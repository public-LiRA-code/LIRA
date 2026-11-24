from typing import Dict

import numpy as np
import cupy as cp
import cupyx
from cupy_backends.cuda.libs import cublas

from .custatevec_backend import _embed, _perm_matrix
from .engine import lower
from .fusion import gen1q, mat1q
from .windows import ONE_Q, window_tables

OP_N, OP_T, OP_C = 0, 1, 2
BATCH_MIN_LOW = 5
CROSS_LOOP_MAX = 64
STATS = dict(gemm=0, batched=0, transposes=0, gathers=0)

_spares: Dict = {}
_consts: Dict = {}


def release():
    _spares.clear()


def _const(dt):
    if dt not in _consts:
        _consts[dt] = (np.ones(1, dtype=dt), np.zeros(1, dtype=dt))
    return _consts[dt]


def _handle():
    h = cp.cuda.device.get_cublas_handle()
    cublas.setStream(h, cp.cuda.get_current_stream().ptr)
    cublas.setPointerMode(h, cublas.CUBLAS_POINTER_MODE_HOST)
    return h


def _gemm(dt, ta, tb, m, n, k, A, lda, B, ldb, C, ldc, accumulate=False):
    one, zero = _const(dt)
    f = cublas.cgemm if dt == np.complex64 else cublas.zgemm
    f(_handle(), ta, tb, m, n, k, one.ctypes.data, A, lda, B, ldb, (one if accumulate else zero).ctypes.data, C, ldc)
    STATS['gemm'] += 1


def _gemm_batched(dt, ta, tb, m, n, k, A, lda, sA, B, ldb, sB, C, ldc, sC, count):
    one, zero = _const(dt)
    f = cublas.cgemmStridedBatched if dt == np.complex64 else cublas.zgemmStridedBatched
    f(_handle(), ta, tb, m, n, k, one.ctypes.data, A, lda, sA, B, ldb, sB, zero.ctypes.data, C, ldc, sC, count)
    STATS['batched'] += 1


def _pool(s):
    key = (cp.cuda.Device().id, s.dtype, s.size)
    if key not in _spares:
        _spares[key] = [[cp.zeros_like(s), 0], [cp.zeros_like(s), 0]]
    return _spares[key]


def _take(s, width, clean):
    pool = _pool(s)
    pool.sort(key=lambda sp: sp[1], reverse=clean)
    sp = pool.pop()
    if clean and sp[1] > width:
        sp[0][1 << width: 1 << sp[1]].fill(0)
    sp[1] = width if clean else s.size.bit_length() - 1
    return sp


def _give(s, dirty):
    _pool(s).append([s, dirty])


def tables(eng, op):
    if 'blas' in op.cache:
        return op.cache['blas']
    if 'whost' not in op.cache:
        op.cache['whost'] = window_tables(op)
    h = op.cache['whost']
    qs = h['qs']
    k, D = len(qs), 1 << len(qs)
    p = sorted(qs)
    rank = [p.index(q) for q in qs]
    srt = np.array([sum(((b >> j) & 1) << rank[j] for j in range(k)) for b in range(D)])
    full = []
    for g in op.gates:
        kind = lower(g)[0]
        t = tuple(qs.index(q) for q in g.targets)
        full.append(_embed(mat1q(g), t[0], D) if kind in ONE_Q else _perm_matrix(kind, t, D))
    T = np.eye(D, dtype=np.complex128)
    rows, slots = [], []
    for g, pi, F in zip(reversed(op.gates), reversed(op.pidx), reversed(full)):
        if pi >= 0:
            A = T @ _embed(gen1q(g), qs.index(g.targets[0]), D) @ T.conj().T
            As = np.empty_like(A)
            As[np.ix_(srt, srt)] = A
            rows.append(As.reshape(-1))
            slots.append(pi)
        T = T @ F
    W = np.empty_like(T)
    W[np.ix_(srt, srt)] = T
    dt = np.dtype(eng.dtype).type
    w = op.width
    a = p[0]
    contiguous = p[-1] - p[0] == k - 1
    c = dict(k=k, D=D, dt=dt, width=w)
    if contiguous and (a == 0 or a >= max(BATCH_MIN_LOW, k)):
        c['layout'] = ('flat',) if a == 0 else ('batched', 1 << (w - a - k), 1 << a)
    else:
        groups, b = [], w - 1
        while b >= 0:
            inw = b in p
            n_ = 0
            while b >= 0 and (b in p) == inw:
                n_ += 1
                b -= 1
            groups.append((1 << n_, inw))
        shape = tuple(sz for sz, _ in groups)
        axes = [i for i, (_, inw) in enumerate(groups) if not inw] + [i for i, (_, inw) in enumerate(groups) if inw]
        c['layout'] = ('transposed', shape, tuple(axes), tuple(int(x) for x in np.argsort(axes)),
                       tuple(shape[i] for i in axes))
    if h['nseg']:
        c['W'] = cp.asarray(np.ascontiguousarray(W.astype(dt)))
        c['V'] = cp.asarray(np.ascontiguousarray(W.conj().T.astype(dt)))
        c['perm'] = None
    else:
        src = np.argmax(np.abs(W), axis=1)
        c['perm'] = cp.asarray(src.astype(np.int64))
        c['iperm'] = cp.asarray(np.argsort(src).astype(np.int64))
    c['C'] = cp.asarray(np.array(rows)) if rows else None
    c['slots'] = cp.asarray(np.array(slots, dtype=np.int64)) if rows else None
    op.cache['blas'] = c
    return c


def prepare(eng, ops):
    for op in ops:
        tables(eng, op)


def _apply_flat(dt, Mrm, x, y, R, D):
    _gemm(dt, OP_T, OP_N, D, R, D, Mrm.data.ptr, D, x.data.ptr, D, y.data.ptr, D)


def _apply_batched(dt, Mrm, x, y, B, D, L):
    _gemm_batched(dt, OP_N, OP_N, L, D, D, x.data.ptr, L, D * L, Mrm.data.ptr, D, 0, y.data.ptr, L, D * L, B)


def _cross_flat(dt, x, l, Mout, R, D):
    _gemm(dt, OP_N, OP_C, D, D, R, x.data.ptr, D, l.data.ptr, D, Mout.data.ptr, D)


def _apply(c, mat, src, dst):
    dt, D, n = c['dt'], c['D'], 1 << c['width']
    x, y = src[:n], dst[:n]
    lay = c['layout']
    if lay[0] == 'flat':
        if c['perm'] is not None:
            cp.take(x.reshape(n // D, D), mat, axis=1, out=y.reshape(n // D, D))
            STATS['gathers'] += 1
        else:
            _apply_flat(dt, mat, x, y, n // D, D)
        return dst
    if lay[0] == 'batched':
        B, L = lay[1], lay[2]
        if c['perm'] is not None:
            cp.take(x.reshape(B, D, L), mat, axis=1, out=y.reshape(B, D, L))
            STATS['gathers'] += 1
        else:
            _apply_batched(dt, mat, x, y, B, D, L)
        return dst
    _, shape, axes, inv, tshape = lay
    cp.copyto(y.reshape(tshape), x.reshape(shape).transpose(axes))
    if c['perm'] is not None:
        cp.take(y.reshape(n // D, D), mat, axis=1, out=x.reshape(n // D, D))
        STATS['gathers'] += 1
    else:
        _apply_flat(dt, mat, y, x, n // D, D)
    cp.copyto(y.reshape(shape), x.reshape(tshape).transpose(inv))
    STATS['transposes'] += 2
    return dst


def forward(eng, s, op):
    c = tables(eng, op)
    if not eng._out_of_place_ok:
        raise RuntimeError('LIRA_WINDOW_KERNEL=cublas works out of place: use LiraEngine.run (single GPU)')
    sp = _take(s, c['width'], True)
    out = _apply(c, c['perm'] if c['perm'] is not None else c['W'], s, sp[0])
    _give(s, c['width'])
    return out


def backward(eng, s, l, gr, op):
    c = tables(eng, op)
    if not eng._out_of_place_ok:
        raise RuntimeError('LIRA_WINDOW_KERNEL=cublas works out of place: use LiraEngine.run (single GPU)')
    dt, D, w = c['dt'], c['D'], c['width']
    n = 1 << w
    spa = _take(s, w, True)
    spb = _take(l, w, False)
    N = s.size.bit_length() - 1
    A, Bf = spa[0], spb[0]
    if c['perm'] is not None:
        out_s = _apply(c, c['iperm'], s, A)
        out_l = _apply(c, c['iperm'], l, Bf)
        _give(s, w)
        _give(l, N)
        return out_s, out_l
    lay = c['layout']
    Mrm = cp.empty((D, D), dtype=dt)
    if lay[0] == 'flat':
        if c['C'] is not None:
            _cross_flat(dt, s[:n], l[:n], Mrm, n // D, D)
        _apply_flat(dt, c['V'], s[:n], A[:n], n // D, D)
        _apply_flat(dt, c['V'], l[:n], Bf[:n], n // D, D)
    elif lay[0] == 'batched':
        Bn, L = lay[1], lay[2]
        if c['C'] is not None and Bn <= CROSS_LOOP_MAX:
            Mcm = cp.empty((D, D), dtype=dt)
            step = D * L * np.dtype(dt).itemsize
            for b in range(Bn):
                _gemm(dt, OP_C, OP_N, D, D, L, l.data.ptr + b * step, L, s.data.ptr + b * step, L, Mcm.data.ptr, D, accumulate=b > 0)
            Mrm[...] = Mcm.T
        elif c['C'] is not None:
            blocks = A[: Bn * D * D]
            _gemm_batched(dt, OP_C, OP_N, D, D, L, l[:n].data.ptr, L, D * L, s[:n].data.ptr, L, D * L, blocks.data.ptr, D, D * D, Bn)
            Mrm[...] = blocks.reshape(Bn, D, D).sum(axis=0).T
        _apply_batched(dt, c['V'], s[:n], A[:n], Bn, D, L)
        _apply_batched(dt, c['V'], l[:n], Bf[:n], Bn, D, L)
    else:
        _, shape, axes, inv, tshape = lay
        x, y = s[:n], l[:n]
        cp.copyto(A[:n].reshape(tshape), x.reshape(shape).transpose(axes))
        cp.copyto(Bf[:n].reshape(tshape), y.reshape(shape).transpose(axes))
        if c['C'] is not None:
            _cross_flat(dt, A[:n], Bf[:n], Mrm, n // D, D)
        _apply_flat(dt, c['V'], A[:n], x, n // D, D)
        _apply_flat(dt, c['V'], Bf[:n], y, n // D, D)
        cp.copyto(A[:n].reshape(shape), x.reshape(tshape).transpose(inv))
        cp.copyto(Bf[:n].reshape(shape), y.reshape(tshape).transpose(inv))
        STATS['transposes'] += 4
    if c['C'] is not None:
        g = 2.0 * (c['C'] * Mrm.reshape(-1).astype(np.complex128)[None, :]).sum(axis=1).imag
        cupyx.scatter_add(gr, c['slots'], g)
    _give(s, w)
    _give(l, N)
    return A, Bf
