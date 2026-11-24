import ctypes
import os
import sysconfig
from typing import Dict

import numpy as np
import cupy as cp
import cupyx

from .engine import lower
from .fusion import gen1q, mat1q
from .windows import HEAD, ONE_Q, PERM, window_tables

SITE = sysconfig.get_paths()['purelib']
LIB = os.environ.get('LIRA_CUSTATEVEC_LIB', os.path.join(SITE, 'cuquantum', 'lib', 'libcustatevec.so.1'))
CUDA_C_32F, CUDA_C_64F = 4, 5
LAYOUT_ROW, COMPUTE_DEFAULT = 1, 0
STATS = dict(apply=0, permute=0, cross=0)

_lib = None
_handles: Dict = {}
_work: Dict = {}
_wsize: Dict = {}
_cross: Dict = {}
_mbuf: Dict = {}


def lib():
    global _lib
    if _lib is None:
        for sub in ('nvidia/cublas/lib', 'nvidia/cuda_runtime/lib', 'nvidia/nvjitlink/lib', 'nvidia/cusparse/lib'):
            d = os.path.join(SITE, sub)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if '.so' in f:
                        try:
                            ctypes.CDLL(os.path.join(d, f), mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass
        L = ctypes.CDLL(LIB)
        vp, u32, i32, sz = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32, ctypes.c_size_t
        L.custatevecCreate.argtypes = [ctypes.POINTER(vp)]
        L.custatevecSetStream.argtypes = [vp, vp]
        L.custatevecApplyMatrixGetWorkspaceSize.argtypes = [vp, ctypes.c_int, u32, vp, ctypes.c_int, ctypes.c_int, i32, u32, u32, ctypes.c_int,
                                                            ctypes.POINTER(sz)]
        tg = ctypes.POINTER(ctypes.c_int32)
        L.custatevecApplyMatrix.argtypes = [vp, vp, ctypes.c_int, u32, vp, ctypes.c_int, ctypes.c_int, i32, tg, u32, vp, vp, u32, ctypes.c_int,
                                            vp, sz]
        L.custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize.argtypes = [vp, ctypes.c_int, u32, vp, vp, ctypes.c_int, tg, u32, u32,
                                                                                  ctypes.POINTER(sz)]
        L.custatevecApplyGeneralizedPermutationMatrix.argtypes = [vp, vp, ctypes.c_int, u32, vp, vp, ctypes.c_int, i32, tg, u32, vp, vp, u32,
                                                                  vp, sz]
        _lib = L
    return _lib


def _check(st, what):
    if st != 0:
        raise RuntimeError(f'cuStateVec {what} returned status {st}')


def handle():
    dev = cp.cuda.Device().id
    h = _handles.get(dev)
    if h is None:
        p = ctypes.c_void_p()
        _check(lib().custatevecCreate(ctypes.byref(p)), 'Create')
        h = _handles[dev] = [p, None]
    st = cp.cuda.get_current_stream().ptr
    if h[1] != st:
        _check(lib().custatevecSetStream(h[0], st), 'SetStream')
        h[1] = st
    return h[0]


def _workspace(key, query):
    dev = cp.cuda.Device().id
    k = (dev,) + key
    if k not in _wsize:
        n = ctypes.c_size_t(0)
        query(ctypes.byref(n))
        _wsize[k] = int(n.value)
    need = _wsize[k]
    if not need:
        return None, 0
    buf = _work.get(dev)
    if buf is None or buf.nbytes < need:
        buf = _work[dev] = cp.empty(need, dtype=np.uint8)
    return buf.data.ptr, need


def _embed(U, j, D):
    F = np.zeros((D, D), dtype=np.complex128)
    for r in range(D):
        for c in range(D):
            if (r ^ c) & ~(1 << j) == 0:
                F[r, c] = U[(r >> j) & 1, (c >> j) & 1]
    return F


def _perm_matrix(kind, t, D):
    F = np.zeros((D, D), dtype=np.complex128)
    for c in range(D):
        if kind == 'cx':
            r = c ^ (((c >> t[0]) & 1) << t[1])
        else:
            a, b = t
            r = c if ((c >> a) & 1) == ((c >> b) & 1) else c ^ (1 << a) ^ (1 << b)
        F[r, c] = 1
    return F


def tables(eng, op):
    if 'csv' in op.cache:
        return op.cache['csv']
    if 'whost' not in op.cache:
        op.cache['whost'] = window_tables(op)
    h = op.cache['whost']
    qs = h['qs']
    k, D = len(qs), 1 << len(qs)
    full = []
    for g in op.gates:
        kind = lower(g)[0]
        t = tuple(qs.index(q) for q in g.targets)
        full.append(_embed(mat1q(g), t[0], D) if kind in ONE_Q else _perm_matrix(kind, t, D))
    T = np.eye(D, dtype=np.complex128)
    rows, slots = [], []
    for g, p, F in zip(reversed(op.gates), reversed(op.pidx), reversed(full)):
        if p >= 0:
            A = T @ _embed(gen1q(g), qs.index(g.targets[0]), D) @ T.conj().T
            rows.append(A.reshape(-1))
            slots.append(p)
        T = T @ F
    W = T
    dt = np.dtype(eng.dtype)
    c = dict(k=k, targets=(ctypes.c_int32 * k)(*qs), sorted=tuple(np.int32(q) for q in sorted(qs)), local=tuple(np.int32(q) for q in qs),
             svtype=CUDA_C_32F if dt == np.complex64 else CUDA_C_64F, perm=None, W=None, C=None, slots=None)
    if h['nseg']:
        c['W'] = cp.asarray(np.ascontiguousarray(W.astype(dt)))
    else:
        src = np.argmax(np.abs(W), axis=1)
        c['perm'] = cp.asarray(src.astype(np.int64))
        c['W'] = cp.asarray(np.ascontiguousarray(W.astype(dt)))
    if rows:
        c['C'] = cp.asarray(np.array(rows))
        c['slots'] = cp.asarray(np.array(slots, dtype=np.int64))
    op.cache['csv'] = c
    return c


def prepare(eng, ops):
    for op in ops:
        tables(eng, op)
        _kernel(eng.real, op.cache['csv']['k'])


def _cross_source(k):
    D = 1 << k
    G = min(32, 1024 // D)
    E = (D * D + 255) // 256
    pargs = ', '.join(f'int p{j}' for j in range(k))
    qargs = ', '.join(f'int q{j}' for j in range(k))
    return f"""
extern "C" __global__ void xm{k}(const T* s, const T* l, double* M, ll n, {pargs}, {qargs}) {{
    __shared__ T ss[{D}][{G + 1}];
    __shared__ T sl[{D}][{G + 1}];
    const int t = threadIdx.x;
    const int pp[{k}] = {{{', '.join(f'p{j}' for j in range(k))}}};
    const int qq[{k}] = {{{', '.join(f'q{j}' for j in range(k))}}};
    double ar[{E}], ai[{E}];
    for (int u = 0; u < {E}; u++) {{ ar[u] = 0.0; ai[u] = 0.0; }}
    for (ll tile = blockIdx.x; tile * {G} < n; tile += gridDim.x) {{
        for (int e = t; e < {G * D}; e += 256) {{
            const int g = e % {G}, b = e / {G};
            const ll grp = tile * {G} + g;
            T x = T(0), y = T(0);
            if (grp < n) {{
                ll base = grp;
                for (int j = 0; j < {k}; j++) base = ins0(base, pp[j]);
                ll o = 0;
                for (int j = 0; j < {k}; j++) if ((b >> j) & 1) o |= 1LL << qq[j];
                x = s[base | o];
                y = l[base | o];
            }}
            ss[b][g] = x;
            sl[b][g] = y;
        }}
        __syncthreads();
        #pragma unroll
        for (int u = 0; u < {E}; u++) {{
            const int en = t + 256 * u;
            if (en < {D * D}) {{
                const int ea = en / {D}, eb = en % {D};
                REAL pr = 0, pi = 0;
                #pragma unroll
                for (int g = 0; g < {G}; g++) {{
                    const T y = sl[ea][g], x = ss[eb][g];
                    pr += y.real() * x.real() + y.imag() * x.imag();
                    pi += y.real() * x.imag() - y.imag() * x.real();
                }}
                ar[u] += pr;
                ai[u] += pi;
            }}
        }}
        __syncthreads();
    }}
    for (int u = 0; u < {E}; u++) {{
        const int en = t + 256 * u;
        if (en < {D * D}) {{ atomicAdd(M + 2 * en, ar[u]); atomicAdd(M + 2 * en + 1, ai[u]); }}
    }}
}}
"""


def _kernel(real, k):
    key = (real, k, cp.cuda.Device().id)
    if key not in _cross:
        mod = cp.RawModule(code=(HEAD + _cross_source(k)).replace('REAL', real))
        _cross[key] = mod.get_function(f'xm{k}')
    return _cross[key]


def _apply(eng, s, op, c, adjoint):
    L, h = lib(), handle()
    n = op.width
    if c['perm'] is not None:
        perm = c['perm'].data.ptr
        ws, wn = _workspace(('gpm', c['svtype'], n, c['k']), lambda out: _check(
            L.custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(h, c['svtype'], n, perm, None, c['svtype'], c['targets'], c['k'], 0, out),
            'GPM workspace'))
        _check(L.custatevecApplyGeneralizedPermutationMatrix(h, s.data.ptr, c['svtype'], n, perm, None, c['svtype'], adjoint, c['targets'], c['k'],
                                                             None, None, 0, ws, wn), 'ApplyGeneralizedPermutationMatrix')
        STATS['permute'] += 1
        return
    W = c['W'].data.ptr
    ws, wn = _workspace(('mat', c['svtype'], n, c['k'], adjoint), lambda out: _check(
        L.custatevecApplyMatrixGetWorkspaceSize(h, c['svtype'], n, W, c['svtype'], LAYOUT_ROW, adjoint, c['k'], 0, COMPUTE_DEFAULT, out),
        'ApplyMatrix workspace'))
    _check(L.custatevecApplyMatrix(h, s.data.ptr, c['svtype'], n, W, c['svtype'], LAYOUT_ROW, adjoint, c['targets'], c['k'], None, None, 0,
                                   COMPUTE_DEFAULT, ws, wn), 'ApplyMatrix')
    STATS['apply'] += 1


def forward(eng, s, op):
    _apply(eng, s, op, tables(eng, op), 0)


def backward(eng, s, l, gr, op):
    c = tables(eng, op)
    if c['C'] is not None:
        dev = cp.cuda.Device().id
        D2 = 1 << (2 * c['k'])
        M = _mbuf.get(dev)
        if M is None or M.size < D2:
            M = _mbuf[dev] = cp.zeros(max(D2, 256), dtype=np.complex128)
        Mv = M[:D2]
        Mv.fill(0)
        groups = 1 << (op.width - c['k'])
        G = min(32, 1024 >> c['k'])
        sms = cp.cuda.Device().attributes['MultiProcessorCount']
        grid = max(1, min((groups + G - 1) // G, 4 * sms))
        _kernel(eng.real, c['k'])((grid,), (256,), (s, l, Mv.view(np.float64), np.int64(groups), *c['sorted'], *c['local']))
        g = 2.0 * (c['C'] * Mv[None, :]).sum(axis=1).imag
        cupyx.scatter_add(gr, c['slots'], g)
        STATS['cross'] += 1
    _apply(eng, s, op, c, 1)
    _apply(eng, l, op, c, 1)
