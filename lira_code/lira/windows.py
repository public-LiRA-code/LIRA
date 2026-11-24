import hashlib
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from .engine import lower
from .fusion import MAX_RUN, Op, gen1q, mat1q

ONE_Q = {'1q', 'diag1', 'phase1'}
PERM = {'cx', 'swap'}
K_MIX = 4
K_PERM = 6
MAX_SEG = 8


def config(window) -> Tuple[int, int, bool]:
    if not window:
        return None
    if window is True:
        return K_MIX, K_PERM, True
    if isinstance(window, (tuple, list)):
        return int(window[0]), int(window[1]), bool(window[2])
    return int(window), max(K_PERM, int(window)), True


class _Win:
    __slots__ = ('qs', 'gates', 'pidx', 'nseg', 'open', 'perms')

    def __init__(self):
        self.qs, self.gates, self.pidx = [], [], []
        self.nseg, self.open, self.perms = 0, set(), 0

    def add_1q(self, q, gates, pidx):
        if q not in self.qs:
            self.qs.append(q)
        if q not in self.open:
            self.open.add(q)
            self.nseg += 1
        self.gates += gates
        self.pidx += pidx

    def add_perm(self, g):
        for q in g.targets:
            if q not in self.qs:
                self.qs.append(q)
            self.open.discard(q)
        self.gates.append(g)
        self.pidx.append(-1)
        self.perms += 1


def hoist(ops: List[Op]) -> List[Op]:
    key, last, floor, cur = [], {}, -1, None
    for i, op in enumerate(ops):
        if op.width != cur:
            cur, floor, last = op.width, i - 1, {}
        if op.kind == 'run' or (op.kind == 'gate' and lower(op.gates[0])[0] in ONE_Q):
            q = op.gates[0].targets[0]
            k = (last.get(q, (floor,))[0], 1, i)
            last[q] = k
        else:
            k = (i, 0, 0)
            for q in op.qubits:
                last[q] = k
        key.append(k)
    return [ops[i] for i in sorted(range(len(ops)), key=key.__getitem__)]


def windows(ops: List[Op], kmix: int = K_MIX, kperm: int = K_PERM, rot: bool = True, max_seg: int = MAX_SEG) -> List[Op]:
    assert kmix >= 2 and kperm >= kmix
    ops = hoist(ops)
    out: List[Op] = []
    wins: Dict[int, _Win] = {}
    qwin: Dict[int, int] = {}
    runs: Dict[int, Tuple[list, list]] = {}
    cur, nid = None, 0

    def emit(w):
        if not w.perms and len(w.qs) == 1 and len(w.gates) <= MAX_RUN:
            out.append(Op('gate' if len(w.gates) == 1 else 'run', cur, w.gates, w.pidx))
        elif len(w.gates) == 1:
            out.append(Op('gate', cur, w.gates, w.pidx))
        else:
            out.append(Op('window', cur, w.gates, w.pidx))

    def close(wid, fill=()):
        w = wins.pop(wid)
        for q in w.qs:
            del qwin[q]
        for q in fill:
            if q in runs and len(w.qs) < kmix and w.nseg < max_seg:
                w.add_1q(q, *runs.pop(q))
        emit(w)

    def pack(need):
        need = [q for q in need if q in runs]
        if not rot:
            for q in need:
                gs, ps = runs.pop(q)
                for i in range(0, len(gs), MAX_RUN):
                    out.append(Op('gate' if len(gs[i:i + MAX_RUN]) == 1 else 'run', cur, gs[i:i + MAX_RUN], ps[i:i + MAX_RUN]))
            return
        while need:
            grp, need = need[:kmix], need[kmix:]
            for q in list(runs):
                if len(grp) >= kmix:
                    break
                if q not in grp and q not in need:
                    grp.append(q)
            w = _Win()
            for q in grp:
                w.add_1q(q, *runs.pop(q))
            emit(w)

    def barrier(qs):
        need = [q for q in qs if q in runs]
        for wid in sorted({qwin[q] for q in qs if q in qwin}):
            close(wid, need)
        pack(need)

    for op in ops:
        k = int(op.width)
        if k != cur:
            if cur is not None:
                barrier(list(runs) + list(qwin))
            cur = k
        if op.kind in ('diag', 'act'):
            barrier(op.qubits)
            out.append(op)
            continue
        g = op.gates[0]
        kind = lower(g)[0]
        if op.kind == 'run' or (op.kind == 'gate' and kind in ONE_Q):
            q = g.targets[0]
            if q in qwin:
                w = wins[qwin[q]]
                if len(w.qs) <= kmix and w.nseg + (q not in w.open) <= max_seg:
                    w.add_1q(q, list(op.gates), list(op.pidx))
                    continue
                close(qwin[q])
            gs, ps = runs.get(q, ([], []))
            runs[q] = (gs + list(op.gates), ps + list(op.pidx))
            continue
        if op.kind == 'gate' and kind in PERM:
            a, b = g.targets
            while True:
                ws = sorted({qwin[q] for q in (a, b) if q in qwin})
                qs = {a, b}.union(*(wins[w].qs for w in ws))
                rq = [q for q in (a, b) if q in runs]
                nseg = sum(wins[w].nseg for w in ws) + len(rq)
                if len(qs) <= (kmix if nseg else kperm) and nseg <= max_seg:
                    break
                close(max(ws, key=lambda w: (len(wins[w].qs), -w)))
            w = _Win()
            for wid in ws:
                o = wins.pop(wid)
                w.qs += o.qs
                w.gates += o.gates
                w.pidx += o.pidx
                w.nseg += o.nseg
                w.open |= o.open
                w.perms += o.perms
            for q in rq:
                w.add_1q(q, *runs.pop(q))
            w.add_perm(g)
            nid += 1
            wins[nid] = w
            for q in w.qs:
                qwin[q] = nid
            continue
        barrier(op.qubits)
        out.append(op)
    if cur is not None:
        barrier(list(runs) + list(qwin))
    return out


def _moves(k, steps):
    reg = list(range(1 << k))
    for st in steps:
        _permute(reg, st)
    return reg


def _permute(reg, st):
    if st[0] == 'x':
        c, t = st[1], st[2]
        for b in range(len(reg)):
            if (b >> c) & 1 and not (b >> t) & 1:
                reg[b], reg[b | (1 << t)] = reg[b | (1 << t)], reg[b]
    elif st[0] == 'w':
        a, c = st[1], st[2]
        for b in range(len(reg)):
            if (b >> a) & 1 and not (b >> c) & 1:
                o = b ^ (1 << a) ^ (1 << c)
                reg[b], reg[o] = reg[o], reg[b]


def window_tables(op: Op) -> Dict:
    qs: List[int] = []
    steps, segs, open_ = [], [], {}
    for g, p in zip(op.gates, op.pidx):
        for q in g.targets:
            if q not in qs:
                qs.append(q)
        t = tuple(qs.index(q) for q in g.targets)
        kind = lower(g)[0]
        if kind in ONE_Q:
            if t[0] not in open_:
                open_[t[0]] = len(segs)
                segs.append(([], []))
                steps.append(['s', t[0], False])
            segs[open_[t[0]]][0].append(g)
            segs[open_[t[0]]][1].append(p)
        else:
            assert kind in PERM, g.name
            for j in t:
                open_.pop(j, None)
            steps.append(['x' if kind == 'cx' else 'w', *t])
    m, v, grads = [], [], []
    si = 0
    for st in steps:
        if st[0] != 's':
            continue
        gates, pidx = segs[si]
        si += 1
        T = np.eye(2, dtype=np.complex128)
        mem = []
        for g, p in zip(reversed(gates), reversed(pidx)):
            if p >= 0:
                Gt = T @ gen1q(g) @ T.conj().T
                mem.append((Gt[0, 0].real, Gt[1, 1].real, Gt[0, 1].real, Gt[0, 1].imag, p))
            T = T @ mat1q(g)
        st[2] = bool(mem)
        m.append(T.reshape(-1))
        v.append(T.conj().T.reshape(-1))
        grads.append(mem)
    k = len(qs)
    coef, slot, off = [], [], [0]
    for mem in reversed(grads):
        if mem:
            coef += [c for x in mem for c in x[:4]]
            slot += [x[4] for x in mem]
            off.append(len(slot))
    reg = _moves(k, steps)
    pad = lambda x, dt: np.array(x if len(x) else [0], dtype=dt)
    return dict(qs=qs, shape=(k, tuple(tuple(s) for s in steps)),
                m=np.concatenate(m) if m else np.zeros(1, dtype=np.complex128),
                v=np.concatenate(v[::-1]) if v else np.zeros(1, dtype=np.complex128),
                coef=pad(coef, np.float64), slot=pad(slot, np.int32), off=np.array(off, dtype=np.int32),
                touch=1.0 if segs else sum(reg[b] != b for b in range(1 << k)) / (1 << k), nseg=len(segs))


HEAD = r'''
#include <cupy/complex.cuh>
typedef complex<REAL> T;
typedef long long ll;
__device__ inline ll ins0(ll x, int t) { return ((x >> t) << (t + 1)) | (x & ((1LL << t) - 1)); }
__device__ inline double imcj(T l, T p) { return (double)l.real() * p.imag() - (double)l.imag() * p.real(); }
__device__ inline double recj(T l, T p) { return (double)l.real() * p.real() + (double)l.imag() * p.imag(); }
__device__ inline REAL imcjr(T l, T p) { return l.real() * p.imag() - l.imag() * p.real(); }
__device__ inline REAL recjr(T l, T p) { return l.real() * p.real() + l.imag() * p.imag(); }
__device__ inline double wsum(double x) {
    for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
    return x;
}
#define LOOP(n) ll stride = (ll)gridDim.x * blockDim.x; \
    ll tid = (ll)blockIdx.x * blockDim.x + threadIdx.x; \
    for (ll p = tid; p < (n); p += stride)
'''


def shape_name(shape) -> str:
    return hashlib.md5(repr(shape).encode()).hexdigest()[:12]


VARIANT = dict(acc='double', mat='ptr', blockred=False, freg=0, breg=0)


def _source(shape, var, tiled: bool) -> str:
    if KERNEL == 'v2-uncapped' and var.get('freg') and shape[0] >= 5:
        var = dict(var, freg=0)
    k, steps = shape
    D = 1 << k
    name = shape_name(shape)
    nseg = sum(s[0] == 's' for s in steps)
    ng = sum(s[0] == 's' and s[2] for s in steps)
    final = _moves(k, steps)
    live = list(range(D)) if nseg else [b for b in range(D) if final[b] != b]
    assert nseg or not tiled, 'tiled kernels are for windows with rotations'
    acc = var['acc']

    def two(x, y, mat, arr):
        return (f'{{ T u = {arr}{x}, w = {arr}{y}; {arr}{x} = {mat}_0 * u + {mat}_1 * w; {arr}{y} = {mat}_2 * u + {mat}_3 * w; }}')

    def pairs(reg, j):
        return [(reg[b], reg[b | (1 << j)]) for b in range(D) if not (b >> j) & 1]

    def mats(arr):
        if var['mat'] == 'param':
            return [f'T {arr}{i}_{e}' for i in range(nseg) for e in range(4)], []
        return [f'const T* {arr}'], [f'const T {arr}{i}_{e} = {arr}[{4 * i + e}];' for i in range(nseg) for e in range(4)]

    if tiled:
        head = ['ll nb', 'int t', 'll nwmask', 'int kh', 'int h0', 'int h1', 'int h2', 'int h3'] + [f'int q{j}' for j in range(k)]
        setup = [
            'const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;', f'const int w0 = wib * {D * 33};',
            'const int hp[4] = {h0, h1, h2, h3};',
            f'const int qq[{k}] = {{' + ', '.join(f'q{j}' for j in range(k)) + '};',
            f'll eo[{D}]; int ex[{D}];',
            '#pragma unroll',
            f'for (int r = 0; r < {D}; r++) {{',
            '  const int e = lane + 32 * r; const ll idx = e & ((1LL << t) - 1); const int h = e >> t;',
            '  ll ho = 0; for (int i = 0; i < kh; i++) if ((h >> i) & 1) ho |= 1LL << hp[i];',
            '  int g = 0, c = 0; for (int i = 0; i < t; i++) if ((nwmask >> i) & 1) { if ((idx >> i) & 1) g |= 1 << c; c++; }',
            '  const ll full = idx | ho; int b = 0;',
            f'  for (int j = 0; j < {k}; j++) if ((full >> qq[j]) & 1) b |= 1 << j;',
            '  eo[r] = full; ex[r] = w0 + 33 * b + g;',
            '}',
            'for (ll p = (ll)blockIdx.x * NW + wib; p < nb; p += (ll)gridDim.x * NW) {',
            'll base = p << t; for (int i = 0; i < kh; i++) base = ins0(base, hp[i]);']
        fpre = [f'#define NW (512 / ({D} * (int)sizeof(REAL)))', f'__shared__ T sa[NW * {D * 33}];']
        bpre = [f'#define NW (256 / ({D} * (int)sizeof(REAL)))', f'__shared__ T sa[NW * {D * 33}];', f'__shared__ T sc[NW * {D * 33}];']

        def load(arrs):
            return ([f'for (int r = 0; r < {D}; r++) {{ ' + ' '.join(f'{sh}[ex[r]] = {buf}[base | eo[r]];' for _, buf, sh in arrs) + ' }',
                     '__syncwarp();'] + [f'T ' + ', '.join(f'{a}{b} = {sh}[w0 + {33 * b} + lane]' for a, _, sh in arrs) + ';' for b in range(D)])

        def store(arrs, reg):
            return (['__syncwarp();'] + [' '.join(f'{sh}[w0 + {33 * b} + lane] = {a}{reg[b]};' for a, _, sh in arrs) for b in range(D)]
                    + ['__syncwarp();', f'for (int r = 0; r < {D}; r++) {{ ' + ' '.join(f'{buf}[base | eo[r]] = {sh}[ex[r]];' for _, buf, sh in arrs) + ' }',
                       '__syncwarp();'])
        close = ['}', '#undef NW']
    else:
        head = ['ll n'] + [f'int p{j}' for j in range(k)] + [f'int q{j}' for j in range(k)]
        off = lambda b: f'const ll o{b} = ' + (' | '.join(f'(1LL << q{j})' for j in range(k) if (b >> j) & 1) or '0') + ';'
        setup = [off(b) for b in live] + ['LOOP(n) {', 'll base = p; ' + ' '.join(f'base = ins0(base, p{j});' for j in range(k))]
        fpre, bpre = [], []

        def load(arrs):
            return [f'T ' + ', '.join(f'{a}{b} = {buf}[base | o{b}]' for a, buf, _ in arrs) + ';' for b in live]

        def store(arrs, reg):
            return [' '.join(f'{buf}[base | o{b}] = {a}{reg[b]};' for a, buf, _ in arrs) for b in live]
        close = ['}']
    sfx = 't' if tiled else ''

    par, dec = mats('m')
    cap = lambda r: f'__maxnreg__({r}) ' if r else ''
    f = [f'extern "C" __global__ void {cap(var.get("freg", 0))}wf{sfx}_{name}(' + ', '.join(['T* s'] + head + par) + ') {'] + fpre + dec + setup
    f += load([('a', 's', 'sa')])
    reg, i = list(range(D)), 0
    for st in steps:
        if st[0] == 's':
            f += [two(x, y, f'm{i}', 'a') for x, y in pairs(reg, st[1])]
            i += 1
        else:
            _permute(reg, st)
    f += store([('a', 's', 'sa')], reg) + close + ['}']

    par, dec = mats('v')
    bw = [f'extern "C" __global__ void {cap(var.get("breg", 0))}wb{sfx}_{name}(' + ', '.join(['T* s', 'T* l', 'double* gr'] + head + par +
                                                                   ['const double* coef', 'const int* slot', 'const int* off']) + ') {']
    bw += bpre + dec
    if acc == 'realall':
        bw += [f'REAL t{g}_0 = 0, t{g}_1 = 0, t{g}_2 = 0, t{g}_3 = 0;' for g in range(ng)]
    else:
        bw += [f'double r{g}_0 = 0.0, r{g}_1 = 0.0, r{g}_2 = 0.0, r{g}_3 = 0.0;' for g in range(ng)]
    bw += setup
    if acc == 'real':
        bw += [f'REAL t{g}_0 = 0, t{g}_1 = 0, t{g}_2 = 0, t{g}_3 = 0;' for g in range(ng)]
    reg, i, g = list(range(D)), 0, 0
    body = []
    for st in reversed(steps):
        if st[0] == 's':
            for x, y in pairs(reg, st[1]):
                if st[2] and acc == 'double':
                    body.append(f'r{g}_0 += imcj(c{x}, a{x}); r{g}_1 += imcj(c{y}, a{y}); '
                                f'r{g}_2 += imcj(c{x}, a{y}) + imcj(c{y}, a{x}); r{g}_3 += recj(c{x}, a{y}) - recj(c{y}, a{x});')
                elif st[2]:
                    body.append(f't{g}_0 += imcjr(c{x}, a{x}); t{g}_1 += imcjr(c{y}, a{y}); '
                                f't{g}_2 += imcjr(c{x}, a{y}) + imcjr(c{y}, a{x}); t{g}_3 += recjr(c{x}, a{y}) - recjr(c{y}, a{x});')
                body += [two(x, y, f'v{i}', 'a'), two(x, y, f'v{i}', 'c')]
            g += st[2]
            i += 1
        else:
            _permute(reg, st)
    if nseg:
        bw += load([('a', 's', 'sa'), ('c', 'l', 'sc')]) + body + store([('a', 's', 'sa'), ('c', 'l', 'sc')], reg)
        if acc == 'real':
            bw += [f'r{g}_{e} += (double)t{g}_{e};' for g in range(ng) for e in range(4)]
    else:
        for arr in (('a', 's', 'sa'), ('c', 'l', 'sc')):
            bw += load([arr]) + store([arr], reg)
    bw += close
    if acc == 'realall':
        bw += [f'double r{g}_{e} = (double)t{g}_{e};' for g in range(ng) for e in range(4)]
    if ng:
        bw += [f'r{g}_{e} = wsum(r{g}_{e});' for g in range(ng) for e in range(4)]
    if ng and var['blockred']:
        nv = 4 * ng
        bw += [f'__shared__ double shr[32 * {nv}];', 'const int w_ = threadIdx.x >> 5;', 'if ((threadIdx.x & 31) == 0) {']
        bw += [f'shr[w_ * {nv} + {4 * g + e}] = r{g}_{e};' for g in range(ng) for e in range(4)]
        bw += ['}', '__syncthreads();', 'if (threadIdx.x == 0) {', 'const int nw_ = blockDim.x >> 5;']
        bw += [f'for (int u = 1; u < nw_; u++) r{g}_{e} += shr[u * {nv} + {4 * g + e}];' for g in range(ng) for e in range(4)]
    elif ng:
        bw += ['if ((threadIdx.x & 31) == 0) {']
    if ng:
        bw += [f'for (int i = off[{g}]; i < off[{g + 1}]; i++) atomicAdd(gr + slot[i], 2.0 * (coef[4 * i] * r{g}_0 + '
               f'coef[4 * i + 1] * r{g}_1 + coef[4 * i + 2] * r{g}_2 + coef[4 * i + 3] * r{g}_3));' for g in range(ng)]
        bw += ['}']
    bw += ['}']
    return '\n'.join(f) + '\n' + '\n'.join(bw) + '\n'


def window_source(shape, variant=None) -> str:
    return _source(shape, dict(VARIANT, **(variant or {})), False)


def tile_layout(qs, width: int):
    t = 5
    while True:
        j = sum(1 for q in qs if q < t)
        if 5 + j == t:
            break
        t = 5 + j
    if t + sum(1 for q in qs if q >= t) > width:
        return None
    nw = sum(1 << b for b in range(t) if b not in qs)
    return t, nw, sorted(q for q in qs if q >= t)


def tiled_warps(k: int, real: str):
    size = 4 if real == 'float' else 8
    if (1 << k) * size > 256:
        return None
    return 512 // ((1 << k) * size), 256 // ((1 << k) * size)


def window_source_tiled(shape, variant=None) -> str:
    return _source(shape, dict(VARIANT, **(variant or {})), True)


KERNEL = os.environ.get('LIRA_WINDOW_KERNEL', 'v2')
BACKWARD_THREADS = 1 << 19 if KERNEL == 'plain' else 1 << 20
TILED_FORWARD_THREADS = 1 << 20
TILED_MIN_SETS = int(os.environ.get('LIRA_WINDOW_TILED_MIN', 1 << 13))


def in_use(real: str) -> Dict:
    if KERNEL == 'plain':
        return dict(VARIANT, tiled=False)
    return dict(VARIANT, acc='real' if real == 'float' else 'double', mat=os.environ.get('LIRA_WINDOW_MATRIX', 'param'),
                tiled=os.environ.get('LIRA_WINDOW_TILED', '1') == '1')


def _variant(real: str, tiled: bool) -> Dict:
    var = in_use(real)
    if KERNEL != 'plain' and real == 'float' and not tiled:
        var = dict(var, freg=96)
    return var


def launch_layout(h: Dict, width: int, real: str, kind: str = 'forward'):
    qs = h['qs']
    if (not in_use(real)['tiled'] or not h['nseg'] or tiled_warps(len(qs), real) is None
            or not (min(qs) < 2 and sum(q < 5 for q in qs) >= 2) or (kind == 'backward' and real != 'float')):
        return None
    lay = tile_layout(qs, width)
    if lay is None or (1 << (width - lay[0] - len(lay[2]))) < TILED_MIN_SETS:
        return None
    return lay


_COMPILED: Dict = {}
SALT = os.environ.get('LIRA_WINDOW_SALT', '')
STATS = dict(compiled=0, seconds=0.0)
COMPILE_THREADS = 16


def kernels(specs, real: str) -> Dict:
    import cupy as cp
    new = [s for s in dict.fromkeys(specs) if (real, *s) not in _COMPILED]
    if new:
        t0 = time.perf_counter()
        dev = cp.cuda.Device().id

        def build(spec):
            shape, tiled = spec
            sfx = 't' if tiled else ''
            with cp.cuda.Device(dev):
                mod = cp.RawModule(code=(HEAD + _source(shape, _variant(real, tiled), tiled)).replace('REAL', real) + (f'// {SALT}\n' if SALT else ''))
                mod.compile()
            return mod.get_function(f'wf{sfx}_' + shape_name(shape)), mod.get_function(f'wb{sfx}_' + shape_name(shape))
        if len(new) == 1:
            built = [build(new[0])]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(min(COMPILE_THREADS, len(new))) as ex:
                built = list(ex.map(build, new))
        for s, k in zip(new, built):
            _COMPILED[(real, *s)] = k
        STATS['compiled'] += len(new)
        STATS['seconds'] += time.perf_counter() - t0
    return {s: _COMPILED[(real, *s)] for s in specs}
