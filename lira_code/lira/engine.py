import time
from typing import Dict, Sequence, Tuple

import cupy as cp
import numpy as np

from lira.gates import Gate
from .planner import Plan
from .scheme import WINDOW

_SRC = r'''
#include <cupy/complex.cuh>
typedef complex<REAL> T;
typedef long long ll;

__device__ inline ll ins0(ll x, int t) { return ((x >> t) << (t + 1)) | (x & ((1LL << t) - 1)); }
__device__ inline ll ins00(ll x, int a, int b) {
    int lo = a < b ? a : b, hi = a < b ? b : a;
    return ins0(ins0(x, lo), hi);
}
__device__ inline ll ins000(ll x, int a, int b, int c) {
    int lo = a < b ? (a < c ? a : c) : (b < c ? b : c), hi = a > b ? (a > c ? a : c) : (b > c ? b : c);
    return ins0(ins0(ins0(x, lo), a + b + c - lo - hi), hi);
}
__device__ inline double imcj(T l, T p) { return (double)l.real() * p.imag() - (double)l.imag() * p.real(); }

#define LOOP(n) ll stride = (ll)gridDim.x * blockDim.x; \
    ll tid = (ll)blockIdx.x * blockDim.x + threadIdx.x; \
    for (ll p = tid; p < (n); p += stride)
__device__ inline void grad_add(double* gr, int slot, double acc) {
    if (slot < 0) return;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(gr + slot, 2.0 * acc);
}
#define GRAD(slot, acc) grad_add(gr, (slot), (acc))

extern "C" __global__ void f_1q(T* s, ll n, int t, T u00, T u01, T u10, T u11) {
    LOOP(n) { ll i0 = ins0(p, t), i1 = i0 | (1LL << t);
        T a = s[i0], b = s[i1];
        s[i0] = u00 * a + u01 * b; s[i1] = u10 * a + u11 * b; }
}
extern "C" __global__ void f_diag1(T* s, ll n, int t, T d0, T d1) {
    LOOP(n) { s[p] *= ((p >> t) & 1) ? d1 : d0; }
}
extern "C" __global__ void f_phase1(T* s, ll n, int t, T d) {
    LOOP(n) { s[ins0(p, t) | (1LL << t)] *= d; }
}
extern "C" __global__ void f_cx(T* s, ll n, int c, int t) {
    LOOP(n) { ll i0 = ins00(p, c, t) | (1LL << c), i1 = i0 | (1LL << t);
        T a = s[i0]; s[i0] = s[i1]; s[i1] = a; }
}
extern "C" __global__ void f_cphase(T* s, ll n, int c, int t, T d) {
    LOOP(n) { s[ins00(p, c, t) | (1LL << c) | (1LL << t)] *= d; }
}
extern "C" __global__ void f_diag2(T* s, ll n, int a, int b, T de, T dq) {
    LOOP(n) { s[p] *= (((p >> a) ^ (p >> b)) & 1) ? dq : de; }
}
extern "C" __global__ void f_swap(T* s, ll n, int a, int b) {
    LOOP(n) { ll base = ins00(p, a, b), i = base | (1LL << a), j = base | (1LL << b);
        T x = s[i]; s[i] = s[j]; s[j] = x; }
}

extern "C" __global__ void f_c1q(T* s, ll n, int c, int t, T u00, T u01, T u10, T u11) {
    LOOP(n) { ll i0 = ins00(p, c, t) | (1LL << c), i1 = i0 | (1LL << t);
        T a = s[i0], b = s[i1];
        s[i0] = u00 * a + u01 * b; s[i1] = u10 * a + u11 * b; }
}
extern "C" __global__ void f_ccx(T* s, ll n, int c1, int c2, int t) {
    LOOP(n) { ll i0 = ins000(p, c1, c2, t) | (1LL << c1) | (1LL << c2), i1 = i0 | (1LL << t);
        T a = s[i0]; s[i0] = s[i1]; s[i1] = a; }
}

extern "C" __global__ void f_pp(T* s, ll n, int a, int b, REAL c, REAL sn, int e) {
    T z(0, -sn), ze(0, -sn * e);
    LOOP(n) { ll i0 = ins00(p, a, b), ia = i0 | (1LL << a), ib = i0 | (1LL << b), i3 = ia | ib;
        T x0 = s[i0], xa = s[ia], xb = s[ib], x3 = s[i3];
        s[i0] = c * x0 + ze * x3; s[i3] = c * x3 + ze * x0;
        s[ia] = c * xa + z * xb;  s[ib] = c * xb + z * xa; }
}

extern "C" __global__ void b_pp(T* s, T* l, double* gr, ll n, int a, int b, REAL c, REAL sn, int e, double g, int slot) {
    double acc = 0.0;
    T z(0, sn), ze(0, sn * e);
    LOOP(n) { ll i0 = ins00(p, a, b), ia = i0 | (1LL << a), ib = i0 | (1LL << b), i3 = ia | ib;
        T x0 = s[i0], xa = s[ia], xb = s[ib], x3 = s[i3], y0 = l[i0], ya = l[ia], yb = l[ib], y3 = l[i3];
        if (slot >= 0) acc += g * (e * (imcj(y0, x3) + imcj(y3, x0)) + imcj(ya, xb) + imcj(yb, xa));
        s[i0] = c * x0 + ze * x3; s[i3] = c * x3 + ze * x0; s[ia] = c * xa + z * xb; s[ib] = c * xb + z * xa;
        l[i0] = c * y0 + ze * y3; l[i3] = c * y3 + ze * y0; l[ia] = c * ya + z * yb; l[ib] = c * yb + z * ya; }
    GRAD(slot, acc);
}
extern "C" __global__ void b_c1q(T* s, T* l, double* gr, ll n, int c, int t,
                                 T v00, T v01, T v10, T v11, T g00, T g01, T g10, T g11, int slot) {
    double acc = 0.0;
    LOOP(n) { ll i0 = ins00(p, c, t) | (1LL << c), i1 = i0 | (1LL << t);
        T a = s[i0], b = s[i1], x = l[i0], d = l[i1];
        if (slot >= 0) acc += imcj(x, g00 * a + g01 * b) + imcj(d, g10 * a + g11 * b);
        s[i0] = v00 * a + v01 * b; s[i1] = v10 * a + v11 * b;
        l[i0] = v00 * x + v01 * d; l[i1] = v10 * x + v11 * d; }
    GRAD(slot, acc);
}
extern "C" __global__ void b_ccx(T* s, T* l, ll n, int c1, int c2, int t) {
    LOOP(n) { ll i0 = ins000(p, c1, c2, t) | (1LL << c1) | (1LL << c2), i1 = i0 | (1LL << t);
        T a = s[i0]; s[i0] = s[i1]; s[i1] = a;
        T x = l[i0]; l[i0] = l[i1]; l[i1] = x; }
}
extern "C" __global__ void b_1q(T* s, T* l, double* gr, ll n, int t,
                                T v00, T v01, T v10, T v11, T g00, T g01, T g10, T g11, int slot) {
    double acc = 0.0;
    LOOP(n) { ll i0 = ins0(p, t), i1 = i0 | (1LL << t);
        T a = s[i0], b = s[i1], c = l[i0], d = l[i1];
        if (slot >= 0) acc += imcj(c, g00 * a + g01 * b) + imcj(d, g10 * a + g11 * b);
        s[i0] = v00 * a + v01 * b; s[i1] = v10 * a + v11 * b;
        l[i0] = v00 * c + v01 * d; l[i1] = v10 * c + v11 * d; }
    GRAD(slot, acc);
}
extern "C" __global__ void b_diag1(T* s, T* l, double* gr, ll n, int t, T e0, T e1, double g0, double g1, int slot) {
    double acc = 0.0;
    LOOP(n) { int bit = (p >> t) & 1; T a = s[p], c = l[p];
        if (slot >= 0) acc += (bit ? g1 : g0) * imcj(c, a);
        T e = bit ? e1 : e0; s[p] = e * a; l[p] = e * c; }
    GRAD(slot, acc);
}
extern "C" __global__ void b_phase1(T* s, T* l, double* gr, ll n, int t, T e, double g, int slot) {
    double acc = 0.0;
    LOOP(n) { ll i = ins0(p, t) | (1LL << t); T a = s[i], c = l[i];
        if (slot >= 0) acc += g * imcj(c, a);
        s[i] = e * a; l[i] = e * c; }
    GRAD(slot, acc);
}
extern "C" __global__ void b_cx(T* s, T* l, ll n, int c, int t) {
    LOOP(n) { ll i0 = ins00(p, c, t) | (1LL << c), i1 = i0 | (1LL << t);
        T a = s[i0]; s[i0] = s[i1]; s[i1] = a;
        T x = l[i0]; l[i0] = l[i1]; l[i1] = x; }
}
extern "C" __global__ void b_cphase(T* s, T* l, double* gr, ll n, int c, int t, T e, double g, int slot) {
    double acc = 0.0;
    LOOP(n) { ll i = ins00(p, c, t) | (1LL << c) | (1LL << t); T a = s[i], x = l[i];
        if (slot >= 0) acc += g * imcj(x, a);
        s[i] = e * a; l[i] = e * x; }
    GRAD(slot, acc);
}
extern "C" __global__ void b_diag2(T* s, T* l, double* gr, ll n, int a, int b, T ee, T eq, double ge, double gq, int slot) {
    double acc = 0.0;
    LOOP(n) { int odd = ((p >> a) ^ (p >> b)) & 1; T x = s[p], y = l[p];
        if (slot >= 0) acc += (odd ? gq : ge) * imcj(y, x);
        T e = odd ? eq : ee; s[p] = e * x; l[p] = e * y; }
    GRAD(slot, acc);
}
extern "C" __global__ void b_swap(T* s, T* l, ll n, int a, int b) {
    LOOP(n) { ll base = ins00(p, a, b), i = base | (1LL << a), j = base | (1LL << b);
        T x = s[i]; s[i] = s[j]; s[j] = x;
        T y = l[i]; l[i] = l[j]; l[j] = y; }
}

#define UNSTEP(V, Gn, slot, acc) \
    if ((slot) >= 0) acc += imcj(c, Gn##0 * a + Gn##1 * b) + imcj(d, Gn##2 * a + Gn##3 * b); \
    { T na = V##0 * a + V##1 * b, nb = V##2 * a + V##3 * b; a = na; b = nb; \
      T nc = V##0 * c + V##1 * d, nd = V##2 * c + V##3 * d; c = nc; d = nd; }
extern "C" __global__ void b_run(T* s, T* l, double* gr, ll n, int t, int m,
        T va0, T va1, T va2, T va3, T ga0, T ga1, T ga2, T ga3, int sa,
        T vb0, T vb1, T vb2, T vb3, T gb0, T gb1, T gb2, T gb3, int sb,
        T vc0, T vc1, T vc2, T vc3, T gc0, T gc1, T gc2, T gc3, int sc) {
    double xa = 0.0, xb = 0.0, xc = 0.0;
    LOOP(n) { ll i0 = ins0(p, t), i1 = i0 | (1LL << t);
        T a = s[i0], b = s[i1], c = l[i0], d = l[i1];
        UNSTEP(va, ga, sa, xa)
        if (m > 1) { UNSTEP(vb, gb, sb, xb) }
        if (m > 2) { UNSTEP(vc, gc, sc, xc) }
        s[i0] = a; s[i1] = b; l[i0] = c; l[i1] = d; }
    GRAD(sa, xa); GRAD(sb, xb); GRAD(sc, xc);
}

#define MAXP 48
__device__ inline T phase_of(double a, int inv) {
    a -= 6.283185307179586 * rint(a * 0.15915494309189535);
    REAL c = cos((REAL)a), sn = sin((REAL)a);
    return T(c, inv ? -sn : sn);
}
__device__ inline double group_angle(ll p, const double* tab, int nch, int chb,
                                     const ll* act, const ll* nbm, const double* ccoef, int ncls,
                                     const ll* cm, const double* cv, int nc, double a0) {
    ll chm = (1LL << chb) - 1;
    double a = a0;
    for (int c = 0; c < nch; c++) a += tab[((ll)c << chb) + ((p >> (c * chb)) & chm)];
    for (int cl = 0; cl < ncls; cl++) {
        int cnt = 0;
        for (ll y = p & act[cl]; y; y &= y - 1) cnt += __popcll(p & nbm[cl * 64 + (__ffsll(y) - 1)]);
        a += ccoef[cl] * cnt;
    }
    for (int g = 0; g < nc; g++) if ((p & cm[g]) == cm[g]) a += cv[g];
    return a;
}
extern "C" __global__ void f_dgroup(T* s, ll n, const double* tab, int nch, int chb, const ll* act, const ll* nbm,
                                    const double* ccoef, int ncls, const ll* cm, const double* cv, int nc, double a0) {
    LOOP(n) { s[p] *= phase_of(group_angle(p, tab, nch, chb, act, nbm, ccoef, ncls, cm, cv, nc, a0), 0); }
}
extern "C" __global__ void b_dgroup(T* s, T* l, double* gr, ll n, const double* tab, int nch, int chb,
                                    const ll* act, const ll* nbm, const double* ccoef, int ncls,
                                    const ll* cm, const double* cv, int nc, double a0,
                                    const ll* pm, const int* pty, const double* g0, const double* g1, const int* slot, int np) {
    double acc[MAXP + 1];
    for (int j = 0; j <= np; j++) acc[j] = 0.0;
    LOOP(n) { T x = s[p], y = l[p];
        if (np) { double w = imcj(y, x); acc[0] += w;
            for (int j = 0; j < np; j++)
                if (pty[j] ? ((p & pm[j]) == pm[j]) : (__popcll(p & pm[j]) & 1)) acc[1 + j] += w; }
        T e = phase_of(group_angle(p, tab, nch, chb, act, nbm, ccoef, ncls, cm, cv, nc, a0), 1);
        s[p] = e * x; l[p] = e * y; }
    for (int j = 0; j < np; j++) grad_add(gr, slot[j], g0[j] * acc[0] + g1[j] * acc[1 + j]);
}

extern "C" __global__ void f_scal(T* s, ll n, T d) {
    LOOP(n) { s[p] *= d; }
}
extern "C" __global__ void b_scal(T* s, T* l, double* gr, ll n, T e, double g, int slot) {
    double acc = 0.0;
    LOOP(n) { T a = s[p], c = l[p];
        if (slot >= 0) acc += g * imcj(c, a);
        s[p] = e * a; l[p] = e * c; }
    GRAD(slot, acc);
}

extern "C" __global__ void vdot_re(const T* a, const T* b, double* part, ll n) {
    double acc = 0.0;
    LOOP(n) { acc += (double)a[p].real() * b[p].real() + (double)a[p].imag() * b[p].imag(); }
    part[tid] = acc;
}

extern "C" __global__ void obs_term(T* l, const T* s, ll n, ll xmask, ll zmask, T coeff, int first) {
    LOOP(n) { ll j = p ^ xmask;
        T v = coeff * s[j];
        if (__popcll(j & zmask) & 1) v = -v;
        if (first) l[p] = v; else l[p] += v; }
}
extern "C" __global__ void obs_group(T* l, const T* s, ll n, ll xmask, const ll* zm, const T* c, int nt, int first) {
    LOOP(n) { ll j = p ^ xmask;
        T w(0, 0);
        for (int t = 0; t < nt; t++) { if (__popcll(j & zm[t]) & 1) w -= c[t]; else w += c[t]; }
        T v = w * s[j];
        if (first) l[p] = v; else l[p] += v; }
}
__device__ inline double recj(T l, T p) { return (double)l.real() * p.real() + (double)l.imag() * p.imag(); }
__device__ inline void sum_add(double* gr, int slot, double acc) {
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(gr + slot, acc);
}
extern "C" __global__ void b_obs_group(T* l, const T* s, double* gr, ll n, ll xmask, const ll* zm, const T* c, int nt, int slot, int first) {
    double acc = 0.0;
    LOOP(n) { ll j = p ^ xmask;
        T w(0, 0);
        for (int t = 0; t < nt; t++) { if (__popcll(j & zm[t]) & 1) w -= c[t]; else w += c[t]; }
        T v = w * s[j];
        if (first) l[p] = v; else l[p] += v;
        acc += recj(s[p], v); }
    sum_add(gr, slot, acc);
}
extern "C" __global__ void b_obs_x(T* l, const T* sc, const T* sl, double* gr, ll n, ll q0, ll xlo, ll zmask, T coeff, int slot) {
    double acc = 0.0;
    LOOP(n) { T v = coeff * sc[p];
        if (__popcll((q0 + p) & zmask) & 1) v = -v;
        l[p ^ xlo] += v; acc += recj(sl[p ^ xlo], v); }
    sum_add(gr, slot, acc);
}
extern "C" __global__ void obs_x(T* l, const T* sc, ll n, ll q0, ll xlo, ll zmask, T coeff) {
    LOOP(n) { T v = coeff * sc[p];
        if (__popcll((q0 + p) & zmask) & 1) v = -v;
        l[p ^ xlo] += v; }
}
'''

OBS_K = 4


def _obs_window_source(k: int) -> str:
    D = 1 << k
    src = [f'extern "C" __global__ void obs_w{k}(T* l, const T* s, ll n, ' + ', '.join(f'int p{j}' for j in range(k)) + ', const T* c, int first) {']
    src += [f'const T c{j}_0 = c[{2 * j}], c{j}_1 = c[{2 * j + 1}];' for j in range(k)]
    src += [f'const ll o{b} = ' + (' | '.join(f'(1LL << p{j})' for j in range(k) if (b >> j) & 1) or '0') + ';' for b in range(D)]
    src += ['LOOP(n) { ll base = p; ' + ' '.join(f'base = ins0(base, p{j});' for j in range(k))]
    src += [f'T a{b} = s[base | o{b}], y{b} = first ? T(0, 0) : l[base | o{b}];' for b in range(D)]
    for j in range(k):
        src += [f'y{b} += c{j}_0 * a{b | (1 << j)}; y{b | (1 << j)} += c{j}_1 * a{b};' for b in range(D) if not (b >> j) & 1]
    src += [f'l[base | o{b}] = y{b};' for b in range(D)] + ['} }']
    return '\n'.join(src) + '\n'


def _late_window_source(k: int) -> str:
    D = 1 << k
    src = [f'extern "C" __global__ void b_obs_w{k}(T* l, const T* s, double* gr, ll n, ' + ', '.join(f'int p{j}' for j in range(k)) + ', const T* c, int slot, int first) {']
    src += [f'const T c{j}_0 = c[{2 * j}], c{j}_1 = c[{2 * j + 1}];' for j in range(k)]
    src += [f'const ll o{b} = ' + (' | '.join(f'(1LL << p{j})' for j in range(k) if (b >> j) & 1) or '0') + ';' for b in range(D)]
    src += ['double acc = 0.0;', 'LOOP(n) { ll base = p; ' + ' '.join(f'base = ins0(base, p{j});' for j in range(k))]
    src += [f'T a{b} = s[base | o{b}], y{b} = first ? T(0, 0) : l[base | o{b}];' for b in range(D)]
    for j in range(k):
        src += [f'{{ T u = c{j}_0 * a{b | (1 << j)}, w = c{j}_1 * a{b}; y{b} += u; y{b | (1 << j)} += w; acc += recj(a{b}, u) + recj(a{b | (1 << j)}, w); }}'
                for b in range(D) if not (b >> j) & 1]
    src += [f'l[base | o{b}] = y{b};' for b in range(D)] + ['}', 'sum_add(gr, slot, acc);', '}']
    return '\n'.join(src) + '\n'


_SRC += ''.join(_obs_window_source(k) for k in range(1, OBS_K + 1)) + ''.join(_late_window_source(k) for k in range(1, OBS_K + 1))

_KERNELS = ['f_1q', 'f_diag1', 'f_phase1', 'f_cx', 'f_cphase', 'f_diag2', 'f_swap', 'f_c1q', 'f_ccx', 'f_pp',
            'b_1q', 'b_diag1', 'b_phase1', 'b_cx', 'b_cphase', 'b_diag2', 'b_swap', 'b_c1q', 'b_ccx', 'b_pp',
            'b_run', 'f_dgroup', 'b_dgroup', 'obs_term', 'vdot_re', 'f_scal', 'b_scal', 'obs_x', 'obs_group', 'b_obs_group', 'b_obs_x'] \
    + [f'obs_w{k}' for k in range(1, OBS_K + 1)] + [f'b_obs_w{k}' for k in range(1, OBS_K + 1)]
MAXP = 48
_BACKWARD_THREADS = 1 << 19


class _Windows:
    def __getattr__(self, name):
        from . import windows
        return getattr(windows, name)


_WIN = _Windows()


def _csv():
    from . import custatevec_backend
    return custatevec_backend


def _alt():
    if _WIN.KERNEL == 'cublas':
        from . import cublas_backend
        return cublas_backend
    return _csv()

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)
_PHASE = {'z': -1.0, 's': 1j, 'sdg': -1j, 't': np.exp(1j * np.pi / 4), 'tdg': np.exp(-1j * np.pi / 4)}
_ZERO4 = np.zeros(4, dtype=np.complex128)


def _rot(axis: str, th: float) -> np.ndarray:
    c, s = np.cos(th / 2), np.sin(th / 2)
    if axis == 'x':
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def lower(g: Gate) -> Tuple:
    n = g.name.lower()
    th = g.params[0] if g.params else 0.0
    sc = g.params[1] if (len(g.params) == 3 and n != 'dg1') else 1.0
    if n == 'h':
        return '1q', (_H, None)
    if n == 'x':
        return '1q', (_X, None)
    if n == 'y':
        return '1q', (_Y, None)
    if n == 'sx':
        return '1q', (_SX, None)
    if n in ('rx', 'frx'):
        return '1q', (_rot('x', th), sc * _X / 2 if n == 'rx' else None)
    if n in ('ry', 'fry'):
        return '1q', (_rot('y', th), sc * _Y / 2 if n == 'ry' else None)
    if n in ('rz', 'frz'):
        return 'diag1', (np.exp(-0.5j * th), np.exp(0.5j * th), *((0.5 * sc, -0.5 * sc) if n == 'rz' else (None, None)))
    if n in ('p', 'u1'):
        return 'phase1', (np.exp(1j * th), -1.0 * sc if g.params else None)
    if n in _PHASE:
        return 'phase1', (complex(_PHASE[n]), None)
    if n in ('cx', 'cnot'):
        return 'cx', ()
    if n == 'cz':
        return 'cphase', (-1.0 + 0j, None)
    if n == 'cp':
        return 'cphase', (np.exp(1j * th), -1.0 * sc)
    if n == 'rzz':
        return 'diag2', (np.exp(-0.5j * th), np.exp(0.5j * th), 0.5 * sc, -0.5 * sc)
    if n == 'swap':
        return 'swap', ()
    if n in ('crx', 'cry'):
        return 'c1q', (_rot(n[-1], th), sc * (_X if n == 'crx' else _Y) / 2 if g.params else None)
    if n == 'cy':
        return 'c1q', (_Y, None)
    if n == 'ch':
        return 'c1q', (_H, None)
    if n in ('ccx', 'toffoli'):
        return 'ccx', ()
    if n in ('rxx', 'ryy'):
        return 'pp2', (np.cos(th / 2), np.sin(th / 2), 1 if n == 'rxx' else -1, 0.5 * sc)
    if n == 'dg1':
        p = g.params
        return 'diag1', (complex(p[0], p[1]), complex(p[2], p[3]), p[4], p[5])
    raise ValueError(f'unsupported gate: {g.name}')


TOUCH = {'1q': 1.0, 'diag1': 1.0, 'phase1': 0.5, 'cx': 0.5, 'cphase': 0.25, 'diag2': 1.0, 'swap': 0.5, 'c1q': 0.5, 'ccx': 0.25, 'pp2': 1.0}

PauliTerm = Tuple[float, Dict[int, str]]


class LiraEngine:
    def __init__(self, n: int, dtype=np.complex64, max_threads: int = 1 << 22):
        self.n = n
        self.dtype = np.dtype(dtype)
        self.ct = self.dtype.type
        self.rt = np.float32 if self.dtype == np.complex64 else np.float64
        self.real = real = 'float' if self.dtype == np.complex64 else 'double'
        mod = cp.RawModule(code=_SRC.replace('REAL', real))
        self.k = {name: mod.get_function(name) for name in _KERNELS}
        self.max_threads = max_threads
        self.part = cp.zeros(max_threads, dtype=np.float64)
        self._ops = {}
        self._obs = {}
        self._psi = self._lam = None
        self._out_of_place_ok = False

    def allocate(self):
        if self._psi is None:
            self._psi = cp.empty(1 << self.n, dtype=self.dtype)
            self._lam = cp.empty(1 << self.n, dtype=self.dtype)
        return self._psi, self._lam

    def release(self):
        self._psi = self._lam = None
        if _WIN.KERNEL == 'cublas':
            _alt().release()
        cp.get_default_memory_pool().free_all_blocks()

    def _launch(self, name: str, n_items: int, args: tuple):
        if name[0] == 'b':
            grid = (min(n_items, _BACKWARD_THREADS) + 255) // 256
            self.k[name]((grid,), (256,), args)
            return grid * 256
        threads = min(n_items, self.max_threads)
        block = min(256, threads)
        grid = (threads + block - 1) // block
        self.k[name]((grid,), (block,), args)
        return grid * block

    def _c(self, m) -> tuple:
        return tuple(self.ct(x) for x in np.asarray(m).reshape(-1))

    def _forward_gate(self, s, g: Gate, k: int):
        kind, pl = lower(g)
        q = tuple(np.int32(t) for t in g.targets)
        N = np.int64(1 << k)
        if kind == '1q':
            self._launch('f_1q', 1 << (k - 1), (s, N >> 1, q[0], *self._c(pl[0])))
        elif kind == 'diag1':
            self._launch('f_diag1', 1 << k, (s, N, q[0], self.ct(pl[0]), self.ct(pl[1])))
        elif kind == 'phase1':
            self._launch('f_phase1', 1 << (k - 1), (s, N >> 1, q[0], self.ct(pl[0])))
        elif kind == 'cx':
            self._launch('f_cx', 1 << (k - 2), (s, N >> 2, q[0], q[1]))
        elif kind == 'cphase':
            self._launch('f_cphase', 1 << (k - 2), (s, N >> 2, q[0], q[1], self.ct(pl[0])))
        elif kind == 'diag2':
            self._launch('f_diag2', 1 << k, (s, N, q[0], q[1], self.ct(pl[0]), self.ct(pl[1])))
        elif kind == 'swap':
            self._launch('f_swap', 1 << (k - 2), (s, N >> 2, q[0], q[1]))
        elif kind == 'c1q':
            self._launch('f_c1q', 1 << (k - 2), (s, N >> 2, q[0], q[1], *self._c(pl[0])))
        elif kind == 'ccx':
            self._launch('f_ccx', 1 << (k - 3), (s, N >> 3, q[0], q[1], q[2]))
        elif kind == 'pp2':
            self._launch('f_pp', 1 << (k - 2), (s, N >> 2, q[0], q[1], self.rt(pl[0]), self.rt(pl[1]), np.int32(pl[2])))

    def _backward_gate(self, s, l, gr, g: Gate, k: int, slot: int):
        kind, pl = lower(g)
        q = tuple(np.int32(t) for t in g.targets)
        N = np.int64(1 << k)
        sl = np.int32(slot)
        if kind == '1q':
            U, G = pl
            G = G if (slot >= 0 and G is not None) else _ZERO4
            self._launch('b_1q', 1 << (k - 1), (s, l, gr, N >> 1, q[0], *self._c(U.conj().T), *self._c(G), sl))
        elif kind == 'diag1':
            d0, d1, g0, g1 = pl
            self._launch('b_diag1', 1 << k, (s, l, gr, N, q[0], self.ct(np.conj(d0)), self.ct(np.conj(d1)),
                                             np.float64(g0 or 0.0), np.float64(g1 or 0.0), sl))
        elif kind == 'phase1':
            d, gv = pl
            self._launch('b_phase1', 1 << (k - 1), (s, l, gr, N >> 1, q[0], self.ct(np.conj(d)), np.float64(gv or 0.0), sl))
        elif kind == 'cx':
            self._launch('b_cx', 1 << (k - 2), (s, l, N >> 2, q[0], q[1]))
        elif kind == 'cphase':
            d, gv = pl
            self._launch('b_cphase', 1 << (k - 2), (s, l, gr, N >> 2, q[0], q[1], self.ct(np.conj(d)),
                                                    np.float64(gv or 0.0), sl))
        elif kind == 'diag2':
            de, dq, ge, gq = pl
            self._launch('b_diag2', 1 << k, (s, l, gr, N, q[0], q[1], self.ct(np.conj(de)), self.ct(np.conj(dq)),
                                             np.float64(ge), np.float64(gq), sl))
        elif kind == 'swap':
            self._launch('b_swap', 1 << (k - 2), (s, l, N >> 2, q[0], q[1]))
        elif kind == 'c1q':
            U, G = pl
            G = G if (slot >= 0 and G is not None) else _ZERO4
            self._launch('b_c1q', 1 << (k - 2), (s, l, gr, N >> 2, q[0], q[1], *self._c(U.conj().T), *self._c(G), sl))
        elif kind == 'ccx':
            self._launch('b_ccx', 1 << (k - 3), (s, l, N >> 3, q[0], q[1], q[2]))
        elif kind == 'pp2':
            self._launch('b_pp', 1 << (k - 2), (s, l, gr, N >> 2, q[0], q[1], self.rt(pl[0]), self.rt(pl[1]), np.int32(pl[2]),
                                                np.float64(pl[3]), sl))

    def _inverse_gate(self, s, g: Gate, k: int):
        kind, pl = lower(g)
        q = tuple(np.int32(t) for t in g.targets)
        N = np.int64(1 << k)
        if kind == '1q':
            self._launch('f_1q', 1 << (k - 1), (s, N >> 1, q[0], *self._c(pl[0].conj().T)))
        elif kind == 'diag1':
            self._launch('f_diag1', 1 << k, (s, N, q[0], self.ct(np.conj(pl[0])), self.ct(np.conj(pl[1]))))
        elif kind == 'phase1':
            self._launch('f_phase1', 1 << (k - 1), (s, N >> 1, q[0], self.ct(np.conj(pl[0]))))
        elif kind == 'cx':
            self._launch('f_cx', 1 << (k - 2), (s, N >> 2, q[0], q[1]))
        elif kind == 'cphase':
            self._launch('f_cphase', 1 << (k - 2), (s, N >> 2, q[0], q[1], self.ct(np.conj(pl[0]))))
        elif kind == 'diag2':
            self._launch('f_diag2', 1 << k, (s, N, q[0], q[1], self.ct(np.conj(pl[0])), self.ct(np.conj(pl[1]))))
        elif kind == 'swap':
            self._launch('f_swap', 1 << (k - 2), (s, N >> 2, q[0], q[1]))
        elif kind == 'c1q':
            self._launch('f_c1q', 1 << (k - 2), (s, N >> 2, q[0], q[1], *self._c(pl[0].conj().T)))
        elif kind == 'ccx':
            self._launch('f_ccx', 1 << (k - 3), (s, N >> 3, q[0], q[1], q[2]))
        elif kind == 'pp2':
            self._launch('f_pp', 1 << (k - 2), (s, N >> 2, q[0], q[1], self.rt(pl[0]), self.rt(-pl[1]), np.int32(pl[2])))

    def _lam_outside(self, lam, g: Gate, k: int):
        for j in range(max(k, len(g.targets)), self.n):
            self._inverse_gate(lam[1 << j:1 << (j + 1)], g, j)

    def fused_ops(self, plan: Plan, margin: float = 1.0, window=0):
        from .windows import config
        key = (id(plan), margin, config(window))
        if key not in self._ops:
            from .fusion import fuse
            self._ops[key] = ops = fuse(plan, margin=margin, window=window)
            self.prepare_windows(ops)
        return self._ops[key]

    def prepare_windows(self, ops):
        from .windows import kernels, launch_layout, window_tables
        wins = [op for op in ops if op.kind == 'window']
        for op in wins:
            if 'whost' not in op.cache:
                op.cache['whost'] = window_tables(op)
        if wins and _WIN.KERNEL in ('custatevec', 'cublas'):
            _alt().prepare(self, wins)
        elif wins:
            kernels([(op.cache['whost']['shape'], launch_layout(op.cache['whost'], op.width, self.real, kind) is not None)
                     for op in wins for kind in ('forward', 'backward')], self.real)

    def _win(self, op):
        if 'wdev' not in op.cache:
            from .windows import in_use, kernels, launch_layout, tiled_warps, window_tables
            if 'whost' not in op.cache:
                op.cache['whost'] = window_tables(op)
            h = op.cache['whost']
            got = []
            for j, kind in enumerate(('forward', 'backward')):
                lay = launch_layout(h, op.width, self.real, kind)
                spec = (h['shape'], lay is not None)
                kern = kernels([spec], self.real)[spec][j]
                if lay is None:
                    ints, tl = tuple(np.int32(q) for q in sorted(h['qs'])) + tuple(np.int32(q) for q in h['qs']), None
                else:
                    t, nwmask, highs = lay
                    ints = (np.int32(t), np.int64(nwmask), np.int32(len(highs)), *[np.int32(q) for q in highs + [0] * (4 - len(highs))],
                            *[np.int32(q) for q in h['qs']])
                    tl = (1 << (op.width - t - len(highs)), tiled_warps(len(h['qs']), self.real)[j])
                got.append((kern, ints, tl))
            if in_use(self.real)['mat'] == 'param':
                ct = np.dtype(self.dtype).type
                m = tuple(ct(x) for x in h['m'][:4 * h['nseg']])
                v = tuple(ct(x) for x in h['v'][:4 * h['nseg']])
            else:
                m, v = (cp.asarray(h['m'].astype(self.dtype)),), (cp.asarray(h['v'].astype(self.dtype)),)
            (fw, fints, ftl), (bw, bints, btl) = got
            op.cache['wdev'] = (fw, bw, len(h['qs']), fints, m, v, cp.asarray(h['coef']), cp.asarray(h['slot']), cp.asarray(h['off']),
                                ftl, bints, btl)
        return op.cache['wdev']

    def _wlaunch(self, op):
        if _WIN.KERNEL in ('custatevec', 'cublas'):
            op.cache['wf'] = op.cache['wb'] = ()
            return (), ()
        fw, bw, kk, fints, m, v, coef, slot, off, ftl, bints, btl = self._win(op)
        items = 1 << (op.width - kk)
        if ftl is None:
            threads = min(items, self.max_threads)
            block = min(256, threads)
            wf = (fw, ((threads + block - 1) // block,), (block,), (np.int64(items), *fints, *m))
        else:
            wf = (fw, (min((ftl[0] + ftl[1] - 1) // ftl[1], max(1, _WIN.TILED_FORWARD_THREADS // (32 * ftl[1]))),), (32 * ftl[1],),
                  (np.int64(ftl[0]), *fints, *m))
        bt = _WIN.BACKWARD_THREADS
        if btl is None:
            wb = (bw, ((min(items, bt) + 255) // 256,), (256,), (np.int64(items), *bints, *v, coef, slot, off))
        else:
            wb = (bw, (min((btl[0] + btl[1] - 1) // btl[1], max(1, bt // (32 * btl[1]))),), (32 * btl[1],),
                  (np.int64(btl[0]), *bints, *v, coef, slot, off))
        op.cache['wf'], op.cache['wb'] = wf, wb
        return wf, wb

    def _diag_dev(self, op):
        if 'dev' not in op.cache:
            h = op.cache['host']
            op.cache['dev'] = {k: cp.asarray(h[k]) for k in ('tab', 'act', 'nbm', 'ccoef', 'cm', 'cv', 'pm', 'pty', 'g0', 'g1', 'slot')}
        return op.cache['dev'], op.cache['host']

    def _run_args(self, op):
        if 'args' not in op.cache:
            from .fusion import gen1q, mat1q
            mats = [mat1q(g) for g in op.gates]
            M = np.eye(2, dtype=np.complex128)
            for U in mats:
                M = U @ M
            back = []
            for j in range(3):
                if j < len(op.gates):
                    g, p = op.gates[-1 - j], op.pidx[-1 - j]
                    back += [*self._c(mats[-1 - j].conj().T), *self._c(gen1q(g) if p >= 0 else _ZERO4), np.int32(p)]
                else:
                    back += [*self._c(_ZERO4), *self._c(_ZERO4), np.int32(-1)]
            op.cache['args'] = (self._c(M), tuple(back))
        return op.cache['args']

    def _forward_op(self, s, op):
        k = op.width
        if op.kind == 'gate':
            self._forward_gate(s, op.gates[0], k)
        elif op.kind == 'run':
            self._launch('f_1q', 1 << (k - 1), (s, np.int64(1 << (k - 1)), np.int32(op.gates[0].targets[0]), *self._run_args(op)[0]))
        elif op.kind == 'window':
            w = op.cache['wf'] if 'wf' in op.cache else self._wlaunch(op)[0]
            if not w:
                return _alt().forward(self, s, op)
            w[0](w[1], w[2], (s,) + w[3])
        else:
            d, h = self._diag_dev(op)
            self._launch('f_dgroup', 1 << k, (s, np.int64(1 << k), d['tab'], np.int32(h['nch']), np.int32(h['chb']),
                                              d['act'], d['nbm'], d['ccoef'], np.int32(h['ncls']),
                                              d['cm'], d['cv'], np.int32(h['nc']), np.float64(h['a0'])))

    def _backward_op(self, s, l, gr, op):
        k = op.width
        if op.kind == 'gate':
            self._backward_gate(s, l, gr, op.gates[0], k, op.pidx[0])
        elif op.kind == 'run':
            self._launch('b_run', 1 << (k - 1), (s, l, gr, np.int64(1 << (k - 1)), np.int32(op.gates[0].targets[0]),
                                                 np.int32(len(op.gates)), *self._run_args(op)[1]))
        elif op.kind == 'window':
            w = op.cache['wb'] if 'wb' in op.cache else self._wlaunch(op)[1]
            if not w:
                return _alt().backward(self, s, l, gr, op)
            w[0](w[1], w[2], (s, l, gr) + w[3])
        else:
            d, h = self._diag_dev(op)
            self._launch('b_dgroup', 1 << k, (s, l, gr, np.int64(1 << k), d['tab'], np.int32(h['nch']), np.int32(h['chb']),
                                              d['act'], d['nbm'], d['ccoef'], np.int32(h['ncls']),
                                              d['cm'], d['cv'], np.int32(h['nc']), np.float64(h['a0']),
                                              d['pm'], d['pty'], d['g0'], d['g1'], d['slot'], np.int32(h['np'])))

    def _obs_sweeps(self, terms: Sequence[PauliTerm]):
        key = tuple((float(c), tuple(sorted(ps.items()))) for c, ps in terms)
        if key not in self._obs:
            groups, single = {}, {}
            for coeff, paulis in terms:
                xm = zm = ny = 0
                for bit, pch in paulis.items():
                    if pch in ('X', 'Y'):
                        xm |= 1 << bit
                    if pch in ('Z', 'Y'):
                        zm |= 1 << bit
                    ny += pch == 'Y'
                if len(paulis) == 1 and xm:
                    f = single.setdefault(xm.bit_length() - 1, [0j, 0j])
                    f[0] += -1j * coeff if ny else coeff
                    f[1] += 1j * coeff if ny else coeff
                else:
                    groups.setdefault(xm, []).append((zm, coeff * (1j) ** ny))
            sweeps = [('group', np.int64(xm), cp.asarray(np.array([z for z, _ in g], dtype=np.int64)),
                       cp.asarray(np.array([c for _, c in g]).astype(self.dtype)), np.int32(len(g))) for xm, g in sorted(groups.items())]
            qs = sorted(single)
            for i in range(0, len(qs), OBS_K):
                part = qs[i:i + OBS_K]
                sweeps.append(('window', tuple(np.int32(q) for q in part),
                               cp.asarray(np.array([x for q in part for x in single[q]]).astype(self.dtype))))
            self._obs[key] = sweeps
        return self._obs[key]

    def _apply_observable_fused(self, lam, psi, terms: Sequence[PauliTerm]):
        N = 1 << self.n
        for i, sw in enumerate(self._obs_sweeps(terms)):
            first = np.int32(i == 0)
            if sw[0] == 'group':
                self._launch('obs_group', N, (lam, psi, np.int64(N), sw[1], sw[2], sw[3], sw[4], first))
            else:
                k = len(sw[1])
                self._launch(f'obs_w{k}', N >> k, (lam, psi, np.int64(N >> k), *sw[1], sw[2], first))

    def _inject(self, lam, psi, gr, terms: Sequence[PauliTerm], width: int, slot: int, first: bool = False):
        N = 1 << width

        def launch(name, items, args):
            self.k[name](((min(items, self.max_threads) + 255) // 256,), (256,), args)
        for i, sw in enumerate(self._obs_sweeps(terms)):
            f = np.int32(first and i == 0)
            if sw[0] == 'group':
                launch('b_obs_group', N, (lam, psi, gr, np.int64(N), sw[1], sw[2], sw[3], sw[4], np.int32(slot), f))
            else:
                k = len(sw[1])
                launch(f'b_obs_w{k}', N >> k, (lam, psi, gr, np.int64(N >> k), *sw[1], sw[2], np.int32(slot), f))

    def _apply_observable(self, lam, psi, terms: Sequence[PauliTerm]):
        N = 1 << self.n
        for i, (coeff, paulis) in enumerate(terms):
            xm = zm = ny = 0
            for bit, pch in paulis.items():
                if pch in ('X', 'Y'):
                    xm |= 1 << bit
                if pch in ('Z', 'Y'):
                    zm |= 1 << bit
                ny += pch == 'Y'
            self._launch('obs_term', N, (lam, psi, np.int64(N), np.int64(xm), np.int64(zm),
                                         self.ct(coeff * (1j) ** ny), np.int32(i == 0)))

    def run(self, plan: Plan, terms: Sequence[PauliTerm], backward: bool = True, fuse: bool = True,
            margin: float = 1.0, init=None, window=WINDOW, fuse_obs: bool = None, late_obs: bool = False,
            lam_full: bool = False):
        assert plan.n == self.n
        if fuse_obs is None:
            fuse_obs = not late_obs
        assert not (lam_full and (fuse or late_obs)), 'lam_full is defined for the per-gate path only'
        sync = cp.cuda.get_current_stream().synchronize
        t = {}
        ops = self.fused_ops(plan, margin, window) if fuse else None

        psi, lam = self.allocate()
        self._out_of_place_ok = True
        sync(); t0 = time.perf_counter()
        if init is None:
            psi.fill(0)
            psi[0] = 1.0
        else:
            psi[...] = init
        if fuse:
            for op in ops:
                r = self._forward_op(psi, op)
                if r is not None:
                    psi = r
        else:
            for g, k in zip(plan.gates, plan.widths):
                self._forward_gate(psi, g, int(k))
        sync(); t['forward'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        late = None
        if late_obs and backward:
            units = [(op.qubits, op.width) for op in ops] if fuse else [(g.targets, int(k)) for g, k in zip(plan.gates, plan.widths)]
            late, cost = late_plan([u[0] for u in units], [u[1] for u in units], terms, start_is_zero=init is None)
            end = len(units) - 1
            fresh = end in late
            if not fresh:
                lam.fill(0)
        else:
            if fuse_obs:
                self._apply_observable_fused(lam, psi, terms)
            else:
                self._apply_observable(lam, psi, terms)
            N = 1 << self.n
            used = self._launch('vdot_re', N, (psi, lam, self.part, np.int64(N)))
            cost = float(self.part[:used].sum())
        sync(); t['observable'] = time.perf_counter() - t0
        self._psi, self._lam = psi, lam
        if not backward:
            self._out_of_place_ok = False
            return cost, None, t

        t0 = time.perf_counter()
        P = plan.n_params
        gr = cp.zeros(P + 1, dtype=np.float64)
        for i in range((len(ops) if fuse else len(plan.gates)) - 1, -1, -1):
            if late and i in late:
                self._inject(lam, psi, gr, late[i], units[i][1], P, first=fresh and i == end)
            if fuse:
                r = self._backward_op(psi, lam, gr, ops[i])
                if r is not None:
                    psi, lam = r
            else:
                self._backward_gate(psi, lam, gr, plan.gates[i], int(plan.widths[i]), int(plan.param_index[i]))
                if lam_full and plan.widths[i] < self.n:
                    self._lam_outside(lam, plan.gates[i], int(plan.widths[i]))
        if late and -1 in late:
            self._inject(lam, psi, gr, late[-1], self.n, P)
        self._psi, self._lam = psi, lam
        self._out_of_place_ok = False
        out = gr.get()
        grads = out[:P]
        if late is not None:
            cost += float(out[P])
        sync(); t['backward'] = time.perf_counter() - t0
        t['total'] = t['forward'] + t['observable'] + t['backward']
        return cost, grads, t


def n_sweeps(terms: Sequence[PauliTerm]) -> int:
    groups, single = set(), set()
    for _, paulis in terms:
        xy = [q for q, p in paulis.items() if p in ('X', 'Y')]
        if len(paulis) == 1 and xy:
            single.add(xy[0])
        else:
            groups.add(tuple(sorted(xy)))
    return len(groups) + (len(single) + OBS_K - 1) // OBS_K


def late_schedule(touched, widths, terms: Sequence[PauliTerm], start_is_zero: bool = True, comm=None, only: int = None):
    last = {}
    for i, qs in enumerate(touched):
        for q in (qs or ()):
            last[q] = i
    points = [i for i, qs in enumerate(touched) if qs is not None]
    out, c0 = {}, 0.0
    for coeff, paulis in terms:
        tp = max((last.get(q, -1) for q in paulis), default=-1)
        if tp < 0:
            if not start_is_zero:
                out.setdefault(-1, []).append((coeff, paulis))
            elif all(p == 'Z' for p in paulis.values()):
                c0 += coeff
            continue
        cands = [only] if only is not None else [i for i in points if i >= tp]
        need = {i: comm(i, paulis) if comm else False for i in cands}
        i = min((i for i in cands if need[i] is not None), key=lambda i: (need[i], widths[i], -i))
        if start_is_zero:
            if any(p != 'Z' and q >= widths[i] for q, p in paulis.items()):
                continue
            paulis = {q: p for q, p in paulis.items() if q < widths[i]}
        out.setdefault(i, []).append((coeff, paulis))
    return out, c0


def late_plan(touched, widths, terms: Sequence[PauliTerm], start_is_zero: bool = True):
    late, c0 = late_schedule(touched, widths, terms, start_is_zero)
    end = len(touched) - 1
    if end >= 0 and sum(n_sweeps(b) * 2.0 ** (widths[i] if i >= 0 else widths[end]) for i, b in late.items()) \
            >= n_sweeps([x for b in late.values() for x in b]) * 2.0 ** widths[end]:
        late, c0 = late_schedule(touched, widths, terms, start_is_zero, only=end)
    return late, c0


def traffic(plan: Plan) -> Dict[str, float]:
    fwd = bwd = 0.0
    for g, k in zip(plan.gates, plan.widths):
        kind, _ = lower(g)
        x = TOUCH[kind] * 2.0 ** (int(k) - plan.n)
        fwd += x
        bwd += 2 * x
    return dict(forward=fwd, backward=bwd)
