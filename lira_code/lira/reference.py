from typing import Sequence

import numpy as np

from lira.gates import Gate
from .engine import PauliTerm, lower
from .planner import Plan, is_parametric


def apply_np(s: np.ndarray, g: Gate, inverse: bool = False) -> None:
    kind, pl = lower(g)
    idx = np.arange(len(s))
    bit = lambda q: (idx >> q) & 1
    cj = np.conj if inverse else (lambda x: x)
    if kind == '1q':
        U = pl[0].conj().T if inverse else pl[0]
        t = g.targets[0]
        i0 = idx[bit(t) == 0]
        i1 = i0 | (1 << t)
        a, b = s[i0].copy(), s[i1].copy()
        s[i0], s[i1] = U[0, 0] * a + U[0, 1] * b, U[1, 0] * a + U[1, 1] * b
    elif kind == 'diag1':
        s *= np.where(bit(g.targets[0]) == 1, cj(pl[1]), cj(pl[0]))
    elif kind == 'phase1':
        s[bit(g.targets[0]) == 1] *= cj(pl[0])
    elif kind == 'cx':
        c, t = g.targets
        i0 = idx[(bit(c) == 1) & (bit(t) == 0)]
        i1 = i0 | (1 << t)
        s[i0], s[i1] = s[i1].copy(), s[i0].copy()
    elif kind == 'cphase':
        c, t = g.targets
        s[(bit(c) == 1) & (bit(t) == 1)] *= cj(pl[0])
    elif kind == 'diag2':
        a, b = g.targets
        s *= np.where((bit(a) ^ bit(b)) == 1, cj(pl[1]), cj(pl[0]))
    elif kind == 'swap':
        a, b = g.targets
        i = idx[(bit(a) == 1) & (bit(b) == 0)]
        j = (i ^ (1 << a)) | (1 << b)
        s[i], s[j] = s[j].copy(), s[i].copy()
    elif kind == 'c1q':
        U = pl[0].conj().T if inverse else pl[0]
        c, t = g.targets
        i0 = idx[(bit(c) == 1) & (bit(t) == 0)]
        i1 = i0 | (1 << t)
        a, b = s[i0].copy(), s[i1].copy()
        s[i0], s[i1] = U[0, 0] * a + U[0, 1] * b, U[1, 0] * a + U[1, 1] * b
    elif kind == 'ccx':
        c1, c2, t = g.targets
        i0 = idx[(bit(c1) == 1) & (bit(c2) == 1) & (bit(t) == 0)]
        i1 = i0 | (1 << t)
        s[i0], s[i1] = s[i1].copy(), s[i0].copy()
    elif kind == 'pp2':
        a, b = g.targets
        c, sn, e = pl[0], pl[1] * (-1 if inverse else 1), pl[2]
        flip = s[idx ^ (1 << a) ^ (1 << b)] * np.where(bit(a) == bit(b), e, 1)
        s[:] = c * s - 1j * sn * flip


def apply_generator(s: np.ndarray, g: Gate) -> np.ndarray:
    kind, pl = lower(g)
    idx = np.arange(len(s))
    bit = lambda q: (idx >> q) & 1
    out = np.zeros_like(s)
    if kind == '1q':
        G, t = pl[1], g.targets[0]
        i0 = idx[bit(t) == 0]
        i1 = i0 | (1 << t)
        out[i0] = G[0, 0] * s[i0] + G[0, 1] * s[i1]
        out[i1] = G[1, 0] * s[i0] + G[1, 1] * s[i1]
    elif kind == 'diag1':
        out = s * np.where(bit(g.targets[0]) == 1, pl[3], pl[2])
    elif kind == 'phase1':
        out = s * np.where(bit(g.targets[0]) == 1, pl[1], 0.0)
    elif kind == 'cphase':
        c, t = g.targets
        out = s * np.where((bit(c) == 1) & (bit(t) == 1), pl[1], 0.0)
    elif kind == 'diag2':
        a, b = g.targets
        out = s * np.where((bit(a) ^ bit(b)) == 1, pl[3], pl[2])
    elif kind == 'c1q':
        G, (c, t) = pl[1], g.targets
        i0 = idx[(bit(c) == 1) & (bit(t) == 0)]
        i1 = i0 | (1 << t)
        out[i0] = G[0, 0] * s[i0] + G[0, 1] * s[i1]
        out[i1] = G[1, 0] * s[i0] + G[1, 1] * s[i1]
    elif kind == 'pp2':
        a, b = g.targets
        out = pl[3] * s[idx ^ (1 << a) ^ (1 << b)] * np.where(bit(a) == bit(b), pl[2], 1)
    return out


def apply_observable(s: np.ndarray, terms: Sequence[PauliTerm]) -> np.ndarray:
    idx = np.arange(len(s))
    out = np.zeros_like(s)
    for coeff, paulis in terms:
        xm = zm = ny = 0
        for b, pch in paulis.items():
            xm |= (1 << b) if pch in 'XY' else 0
            zm |= (1 << b) if pch in 'ZY' else 0
            ny += pch == 'Y'
        j = idx ^ xm
        par = np.zeros(len(s), dtype=np.int64)
        m, sh = zm, 0
        while m:
            if m & 1:
                par ^= (j >> sh) & 1
            m >>= 1
            sh += 1
        out += coeff * (1j) ** ny * np.where(par == 1, -1.0, 1.0) * s[j]
    return out


def forward_state(n: int, gates: Sequence[Gate], init=None) -> np.ndarray:
    if init is None:
        s = np.zeros(1 << n, dtype=np.complex128)
        s[0] = 1.0
    else:
        s = np.array(init, dtype=np.complex128)
    for g in gates:
        apply_np(s, g)
    return s


def cost_of(n: int, gates: Sequence[Gate], terms: Sequence[PauliTerm]) -> float:
    s = forward_state(n, gates)
    return float(np.vdot(s, apply_observable(s, terms)).real)


def dense_adjoint(n: int, gates: Sequence[Gate], terms: Sequence[PauliTerm], init=None):
    psi = forward_state(n, gates, init)
    lam = apply_observable(psi, terms)
    cost = float(np.vdot(psi, lam).real)
    grads = []
    for g in reversed(gates):
        if is_parametric(g):
            grads.append(2.0 * np.vdot(lam, apply_generator(psi, g)).imag)
        apply_np(psi, g, inverse=True)
        apply_np(lam, g, inverse=True)
    return cost, np.array(grads[::-1])


def contracted_adjoint(plan: Plan, terms: Sequence[PauliTerm]):
    k = int(plan.widths[0]) if len(plan.widths) else plan.n
    psi = np.zeros(1 << k, dtype=np.complex128)
    psi[0] = 1.0
    peak_work = 0.0
    for g, w in zip(plan.gates, plan.widths):
        w = int(w)
        if w > len(psi).bit_length() - 1:
            psi = np.concatenate([psi, np.zeros((1 << w) - len(psi), dtype=psi.dtype)])
        apply_np(psi, g)
        peak_work += len(psi)
    if len(psi) < (1 << plan.n):
        psi = np.concatenate([psi, np.zeros((1 << plan.n) - len(psi), dtype=psi.dtype)])
    lam = apply_observable(psi, terms)
    cost = float(np.vdot(psi, lam).real)

    grads = np.zeros(plan.n_params)
    psi_leak = lam_dropped = 0.0
    for i in range(len(plan.gates) - 1, -1, -1):
        g, w = plan.gates[i], int(plan.widths[i])
        if (1 << w) < len(psi):
            psi_leak = max(psi_leak, float(np.linalg.norm(psi[1 << w:])))
            lam_dropped = max(lam_dropped, float(np.linalg.norm(lam[1 << w:])))
            psi = psi[:1 << w].copy()
            lam = lam[:1 << w].copy()
        if plan.param_index[i] >= 0:
            grads[plan.param_index[i]] = 2.0 * np.vdot(lam, apply_generator(psi, g)).imag
        apply_np(psi, g, inverse=True)
        apply_np(lam, g, inverse=True)
    return cost, grads, dict(psi_leak=psi_leak, lam_dropped=lam_dropped,
                             work=peak_work / (len(plan.gates) * 2.0 ** plan.n))


def parameter_shift(n: int, gates: Sequence[Gate], terms: Sequence[PauliTerm]) -> np.ndarray:
    gates = list(gates)
    out = []
    for i, g in enumerate(gates):
        if not is_parametric(g):
            continue
        vals = []
        for sgn in (+1, -1):
            gates[i] = Gate(g.name, g.targets, (g.params[0] + sgn * np.pi / 2,) + tuple(g.params[1:]))
            vals.append(cost_of(n, gates, terms))
        gates[i] = g
        out.append(0.5 * (vals[0] - vals[1]))
    return np.array(out)
