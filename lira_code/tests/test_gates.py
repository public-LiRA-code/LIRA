import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lira.decompose import expand
from lira.engine import LiraEngine
from lira.exchanges import MultiGPUEngine
from lira.planner import make_plan
from lira.reference import apply_np
from lira.gates import Gate

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def rot(P, th):
    return np.cos(th / 2) * np.eye(len(P)) - 1j * np.sin(th / 2) * P


def ctrl(U):
    k = len(U)
    out = np.eye(2 * k, dtype=complex)
    out[k:, k:] = U
    return out


def matrix(name, p):
    if name == 'u3':
        th, ph, lam = p
        return np.array([[np.cos(th / 2), -np.exp(1j * lam) * np.sin(th / 2)],
                         [np.exp(1j * ph) * np.sin(th / 2), np.exp(1j * (ph + lam)) * np.cos(th / 2)]])
    if name == 'u2':
        return matrix('u3', (np.pi / 2, p[0], p[1]))
    one = {'h': H, 'x': X, 'y': Y, 'z': Z, 's': np.diag([1, 1j]), 't': np.diag([1, np.exp(1j * np.pi / 4)]),
           'sx': 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])}
    if name in one:
        return one[name].astype(complex)
    if name in ('rx', 'ry', 'rz'):
        return rot({'rx': X, 'ry': Y, 'rz': Z}[name], p[0])
    if name == 'p':
        return np.diag([1, np.exp(1j * p[0])])
    if name in ('crx', 'cry', 'crz'):
        return ctrl(rot({'crx': X, 'cry': Y, 'crz': Z}[name], p[0]))
    if name in ('cx', 'cy', 'cz', 'ch'):
        return ctrl({'cx': X, 'cy': Y, 'cz': Z, 'ch': H}[name])
    if name == 'cp':
        return np.diag([1, 1, 1, np.exp(1j * p[0])])
    if name in ('rxx', 'ryy', 'rzz'):
        P = {'rxx': X, 'ryy': Y, 'rzz': Z}[name]
        return rot(np.kron(P, P), p[0])
    if name == 'swap':
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    if name == 'ccx':
        return ctrl(ctrl(X))
    raise ValueError(name)


N_ANGLES = {'u3': 3, 'u2': 2, 'rx': 1, 'ry': 1, 'rz': 1, 'p': 1, 'crx': 1, 'cry': 1, 'crz': 1, 'cp': 1, 'rxx': 1, 'ryy': 1, 'rzz': 1}


def apply_dense(s, M, targets):
    n = int(np.log2(len(s)))
    k = len(targets)
    idx = np.arange(1 << n)
    base = idx[np.all([((idx >> q) & 1) == 0 for q in targets], axis=0)]
    offs = [sum(((b >> (k - 1 - j)) & 1) << targets[j] for j in range(k)) for b in range(1 << k)]
    new = M @ np.stack([s[base | o] for o in offs])
    for b, o in enumerate(offs):
        s[base | o] = new[b]


def pauli_apply(s, terms):
    n = int(np.log2(len(s)))
    out = np.zeros_like(s)
    for c, ps in terms:
        v = s.copy()
        for q, ch in ps.items():
            apply_dense(v, {'X': X, 'Y': Y, 'Z': Z}[ch], (q,))
        out += c * v
    return out


def cost(n, circ, theta, terms):
    s = np.zeros(1 << n, dtype=complex)
    s[0] = 1.0
    k = 0
    for name, t in circ:
        m = N_ANGLES.get(name, 0)
        apply_dense(s, matrix(name, theta[k:k + m]), t)
        k += m
    return float(np.vdot(s, pauli_apply(s, terms)).real)


def to_gates(circ, theta):
    out, k = [], 0
    for name, t in circ:
        m = N_ANGLES.get(name, 0)
        out.append(Gate(name, tuple(t), tuple(float(x) for x in theta[k:k + m])))
        k += m
    return out


def random_circuit(n, depth, names, rng):
    circ = []
    for _ in range(depth):
        name = names[rng.integers(len(names))]
        k = {'ccx': 3}.get(name, 2 if (name[0] == 'c' or name in ('rxx', 'ryy', 'rzz', 'swap')) else 1)
        circ.append((name, tuple(int(q) for q in rng.choice(n, size=k, replace=False))))
    return circ


def check_rewrites(rng):
    worst = 0.0
    for name, t in (('u3', (1,)), ('u2', (0,)), ('crz', (2, 0)), ('crz', (0, 1)), ('rxx', (0, 2)), ('ryy', (1, 0)), ('ryy', (0, 2)),
                    ('crx', (2, 0)), ('cry', (0, 1)), ('cy', (1, 2)), ('ch', (2, 1)), ('ccx', (2, 0, 1))):
        th = rng.uniform(-3, 3, N_ANGLES.get(name, 0))
        n = 3
        A = np.zeros((1 << n, 1 << n), dtype=complex)
        B = np.zeros_like(A)
        for col in range(1 << n):
            a = np.zeros(1 << n, dtype=complex)
            a[col] = 1.0
            b = a.copy()
            apply_dense(a, matrix(name, th), t)
            for g in expand([Gate(name, t, tuple(th))]):
                apply_np(b, g)
            A[:, col], B[:, col] = a, b
        ph = np.vdot(A.ravel(), B.ravel())
        worst = max(worst, float(np.abs(B - A * ph / abs(ph)).max()))
    return worst


def lightning_grad(n, circ, theta, terms):
    import pennylane as qml
    from pennylane import numpy as pnp
    ops = {'u3': qml.U3, 'u2': qml.U2, 'rx': qml.RX, 'ry': qml.RY, 'rz': qml.RZ, 'p': qml.PhaseShift, 'crx': qml.CRX,
           'cry': qml.CRY, 'crz': qml.CRZ, 'cp': qml.ControlledPhaseShift, 'rxx': qml.IsingXX, 'ryy': qml.IsingYY,
           'rzz': qml.IsingZZ, 'h': qml.Hadamard, 'x': qml.PauliX, 'y': qml.PauliY, 'z': qml.PauliZ, 's': qml.S, 't': qml.T,
           'sx': qml.SX, 'cx': qml.CNOT, 'cy': qml.CY, 'cz': qml.CZ, 'ch': qml.CH, 'swap': qml.SWAP, 'ccx': qml.Toffoli}
    P = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}
    Hm = qml.Hamiltonian([c for c, _ in terms],
                         [qml.prod(*[P[ch](q) for q, ch in sorted(ps.items())]) if len(ps) > 1 else P[next(iter(ps.values()))](next(iter(ps)))
                          for _, ps in terms])
    dev = qml.device('lightning.gpu', wires=n, c_dtype=np.complex128)

    @qml.qnode(dev, diff_method='adjoint')
    def f(w):
        k = 0
        for name, t in circ:
            m = N_ANGLES.get(name, 0)
            ops[name](*[w[k + j] for j in range(m)], wires=list(t))
            k += m
        return qml.expval(Hm)
    return np.asarray(qml.grad(f)(pnp.array(theta, requires_grad=True)), dtype=float)


def main():
    import cupy as cp
    ngpu = cp.cuda.runtime.getDeviceCount()
    rng = np.random.default_rng(11)
    w = check_rewrites(rng)
    print(f'added gates against the dense matrices of their definitions: max difference {w:.1e}')
    ok = w < 1e-12

    new = ['u3', 'u2', 'crx', 'cry', 'crz', 'rxx', 'ryy', 'cy', 'ch', 'ccx']
    old = ['h', 'rx', 'ry', 'rz', 'p', 'cx', 'cz', 'cp', 'rzz', 'swap', 'sx', 's', 't']
    cases = [('all new gates', 7, random_circuit(7, 70, new, rng)),
             ('new + old gates', 8, random_circuit(8, 110, new + old, rng)),
             ('kernels only', 7, random_circuit(7, 60, ['crx', 'cry', 'cy', 'ch', 'ccx', 'rxx', 'ryy', 'ry', 'h'], rng)),
             ('rewrites only', 7, random_circuit(7, 60, ['u3', 'u2', 'crz', 'cx'], rng)),
             ('Heisenberg layers', 8, [(g, (q, q + 1)) for _ in range(2) for q in list(range(0, 7, 2)) + list(range(1, 7, 2)) for g in ('rxx', 'ryy', 'rzz')]),
             ('u3 + cx layers', 8, [(('u3', (q,)) if i % 2 == 0 else ('cx', (q, (q + 1) % 8))) for _ in range(3) for i in range(2) for q in range(8)])]
    cols = ('gpu-full', 'gpu-lira', 'fuse-full', 'fuse-lira', 'win-full', 'win-lira', '2r lira', '4r lira', '4r full', '4r lira w4', '8r nofuse', 'lgpu')
    print(f"\n{'circuit':<18}{'n':>3}{'G':>5}{'P':>5}{'native':>8} |" + ''.join(f'{c:>11}' for c in cols) + ' | max |grad|')
    worst = 0.0
    for name, n, circ in cases:
        P = sum(N_ANGLES.get(nm, 0) for nm, _ in circ)
        theta = rng.uniform(-3, 3, P)
        terms = [(float(rng.normal()), {q: 'Z', (q + 1) % n: 'Z'}) for q in range(n)] + [(float(rng.normal()), {q: 'X'}) for q in range(n)]
        terms += [(float(rng.normal()), {0: 'Y', n - 1: 'Y', n // 2: 'X'})]
        c0 = cost(n, circ, theta, terms)
        h = 1e-5
        fd = np.array([(cost(n, circ, theta + h * e, terms) - cost(n, circ, theta - h * e, terms)) / (2 * h) for e in np.eye(P)])
        gates = to_gates(circ, theta)
        plan = make_plan(n, gates)
        full = make_plan(n, gates, order=plan.order, full_width=True, keep_order=True)
        assert plan.n_params == P
        bit = {q: j for j, q in enumerate(plan.order)}
        t = [(c, {bit[q]: p for q, p in ps.items()}) for c, ps in terms]
        errs = []
        eng = LiraEngine(n, np.complex128)
        for pl, fz, win in ((full, False, 0), (plan, False, 0), (full, True, 0), (plan, True, 0), (full, True, 4), (plan, True, 4)):
            c, g, _ = eng.run(pl, t, fuse=fz, margin=0.0, window=win, fuse_obs=False)
            errs.append(max(np.abs(g - fd).max(), abs(c - c0)))
        eng.release()
        for ranks, pl, fz, win in ((2, plan, True, 0), (4, plan, True, 0), (4, full, True, 0), (4, plan, True, 4), (8, plan, False, 0)):
            mg = MultiGPUEngine(n, [r % ngpu for r in range(ranks)], dtype=np.complex128, chunk_bits=2)
            c, g, _, _ = mg.run(pl, t, fuse=fz, margin=0.0, window=win)
            mg.release()
            errs.append(max(np.abs(g - fd).max(), abs(c - c0)))
        try:
            errs.append(float(np.abs(lightning_grad(n, circ, theta, terms) - fd).max()))
        except Exception as e:
            errs.append(float('nan'))
            print(f'  [lightning.gpu not run: {str(e)[:120]}]')
        good = [e for e in errs if e == e]
        bad = max(good) > 2e-7
        ok &= not bad
        worst = max(worst, *good)
        print(f'{name:<18}{n:>3}{len(circ):>5}{P:>5}{len(plan.gates):>8} |' + ''.join(f'{e:>11.1e}' for e in errs)
              + f' | {np.abs(fd).max():.2e}' + ('   <-- FAIL' if bad else ''))
    print(f"\nmax |grad - central difference of the dense reference| = {worst:.2e}   ->  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
