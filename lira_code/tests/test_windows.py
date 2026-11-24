import os
import sys

import cupy as cp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lira import circuits as C
from lira import windows as W
from lira.engine import LiraEngine
from lira.fusion import summary
from lira.planner import make_plan
from lira.reference import parameter_shift
from lira.gates import Gate

from lira import paths
SETTINGS = [3, 4, (3, 5, False), (2, 2, True), (3, 3, True), (5, 6, True)]


def random_circuit(n, count, seed):
    r = np.random.default_rng(seed)
    g = []
    for _ in range(count):
        x = r.random()
        if x < 0.45:
            g.append(Gate(str(r.choice(['rx', 'ry', 'rz', 'p'])), (int(r.integers(n)),), (float(r.uniform(0, 6)),)))
        elif x < 0.55:
            g.append(Gate(str(r.choice(['h', 's', 't', 'x', 'sx', 'sdg'])), (int(r.integers(n)),), ()))
        elif x < 0.9:
            a, b = r.choice(n, 2, replace=False)
            g.append(Gate(str(r.choice(['cx', 'cx', 'cx', 'swap'])), (int(a), int(b)), ()))
        else:
            a, b = r.choice(n, 2, replace=False)
            nm = str(r.choice(['cz', 'rzz', 'cp']))
            g.append(Gate(nm, (int(a), int(b)), () if nm == 'cz' else (float(r.uniform(0, 6)),)))
    return n, g


def cases():
    q = lambda fam, n: C.from_qasm(paths.circuit(fam, 28), n)
    return [('HEA CX line', *C.hea(10, 3, 'linear')), ('HEA CX ring', *C.hea(8, 3, 'circular')), ('HEA CX all pairs', *C.hea(7, 2, 'full')),
            ('HEA CZ ring', *C.kawase_hea(9, 3)), ('QAOA 3-regular', *C.qaoa(10, 2, '3reg')), ('TFIM 1D', *C.qaoa(9, 3, 'line')),
            ('UCCSD 6q2e', *C.uccsd(6, 2)), ('MPS staircase', *C.mps_staircase(9, 2)),
            ('MQT vqc (9q)', *q('vqc', 9)), ('MQT realamprandom (8q)', *q('realamprandom', 8)), ('MQT su2random (7q)', *q('su2random', 7)),
            ('random 8q', *random_circuit(8, 150, 1)), ('random 9q', *random_circuit(9, 220, 2)), ('random 6q', *random_circuit(6, 300, 3))]


def main():
    rng = np.random.default_rng(0)
    worst = worst32 = restore = 0.0
    print(f"{'circuit':<24}{'n':>3}{'G':>5}{'P':>5} |" + ''.join(f"{str(s).replace(' ', ''):>14}" for s in SETTINGS)
          + f" |{'obs':>9}{'late':>9}{'c64':>9}{'restore':>9} | stage 1 -> window=4: ops (windows), sweeps")
    for name, n, gates in cases():
        terms = [(float(rng.normal()), {q: 'Z', (q + 1) % n: 'Z'}) for q in range(n)] + [(float(rng.normal()), {q: 'X'}) for q in range(n)]
        terms += [(float(rng.normal()), {0: 'Y', n - 1: 'Y', n // 2: 'X'}), (float(rng.normal()), {1: 'Y'}), (float(rng.normal()), {2: 'X', 3: 'Z'})]
        plan = make_plan(n, gates)
        full = make_plan(n, gates, order=plan.order, full_width=True)
        ps = parameter_shift(n, plan.gates, [(c, {plan.order.index(q): p for q, p in t.items()}) for c, t in terms]) \
            if len(plan.gates) != len(gates) else None
        bit = {q: j for j, q in enumerate(plan.order)}
        t = [(c, {bit[q]: p for q, p in ps_.items()}) for c, ps_ in terms]
        eng = LiraEngine(n, np.complex128)
        c0, g0, _ = eng.run(plan, t, fuse=False, window=0, fuse_obs=False)
        if ps is None:
            ps = parameter_shift(n, gates, terms)
        else:
            ps = g0
        errs = []
        for s in SETTINGS:
            e = 0.0
            for pl in (full, plan):
                for mg in (0.0, 1.0):
                    c, g, _ = eng.run(pl, t, fuse=True, margin=mg, window=s, fuse_obs=False)
                    e = max(e, abs(c - c0), float(np.abs(g - ps).max()))
                    restore = max(restore, float(cp.abs(eng._psi[1:]).max()), abs(abs(complex(eng._psi[0])) - 1.0))
            errs.append(e)
        eo = 0.0
        for pl in (full, plan):
            c, g, _ = eng.run(pl, t, fuse=True, window=4, fuse_obs=True)
            eo = max(eo, abs(c - c0), float(np.abs(g - ps).max()))
        errs.append(eo)
        el = 0.0
        for pl, kw in ((full, dict(fuse=True, window=4)), (plan, dict(fuse=True, window=4)), (plan, dict(fuse=True, window=0)), (plan, dict(fuse=False))):
            c, g, _ = eng.run(pl, t, late_obs=True, **kw)
            el = max(el, abs(c - c0), float(np.abs(g - ps).max()))
        errs.append(el)
        e32 = LiraEngine(n, np.complex64)
        c1, g1, _ = e32.run(plan, t, fuse=True, window=0, fuse_obs=False)
        d32 = 0.0
        for s in (3, 4):
            c, g, _ = e32.run(plan, t, fuse=True, window=s, fuse_obs=False)
            d32 = max(d32, abs(c - c1), float(np.abs(g - g1).max()))
        worst, worst32 = max(worst, *errs), max(worst32, d32)
        s1, s4 = summary(eng.fused_ops(full, 1.0), n), summary(eng.fused_ops(full, 1.0, 4), n)
        print(f"{name:<24}{n:>3}{len(gates):>5}{plan.n_params:>5} |" + ''.join(f"{e:>14.1e}" for e in errs[:-1]) + f" |{eo:>9.1e}{d32:>9.1e}{restore:>9.1e} | "
              f"{s1['ops']:>4} -> {s4['ops']:>3} ({s4['windows']:>3}), {s1['sweeps']:>6.1f} -> {s4['sweeps']:>5.1f}")
    ok = worst < 1e-9 and worst32 < 2e-5 and restore < 1e-9
    print(f"\nmax |grad - parameter shift| with windows = {worst:.1e};  single precision vs stage 1 = {worst32:.1e};  psi restored to {restore:.1e}"
          f"   ->  {'PASS' if ok else 'FAIL'}   ({W.STATS['compiled']} window shapes compiled in {W.STATS['seconds']:.1f} s)")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
