import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lira import circuits as C
from lira.engine import LiraEngine
from lira.planner import make_plan
from lira.reference import contracted_adjoint, dense_adjoint, parameter_shift

try:
    from lira.lightning_ref import LightningBaseline
except Exception as e:
    LightningBaseline = None
    print(f'[warn] lightning.gpu unavailable: {e!r}')

from lira import paths


def hamiltonian(n, rng):
    terms = [(float(rng.normal()), {q: 'Z', (q + 1) % n: 'Z'}) for q in range(n)]
    terms += [(float(rng.normal()), {q: 'X'}) for q in range(n)]
    terms += [(float(rng.normal()), {0: 'Y', n - 1: 'Y', n // 2: 'X'})]
    return terms


def relabel(terms, plan):
    bit = {q: j for j, q in enumerate(plan.order)}
    return [(c, {bit[q]: p for q, p in ps.items()}) for c, ps in terms]


def cases():
    out = [('HEA linear', *C.hea(10, 3, 'linear')),
           ('HEA brick cz', *C.hea(9, 4, 'brick', gate2='cz')),
           ('HEA circular', *C.hea(8, 2, 'circular')),
           ('HEA full', *C.hea(7, 2, 'full', rot=('rx', 'rz'))),
           ('QAOA 3-reg', *C.qaoa(10, 2, '3reg')),
           ('QAOA ring', *C.qaoa(9, 3, 'ring')),
           ('MPS staircase', *C.mps_staircase(10, 2)),
           ('UCCSD 6q2e', *C.uccsd(6, 2)),
           ('SEL L=4', *C.sel(7, 4)),
           ('TFIM HVA ring', *C.qaoa(8, 4, 'ring'))]
    q = paths.circuit('vqc', 28)
    if os.path.exists(q):
        out.append(('MQT vqc (first 9q)', *C.from_qasm(q, n_keep=9)))
    from lira.gates import Gate
    r = np.random.default_rng(7)
    g = [Gate('h', (3,), ()), Gate('ry', (3,), (0.3,)), Gate('cx', (3, 1), ()), Gate('p', (1,), (0.7,)),
         Gate('s', (1,), ()), Gate('rzz', (1, 5), (1.1,)), Gate('rx', (5,), (0.4,)), Gate('cp', (5, 0), (0.9,)),
         Gate('t', (0,), ()), Gate('swap', (0, 4), ()), Gate('cz', (4, 2), ()), Gate('sdg', (2,), ()),
         Gate('frx', (2,), (1.2,)), Gate('y', (4,), ()), Gate('x', (0,), ()), Gate('z', (5,), ()),
         Gate('tdg', (3,), ()), Gate('rz', (2,), (0.5,)), Gate('cx', (2, 3), ()), Gate('ry', (1,), (2.2,)),
         Gate('u1', (4,), (0.33,)), Gate('rzz', (3, 0), (0.6,)), Gate('rx', (4,), (1.9,))]
    g += [Gate(str(r.choice(['rx', 'ry', 'rz'])), (int(q),), (float(r.uniform(0, 6)),)) for q in r.integers(0, 6, 12)]
    out.append(('all gate kinds', 6, g))
    return out


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    ok = True
    print(f"{'circuit':<20}{'n':>3}{'G':>6}{'P':>5}{'R_grad':>8} |{'dense':>9}{'literal':>9}{'gpu-full':>9}"
          f"{'gpu-lira':>9}{'fuse-full':>10}{'fuse-lira':>9}{'window':>9}{'init':>9}{'lgpu':>9} | {'psi leak':>9}{'lam drop':>9}{'ops':>6}")
    for name, n, gates in cases():
        terms = hamiltonian(n, rng)
        ps = parameter_shift(n, gates, terms)
        cost0, dense = dense_adjoint(n, gates, terms)

        plan = make_plan(n, gates)
        full = make_plan(n, gates, order=plan.order, full_width=True)
        t = relabel(terms, plan)

        cost_l, lit, info = contracted_adjoint(plan, t)
        errs = {'dense': np.abs(dense - ps).max(), 'literal': np.abs(lit - ps).max()}
        eng = LiraEngine(n, dtype=np.complex128)
        for key, pl, fz in (('gpu-full', full, False), ('gpu-lira', plan, False),
                            ('fuse-full', full, True), ('fuse-lira', plan, True)):
            c, gr, _ = eng.run(pl, t, fuse=fz, margin=0.0, window=0, fuse_obs=False)
            errs[key] = max(np.abs(gr - ps).max(), abs(c - cost0))
        errs['window'] = 0.0
        for pl in (full, plan):
            c, gr, _ = eng.run(pl, t, fuse=True, margin=0.0, window=4, fuse_obs=False)
            errs['window'] = max(errs['window'], np.abs(gr - ps).max(), abs(c - cost0))
        init = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        init /= np.linalg.norm(init)
        c_i, g_i = dense_adjoint(n, full.gates, t, init)
        ref_i = np.zeros(plan.n_params)
        ref_i[full.param_index[full.param_index >= 0]] = g_i
        errs['init'] = 0.0
        import cupy as cp
        for fz in (False, True):
            c, gr, _ = eng.run(full, t, fuse=fz, margin=0.0, init=cp.asarray(init), window=0, fuse_obs=False)
            errs['init'] = max(errs['init'], np.abs(gr - ref_i).max(), abs(c - c_i))
        if LightningBaseline is not None:
            c, gr, _ = LightningBaseline(n, dtype=np.complex128).run(plan, t)
            errs['lgpu'] = max(np.abs(gr - ps).max(), abs(c - cost0))
        else:
            errs['lgpu'] = float('nan')

        bad = max(v for v in errs.values() if v == v) > 1e-9 or abs(cost_l - cost0) > 1e-9 \
            or info['psi_leak'] > 1e-9
        ok &= not bad
        worst = max(worst, *[v for v in errs.values() if v == v])
        print(f"{name:<20}{n:>3}{len(gates):>6}{plan.n_params:>5}{plan.r_work['grad']:>8.3f} |"
              + ''.join(f"{errs[k]:>{10 if k == 'fuse-full' else 9}.1e}" for k in ('dense', 'literal', 'gpu-full', 'gpu-lira', 'fuse-full', 'fuse-lira', 'window', 'init', 'lgpu'))
              + f" | {info['psi_leak']:>9.1e}{info['lam_dropped']:>9.2f}{len(eng.fused_ops(plan, 0.0)):>6}" + ('   <-- FAIL' if bad else ''))
    print(f"\nmax |grad - parameter_shift| over all engines = {worst:.2e}   ->  {'PASS' if ok else 'FAIL'}")
    print("psi leak = norm of psi outside the prefix at a boundary (must be ~0: psi restore needs no checkpoint)")
    print("lam drop = norm of lambda outside the prefix (non-zero: the contraction really discards amplitude)")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
