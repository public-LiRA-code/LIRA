import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lira import circuits as C
from lira.engine import LiraEngine
from lira.planner import make_plan
from lira.reference import parameter_shift
from test_lira import cases, hamiltonian, relabel


def main():
    rng = np.random.default_rng(0)
    ok, worst = True, 0.0
    print(f"{'circuit':<20}{'n':>3}{'G':>6} | {'grad vs ps':>11}{'contracted':>11}{'lambda vs full':>15}")
    for name, n, gates in cases() + [('SEL L=4', *C.sel(7, 4)), ('SEL L=2', *C.sel(9, 2))]:
        terms = hamiltonian(n, rng)
        ps = parameter_shift(n, gates, terms)
        plan = make_plan(n, gates)
        full = make_plan(n, gates, order=plan.order, full_width=True)
        t = relabel(terms, plan)
        eng = LiraEngine(n, dtype=np.complex128)
        c_c, g_c, _ = eng.run(plan, t, fuse=False, window=0, fuse_obs=False)
        c_v, g_v, _ = eng.run(plan, t, fuse=False, window=0, fuse_obs=False, lam_full=True)
        lam_v = eng._lam.get()
        eng.run(full, t, fuse=False, window=0, fuse_obs=False)
        lam_f = eng._lam.get()
        e_ps = max(float(np.abs(g_v - ps).max()), abs(c_v - c_c))
        e_c = float(np.abs(g_v - g_c).max())
        e_l = float(np.abs(lam_v - lam_f).max())
        bad = max(e_ps, e_c, e_l) > 1e-9
        ok &= not bad
        worst = max(worst, e_ps, e_c, e_l)
        print(f"{name:<20}{n:>3}{len(gates):>6} | {e_ps:>11.1e}{e_c:>11.1e}{e_l:>15.1e}" + ('   <-- FAIL' if bad else ''))
    print(f"\nworst difference {worst:.1e}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
