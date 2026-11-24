import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lira.exchanges as MG
from lira.distributed import MPI, DistributedEngine
from lira.planner import make_plan
from lira.reference import dense_adjoint, parameter_shift
from test_lira import cases, hamiltonian, relabel


def main():
    comm = MPI.COMM_WORLD
    rank, G = comm.Get_rank(), comm.Get_size()
    g = G.bit_length() - 1
    rng = np.random.default_rng(0)
    worst, ok = 0.0, True
    MG.STREAM_MIN, MG.STREAM_MIN_OPS = 1, 0
    if rank == 0:
        print(f'{G} ranks')
        print(f"{'circuit':<20}{'n':>3}{'G':>6}{'P':>5} |" + ''.join(f'{c:>12}' for c in ('lira', 'full', 'lira nofuse', 'lira w4', 'full w3', 'lira w4 obs', 'lira early', 'lira late', 'full late w4', 'full s1', 'lira s2 w4', 'lira all', 'full all w4', 'lira mix', 'all mix h2', 'lira mix2', 'all mix2 h2'))
              + ' | exchanges (two-way, one-way) lira / full', flush=True)
    for name, n, gates in cases():
        job = None
        if rank == 0:
            terms = hamiltonian(n, rng)
            plan = make_plan(n, gates)
            job = (plan, make_plan(n, gates, order=plan.order, full_width=True, keep_order=True), relabel(terms, plan),
                   parameter_shift(n, gates, terms), dense_adjoint(n, gates, terms)[0])
        plan, full, t, ps, cost0 = comm.bcast(job)
        if n - g < 3:
            continue
        errs, info = [], []
        for pl, fz, win, fo, early, late, stream, *extra in (
                (plan, True, 0, False, 0, False, 0), (full, True, 0, False, 0, False, 0), (plan, False, 0, False, 0, False, 0),
                (plan, True, 4, False, 0, False, 0), (full, True, 3, False, 0, False, 0), (plan, True, 4, True, 0, False, 0),
                (plan, True, 4, True, 3, False, 0), (plan, True, 0, False, 0, True, 0), (full, True, 4, False, 0, True, 0),
                (full, True, 0, False, 0, True, 1), (plan, True, 4, False, 0, True, 2), (plan, True, 4, False, 3, True, 1),
                (full, True, 4, False, 3, True, 2), (plan, True, 4, True, 0, False, 0, 0, 1), (full, True, 4, False, 3, True, 2, 2, 1),
                (plan, True, 4, True, 0, False, 0, 0, 2), (full, True, 4, False, 3, True, 2, 2, 2)):
            intra, mix = (extra + [0, False])[:2] if extra else (0, False)
            eng = DistributedEngine(n, comm, dtype=np.complex128, chunk_bits=2, early=early, stream=stream, intra=intra, mix=mix)
            eng._force_early = bool(early)
            c, gr, _, st = eng.run(pl, t, fuse=fz, margin=0.0, window=win, fuse_obs=fo, late_obs=late)
            eng.release()
            errs.append(max(np.abs(gr - ps).max(), abs(c - cost0)))
            info.append(st)
        bad = max(errs) > 1e-9
        ok &= not bad
        worst = max(worst, *errs)
        if rank == 0:
            print(f"{name:<20}{n:>3}{len(gates):>6}{plan.n_params:>5} |" + ''.join(f'{e:>12.1e}' for e in errs)
                  + f" | {info[0]['exchanges']}, {info[0]['oneway']} / {info[1]['exchanges']}, {info[1]['oneway']}"
                  + ('   <-- FAIL' if bad else ''), flush=True)
    if rank == 0:
        print(f"\n{G} ranks: max |grad - parameter_shift| = {worst:.2e}   ->  {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
