import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lira.distributed import MPI, DistributedEngine
from lira import circuits as C
from lira.decompose import is_tagged
from lira.planner import Plan, make_plan
from lira.reference import contracted_adjoint
from benchmarks.single_gpu import build, observable
from lira.gates import Gate

from lira import paths


def with_angles(plan, theta):
    gates = []
    for g, p in zip(plan.gates, plan.param_index):
        if p >= 0:
            g = Gate(g.name, g.targets, (float(g.params[1]) * theta[p], g.params[1], g.params[2]) if is_tagged(g)
                     else (float(theta[p]),) + tuple(g.params[1:]))
        gates.append(g)
    return Plan(n=plan.n, order=plan.order, gates=gates, widths=plan.widths, orig_index=plan.orig_index,
                param_index=plan.param_index, r_work=plan.r_work, n_params=plan.n_params)


def main():
    comm = MPI.COMM_WORLD
    n = int(os.environ.get('N', 14))
    cases = [(c, *build(c, n, 0)) for c in ('khea_4', 'hea_circular_4', 'hea_full_4', 'qaoa_3reg_4', 'tfim1d_4')]
    nn, g = C.from_qasm(paths.circuit('su2random', 28), n_keep=n)
    cases.append(('su2random (u3, first %d qubits)' % n, nn, g))
    worst, rng = 0.0, np.random.default_rng(1)
    for name, nn, gates in cases:
        if name.startswith('qaoa') and nn % 2:
            continue
        plan = make_plan(nn, gates)
        terms = observable('tfim', nn, plan, name)
        for w, fo, late in ((0, False, False), ((4, 6, 1), True, False), ((4, 6, 1), False, True)):
            eng = DistributedEngine(nn, comm, np.complex128, early=6 if comm.Get_size() > 1 else 0, stream=2 if comm.Get_size() > 1 else 0)
            eng.compile(plan, window=w)
            for _ in range(2):
                theta = comm.bcast(rng.uniform(0, 2 * np.pi, plan.n_params))
                eng.rebind(plan, theta, window=w)
                c, gr, _, _ = eng.run(plan, terms, window=w, fuse_obs=fo, late_obs=late)
                c0, g0, _ = contracted_adjoint(with_angles(plan, theta), terms)
                err = max(abs(c - c0), float(np.abs(gr - g0).max()))
                worst = max(worst, err)
                if comm.Get_rank() == 0 and err > 1e-9:
                    print(f'  {name} window={w} fuse_obs={fo} late={late}: error {err:.2e}', flush=True)
            eng.release()
        if comm.Get_rank() == 0:
            print(f'{name:34s} n={nn} params={plan.n_params}: done', flush=True)
    if comm.Get_rank() == 0:
        print(f"rebind: {'PASS' if worst < 1e-9 else 'FAIL'} (largest difference {worst:.2e}, {comm.Get_size()} ranks)", flush=True)


if __name__ == '__main__':
    main()
