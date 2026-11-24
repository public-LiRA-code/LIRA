import argparse
import json
import os
import sys
import time

_prof = None
if os.environ.get('LIRA_MPIPROF'):
    import ctypes
    _prof = ctypes.CDLL(os.environ['LIRA_MPIPROF'], mode=os.RTLD_LAZY | os.RTLD_GLOBAL)

if os.environ.get('LIRA_GPU_BEFORE_MPI') and os.environ.get('SLURM_LOCALID') is not None:
    from cupy.cuda import runtime as _rt
    _rt.setDevice(int(os.environ['SLURM_LOCALID']) % _rt.getDeviceCount())

if _prof:
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
from mpi4py import MPI
if _prof:
    sys.setdlopenflags(_flags)

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lira.lightning_ref import _FIXED, _PAULI, _ROT
from lira.planner import make_plan
from benchmarks.single_gpu import build, observable


def make_qnode(dev, plan, terms):
    n = plan.n
    w = lambda b: n - 1 - int(b)
    init = [g.params[0] for g, p in zip(plan.gates, plan.param_index) if g.name.lower() in _ROT and p >= 0]
    slots = [int(p) for g, p in zip(plan.gates, plan.param_index) if g.name.lower() in _ROT and p >= 0]

    def word(ps):
        if len(ps) == 1:
            return _PAULI[next(iter(ps.values()))](w(next(iter(ps))))
        return qml.prod(*[_PAULI[p](w(b)) for b, p in sorted(ps.items())])
    obs = [word(ps) for _, ps in terms]
    coeffs = [c for c, _ in terms]
    H = obs[0] if (len(obs) == 1 and coeffs[0] == 1.0) else qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev, diff_method='adjoint')
    def circuit(weights):
        k = 0
        for g, pidx in zip(plan.gates, plan.param_index):
            name = g.name.lower()
            wires = [w(t) for t in g.targets]
            if name in _ROT:
                if pidx >= 0:
                    _ROT[name](weights[k], wires=wires)
                    k += 1
                else:
                    _ROT[name](g.params[0] if g.params else 0.0, wires=wires)
            elif name == 'sdg':
                qml.adjoint(qml.S(wires=wires))
            elif name == 'tdg':
                qml.adjoint(qml.T(wires=wires))
            else:
                _FIXED[name](wires=wires)
        return qml.expval(H)

    return circuit, pnp.array(init, requires_grad=True), slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--cases', default='khea_4')
    ap.add_argument('--obs', choices=['tfim', 'parity', 'native', 'zzring', 'meanz'], default='tfim')
    ap.add_argument('--order', choices=['circuit', 'planner'], default='circuit',
                    help="circuit: the circuit as written (own qubit labels and gate order; no technique of lira); "
                         "planner: the relabelling and gate order of the execution plan")
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--dtype', choices=['c64', 'c128'], default='c64', help='c128 for the accuracy check')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-grads', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if _prof:
        names = comm.allgather(MPI.Get_processor_name())
        _prof.mpiprof_set_far((ctypes.c_int * size)(*[int(nm != names[rank]) for nm in names]), size)

    def prof():
        buf = (ctypes.c_double * 16)()
        _prof.mpiprof_get(buf)
        keys = ('far', 'near', 'barrier', 'coll', 'bytes_far', 'bytes_near', 'n_data', 'n_wait', 'n_barrier', 'n_coll', 'n_test', 't_test',
                'swap_far', 'swap_near', 'n_swap_far', 'n_swap_near')
        return comm.gather(dict(zip(keys, list(buf))))
    rows = []
    for case in args.cases.split(','):
        n, gates = build(case, args.n, args.seed)
        plan = (make_plan(n, gates, order=list(range(n)), full_width=True, keep_order=True) if args.order == 'circuit'
                else make_plan(n, gates))
        terms = observable(args.obs, n, plan, case)
        row = dict(case=case, n=n, ranks=size, obs=args.obs, order=args.order, engine='lightning.kokkos', workflow='qnode + qml.jacobian',
                   dtype=args.dtype, match_fix=bool(os.environ.get('LKK_RECEIVE_RANK_FIX')))
        try:
            dev = qml.device('lightning.kokkos', wires=n, c_dtype=np.complex128 if args.dtype == 'c128' else np.complex64,
                             **(dict(mpi=True) if size > 1 else {}))
            circuit, weights, slots = make_qnode(dev, plan, terms)
            runs = []
            for _ in range(args.repeat):
                comm.Barrier()
                if _prof:
                    _prof.mpiprof_reset()
                t0 = time.perf_counter()
                jac = qml.jacobian(circuit)(weights)
                comm.Barrier()
                runs.append(time.perf_counter() - t0)
                if _prof:
                    pr = prof()
                    if rank == 0 and runs[-1] == min(runs):
                        row['mpiprof_total'] = pr
            total = min(runs)
            row['runs'] = runs
            grads = np.zeros(plan.n_params)
            grads[slots] = np.asarray(jac).reshape(-1)
            row.update(total=total)
            if rank == 0:
                print(f"[lightning.kokkos, {size} ranks] {case} n={n}: forward + adjoint {total:8.3f}s", flush=True)
                if args.save_grads:
                    os.makedirs(os.path.dirname(args.save_grads), exist_ok=True)
                    np.save(f"{args.save_grads}_{case.replace('/', '-')}_n{n}.npy", np.concatenate([[np.nan], grads]))
        except Exception as e:
            row['error'] = str(e)[-200:]
            if rank == 0:
                print(f"[lightning.kokkos, {size} ranks] {case} n={n}: FAILED {row['error']}", flush=True)
        rows.append(row)
        if rank == 0 and args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
