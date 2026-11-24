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

_flags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
import pennylane_lightning.lightning_gpu_ops
sys.setdlopenflags(_flags)
from mpi4py import MPI

import numpy as np
import pennylane as qml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lira.lightning_ref import build_tape
from lira.planner import make_plan
from benchmarks.single_gpu import build, observable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--cases', default='khea_4')
    ap.add_argument('--obs', choices=['tfim', 'z0', 'ixyz', 'parity', 'native', 'meanz'], default='tfim')
    ap.add_argument('--order', choices=['planner', 'circuit'], default='circuit',
                    help="circuit: the circuit as written, own qubit labels and gate order (no technique "
                         "of lira); planner: the relabelling and gate order of the execution plan")
    ap.add_argument('--batch-obs', action='store_true')
    ap.add_argument('--mpi-buf-size', type=int, default=0,
                    help='MiB of GPU memory for MPI transfers (power of two; 0 = the default: local state size, at most 64 MiB)')
    ap.add_argument('--no-mpi', action='store_true', help='one process, the device without mpi=True (the single-GPU point)')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-grads', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    names = comm.allgather(MPI.Get_processor_name())
    if _prof:
        _prof.mpiprof_set_far((ctypes.c_int * size)(*[int(nm != names[rank]) for nm in names]), size)

    def prof():
        if not _prof:
            return None
        buf = (ctypes.c_double * 16)()
        _prof.mpiprof_get(buf)
        keys = ('far', 'near', 'barrier', 'coll', 'bytes_far', 'bytes_near', 'n_data', 'n_wait', 'n_barrier', 'n_coll', 'n_test', 't_test',
                'swap_far', 'swap_near', 'n_swap_far', 'n_swap_near')
        mine = dict(zip(keys, list(buf)))
        every = comm.gather(mine)
        return every
    rows = []
    for case in args.cases.split(','):
        n, gates = build(case, args.n, args.seed)
        plan = (make_plan(n, gates, order=list(range(n)), full_width=True, keep_order=True) if args.order == 'circuit'
                else make_plan(n, gates))
        terms = observable(args.obs, n, plan, case)
        tape, slots = build_tape(plan, terms)
        row = dict(case=case, n=n, ranks=size, order=args.order, obs=args.obs, batch_obs=args.batch_obs, mpi_buf_size=args.mpi_buf_size)
        try:
            if args.no_mpi:
                dev = qml.device('lightning.gpu', wires=n, c_dtype=np.complex64, batch_obs=args.batch_obs)
                if dev._statevector is None:
                    dev._statevector = dev.LightningStateVector(num_wires=n, dtype=dev.c_dtype)
            else:
                dev = qml.device('lightning.gpu', wires=n, mpi=True, c_dtype=np.complex64, batch_obs=args.batch_obs,
                                 mpi_buf_size=args.mpi_buf_size)
            state = dev._statevector
            best = None
            for _ in range(args.repeat):
                comm.Barrier()
                if _prof:
                    _prof.mpiprof_reset()
                t0 = time.perf_counter()
                state.reset_state()
                final = state.get_final_state(tape)
                cost = float(np.squeeze(dev.LightningMeasurements(final).measure_final_state(tape)))
                comm.Barrier()
                t1 = time.perf_counter()
                p1 = prof()
                jac = np.asarray(dev.LightningAdjointJacobian(final, batch_obs=args.batch_obs).calculate_jacobian(tape))
                comm.Barrier()
                t2 = time.perf_counter()
                p2 = prof()
                if best is None or t2 - t0 < best[0]:
                    best = (t2 - t0, t1 - t0, t2 - t1, p1, p2)
            grads = np.zeros(plan.n_params)
            grads[slots] = jac.reshape(-1)
            row.update(total=best[0], forward=best[1], backward=best[2], cost=cost, mpi=not args.no_mpi,
                       nodes=len(set(names)))
            if rank == 0 and best[3]:
                row.update(mpiprof_forward=best[3], mpiprof_total=best[4])
                mx = lambda k: max(r[k] for r in best[4])
                print(f"[lightning.gpu MPI, {size} ranks]    index-bit swaps (slowest rank): with another node {mx('swap_far'):.3f}s ({best[4][0]['n_swap_far']:.0f}), "
                      f"inside the node {mx('swap_near'):.3f}s ({best[4][0]['n_swap_near']:.0f})", flush=True)
                print(f"[lightning.gpu MPI, {size} ranks]    inside MPI calls (slowest rank): to another node {mx('far'):.3f}s, same node {mx('near'):.3f}s, "
                      f"Barrier {mx('barrier'):.3f}s, collectives {mx('coll'):.3f}s; sent per rank {best[4][0]['bytes_far'] / 2 ** 30:.1f} GiB to another node, "
                      f"{best[4][0]['bytes_near'] / 2 ** 30:.1f} GiB inside the node; calls: data {best[4][0]['n_data']:.0f}, wait {best[4][0]['n_wait']:.0f}, "
                      f"test {best[4][0]['n_test']:.0f}, Barrier {best[4][0]['n_barrier']:.0f}", flush=True)
            if rank == 0:
                print(f"[lightning.gpu MPI, {size} ranks] {case} n={n} order={args.order}: fwd {best[1]:8.3f}s  bwd {best[2]:8.3f}s  "
                      f"total {best[0]:8.3f}s   cost={cost:+.8f}" + (f"   mpi_buf_size={args.mpi_buf_size} MiB" if args.mpi_buf_size else ''),
                      flush=True)
                if args.save_grads:
                    os.makedirs(os.path.dirname(args.save_grads), exist_ok=True)
                    np.save(f"{args.save_grads}_{case.replace('/', '-')}_n{n}.npy", np.concatenate([[cost], grads]))
            del dev, state, final
        except Exception as e:
            row['error'] = str(e)[-200:]
            if rank == 0:
                print(f"[lightning.gpu MPI, {size} ranks] {case} n={n}: FAILED {row['error']}", flush=True)
        rows.append(row)
        if rank == 0 and args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
