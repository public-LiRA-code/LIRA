import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_prof = None
if os.environ.get('LIRA_MPIPROF'):
    import ctypes
    _prof = ctypes.CDLL(os.environ['LIRA_MPIPROF'], mode=os.RTLD_LAZY | os.RTLD_GLOBAL)
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    from mpi4py import MPI as _MPI
    sys.setdlopenflags(_flags)
from lira.distributed import MPI, DistributedEngine
from lira.planner import make_plan
from benchmarks.single_gpu import build, observable
from lira.scheme import CHUNK_BITS, VARIANTS, variant, window as parse_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--cases', default='sel_4')
    ap.add_argument('--obs', choices=['tfim', 'z0', 'ixyz', 'parity', 'native', 'meanz'], default='meanz')
    ap.add_argument('--variants', default='full', help='comma list of ' + ', '.join(VARIANTS) + ' (lira/scheme.py)')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--no-warm', action='store_true', help='time the first pass too (large runs)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk-bits', type=int, default=CHUNK_BITS)
    ap.add_argument('--window', default='4,6,1', help='window kernels (lira/windows.py): K or K,Kperm,rot')
    ap.add_argument('--save-grads', default='', help='prefix: writes <prefix>_<case>_n<n>.npy (first variant)')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    rank, G = comm.Get_rank(), comm.Get_size()
    win = parse_window(args.window)
    sel = [v for v in args.variants.split(',') if v]
    unknown = [v for v in sel if v not in VARIANTS]
    if unknown:
        raise SystemExit(f'unknown variants {unknown}; choose from {list(VARIANTS)}')
    rows = []
    for case in args.cases.split(','):
        job = None
        if rank == 0:
            t0 = time.perf_counter()
            n, gates = build(case, args.n, args.seed)
            plan = make_plan(n, gates)
            plans = dict(adaptive=plan)
            if any(VARIANTS[v]['plan'] == 'full-width' for v in sel):
                plans['full-width'] = make_plan(n, gates, order=plan.order, full_width=True)
            terms = observable(args.obs, n, plan, case)
            print(f"\n=== {case}: n={n} gates={len(gates)} params={plan.n_params} obs={args.obs} ({len(terms)} terms)  {G} ranks, "
                  f"{n - G.bit_length() + 1} local qubits  width-adaptive bound {1 / plan.r_work['grad']:.2f}x  "
                  f"(planning {time.perf_counter() - t0:.1f}s)", flush=True)
            job = (n, plans, terms, time.perf_counter() - t0)
        n, plans, terms, plan_time = comm.bcast(job)
        saved = False
        eng, made = None, None
        for name in sel:
            opt = variant(name, G)
            pname, w, sch = opt['plan'], win, name
            if made != (opt['early'], opt['stream'], opt['mix']):
                if eng is not None:
                    eng.release()
                eng = DistributedEngine(n, comm, np.complex64, chunk_bits=args.chunk_bits, joint=True, early=opt['early'], stream=opt['stream'],
                                   mix=opt['mix'])
                made = (opt['early'], opt['stream'], opt['mix'])
            fo, late = opt['fuse_obs'], opt['late']
            pl = plans[pname]
            t0 = time.perf_counter()
            eng.compile(pl, window=w)
            t_compile = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
            best = None
            for i in range(args.repeat + (0 if args.no_warm else 1)):
                if _prof:
                    _prof.mpiprof_reset()
                res = eng.run(pl, terms, fuse=True, window=w, fuse_obs=fo, late_obs=late)
                if _prof:
                    buf = (ctypes.c_double * 16)()
                    _prof.mpiprof_get(buf)
                    res[3]['mpiprof'] = comm.gather(dict(far=buf[0], near=buf[1], barrier=buf[2], coll=buf[3], bytes_far=buf[4], bytes_near=buf[5]))
                if (i > 0 or args.no_warm) and (best is None or res[2]['total'] < best[2]['total']):
                    best = res
            c, g, t, st = best
            if rank:
                continue
            vol = (st['moved_forward'] + st['moved_observable'] + st['moved_backward']) / 2.0 ** n
            rows.append(dict(case=case, n=n, engine=f'mpi_{G}', plan=pname, window=args.window if w else '', fuse_obs=fo, joint=True,
                             variant=sch, late=late, mix=eng.mix, spread=st.get('spread', 0), stream=st.get('stream', 0), stream_ops=st.get('stream_ops', 0),
                             grad_l2=float(np.linalg.norm(g)),
                             bits=st.get('bits'), gpus=G, obs=args.obs, **t, exchanges=st['exchanges'],
                             oneway=st['oneway'], lswaps=st['lswaps'], moved_state_vectors=vol, moved_forward=st['moved_forward'] / 2.0 ** n,
                             moved_backward=st['moved_backward'] / 2.0 ** n, moved_observable=st['moved_observable'] / 2.0 ** n,
                             cost=c, grad_max=float(np.abs(g).max()), local_qubits=st['local_qubits'], compile=t_compile,
                             comm_forward=st['comm_forward'], comm_observable=st['comm_observable'], comm_backward=st['comm_backward'],
                             comm_far=st['comm_far'], moved_far=st['moved_far'] / 2.0 ** n, nodes=st['nodes'],
                             comm_far_forward=st.get('comm_far_forward'), comm_far_observable=st.get('comm_far_observable'),
                             comm_far_backward=st.get('comm_far_backward'), plan_time=plan_time,
                             layout=os.environ.get('SLURM_DISTRIBUTION', ''),
                             chunk_bits=args.chunk_bits))
            print(f"[lira MPI, {G} ranks] {case} n={n} {sch + ' (' + pname + ')':38s}: fwd {t['forward']:8.3f}s  obs {t['observable']:7.3f}s  bwd {t['backward']:8.3f}s  "
                  f"total {t['total']:8.3f}s   exchanges {st['exchanges']:3d} + {st['oneway']} one-way, moved {vol:6.2f} state vectors   "
                  f"cost={c:+.8f}  (compile {t_compile:.1f}s)\n"
                  f"[lira MPI, {G} ranks]    of which inside MPI calls (slowest rank): fwd {st['comm_forward']:.3f}s  obs {st['comm_observable']:.3f}s  "
                  f"bwd {st['comm_backward']:.3f}s   | to other nodes ({st['nodes']} nodes): {st['moved_far'] / 2.0 ** n:.2f} state vectors, "
                  f"{st['comm_far']:.3f}s in those calls", flush=True)
            if st.get('mpiprof'):
                rows[-1]['mpiprof'] = st['mpiprof']
                print(f"[lira MPI, {G} ranks]    the same from tools/libmpiprof.so (slowest rank): to another node {max(x['far'] for x in st['mpiprof']):.3f}s, "
                      f"same node {max(x['near'] for x in st['mpiprof']):.3f}s, Barrier {max(x['barrier'] for x in st['mpiprof']):.3f}s", flush=True)
            if args.save_grads and not saved:
                saved = True
                if rank == 0:
                    os.makedirs(os.path.dirname(args.save_grads), exist_ok=True)
                    np.save(f"{args.save_grads}_{case.replace('/', '-')}_n{n}.npy", np.concatenate([[c], g]))
            if args.out:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out, 'w') as f:
                    json.dump(rows, f, indent=1)
        eng.release()
        del eng


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
    MPI.COMM_WORLD.Barrier()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
