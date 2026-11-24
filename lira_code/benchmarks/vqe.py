import argparse
import json
import os
import sys
import time

_t_start = time.perf_counter()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = next((a.split('=')[1] for a in sys.argv if a.startswith('--engine=')), None) or \
    (sys.argv[sys.argv.index('--engine') + 1] if '--engine' in sys.argv else 'lira')
if ENGINE in ('lgpu', 'kokkos'):
    if os.environ.get('LIRA_GPU_BEFORE_MPI') and os.environ.get('SLURM_LOCALID') is not None:
        from cupy.cuda import runtime as _rt
        _rt.setDevice(int(os.environ['SLURM_LOCALID']) % _rt.getDeviceCount())
    if ENGINE == 'lgpu':
        _flags = sys.getdlopenflags()
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
        import pennylane_lightning.lightning_gpu_ops
        sys.setdlopenflags(_flags)
    else:
        import pennylane_lightning.lightning_kokkos_ops
    from mpi4py import MPI
else:
    from lira.distributed import MPI, DistributedEngine

import numpy as np

from lira.planner import make_plan
from benchmarks.single_gpu import build, observable


def tfim_line_e0(n: int, h: float = 1.0) -> float:
    M = np.diag([h] * n) + np.diag([1.0] * (n - 1), 1)
    return -float(np.linalg.svd(M, compute_uv=False).sum())


def tfim_ring_e0(n: int, h: float = 1.0) -> float:
    k = np.pi * (2 * np.arange(n) + 1) / n
    return -float(np.sqrt(1 + h * h - 2 * h * np.cos(k)).sum())


class Adam:
    def __init__(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps, self.k, self.m, self.v = lr, b1, b2, eps, 0, None, None

    def step(self, x, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(x), np.zeros_like(x)
        self.k += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh, vh = self.m / (1 - self.b1 ** self.k), self.v / (1 - self.b2 ** self.k)
        return x - self.lr * mh / (np.sqrt(vh) + self.eps)


class Lira:
    def __init__(self, plan, terms, comm, name: str):
        from lira.scheme import WINDOW, variant
        G = comm.Get_size()
        opt = variant(name, G)
        self.plan, self.terms, self.w = plan, terms, WINDOW
        self.late = opt['late']
        self.eng = DistributedEngine(plan.n, comm, np.complex64, early=opt['early'], stream=opt['stream'], mix=opt['mix'])
        self.eng.compile(plan, window=self.w)

    def __call__(self, theta):
        self.eng.rebind(self.plan, theta, window=self.w)
        c, g, t, st = self.eng.run(self.plan, self.terms, window=self.w, fuse_obs=not self.late, late_obs=self.late)
        return c, g


class Lightning:
    def __init__(self, plan, terms, comm, device='lightning.gpu'):
        import pennylane as qml
        from lira.lightning_ref import _ROT, build_tape, tape_grads
        self.plan, self.tape_grads = plan, tape_grads
        self.tape, self.slots = build_tape(plan, terms)
        self.scale = [g.params[1] if len(g.params) == 3 else 1.0
                      for g, p in zip(plan.gates, plan.param_index) if g.name.lower() in _ROT and p >= 0]
        self.allp = list(self.tape.get_parameters(trainable_only=False))
        self.pos = list(self.tape.trainable_params)
        if comm.Get_size() == 1:
            self.dev = qml.device(device, wires=plan.n, c_dtype=np.complex64)
            if self.dev._statevector is None:
                self.dev._statevector = self.dev.LightningStateVector(num_wires=plan.n, dtype=self.dev.c_dtype)
        else:
            self.dev = qml.device(device, wires=plan.n, mpi=True, c_dtype=np.complex64)

    def __call__(self, theta):
        p = list(self.allp)
        for k, s, sc in zip(self.pos, self.slots, self.scale):
            p[k] = sc * float(theta[s])
        tape = self.tape.bind_new_parameters(p, list(range(len(p))))
        dev, state = self.dev, self.dev._statevector
        state.reset_state()
        final = state.get_final_state(tape)
        cost = float(np.squeeze(dev.LightningMeasurements(final).measure_final_state(tape)))
        jac = np.asarray(dev.LightningAdjointJacobian(final, batch_obs=False).calculate_jacobian(tape))
        return cost, self.tape_grads(self.plan, self.slots, jac)


def _row(args, n, G, plan, e0, t_plan, t_build, t_first, hist, partial):
    return dict(engine=args.engine, case=args.case, n=n, gpus=G, obs=args.obs, iters=args.iters, done=len(hist), lr=args.lr,
                init=args.init, seed=args.seed, variant=args.variant if args.engine == 'lira' else '',
                order='circuit' if args.engine in ('lgpu', 'kokkos') else 'planner', params=plan.n_params,
                e0=e0, t_import=_t_import, t_plan=t_plan, t_build=t_build, t_first=t_first,
                t_iter=float(np.median([h['dt'] for h in hist[1:]])) if len(hist) > 1 else None, partial=partial, history=hist)


def _write(path, row):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(row, fh, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['lira', 'lgpu', 'kokkos'], default='lira')
    ap.add_argument('--case', default='tfim1d_4')
    ap.add_argument('--obs', default='native')
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--init', type=float, default=0.1, help='start: every parameter uniform in [-init, init]')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', default='full', help='lira: a variant of lira/scheme.py')
    ap.add_argument('--time-limit', type=float, default=0.0, help='seconds of iterations; 0 = all')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    rank, G = comm.Get_rank(), comm.Get_size()

    t0 = time.perf_counter()
    n, gates = build(args.case, args.n, 0)
    if args.engine in ('lgpu', 'kokkos'):
        plan = make_plan(n, gates, order=list(range(n)), full_width=True, keep_order=True)
    else:
        plan = make_plan(n, gates)
    terms = observable(args.obs, n, plan, args.case)
    t_plan = time.perf_counter() - t0
    t0 = time.perf_counter()
    f = Lira(plan, terms, comm, args.variant) if args.engine == 'lira' else \
        Lightning(plan, terms, comm, 'lightning.kokkos' if args.engine == 'kokkos' else 'lightning.gpu')
    t_build = time.perf_counter() - t0
    theta = np.random.default_rng(args.seed).uniform(-args.init, args.init, plan.n_params)
    opt = Adam(args.lr)
    e0 = None
    if args.obs == 'native' and args.case.startswith('tfim1d_'):
        e0 = tfim_line_e0(n)
    elif args.obs == 'native' and args.case.startswith('tfim1dp_'):
        e0 = tfim_ring_e0(n)

    hist, t_first = [], None
    comm.Barrier()
    t_loop = time.perf_counter()
    for it in range(args.iters):
        t1 = time.perf_counter()
        cost, grads = f(theta)
        comm.Barrier()
        dt = time.perf_counter() - t1
        if t_first is None:
            t_first = dt
        hist.append(dict(it=it, energy=cost, t=time.perf_counter() - t_loop, dt=dt, gnorm=float(np.linalg.norm(grads))))
        theta = opt.step(theta, grads)
        if rank == 0 and args.out:
            _write(args.out, _row(args, n, G, plan, e0, t_plan, t_build, t_first, hist, partial=True))
        if rank == 0 and (it < 3 or it % 20 == 0 or it == args.iters - 1):
            print(f'[{args.engine} {G} GPUs] {args.case} n={n} it {it:4d}  E={cost:+.8f}' + (f'  E-E0={cost - e0:.2e}' if e0 is not None else '')
                  + f'  {dt:.3f}s/it', flush=True)
        if args.time_limit and comm.bcast(time.perf_counter() - t_loop > args.time_limit, root=0):
            break
    row = _row(args, n, G, plan, e0, t_plan, t_build, t_first, hist, partial=False)
    if rank == 0:
        print(f"[{args.engine} {G} GPUs] {args.case} n={n}: plan {t_plan:.2f}s, build {t_build:.2f}s, first iteration {t_first:.2f}s, "
              f"then {row['t_iter']:.3f}s per iteration; {len(hist)} iterations, final E={hist[-1]['energy']:+.8f}", flush=True)
        if args.out:
            _write(args.out, row)


_t_import = time.perf_counter() - _t_start

if __name__ == '__main__':
    main()
    MPI.COMM_WORLD.Barrier()
    sys.stdout.flush()
    os._exit(0)
