import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lira.engine as _engine
import lira.distributed as _mg
import lira.exchanges as _mu
import lira.planner as _pl
from lira.distributed import MPI, DistributedEngine
from benchmarks.single_gpu import build, observable
from lira.scheme import variant

SETUP = defaultdict(float)
_stack = []


def _timed(name, fn):
    def wrap(*a, **k):
        _stack.append(0.0)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            inner = _stack.pop()
            SETUP[name] += dt - inner
            if _stack:
                _stack[-1] += dt
    return wrap


_pl.ancestor_supports = _timed('dependency', _pl.ancestor_supports)
_pl.gate_weights = _timed('activation', _pl.gate_weights)
_pl.WidthProblem.optimise = _timed('activation', _pl.WidthProblem.optimise)
_make_plan = _timed('reorder', _pl.make_plan)
_mu.fuse = _timed('fusion', _mu.fuse)
_mu.windows = _timed('fusion', _mu.windows)
_mu.compile_plan = _timed('distribution', _mu.compile_plan)
_mg.compile_plan = _mu.compile_plan
_mg._choose_early = _timed('distribution', _mg._choose_early)
_engine.LiraEngine.prepare_windows = _timed('kernel_compile', _engine.LiraEngine.prepare_windows)
_engine.LiraEngine.allocate = _timed('allocation', _engine.LiraEngine.allocate)


class ProfEngine(DistributedEngine):
    profile = False

    def reset_prof(self):
        self.P = defaultdict(float)
        self._ph = 'forward'

    @contextmanager
    def _tim(self, cat):
        self._sync()
        self._over.synchronize()
        c0, f0, t0 = self.tcomm, self.tcomm_far, time.perf_counter()
        yield
        self._sync()
        self._over.synchronize()
        dt = time.perf_counter() - t0
        dc, df = self.tcomm - c0, self.tcomm_far - f0
        P, ph = self.P, self._ph
        P[f'{ph}/{cat}'] += max(dt - dc, 0.0)
        P[f'{ph}/comm_intra'] += dc - df
        P[f'{ph}/comm_inter'] += df

    def _late(self, steps, terms):
        if not self.profile:
            return super()._late(steps, terms)
        self._ph = 'observable'
        with self._tim('term_entry_planning'):
            return super()._late(steps, terms)

    def _observable(self, terms, loc, fuse_obs=False):
        if not self.profile:
            return super()._observable(terms, loc, fuse_obs)
        self._ph = 'observable'
        with self._tim('observable'):
            return super()._observable(terms, loc, fuse_obs)

    def _run_steps(self, steps, forward: bool, late=None):
        if not self.profile:
            return super()._run_steps(steps, forward, late)
        self._ph = 'forward' if forward else 'backward'
        r = self.r
        e = self.eng[r]
        self._backward = not forward
        for i in (range(len(steps)) if forward else range(len(steps) - 1, -1, -1)):
            st = steps[i]
            if late and i in late:
                with self._tim('late_term_entry'):
                    self._inject(*late[i])
            bufs = [self.psi] if forward or not self._live else [self.psi, self.lam]
            kind = st[0]
            cat = {'single': 'one_gpu', 'multi': 'kernels', 'lswap': 'local_swap', 'jxchg': 'exchange_copy',
                   'sx': 'streamed_exchange', 'spread': 'spread'}.get(kind, 'exchange_copy')
            with self._tim(cat):
                if kind == 'single':
                    if r == 0:
                        if forward:
                            e._forward_op(self.psi[r], st[1])
                        else:
                            e._backward_op(self.psi[r], self.lam[r], self.gr[r], st[1])
                elif kind == 'multi':
                    acts = st[2].get(r)
                    if acts:
                        for act in (acts if forward else reversed(acts)):
                            (self._fwd if forward else self._bwd)(r, act, st[3])
                elif kind == 'lswap':
                    if r in st[1]:
                        a, b, n4 = np.int32(st[2]), np.int32(st[3]), 1 << (st[4] - 2)
                        if forward:
                            e._launch('f_swap', n4, (self.psi[r], np.int64(n4), a, b))
                        else:
                            e._launch('b_swap', n4, (self.psi[r], self.lam[r], np.int64(n4), a, b))
                elif kind == 'jxchg':
                    self._jexchange(bufs, st[1], st[2], st[3])
                elif kind == 'sx':
                    self._sexchange(bufs, st[1], st[2], st[3], st[4], *self._sx_work(st, i, forward, late))
                    self._live |= bool(late and ('mid', i) in late)
                elif kind == 'spread':
                    self._spread(bufs, st[1], st[2], forward)
                else:
                    self._exchange(bufs, st[1], st[2], forward, st[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--cases', default='sel_4,tfim1dp_n/2')
    ap.add_argument('--obs', default='auto', help="auto: native for tfim*, meanz otherwise")
    ap.add_argument('--window', default='4,6,1')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    rank, G = comm.Get_rank(), comm.Get_size()
    sch = 'full'
    opt = variant(sch, G)
    w = [int(x) for x in args.window.split(',')]
    win = w[0] if len(w) == 1 else (w[0], w[1], bool(w[2]))
    rows = []
    eng, alloc_t = None, {}
    for case in args.cases.split(','):
        obs = args.obs if args.obs != 'auto' else ('native' if case.startswith('tfim') else 'meanz')
        SETUP.clear()
        job = None
        if rank == 0:
            n, gates = build(case, args.n, 0)
            plan = _make_plan(n, gates)
            terms = observable(obs, n, plan, case)
            job = (n, plan, terms, dict(SETUP))
        n, plan, terms, plan_setup = comm.bcast(job)
        SETUP.clear()
        if eng is None:
            eng = ProfEngine(n, comm, np.complex64, joint=True, early=opt['early'], stream=opt['stream'], mix=opt['mix'])
        assert eng.n == n, 'all circuits of one run must have the same number of qubits'
        eng._compiled.clear()
        eng._keep.clear()
        eng.compile(plan, window=win)
        if not alloc_t:
            eng.allocate()
            alloc_t = {'allocation': SETUP.get('allocation', 0.0)}
        else:
            SETUP.update(alloc_t)
        fo, late = opt['fuse_obs'], opt['late']
        os.environ['LIRA_PHASE_SYNC'] = '0'
        t0 = time.perf_counter()
        eng.run(plan, terms, fuse=True, window=win, fuse_obs=fo, late_obs=late)
        first = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
        prod = []
        for _ in range(args.repeat):
            c, g, t, _st = eng.run(plan, terms, fuse=True, window=win, fuse_obs=fo, late_obs=late)
            prod.append(t['total'])
        g_prod = g
        os.environ['LIRA_PHASE_SYNC'] = '1'
        eng.profile = True
        best = None
        for _ in range(args.repeat):
            eng.reset_prof()
            c2, g2, t2, st2 = eng.run(plan, terms, fuse=True, window=win, fuse_obs=fo, late_obs=late)
            if best is None or t2['total'] < best[1]['total']:
                best = (dict(eng.P), t2, st2)
        eng.profile = False
        P, t2, st2 = best
        allP = comm.gather(P)
        setup = comm.gather(dict(SETUP))
        diff = float(np.max(np.abs(np.asarray(g2) - np.asarray(g_prod)))) if g2 is not None else None
        if rank == 0:
            ph_sum = {ph: sum(v for k, v in P.items() if k.startswith(ph + '/')) for ph in ('forward', 'observable', 'backward')}
            other = {ph: t2[ph] - ph_sum[ph] for ph in ph_sum}
            keys = sorted({k for p in allP for k in p})
            row = dict(case=case, n=n, gpus=G, obs=obs, variant=sch, window=args.window, nodes=st2['nodes'],
                       total_prod=min(prod), first_run=first, total_prof=t2['total'], phases={k: t2[k] for k in ('forward', 'observable', 'backward')},
                       rank0=P, other=other, mean={k: float(np.mean([p.get(k, 0.0) for p in allP])) for k in keys},
                       max={k: float(np.max([p.get(k, 0.0) for p in allP])) for k in keys},
                       setup=dict(plan_setup, **setup[0]), setup_max={k: max(s.get(k, 0.0) for s in setup) for k in setup[0]},
                       cost=c2, grad_diff_prof_vs_prod=diff, exchanges=st2['exchanges'], oneway=st2['oneway'], lswaps=st2['lswaps'])
            rows.append(row)
            print(f"\n[profile {G} GPUs] {case} n={n} obs={obs} {sch}: normal run {min(prod):.3f}s, profiled {t2['total']:.3f}s "
                  f"(first run {first:.2f}s), |grad prof - grad normal| = {diff:.1e}", flush=True)
            print('  setup (rank 0, s): ' + ', '.join(f'{k} {v:.3f}' for k, v in row['setup'].items()), flush=True)
            for ph in ('forward', 'observable', 'backward'):
                parts = sorted(((k.split('/', 1)[1], v) for k, v in P.items() if k.startswith(ph + '/')), key=lambda x: -x[1])
                print(f"  {ph:10s} {t2[ph]:8.3f}s = " + ', '.join(f'{k} {v:.3f}' for k, v in parts if v > 1e-4)
                      + f", other {other[ph]:.3f}", flush=True)
            if args.out:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out, 'w') as f:
                    json.dump(rows, f, indent=1)
    if eng is not None:
        eng.release()


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
