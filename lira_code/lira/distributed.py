import os
import threading
import time
from typing import List, Sequence

import cupy as cp
import numpy as np

_local = os.environ.get('SLURM_LOCALID')
if _local is not None:
    cp.cuda.Device(int(_local) % cp.cuda.runtime.getDeviceCount()).use()
from mpi4py import MPI

from .engine import LiraEngine, PauliTerm
from .exchanges import MultiGPUEngine, _choose_early, _local_terms, _masks, _part_of, _rank_ops, _rank_with, compile_plan, rebind_steps
from .windows import config
from .planner import Plan
from .scheme import CHUNK_BITS, EARLY, STREAM, WINDOW

FAR_INFLIGHT = 2


class DistributedEngine(MultiGPUEngine):

    def __init__(self, n: int, comm=None, dtype=np.complex64, chunk_bits: int = CHUNK_BITS, device: int = None, joint: bool = True,
                 early: int = EARLY, stream: int = STREAM, intra: int = 0, mix: bool = None):
        self.comm = comm = comm or MPI.COMM_WORLD
        G, r = comm.Get_size(), comm.Get_rank()
        assert G & (G - 1) == 0, 'number of ranks must be a power of two'
        self.n, self.G, self.g, self.r = n, G, G.bit_length() - 1, r
        self.m = n - self.g
        assert self.m >= 2
        local = os.environ.get('SLURM_LOCALID')
        self.dev = (int(local) if local is not None else r) % cp.cuda.runtime.getDeviceCount() if device is None else device
        cp.cuda.Device(self.dev).use()
        names = comm.allgather(MPI.Get_processor_name())
        self.far = [nm != names[r] for nm in names]
        if names.count(names[r]) > 1 and cp.cuda.runtime.getDeviceCount() == 1:
            raise RuntimeError('several ranks per node but one visible GPU each: in-node GPU exchanges would hang; '
                               'launch with --gpus-per-node (all GPUs of the node visible), not --gpus-per-task=1')
        self.devs = {r: self.dev}
        self.dtype = np.dtype(dtype)
        self.chunk = 1 << min(chunk_bits, self.m - 1)
        self.eng = {r: LiraEngine(self.m, dtype)}
        self.ct = self.eng[r].ct
        self.psi = self.lam = self.scr = None
        self.joint, self.early, self.stream, self.intra = joint, early, stream, intra
        self.mix = int(any(self.far) if mix is None else mix)
        self._lock = threading.Lock()
        self._fscr = []
        self._live, self._fresh = True, False
        self._over = cp.cuda.Stream(non_blocking=True)
        self._compiled = {}
        self._keep = {}

    def allocate(self):
        if self.psi is None:
            s, l = self.eng[self.r].allocate()
            self.psi, self.lam = {self.r: s}, {self.r: l}
            self.scr = cp.empty(self.chunk, dtype=self.dtype)
            if self.mix:
                self._fscr = [cp.empty(self.chunk, dtype=self.dtype) for _ in range(FAR_INFLIGHT)]

    def release(self):
        self.psi = self.lam = self.scr = None
        self._fscr = []
        self.eng[self.r].release()

    def _sync(self):
        cp.cuda.get_current_stream().synchronize()

    def compile(self, plan: Plan, fuse_ops: bool = True, margin: float = 1.0, window=WINDOW):
        key = (id(plan), fuse_ops, margin, config(window))
        if key not in self._compiled:
            early = self.early if getattr(self, '_force_early', False) else _choose_early(plan, self.g, fuse_ops, margin, self.joint, self.early)
            self._keep[key] = keep = {}
            self._compiled[key] = compile_plan(plan, self.g, fuse_ops, margin, only_rank=self.r, window=window, joint=self.joint, early=early,
                                               stream=self.stream if self.joint else 0, intra=self.intra, keep=keep)
            if window:
                self.eng[self.r].prepare_windows(_rank_ops(self._compiled[key][0], self.r))
        return self._compiled[key]

    def rebind(self, plan: Plan, theta, fuse_ops: bool = True, margin: float = 1.0, window=WINDOW):
        key = (id(plan), fuse_ops, margin, config(window))
        if key not in self._compiled:
            self.compile(plan, fuse_ops, margin, window)
        steps = rebind_steps(self._keep[key], theta)
        self._compiled[key] = (steps,) + tuple(self._compiled[key][1:])
        if window:
            self.eng[self.r].prepare_windows(_rank_ops(steps, self.r))

    def _count(self, partner: int, seconds: float, amplitudes: int):
        with self._lock:
            self._count_(partner, seconds, amplitudes)

    def _count_(self, partner: int, seconds: float, amplitudes: int):
        self.tcomm += seconds
        self.sent += amplitudes
        if self.far[partner]:
            self.tcomm_far += seconds
            self.sent_far += amplitudes

    def _swap(self, view, partner: int, scr=None):
        C, scr = self.chunk, self.scr if scr is None else scr
        for off in range(0, view.size, C):
            mine = view[off:off + C]
            self._sync()
            t0 = time.perf_counter()
            self.comm.Sendrecv(mine, dest=partner, sendtag=1, recvbuf=scr[:mine.size], source=partner, recvtag=1)
            self._count(partner, time.perf_counter() - t0, mine.size)
            mine[...] = scr[:mine.size]

    def _swap_many(self, items):
        far = [(v, p) for v, p in items if self.far[p]]
        near = [(v, p) for v, p in items if not self.far[p]]
        if not self.mix or not far or not near:
            for view, partner in items:
                self._swap(view, partner)
            return
        if self.mix == 2:
            errors = []

            def far_partners():
                try:
                    with cp.cuda.Device(self.dev):
                        for v, p in far:
                            self._swap(v, p, self._fscr[0])
                except BaseException as exc:
                    errors.append(exc)
            self._sync()
            th = threading.Thread(target=far_partners)
            th.start()
            for v, p in near:
                self._swap(v, p)
            th.join()
            if errors:
                raise errors[0]
            return
        C = self.chunk
        seq = {}
        fchunks = []
        for v, p in far:
            for off in range(0, v.size, C):
                seq[p] = seq.get(p, -1) + 1
                fchunks.append((v[off:off + C], p, 100 + seq[p]))
        nchunks = [(v[off:off + C], p) for v, p in near for off in range(0, v.size, C)]
        self._sync()
        pending, nxt = [], [0]

        def post():
            mine, p, tag = fchunks[nxt[0]]
            s = self._fscr[nxt[0] % FAR_INFLIGHT][:mine.size]
            pending.append(([self.comm.Irecv(s, source=p, tag=tag), self.comm.Isend(mine, dest=p, tag=tag)], mine, s, p))
            nxt[0] += 1

        def retire(block: bool) -> bool:
            reqs, mine, s, p = pending[0]
            t0 = time.perf_counter()
            if block:
                MPI.Request.Waitall(reqs)
            elif not MPI.Request.Testall(reqs):
                return False
            self._count(p, time.perf_counter() - t0, mine.size)
            mine[...] = s
            self._sync()
            pending.pop(0)
            if nxt[0] < len(fchunks):
                post()
            return True
        while nxt[0] < len(fchunks) and len(pending) < FAR_INFLIGHT:
            post()
        for mine, p in nchunks:
            t0 = time.perf_counter()
            self.comm.Sendrecv(mine, dest=p, sendtag=1, recvbuf=self.scr[:mine.size], source=p, recvtag=1)
            self._count(p, time.perf_counter() - t0, mine.size)
            mine[...] = self.scr[:mine.size]
            self._sync()
            while pending and retire(False):
                pass
        while pending:
            retire(True)

    def _send(self, view, partner: int):
        for off in range(0, view.size, self.chunk):
            self._sync()
            t0 = time.perf_counter()
            self.comm.Send(view[off:off + self.chunk], dest=partner, tag=3)
            self._count(partner, time.perf_counter() - t0, min(self.chunk, view.size - off))

    def _recv(self, view, partner: int):
        for off in range(0, view.size, self.chunk):
            self._sync()
            t0 = time.perf_counter()
            self.comm.Recv(view[off:off + self.chunk], source=partner, tag=3)
            self._count(partner, time.perf_counter() - t0, 0)

    def _exchange(self, bufs, pairs, oneway: bool, forward: bool, wl: int):
        me, half = self.r, 1 << (wl - 1)
        for a, b in pairs:
            if me != a and me != b:
                continue
            for B in bufs:
                buf = B[me]
                if not oneway:
                    self._swap(buf[half:2 * half] if me == a else buf[:half], b if me == a else a)
                elif forward:
                    if me == a:
                        self._send(buf[half:2 * half], b)
                        buf[half:2 * half].fill(0)
                    else:
                        self._recv(buf[:half], a)
                elif me == b:
                    self._send(buf[:half], a)
                else:
                    self._recv(buf[half:2 * half], b)
            return

    def _spread(self, bufs, J, wl: int, forward: bool):
        me, blk = self.r, 1 << (wl - len(J))
        dest = {_rank_with(0, T, J): T for T in range(1, 1 << len(J))}
        if me != 0 and me not in dest:
            return
        for B in bufs:
            buf = B[me]
            if me == 0:
                for r, T in sorted(dest.items()):
                    (self._send if forward else self._recv)(buf[T * blk:(T + 1) * blk], r)
                if forward:
                    buf[blk:1 << wl].fill(0)
            else:
                (self._recv if forward else self._send)(buf[:blk], 0)

    def _jexchange(self, bufs, ranks, J, wl: int):
        me = self.r
        if me not in ranks:
            return
        blk = 1 << (wl - len(J))
        items = []
        for d in range(1, 1 << len(J)):
            T = _part_of(me, J) ^ d
            partner = _rank_with(me, T, J)
            for B in bufs:
                items.append((B[me][T * blk:(T + 1) * blk], partner))
        self._swap_many(items)

    def _sexchange(self, bufs, ranks, J, wl: int, js: int, before, after):
        me = self.r
        if me not in ranks:
            return
        k, w = len(J), wl - js
        blk, ng = 1 << (w - k), 1 << js
        own = _part_of(me, J)
        self._sync()
        with self._over:
            before(me, 0)
        for M in range(ng):
            self._over.synchronize()
            self._sync()
            with self._over:
                if M + 1 < ng:
                    before(me, M + 1)
                if M >= 1:
                    after(me, M - 1)
            items = []
            for d in range(1, 1 << k):
                T = own ^ d
                partner = _rank_with(me, T, J)
                for B in bufs:
                    items.append((B[me][(M << w) + T * blk:(M << w) + (T + 1) * blk], partner))
            self._swap_many(items)
        self._over.synchronize()
        self._sync()
        after(me, ng - 1)

    def _inject(self, terms, ranks, wl: int, loc):
        m, r, slot = self.m, self.r, np.int32(self._slot)
        first = self._fresh and not self._live
        self._live = True
        if r not in ranks:
            return
        loc = list(range(self.n)) if loc is None else loc
        e, psi, lam, gr = self.eng[r], self.psi[r], self.lam[r], self.gr[r]
        local = _local_terms(terms, loc, m, [r])[r]
        if local:
            e._inject(lam, psi, gr, local, wl, int(slot), first=first)
        C = min(self.chunk, 1 << wl)
        for coeff, paulis in terms:
            xl, zl, xg, zg, ny = _masks(paulis, loc, m)
            if xg == 0:
                continue
            src = r ^ xg
            c = coeff * (1j) ** ny * (-1.0 if bin(src & zg).count('1') & 1 else 1.0)
            xlo, xhi = xl & (C - 1), xl & ~(C - 1)
            for q0 in range(0, 1 << wl, C):
                self._sync()
                t0 = time.perf_counter()
                self.comm.Sendrecv(psi[q0:q0 + C], dest=src, sendtag=2, recvbuf=self.scr[:C], source=src, recvtag=2)
                self._count(src, time.perf_counter() - t0, C)
                p0 = q0 ^ xhi
                e._launch('b_obs_x', C, (lam[p0:p0 + C], self.scr[:C], psi[p0:p0 + C], gr, np.int64(C), np.int64(q0), np.int64(xlo),
                                         np.int64(zl), self.ct(c), slot))

    def _run_steps(self, steps, forward: bool, late=None):
        r = self.r
        e = self.eng[r]
        self._backward = not forward
        for i in (range(len(steps)) if forward else range(len(steps) - 1, -1, -1)):
            st = steps[i]
            if late and i in late:
                self._inject(*late[i])
            bufs = [self.psi] if forward or not self._live else [self.psi, self.lam]
            if st[0] == 'single':
                if r == 0:
                    if forward:
                        e._forward_op(self.psi[r], st[1])
                    else:
                        e._backward_op(self.psi[r], self.lam[r], self.gr[r], st[1])
            elif st[0] == 'multi':
                acts = st[2].get(r)
                if acts:
                    for act in (acts if forward else reversed(acts)):
                        (self._fwd if forward else self._bwd)(r, act, st[3])
            elif st[0] == 'lswap':
                if r in st[1]:
                    a, b, n4 = np.int32(st[2]), np.int32(st[3]), 1 << (st[4] - 2)
                    if forward:
                        e._launch('f_swap', n4, (self.psi[r], np.int64(n4), a, b))
                    else:
                        e._launch('b_swap', n4, (self.psi[r], self.lam[r], np.int64(n4), a, b))
            elif st[0] == 'jxchg':
                self._jexchange(bufs, st[1], st[2], st[3])
            elif st[0] == 'sx':
                self._sexchange(bufs, st[1], st[2], st[3], st[4], *self._sx_work(st, i, forward, late))
                self._live |= bool(late and ('mid', i) in late)
            elif st[0] == 'spread':
                self._spread(bufs, st[1], st[2], forward)
            else:
                self._exchange(bufs, st[1], st[2], forward, st[3])

    def _observable(self, terms: Sequence[PauliTerm], loc: List[int], fuse_obs: bool = False):
        m, C, r = self.m, self.chunk, self.r
        N = 1 << m
        e, psi, lam, scr = self.eng[r], self.psi[r], self.lam[r], self.scr
        local = _local_terms(terms, loc, m, [r])[r] if fuse_obs else []
        if local:
            e._apply_observable_fused(lam, psi, local)
        else:
            lam.fill(0)
        for coeff, paulis in terms:
            xl = zl = xg = zg = ny = 0
            for q, pch in paulis.items():
                b = loc[q]
                if pch in ('X', 'Y'):
                    if b < m:
                        xl |= 1 << b
                    else:
                        xg |= 1 << (b - m)
                if pch in ('Z', 'Y'):
                    if b < m:
                        zl |= 1 << b
                    else:
                        zg |= 1 << (b - m)
                ny += pch == 'Y'
            src = r ^ xg
            c = coeff * (1j) ** ny * (-1.0 if bin(src & zg).count('1') & 1 else 1.0)
            if xg == 0:
                if not fuse_obs:
                    e._launch('obs_term', N, (lam, psi, np.int64(N), np.int64(xl), np.int64(zl), self.ct(c), np.int32(0)))
                continue
            xlo, xhi = xl & (C - 1), xl & ~(C - 1)
            for q0 in range(0, N, C):
                self._sync()
                t0 = time.perf_counter()
                self.comm.Sendrecv(psi[q0:q0 + C], dest=src, sendtag=2, recvbuf=scr, source=src, recvtag=2)
                self._count(src, time.perf_counter() - t0, C)
                p0 = q0 ^ xhi
                e._launch('obs_x', C, (lam[p0:p0 + C], scr, np.int64(C), np.int64(q0), np.int64(xlo), np.int64(zl), self.ct(c)))
        used = e._launch('vdot_re', N, (psi, lam, e.part, np.int64(N)))
        return self.comm.allreduce(float(e.part[:used].sum()))

    def run(self, plan: Plan, terms: Sequence[PauliTerm], backward: bool = True, fuse: bool = True, margin: float = 1.0, window=WINDOW,
            fuse_obs: bool = None, late_obs: bool = None):
        assert plan.n == self.n
        if late_obs is None:
            late_obs = self.G > 1
        if fuse_obs is None:
            fuse_obs = not late_obs
        comm, r = self.comm, self.r
        steps, counts, loc = self.compile(plan, fuse, margin, window)
        self.allocate()
        self.sent, self.tcomm = 0, 0.0
        self.sent_far, self.tcomm_far = 0, 0.0

        phases = os.environ.get('LIRA_PHASE_SYNC', '0') != '0'

        def lap(t0, last=False):
            if phases or last:
                self._sync()
                comm.Barrier()
            return time.perf_counter() - t0

        t = {}
        lap(0.0, last=True)
        t0 = time.perf_counter()
        self.psi[r].fill(0)
        if r == 0:
            self.psi[r][0] = 1.0
        self._run_steps(steps, True)
        t['forward'] = lap(t0)
        sent_f, tc_f, tf_f = self.sent, self.tcomm, self.tcomm_far

        t0 = time.perf_counter()
        late = None
        if late_obs and backward:
            late, cost = self._late(steps, terms)
            if not self._fresh:
                self.lam[r].fill(0)
        else:
            cost = self._observable(terms, loc, fuse_obs)
        t['observable'] = lap(t0, last=not backward)
        sent_o, tc_o, tf_o = self.sent - sent_f, self.tcomm - tc_f, self.tcomm_far - tf_f

        def phase_stats():
            return dict(counts, ranks=self.G, local_qubits=self.m, moved_forward=comm.allreduce(sent_f),
                        moved_observable=comm.allreduce(sent_o),
                        comm_forward=comm.allreduce(tc_f, op=MPI.MAX), comm_observable=comm.allreduce(tc_o, op=MPI.MAX),
                        comm_far_forward=comm.allreduce(tf_f, op=MPI.MAX), comm_far_observable=comm.allreduce(tf_o, op=MPI.MAX))
        stats = phase_stats() if phases or not backward else None
        if not backward:
            return cost, None, t, stats

        t0 = time.perf_counter()
        self.gr = {r: cp.zeros(plan.n_params + 1, dtype=np.float64)}
        self._slot, self._live = plan.n_params, late is None
        self._run_steps(steps, False, late)
        self._live = True
        mine = np.ascontiguousarray(self.gr[r].get())
        both = np.zeros(plan.n_params + 1)
        comm.Allreduce(mine, both, op=MPI.SUM)
        grads = both[:plan.n_params]
        if late is not None:
            cost += float(both[plan.n_params])
        t['backward'] = lap(t0, last=True)
        t['total'] = t['forward'] + t['observable'] + t['backward']
        if stats is None:
            stats = phase_stats()
        stats['moved_backward'] = comm.allreduce(self.sent - sent_f - sent_o)
        stats['comm_backward'] = comm.allreduce(self.tcomm - tc_f - tc_o, op=MPI.MAX)
        stats['comm_far'] = comm.allreduce(self.tcomm_far, op=MPI.MAX)
        stats['comm_far_backward'] = comm.allreduce(self.tcomm_far - tf_f - tf_o, op=MPI.MAX)
        stats['moved_far'] = comm.allreduce(self.sent_far)
        stats['nodes'] = len(set(comm.allgather(MPI.Get_processor_name())))
        return cost, grads, t, stats
