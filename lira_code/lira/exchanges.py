import time
from typing import Dict, List, Sequence

import cupy as cp
import numpy as np

from lira.gates import Gate
from .engine import LiraEngine, PauliTerm, late_schedule, lower
from .fusion import DIAG_KINDS, Op, diag_tables, fuse
from .windows import config, windows
from .planner import Plan


STREAM_MIN = 12
STREAM_MIN_OPS = 8


def compile_plan(plan: Plan, g: int, fuse_ops: bool = True, margin: float = 1.0, actions: bool = True, only_rank: int = None,
                 window=0, joint: bool = False, early: int = 0, stream: int = 0, intra: int = 0, keep: Dict = None, _plain=frozenset()):
    n = plan.n
    m, G = n - g, 1 << g
    if fuse_ops:
        ops = fuse(plan, margin=margin)
    else:
        ops = [Op('gate', int(k), [g], [int(p)]) for g, k, p in zip(plan.gates, plan.widths, plan.param_index)]

    def needs(op):
        if op.kind == 'run':
            return [op.gates[0].targets[0]]
        if op.kind == 'diag':
            return []
        g = op.gates[0]
        kind = lower(g)[0]
        if kind == '1q':
            return [g.targets[0]]
        if kind in ('cx', 'c1q'):
            return [g.targets[1]]
        if kind == 'ccx':
            return [g.targets[2]]
        if kind in ('swap', 'pp2'):
            return list(g.targets)
        return []

    req = [needs(op) for op in ops]
    uses = [[] for _ in range(n)]
    for i, qs in enumerate(req):
        for q in qs:
            uses[q].append(i)
    ptr = [0] * n

    def next_use(q, i):
        while ptr[q] < len(uses[q]) and uses[q][ptr[q]] <= i:
            ptr[q] += 1
        return uses[q][ptr[q]] if ptr[q] < len(uses[q]) else 1 << 60

    spread_at, spread_victims = -1, []
    widths = [o.width for o in ops]
    if early > 0 and g > 0 and widths and max(widths) > m:
        import bisect
        wmax = max(widths)
        t_end = widths.index(wmax)
        t_full = next(i for i, w in enumerate(widths) if w > m)
        after = lambda q, i: (uses[q][bisect.bisect_left(uses[q], i)] if bisect.bisect_left(uses[q], i) < len(uses[q]) else 1 << 60)
        best = None
        for i in range(1, t_full):
            ka = widths[i - 1]
            if ka < max(g + 1, m - early):
                continue
            free = sorted((q for q in range(ka) if after(q, i) >= t_end), key=lambda q: -after(q, i))
            if best is None or len(free) > len(best[1]):
                best = (i, free[:g])
            if len(free) >= g:
                break
        if best and best[1]:
            spread_at, spread_victims = best

    loc = list(range(n))
    at = list(range(n))
    steps, kprev = [], 0
    counts = dict(exchanges=0, oneway=0, lswaps=0, bits=0, spread=0, stream=0)
    loc_t = tuple(loc)
    for i, op in enumerate(ops):
        k = op.width
        if i == spread_at:
            ka, e = kprev, len(spread_victims)
            for idx, v in enumerate(spread_victims):
                target = ka - 1 - idx
                if loc[v] != target:
                    a, w = loc[v], at[target]
                    steps.append(('lswap', [0], a, target, ka))
                    loc[v], loc[w], at[target], at[a] = target, a, v, w
                    counts['lswaps'] += 1
            J = list(range(e))
            steps.append(('spread', J, ka))
            counts['spread'] = e
            for idx, v in enumerate(spread_victims):
                loc[v] = m + J[idx]
            for q in range(ka, m + e):
                loc[q] = q - e
            for q in range(n):
                at[loc[q]] = q
            loc_t = tuple(loc)
        hot_bits = sum(1 for q in range(k) if loc[q] >= m)
        if k <= m and hot_bits == 0:
            steps.append(('single', op, frozenset(op.qubits)))
            kprev = k
            continue
        wl = k - hot_bits
        cold_before = sum(1 << (loc[q] - m) for q in range(kprev, n) if loc[q] >= m)
        ranks_before = [r for r in range(G) if r & cold_before == 0]
        need = [q for q in req[i] if loc[q] >= m]
        hot = [q for q in need if q < kprev]
        active_local = lambda: (v for v in range(k) if loc[v] < m and v not in req[i])
        if hot and joint:
            cands = sorted(active_local(), key=lambda v: -next_use(v, i))
            join = list(hot)
            for c in sorted((v for v in range(kprev) if loc[v] >= m and v not in req[i]), key=lambda v: next_use(v, i)):
                if len(join) < len(cands) and next_use(c, i) < next_use(cands[len(join)], i):
                    join.append(c)
                else:
                    break
            victims = cands[:len(join)]
            if intra:
                join.sort(key=lambda q: loc[q] - m < intra)
            js = stream if stream and wl - len(join) - stream >= STREAM_MIN and counts['exchanges'] not in _plain else 0
            top = wl - js
            for idx, v in enumerate(victims):
                target = top - 1 - idx
                if loc[v] != target:
                    a, w = loc[v], at[target]
                    steps.append(('lswap', ranks_before, a, target, wl, (v, w)))
                    loc[v], loc[w], at[target], at[a] = target, a, v, w
                    counts['lswaps'] += 1
            J = [loc[q] - m for q in join]
            counts['exchanges'] += 1
            counts['bits'] += len(J)
            for idx, (q, v) in enumerate(zip(join, victims)):
                gj = loc[q] - m
                loc[v], at[m + gj] = m + gj, v
                loc[q], at[top - 1 - idx] = top - 1 - idx, q
            loc_t = tuple(loc)
            if js:
                steps.append(('jxchg', list(ranks_before), J, wl, js, loc_t, counts['exchanges'] - 1))
                counts['stream'] += 1
            elif len(J) == 1:
                steps.append(('xchg', [(r, r | (1 << J[0])) for r in ranks_before if not (r >> J[0]) & 1], False, wl))
            else:
                steps.append(('jxchg', list(ranks_before), J, wl))
        elif hot:
            for q in hot:
                gj = loc[q] - m
                victim = max(active_local(), key=lambda v: next_use(v, i))
                if loc[victim] != wl - 1:
                    a, w = loc[victim], at[wl - 1]
                    steps.append(('lswap', ranks_before, a, wl - 1, wl))
                    loc[victim], loc[w], at[wl - 1], at[a] = wl - 1, a, victim, w
                    counts['lswaps'] += 1
                steps.append(('xchg', [(r, r | (1 << gj)) for r in ranks_before if not (r >> gj) & 1], False, wl))
                counts['exchanges'] += 1
                counts['bits'] += 1
                loc[victim], at[m + gj] = m + gj, victim
                loc[q], at[wl - 1] = wl - 1, q
            loc_t = tuple(loc)
        for q in need:
            if q < kprev:
                continue
            gj = loc[q] - m
            assert wl == m, 'a rank bit is opened the old way only when the local buffer is full'
            cands = [v for v in range(n) if loc[v] < m and v not in req[i]]
            victim = max(cands, key=lambda v: next_use(v, i))
            if loc[victim] != m - 1:
                a, w = loc[victim], at[m - 1]
                steps.append(('lswap', ranks_before, a, m - 1, m))
                loc[victim], loc[w], at[m - 1], at[a] = m - 1, a, victim, w
                counts['lswaps'] += 1
            pairs = [(r, r | (1 << gj)) for r in ranks_before if not (r >> gj) & 1]
            steps.append(('xchg', pairs, True, m))
            counts['oneway'] += 1
            loc[victim], at[m + gj] = m + gj, victim
            loc[q], at[m - 1] = m - 1, q
            ranks_before = sorted(set(ranks_before) | {b for _, b in pairs})
            loc_t = tuple(loc)
        cold = sum(1 << (loc[q] - m) for q in range(k, n) if loc[q] >= m)
        ranks = [r for r in range(G) if r & cold == 0]
        wl = k - sum(1 for q in range(k) if loc[q] >= m)
        steps.append(('mop', ranks, op, wl, loc_t))
        kprev = max(kprev, k)
    if counts['stream']:
        steps, counts['stream_ops'], few = _stream_split(steps, needs)
        if few:
            return compile_plan(plan, g, fuse_ops, margin, actions, only_rank, window, joint, early, stream, intra, keep, _plain | few)
    if keep is not None:
        keep.update(raw=steps, ops=ops, m=m, only_rank=only_rank, window=window if fuse_ops else 0)
    return _finish(steps, m, actions, only_rank, window if fuse_ops else 0), counts, list(loc)


def _finish(steps, m: int, actions: bool, only_rank: int, window):
    steps = _materialize(steps, m, actions, only_rank)
    if window and actions:
        steps = _window_steps(steps, config(window))
    return steps


def set_angles(ops: List[Op], theta) -> None:
    from .decompose import is_tagged
    for op in ops:
        hit = False
        for j, (g, p) in enumerate(zip(op.gates, op.pidx)):
            if p < 0:
                continue
            if is_tagged(g):
                op.gates[j] = Gate(g.name, g.targets, (float(g.params[1]) * float(theta[p]), g.params[1], g.params[2]))
            else:
                op.gates[j] = Gate(g.name, g.targets, (float(theta[p]),) + tuple(g.params[1:]))
            hit = True
        if hit:
            op.cache.clear()
            if op.kind == 'diag':
                op.cache['host'] = diag_tables(op)


def rebind_steps(keep: Dict, theta, actions: bool = True):
    set_angles(keep['ops'], theta)
    return _finish(keep['raw'], keep['m'], actions, keep['only_rank'], keep['window'])


def _stream_split(steps, needs):
    n = len(steps)
    is_sx = lambda st: st[0] == 'jxchg' and len(st) > 4
    pre, post, gone = {}, {}, set()
    for i, st in enumerate(steps):
        if not is_sx(st):
            continue
        ranks, wl, w = st[1], st[3], st[3] - st[4]
        mine = lambda t: steps[t][0] == 'mop' and steps[t][1] == ranks and steps[t][3] == wl
        swap = lambda t: steps[t][0] == 'lswap' and len(steps[t]) > 5 and steps[t][1] == ranks and steps[t][4] == wl
        twin = lambda t: 0 <= t < n and is_sx(steps[t]) and steps[t][1] == ranks and steps[t][3] == wl
        j = i + 1
        while j < n and mine(j):
            j += 1
        seg, e = list(range(i + 1, j)), j
        while e < n and swap(e):
            e += 1
        if twin(e):
            seg = seg[:(len(seg) + 1) // 2]
        B, post[i] = set(), []
        for t in seg:
            op, loc = steps[t][2], steps[t][4]
            if all(loc[q] < w for q in needs(op)) and not (B & set(op.qubits)):
                post[i].append(t)
            else:
                B |= set(op.qubits)
        j = i - 1
        while j >= 0 and swap(j):
            j -= 1
        e = j
        while e >= 0 and mine(e):
            e -= 1
        seg = list(range(j, e, -1))
        if twin(e):
            seg = seg[:len(seg) // 2]
        B, pre[i] = set(), []
        for t in list(range(i - 1, j, -1)) + seg:
            s2 = steps[t]
            if s2[0] == 'lswap':
                qs, ok = set(s2[5]), s2[2] < w and s2[3] < w
            else:
                qs, ok = set(s2[2].qubits), all(s2[4][q] < w for q in needs(s2[2]))
            if ok and not (B & qs):
                pre[i].append(t)
            else:
                B |= qs
        pre[i].reverse()
        gone.update(pre[i], post[i])
    out = []
    for i, st in enumerate(steps):
        if i in gone:
            continue
        out.append(('sx', st[1], st[2], st[3], st[4], [steps[t] for t in pre[i]], [steps[t] for t in post[i]], st[5]) if i in pre else st)
    few = frozenset(steps[i][6] for i in pre if sum(1 for t in pre[i] + post[i] if steps[t][0] == 'mop') < STREAM_MIN_OPS)
    return out, sum(1 for t in gone if steps[t][0] == 'mop'), few


def _group_actions(item, r: int, M: int, m: int, w: int, js: int, wl: int, memo: Dict):
    if item[0] == 'lswap':
        return [('lswap', item[2], item[3])]
    op, loc = item[2], item[4]
    vloc = [l if l < wl else w + js + (l - m) if l >= m else 1 << 20 for l in loc]
    return _actions(op, vloc, M | (r << js), w, memo, w)


def _materialize(steps, m: int, actions: bool, only_rank: int):
    out = []
    for st in steps:
        if st[0] == 'mop':
            _, ranks, op, wl, loc = st
            memo = {}
            out.append(('multi', ranks, {r: _actions(op, loc, r, m, memo, wl) for r in ranks if only_rank is None or r == only_rank}
                        if actions else None, wl, frozenset(op.qubits), loc))
        elif st[0] == 'sx':
            _, ranks, J, wl, js, pre, post, loc = st
            touch = [frozenset(q for it in items for q in (it[5] if it[0] == 'lswap' else it[2].qubits)) for items in (pre, post)]
            before, at = list(loc), {b: q for q, b in enumerate(loc)}
            for idx, gj in enumerate(J):
                a, b = wl - js - 1 - idx, m + gj
                before[at[a]], before[at[b]] = b, a
            built = []
            for items in (pre, post):
                memos = [{} for _ in items]
                built.append({r: [[a for it, mm in zip(items, memos) for a in _group_actions(it, r, M, m, wl - js, js, wl, mm)]
                                  for M in range(1 << js)] for r in ranks if only_rank is None or r == only_rank} if actions else None)
            out.append(('sx', ranks, J, wl, js, built[0], built[1], touch[0], touch[1], tuple(before), loc))
        else:
            out.append(st)
    return out


def plan_cost(steps) -> float:
    v = 0.0
    for st in steps:
        if st[0] == 'xchg':
            v += 3.0 * 2.0 ** (st[3] - 1)
        elif st[0] in ('jxchg', 'sx'):
            v += 3.0 * ((1 << len(st[2])) - 1) * 2.0 ** (st[3] - len(st[2]))
        elif st[0] == 'spread':
            v += 3.0 * ((1 << len(st[1])) - 1) * 2.0 ** (st[2] - len(st[1]))
    return v


def _window_steps(steps, cfg):
    def regroup(acts, width):
        qs = lambda a: set() if a[0] == 'scal' else {a[1], a[2]} if a[0] == 'lswap' else {a[1]}
        ops = [a[1] if a[0] == 'op' else Op('act', width, [], [], dict(act=a, qs=qs(a))) for a in acts]
        return [o.cache['act'] if o.kind == 'act' else ('op', o) for o in windows(ops, *cfg)]

    out, i = [], 0
    while i < len(steps):
        st, j = steps[i], i + 1
        if st[0] == 'single':
            while j < len(steps) and steps[j][0] == 'single':
                j += 1
            out += [('single', op, frozenset(op.qubits)) for op in windows([s[1] for s in steps[i:j]], *cfg)]
        elif st[0] == 'multi':
            while j < len(steps) and steps[j][0] == 'multi' and steps[j][1] == st[1] and steps[j][3] == st[3]:
                j += 1
            touch = frozenset().union(*(s[4] for s in steps[i:j]))
            out.append(('multi', st[1], {r: regroup([a for s in steps[i:j] for a in s[2][r]], st[3]) for r in st[2]}, st[3], touch, st[5]))
        elif st[0] == 'sx':
            w = st[3] - st[4]
            out.append(st[:5] + tuple({r: [regroup(acts, w) for acts in d[r]] for r in d} for d in st[5:7]) + st[7:])
        else:
            out.append(st)
        i = j
    return out


def _actions(op: Op, loc: List[int], r: int, m: int, memo: Dict, wl: int = None):
    bitval = lambda q: (r >> (loc[q] - m)) & 1
    acts = []
    wl = m if wl is None else wl

    def gate_actions(g: Gate, pidx: int):
        kind, pl = lower(g)
        t = g.targets
        glob = [q for q in t if loc[q] >= m]
        if not glob:
            return [('op', Op('gate', wl, [Gate(g.name, tuple(loc[q] for q in t), g.params)], [pidx]))]
        if kind == 'diag1':
            v = bitval(t[0])
            return [('scal', pl[v], pl[2 + v] or 0.0, pidx)]
        if kind == 'phase1':
            return [('scal', pl[0], pl[1] or 0.0, pidx)] if bitval(t[0]) else []
        if kind == 'cphase':
            if len(glob) == 2:
                return [('scal', pl[0], pl[1] or 0.0, pidx)] if (bitval(t[0]) and bitval(t[1])) else []
            other = t[1] if glob[0] == t[0] else t[0]
            return [('phase1', loc[other], pl[0], pl[1] or 0.0, pidx)] if bitval(glob[0]) else []
        if kind == 'diag2':
            de, dq, ge, gq = pl
            if len(glob) == 2:
                odd = bitval(t[0]) ^ bitval(t[1])
                return [('scal', dq if odd else de, gq if odd else ge, pidx)]
            other = t[1] if glob[0] == t[0] else t[0]
            if bitval(glob[0]):
                return [('diag1', loc[other], dq, de, gq, ge, pidx)]
            return [('diag1', loc[other], de, dq, ge, gq, pidx)]
        if kind == 'cx':
            return [('op', Op('gate', wl, [Gate('x', (loc[t[1]],), ())], [-1]))] if bitval(t[0]) else []
        if kind == 'c1q':
            one = {'crx': 'rx', 'cry': 'ry', 'cy': 'y', 'ch': 'h'}[g.name.lower()]
            return [('op', Op('gate', wl, [Gate(one, (loc[t[1]],), g.params)], [pidx]))] if bitval(t[0]) else []
        if kind == 'ccx':
            if not all(bitval(q) for q in t[:2] if loc[q] >= m):
                return []
            here = [q for q in t[:2] if loc[q] < m]
            if here:
                return [('op', Op('gate', wl, [Gate('cx', (loc[here[0]], loc[t[2]]), ())], [-1]))]
            return [('op', Op('gate', wl, [Gate('x', (loc[t[2]],), ())], [-1]))]
        raise AssertionError(f'{g.name}: a qubit that must be local is not')

    if op.kind == 'run':
        q = loc[op.gates[0].targets[0]]
        acts.append(('op', Op('run', wl, [Gate(g.name, (q,), g.params) for g in op.gates], list(op.pidx))))
    elif op.kind == 'gate':
        acts += gate_actions(op.gates[0], op.pidx[0])
    else:
        gates, pidx, mixed = [], [], False
        for g, p in zip(op.gates, op.pidx):
            if all(loc[q] < m for q in g.targets):
                gates.append(Gate(g.name, tuple(loc[q] for q in g.targets), g.params))
                pidx.append(p)
                continue
            mixed = True
            for a in gate_actions(g, p):
                if a[0] == 'scal':
                    bit, d0, d1, g0, g1 = 0, a[1], a[1], a[2], a[2]
                elif a[0] == 'phase1':
                    bit, d0, d1, g0, g1 = a[1], 1.0 + 0j, a[2], 0.0, a[3]
                else:
                    bit, d0, d1, g0, g1 = a[1], a[2], a[3], a[4], a[5]
                d0, d1 = complex(d0), complex(d1)
                gates.append(Gate('dg1', (bit,), (d0.real, d0.imag, d1.real, d1.imag, float(g0), float(g1))))
                pidx.append(p)
        if len(gates) >= 2:
            sub = Op('diag', wl, gates, pidx)
            if mixed:
                sub.cache['host'] = diag_tables(sub)
            else:
                if 'host' not in memo:
                    memo['host'] = diag_tables(sub)
                sub.cache['host'] = memo['host']
            acts.append(('op', sub))
        elif gates:
            acts.append(('op', Op('gate', wl, gates, pidx)))
    return acts


def _part_of(r: int, J) -> int:
    k = len(J)
    return sum(((r >> J[i]) & 1) << (k - 1 - i) for i in range(k))


def _rank_with(r: int, T: int, J) -> int:
    k = len(J)
    for i in range(k):
        r = (r & ~(1 << J[i])) | (((T >> (k - 1 - i)) & 1) << J[i])
    return r


def _local_terms(terms, loc: List[int], m: int, ranks):
    out = {r: [] for r in ranks}
    for coeff, paulis in terms:
        if any(p in ('X', 'Y') and loc[q] >= m for q, p in paulis.items()):
            continue
        zg = sum(1 << (loc[q] - m) for q, p in paulis.items() if loc[q] >= m)
        here = {loc[q]: p for q, p in paulis.items() if loc[q] < m}
        for r in out:
            out[r].append((-coeff if bin(r & zg).count('1') & 1 else coeff, here))
    return out


def late_points(steps, m: int):
    units = []
    for i, st in enumerate(steps):
        if st[0] == 'single':
            k = st[1].width
            units.append((i, st[2], k, ([0], k, None), 0))
        elif st[0] == 'multi':
            units.append((i, st[4], st[3] + len(st[1]).bit_length() - 1, (st[1], st[3], st[5]), 0))
        elif st[0] == 'sx':
            k = st[3] + len(st[1]).bit_length() - 1
            units.append((('mid', i), st[7], k, (st[1], st[3], st[9]), st[4]))
            units.append((i, st[8], k, (st[1], st[3], st[10]), 0))
    return units


def _masks(paulis, loc, m: int):
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
    return xl, zl, xg, zg, ny


EXCH_SWEEPS = 20.0


def plan_time(steps, m: int) -> float:
    t = 0.0
    for st in steps:
        if st[0] == 'single':
            t += 2.0 ** (st[1].width - m)
        elif st[0] == 'multi':
            t += 2.0 ** (st[3] - m)
    return t + EXCH_SWEEPS * plan_cost(steps) / 2.0 ** m


def _choose_early(plan: Plan, g: int, fuse_ops: bool, margin: float, joint: bool, early: int) -> int:
    if not early:
        return 0
    m = plan.n - g
    t = [plan_time(compile_plan(plan, g, fuse_ops, margin, actions=False, joint=joint, early=e)[0], m) for e in (0, early)]
    return early if t[1] < t[0] else 0


def _rank_ops(steps, r: int):
    ops = []
    for st in steps:
        if st[0] == 'single' and r == 0:
            ops.append(st[1])
        elif st[0] == 'multi' and r in st[2]:
            ops += [a[1] for a in st[2][r] if a[0] == 'op']
        elif st[0] == 'sx' and r in st[5]:
            ops += [a[1] for d in st[5:7] for acts in d[r] for a in acts if a[0] == 'op']
    return ops


class MultiGPUEngine:
    def __init__(self, n: int, devices: Sequence[int], dtype=np.complex64, chunk_bits: int = 27, joint: bool = True, early: int = 0,
                 stream: int = 0, intra: int = 0):
        G = len(devices)
        assert G >= 1 and G & (G - 1) == 0, 'number of ranks must be a power of two'
        self.n, self.G, self.g = n, G, G.bit_length() - 1
        self.m = n - self.g
        assert self.m >= 2
        self.devs = list(devices)
        self.dtype = np.dtype(dtype)
        self.chunk = 1 << min(chunk_bits, self.m - 1)
        self.eng: List[LiraEngine] = []
        for d in self.devs:
            with cp.cuda.Device(d):
                self.eng.append(LiraEngine(self.m, dtype))
        self.ct = self.eng[0].ct
        self.psi = self.lam = self.scr = None
        self._compiled = {}
        self.joint, self.early, self.stream, self.intra = joint, early, stream, intra
        self._live = True
        self._fresh = False
        self._streams = {}
        self._side = {}
        for a in set(self.devs):
            for b in set(self.devs):
                if a != b:
                    try:
                        with cp.cuda.Device(a):
                            if cp.cuda.runtime.deviceCanAccessPeer(a, b):
                                cp.cuda.runtime.deviceEnablePeerAccess(b)
                    except Exception:
                        pass

    def allocate(self):
        if self.psi is None:
            self.psi, self.lam, self.scr = [], [], []
            for r, d in enumerate(self.devs):
                with cp.cuda.Device(d):
                    s, l = self.eng[r].allocate()
                    self.psi.append(s)
                    self.lam.append(l)
                    self.scr.append(cp.empty(self.chunk, dtype=self.dtype))

    def release(self):
        self.psi = self.lam = self.scr = None
        for r, d in enumerate(self.devs):
            with cp.cuda.Device(d):
                self.eng[r].release()

    def _sync(self):
        for d in set(self.devs):
            with cp.cuda.Device(d):
                cp.cuda.get_current_stream().synchronize()

    def _copies(self, jobs, wait: bool = True):
        devs = {self.devs[r] for j in jobs for r in (j[1], j[3])}
        if wait:
            for d in devs:
                with cp.cuda.Device(d):
                    cp.cuda.Device(d).synchronize()
        t0 = time.perf_counter()
        used = set()
        for dst, rd, src, rs in jobs:
            a, b = self.devs[rs], self.devs[rd]
            with cp.cuda.Device(a):
                if a == b:
                    with self._side_stream(a):
                        dst[...] = src
                    used.add((a, a))
                else:
                    if (a, b) not in self._streams:
                        self._streams[(a, b)] = cp.cuda.Stream(non_blocking=True)
                    cp.cuda.runtime.memcpyPeerAsync(dst.data.ptr, b, src.data.ptr, a, dst.nbytes, self._streams[(a, b)].ptr)
                    used.add((a, b))
            self.stats['moved'] += dst.size
        for a, b in used:
            with cp.cuda.Device(a):
                (self._side[a] if a == b else self._streams[(a, b)]).synchronize()
        if wait:
            for d in devs:
                with cp.cuda.Device(d):
                    cp.cuda.Device(d).synchronize()
        self.stats['comm'] += time.perf_counter() - t0

    def _side_stream(self, d: int):
        if d not in self._side:
            self._side[d] = cp.cuda.Stream(non_blocking=True)
        return self._side[d]

    def compile(self, plan: Plan, fuse_ops: bool = True, margin: float = 1.0, window=0):
        key = (id(plan), fuse_ops, margin, config(window))
        if key not in self._compiled:
            early = self.early if getattr(self, '_force_early', False) else _choose_early(plan, self.g, fuse_ops, margin, self.joint, self.early)
            self._compiled[key] = compile_plan(plan, self.g, fuse_ops, margin, window=window, joint=self.joint, early=early,
                                               stream=self.stream if self.joint else 0, intra=self.intra)
            if window:
                for r, d in enumerate(self.devs):
                    with cp.cuda.Device(d):
                        self.eng[r].prepare_windows(_rank_ops(self._compiled[key][0], r))
        return self._compiled[key]

    def _fwd(self, r: int, act, wl: int, s=None):
        e, N = self.eng[r], np.int64(1 << wl)
        s = self.psi[r] if s is None else s
        if act[0] == 'op':
            e._forward_op(s, act[1])
        elif act[0] == 'lswap':
            e._launch('f_swap', 1 << (wl - 2), (s, N >> 2, np.int32(act[1]), np.int32(act[2])))
        elif act[0] == 'scal':
            e._launch('f_scal', 1 << wl, (s, N, self.ct(act[1])))
        elif act[0] == 'phase1':
            e._launch('f_phase1', 1 << (wl - 1), (s, N >> 1, np.int32(act[1]), self.ct(act[2])))
        else:
            e._launch('f_diag1', 1 << wl, (s, N, np.int32(act[1]), self.ct(act[2]), self.ct(act[3])))

    def _bwd(self, r: int, act, wl: int, s=None, l=None):
        e, gr, N = self.eng[r], self.gr[r], np.int64(1 << wl)
        s, l = (self.psi[r], self.lam[r]) if s is None else (s, l)
        if act[0] == 'op':
            e._backward_op(s, l, gr, act[1])
        elif act[0] == 'lswap':
            e._launch('b_swap', 1 << (wl - 2), (s, l, N >> 2, np.int32(act[1]), np.int32(act[2])))
        elif act[0] == 'scal':
            e._launch('b_scal', 1 << wl, (s, l, gr, N, self.ct(np.conj(act[1])), np.float64(act[2]), np.int32(act[3])))
        elif act[0] == 'phase1':
            e._launch('b_phase1', 1 << (wl - 1), (s, l, gr, N >> 1, np.int32(act[1]), self.ct(np.conj(act[2])),
                                                  np.float64(act[3]), np.int32(act[4])))
        else:
            e._launch('b_diag1', 1 << wl, (s, l, gr, N, np.int32(act[1]), self.ct(np.conj(act[2])), self.ct(np.conj(act[3])),
                                           np.float64(act[4]), np.float64(act[5]), np.int32(act[6])))

    def _exchange(self, bufs, pairs, oneway: bool, forward: bool, wl: int):
        half = 1 << (wl - 1)
        if not oneway:
            gj = (pairs[0][1] - pairs[0][0]).bit_length() - 1
            return self._jexchange(bufs, [r for p in pairs for r in p], [gj], wl)
        for B in bufs:
            if forward:
                self._copies([(B[b][:half], b, B[a][half:2 * half], a) for a, b in pairs])
                for a, _ in pairs:
                    with cp.cuda.Device(self.devs[a]):
                        B[a][half:2 * half].fill(0)
            else:
                self._copies([(B[a][half:2 * half], a, B[b][:half], b) for a, b in pairs])

    def _spread(self, bufs, J, wl: int, forward: bool):
        blk = 1 << (wl - len(J))
        dest = [(T, _rank_with(0, T, J)) for T in range(1, 1 << len(J))]
        for B in bufs:
            if forward:
                self._copies([(B[r][:blk], r, B[0][T * blk:(T + 1) * blk], 0) for T, r in dest])
                with cp.cuda.Device(self.devs[0]):
                    B[0][blk:1 << wl].fill(0)
            else:
                self._copies([(B[0][T * blk:(T + 1) * blk], 0, B[r][:blk], r) for T, r in dest])

    def _jexchange(self, bufs, ranks, J, wl: int):
        k = len(J)
        blk = 1 << (wl - k)
        allat = (self.chunk >> k) >= 1
        groups = [list(range(1, 1 << k))] if allat else [[d] for d in range(1, 1 << k)]
        c = min(blk, (self.chunk >> k) if allat else self.chunk)
        for B in bufs:
            for grp in groups:
                for off in range(0, blk, c):
                    ln = min(c, blk - off)
                    jobs = []
                    for r in ranks:
                        for slot, d in enumerate(grp):
                            T = _part_of(r, J) ^ d
                            rp = _rank_with(r, T, J)
                            jobs.append((self.scr[rp][slot * c:slot * c + ln], rp, B[r][T * blk + off:T * blk + off + ln], r))
                    self._copies(jobs)
                    for r in ranks:
                        with cp.cuda.Device(self.devs[r]):
                            for slot, d in enumerate(grp):
                                T = _part_of(r, J) ^ d
                                B[r][T * blk + off:T * blk + off + ln] = self.scr[r][slot * c:slot * c + ln]

    def _group(self, r: int, acts, w: int, M: int, forward: bool):
        s = self.psi[r][M << w:(M + 1) << w]
        if forward:
            for act in acts:
                self._fwd(r, act, w, s)
        else:
            l = self.lam[r][M << w:(M + 1) << w]
            for act in reversed(acts):
                self._bwd(r, act, w, s, l)

    def _sx_work(self, st, i: int, forward: bool, late):
        w = st[3] - st[4]
        if forward:
            return (lambda r, M: self._group(r, st[5][r][M], w, M, True)), (lambda r, M: self._group(r, st[6][r][M], w, M, True))
        mid = late.get(('mid', i)) if late else None

        def after(r, M):
            if mid:
                self._inject_group(mid[0], r, M, w, st[4], st[3], st[9])
            self._group(r, st[5][r][M], w, M, False)
        return (lambda r, M: self._group(r, st[6][r][M], w, M, False)), after

    def _sexchange(self, bufs, ranks, J, wl: int, js: int, before, after):
        k, w = len(J), wl - js
        blk, ng = 1 << (w - k), 1 << js
        allat = (self.chunk >> k) >= 1
        partners = [list(range(1, 1 << k))] if allat else [[d] for d in range(1, 1 << k)]
        c = min(blk, (self.chunk >> k) if allat else self.chunk)

        def work(fn, M):
            for r in ranks:
                with cp.cuda.Device(self.devs[r]):
                    fn(r, M)

        def wait():
            for d in {self.devs[r] for r in ranks}:
                with cp.cuda.Device(d):
                    cp.cuda.Device(d).synchronize()

        work(before, 0)
        for M in range(ng):
            wait()
            if M + 1 < ng:
                work(before, M + 1)
            if M >= 1:
                work(after, M - 1)
            base = M << w
            for B, grp in ((B, grp) for B in bufs for grp in partners):
                for off in range(0, blk, c):
                    ln = min(c, blk - off)
                    jobs, back = [], []
                    for r in ranks:
                        own = _part_of(r, J)
                        for slot, d in enumerate(grp):
                            T = own ^ d
                            rp = _rank_with(r, T, J)
                            jobs.append((self.scr[rp][slot * c:slot * c + ln], rp, B[r][base + T * blk + off:base + T * blk + off + ln], r))
                            back.append((B[r][base + T * blk + off:base + T * blk + off + ln], r, self.scr[r][slot * c:slot * c + ln], r))
                    self._copies(jobs, wait=False)
                    moved = self.stats['moved']
                    self._copies(back, wait=False)
                    self.stats['moved'] = moved
        wait()
        work(after, ng - 1)

    def _inject(self, terms, ranks, wl: int, loc):
        m, slot = self.m, np.int32(self._slot)
        loc = list(range(self.n)) if loc is None else loc
        local = _local_terms(terms, loc, m, ranks)
        for r in ranks:
            if local[r]:
                with cp.cuda.Device(self.devs[r]):
                    self.eng[r]._inject(self.lam[r], self.psi[r], self.gr[r], local[r], wl, int(slot), first=self._fresh and not self._live)
        C = min(self.chunk, 1 << wl)
        for coeff, paulis in terms:
            xl, zl, xg, zg, ny = _masks(paulis, loc, m)
            if xg == 0:
                continue
            sign = lambda r: coeff * (1j) ** ny * (-1.0 if bin((r ^ xg) & zg).count('1') & 1 else 1.0)
            xlo, xhi = xl & (C - 1), xl & ~(C - 1)
            for q0 in range(0, 1 << wl, C):
                self._copies([(self.scr[r][:C], r, self.psi[r ^ xg][q0:q0 + C], r ^ xg) for r in ranks])
                for r in ranks:
                    with cp.cuda.Device(self.devs[r]):
                        p0 = q0 ^ xhi
                        self.eng[r]._launch('b_obs_x', C, (self.lam[r][p0:p0 + C], self.scr[r], self.psi[r][p0:p0 + C], self.gr[r], np.int64(C),
                                                           np.int64(q0), np.int64(xlo), np.int64(zl), self.ct(sign(r)), slot))
        self._live = True

    def _inject_group(self, terms, r: int, M: int, w: int, js: int, wl: int, loc):
        m = self.m
        vloc = [b if b < wl else w + js + (b - m) if b >= m else 1 << 20 for b in loc]
        rv = M | (r << js)
        local = _local_terms(terms, vloc, w, [rv])[rv]
        if local:
            self.eng[r]._inject(self.lam[r][M << w:(M + 1) << w], self.psi[r][M << w:(M + 1) << w], self.gr[r], local, w, self._slot)

    def _late(self, steps, terms):
        units = late_points(steps, self.m)

        def comm(u, paulis):
            (ranks, wl, loc), js = units[u][3], units[u][4]
            if loc is None:
                return False
            if js:
                return None if any(p != 'Z' and loc[q] >= wl - js for q, p in paulis.items()) else False
            return any(p != 'Z' and loc[q] >= self.m for q, p in paulis.items())

        sched, c0 = late_schedule([u[1] for u in units], [u[2] for u in units], terms, comm=comm)
        end = len(units) - 1
        self._fresh = end in sched and isinstance(units[end][0], int) and any(not comm(end, ps) for _, ps in sched[end])
        return {units[u][0]: (b,) + units[u][3] for u, b in sched.items()}, c0

    def _run_steps(self, steps, forward: bool, late=None):
        self._backward = not forward
        for i in (range(len(steps)) if forward else range(len(steps) - 1, -1, -1)):
            st = steps[i]
            if late and i in late:
                self._inject(*late[i])
            bufs = [self.psi] if forward or not self._live else [self.psi, self.lam]
            if st[0] == 'single':
                with cp.cuda.Device(self.devs[0]):
                    if forward:
                        self.eng[0]._forward_op(self.psi[0], st[1])
                    else:
                        self.eng[0]._backward_op(self.psi[0], self.lam[0], self.gr[0], st[1])
            elif st[0] == 'multi':
                for r in st[1]:
                    with cp.cuda.Device(self.devs[r]):
                        for act in (st[2][r] if forward else reversed(st[2][r])):
                            (self._fwd if forward else self._bwd)(r, act, st[3])
            elif st[0] == 'lswap':
                a, b, n4 = np.int32(st[2]), np.int32(st[3]), 1 << (st[4] - 2)
                for r in st[1]:
                    with cp.cuda.Device(self.devs[r]):
                        if forward:
                            self.eng[r]._launch('f_swap', n4, (self.psi[r], np.int64(n4), a, b))
                        else:
                            self.eng[r]._launch('b_swap', n4, (self.psi[r], self.lam[r], np.int64(n4), a, b))
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
        m, G, C = self.m, self.G, self.chunk
        N = 1 << m
        local = _local_terms(terms, loc, m, range(G)) if fuse_obs else {}
        for r in range(G):
            with cp.cuda.Device(self.devs[r]):
                if local.get(r):
                    self.eng[r]._apply_observable_fused(self.lam[r], self.psi[r], local[r])
                else:
                    self.lam[r].fill(0)
        for coeff, paulis in terms:
            xl, zl, xg, zg, ny = _masks(paulis, loc, m)
            if fuse_obs and xg == 0:
                continue
            sign = lambda r: coeff * (1j) ** ny * (-1.0 if bin((r ^ xg) & zg).count('1') & 1 else 1.0)
            if xg == 0:
                for r in range(G):
                    with cp.cuda.Device(self.devs[r]):
                        self.eng[r]._launch('obs_term', N, (self.lam[r], self.psi[r], np.int64(N), np.int64(xl), np.int64(zl),
                                                            self.ct(sign(r)), np.int32(0)))
                continue
            xlo, xhi = xl & (C - 1), xl & ~(C - 1)
            for q0 in range(0, N, C):
                self._copies([(self.scr[r], r, self.psi[r ^ xg][q0:q0 + C], r ^ xg) for r in range(G)])
                for r in range(G):
                    with cp.cuda.Device(self.devs[r]):
                        p0 = q0 ^ xhi
                        self.eng[r]._launch('obs_x', C, (self.lam[r][p0:p0 + C], self.scr[r], np.int64(C), np.int64(q0),
                                                         np.int64(xlo), np.int64(zl), self.ct(sign(r))))
        cost = 0.0
        for r in range(G):
            with cp.cuda.Device(self.devs[r]):
                e = self.eng[r]
                used = e._launch('vdot_re', N, (self.psi[r], self.lam[r], e.part, np.int64(N)))
                cost += float(e.part[:used].sum())
        return cost

    def run(self, plan: Plan, terms: Sequence[PauliTerm], backward: bool = True, fuse: bool = True, margin: float = 1.0, window=0,
            fuse_obs: bool = False, late_obs: bool = False):
        assert plan.n == self.n
        steps, counts, loc = self.compile(plan, fuse, margin, window)
        self.allocate()
        self.stats = dict(moved=0, comm=0.0)
        t = {}
        self._sync()
        t0 = time.perf_counter()
        for r in range(self.G):
            with cp.cuda.Device(self.devs[r]):
                self.psi[r].fill(0)
        with cp.cuda.Device(self.devs[0]):
            self.psi[0][0] = 1.0
        self._run_steps(steps, True)
        self._sync()
        t['forward'] = time.perf_counter() - t0
        moved_f, comm_f = self.stats['moved'], self.stats['comm']

        t0 = time.perf_counter()
        late = None
        if late_obs and backward:
            late, cost = self._late(steps, terms)
            if not self._fresh:
                for r in range(self.G):
                    with cp.cuda.Device(self.devs[r]):
                        self.lam[r].fill(0)
        else:
            cost = self._observable(terms, loc, fuse_obs)
        self._sync()
        t['observable'] = time.perf_counter() - t0
        moved_o, comm_o = self.stats['moved'] - moved_f, self.stats['comm'] - comm_f
        stats = dict(counts, ranks=self.G, local_qubits=self.m, moved_forward=moved_f, moved_observable=moved_o,
                     comm_forward=comm_f, comm_observable=comm_o)
        if not backward:
            return cost, None, t, stats

        t0 = time.perf_counter()
        self.gr, self._slot, self._live = [], plan.n_params, late is None
        for r in range(self.G):
            with cp.cuda.Device(self.devs[r]):
                self.gr.append(cp.zeros(plan.n_params + 1, dtype=np.float64))
        self._run_steps(steps, False, late)
        self._live = True
        grads = np.zeros(plan.n_params)
        for r in range(self.G):
            with cp.cuda.Device(self.devs[r]):
                out = self.gr[r].get()
                grads += out[:plan.n_params]
                if late is not None:
                    cost += float(out[plan.n_params])
        self._sync()
        t['backward'] = time.perf_counter() - t0
        t['total'] = t['forward'] + t['observable'] + t['backward']
        stats['moved_backward'] = self.stats['moved'] - moved_f - moved_o
        stats['comm_backward'] = self.stats['comm'] - comm_f - comm_o
        return cost, grads, t, stats
