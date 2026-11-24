from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lira.gates import Gate
from .decompose import expand, is_tagged

PARAMETRIC = {'rx', 'ry', 'rz', 'p', 'u1', 'cp', 'rzz', 'crx', 'cry', 'rxx', 'ryy'}

_Z1 = {'rz', 'frz', 'p', 'u1', 'z', 's', 'sdg', 't', 'tdg', 'i'}
_X1 = {'rx', 'frx', 'x', 'sx', 'sxdg'}
_Y1 = {'ry', 'fry', 'y'}
_ZZ2 = {'cz', 'cp', 'rzz'}


def gate_axes(g: Gate) -> Tuple[str, ...]:
    n = g.name.lower()
    if len(g.targets) == 1:
        if n in _Z1:
            return ('Z',)
        if n in _X1:
            return ('X',)
        if n in _Y1:
            return ('Y',)
        return ('G',)
    if n in _ZZ2:
        return ('Z', 'Z')
    if n == 'rxx':
        return ('X', 'X')
    if n == 'ryy':
        return ('Y', 'Y')
    if n in ('cx', 'cnot', 'crx'):
        return ('Z', 'X')
    if n in ('cry', 'cy'):
        return ('Z', 'Y')
    if n == 'ch':
        return ('Z', 'G')
    if n in ('ccx', 'toffoli'):
        return ('Z', 'Z', 'X')
    return tuple('G' for _ in g.targets)


def is_parametric(g: Gate) -> bool:
    return g.name.lower() in PARAMETRIC and len(g.params) > 0


def param_slots(gates: Sequence[Gate]) -> Tuple[np.ndarray, int]:
    pidx = np.full(len(gates), -1, dtype=np.int64)
    if any(is_tagged(g) for g in gates):
        for i, g in enumerate(gates):
            if is_parametric(g):
                pidx[i] = int(round(g.params[2]))
        return pidx, int(pidx.max()) + 1
    k = 0
    for i, g in enumerate(gates):
        if is_parametric(g):
            pidx[i] = k
            k += 1
    return pidx, k


def ancestor_supports(n: int, gates: Sequence[Gate], commute: bool = True) -> np.ndarray:
    prev = [0] * n
    cur = [0] * n
    axis = [None] * n
    out = np.zeros(len(gates), dtype=np.uint64)
    for i, g in enumerate(gates):
        axes = gate_axes(g) if commute else tuple('G' for _ in g.targets)
        a = 0
        for q, ax in zip(g.targets, axes):
            a |= 1 << q
            if ax == 'G' or ax != axis[q]:
                prev[q], cur[q], axis[q] = cur[q], 0, ax
            a |= prev[q]
        for q in g.targets:
            cur[q] |= a
        out[i] = a
    return out


def gate_weights(gates: Sequence[Gate], mode: str = 'grad',
                 w_param_extra: float = 1.0) -> np.ndarray:
    if mode == 'forward':
        return np.ones(len(gates))
    w = np.full(len(gates), 3.0)
    w[[is_parametric(g) for g in gates]] += w_param_extra
    return w


class WidthProblem:

    def __init__(self, n: int, supports: np.ndarray, weights: np.ndarray):
        self.n = n
        self.masks, inv = np.unique(supports, return_inverse=True)
        self.w = np.bincount(inv, weights=weights, minlength=len(self.masks))
        self.member = ((self.masks[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
        self.total = float(weights.sum())

    def cost(self, order: Sequence[int]) -> float:
        rank = np.empty(self.n, dtype=np.int64)
        rank[np.asarray(order)] = np.arange(1, self.n + 1)
        pos = (self.member * rank[None, :]).max(axis=1)
        return float((self.w * np.exp2(pos - self.n)).sum() / self.total)

    def greedy_reverse(self) -> List[int]:
        alive = np.ones(len(self.masks), dtype=bool)
        remaining = list(range(self.n))
        rev = []
        while remaining:
            load = (self.member[alive][:, remaining] * self.w[alive][:, None]).sum(axis=0)
            q = remaining[int(np.argmin(load))]
            rev.append(q)
            remaining.remove(q)
            alive &= ~self.member[:, q]
        return rev[::-1]

    def local_search(self, order: Sequence[int], max_sweeps: int = 20) -> List[int]:
        best = list(order)
        best_c = self.cost(best)
        for _ in range(max_sweeps):
            improved = False
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        continue
                    cand = best.copy()
                    cand.insert(j, cand.pop(i))
                    c = self.cost(cand)
                    if c < best_c * (1 - 1e-12):
                        best, best_c, improved = cand, c, True
            if not improved:
                break
        return best

    def exact(self) -> List[int]:
        n = self.n
        size = 1 << n
        cnt = np.zeros(size)
        np.add.at(cnt, self.masks.astype(np.int64), self.w)
        for b in range(n):
            v = cnt.reshape(-1, 2, 1 << b)
            v[:, 1, :] += v[:, 0, :]
        idx = np.arange(size, dtype=np.int64)
        pc = np.zeros(size, dtype=np.int64)
        for b in range(n):
            pc += (idx >> b) & 1
        f = np.full(size, np.inf)
        f[0] = 0.0
        arg = np.full(size, -1, dtype=np.int8)
        for s in range(1, n + 1):
            layer = idx[pc == s]
            scale = float(2.0 ** (s - n))
            best = np.full(len(layer), np.inf)
            bq = np.full(len(layer), -1, dtype=np.int8)
            for q in range(n):
                has = ((layer >> q) & 1).astype(bool)
                S = layer[has]
                c = f[S ^ (1 << q)] + scale * (cnt[S] - cnt[S ^ (1 << q)])
                sub = best[has]
                better = c < sub
                sub[better] = c[better]
                best[has] = sub
                subq = bq[has]
                subq[better] = q
                bq[has] = subq
            f[layer] = best
            arg[layer] = bq
        order, S = [], size - 1
        while S:
            q = int(arg[S])
            order.append(q)
            S ^= 1 << q
        return order[::-1]

    def optimise(self, exact_max_n: int = 20) -> List[int]:
        if self.n <= exact_max_n:
            return self.exact()
        cands = [self.greedy_reverse(), list(range(self.n)), list(range(self.n))[::-1]]
        best = min(cands, key=self.cost)
        return self.local_search(best)


@dataclass
class Plan:
    n: int
    order: List[int]
    gates: List[Gate]
    widths: np.ndarray
    orig_index: np.ndarray
    param_index: np.ndarray
    r_work: Dict[str, float]
    n_params: int

    def segments(self) -> List[Tuple[int, int]]:
        ks, counts = np.unique(self.widths, return_counts=True)
        return list(zip(ks.tolist(), counts.tolist()))


def first_use_order(n: int, gates: Sequence[Gate]) -> List[int]:
    seen, order = set(), []
    for g in gates:
        for q in g.targets:
            if q not in seen:
                seen.add(q)
                order.append(q)
    return order + [q for q in range(n) if q not in seen]


def naive_widths(n: int, gates: Sequence[Gate]) -> np.ndarray:
    seen, out = set(), np.zeros(len(gates), dtype=np.int64)
    for i, g in enumerate(gates):
        seen.update(g.targets)
        out[i] = len(seen)
    return out


def r_work_from_widths(n: int, widths: np.ndarray, weights: np.ndarray) -> float:
    return float((weights * np.exp2(widths - n)).sum() / weights.sum())


def make_plan(n: int, gates: Sequence[Gate], order: Optional[Sequence[int]] = None,
              objective: str = 'grad', commute: bool = True, full_width: bool = False,
              exact_max_n: int = 20, keep_order: bool = False) -> Plan:
    gates = expand(gates)
    sup = ancestor_supports(n, gates, commute=commute)
    probs = {m: WidthProblem(n, sup, gate_weights(gates, m)) for m in ('forward', 'grad')}
    if order is None:
        order = probs[objective].optimise(exact_max_n=exact_max_n)
    order = list(order)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    member = ((sup[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
    pos = (member * rank[None, :]).max(axis=1)
    sched = np.arange(len(gates)) if (keep_order and full_width) else np.argsort(pos, kind='stable')

    pidx, k = param_slots(gates)
    new_gates = [Gate(gates[i].name.lower(), tuple(int(rank[q]) - 1 for q in gates[i].targets),
                      tuple(gates[i].params)) for i in sched]
    widths = np.full(len(gates), n, dtype=np.int64) if full_width else pos[sched]
    return Plan(n=n, order=order, gates=new_gates, widths=widths, orig_index=sched,
                param_index=pidx[sched],
                r_work={m: (1.0 if full_width else p.cost(order)) for m, p in probs.items()},
                n_params=k)
