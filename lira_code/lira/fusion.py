from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from lira.gates import Gate
from .engine import TOUCH, lower
from .planner import Plan

DIAG_KINDS = {'diag1', 'phase1', 'cphase', 'diag2'}
MAX_RUN = 3
MAX_DIAG_PARAMS = 48
MAX_CLASSES = 4
CHUNK_BITS = 11

C_BASE, C_CLASS, C_TERM = 1.2, 1.0, 0.09


@dataclass
class Op:
    kind: str
    width: int
    gates: List[Gate]
    pidx: List[int]
    cache: Dict = field(default_factory=dict)

    @property
    def qubits(self):
        if self.kind == 'act':
            return self.cache['qs']
        return {q for g in self.gates for q in g.targets}


def fuse(plan: Plan, margin: float = 1.0, window=0) -> List[Op]:
    out: List[Op] = []
    runs: Dict[int, Op] = {}
    diag: List = [None]

    def emit_gate(g, k, p):
        out.append(Op('gate', k, [g], [p]))

    def close_run(q):
        op = runs.pop(q, None)
        if op is None:
            return
        if len(op.gates) == 1:
            op.kind = 'gate'
        out.append(op)

    def close_diag():
        op, diag[0] = diag[0], None
        if op is None:
            return
        if len(op.gates) > 1:
            h = diag_tables(op)
            separate = 3.0 * sum(TOUCH[lower(g)[0]] for g in op.gates)
            fused = 3.0 * C_BASE + 2.0 * (C_CLASS * h['ncls'] + C_TERM * h['nc']) + C_TERM * h['np']
            if separate > margin * fused:
                op.cache['host'] = h
                out.append(op)
                return
        for g, p in zip(op.gates, op.pidx):
            emit_gate(g, op.width, p)

    def add_diag(g, k, p):
        op = diag[0]
        if op is not None and p >= 0 and sum(x >= 0 for x in op.pidx) >= MAX_DIAG_PARAMS:
            close_diag()
            op = None
        if op is None:
            diag[0] = Op('diag', k, [g], [p])
        else:
            op.gates.append(g)
            op.pidx.append(p)

    for g, k, p in zip(plan.gates, plan.widths, plan.param_index):
        k, p = int(k), int(p)
        for q in [q for q, op in runs.items() if op.width < k]:
            close_run(q)
        if diag[0] is not None and diag[0].width < k:
            close_diag()

        kind = lower(g)[0]
        if len(g.targets) == 1:
            q = g.targets[0]
            if q in runs:
                if len(runs[q].gates) >= MAX_RUN:
                    close_run(q)
                    runs[q] = Op('run', k, [g], [p])
                else:
                    runs[q].gates.append(g)
                    runs[q].pidx.append(p)
            elif kind in DIAG_KINDS:
                add_diag(g, k, p)
            else:
                if diag[0] is not None and q in diag[0].qubits:
                    close_diag()
                runs[q] = Op('run', k, [g], [p])
            continue

        for q in g.targets:
            close_run(q)
        if kind in DIAG_KINDS:
            add_diag(g, k, p)
        else:
            if diag[0] is not None and set(g.targets) & diag[0].qubits:
                close_diag()
            emit_gate(g, k, p)

    for q in list(runs):
        close_run(q)
    close_diag()
    if window:
        from .windows import config, windows
        out = windows(out, *config(window))
    return out


def summary(ops: List[Op], n: int) -> Dict[str, float]:
    sweeps = 0.0
    for op in ops:
        if op.kind == 'window':
            if 'whost' not in op.cache:
                from .windows import window_tables
                op.cache['whost'] = window_tables(op)
            f = op.cache['whost']['touch']
        else:
            f = TOUCH[lower(op.gates[0])[0]] if op.kind == 'gate' else 1.0
        sweeps += f * 2.0 ** (op.width - n)
    return dict(ops=len(ops), gates=sum(len(o.gates) for o in ops), runs=sum(o.kind == 'run' for o in ops),
                diags=sum(o.kind == 'diag' for o in ops), windows=sum(o.kind == 'window' for o in ops), sweeps=sweeps)


def mat1q(g: Gate) -> np.ndarray:
    kind, pl = lower(g)
    if kind == '1q':
        return pl[0]
    if kind == 'diag1':
        return np.diag([pl[0], pl[1]]).astype(np.complex128)
    if kind == 'phase1':
        return np.diag([1.0, pl[0]]).astype(np.complex128)
    raise ValueError(g.name)


def gen1q(g: Gate) -> np.ndarray:
    kind, pl = lower(g)
    if kind == '1q':
        return pl[1]
    if kind == 'diag1':
        return np.diag([pl[2], pl[3]]).astype(np.complex128)
    if kind == 'phase1':
        return np.diag([0.0, pl[1]]).astype(np.complex128)
    raise ValueError(g.name)


def diag_tables(op: Op, chunk_bits: int = CHUNK_BITS):
    k = op.width
    nch = (k + chunk_bits - 1) // chunk_bits
    single = np.zeros(max(k, 1))
    pairs: Dict = {}
    a0 = 0.0
    pm, pty, g0, g1, slot = [], [], [], [], []
    two_pi = 2.0 * np.pi
    for g, p in zip(op.gates, op.pidx):
        kind, pl = lower(g)
        t = g.targets
        mask = 0
        for q in t:
            mask |= 1 << q
        if kind == 'diag1':
            ang0, ang1 = np.angle(pl[0]), np.angle(pl[1])
            a0 += ang0
            single[t[0]] += ang1 - ang0
            par = (0, pl[2], None if pl[2] is None else pl[3] - pl[2])
        elif kind == 'phase1':
            single[t[0]] += np.angle(pl[0])
            par = (1, 0.0, pl[1])
        elif kind == 'cphase':
            key = (min(t), max(t))
            pairs[key] = pairs.get(key, 0.0) + np.angle(pl[0])
            par = (1, 0.0, pl[1])
        elif kind == 'diag2':
            ang0, c1 = np.angle(pl[0]), np.angle(pl[1]) - np.angle(pl[0])
            a0 += ang0
            single[t[0]] += c1
            single[t[1]] += c1
            key = (min(t), max(t))
            pairs[key] = pairs.get(key, 0.0) - 2.0 * c1
            par = (0, pl[2], pl[3] - pl[2])
        else:
            raise ValueError(g.name)
        if p >= 0:
            pm.append(mask); pty.append(par[0]); g0.append(par[1]); g1.append(par[2]); slot.append(p)

    v = np.arange(1 << chunk_bits, dtype=np.int64)
    tab = np.zeros((nch, 1 << chunk_bits))
    for q in range(k):
        if single[q] != 0.0:
            tab[q // chunk_bits] += single[q] * ((v >> (q % chunk_bits)) & 1)

    classes: Dict = {}
    for (a, b), c in pairs.items():
        c = c - two_pi * np.round(c / two_pi)
        if abs(c) > 1e-14:
            classes.setdefault(round(c, 12), []).append((a, b, c))
    ordered = sorted(classes.values(), key=len, reverse=True)
    kept = [cl for cl in ordered[:MAX_CLASSES] if len(cl) >= 4]
    rest = [x for cl in ordered if cl not in kept for x in cl]
    act = np.zeros(MAX_CLASSES, dtype=np.int64)
    nbm = np.zeros(MAX_CLASSES * 64, dtype=np.int64)
    ccoef = np.zeros(MAX_CLASSES)
    for i, cl in enumerate(kept):
        ccoef[i] = cl[0][2]
        for a, b, _ in cl:
            act[i] |= 1 << a
            nbm[i * 64 + a] |= 1 << b
    cm, cv = [], []
    for a, b, c in rest:
        if a // chunk_bits == b // chunk_bits:
            ch = a // chunk_bits
            tab[ch] += c * (((v >> (a % chunk_bits)) & 1) & ((v >> (b % chunk_bits)) & 1))
        else:
            cm.append((1 << a) | (1 << b)); cv.append(c)
    pad = lambda x, dt: np.array(x if len(x) else [0], dtype=dt)
    return dict(tab=tab.reshape(-1), nch=nch, chb=chunk_bits, a0=float(a0),
                act=act, nbm=nbm, ccoef=ccoef, ncls=len(kept),
                cm=pad(cm, np.int64), cv=pad(cv, np.float64), nc=len(cm),
                pm=pad(pm, np.int64), pty=pad(pty, np.int32), g0=pad(g0, np.float64), g1=pad(g1, np.float64),
                slot=pad(slot, np.int32), np=len(pm))
