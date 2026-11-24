from typing import List, Sequence

import numpy as np

from lira.gates import Gate

REWRITE = {'u3', 'u', 'u2', 'crz', 'id'}


def is_tagged(g: Gate) -> bool:
    return len(g.params) == 3 and g.name.lower() not in REWRITE and g.name != 'dg1'


def expand(gates: Sequence[Gate]) -> List[Gate]:
    from .planner import is_parametric
    gates = list(gates)
    if not any(g.name.lower() in REWRITE for g in gates):
        return gates
    out: List[Gate] = []
    slot = 0

    def par(name, targets, angle, scale, s):
        return Gate(name, tuple(targets), (float(angle), float(scale), float(s)))

    for g in gates:
        n, t, p = g.name.lower(), tuple(g.targets), g.params
        if n in ('u3', 'u'):
            th, ph, lam = p
            out += [par('rz', t, lam, 1, slot + 2), par('ry', t, th, 1, slot), par('rz', t, ph, 1, slot + 1)]
            slot += 3
        elif n == 'u2':
            ph, lam = p
            out += [par('rz', t, lam, 1, slot + 1), Gate('fry', t, (np.pi / 2,)), par('rz', t, ph, 1, slot)]
            slot += 2
        elif n == 'crz':
            out += [par('p', t[:1], -0.5 * p[0], -0.5, slot), par('cp', t, p[0], 1, slot)]
            slot += 1
        elif n == 'id':
            continue
        elif is_tagged(g):
            out.append(g)
            slot = max(slot, int(round(p[2])) + 1)
        elif is_parametric(g):
            out.append(par(n, t, p[0], 1, slot))
            slot += 1
        else:
            out.append(g)
    return out
