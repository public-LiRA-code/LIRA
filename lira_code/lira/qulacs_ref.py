import os
import time
from typing import Sequence

import numpy as np

from .engine import PauliTerm
from .planner import Plan

_FIXED1 = {'h': 'add_H_gate', 'x': 'add_X_gate', 'y': 'add_Y_gate', 'z': 'add_Z_gate', 's': 'add_S_gate',
           't': 'add_T_gate', 'sdg': 'add_Sdag_gate', 'tdg': 'add_Tdag_gate'}


def build_circuit(plan: Plan):
    from qulacs import ParametricQuantumCircuit
    c = ParametricQuantumCircuit(plan.n)
    slots = []
    for g, pidx in zip(plan.gates, plan.param_index):
        name, t = g.name.lower(), [int(x) for x in g.targets]
        th = float(g.params[0]) if g.params else 0.0
        if name in ('rx', 'ry', 'rz'):
            if pidx < 0:
                raise ValueError('parametric rotation without a parameter slot')
            getattr(c, f'add_parametric_{name.upper()}_gate')(t[0], -th)
            slots.append(int(pidx))
        elif name == 'rzz':
            c.add_parametric_multi_Pauli_rotation_gate(t, [3, 3], -th)
            slots.append(int(pidx))
        elif name in ('frx', 'fry', 'frz'):
            getattr(c, f'add_{name[1:].upper()}_gate')(t[0], -th)
        elif name in _FIXED1:
            getattr(c, _FIXED1[name])(t[0])
        elif name in ('cx', 'cnot'):
            c.add_CNOT_gate(t[0], t[1])
        elif name == 'cz':
            c.add_CZ_gate(t[0], t[1])
        elif name == 'swap':
            c.add_SWAP_gate(t[0], t[1])
        else:
            raise ValueError(f'gate not available as a Qulacs (parametric) gate: {g.name}')
    return c, slots


def build_observable(n: int, terms: Sequence[PauliTerm]):
    from qulacs import Observable
    obs = Observable(n)
    for coeff, paulis in terms:
        obs.add_operator(float(coeff), ' '.join(f'{p} {b}' for b, p in sorted(paulis.items())))
    return obs


class QulacsBaseline:
    def __init__(self, n: int):
        import qulacs
        self.n = n
        self.threads = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count()))

    def run(self, plan: Plan, terms: Sequence[PauliTerm]):
        from qulacs import QuantumState
        c, slots = build_circuit(plan)
        obs = build_observable(self.n, terms)
        t = {}

        t0 = time.perf_counter()
        state = QuantumState(self.n)
        state.set_zero_state()
        c.update_quantum_state(state)
        cost = float(np.real(obs.get_expectation_value(state)))
        t['forward'] = time.perf_counter() - t0
        del state

        t0 = time.perf_counter()
        g = np.asarray(c.backprop(obs), dtype=np.float64)
        t['total'] = time.perf_counter() - t0
        t['backward'] = max(t['total'] - t['forward'], 0.0)
        grads = np.zeros(plan.n_params)
        grads[slots] = -g
        return cost, grads, t
