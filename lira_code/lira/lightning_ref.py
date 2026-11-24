import time
from typing import Sequence

import numpy as np
import pennylane as qml

from .engine import PauliTerm
from .planner import Plan

_FIXED = {'h': qml.Hadamard, 'x': qml.PauliX, 'y': qml.PauliY, 'z': qml.PauliZ, 's': qml.S, 't': qml.T,
          'sx': qml.SX, 'cx': qml.CNOT, 'cnot': qml.CNOT, 'cz': qml.CZ, 'swap': qml.SWAP,
          'cy': qml.CY, 'ch': qml.CH, 'ccx': qml.Toffoli}
_ROT = {'rx': qml.RX, 'ry': qml.RY, 'rz': qml.RZ, 'frx': qml.RX, 'fry': qml.RY, 'frz': qml.RZ,
        'p': qml.PhaseShift, 'u1': qml.PhaseShift, 'cp': qml.ControlledPhaseShift, 'rzz': qml.IsingZZ,
        'crx': qml.CRX, 'cry': qml.CRY, 'rxx': qml.IsingXX, 'ryy': qml.IsingYY}
_PAULI = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}


def build_tape(plan: Plan, terms: Sequence[PauliTerm]):
    n = plan.n
    w = lambda b: n - 1 - int(b)
    ops, trainable, slots, npar = [], [], [], 0
    for g, pidx in zip(plan.gates, plan.param_index):
        name = g.name.lower()
        wires = [w(t) for t in g.targets]
        if name in _ROT:
            ops.append(_ROT[name](g.params[0] if g.params else 0.0, wires=wires))
            if pidx >= 0:
                trainable.append(npar)
                slots.append(int(pidx))
            npar += 1
        elif name == 'sdg':
            ops.append(qml.adjoint(qml.S(wires=wires)))
        elif name == 'tdg':
            ops.append(qml.adjoint(qml.T(wires=wires)))
        else:
            ops.append(_FIXED[name](wires=wires))
    coeffs = [c for c, _ in terms]
    def word(ps):
        if not ps:
            return qml.Identity(0)
        if len(ps) == 1:
            return _PAULI[next(iter(ps.values()))](w(next(iter(ps))))
        return qml.prod(*[_PAULI[p](w(b)) for b, p in sorted(ps.items())])
    obs = [word(ps) for _, ps in terms]
    meas = qml.expval(obs[0] if (len(obs) == 1 and coeffs[0] == 1.0) else qml.Hamiltonian(coeffs, obs))
    tape = qml.tape.QuantumScript(ops, [meas], trainable_params=trainable)
    return tape, slots


def tape_grads(plan: Plan, slots, jac) -> np.ndarray:
    scale = [g.params[1] if len(g.params) == 3 else 1.0
             for g, p in zip(plan.gates, plan.param_index) if g.name.lower() in _ROT and p >= 0]
    grads = np.zeros(plan.n_params)
    np.add.at(grads, np.asarray(slots, dtype=np.int64), np.asarray(scale) * np.asarray(jac, dtype=float).reshape(-1))
    return grads


class LightningBaseline:
    def __init__(self, n: int, dtype=np.complex64, device: str = 'lightning.gpu'):
        self.n = n
        self.dev = qml.device(device, wires=n, c_dtype=dtype)

    def run(self, plan: Plan, terms: Sequence[PauliTerm], backward: bool = True):
        dev = self.dev
        tape, slots = build_tape(plan, terms)
        if getattr(dev, '_statevector', None) is None:
            dev._statevector = dev.LightningStateVector(num_wires=self.n, dtype=dev.c_dtype)
        state = dev._statevector
        t = {}

        t0 = time.perf_counter()
        state.reset_state()
        final = state.get_final_state(tape)
        cost = float(np.squeeze(dev.LightningMeasurements(final).measure_final_state(tape)))
        t['forward'] = time.perf_counter() - t0
        if not backward:
            return cost, None, t

        t0 = time.perf_counter()
        jac = np.asarray(dev.LightningAdjointJacobian(final, batch_obs=False).calculate_jacobian(tape))
        t['backward'] = time.perf_counter() - t0
        t['total'] = t['forward'] + t['backward']
        return cost, tape_grads(plan, slots, jac), t
