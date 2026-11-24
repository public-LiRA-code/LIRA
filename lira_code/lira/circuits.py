import os
from typing import List, Tuple

import numpy as np

from lira.gates import Gate
from lira.qasm import parse_qasm


def _rng(seed):
    return np.random.default_rng(seed)


def _entangler_pairs(n: int, ent: str, layer: int) -> List[Tuple[int, int]]:
    if ent == 'linear':
        return [(i, i + 1) for i in range(n - 1)]
    if ent == 'circular':
        return [(n - 1, 0)] + [(i, i + 1) for i in range(n - 1)]
    if ent == 'full':
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if ent == 'brick':
        return [(i, i + 1) for i in range(layer % 2, n - 1, 2)]
    if ent == 'pairwise':
        return [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]
    raise ValueError(ent)


def hea(n: int, layers: int, ent: str = 'linear', rot=('ry', 'rz'), gate2: str = 'cx', seed: int = 0):
    r = _rng(seed)
    gates = []
    for layer in range(layers):
        for name in rot:
            gates += [Gate(name, (q,), (float(r.uniform(0, 2 * np.pi)),)) for q in range(n)]
        gates += [Gate(gate2, p, ()) for p in _entangler_pairs(n, ent, layer)]
    for name in rot:
        gates += [Gate(name, (q,), (float(r.uniform(0, 2 * np.pi)),)) for q in range(n)]
    return n, gates


def sel(n: int, layers: int, ranges=None, seed: int = 0):
    r = _rng(seed)
    if ranges is None:
        ranges = [l % (n - 1) + 1 for l in range(layers)] if n > 1 else [0] * layers
    gates = []
    for l in range(layers):
        for q in range(n):
            phi, theta, omega = (float(x) for x in r.uniform(0, 2 * np.pi, 3))
            gates += [Gate('rz', (q,), (phi,)), Gate('ry', (q,), (theta,)), Gate('rz', (q,), (omega,))]
        if n > 1:
            gates += [Gate('cx', (q, (q + ranges[l]) % n), ()) for q in range(n)]
    return n, gates


def _graph_edges(n: int, graph: str, seed: int) -> List[Tuple[int, int]]:
    if graph == 'line':
        return [(i, i + 1) for i in range(n - 1)]
    if graph == 'ring':
        return [(i, (i + 1) % n) for i in range(n)]
    if graph == 'complete':
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if graph == '3reg':
        import networkx as nx
        return [tuple(sorted(e)) for e in nx.random_regular_graph(3, n, seed=seed).edges()]
    if graph.startswith('grid2d:'):
        R, Cc = map(int, graph.split(':')[1].split('x'))
        e = [(r * Cc + c, r * Cc + c + 1) for r in range(R) for c in range(Cc - 1)]
        return e + [(r * Cc + c, (r + 1) * Cc + c) for r in range(R - 1) for c in range(Cc)]
    if graph == 'grid':
        w = n // 2
        e = [(i, i + 1) for i in range(w - 1)] + [(w + i, w + i + 1) for i in range(w - 1)]
        return e + [(i, w + i) for i in range(w)]
    raise ValueError(graph)


def qaoa(n: int, p: int, graph: str = '3reg', seed: int = 0):
    r = _rng(seed)
    edges = _graph_edges(n, graph, seed)
    gates = [Gate('h', (q,), ()) for q in range(n)]
    for _ in range(p):
        gamma, beta = r.uniform(0, np.pi, 2)
        gates += [Gate('rzz', e, (float(gamma),)) for e in edges]
        gates += [Gate('rx', (q,), (float(beta),)) for q in range(n)]
    return n, gates


def mps_staircase(n: int, layers: int = 1, seed: int = 0):
    r = _rng(seed)
    gates = []
    for _ in range(layers):
        for i in range(n - 1):
            for q in (i, i + 1):
                gates.append(Gate('ry', (q,), (float(r.uniform(0, 2 * np.pi)),)))
            gates.append(Gate('cx', (i, i + 1), ()))
    gates += [Gate('ry', (q,), (float(r.uniform(0, 2 * np.pi)),)) for q in range(n)]
    return n, gates


def uccsd(n: int, electrons: int, seed: int = 0):
    import warnings
    import pennylane as qml
    warnings.filterwarnings('ignore', message='(?s).*deprecated in PennyLane')
    r = _rng(seed)
    singles, doubles = qml.qchem.excitations(electrons, n)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    weights = r.uniform(-0.1, 0.1, len(singles) + len(doubles))
    op = qml.UCCSD(weights, wires=range(n), s_wires=s_wires, d_wires=d_wires,
                   init_state=qml.qchem.hf_state(electrons, n))
    tape = qml.tape.QuantumScript([op], [])
    stop = {'RX', 'RZ', 'Hadamard', 'CNOT', 'PauliX'}
    tape = tape.expand(depth=10, stop_at=lambda o: o.name in stop)
    gates = []
    for o in tape.operations:
        w = tuple(int(x) for x in o.wires)
        if o.name == 'RX':
            gates.append(Gate('frx', w, (float(o.parameters[0]),)))
        elif o.name == 'RZ':
            gates.append(Gate('rz', w, (float(o.parameters[0]),)))
        elif o.name == 'Hadamard':
            gates.append(Gate('h', w, ()))
        elif o.name == 'CNOT':
            gates.append(Gate('cx', w, ()))
        elif o.name == 'PauliX':
            gates.append(Gate('x', w, ()))
        else:
            raise ValueError(f'unexpected op {o.name}')
    return n, gates


SUPPORTED = {'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx', 'rx', 'ry', 'rz', 'p', 'u1', 'frx', 'fry',
             'frz', 'cx', 'cnot', 'cz', 'cp', 'rzz', 'swap',
             'crx', 'cry', 'cy', 'ch', 'ccx', 'rxx', 'ryy',
             'u', 'u2', 'u3', 'crz', 'id'}


def from_qasm(path: str, n_keep: int = 0):
    n, gates = parse_qasm(path)
    if n_keep > 0:
        n = n_keep
        gates = [g for g in gates if all(t < n for t in g.targets)]
    bad = sorted({g.name for g in gates if g.name.lower() not in SUPPORTED})
    if bad:
        raise ValueError(f'unsupported gates in {os.path.basename(path)}: {bad}')
    return n, gates


def kawase_hea(n: int, layers: int, seed: int = 0):
    theta = np.random.default_rng(seed).uniform(0, 2 * np.pi, (layers, n, 3))
    gates = []
    for l in range(layers):
        for q in range(n):
            for j, name in enumerate(('rx', 'ry', 'rz')):
                gates.append(Gate(name, (q,), (float(theta[l, q, j]),)))
        gates += [Gate('cz', (q, (q + 1) % n), ()) for q in range(n)]
    return n, gates


def qaoa_from_edges(n: int, edges, gammas, betas):
    gates = [Gate('h', (q,), ()) for q in range(n)]
    for gamma, beta in zip(gammas, betas):
        gates += [Gate('rzz', (int(i), int(j)), (float(-gamma * w),)) for i, j, w in edges]
        gates += [Gate('rx', (q,), (float(2 * beta),)) for q in range(n)]
    return n, gates


def tfim_terms(n: int, graph: str, h: float = 1.0, seed: int = 0):
    return ([(-1.0, {i: 'Z', j: 'Z'}) for i, j in _graph_edges(n, graph, seed)]
            + [(-h, {q: 'X'}) for q in range(n)])
