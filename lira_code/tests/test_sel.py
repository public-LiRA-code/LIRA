import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pennylane as qml
from pennylane import numpy as pnp

from lira import circuits as C
from lira.reference import dense_adjoint

PAULI = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}


def tfim_terms(n):
    return [(1.0, {q: 'Z', (q + 1) % n: 'Z'}) for q in range(n)] + [(0.7, {q: 'X'}) for q in range(n)]


def template(n, layers, weights, terms):
    obs = [qml.prod(*[PAULI[p](q) for q, p in ps.items()]) for _, ps in terms]
    H = qml.Hamiltonian([c for c, _ in terms], obs)

    @qml.qnode(qml.device('default.qubit', wires=n))
    def cost(w):
        qml.StronglyEntanglingLayers(w, wires=range(n))
        return qml.expval(H)

    w = pnp.array(weights, requires_grad=True)
    return float(cost(w)), np.asarray(qml.grad(cost)(w)).reshape(-1)


def main():
    worst, ok = 0.0, True
    print(f"{'n':>3}{'L':>4}{'ranges':>22}{'gates':>7} | {'|cost diff|':>12}{'max|grad diff|':>16}")
    for n, layers in ((2, 3), (3, 2), (5, 4), (6, 7), (8, 2), (9, 4)):
        nn, gates = C.sel(n, layers, seed=n)
        weights = np.array([g.params[0] for g in gates if g.name in ('rz', 'ry')]).reshape(layers, n, 3)
        terms = tfim_terms(n)
        c_lira, g_lira = dense_adjoint(n, gates, terms)
        c_ref, g_ref = template(n, layers, weights, terms)
        dc, dg = abs(c_lira - c_ref), float(np.abs(g_lira - g_ref).max())
        bad = dc > 1e-10 or dg > 1e-10
        ok &= not bad
        worst = max(worst, dc, dg)
        ranges = [l % (n - 1) + 1 for l in range(layers)]
        print(f"{n:>3}{layers:>4}{str(ranges):>22}{len(gates):>7} | {dc:>12.1e}{dg:>16.1e}" + ('   <-- FAIL' if bad else ''))
    print(f"\nworst difference {worst:.1e}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
