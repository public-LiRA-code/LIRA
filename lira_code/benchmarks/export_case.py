import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lira.decompose import expand
from lira.planner import is_parametric
from lira import paths
from benchmarks.single_gpu import NATIVE, build


def export(case: str, n: int, seed: int = 0, obs: str = 'meanz') -> dict:
    nn, gates = build(case, n, seed)
    gates = expand(gates)
    out, k = [], 0
    for g in gates:
        p = -1
        if is_parametric(g):
            p, k = k, k + 1
        out.append(dict(name=g.name.lower(), q=[int(x) for x in g.targets],
                        theta=float(g.params[0]) if g.params else None, p=p))
    d = dict(case=case.replace('/', '-'), n=nn, seed=seed, n_params=k, gates=out, obs=obs)
    if obs == 'native':
        d['terms'] = [[float(c), {str(q): pl for q, pl in ps.items()}] for c, ps in NATIVE[case]]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='12')
    ap.add_argument('--cases', default='khea_4,hea_linear_4,qaoa_3reg_4')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--obs', choices=['meanz', 'native'], default='meanz',
                    help='meanz: mean over qubits of <Z> (the engines build it themselves); native: the Hamiltonian of the case, written as "terms"')
    ap.add_argument('--out', default=paths.CASES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for n in map(int, args.n.split(',')):
        for case in args.cases.split(','):
            d = export(case, n, args.seed, args.obs)
            path = os.path.join(args.out, f"{case.replace('/', '-')}_n{d['n']}.json")
            with open(path, 'w') as f:
                json.dump(d, f)
            print(f"{path}: {len(d['gates'])} gates, {d['n_params']} parameters")


if __name__ == '__main__':
    main()
