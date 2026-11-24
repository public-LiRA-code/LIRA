import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'lira_code'))
from benchmarks.single_gpu import build
from lira.planner import make_plan

CASES = ['sel_4', 'sel_1', 'tfim1dp_n/2', 'khea_14', 'vqc', 'mqt_su2random', 'mqt_realamprandom']

rows = {}
for case in CASES:
    row = {}
    for n in (30, 37):
        nn, gates = build(case, n, 0)
        row[f'gates_{n}q'] = len(gates)
        if n == 30:
            row['params_30q'] = make_plan(nn, gates).n_params
    rows[case] = row
    print(f"{case:18s} gates {row['gates_30q']:6d} (30 qubits) {row['gates_37q']:6d} (37 qubits)  parameters {row['params_30q']:5d}")
if len(sys.argv) > 1:
    json.dump(rows, open(sys.argv[1], 'w'), indent=1)
