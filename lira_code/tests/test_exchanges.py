import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cupy as cp

import lira.exchanges as MG
from lira.exchanges import MultiGPUEngine
from lira.planner import make_plan
from lira.reference import dense_adjoint, parameter_shift
from test_lira import cases, hamiltonian, relabel


def main():
    ngpu = cp.cuda.runtime.getDeviceCount()
    rng = np.random.default_rng(0)
    worst, ok = 0.0, True
    MG.STREAM_MIN, MG.STREAM_MIN_OPS = 1, 0
    streamed = [0, 0]
    print(f'{ngpu} GPU(s) visible')
    print(f"{'circuit':<20}{'n':>3}{'G':>6}{'P':>5} |" + ''.join(f'{c:>12}' for c in
          ('2r lira', '2r full', '4r lira', '4r full', '8r lira', '8r nofuse', '2r lira w4', '4r lira w3', '4r full w4', '8r lira w4', '4r w4 obs', '8r w4 obs', '2r early', '4r early', '8r early', '4r late', '8r late w4', '4r s1', '4r s2 w4', '8r s1 w4', '2r all', '4r all', '8r all'))
          + ' | exchanges (two-way, one-way) for 4r lira / 4r full / 4r full w4')
    for name, n, gates in cases():
        terms = hamiltonian(n, rng)
        ps = parameter_shift(n, gates, terms)
        cost0, _ = dense_adjoint(n, gates, terms)
        plan = make_plan(n, gates)
        full = make_plan(n, gates, order=plan.order, full_width=True, keep_order=True)
        t = relabel(terms, plan)
        errs, info = [], {}
        for ranks, pl, fz, win, tag in ((2, plan, True, 0, 'a2'), (2, full, True, 0, 'f2'), (4, plan, True, 0, 'a4'), (4, full, True, 0, 'f4'),
                                        (8, plan, True, 0, 'a8'), (8, plan, False, 0, 'n8'), (2, plan, True, 4, 'w2'), (4, plan, True, 3, 'w4'),
                                        (4, full, True, 4, 'wf4'), (8, plan, True, 4, 'w8'), (4, plan, True, 4, 'o4'), (8, full, True, 4, 'o8'),
                                        (2, plan, True, 4, 'e2'), (4, plan, True, 4, 'e4'), (8, plan, True, 0, 'e8'),
                                        (4, plan, True, 0, 'l4'), (8, full, True, 4, 'l8'), (4, full, True, 0, 's4:1'), (4, plan, True, 4, 's4:2'),
                                        (8, full, True, 4, 's8:1'), (2, plan, True, 4, 'x2:2'), (4, plan, True, 4, 'x4:1'), (8, full, True, 3, 'x8:1')):
            if n - (ranks.bit_length() - 1) < 3:
                errs.append(float('nan'))
                continue
            eng = MultiGPUEngine(n, [r % ngpu for r in range(ranks)], dtype=np.complex128, chunk_bits=2, early=3 if tag[0] in 'ex' else 0,
                                 stream=int(tag[3:]) if tag[0] in 'sx' else 0)
            if tag[0] in 'ex':
                eng._force_early = True
            c, g, _, st = eng.run(pl, t, fuse=fz, margin=0.0, window=win, fuse_obs=tag[0] in 'oe', late_obs=tag[0] in 'lsx')
            streamed[0] += st.get('stream', 0)
            streamed[1] += st.get('stream_ops', 0)
            eng.release()
            errs.append(max(np.abs(g - ps).max(), abs(c - cost0)))
            info[tag] = st
        good = [e for e in errs if e == e]
        bad = max(good) > 1e-9
        ok &= not bad
        worst = max(worst, *good)
        ex = ' / '.join(f"{info[k]['exchanges']}, {info[k]['oneway']}" for k in ('a4', 'f4', 'wf4') if k in info)
        print(f"{name:<20}{n:>3}{len(gates):>6}{plan.n_params:>5} |" + ''.join(f'{e:>12.1e}' for e in errs) + f' | {ex}'
              + ('   <-- FAIL' if bad else ''))
    print(f"\nstreamed exchanges {streamed[0]}, ops run group by group {streamed[1]}")
    print(f"max |grad - parameter_shift| = {worst:.2e}   ->  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
