import glob
import json
import os
import sys

import numpy as np

results = sys.argv[1]
grads = os.path.join(results, 'grads')
out = {'single_gpu_c128': {}, 'distributed_c64_8gpus': {}}

for f in sorted(glob.glob(os.path.join(grads, 'lkk_c128_*.npy'))):
    tail = os.path.basename(f)[len('lkk_c128'):]
    g = os.path.join(grads, 'lira_c128' + tail)
    if not os.path.exists(g):
        continue
    case = tail[1:-4]
    a, b = np.load(f)[1:], np.load(g)[1:]
    row = dict(params=int(b.size), vs_lightning_kokkos=float(np.nanmax(np.abs(a - b))), largest_derivative=float(np.nanmax(np.abs(b))))
    one = glob.glob(os.path.join(results, f"onegpu_{case.rsplit('_n', 1)[0]}_*_g1_n28_c128.json"))
    if one:
        r = json.load(open(one[0]))['rows'][0]
        row['vs_lightning_gpu'] = r.get('full_maxdiff')
        row['full_width_vs_lightning_gpu'] = r.get('full-width-gate-by-gate_maxdiff')
    worst = max(v for v in (row['vs_lightning_kokkos'], row.get('vs_lightning_gpu')) if v is not None)
    row['relative'] = worst / row['largest_derivative']
    out['single_gpu_c128'][case] = row
    print(f"{case:24s} params {row['params']:5d}  vs lightning.gpu {row.get('vs_lightning_gpu', float('nan')):.1e}  "
          f"vs lightning.kokkos {row['vs_lightning_kokkos']:.1e}  largest {row['largest_derivative']:.2g}")

for case in ('sel_4', 'tfim1dp_n-2'):
    ref_f, kk_f = os.path.join(grads, f'lira_c128_{case}_n28.npy'), os.path.join(grads, f'lkk_c128_{case}_n28.npy')
    if not (os.path.exists(ref_f) and os.path.exists(kk_f)):
        continue
    ref_all = np.load(ref_f)
    ref, kk = ref_all[1:], np.load(kk_f)[1:]
    big = float(np.abs(ref).max())
    for v in ('full', 'no-early-distribution'):
        f = os.path.join(grads, f'{v}_g8_{case}_n28.npy')
        if not os.path.exists(f):
            continue
        g_all = np.load(f).astype(np.float64)
        g = g_all[1:]
        dw, dk = float(np.abs(g - ref).max()), float(np.abs(g - kk).max())
        out['distributed_c64_8gpus'][f'{case} {v}'] = dict(params=int(g.size), vs_lira_c128=dw, vs_lightning_kokkos_c128=dk,
                                                           relative=max(dw, dk) / big,
                                                           cost_relative=float(abs(g_all[0] - ref_all[0]) / abs(ref_all[0])))
        print(f"8 GPUs {case:12s} {v:22s} largest difference relative to the largest derivative {max(dw, dk) / big:.1e}")

if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
