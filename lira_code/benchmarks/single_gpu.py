import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cupy as cp

from lira import circuits as C
from lira.gates import Gate
from lira.engine import LiraEngine, traffic
from lira.planner import make_plan
from lira.scheme import window as parse_window
from lira import paths


NATIVE = {}


def build(case: str, n: int, seed: int = 0):
    a = case.split('_')
    if a[0] in ('tfim1d', 'tfim1dp', 'tfim2d'):
        if a[0] == 'tfim2d':
            R, Cc = map(int, a[1].split('x'))
            nn, graph, p = R * Cc, 'grid2d:' + a[1], int(a[2])
        else:
            nn, graph, p = n, ('line' if a[0] == 'tfim1d' else 'ring'), {'n': n, 'n/2': n // 2}.get(a[1]) or int(a[1])
        NATIVE[case] = C.tfim_terms(nn, graph, seed=seed)
        return C.qaoa(nn, p, graph, seed=seed)
    if a[0] in ('vqc', 'vqca'):
        nn, gates = C.from_qasm(paths.circuit('vqc', n))
        if a[0] == 'vqca':
            first_ry = next(i for i, g in enumerate(gates) if g.name == 'ry')
            gates = [Gate('frz', g.targets, g.params) if (i < first_ry and g.name == 'rz') else g for i, g in enumerate(gates)]
        return nn, gates
    if a[0] == 'mqt':
        return C.from_qasm(paths.circuit(a[1], n))
    if a[0] == 'hea':
        L = {'n': n, 'n/2': n // 2, '2n': 2 * n}.get(a[2]) or int(a[2])
        return C.hea(n, L, a[1], gate2='cz' if a[1] == 'full' else 'cx', seed=seed)
    if a[0] == 'qaoa':
        NATIVE[case] = [(-0.5, {i: 'Z', j: 'Z'}) for i, j in C._graph_edges(n, a[1], seed)]
        return C.qaoa(n, int(a[2]), a[1], seed=seed)
    if a[0] == 'sel':
        return C.sel(n, {'n': n, 'n/2': n // 2}.get(a[1]) or int(a[1]), seed=seed)
    if a[0] == 'khea':
        return C.kawase_hea(n, {'n': n, 'n/2': n // 2}.get(a[1]) or int(a[1]), seed=seed)
    if a[0] == 'mps':
        return C.mps_staircase(n, int(a[1]))
    if a[0] == 'uccsd':
        return C.uccsd(int(a[1]), int(a[2]))
    raise ValueError(case)


def observable(kind: str, n: int, plan, case: str = ''):
    bit = {q: j for j, q in enumerate(plan.order)}
    if kind == 'native':
        return [(c, {bit[q]: p for q, p in ps.items()}) for c, ps in NATIVE[case]]
    if kind == 'z0':
        return [(1.0, {bit[0]: 'Z'})]
    if kind == 'z0z1':
        return [(1.0, {bit[0]: 'Z', bit[1]: 'Z'})]
    if kind == 'parity':
        return [(1.0, {bit[q]: 'Z' for q in range(n)})]
    if kind == 'meanz':
        return [(1.0 / n, {bit[q]: 'Z'}) for q in range(n)]
    if kind == 'zzring':
        return [(1.0, {bit[q]: 'Z', bit[(q + 1) % n]: 'Z'}) for q in range(n)]
    if kind == 'ixyz':
        return [(1.0, {bit[q]: 'IXYZ'[q % 4] for q in range(n) if q % 4})]
    terms = [(1.0, {bit[q]: 'Z', bit[(q + 1) % n]: 'Z'}) for q in range(n)]
    return terms + [(0.7, {bit[q]: 'X'}) for q in range(n)]


class PeakMem:

    def __enter__(self):
        import subprocess
        import tempfile
        self._f = tempfile.NamedTemporaryFile('w+', suffix='.mem')
        bus = cp.cuda.Device().pci_bus_id
        self._p = subprocess.Popen(['nvidia-smi', '-i', bus, '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                                    '-lms', '20'], stdout=self._f, stderr=subprocess.DEVNULL)
        while not self._samples():
            time.sleep(0.01)
        return self

    def _samples(self):
        with open(self._f.name) as f:
            return [float(x) for x in f.read().split() if x.strip().replace('.', '').isdigit()]

    def __exit__(self, *exc):
        cp.cuda.get_current_stream().synchronize()
        time.sleep(0.05)
        self._p.terminate()
        self._p.wait()
        v = self._samples()
        self.gib = (max(v) - v[0]) / 1024.0
        self._f.close()


def best(runs):
    keys = runs[0].keys()
    return {k: min(r[k] for r in runs) for k in keys}


def record(row, key, runs):
    row[key] = best(runs)
    row[key + '_runs'] = runs


def free():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


VARIANTS = {
    'full-width-gate-by-gate': ('full', False, False, False, False),
    'gate-by-gate': ('adaptive', False, False, False, False),
    'no-adjoint-restriction': ('adaptive', False, False, False, True),
    'full-width-windows': ('full-ordered', True, True, True, False),
    'full': ('adaptive', True, True, True, False),
}
REFERENCE = 'full-width-gate-by-gate'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=28)
    ap.add_argument('--cases', default='sel_4')
    ap.add_argument('--obs', choices=['tfim', 'z0', 'ixyz', 'parity', 'native', 'meanz', 'zzring'], default='meanz')
    ap.add_argument('--dtype', choices=['c64', 'c128'], default='c64')
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--no-lightning', action='store_true')
    ap.add_argument('--qulacs', action='store_true', help="also run Qulacs' own backprop (CPU, complex128)")
    ap.add_argument('--window', default='4,6,1', help='window kernels (lira/windows.py): K or K,Kperm,rot')
    ap.add_argument('--margin', type=float, default=1.0, help='fuse a diagonal group if separate cost > margin x fused estimate')
    ap.add_argument('--seed', type=int, default=0, help='seed of the circuit parameters (and of the random graph)')
    ap.add_argument('--variants', default='full', help='comma list of ' + ', '.join(VARIANTS) + '; ' + REFERENCE +
                                                       ' always runs once as the gradient reference')
    ap.add_argument('--out', default='')
    ap.add_argument('--save-grads', default='', help='prefix: writes <prefix>_<case>_n<n>.npy (the last variant)')
    args = ap.parse_args()
    dtype = np.complex64 if args.dtype == 'c64' else np.complex128
    sel = [v for v in args.variants.split(',') if v]
    unknown = [v for v in sel if v not in VARIANTS]
    if unknown:
        raise SystemExit(f'unknown variants {unknown}; choose from {list(VARIANTS)}')
    window = parse_window(args.window)

    gpu = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    free_b, total_b = cp.cuda.Device().mem_info
    print(f'GPU: {gpu}  free {free_b / 2**30:.1f} / {total_b / 2**30:.1f} GiB   dtype={args.dtype}  obs={args.obs}', flush=True)

    Lightning = None
    if not args.no_lightning:
        from lira.lightning_ref import LightningBaseline as Lightning

    wn, wg = C.hea(12, 2, 'linear')
    wp = make_plan(wn, wg)
    wobs = 'tfim' if args.obs == 'native' else args.obs
    LiraEngine(wn, dtype).run(wp, observable(wobs, wn, wp), fuse=False, window=0, fuse_obs=False)
    if Lightning:
        Lightning(wn, dtype).run(wp, observable(wobs, wn, wp))

    rows = []
    for case in args.cases.split(','):
        n, gates = build(case, args.n, args.seed)
        t0 = time.perf_counter()
        plan = make_plan(n, gates)
        t_plan = time.perf_counter() - t0
        full = make_plan(n, gates, order=plan.order, full_width=True)
        full_o = make_plan(n, gates, order=plan.order, full_width=True, keep_order=True)
        plans = {'adaptive': plan, 'full': full, 'full-ordered': full_o}
        terms = observable(args.obs, n, plan, case)
        row = dict(case=case, n=n, gates=len(gates), params=plan.n_params, r_work=plan.r_work['grad'], obs_terms=len(terms),
                   t_plan=t_plan, traffic_full=traffic(full), traffic_lira=traffic(plan), obs=args.obs, dtype=args.dtype,
                   seed=args.seed, window=args.window)
        print(f"\n=== {case}: n={n} gates={len(gates)} params={plan.n_params} "
              f"R_work={plan.r_work['grad']:.3f} (bound {1 / plan.r_work['grad']:.2f}x)  plan {t_plan * 1e3:.0f} ms  obs terms={len(terms)}", flush=True)

        ref = None
        plan_l = make_plan(n, gates, order=list(range(n)), full_width=True, keep_order=True)
        terms_l = observable(args.obs, n, plan_l, case)
        if Lightning:
            try:
                with PeakMem() as pm:
                    lg = Lightning(n, dtype)
                    runs = []
                    for _ in range(args.repeat):
                        c_l, g_l, t = lg.run(plan_l, terms_l)
                        runs.append(t)
                record(row, 'lightning', runs)
                row['lightning_peak_gib'] = pm.gib
                ref = (c_l, g_l)
                print(f"  lightning.gpu : fwd {row['lightning']['forward']:8.3f}s  bwd {row['lightning']['backward']:8.3f}s  "
                      f"total {row['lightning']['total']:8.3f}s   cost={c_l:+.8f}   peak mem {pm.gib:.1f} GiB", flush=True)
            except Exception as e:
                row['lightning_error'] = repr(e)[:300]
                print(f'  lightning.gpu : FAILED {row["lightning_error"]}', flush=True)
            lg = None
            free()

        q_ref = None
        if args.qulacs:
            from lira.qulacs_ref import QulacsBaseline
            try:
                qb = QulacsBaseline(n)
                runs = []
                for _ in range(args.repeat):
                    c_q, g_q, t = qb.run(plan_l, terms_l)
                    runs.append(t)
                record(row, 'qulacs', runs)
                row['qulacs_threads'] = qb.threads
                q_ref = g_q
                print(f"  qulacs (CPU)  : fwd {row['qulacs']['forward']:8.3f}s  bwd {row['qulacs']['backward']:8.3f}s  "
                      f"total {row['qulacs']['total']:8.3f}s   cost={c_q:+.8f}   ({qb.threads} threads, complex128)", flush=True)
            except Exception as e:
                row['qulacs_error'] = repr(e)[:300]
                print(f'  qulacs (CPU)  : FAILED {row["qulacs_error"]}', flush=True)

        free()
        with PeakMem() as pm_lira:
            eng = LiraEngine(n, dtype)
            eng.allocate()
            eng.run(plan, terms, fuse=False, window=0, fuse_obs=False)
        row['lira_peak_gib'] = pm_lira.gib
        g_full = None
        for key in [k for k in VARIANTS if k in sel or k == REFERENCE]:
            pname, fz, wz, fo, lf = VARIANTS[key]
            pl, win = plans[pname], (window if wz else 0)
            runs = []
            if fz:
                from lira import windows as W
                from lira.fusion import summary
                t0, c0, s0 = time.perf_counter(), W.STATS['compiled'], W.STATS['seconds']
                ops = eng.fused_ops(pl, args.margin, win)
                row[key + '_ops'] = summary(ops, n)
                if win:
                    row[key + '_ops'].update(prepare=time.perf_counter() - t0, shapes=len({o.cache['whost']['shape'] for o in ops if o.kind == 'window'}),
                                             compiled=W.STATS['compiled'] - c0, compile=W.STATS['seconds'] - s0)
                if fo:
                    row[key + '_ops']['obs_sweeps'] = len(eng._obs_sweeps(terms))
                eng.run(pl, terms, fuse=True, margin=args.margin, window=win, fuse_obs=fo, late_obs=False)
            for _ in range(args.repeat if key in sel else 1):
                c, g, t = eng.run(pl, terms, fuse=fz, margin=args.margin, window=win, fuse_obs=fo, late_obs=False, lam_full=lf)
                runs.append(t)
            if key == REFERENCE:
                g_full = g
            else:
                row[key + '_vs_reference_maxdiff'] = float(np.abs(g - g_full).max())
            record(row, key, runs)
            err = ''
            if ref is not None:
                row[key + '_maxdiff'] = float(np.abs(g - ref[1]).max())
                err = f"  max|grad-lgpu|={row[key + '_maxdiff']:.1e}"
            if fo:
                err += f"   [observable: {len(terms)} terms in {row[key + '_ops']['obs_sweeps']} sweeps]"
            print(f"  {key:<24}: fwd {row[key]['forward']:8.3f}s  bwd {row[key]['backward']:8.3f}s  "
                  f"total {row[key]['total']:8.3f}s   cost={c:+.8f}{err}   (obs {row[key]['observable']:.3f}s)", flush=True)
        eng.release()
        eng = None
        free()
        if args.save_grads:
            np.save(f"{args.save_grads}_{case.replace('/', '-')}_n{n}.npy", np.concatenate([[c], g]))

        if q_ref is not None:
            row['lira_vs_qulacs_maxdiff'] = float(np.abs(g - q_ref).max())
            print(f"  max|grad(lira) - grad(qulacs)| = {row['lira_vs_qulacs_maxdiff']:.1e}", flush=True)
        rows.append(row)
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as fh:
                json.dump(dict(gpu=gpu, rows=rows), fh, indent=1)
    if args.out:
        print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
