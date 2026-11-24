import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CODE = os.path.join(REPO, 'lira_code')
BENCH = os.path.join(CODE, 'benchmarks')
sys.path.insert(0, CODE)
from lira import paths

RESULTS = paths.RESULTS
os.environ.setdefault('LIRA_RESULTS', RESULTS)
os.environ.setdefault('LIRA_MPIPROF_LIB', os.path.join(CODE, 'tools', 'libmpiprof.so'))
GPN = int(os.environ.get('LIRA_GPUS_PER_NODE', 4))
CPN = os.environ.get('LIRA_CPUS_PER_TASK', '32')


def interpreter(var):
    return os.environ.get(var) or sys.executable


def srun(G, minutes):
    if G <= 1:
        return ['srun', '-N', '1', '-n', '1', '-c', CPN, '--gpus-per-node=1', '--exact', f'--time={minutes}']
    nodes = (G + GPN - 1) // GPN
    return ['srun', '-N', str(nodes), '-n', str(G), f'--ntasks-per-node={min(G, GPN)}', '-c', CPN, f'--gpus-per-node={GPN}',
            '--kill-on-bad-exit=1', f'--time={minutes}']


def expand(xs):
    return [os.path.expandvars(x) for x in xs]


def command(p, out):
    E, G, n, case, obs = p['engine'], p['gpus'], str(p['qubits']), p['case'], p['obs']
    args, env = expand(p['args']), {k: os.path.expandvars(v) for k, v in p['env'].items()}
    lira, lkk, torch = interpreter('LIRA_PYTHON'), interpreter('LIRA_LKK_PYTHON'), interpreter('LIRA_TORCH_PYTHON')
    common = ['--n', n, '--cases', case, '--obs', obs]
    if E == 'lira':
        return srun(G, p['minutes']) + [lira, f'{BENCH}/multi_gpu.py'] + common + args + ['--out', out], env
    if E == 'lgpu':
        env.setdefault('LIRA_GPU_BEFORE_MPI', '1')
        return srun(G, p['minutes']) + [lira, f'{BENCH}/lightning_gpu.py'] + common + args + ['--out', out], env
    if E == 'lkk':
        env.update(LIRA_GPU_BEFORE_MPI='1', LKK_RECEIVE_RANK_FIX='1')
        return srun(G, p['minutes']) + [lkk, f'{BENCH}/lightning_kokkos.py'] + common + args + ['--out', out], env
    if E in ('onegpu', 'qulacs'):
        if E == 'qulacs':
            env.setdefault('OMP_NUM_THREADS', CPN)
            if os.environ.get('LIRA_QULACS_PATH'):
                env['PYTHONPATH'] = os.environ['LIRA_QULACS_PATH'] + os.pathsep + os.environ.get('PYTHONPATH', '')
        return srun(1, p['minutes']) + [lira, f'{BENCH}/single_gpu.py'] + common + args + ['--out', out], env
    if E in ('tq', 'vqcsim'):
        engine = 'torchquantum' if E == 'tq' else 'vqcsim'
        case_file = os.path.join(paths.CASES, f"{case.replace('/', '-')}_n{n}.json")
        return srun(1, p['minutes']) + [torch, f'{BENCH}/autograd_baselines.py', '--engine', engine, '--case', case_file] + args + \
            ['--out', out], env
    if E == 'yao':
        case_file = os.path.join(paths.CASES, f"{case.replace('/', '-')}_n{n}.json")
        tmp = out[:-5] + '.tmp'
        julia = os.environ.get('LIRA_JULIA', 'julia')
        return srun(1, p['minutes']) + [julia, f'--project={BENCH}/yao_env', f'{BENCH}/yao.jl', tmp] + args + [case_file], env
    if E == 'kawase':
        return srun(1, p['minutes']) + [torch, f'{BENCH}/kawase.py', '--n', n, '--obs', obs] + args + ['--out', out[:-5]], env
    if E == 'vqe':
        py = lkk if '--engine' in args and args[args.index('--engine') + 1] == 'kokkos' else lira
        if py == lkk:
            env.update(LIRA_GPU_BEFORE_MPI='1', LKK_RECEIVE_RANK_FIX='1')
        elif 'lgpu' in args:
            env.setdefault('LIRA_GPU_BEFORE_MPI', '1')
        return srun(G, p['minutes']) + [py, f'{BENCH}/vqe.py', '--case', case, '--obs', obs, '--n', n, '--iters', '200'] + args + \
            ['--out', out], env
    if E == 'breakdown':
        return srun(G, p['minutes']) + [lira, f'{BENCH}/time_breakdown.py', '--n', n, '--cases', case, '--out', out], env
    if E == 'counts':
        if '--check' in args:
            return [lira, f'{BENCH}/exchange_counts.py'] + args + ['--results', RESULTS], env
        return [lira, f'{BENCH}/exchange_counts.py'] + args + ['--cases', case, '--out', out], env
    if E == 'workloads':
        return [lira, f'{HERE}/workloads.py', out], env
    if E == 'compare':
        return [lira, f'{HERE}/compare_gradients.py', RESULTS, out], env
    raise ValueError(E)


def finish(p, out):
    if p['engine'] == 'yao':
        tmp = out[:-5] + '.tmp'
        found = glob.glob(os.path.join(tmp, '*.json'))
        if found:
            shutil.move(found[0], out)
        shutil.rmtree(tmp, ignore_errors=True)
    if p['engine'] == 'kawase' and os.path.exists(out[:-5] + '.json') and not os.path.exists(out):
        shutil.move(out[:-5] + '.json', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('section')
    ap.add_argument('--only', default='', help='run only the points whose tag contains this text')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, 'config', a.section + '.json')))
    gpus = int(os.environ.get('SLURM_JOB_NUM_NODES', 0)) * GPN
    os.makedirs(os.path.join(RESULTS, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, 'grads'), exist_ok=True)
    for p in cfg['points']:
        if a.only and a.only not in p['tag']:
            continue
        out = os.path.join(RESULTS, p['tag'] + '.json')
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        if p['gpus'] > gpus:
            print(f"[skip] {p['tag']}: needs {p['gpus']} GPUs, the allocation has {gpus}", flush=True)
            continue
        cmd, env = command(p, out)
        if a.dry_run:
            print(' '.join(cmd))
            continue
        log = os.path.join(RESULTS, 'logs', p['tag'] + '.log')
        t0 = time.time()
        try:
            with open(log, 'w') as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, **env),
                                   timeout=p['minutes'] * 60 + 180, stdin=subprocess.DEVNULL)
            status = 'ok' if r.returncode == 0 else f'failed (exit {r.returncode})'
        except subprocess.TimeoutExpired:
            status = 'timeout'
        if status != 'ok' and time.time() - t0 >= 0.97 * p['minutes'] * 60:
            status = 'timeout'
        finish(p, out)
        if p['engine'] == 'counts' and '--check' in p['args']:
            shutil.copy(log, out[:-5] + '.txt')
            json.dump(dict(status=status, log=os.path.basename(log)), open(out, 'w'))
        elif not os.path.exists(out) or os.path.getsize(out) == 0:
            json.dump(dict(status=status, time_limit_min=p['minutes'], tag=p['tag']), open(out, 'w'))
        print(f"[{status}] {p['tag']}  {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
