import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lira import paths


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_QASM = {'cx': 'cx', 'cnot': 'cx', 'cz': 'cz', 'h': 'h', 'x': 'x', 'y': 'y', 'z': 'z', 's': 's', 't': 't',
         'sdg': 'sdg', 'tdg': 'tdg', 'swap': 'swap', 'rx': 'rx', 'ry': 'ry', 'rz': 'rz', 'rzz': 'rzz', 'p': 'p', 'cp': 'cp',
         'frx': 'rx', 'fry': 'ry', 'frz': 'rz'}


def to_qasm(case: dict) -> str:
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{case['n']}];"]
    for g in case['gates']:
        qs = ','.join(f'q[{q}]' for q in g['q'])
        if g['theta'] is None:
            lines.append(f"{_QASM[g['name']]} {qs};")
        else:
            tok = f"t{g['p']}" if g['p'] >= 0 else repr(float(g['theta']))
            lines.append(f"{_QASM[g['name']]}({tok}) {qs};")
    return '\n'.join(lines) + '\n'


class VQCSimRunner:
    def __init__(self, case, device):
        sys.path.insert(0, os.path.join(paths.THIRD_PARTY, 'VQCSim', 'code'))
        from converter.fast_model_converter import compile_generic_qasm_to_fast_kernel
        if 'qasm' in case:
            qasm, self.names, self.init = case['qasm'], list(case['param_names']), list(case['param_values'])
        else:
            qasm, self.names = to_qasm(case), [f't{i}' for i in range(case['n_params'])]
            self.init = [g['theta'] for g in case['gates'] if g['p'] >= 0]
        self.kernel = compile_generic_qasm_to_fast_kernel(qasm, max_supported_qubits=case['n']).to(device=device)

    def loss(self, theta, batch=1):
        mapping = {name: theta[i] for i, name in enumerate(self.names)}
        return torch.real(self.kernel.expectation_z(parameter_values=mapping, batch_size=batch)).mean()


class TorchQuantumRunner:
    def __init__(self, case, device, static=0):
        import torchquantum as tq
        import torchquantum.functional as tqf
        self.tq, self.n, self.device = tq, case['n'], device
        self.qdev = tq.QuantumDevice(n_wires=self.n, bsz=1, device=device)
        fn = {'rx': tqf.rx, 'ry': tqf.ry, 'rz': tqf.rz, 'rzz': tqf.rzz, 'p': tqf.phaseshift, 'frx': tqf.rx, 'fry': tqf.ry,
              'frz': tqf.rz, 'h': tqf.hadamard, 'x': tqf.paulix, 'y': tqf.pauliy, 'z': tqf.pauliz, 's': tqf.s, 't': tqf.t,
              'cx': tqf.cnot, 'cnot': tqf.cnot, 'cz': tqf.cz, 'swap': tqf.swap}
        self.ops = [(fn[g['name']], g['q'] if len(g['q']) > 1 else g['q'][0], g['p'], g['theta']) for g in case['gates']]
        self.obs = [tq.PauliZ() for _ in range(self.n)]
        self.init = [g['theta'] for g in case['gates'] if g['p'] >= 0]

        self.static = static
        if static:
            cls = {'rx': tq.RX, 'ry': tq.RY, 'rz': tq.RZ, 'rzz': tq.RZZ, 'p': tq.PhaseShift, 'frx': tq.RX, 'fry': tq.RY,
                   'frz': tq.RZ, 'h': tq.Hadamard, 'x': tq.PauliX, 'y': tq.PauliY, 'z': tq.PauliZ, 's': tq.S, 't': tq.T,
                   'cx': tq.CNOT, 'cnot': tq.CNOT, 'cz': tq.CZ, 'swap': tq.SWAP}
            gates = case['gates']

            class Layer(tq.QuantumModule):
                def __init__(self):
                    super().__init__()
                    self.n_wires = case['n']
                    self.gates = tq.QuantumModuleList(
                        [cls[g['name']]() if g['theta'] is None else
                         cls[g['name']](has_params=True, trainable=g['p'] >= 0, init_params=float(g['theta'])) for g in gates])

                @tq.static_support
                def forward(self, q_device):
                    self.q_device = q_device
                    for op, g in zip(self.gates, gates):
                        op(q_device, wires=g['q'] if len(g['q']) > 1 else g['q'][0])

            self.layer = Layer().to(device)
            self.layer.static_on(wires_per_block=static)
            self.trainable = [(g['p'], op) for g, op in zip(gates, self.layer.gates) if g['theta'] is not None and g['p'] >= 0]

    def plan_time(self):
        def fwd(r):
            sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                r.qdev.reset_states(1)
                r.layer(r.qdev)
            sync()
            return time.perf_counter() - t0
        small = dict(n=6, gates=[dict(name='rx', q=[q], p=q, theta=0.1) for q in range(6)] + [dict(name='cz', q=[q, q + 1], p=-1, theta=None) for q in range(5)])
        warm = TorchQuantumRunner(small, self.device, 2)
        fwd(warm), fwd(warm)
        first, second = fwd(self), fwd(self)
        return dict(plan_time=first - second, first_forward=first, second_forward=second)

    def grads(self, theta):
        if not self.static:
            return theta.grad
        out = theta.new_zeros(theta.shape)
        for p, op in self.trainable:
            out[p] += op.params.grad.reshape(())
            op.params.grad = None
        return out

    def loss(self, theta, batch=1):
        qdev = self.qdev
        qdev.reset_states(batch)
        if self.static:
            self.layer(qdev)
            return self.tq.expval(qdev, list(range(self.n)), self.obs).mean()
        for f, wires, p, th in self.ops:
            if th is None:
                f(qdev, wires=wires)
            elif p >= 0:
                f(qdev, wires=wires, params=theta[p])
            else:
                f(qdev, wires=wires, params=th)
        return self.tq.expval(qdev, list(range(self.n)), self.obs).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['vqcsim', 'torchquantum'], required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--batch', default='1', help='comma-separated batch sizes (identical samples, as in VQCSim\'s benchmark)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tq-static', type=int, default=0,
                    help='TorchQuantum static mode with this many wires per block (0: dynamic mode, its default)')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    case = json.load(open(args.case))
    device = torch.device(args.device)
    row = dict(engine=args.engine, case=case['case'], n=case['n'], gates=len(case['gates']), params=case['n_params'], tq_static=args.tq_static,
               torch=torch.__version__, device=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', batches={})
    try:
        runner = VQCSimRunner(case, device) if args.engine == 'vqcsim' else TorchQuantumRunner(case, device, args.tq_static)
        init = torch.tensor(runner.init, dtype=torch.float32, device=device)
        if args.engine == 'torchquantum' and args.tq_static:
            row.update(runner.plan_time())
            print(f"torchquantum  {case['case']:16s} n={case['n']:2d} static={args.tq_static}  schedule built in {row['plan_time']:8.3f}s  "
                  f"(first forward pass {row['first_forward']:.3f}s, second {row['second_forward']:.3f}s)", flush=True)
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {str(e)[:300]}'
        print(f"{args.engine:13s} {case['case']:16s} n={case['n']:2d}  FAILED {row['error'][:160]}", flush=True)
        runner = None
    for b in ([int(x) for x in args.batch.split(',')] if runner else []):
        theta = torch.nn.Parameter(init.clone())
        res = {}
        try:
            runs = []
            for it in range(args.repeat + 1):
                theta.grad = None
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                sync()
                t0 = time.perf_counter()
                loss = runner.loss(theta, b)
                sync()
                t1 = time.perf_counter()
                loss.backward()
                sync()
                t2 = time.perf_counter()
                grad = runner.grads(theta) if hasattr(runner, 'grads') else theta.grad
                if it:
                    runs.append(dict(forward=t1 - t0, backward=t2 - t1, total=t2 - t0))
                else:
                    first = t2 - t0
            best = min(runs, key=lambda r: r['total'])
            res.update(best, runs=runs, first_pass=first, cost=float(loss.detach()), grad=grad.detach().double().cpu().tolist())
            if device.type == 'cuda':
                res['peak_alloc_gib'] = torch.cuda.max_memory_allocated() / 2**30
                res['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
            print(f"{args.engine:13s} {case['case']:16s} n={case['n']:2d} batch={b:3d}  fwd {best['forward']:8.4f}s  bwd {best['backward']:8.4f}s  "
                  f"total {best['total']:8.4f}s  cost={res['cost']:+.8f}  peak alloc {res.get('peak_alloc_gib', 0):.2f} GiB  first pass {first:.1f}s", flush=True)
        except Exception as e:
            res['error'] = f'{type(e).__name__}: {str(e)[:300]}'
            print(f"{args.engine:13s} {case['case']:16s} n={case['n']:2d} batch={b:3d}  FAILED {res['error'][:140]}", flush=True)
        row['batches'][str(b)] = res
        del theta
        loss = None
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if 'error' in res:
            break
    first = next(iter(row['batches'].values()), None)
    if first:
        row.update({k: v for k, v in first.items() if k != 'runs'})
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(row, f)


if __name__ == '__main__':
    main()
