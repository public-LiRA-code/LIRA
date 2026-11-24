import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lira import paths
sys.path.insert(0, os.path.join(paths.THIRD_PARTY, 'triton_qml'))
from custom_quantum_simulator.custom_quantum_simulator import FunctionalQuantumSimulator

ROT = ["rx", "ry", "rz"]


class TritonHEA(torch.nn.Module):
    def __init__(self, n, layers, theta, mem_save, device, ckpt_blocks=-1, ckpt_layers=0, obs='ixyz'):
        super().__init__()
        self.n, self.layers = n, layers
        self.ckpt_blocks = (2 if (n >= 28 and not mem_save) else 0) if ckpt_blocks < 0 else ckpt_blocks
        self.ckpt_layers = ckpt_layers
        self.sim = FunctionalQuantumSimulator(n, device=device, backend_type='triton',
                                              dtype_str='float32', memory_save_mode=mem_save)
        self.theta = torch.nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        if obs == 'meanz':
            self.obs = ['I' * q + 'Z' + 'I' * (n - q - 1) for q in range(n)]
            self.coeff = 1.0 / n
        else:
            self.obs = ["IXYZ" * (n // 4) + "IXYZ"[:n % 4]]
            self.coeff = 1.0

    def _blocks(self, sv, th, start, stop):
        for i in range(start, stop):
            a, b, c = 3 * i, 3 * i + 1, 3 * i + 2
            sv = self.sim.apply_fused_three_qubit_block(sv, a, ROT, list(th[a]), b, ROT, list(th[b]), c, ROT, list(th[c]))
        return sv

    def _layer(self, sv, th):
        n, nb = self.n, self.n // 3
        if self.ckpt_blocks > 0:
            nf = self.ckpt_blocks
            done = 0
            for i in range(nb // nf):
                sv = checkpoint.checkpoint(self._blocks, sv, th, i * nf, (i + 1) * nf, use_reentrant=False)
                done = (i + 1) * nf
            sv = self._blocks(sv, th, done, nb)
        else:
            sv = self._blocks(sv, th, 0, nb)
        if n % 3 == 2:
            sv = self.sim.apply_fused_two_qubit_block(sv, n - 2, ROT, list(th[n - 2]), n - 1, ROT, list(th[n - 1]))
        elif n % 3 == 1:
            sv = self.sim.apply_single_qubits_gate_sequence(sv, ROT, list(th[n - 1]), n - 1)
        return self.sim.apply_fused_cz_gate(sv, list(range(n)), [(i + 1) % n for i in range(n)])

    def _layers(self, sv, theta, l0, l1):
        for l in range(l0, l1):
            sv = self._layer(sv, theta[l])
        return sv

    def forward(self, sv):
        s = self.ckpt_layers
        if s > 0:
            for l0 in range(0, self.layers, s):
                sv = checkpoint.checkpoint(self._layers, sv, self.theta, l0, min(l0 + s, self.layers), use_reentrant=False)
        else:
            sv = self._layers(sv, self.theta, 0, self.layers)
        e = self.sim.calc_exp_val(sv, self.obs[0])
        for o in self.obs[1:]:
            e = e + self.sim.calc_exp_val(sv, o)
        return e * self.coeff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=28)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', choices=['zero', 'random'], default='zero')
    ap.add_argument('--mem-save', action='store_true')
    ap.add_argument('--ckpt-blocks', type=int, default=-1,
                    help='fused blocks per checkpoint segment inside a layer (-1: as in his benchmark script, 0: none)')
    ap.add_argument('--ckpt-layers', type=int, default=0, help='also checkpoint groups of this many layers (0: off)')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--obs', choices=['ixyz', 'meanz'], default='ixyz',
                    help='ixyz: his benchmark observable; meanz: mean over qubits of <Z_q> (single_gpu.py --obs meanz)')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    dev = torch.device('cuda:0')
    n = args.n
    theta = np.random.default_rng(args.seed).uniform(0, 2 * np.pi, (args.layers, n, 3))
    model = TritonHEA(n, args.layers, theta, args.mem_save, dev, args.ckpt_blocks, args.ckpt_layers, args.obs).to(dev)

    if args.init == 'zero':
        sv0 = torch.zeros(1, 2 ** n, dtype=torch.complex64, device=dev)
        sv0[:, 0] = 1.0
    else:
        g = torch.Generator(device=dev).manual_seed(args.seed)
        sv0 = torch.complex(2 * torch.rand(1, 2 ** n, device=dev, generator=g) - 1,
                            2 * torch.rand(1, 2 ** n, device=dev, generator=g) - 1)
        sv0 = sv0 / torch.linalg.vector_norm(sv0)

    def step():
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e = model(sv0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e.sum().backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return float(e.detach().sum()), t1 - t0, t2 - t1

    name = ('triton fused' + (' (mem save)' if args.mem_save else '')
            + f' ckpt blocks={model.ckpt_blocks} layers={model.ckpt_layers}')
    try:
        step()
        torch.cuda.reset_peak_memory_stats(dev)
        runs = [step() for _ in range(args.repeat)]
    except torch.OutOfMemoryError:
        print(f"[Kawase {name}] {torch.cuda.get_device_name(0)}  n={n} layers={args.layers} init={args.init}  "
              f"OUT OF MEMORY ({torch.cuda.get_device_properties(dev).total_memory / 2 ** 30:.0f} GiB GPU)", flush=True)
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out + '.json', 'w') as f:
                json.dump(dict(n=n, layers=args.layers, init=args.init, mem_save=args.mem_save, ckpt_blocks=model.ckpt_blocks,
                               ckpt_layers=model.ckpt_layers, oom=True, gpu=torch.cuda.get_device_name(0)), f, indent=1)
        return
    cost = runs[-1][0]
    fwd, bwd = min(r[1] for r in runs), min(r[2] for r in runs)
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    reserved = torch.cuda.max_memory_reserved(dev) / 2 ** 30
    grads = model.theta.grad.detach().cpu().numpy().astype(np.float64).reshape(-1)
    print(f"[Kawase {name}] {torch.cuda.get_device_name(0)}  n={n} layers={args.layers} init={args.init}  "
          f"fwd {fwd:.3f}s  bwd {bwd:.3f}s  total {fwd + bwd:.3f}s  peak {peak:.1f} GiB (reserved {reserved:.1f})  cost={cost:+.8f}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out + '.npy', np.concatenate([[cost], grads]))
        with open(args.out + '.json', 'w') as f:
            json.dump(dict(n=n, layers=args.layers, init=args.init, mem_save=args.mem_save, obs=args.obs,
                           ckpt_blocks=model.ckpt_blocks, ckpt_layers=model.ckpt_layers, forward=fwd,
                           backward=bwd, total=fwd + bwd, peak_gib=peak, reserved_gib=reserved, cost=cost,
                           runs=[dict(forward=r[1], backward=r[2]) for r in runs],
                           gpu=torch.cuda.get_device_name(0)), f, indent=1)


if __name__ == '__main__':
    main()
