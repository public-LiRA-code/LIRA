# LIRA
**LIRA: Live-Range Adjoint Differentiation for Exact Variational Quantum Gradients on Large-scale HPC Systems (Submitted ICS'27)**

## Introduction
LIRA is a live-range adjoint differentiation framework that computes exact gradients of variational quantum algorithm (VQA) circuits on many GPUs.
Existing adjoint engines apply every gate of the backward pass to all amplitudes of the state vector and the adjoint state.
Across GPUs, these engines also exchange both vectors for every gate on a global qubit.
Since a VQA circuit starts from the all-zero state, LIRA restricts the dense adjoint state exactly to the amplitudes of the qubits already acted on.
LIRA then plans both passes from the live range of each qubit (i.e., the gates from its first gate to its last gate).
To reduce both the amplitude updates and the exchanges, LIRA builds four techniques on the live ranges:
- **Live-range gate order.** *Planner* delays the first gate of each qubit and reorders the gates that commute for every angle (e.g., diagonal gates on a shared qubit).
- **Live-range-aware exchanges.** *Planner* moves each qubit to a global position after its live range and exchanges only restricted amplitudes at planned points. With early distribution, GPU 0 deals out the restricted state to all GPUs while the state is small.
- **Width-adaptive backward pass.** After the first gate of a qubit, the backward pass drops the half of the adjoint state in which the qubit is |1⟩. Window kernels compute the derivatives of consecutive gates in one sweep without intermediate state vectors.
- **Late term entry.** Each term of the Hamiltonian enters the adjoint state after the last gate on its qubits at a point where these qubits are local.

## Key Features
- **Exact gradients.** The adjoint restriction drops only the amplitudes that no remaining derivative uses. At 28 qubits in double precision, every derivative differs from lightning.gpu and lightning.kokkos by at most 2.0 × 10⁻¹⁴.
- **Two state vectors per gradient.** LIRA stores only the state vector and the adjoint state at any circuit depth. It runs 32 qubits per A100 GPU against 30 qubits of the lightning engines and computes exact gradients of 40 qubits on 256 GPUs.
- **Fewer exchanges.** LIRA requires 25-187× fewer exchange rounds than lightning.gpu on 2 to 512 GPUs (512 GPUs from the exchange rules alone). On 128 GPUs with 30 qubits per GPU, LIRA is 34.5-173.5× faster than lightning.gpu.
- **One execution plan per circuit.** Each MPI process compiles the execution plan of its GPU once without communication. At every optimization step, the host only rebuilds the matrices of the gates and windows from the new angles.
- **GPU-aware MPI.** LIRA runs one MPI process per GPU. A two-way exchange overlaps with the operations around it when at least eight of them can run group by group. The transfers across nodes run during the transfers inside a node.
- **Ablation variants.** LIRA runs the full scheme unless a named variant switches off one technique for the ablation study (`lira_code/lira/scheme.py`).

## Core Components
- `lira_code/lira/planner.py` — *Planner*: live ranges, activation order of the qubits, reordering of commuting gates and the active width of every gate.
- `lira_code/lira/exchanges.py` — exchange planning of both passes and early distribution for multiple GPUs.
- `lira_code/lira/engine.py` — single-GPU *Execution*: forward pass, restricted backward pass, late term entry.
- `lira_code/lira/windows.py` — window kernels, generated and compiled per window shape.
- `lira_code/lira/fusion.py` — grouping of the gates into the operations of a window.
- `lira_code/lira/distributed.py` — distributed *Execution*: joint and overlapped exchanges, rebinding of the angles.
- `lira_code/lira/scheme.py` — the full scheme of the paper and the variants of the ablation study.
- `lira_code/lira/circuits.py`, `decompose.py`, `gates.py`, `qasm.py` — benchmark circuits and gate set.
- `lira_code/benchmarks/` — drivers of LIRA and of the baselines used in the evaluation.

## Installation

### Requirements
- NVIDIA A100 GPUs (four per node), CUDA 12.9
- GPU-aware MPI (Cray MPICH 9.0.1) and Slurm
- Python 3.13 with the packages of `requirements.txt` (CuPy 14.2, PennyLane 0.45.1) and mpi4py 4.1.2 built against the MPI library
- Baselines: CMake and Ninja (Kokkos 5.1.0, lightning.kokkos), PyTorch 2.13 (TorchQuantum, VQCSim, Kawase's simulator), Julia 1.11 (Yao.jl), Boost (Qulacs)

### Set Up Environment
```bash
git clone <repository URL> LIRA
cd LIRA

# Adjust the module lines of setup_env.sh for your system
source setup_env.sh

# Build the engines, the MPI profiling library and the inputs
bash script_example/00_setup/run.sh
```

### Verify Installation
```bash
srun -N 1 -n 1 --gpus-per-node=1 $LIRA_PYTHON lira_code/tests/test_lira.py
srun -N 1 -n 4 --gpus-per-node=4 $LIRA_PYTHON lira_code/tests/test_distributed.py
```
Both tests compare every derivative with parameter-shift and NumPy references and print `PASS`.

## Usage
```python
import numpy as np
from lira import circuits
from lira.planner import make_plan
from lira.engine import LiraEngine

n, gates = circuits.sel(28, 4)
plan = make_plan(n, gates)
bit = {q: j for j, q in enumerate(plan.order)}
terms = [(1.0 / n, {bit[q]: 'Z'}) for q in range(n)]
cost, grads, times = LiraEngine(n, np.complex64).run(plan, terms)
```
On multiple GPUs, `DistributedEngine(n)` from `lira.distributed` takes the same plan and terms (`compile(plan)` once,
`rebind(plan, theta)` and `run(plan, terms)` at every step), with one MPI process per GPU.

### Data Layout
- `data/circuits/MQTBench_<n>q/` — VQC, SU2Random and RealAmpRandom of MQT Bench 0.2.2 (target-independent level, measurements removed), with `SHA256SUMS`. SEL, TFIM HVA ring and HEA are generated in code.
- `data/cases/` — circuits exported for TorchQuantum, VQCSim and Yao.jl by `prepare_workloads.sh`.
- `results/` — one JSON file per measured point, `<engine>_<circuit>_<cost>_g<GPUs>_n<qubits>[_<variant>].json`, with `logs/` and `grads/`. A point that hits its time limit is written with `"status": "timeout"`.

The paths can be moved with `LIRA_DATA`, `LIRA_CIRCUITS`, `LIRA_CASES`, `LIRA_RESULTS` and `LIRA_THIRD_PARTY`.

### Building Engines and Preparing Inputs
`script_example/00_setup/` builds everything the evaluation runs:
- `build_engines.sh` — lightning.gpu 0.45.0 with MPI, lightning.kokkos 0.45.0 with Kokkos 5.1.0, Qulacs, TorchQuantum, VQCSim, Kawase's simulator and Yao.jl, each at a fixed commit.
- `build_drivers.py` — `libmpiprof.so`, which counts the MPI calls of lightning.gpu for the communication analysis.
- `prepare_workloads.sh` — checks the circuit files and exports the cases of the single-GPU baselines.

lightning.kokkos deadlocks from eight GPUs in the exchange that aligns the qubit mappings of two vectors.
`lira_code/patches/lightning-kokkos-v0.45.0-receive-rank.patch` corrects the receiving rank of this exchange; the
runners enable it with `LKK_RECEIVE_RANK_FIX=1`.

### Reproducing paper results
Each directory of `script_example/` reproduces one subsection of the evaluation. Its `run.sh` runs the points of
`script_example/common/config/<section>.json` that fit the current Slurm allocation and skips points with a result.

| Directory | Paper | GPUs |
|---|---|---|
| `01_single_node` | Single-node performance, Kawase comparison, benchmark table | 1-4 |
| `02_multi_node_scalability` | Weak and strong scaling, variability | 1-256 |
| `03_diverse_circuits` | Six circuits on 128 GPUs | 128 |
| `04_extreme_scale_out` | Largest circuits on 8 to 256 GPUs | 8-256 |
| `05_time_and_communication` | Time breakdown, exchange rounds and data | 32-128 |
| `06_ablation` | Design choices, late term entry, window kernels | 1-128 |
| `07_end_to_end_vqe` | VQE runs of TFIM HVA ring | 4-128 |
| `08_accuracy` | Gradient accuracy | 1-8 |

```bash
salloc -N 32 --gpus-per-node=4 -t 04:00:00
bash script_example/03_diverse_circuits/run.sh
```

The variants of LIRA (`lira/scheme.py`) used by the ablation study:

| Variant | Settings |
|---|---|
| `full` | early distribution, overlapped exchanges, late term entry (from 2 GPUs), window kernels |
| `full-width` | every gate at the full width, window kernels |
| `restriction-only` | restricted adjoint state, no early distribution or overlap |
| `no-late-term-entry` | `full` without late term entry |
| `early-distribution-only`, `overlap-only` | one of the two on top of `restriction-only` |
| `no-early-distribution` | `full` without early distribution |

The single-GPU driver also runs `full-width-gate-by-gate`, `gate-by-gate`, `no-adjoint-restriction` and
`full-width-windows`. `LIRA_WINDOW_KERNEL` selects the window kernels (`v2` by default; `v2-uncapped`, `custatevec`,
`cublas` for the kernel comparison).

## Repository Structure
```
LIRA/
├── lira_code/
│   ├── lira/
│   │   ├── planner.py
│   │   ├── exchanges.py
│   │   ├── engine.py
│   │   ├── windows.py
│   │   ├── fusion.py
│   │   ├── distributed.py
│   │   ├── scheme.py
│   │   ├── circuits.py
│   │   ├── decompose.py
│   │   ├── gates.py
│   │   ├── qasm.py
│   │   ├── paths.py
│   │   ├── reference.py
│   │   ├── lightning_ref.py
│   │   ├── qulacs_ref.py
│   │   ├── custatevec_backend.py
│   │   └── cublas_backend.py
│   ├── benchmarks/
│   ├── patches/
│   ├── tests/
│   └── tools/
├── script_example/
│   ├── 00_setup/
│   ├── 01_single_node/
│   ├── 02_multi_node_scalability/
│   ├── 03_diverse_circuits/
│   ├── 04_extreme_scale_out/
│   ├── 05_time_and_communication/
│   ├── 06_ablation/
│   ├── 07_end_to_end_vqe/
│   ├── 08_accuracy/
│   └── common/
├── data/
├── requirements.txt
└── setup_env.sh
```
