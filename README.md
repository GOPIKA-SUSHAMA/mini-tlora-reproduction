# mini-tLoRA Reproduction

A correctness-first, progressively optimized reproduction of the systems ideas in **[tLoRA: Efficient Multi-LoRA Training with Elastic Shared Super-Models](https://arxiv.org/abs/2602.07263)**.

This repository rebuilds the paper's main mechanisms in small, inspectable stages: a shared frozen backbone with heterogeneous LoRA adapters, fused Triton execution, nano-batch compute/communication overlap, AIMD control, residual-capacity-aware scheduling, and ACMETrace-driven online simulation.

> **Scope:** this is an independent research reproduction, not the authors' official implementation. The project reproduces mechanisms and validates them on CPU and free NVIDIA T4 hardware. It does **not** claim to reproduce the paper's full 12×A100 profiling setup or production 128-GPU simulator.

## Why this project is useful

The code is designed for developers and researchers who want to understand tLoRA as a working systems pipeline rather than only as a paper.

Key properties:

- **Correctness first:** shared-model outputs, adapter gradients, optimizer isolation, and training updates are checked against independent LoRA references.
- **Heterogeneous LoRA support:** adapters can use different ranks and batch sizes while sharing one frozen backbone.
- **Progressive optimization:** CPU baselines expose routing overhead before moving to CUDA and Triton.
- **GPU kernel experiments:** custom Triton kernels avoid materializing full LoRA update matrices and test fused low-rank execution.
- **Distributed overlap:** two-GPU NCCL experiments measure serialized vs pipelined nano-batch execution.
- **Adaptive control:** the paper's AIMD nano-batch rule is evaluated first synthetically and then against real two-GPU timings.
- **Scheduler reproduction:** urgency, residual capacity, bounded slowdown, hierarchical grouping, and binary-cut search are implemented with clearly labeled surrogate performance models.
- **Real trace input:** the final simulator consumes public ACMETrace arrivals, durations, and GPU allocations.

## Reproduction status

| Area | Status | Notes |
| --- | --- | --- |
| Shared Super-Model semantics | Reproduced | Shared frozen backbone, heterogeneous adapters, routing, gradients, optimizer isolation |
| CPU execution diagnostics | Reproduced | Shared-backbone benefit and Python routing overhead isolated |
| Nano-batching | Reproduced | Heterogeneous packed jobs and paper-aligned `N = number of nano-batches` semantics |
| AIMD controller | Reproduced | `alpha=4`, `beta=0.5`; configurable stability margin |
| CUDA baseline | Reproduced | Independent, grouped, and shared execution on NVIDIA T4 |
| Triton LoRA forward | Reproduced | Two-stage and single-kernel LoRA paths with correctness checks |
| Training semantics | Reproduced | Triton forward + analytical PyTorch backward; gradient/update equivalence |
| Two-GPU overlap | Reduced reproduction | Controlled NCCL microbenchmark, not full FSDP/Megatron execution |
| Residual-capacity scheduler | Reduced reproduction | Scheduling logic reproduced; throughput predictor is a documented surrogate |
| Hierarchical/binary-cut scheduler | Reduced reproduction | Topology hierarchy is an executable abstraction; not the authors' private planner |
| ACMETrace replay | Reduced reproduction | Real trace data + sampled LoRA attributes + surrogate relative performance |
| Paper headline A100/128-GPU results | Not reproduced | Requires the authors' measured A100 profile database and production-scale simulator |

## Representative results

These are measurements from the development runs used to validate this repository. Treat timings as hardware- and environment-specific.

| Experiment | Result |
| --- | --- |
| Phase 1 correctness | Forward max error `1.192e-07`; adapter gradient errors `0`; optimizer isolation passed |
| Phase 2 parameter storage | Shared SSM reduced model parameter storage by `72.87%` vs four independent models |
| Phase 2B routing diagnostic | One combined backbone call was `1.578x` faster than four separate calls; contiguous routing was `1.778x` faster than boolean-mask routing |
| Phase 5C, Tesla T4 | PyTorch Shared SSM `0.5449 ms` → fused Triton SSM `0.4637 ms` (`1.175x` time ratio) |
| Phase 5D training semantics | `dX`, `dA`, `dB`, and one-step SGD update matched the PyTorch reference in the validation run |
| Phase 6A, T4×2 | Best same-`N` pipelining reduction: `12.41%` at `N=8`; lowest measured pipelined time in that sweep: `4.518 ms` at `N=2` |
| Phase 6B, T4×2 | AIMD best observed `N=4`, `4.125 ms`, `11.76%` below the initial `N=1` horizon |
| Phase 7A | `1.196x` **surrogate** throughput vs standalone, with zero slowdown-bound violations |
| Phase 7B | 32 jobs → 10 groups, `1.106x` **surrogate** throughput, zero slowdown-bound violations |
| Phase 8 | Real ACMETrace replay completed; the surrogate model did **not** reproduce the paper's end-to-end headline gains |

Two important interpretation notes:

1. In Phase 5C, the single fused Triton adapter kernel beat the PyTorch adapter path, but the two-stage Triton adapter path was slightly faster than the single-kernel version in that isolated microbenchmark. The useful result is the measured end-to-end Triton-vs-PyTorch improvement, not a blanket claim that one-kernel fusion always wins.
2. Phase 7 and Phase 8 throughput values come from this repository's documented surrogate predictor. They are **not** the paper's measured A100 results.

Selected captured outputs are stored in [`results/`](results/).

## Project layout

```text
.
├── tlora/
│   ├── __init__.py
│   └── multi_lora_linear.py
├── tests/
│   └── test_equivalence.py
├── results/
├── run_demo.py
├── benchmark_phase2.py
├── benchmark_phase2b.py
├── phase3_nano_batch.py
├── phase4_aimd.py
├── phase5_gpu_baseline.py
├── phase5b_triton_forward.py
├── phase5c_fused_triton.py
├── phase5d_triton_training.py
├── phase6_two_gpu_overlap.py
├── phase6b_real_aimd.py
├── phase7a_adapter_scheduler.py
├── phase7b_hierarchical_scheduler.py
├── phase8_acmetrace_sim.py
├── requirements.txt
└── README.md
```

The core reference primitive lives in [`tlora/multi_lora_linear.py`](tlora/multi_lora_linear.py). It defines an independent LoRA linear layer and the shared `MultiLoRALinear` correctness baseline.

## Requirements

### CPU phases

- Python 3.10+
- PyTorch 2.5+
- pytest 8+

Install the repository requirements with:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU phases

Phases 5 and 6 additionally require:

- Linux
- NVIDIA CUDA-capable GPU
- a PyTorch CUDA build
- Triton
- NCCL for the two-GPU experiments

The GPU experiments in this repository were validated in a Kaggle environment with:

```text
PyTorch 2.10.0+cu128
Triton 3.6.0
Tesla T4
Compute capability 7.5
```

For the distributed Phase 6 experiments, two T4 GPUs were used.

`requirements.txt` intentionally contains only the portable baseline dependencies. CUDA, Triton, and NCCL should be installed as a mutually compatible stack for the target Linux environment.

## Getting started

Clone the repository:

```bash
git clone https://github.com/GOPIKA-SUSHAMA/mini-tlora-reproduction.git
cd mini-tlora-reproduction
```

Create a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the smallest demo:

```bash
python run_demo.py
```

Run the correctness tests:

```bash
python -m pytest -s
```

The tests verify forward equivalence, adapter-gradient equivalence, and optimizer isolation against independently instantiated LoRA jobs.

## Running the experiments

### CPU progression

Start with the baseline and follow the phases in order:

```bash
python benchmark_phase2.py
python benchmark_phase2b.py
python phase3_nano_batch.py
python phase4_aimd.py
```

The scheduler phases are also CPU-only:

```bash
python phase7a_adapter_scheduler.py
python phase7b_hierarchical_scheduler.py
```

### Single-GPU Triton experiments

Run these in a Linux CUDA environment:

```bash
python phase5_gpu_baseline.py
python phase5b_triton_forward.py
python phase5c_fused_triton.py
python phase5d_triton_training.py
```

Phase 5D validates training semantics with a fused Triton forward and explicit analytical backward formulas implemented with PyTorch matrix operations. It is **not** a custom Triton backward implementation.

### Two-GPU overlap and AIMD

With exactly two visible NVIDIA GPUs:

```bash
torchrun --standalone --nproc_per_node=2 phase6_two_gpu_overlap.py
torchrun --standalone --nproc_per_node=2 phase6b_real_aimd.py
```

Phase 6A is a controlled intra-node NCCL experiment. It uses heterogeneous fused LoRA computation as the compute workload and asynchronous `all_reduce` calls to study nano-batch overlap. It does not attempt to reproduce the paper's full distributed training stack.

### ACMETrace-driven simulation

Phase 8 expects a public ACMETrace job CSV with fields including `job_id`, `gpu_num`, `state`, `submit_time`, and `duration`.

One convenient layout is:

```text
parent/
├── mini-tlora-reproduction/
└── AcmeTrace/
```

Clone ACMETrace beside this repository:

```bash
cd ..
git clone https://github.com/InternLM/AcmeTrace.git
cd mini-tlora-reproduction
```

Run the simulator:

```bash
python phase8_acmetrace_sim.py \
  --trace ../AcmeTrace/data/job_trace/trace_seren.csv \
  --jobs 500
```

The simulator compares independent execution, FIFO grouping, and the tLoRA-inspired scheduler at multiple arrival-rate multipliers.

## Experiment map

| Phase | Entry point | Purpose | Hardware |
| --- | --- | --- | --- |
| 1 | [`run_demo.py`](run_demo.py), [`tests/test_equivalence.py`](tests/test_equivalence.py) | Shared Super-Model correctness | CPU |
| 2 | [`benchmark_phase2.py`](benchmark_phase2.py) | Independent vs grouped vs shared baseline | CPU |
| 2B | [`benchmark_phase2b.py`](benchmark_phase2b.py) | Separate backbone benefit from routing overhead | CPU |
| 3 | [`phase3_nano_batch.py`](phase3_nano_batch.py) | Heterogeneous packed jobs and chunked execution | CPU |
| 4 | [`phase4_aimd.py`](phase4_aimd.py) | Paper-aligned nano-batch count and AIMD controller | CPU |
| 5A | [`phase5_gpu_baseline.py`](phase5_gpu_baseline.py) | CUDA training-step baseline | 1× NVIDIA GPU |
| 5B | [`phase5b_triton_forward.py`](phase5b_triton_forward.py) | Two-stage heterogeneous Triton LoRA forward | 1× NVIDIA GPU |
| 5C | [`phase5c_fused_triton.py`](phase5c_fused_triton.py) | Single-kernel LoRA path with Triton autotuning | 1× NVIDIA GPU |
| 5D | [`phase5d_triton_training.py`](phase5d_triton_training.py) | Training-gradient and update equivalence | 1× NVIDIA GPU |
| 6A | [`phase6_two_gpu_overlap.py`](phase6_two_gpu_overlap.py) | Serialized vs pipelined NCCL nano-batches | 2× NVIDIA GPU |
| 6B | [`phase6b_real_aimd.py`](phase6b_real_aimd.py) | AIMD driven by real two-GPU timing | 2× NVIDIA GPU |
| 7A | [`phase7a_adapter_scheduler.py`](phase7a_adapter_scheduler.py) | Urgency/residual-capacity grouping | CPU |
| 7B | [`phase7b_hierarchical_scheduler.py`](phase7b_hierarchical_scheduler.py) | Hierarchical binary-cut scheduling | CPU |
| 8 | [`phase8_acmetrace_sim.py`](phase8_acmetrace_sim.py) | Trace-driven online scheduling simulation | CPU |

## Methodology and limitations

This repository deliberately separates **measured reproduction** from **surrogate simulation**.

The original paper reports results from a substantially larger environment. This project does not have access to the authors' per-job A100 speed-profile database or their production-scale distributed simulator. As a result:

- Phase 1–6 focus on executable correctness and systems mechanisms that can be measured directly on CPU/T4 hardware.
- Phase 7 reproduces scheduling logic but uses a local surrogate throughput predictor based on compute/memory headroom.
- Phase 7B's scaling experiment demonstrates pruning behavior, but should not be presented as empirical proof of strict `O(K log K)` complexity; Python prefix construction and repeated surrogate evaluation add extra work.
- Phase 8 uses real ACMETrace arrivals, durations, and GPU allocations, but samples the LoRA-specific attributes and uses the Phase 7 surrogate for relative grouped performance.
- Phase 8's `allocation_util` metric is allocated GPU-time occupancy, **not** GPU SM utilization.
- The paper's reported `1.2–1.8x` throughput, `2.3–5.4x` JCT, and GPU-utilization improvements are paper results, not results reproduced by this repository.

The goal is to make every approximation visible instead of tuning the simulator until it matches the paper.

## Testing

Run the repository tests with:

```bash
python -m pytest -s
```

The current test suite is intentionally focused on the correctness foundation in [`tests/test_equivalence.py`](tests/test_equivalence.py). Later phases include additional runtime assertions and equivalence checks inside their experiment scripts.

When modifying kernels or scheduling logic, run the relevant phase script as well as the base test suite.

## Getting help

For repository-specific bugs, reproducibility questions, or experiment discrepancies, open a GitHub issue:

- https://github.com/GOPIKA-SUSHAMA/mini-tlora-reproduction/issues

Useful references:

- Original paper: https://arxiv.org/abs/2602.07263
- Public ACMETrace repository: https://github.com/InternLM/AcmeTrace
- PyTorch documentation: https://pytorch.org/docs/stable/
- Triton documentation: https://triton-lang.org/main/index.html
- Captured project outputs: [`results/`](results/)

When reporting a GPU issue, include the output of:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

and, for Triton phases:

```bash
python -c "import triton; print(triton.__version__)"
```

## Maintainer and contributions

This repository is maintained by **[@GOPIKA-SUSHAMA](https://github.com/GOPIKA-SUSHAMA)**.

Contributions that improve correctness, reproducibility, benchmarking discipline, kernel quality, documentation, or scheduler fidelity are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

For research-result changes, keep the distinction between measured results, surrogate results, and paper-reported results explicit.

## License

This repository does not currently include a `LICENSE` file. If you plan to reuse or redistribute the code, contact the maintainer or open an issue before doing so.
