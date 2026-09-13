# Mini-tLoRA — Phase 1

A correctness-first reproduction of the Shared Super-Model idea from the
`tLoRA: Efficient Multi-LoRA Training with Elastic Shared Super-Models` paper.

## What this phase reproduces

- One frozen shared backbone.
- Multiple LoRA branches.
- Heterogeneous LoRA ranks.
- Per-sample adapter/job routing.
- Independent gradients.
- Independent optimiser state / updates.
- Forward equivalence against separately instantiated LoRA jobs.

This deliberately does **not** optimise execution yet. The Python routing loop
is our correctness reference. A later phase replaces it with GPU/Triton fused
execution.

## Windows / PowerShell

```powershell
cd mini-tlora-phase1

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python run_demo.py
python -m pytest -s
```

Expected final line:

```text
ALL PHASE-1 TESTS PASSED
```

## What comes next

1. Add a microbenchmark: independent vs naive grouped vs shared execution.
2. Add heterogeneous batch sizes / job queues.
3. Add nano-batching.
4. Add AIMD controller.
5. Move to a free NVIDIA GPU notebook.
6. Replace the reference loop with a Triton fused LoRA kernel.
7. Add two-GPU scheduling experiments when free hardware permits.
