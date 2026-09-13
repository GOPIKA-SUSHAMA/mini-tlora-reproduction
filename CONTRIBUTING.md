# Contributing

Thanks for contributing to `mini-tlora-reproduction`.

## Development setup

Create a virtual environment and install the portable dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Activate the environment using the command appropriate for your platform, then run:

```bash
python -m pytest -s
```

GPU changes should be tested in a Linux CUDA environment with a compatible PyTorch/Triton stack. Phase 6 changes require two GPUs with NCCL support.

## Pull requests

Keep pull requests focused and include:

- a concise description of the problem and approach;
- the phase(s) affected;
- commands used to validate the change;
- relevant correctness or benchmark output;
- hardware/software details for GPU performance claims.

Do not mix unrelated refactors with benchmark-result changes.

## Reproduction discipline

This project distinguishes three kinds of evidence:

1. **Measured reproduction** — directly executed on the stated hardware.
2. **Reduced/surrogate reproduction** — the mechanism is implemented, but scale, topology, or performance inputs differ from the paper.
3. **Paper-reported result** — a number quoted from the original paper and not reproduced here.

Do not present surrogate scheduler output as a measured tLoRA result. Do not tune Phase 7/8 surrogate coefficients solely to match the paper's headline numbers.

When adding a performance result, record enough information to make it interpretable: device, PyTorch version, CUDA runtime, Triton version where relevant, workload shape, warmup, and iteration count.

## Code style

Prefer readable, explicit experiment code over premature abstraction. Preserve deterministic seeds where they are part of an existing experiment, and keep limitations close to the code that introduces them.

## Issues

Use GitHub Issues for bugs, reproducibility questions, and proposed experimental extensions:

https://github.com/GOPIKA-SUSHAMA/mini-tlora-reproduction/issues
