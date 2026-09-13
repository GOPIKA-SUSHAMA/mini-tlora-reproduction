import copy
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tlora.multi_lora_linear import MultiLoRALinear


@dataclass
class JobConfig:
    adapter_id: int
    rank: int
    batch_size: int


JOBS = [
    JobConfig(adapter_id=0, rank=2, batch_size=8),
    JobConfig(adapter_id=1, rank=4, batch_size=16),
    JobConfig(adapter_id=2, rank=8, batch_size=4),
    JobConfig(adapter_id=3, rank=16, batch_size=32),
]

IN_FEATURES = 512
OUT_FEATURES = 512
WARMUP = 5
ITERS = 30


def make_jobs():
    xs = {}
    ys = {}

    for job in JOBS:
        xs[job.adapter_id] = torch.randn(job.batch_size, IN_FEATURES)
        ys[job.adapter_id] = torch.randn(job.batch_size, OUT_FEATURES)

    return xs, ys


def pack_jobs(xs, ys):
    """
    Pack heterogeneous jobs contiguously:

    [job0 samples][job1 samples][job2 samples][job3 samples]

    Returns:
      x_all
      y_all
      spans: adapter_id -> (start, end)
    """
    x_parts = []
    y_parts = []
    spans = {}

    cursor = 0

    for job in JOBS:
        aid = job.adapter_id
        x_parts.append(xs[aid])
        y_parts.append(ys[aid])

        start = cursor
        end = start + job.batch_size
        spans[aid] = (start, end)

        cursor = end

    return torch.cat(x_parts, dim=0), torch.cat(y_parts, dim=0), spans


def build_model():
    ranks = {job.adapter_id: job.rank for job in JOBS}

    model = MultiLoRALinear(
        IN_FEATURES,
        OUT_FEATURES,
        ranks,
    )

    # Make LoRA deltas non-zero.
    for adapter in model.adapters.values():
        torch.nn.init.normal_(adapter.B, mean=0.0, std=0.01)

    return model


def build_opts(model, lr=1e-5):
    return {
        job.adapter_id: torch.optim.SGD(
            model.adapter_parameters(job.adapter_id),
            lr=lr,
        )
        for job in JOBS
    }


def zero_opts(opts):
    for opt in opts.values():
        opt.zero_grad(set_to_none=True)


def step_opts(opts):
    for opt in opts.values():
        opt.step()


def grouped_forward(model, x_all, spans):
    """
    One combined frozen-backbone call.
    Each heterogeneous job then applies its LoRA branch to its contiguous slice.
    """
    out = model.base(x_all)
    result = out.clone()

    for job in JOBS:
        aid = job.adapter_id
        start, end = spans[aid]

        result[start:end] = (
            result[start:end]
            + model.adapters[str(aid)](x_all[start:end])
        )

    return result


def nano_forward(model, x_all, spans, nano_batch_size):
    """
    Correctness-first nano-batch abstraction.

    Backbone still executes once on the combined packed batch.
    Adapter work is broken into small contiguous nano-batches.

    This does NOT yet reproduce GPU compute/communication overlap.
    That comes later when we move to free GPU + Triton/distributed execution.
    """
    out = model.base(x_all)
    result = out.clone()

    for job in JOBS:
        aid = job.adapter_id
        start, end = spans[aid]

        cursor = start

        while cursor < end:
            nano_end = min(cursor + nano_batch_size, end)

            result[cursor:nano_end] = (
                result[cursor:nano_end]
                + model.adapters[str(aid)](
                    x_all[cursor:nano_end]
                )
            )

            cursor = nano_end

    return result


def train_step_grouped(model, opts, x_all, y_all, spans):
    zero_opts(opts)

    pred = grouped_forward(
        model,
        x_all,
        spans,
    )

    loss = F.mse_loss(
        pred,
        y_all,
        reduction="sum",
    )

    loss.backward()
    step_opts(opts)

    return float(loss.detach())


def train_step_nano(
    model,
    opts,
    x_all,
    y_all,
    spans,
    nano_batch_size,
):
    zero_opts(opts)

    pred = nano_forward(
        model,
        x_all,
        spans,
        nano_batch_size,
    )

    loss = F.mse_loss(
        pred,
        y_all,
        reduction="sum",
    )

    loss.backward()
    step_opts(opts)

    return float(loss.detach())


def benchmark(fn):
    for _ in range(WARMUP):
        fn()

    times = []

    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
    }


def gradient_snapshot(model):
    snap = {}

    for job in JOBS:
        aid = job.adapter_id
        adapter = model.adapters[str(aid)]

        snap[aid] = {
            "A": adapter.A.grad.detach().clone(),
            "B": adapter.B.grad.detach().clone(),
        }

    return snap


def check_gradient_equivalence(
    reference_model,
    nano_model,
    x_all,
    y_all,
    spans,
    nano_batch_size,
):
    # Reference grouped gradients.
    reference_model.zero_grad(set_to_none=True)

    ref_pred = grouped_forward(
        reference_model,
        x_all,
        spans,
    )

    ref_loss = F.mse_loss(
        ref_pred,
        y_all,
        reduction="sum",
    )

    ref_loss.backward()

    ref_grads = gradient_snapshot(reference_model)

    # Nano-batched gradients.
    nano_model.zero_grad(set_to_none=True)

    nano_pred = nano_forward(
        nano_model,
        x_all,
        spans,
        nano_batch_size,
    )

    nano_loss = F.mse_loss(
        nano_pred,
        y_all,
        reduction="sum",
    )

    nano_loss.backward()

    nano_grads = gradient_snapshot(nano_model)

    max_error = 0.0

    for job in JOBS:
        aid = job.adapter_id

        for name in ("A", "B"):
            err = (
                ref_grads[aid][name]
                - nano_grads[aid][name]
            ).abs().max().item()

            max_error = max(max_error, err)

    return max_error


def main():
    torch.manual_seed(2026)

    xs, ys = make_jobs()
    x_all, y_all, spans = pack_jobs(xs, ys)

    seed_model = build_model()

    print("=" * 78)
    print("MINI-tLoRA PHASE 3 - HETEROGENEOUS JOBS + NANO-BATCHING")
    print("=" * 78)

    print("\nJOB CONFIGURATION")
    print("-" * 78)

    total = 0

    for job in JOBS:
        print(
            f"Job {job.adapter_id}: "
            f"rank={job.rank:2d}, "
            f"batch={job.batch_size:2d}, "
            f"span={spans[job.adapter_id]}"
        )
        total += job.batch_size

    print(f"Total packed samples: {total}")

    # ---------------------------------------------------------
    # 1. Verify forward equivalence for several nano-batch sizes.
    # ---------------------------------------------------------
    print("\nFORWARD EQUIVALENCE")
    print("-" * 78)

    reference = grouped_forward(
        seed_model,
        x_all,
        spans,
    )

    nano_sizes = [1, 2, 4, 8, 16, 32]

    for nano_size in nano_sizes:
        out = nano_forward(
            seed_model,
            x_all,
            spans,
            nano_size,
        )

        max_error = (
            reference - out
        ).abs().max().item()

        print(
            f"nano_batch={nano_size:2d} "
            f"max_abs_error={max_error:.3e}"
        )

        assert torch.allclose(
            reference,
            out,
            atol=1e-6,
            rtol=1e-6,
        )

    # ---------------------------------------------------------
    # 2. Verify gradients are preserved by nano-batching.
    # ---------------------------------------------------------
    print("\nGRADIENT EQUIVALENCE")
    print("-" * 78)
    print("Float32 acceptance tolerance: max_grad_error <= 2.0e-05")

    for nano_size in [1, 4, 8, 16]:
        ref_model = copy.deepcopy(seed_model)
        nano_model = copy.deepcopy(seed_model)

        grad_error = check_gradient_equivalence(
            ref_model,
            nano_model,
            x_all,
            y_all,
            spans,
            nano_size,
        )

        print(
            f"nano_batch={nano_size:2d} "
            f"max_grad_error={grad_error:.3e}"
        )

        # Different nano-batch sizes change floating-point accumulation order.
        # For float32 CPU matmuls, tiny gradient differences around 1e-5 are
        # expected even when the computation is mathematically equivalent.
        GRAD_ATOL = 2e-5
        assert grad_error <= GRAD_ATOL, (
            f"gradient mismatch too large: {grad_error:.3e} > {GRAD_ATOL:.1e}"
        )

    # ---------------------------------------------------------
    # 3. CPU timing exploration.
    # ---------------------------------------------------------
    print("\nCPU TRAINING-STEP TIMINGS")
    print("-" * 78)

    grouped_model = copy.deepcopy(seed_model)
    grouped_opts = build_opts(grouped_model)

    grouped_result = benchmark(
        lambda: train_step_grouped(
            grouped_model,
            grouped_opts,
            x_all,
            y_all,
            spans,
        )
    )

    print(
        f"{'Execution':20s}"
        f"{'Mean ms':>12s}"
        f"{'Median ms':>14s}"
    )

    print(
        f"{'Grouped full-job':20s}"
        f"{grouped_result['mean']*1000:12.3f}"
        f"{grouped_result['median']*1000:14.3f}"
    )

    for nano_size in nano_sizes:
        model = copy.deepcopy(seed_model)
        opts = build_opts(model)

        result = benchmark(
            lambda ns=nano_size, m=model, o=opts:
            train_step_nano(
                m,
                o,
                x_all,
                y_all,
                spans,
                ns,
            )
        )

        print(
            f"{('Nano=' + str(nano_size)):20s}"
            f"{result['mean']*1000:12.3f}"
            f"{result['median']*1000:14.3f}"
        )

    print("\nINTERPRETATION")
    print("-" * 78)
    print(
        "This phase proves that heterogeneous jobs with different ranks and\n"
        "batch sizes can be packed contiguously and split into nano-batches\n"
        "without changing outputs or LoRA gradients.\n\n"
        "On CPU, small nano-batches may be slower because they create more\n"
        "Python/kernel-call overhead. That is expected. In the paper, the\n"
        "purpose of nano-batching is to enable GPU compute/communication\n"
        "overlap, which we will test later on free GPU hardware."
    )


if __name__ == "__main__":
    main()
