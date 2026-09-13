import argparse
import copy
import math
import statistics
import time

import torch
import torch.nn.functional as F

from phase3_nano_batch import (
    JOBS,
    IN_FEATURES,
    OUT_FEATURES,
    build_model,
    build_opts,
    grouped_forward,
    make_jobs,
    pack_jobs,
)


def split_into_n(total_samples: int, n: int):
    """Return exactly n near-equal contiguous [start, end) ranges."""
    n = max(1, min(int(n), total_samples))
    q, r = divmod(total_samples, n)

    ranges = []
    start = 0

    for i in range(n):
        size = q + (1 if i < r else 0)
        end = start + size
        ranges.append((start, end))
        start = end

    return ranges


def nano_forward_by_count(model, x_all, spans, n_nano):
    """
    Paper-aligned abstraction:
      N = NUMBER of nano-batches, not samples per nano-batch.

    The combined packed batch is split into N near-equal contiguous pieces.
    Each piece executes backbone + the adapter slices that overlap it.
    """
    total = x_all.shape[0]
    ranges = split_into_n(total, n_nano)

    outputs = []

    for nano_start, nano_end in ranges:
        x_nano = x_all[nano_start:nano_end]
        out_nano = model.base(x_nano)
        result_nano = out_nano.clone()

        # Apply every adapter whose packed span overlaps this nano-batch.
        for job in JOBS:
            aid = job.adapter_id
            job_start, job_end = spans[aid]

            overlap_start = max(nano_start, job_start)
            overlap_end = min(nano_end, job_end)

            if overlap_start >= overlap_end:
                continue

            local_start = overlap_start - nano_start
            local_end = overlap_end - nano_start

            x_slice = x_all[overlap_start:overlap_end]

            result_nano[local_start:local_end] = (
                result_nano[local_start:local_end]
                + model.adapters[str(aid)](x_slice)
            )

        outputs.append(result_nano)

    return torch.cat(outputs, dim=0)


def gradient_snapshot(model):
    snap = {}
    for job in JOBS:
        adapter = model.adapters[str(job.adapter_id)]
        snap[job.adapter_id] = {
            "A": adapter.A.grad.detach().clone(),
            "B": adapter.B.grad.detach().clone(),
        }
    return snap


def correctness_check(seed_model, x_all, y_all, spans):
    print("\nCORRECTNESS: N MEANS NUMBER OF NANO-BATCHES")
    print("-" * 78)

    reference = grouped_forward(seed_model, x_all, spans)

    candidates = [1, 2, 4, 5, 8, 9, 16, 32]
    candidates = [n for n in candidates if n <= x_all.shape[0]]

    for n in candidates:
        out = nano_forward_by_count(seed_model, x_all, spans, n)
        err = (reference - out).abs().max().item()

        print(f"N={n:2d} forward max_abs_error={err:.3e}")
        assert torch.allclose(reference, out, atol=1e-6, rtol=1e-6)

    # One gradient check at a non-trivial N.
    n = min(9, x_all.shape[0])

    ref_model = copy.deepcopy(seed_model)
    nano_model = copy.deepcopy(seed_model)

    ref_model.zero_grad(set_to_none=True)
    ref_pred = grouped_forward(ref_model, x_all, spans)
    ref_loss = F.mse_loss(ref_pred, y_all, reduction="sum")
    ref_loss.backward()
    ref_grads = gradient_snapshot(ref_model)

    nano_model.zero_grad(set_to_none=True)
    nano_pred = nano_forward_by_count(nano_model, x_all, spans, n)
    nano_loss = F.mse_loss(nano_pred, y_all, reduction="sum")
    nano_loss.backward()
    nano_grads = gradient_snapshot(nano_model)

    max_grad_error = 0.0

    for job in JOBS:
        aid = job.adapter_id
        for name in ("A", "B"):
            err = (
                ref_grads[aid][name] - nano_grads[aid][name]
            ).abs().max().item()
            max_grad_error = max(max_grad_error, err)

    print(f"N={n:2d} gradient max_abs_error={max_grad_error:.3e}")

    # Different nano-batch counts change floating-point accumulation order.
    # For float32 CPU matmuls, tiny gradient differences around 1e-5 are
    # expected even when the computation is mathematically equivalent.
    GRAD_ATOL = 2e-5
    print(f"Gradient acceptance tolerance: {GRAD_ATOL:.1e}")
    assert max_grad_error <= GRAD_ATOL, (
        f"gradient mismatch too large: {max_grad_error:.3e} > {GRAD_ATOL:.1e}"
    )


class AIMDController:
    """
    Implements Eq. (2) from tLoRA:

      N_{t+1} = N_t + alpha,
          if T_t <= T_{t-1} - tau

      N_{t+1} = max(1, floor(beta * N_t)),
          otherwise

    alpha=4 and beta=1/2 are the paper's defaults.

    The paper does not specify a universal numeric tau; it is a stability
    margin, so this reproduction exposes tau as a configurable parameter.

    Bootstrap:
      after the first timing measurement, we perform one additive probe so
      there is a second point to compare against.
    """

    def __init__(
        self,
        start_n=1,
        alpha=4,
        beta=0.5,
        tau_seconds=0.00002,
        max_n=60,
    ):
        self.n = max(1, int(start_n))
        self.alpha = int(alpha)
        self.beta = float(beta)
        self.tau = float(tau_seconds)
        self.max_n = max(1, int(max_n))

        self.previous_time = None
        self.bootstrap_done = False

        self.best_n = self.n
        self.best_time = float("inf")

    def observe(self, current_time):
        current_time = float(current_time)

        if current_time < self.best_time:
            self.best_time = current_time
            self.best_n = self.n

        if self.previous_time is None:
            self.previous_time = current_time
            self.n = min(self.max_n, self.n + self.alpha)
            self.bootstrap_done = True
            return self.n, "bootstrap additive probe"

        improved = current_time <= (self.previous_time - self.tau)

        if improved:
            next_n = self.n + self.alpha
            decision = "additive increase"
        else:
            next_n = max(1, math.floor(self.beta * self.n))
            decision = "multiplicative decrease"

        self.previous_time = current_time
        self.n = max(1, min(self.max_n, next_n))

        return self.n, decision


def synthetic_iteration_time(n):
    """
    Deterministic toy landscape used ONLY to verify controller behaviour.

    It is not the tLoRA simulator and is not a paper result.

    The curve rewards increased overlap initially, then penalizes excessive
    nano-batching through launch/fragmentation overhead.
    """
    base_ms = 5.0
    overlap_gain_ms = 2.2 * (1.0 - math.exp(-n / 5.0))
    overhead_ms = 0.035 * n
    return (base_ms - overlap_gain_ms + overhead_ms) / 1000.0


def synthetic_demo(horizons, tau_seconds):
    print("\nAIMD CONTROL TEST ON A SYNTHETIC U-SHAPED TIMING LANDSCAPE")
    print("-" * 78)
    print("This validates the controller only; these are NOT paper measurements.")

    controller = AIMDController(
        start_n=1,
        alpha=4,
        beta=0.5,
        tau_seconds=tau_seconds,
        max_n=60,
    )

    print(f"{'horizon':>8s} {'N':>5s} {'time ms':>12s}  decision")

    for horizon in range(horizons):
        n_used = controller.n
        t = synthetic_iteration_time(n_used)
        next_n, decision = controller.observe(t)

        print(
            f"{horizon:8d} {n_used:5d} {t*1000:12.4f}  "
            f"{decision} -> next N={next_n}"
        )

    print(
        f"Synthetic best observed: N={controller.best_n}, "
        f"time={controller.best_time*1000:.4f} ms"
    )


def timed_training_horizon(
    model,
    opts,
    x_all,
    y_all,
    spans,
    n_nano,
    repeats,
):
    times = []

    for _ in range(repeats):
        for opt in opts.values():
            opt.zero_grad(set_to_none=True)

        t0 = time.perf_counter()

        pred = nano_forward_by_count(
            model,
            x_all,
            spans,
            n_nano,
        )

        loss = F.mse_loss(pred, y_all, reduction="sum")
        loss.backward()

        for opt in opts.values():
            opt.step()

        times.append(time.perf_counter() - t0)

    # Median helps reduce CPU timing noise.
    return statistics.median(times)


def real_cpu_aimd(seed_model, x_all, y_all, spans, horizons, repeats, tau_seconds):
    print("\nAIMD ON THE REAL LOCAL CPU TRAINING LOOP")
    print("-" * 78)
    print(
        "CPU has no cross-GPU communication to overlap, so the selected N is\n"
        "a local baseline only. We do NOT expect it to match the paper's GPU N."
    )

    model = copy.deepcopy(seed_model)
    opts = build_opts(model)

    controller = AIMDController(
        start_n=1,
        alpha=4,
        beta=0.5,
        tau_seconds=tau_seconds,
        max_n=x_all.shape[0],
    )

    print(f"\n{'horizon':>8s} {'N':>5s} {'median ms':>12s}  decision")

    for horizon in range(horizons):
        n_used = controller.n

        t = timed_training_horizon(
            model,
            opts,
            x_all,
            y_all,
            spans,
            n_used,
            repeats,
        )

        next_n, decision = controller.observe(t)

        print(
            f"{horizon:8d} {n_used:5d} {t*1000:12.4f}  "
            f"{decision} -> next N={next_n}"
        )

    print(
        f"\nBest CPU N observed: {controller.best_n} "
        f"({controller.best_time*1000:.4f} ms median)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--tau-ms",
        type=float,
        default=0.02,
        help="stability margin tau in milliseconds",
    )
    args = parser.parse_args()

    torch.manual_seed(2026)

    xs, ys = make_jobs()
    x_all, y_all, spans = pack_jobs(xs, ys)
    seed_model = build_model()

    tau_seconds = args.tau_ms / 1000.0

    print("=" * 78)
    print("MINI-tLoRA PHASE 4 - PAPER-ALIGNED AIMD NANO-BATCH CONTROLLER")
    print("=" * 78)
    print(f"Combined batch samples : {x_all.shape[0]}")
    print("Paper defaults         : alpha=4, beta=0.5")
    print(f"Local tau              : {args.tau_ms:.3f} ms")
    print(
        "Important              : N is the NUMBER of nano-batches, "
        "not samples per nano-batch."
    )

    correctness_check(
        seed_model,
        x_all,
        y_all,
        spans,
    )

    synthetic_demo(
        horizons=args.horizons,
        tau_seconds=tau_seconds,
    )

    real_cpu_aimd(
        seed_model,
        x_all,
        y_all,
        spans,
        horizons=args.horizons,
        repeats=args.repeats,
        tau_seconds=tau_seconds,
    )


if __name__ == "__main__":
    main()
