import argparse
import math
import os

import torch
import torch.distributed as dist

from phase6_two_gpu_overlap import (
    DEFAULT_DEPTH,
    DEFAULT_MULTIPLIER,
    allocate_chunk_buffers,
    make_large_packed_batch,
    measure_distributed,
    run_pipelined,
    split_ranges,
)
from phase5c_fused_triton import build_model, pack_weights


# =============================================================================
# Phase 6B
# Real two-GPU AIMD control over the pipelined nano-batch execution measured
# in Phase 6A.
#
# Paper update rule:
#
#   N_{t+1} = N_t + alpha
#       if T_t <= T_{t-1} - tau
#
#   N_{t+1} = max(1, floor(beta * N_t))
#       otherwise
#
# Defaults from tLoRA:
#   alpha = 4
#   beta  = 1/2
#
# The paper leaves tau as a stability margin rather than a universal constant.
# We therefore expose tau as a CLI parameter.
# =============================================================================


def init_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # device_id avoids NCCL barrier/device-guess warnings on current PyTorch.
    dist.init_process_group(
        backend="nccl",
        device_id=device,
    )

    return rank, local_rank, world_size, device


class AIMDController:
    def __init__(
        self,
        start_n=1,
        alpha=4,
        beta=0.5,
        tau_ms=0.05,
        max_n=64,
    ):
        self.n = max(1, int(start_n))
        self.alpha = int(alpha)
        self.beta = float(beta)
        self.tau_ms = float(tau_ms)
        self.max_n = max(1, int(max_n))

        self.prev_ms = None

        self.best_n = self.n
        self.best_ms = float("inf")

    def observe(self, current_ms):
        current_ms = float(current_ms)
        n_used = self.n

        if current_ms < self.best_ms:
            self.best_ms = current_ms
            self.best_n = n_used

        # Bootstrap exactly as in our Phase 4 reproduction:
        # one conservative N=1 measurement, followed by an additive probe.
        if self.prev_ms is None:
            self.prev_ms = current_ms
            self.n = min(
                self.max_n,
                self.n + self.alpha,
            )
            return self.n, "bootstrap additive probe"

        improved = (
            current_ms
            <= self.prev_ms - self.tau_ms
        )

        if improved:
            next_n = self.n + self.alpha
            decision = "additive increase"
        else:
            next_n = max(
                1,
                math.floor(
                    self.beta * self.n
                ),
            )
            decision = "multiplicative decrease"

        self.prev_ms = current_ms
        self.n = max(
            1,
            min(
                self.max_n,
                next_n,
            ),
        )

        return self.n, decision


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--multiplier",
        type=int,
        default=DEFAULT_MULTIPLIER,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
    )

    parser.add_argument(
        "--horizons",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--iterations-per-horizon",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--tau-ms",
        type=float,
        default=0.05,
        help=(
            "AIMD stability margin tau in milliseconds. "
            "The paper defines tau but does not prescribe one universal value."
        ),
    )

    parser.add_argument(
        "--max-n",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    rank, local_rank, world_size, device = init_dist()

    if world_size != 2:
        raise RuntimeError(
            f"Phase 6B expects exactly 2 GPUs, got {world_size}"
        )

    torch.manual_seed(2026 + rank)
    torch.cuda.manual_seed_all(2026 + rank)

    model = build_model(device)

    A, B, ranks, scales = pack_weights(
        model,
        device,
    )

    x, adapter_ids = make_large_packed_batch(
        device,
        args.multiplier,
    )

    total_samples = x.shape[0]

    controller = AIMDController(
        start_n=1,
        alpha=4,
        beta=0.5,
        tau_ms=args.tau_ms,
        max_n=min(
            args.max_n,
            total_samples,
        ),
    )

    if rank == 0:
        print("=" * 84)
        print("MINI-tLoRA PHASE 6B - REAL TWO-GPU AIMD NANO-BATCH CONTROL")
        print("=" * 84)

        print(f"World size              : {world_size}")
        print(f"GPU 0 / GPU 1           : Tesla T4 / Tesla T4")
        print(f"Local samples/GPU       : {total_samples}")
        print(f"Compute depth           : {args.depth}")
        print(f"Scheduling horizons     : {args.horizons}")
        print(
            f"Iterations/horizon      : "
            f"{args.iterations_per_horizon}"
        )
        print("Paper alpha             : 4")
        print("Paper beta              : 0.5")
        print(
            f"Local tau               : "
            f"{args.tau_ms:.3f} ms"
        )

        print("\nAIMD TRACE")
        print("-" * 84)
        print(
            f"{'Horizon':>8s}"
            f"{'N used':>10s}"
            f"{'Mean ms':>14s}"
            f"{'Median ms':>14s}"
            f"{'Next N':>10s}  Decision"
        )

    history = []

    for horizon in range(args.horizons):
        n_used = controller.n

        ranges = split_ranges(
            total_samples,
            n_used,
        )

        chunks = allocate_chunk_buffers(
            x,
            adapter_ids,
            ranges,
        )

        result = measure_distributed(
            lambda: run_pipelined(
                chunks,
                A,
                B,
                ranks,
                scales,
                args.depth,
            ),
            device,
            args.warmup,
            args.iterations_per_horizon,
        )

        # The paper defines T_t as end-to-end batch completion time.
        # We use the distributed mean of our repeated iterations as the
        # horizon feedback signal; median is printed as a noise diagnostic.
        feedback_ms = result["mean"]

        next_n, decision = controller.observe(
            feedback_ms
        )

        history.append(
            {
                "horizon": horizon,
                "n": n_used,
                "mean": result["mean"],
                "median": result["median"],
                "stdev": result["stdev"],
                "next_n": next_n,
                "decision": decision,
            }
        )

        if rank == 0:
            print(
                f"{horizon:8d}"
                f"{n_used:10d}"
                f"{result['mean']:14.3f}"
                f"{result['median']:14.3f}"
                f"{next_n:10d}  "
                f"{decision}"
            )

    if rank == 0:
        print("\nBEST OBSERVED CONFIGURATION")
        print("-" * 84)

        print(
            f"N={controller.best_n}, "
            f"mean iteration time="
            f"{controller.best_ms:.3f} ms"
        )

        first = history[0]
        best_reduction = (
            100.0
            * (
                1.0
                - controller.best_ms
                / first["mean"]
            )
        )

        print(
            f"Relative to initial N={first['n']}: "
            f"{best_reduction:.2f}% lower mean iteration time"
        )

        print("\nTRACE SUMMARY")
        print("-" * 84)

        visited = " -> ".join(
            str(row["n"])
            for row in history
        )

        print(f"Visited N values: {visited}")

        print("\nINTERPRETATION")
        print("-" * 84)

        print(
            "This phase connects the paper's AIMD rule to real two-GPU NCCL\n"
            "execution rather than the synthetic/CPU feedback used in Phase 4.\n"
            "Because the rule compares consecutive horizons, noisy timings can\n"
            "produce oscillation; tau controls how much improvement must be\n"
            "observed before increasing N. We report the full trace rather than\n"
            "claiming that the final N is necessarily the global optimum."
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
