import argparse
import os
import statistics

import torch
import torch.distributed as dist

try:
    import triton
except Exception as exc:
    raise RuntimeError(f"Triton import failed: {exc}")

from phase5c_fused_triton import (
    IN_FEATURES,
    OUT_FEATURES,
    MAX_RANK,
    DTYPE,
    JOBS,
    build_model,
    pack_weights,
    fused_lora_residual_kernel,
)


# =============================================================================
# Phase 6A
# =============================================================================
#
# Goal:
#   Reproduce the paper's nano-batch compute/communication overlap mechanism
#   on the free two-GPU Kaggle T4 environment.
#
# This is a controlled microbenchmark, not full FSDP/Megatron:
#
#   compute      = heterogeneous fused Triton LoRA work
#   communication = NCCL all-reduce of the nano-batch output tensor
#
# Total local samples and therefore total communicated tensor elements remain
# constant across N. Increasing N only changes granularity / overlap.
#
# Compare:
#
#   SERIAL
#       compute(n) -> all_reduce(n) -> compute(n+1) -> ...
#
#   PIPELINED
#       compute(n) -> async all_reduce(n)
#                      overlaps with compute(n+1)
#
# =============================================================================


DEFAULT_MULTIPLIER = 4
DEFAULT_DEPTH = 8

N_VALUES = [1, 2, 4, 8, 16]


def init_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
    )

    return rank, local_rank, world_size


def split_ranges(total, n):
    n = min(max(1, n), total)

    q, r = divmod(total, n)

    ranges = []
    start = 0

    for i in range(n):
        size = q + (1 if i < r else 0)
        end = start + size
        ranges.append((start, end))
        start = end

    return ranges


def make_large_packed_batch(
    device,
    multiplier,
):
    """
    Preserve the Phase 5 adapter mixture while scaling total samples.

    Base Phase 5 batches:
        rank 2  -> 32
        rank 4  -> 64
        rank 8  -> 16
        rank 16 -> 128

    multiplier=4 gives local M=960 per GPU.
    """

    xs = []
    ids = []

    for aid, cfg in JOBS.items():
        batch = cfg["batch"] * multiplier

        xs.append(
            torch.randn(
                batch,
                IN_FEATURES,
                device=device,
                dtype=DTYPE,
            )
        )

        ids.append(
            torch.full(
                (batch,),
                aid,
                device=device,
                dtype=torch.int32,
            )
        )

    return (
        torch.cat(xs, dim=0).contiguous(),
        torch.cat(ids, dim=0).contiguous(),
    )


def fused_compute_once(
    x,
    adapter_ids,
    A,
    B,
    ranks,
    scales,
    out,
):
    M = x.shape[0]

    # Base pointer is ignored for ADD_BASE=False.
    fused_lora_residual_kernel[
        (M,)
    ](
        x,
        A,
        B,
        adapter_ids,
        ranks,
        scales,
        out,
        out,
        M=M,
        K=IN_FEATURES,
        N=OUT_FEATURES,
        RMAX=MAX_RANK,
        ADD_BASE=False,
    )


def compute_chunk(
    x,
    adapter_ids,
    A,
    B,
    ranks,
    scales,
    out,
    depth,
):
    """
    Repeat the fused LoRA compute to approximate multiple adapter-bearing
    layers. We intentionally reuse x; this is a controlled systems
    microbenchmark rather than a neural-network quality experiment.
    """

    for _ in range(depth):
        fused_compute_once(
            x,
            adapter_ids,
            A,
            B,
            ranks,
            scales,
            out,
        )


def allocate_chunk_buffers(
    x,
    adapter_ids,
    ranges,
):
    chunks = []

    for start, end in ranges:
        x_i = x[start:end]
        ids_i = adapter_ids[start:end]

        out_i = torch.empty(
            end - start,
            OUT_FEATURES,
            device=x.device,
            dtype=DTYPE,
        )

        chunks.append(
            (x_i, ids_i, out_i)
        )

    return chunks


def run_serial(
    chunks,
    A,
    B,
    ranks,
    scales,
    depth,
):
    for x_i, ids_i, out_i in chunks:
        compute_chunk(
            x_i,
            ids_i,
            A,
            B,
            ranks,
            scales,
            out_i,
            depth,
        )

        # Synchronous communication: no cross-nano-batch overlap.
        dist.all_reduce(
            out_i,
            op=dist.ReduceOp.SUM,
        )


def run_pipelined(
    chunks,
    A,
    B,
    ranks,
    scales,
    depth,
):
    works = []

    for x_i, ids_i, out_i in chunks:
        compute_chunk(
            x_i,
            ids_i,
            A,
            B,
            ranks,
            scales,
            out_i,
            depth,
        )

        # NCCL communication is enqueued asynchronously. The next chunk's
        # compute can proceed while this all-reduce is in flight.
        work = dist.all_reduce(
            out_i,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        works.append(work)

    for work in works:
        work.wait()


def measure_distributed(
    fn,
    device,
    warmup,
    iterations,
):
    for _ in range(warmup):
        dist.barrier()
        fn()

    torch.cuda.synchronize()
    dist.barrier()

    samples = []

    for _ in range(iterations):
        dist.barrier()
        torch.cuda.synchronize()

        start = torch.cuda.Event(
            enable_timing=True,
        )

        end = torch.cuda.Event(
            enable_timing=True,
        )

        start.record()

        fn()

        end.record()
        end.synchronize()

        elapsed = start.elapsed_time(end)

        # Distributed iteration time is determined by the slowest GPU.
        t = torch.tensor(
            [elapsed],
            device=device,
            dtype=torch.float32,
        )

        dist.all_reduce(
            t,
            op=dist.ReduceOp.MAX,
        )

        samples.append(
            float(t.item())
        )

    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": (
            statistics.stdev(samples)
            if len(samples) > 1
            else 0.0
        ),
    }


def bytes_per_iteration(total_samples):
    # Every sample communicates one OUT_FEATURES float16 vector.
    return (
        total_samples
        * OUT_FEATURES
        * torch.tensor(
            [],
            dtype=DTYPE,
        ).element_size()
    )


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
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    rank, local_rank, world_size = init_dist()

    if world_size != 2:
        raise RuntimeError(
            f"Phase 6A expects exactly 2 GPUs; got world_size={world_size}"
        )

    device = torch.device(
        f"cuda:{local_rank}"
    )

    torch.manual_seed(
        2026 + rank
    )

    torch.cuda.manual_seed_all(
        2026 + rank
    )

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

    # -------------------------------------------------------------------------
    # Environment summary
    # -------------------------------------------------------------------------

    if rank == 0:
        print("=" * 82)
        print("MINI-tLoRA PHASE 6A - TWO-GPU NANO-BATCH COMMUNICATION OVERLAP")
        print("=" * 82)

        print(f"PyTorch           : {torch.__version__}")
        print(f"Triton            : {triton.__version__}")
        print(f"NCCL              : {torch.cuda.nccl.version()}")
        print(f"World size        : {world_size}")
        print(f"Local samples/GPU : {total_samples}")
        print(f"Compute depth     : {args.depth}")
        print(
            f"Comm payload total: "
            f"{bytes_per_iteration(total_samples) / 1024**2:.2f} MiB/GPU/iteration"
        )

        for r in range(world_size):
            print(
                f"GPU {r}             : "
                f"{torch.cuda.get_device_name(r)}"
            )

        print("\nIMPORTANT")
        print("-" * 82)
        print(
            "This is a controlled intra-node NCCL microbenchmark. It reproduces\n"
            "the paper's nano-batch overlap mechanism, not its full Megatron/FSDP\n"
            "distributed model-parallel execution."
        )

    # Compile/warm the fused kernel once using full batch before timing.
    compile_out = torch.empty(
        total_samples,
        OUT_FEATURES,
        device=device,
        dtype=DTYPE,
    )

    fused_compute_once(
        x,
        adapter_ids,
        A,
        B,
        ranks,
        scales,
        compile_out,
    )

    torch.cuda.synchronize()
    dist.barrier()

    rows = []

    # -------------------------------------------------------------------------
    # Sweep nano-batch counts
    # -------------------------------------------------------------------------

    for n in N_VALUES:
        ranges = split_ranges(
            total_samples,
            n,
        )

        chunks_serial = allocate_chunk_buffers(
            x,
            adapter_ids,
            ranges,
        )

        chunks_pipe = allocate_chunk_buffers(
            x,
            adapter_ids,
            ranges,
        )

        serial = measure_distributed(
            lambda: run_serial(
                chunks_serial,
                A,
                B,
                ranks,
                scales,
                args.depth,
            ),
            device,
            args.warmup,
            args.iterations,
        )

        pipelined = measure_distributed(
            lambda: run_pipelined(
                chunks_pipe,
                A,
                B,
                ranks,
                scales,
                args.depth,
            ),
            device,
            args.warmup,
            args.iterations,
        )

        speedup = (
            serial["mean"]
            / pipelined["mean"]
        )

        overlap_reduction = (
            100.0
            * (
                1.0
                - pipelined["mean"]
                / serial["mean"]
            )
        )

        rows.append(
            (
                n,
                serial,
                pipelined,
                speedup,
                overlap_reduction,
            )
        )

        if rank == 0:
            print(
                f"N={n:2d} completed: "
                f"serial={serial['mean']:.3f} ms, "
                f"pipeline={pipelined['mean']:.3f} ms, "
                f"ratio={speedup:.3f}x"
            )

    # -------------------------------------------------------------------------
    # Final table
    # -------------------------------------------------------------------------

    if rank == 0:
        print("\nRESULTS")
        print("-" * 82)

        print(
            f"{'N':>4s}"
            f"{'Serial ms':>14s}"
            f"{'Pipeline ms':>16s}"
            f"{'Serial/Pipe':>16s}"
            f"{'Reduction %':>16s}"
        )

        for (
            n,
            serial,
            pipelined,
            speedup,
            reduction,
        ) in rows:
            print(
                f"{n:4d}"
                f"{serial['mean']:14.3f}"
                f"{pipelined['mean']:16.3f}"
                f"{speedup:16.3f}"
                f"{reduction:16.2f}"
            )

        best = min(
            rows,
            key=lambda row: row[2]["mean"],
        )

        print("\nBEST PIPELINED N")
        print("-" * 82)

        print(
            f"N={best[0]} with "
            f"{best[2]['mean']:.3f} ms mean iteration time"
        )

        print(
            f"Compared with serialized execution at the same N: "
            f"{best[3]:.3f}x ratio "
            f"({best[4]:.2f}% latency reduction)"
        )

        print("\nINTERPRETATION")
        print("-" * 82)

        print(
            "N=1 exposes essentially no cross-nano-batch overlap. As N grows,\n"
            "communication from one nano-batch can overlap computation of the\n"
            "next. If N becomes too large, smaller kernels and more launches can\n"
            "erase those gains. A non-monotonic optimum would reproduce the\n"
            "trade-off that motivates tLoRA's adaptive AIMD controller."
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
