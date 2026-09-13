import statistics
import sys

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:
    print("ERROR: Triton could not be imported.")
    print("Do not change the environment yet. Report this error:")
    print(repr(exc))
    sys.exit(1)

from tlora.multi_lora_linear import MultiLoRALinear


# ---------------------------------------------------------------------------
# Phase 5B workload
# ---------------------------------------------------------------------------

IN_FEATURES = 4096
OUT_FEATURES = 4096
MAX_RANK = 16

JOBS = {
    0: {"rank": 2, "batch": 32},
    1: {"rank": 4, "batch": 64},
    2: {"rank": 8, "batch": 16},
    3: {"rank": 16, "batch": 128},
}

DTYPE = torch.float16
WARMUP = 20
ITERS = 100


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def heterogeneous_lora_down_kernel(
    X_ptr,
    A_ptr,
    adapter_ids_ptr,
    ranks_ptr,
    Z_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    RMAX: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes, for each sample m:

        Z[m, r] = sum_k X[m, k] * A[adapter[m], r, k]

    Different samples may select different adapters/ranks.

    Padded A layout:
        [num_adapters, RMAX, K]

    This is a correctness-first grouped/heterogeneous kernel.
    """

    row = tl.program_id(axis=0)

    adapter_id = tl.load(adapter_ids_ptr + row)
    rank = tl.load(ranks_ptr + adapter_id)

    offs_r = tl.arange(0, RMAX)
    acc = tl.zeros((RMAX,), dtype=tl.float32)

    # Use a pipelined dynamic loop rather than aggressively unrolling all K.
    for k0 in tl.range(0, K, BLOCK_K, num_stages=2):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x = tl.load(
            X_ptr + row * K + offs_k,
            mask=offs_k < K,
            other=0.0,
        ).to(tl.float32)

        a_ptrs = (
            A_ptr
            + adapter_id * (RMAX * K)
            + offs_r[:, None] * K
            + offs_k[None, :]
        )

        a = tl.load(
            a_ptrs,
            mask=(offs_r[:, None] < rank) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(
            a * x[None, :],
            axis=1,
        )

    tl.store(
        Z_ptr + row * RMAX + offs_r,
        acc,
        mask=offs_r < rank,
    )

    # Explicitly zero padded rank entries.
    tl.store(
        Z_ptr + row * RMAX + offs_r,
        0.0,
        mask=offs_r >= rank,
    )


@triton.jit
def heterogeneous_lora_up_kernel(
    Z_ptr,
    B_ptr,
    adapter_ids_ptr,
    ranks_ptr,
    scales_ptr,
    Y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    RMAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Computes:

        Y[m, n] =
          scale[adapter[m]]
          * sum_r Z[m, r] * B[adapter[m], n, r]

    Padded B layout:
        [num_adapters, N, RMAX]
    """

    row = tl.program_id(axis=0)
    block_n = tl.program_id(axis=1)

    adapter_id = tl.load(adapter_ids_ptr + row)
    rank = tl.load(ranks_ptr + adapter_id)
    scale = tl.load(scales_ptr + adapter_id)

    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RMAX)

    z = tl.load(
        Z_ptr + row * RMAX + offs_r,
        mask=offs_r < rank,
        other=0.0,
    ).to(tl.float32)

    b_ptrs = (
        B_ptr
        + adapter_id * (N * RMAX)
        + offs_n[:, None] * RMAX
        + offs_r[None, :]
    )

    b = tl.load(
        b_ptrs,
        mask=(offs_n[:, None] < N) & (offs_r[None, :] < rank),
        other=0.0,
    ).to(tl.float32)

    out = tl.sum(
        b * z[None, :],
        axis=1,
    )

    out *= scale

    tl.store(
        Y_ptr + row * N + offs_n,
        out,
        mask=offs_n < N,
    )


# ---------------------------------------------------------------------------
# Packing utilities
# ---------------------------------------------------------------------------

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. Run this on the Kaggle T4 GPU notebook."
        )


def build_model(device):
    ranks = {
        aid: cfg["rank"]
        for aid, cfg in JOBS.items()
    }

    model = MultiLoRALinear(
        IN_FEATURES,
        OUT_FEATURES,
        ranks,
    ).to(
        device=device,
        dtype=DTYPE,
    )

    with torch.no_grad():
        for adapter in model.adapters.values():
            torch.nn.init.normal_(
                adapter.B,
                mean=0.0,
                std=0.01,
            )

    return model


def make_packed_batch(device):
    x_parts = []
    adapter_parts = []
    spans = {}

    cursor = 0

    for aid, cfg in JOBS.items():
        batch = cfg["batch"]

        x = torch.randn(
            batch,
            IN_FEATURES,
            device=device,
            dtype=DTYPE,
        )

        x_parts.append(x)

        adapter_parts.append(
            torch.full(
                (batch,),
                aid,
                device=device,
                dtype=torch.int32,
            )
        )

        spans[aid] = (
            cursor,
            cursor + batch,
        )
        cursor += batch

    return (
        torch.cat(x_parts, dim=0).contiguous(),
        torch.cat(adapter_parts, dim=0).contiguous(),
        spans,
    )


def pack_adapter_weights(model, device):
    """
    Convert heterogeneous LoRA tensors into padded dense GPU buffers.

    A_padded:
      [num_adapters, MAX_RANK, IN_FEATURES]

    B_padded:
      [num_adapters, OUT_FEATURES, MAX_RANK]
    """

    num_adapters = len(JOBS)

    A = torch.zeros(
        num_adapters,
        MAX_RANK,
        IN_FEATURES,
        device=device,
        dtype=DTYPE,
    )

    B = torch.zeros(
        num_adapters,
        OUT_FEATURES,
        MAX_RANK,
        device=device,
        dtype=DTYPE,
    )

    ranks = torch.zeros(
        num_adapters,
        device=device,
        dtype=torch.int32,
    )

    scales = torch.ones(
        num_adapters,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for aid, cfg in JOBS.items():
            adapter = model.adapters[str(aid)]
            rank = cfg["rank"]

            # Existing implementation:
            # A shape = [rank, in_features]
            # B shape = [out_features, rank]
            A[aid, :rank].copy_(adapter.A)
            B[aid, :, :rank].copy_(adapter.B)

            ranks[aid] = rank
            scales[aid] = float(adapter.scaling)

    return (
        A.contiguous(),
        B.contiguous(),
        ranks.contiguous(),
        scales.contiguous(),
    )


# ---------------------------------------------------------------------------
# Reference and Triton execution
# ---------------------------------------------------------------------------

def pytorch_adapter_delta(
    model,
    x,
    spans,
):
    out = torch.empty(
        x.shape[0],
        OUT_FEATURES,
        device=x.device,
        dtype=x.dtype,
    )

    for aid in JOBS:
        start, end = spans[aid]

        out[start:end] = model.adapters[str(aid)](
            x[start:end]
        )

    return out


def triton_adapter_delta(
    x,
    adapter_ids,
    A,
    B,
    ranks,
    scales,
):
    M = x.shape[0]

    # Intermediate is only [M, max_rank], never [out, in].
    z = torch.empty(
        M,
        MAX_RANK,
        device=x.device,
        dtype=DTYPE,
    )

    y = torch.empty(
        M,
        OUT_FEATURES,
        device=x.device,
        dtype=DTYPE,
    )

    heterogeneous_lora_down_kernel[
        (M,)
    ](
        x,
        A,
        adapter_ids,
        ranks,
        z,
        M=M,
        K=IN_FEATURES,
        RMAX=MAX_RANK,
        BLOCK_K=128,
        num_warps=4,
    )

    grid_up = (
        M,
        triton.cdiv(
            OUT_FEATURES,
            128,
        ),
    )

    heterogeneous_lora_up_kernel[
        grid_up
    ](
        z,
        B,
        adapter_ids,
        ranks,
        scales,
        y,
        M=M,
        N=OUT_FEATURES,
        RMAX=MAX_RANK,
        BLOCK_N=128,
        num_warps=4,
    )

    return y


def pytorch_shared_forward(
    model,
    x,
    spans,
):
    base = model.base(x)
    delta = pytorch_adapter_delta(
        model,
        x,
        spans,
    )
    return base + delta


def triton_shared_forward(
    model,
    x,
    adapter_ids,
    A,
    B,
    ranks,
    scales,
):
    base = model.base(x)

    delta = triton_adapter_delta(
        x,
        adapter_ids,
        A,
        B,
        ranks,
        scales,
    )

    return base + delta


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_cuda(fn):
    for _ in range(WARMUP):
        fn()

    torch.cuda.synchronize()

    times = []

    for _ in range(ITERS):
        start = torch.cuda.Event(
            enable_timing=True
        )
        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()
        fn()
        end.record()

        end.synchronize()
        times.append(
            start.elapsed_time(end)
        )

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times),
    }


def error_stats(reference, candidate):
    diff = (
        reference.float()
        - candidate.float()
    ).abs()

    return {
        "max": diff.max().item(),
        "mean": diff.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    require_cuda()

    device = torch.device("cuda:0")

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    print("=" * 80)
    print("MINI-tLoRA PHASE 5B - TRITON HETEROGENEOUS LoRA FORWARD")
    print("=" * 80)

    print(f"PyTorch        : {torch.__version__}")
    print(f"Triton         : {triton.__version__}")
    print(f"PyTorch CUDA   : {torch.version.cuda}")
    print(
        f"GPU            : "
        f"{torch.cuda.get_device_name(0)}"
    )

    props = torch.cuda.get_device_properties(0)

    print(
        f"Compute cap.   : "
        f"{props.major}.{props.minor}"
    )

    model = build_model(device)

    x, adapter_ids, spans = make_packed_batch(
        device
    )

    A, B, ranks, scales = pack_adapter_weights(
        model,
        device,
    )

    print("\nWORKLOAD")
    print("-" * 80)

    for aid, cfg in JOBS.items():
        print(
            f"Adapter {aid}: "
            f"rank={cfg['rank']:2d}, "
            f"batch={cfg['batch']:3d}"
        )

    print(
        f"Combined M     : {x.shape[0]}"
    )
    print(
        f"Input K        : {IN_FEATURES}"
    )
    print(
        f"Output N       : {OUT_FEATURES}"
    )
    print(
        f"Padded max rank: {MAX_RANK}"
    )

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    print("\nADAPTER-DELTA CORRECTNESS")
    print("-" * 80)

    with torch.no_grad():
        torch_ref = pytorch_adapter_delta(
            model,
            x,
            spans,
        )

        triton_out = triton_adapter_delta(
            x,
            adapter_ids,
            A,
            B,
            ranks,
            scales,
        )

    stats = error_stats(
        torch_ref,
        triton_out,
    )

    print(
        f"max_abs_error  : "
        f"{stats['max']:.6e}"
    )
    print(
        f"mean_abs_error : "
        f"{stats['mean']:.6e}"
    )

    # This is FP16 data with FP32 accumulation in the Triton reduction.
    # We use a documented tolerance rather than requiring bitwise equality.
    torch.testing.assert_close(
        triton_out.float(),
        torch_ref.float(),
        rtol=2e-2,
        atol=2e-2,
    )

    print("Adapter delta correctness: PASS")

    print("\nEND-TO-END SHARED FORWARD CORRECTNESS")
    print("-" * 80)

    with torch.no_grad():
        full_ref = pytorch_shared_forward(
            model,
            x,
            spans,
        )

        full_triton = triton_shared_forward(
            model,
            x,
            adapter_ids,
            A,
            B,
            ranks,
            scales,
        )

    full_stats = error_stats(
        full_ref,
        full_triton,
    )

    print(
        f"max_abs_error  : "
        f"{full_stats['max']:.6e}"
    )
    print(
        f"mean_abs_error : "
        f"{full_stats['mean']:.6e}"
    )

    torch.testing.assert_close(
        full_triton.float(),
        full_ref.float(),
        rtol=2e-2,
        atol=2e-2,
    )

    print("Shared forward correctness: PASS")

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    intermediate_mb = (
        x.shape[0]
        * MAX_RANK
        * torch.tensor(
            [],
            dtype=DTYPE,
        ).element_size()
        / 1024**2
    )

    materialized_weight_mb = (
        len(JOBS)
        * OUT_FEATURES
        * IN_FEATURES
        * torch.tensor(
            [],
            dtype=DTYPE,
        ).element_size()
        / 1024**2
    )

    print("\nINTERMEDIATE MEMORY")
    print("-" * 80)
    print(
        f"Triton Z = [M, RMAX] : "
        f"{intermediate_mb:.4f} MB"
    )
    print(
        "Hypothetical materialised "
        f"A@B update matrices: "
        f"{materialized_weight_mb:.2f} MB"
    )
    print(
        "The Triton path does not materialise "
        "the full LoRA weight update."
    )

    # ------------------------------------------------------------------
    # Benchmark adapter path in isolation.
    # ------------------------------------------------------------------

    print("\nADAPTER-PATH FORWARD BENCHMARK")
    print("-" * 80)

    pytorch_adapter_result = benchmark_cuda(
        lambda: pytorch_adapter_delta(
            model,
            x,
            spans,
        )
    )

    triton_adapter_result = benchmark_cuda(
        lambda: triton_adapter_delta(
            x,
            adapter_ids,
            A,
            B,
            ranks,
            scales,
        )
    )

    print(
        f"{'Path':22s}"
        f"{'Mean ms':>12s}"
        f"{'Median ms':>14s}"
        f"{'Std ms':>12s}"
    )

    print(
        f"{'PyTorch adapters':22s}"
        f"{pytorch_adapter_result['mean_ms']:12.4f}"
        f"{pytorch_adapter_result['median_ms']:14.4f}"
        f"{pytorch_adapter_result['stdev_ms']:12.4f}"
    )

    print(
        f"{'Triton prototype':22s}"
        f"{triton_adapter_result['mean_ms']:12.4f}"
        f"{triton_adapter_result['median_ms']:14.4f}"
        f"{triton_adapter_result['stdev_ms']:12.4f}"
    )

    print(
        "PyTorch / Triton ratio : "
        f"{pytorch_adapter_result['mean_ms'] / triton_adapter_result['mean_ms']:.3f}x"
    )

    # ------------------------------------------------------------------
    # Benchmark end-to-end shared forward.
    # ------------------------------------------------------------------

    print("\nEND-TO-END SHARED FORWARD BENCHMARK")
    print("-" * 80)

    pytorch_full_result = benchmark_cuda(
        lambda: pytorch_shared_forward(
            model,
            x,
            spans,
        )
    )

    triton_full_result = benchmark_cuda(
        lambda: triton_shared_forward(
            model,
            x,
            adapter_ids,
            A,
            B,
            ranks,
            scales,
        )
    )

    total_samples = x.shape[0]

    print(
        f"{'Path':22s}"
        f"{'Mean ms':>12s}"
        f"{'Samples/s':>16s}"
    )

    pytorch_sps = (
        total_samples
        / (
            pytorch_full_result["mean_ms"]
            / 1000.0
        )
    )

    triton_sps = (
        total_samples
        / (
            triton_full_result["mean_ms"]
            / 1000.0
        )
    )

    print(
        f"{'PyTorch Shared SSM':22s}"
        f"{pytorch_full_result['mean_ms']:12.4f}"
        f"{pytorch_sps:16.1f}"
    )

    print(
        f"{'Triton Shared SSM':22s}"
        f"{triton_full_result['mean_ms']:12.4f}"
        f"{triton_sps:16.1f}"
    )

    print(
        "PyTorch / Triton ratio : "
        f"{pytorch_full_result['mean_ms'] / triton_full_result['mean_ms']:.3f}x"
    )

    print("\nINTERPRETATION")
    print("-" * 80)
    print(
        "Phase 5B is a correctness-first Triton reproduction of the\n"
        "heterogeneous LoRA execution structure. It uses two custom kernels:\n"
        "down projection (X @ A_i^T) and immediate up projection with B_i.\n\n"
        "It is NOT yet the paper's fully optimised fused training kernel.\n"
        "This prototype uses scalar/reduction-style grouped math rather than\n"
        "a heavily autotuned tensor-core grouped GEMM, so it may be slower\n"
        "than PyTorch/cuBLAS. A slowdown here is useful evidence: Phase 5C\n"
        "will optimise the kernel rather than hiding a weak first result."
    )


if __name__ == "__main__":
    main()
