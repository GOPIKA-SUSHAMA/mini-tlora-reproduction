import statistics
import sys

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:
    print("ERROR: Triton import failed:", repr(exc))
    sys.exit(1)

from tlora.multi_lora_linear import MultiLoRALinear


# =============================================================================
# Workload: identical to Phase 5A/5B
# =============================================================================

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


# =============================================================================
# Phase 5B two-stage kernels, now using PREALLOCATED buffers
# =============================================================================

@triton.jit
def lora_down_kernel(
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
    row = tl.program_id(0)

    aid = tl.load(adapter_ids_ptr + row)
    rank = tl.load(ranks_ptr + aid)

    offs_r = tl.arange(0, RMAX)
    acc = tl.zeros((RMAX,), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K, num_stages=2):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x = tl.load(
            X_ptr + row * K + offs_k,
            mask=offs_k < K,
            other=0.0,
        ).to(tl.float32)

        a_ptrs = (
            A_ptr
            + aid * (RMAX * K)
            + offs_r[:, None] * K
            + offs_k[None, :]
        )

        a = tl.load(
            a_ptrs,
            mask=(offs_r[:, None] < rank)
            & (offs_k[None, :] < K),
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

    tl.store(
        Z_ptr + row * RMAX + offs_r,
        0.0,
        mask=offs_r >= rank,
    )


@triton.jit
def lora_up_kernel(
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
    row = tl.program_id(0)
    block_n = tl.program_id(1)

    aid = tl.load(adapter_ids_ptr + row)
    rank = tl.load(ranks_ptr + aid)
    scale = tl.load(scales_ptr + aid)

    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, RMAX)

    z = tl.load(
        Z_ptr + row * RMAX + offs_r,
        mask=offs_r < rank,
        other=0.0,
    ).to(tl.float32)

    b_ptrs = (
        B_ptr
        + aid * (N * RMAX)
        + offs_n[:, None] * RMAX
        + offs_r[None, :]
    )

    b = tl.load(
        b_ptrs,
        mask=(offs_n[:, None] < N)
        & (offs_r[None, :] < rank),
        other=0.0,
    ).to(tl.float32)

    out = tl.sum(
        b * z[None, :],
        axis=1,
    ) * scale

    tl.store(
        Y_ptr + row * N + offs_n,
        out,
        mask=offs_n < N,
    )


# =============================================================================
# Phase 5C: one fused kernel
#
# Each Triton program handles ONE input row:
#   1. compute compact z = x @ A_i^T once
#   2. keep z in registers
#   3. stream across output blocks B_i
#   4. optionally add the shared-backbone output before storing
#
# Therefore:
#   - no global [M, RMAX] intermediate is required
#   - no second LoRA kernel launch is required
#   - A_i B_i^T is never materialised
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_K": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_K": 128, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_K": 128, "BLOCK_N": 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_K": 256, "BLOCK_N": 64},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_K": 256, "BLOCK_N": 128},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["K", "N", "RMAX"],
)
@triton.jit
def fused_lora_residual_kernel(
    X_ptr,
    A_ptr,
    B_ptr,
    adapter_ids_ptr,
    ranks_ptr,
    scales_ptr,
    Base_ptr,
    Y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    RMAX: tl.constexpr,
    ADD_BASE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    aid = tl.load(adapter_ids_ptr + row)
    rank = tl.load(ranks_ptr + aid)
    scale = tl.load(scales_ptr + aid)

    offs_r = tl.arange(0, RMAX)

    # -------------------------------------------------------------------------
    # Down projection: z = x @ A_i^T.
    # z remains in registers for the rest of this program.
    # -------------------------------------------------------------------------
    z = tl.zeros((RMAX,), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K, num_stages=2):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x = tl.load(
            X_ptr + row * K + offs_k,
            mask=offs_k < K,
            other=0.0,
        ).to(tl.float32)

        a_ptrs = (
            A_ptr
            + aid * (RMAX * K)
            + offs_r[:, None] * K
            + offs_k[None, :]
        )

        a = tl.load(
            a_ptrs,
            mask=(offs_r[:, None] < rank)
            & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        z += tl.sum(
            a * x[None, :],
            axis=1,
        )

    # -------------------------------------------------------------------------
    # Up projection: immediately consume z using B_i.
    # No global Z tensor is allocated.
    # -------------------------------------------------------------------------
    for n0 in tl.range(0, N, BLOCK_N, num_stages=1):
        offs_n = n0 + tl.arange(0, BLOCK_N)

        b_ptrs = (
            B_ptr
            + aid * (N * RMAX)
            + offs_n[:, None] * RMAX
            + offs_r[None, :]
        )

        b = tl.load(
            b_ptrs,
            mask=(offs_n[:, None] < N)
            & (offs_r[None, :] < rank),
            other=0.0,
        ).to(tl.float32)

        delta = tl.sum(
            b * z[None, :],
            axis=1,
        ) * scale

        if ADD_BASE:
            base = tl.load(
                Base_ptr + row * N + offs_n,
                mask=offs_n < N,
                other=0.0,
            ).to(tl.float32)

            delta += base

        tl.store(
            Y_ptr + row * N + offs_n,
            delta,
            mask=offs_n < N,
        )


# =============================================================================
# Setup helpers
# =============================================================================

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")


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


def make_batch(device):
    xs = []
    aids = []
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

        xs.append(x)

        aids.append(
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
        torch.cat(xs, dim=0).contiguous(),
        torch.cat(aids, dim=0).contiguous(),
        spans,
    )


def pack_weights(model, device):
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
            rank = cfg["rank"]
            adapter = model.adapters[str(aid)]

            A[aid, :rank].copy_(
                adapter.A
            )

            B[aid, :, :rank].copy_(
                adapter.B
            )

            ranks[aid] = rank
            scales[aid] = float(
                adapter.scaling
            )

    return (
        A.contiguous(),
        B.contiguous(),
        ranks.contiguous(),
        scales.contiguous(),
    )


# =============================================================================
# Execution paths
# =============================================================================

@torch.no_grad()
def pytorch_delta_into(
    model,
    x,
    spans,
    out,
):
    for aid in JOBS:
        start, end = spans[aid]

        out[start:end].copy_(
            model.adapters[str(aid)](
                x[start:end]
            )
        )

    return out


def two_stage_delta_into(
    x,
    adapter_ids,
    A,
    B,
    ranks,
    scales,
    z,
    out,
):
    M = x.shape[0]

    lora_down_kernel[
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

    lora_up_kernel[
        (
            M,
            triton.cdiv(
                OUT_FEATURES,
                128,
            ),
        )
    ](
        z,
        B,
        adapter_ids,
        ranks,
        scales,
        out,
        M=M,
        N=OUT_FEATURES,
        RMAX=MAX_RANK,
        BLOCK_N=128,
        num_warps=4,
    )

    return out


def fused_delta_into(
    x,
    adapter_ids,
    A,
    B,
    ranks,
    scales,
    dummy_base,
    out,
):
    M = x.shape[0]

    fused_lora_residual_kernel[
        (M,)
    ](
        x,
        A,
        B,
        adapter_ids,
        ranks,
        scales,
        dummy_base,
        out,
        M=M,
        K=IN_FEATURES,
        N=OUT_FEATURES,
        RMAX=MAX_RANK,
        ADD_BASE=False,
    )

    return out


def fused_full_forward(
    model,
    x,
    adapter_ids,
    A,
    B,
    ranks,
    scales,
    out,
):
    base = model.base(x)

    M = x.shape[0]

    fused_lora_residual_kernel[
        (M,)
    ](
        x,
        A,
        B,
        adapter_ids,
        ranks,
        scales,
        base,
        out,
        M=M,
        K=IN_FEATURES,
        N=OUT_FEATURES,
        RMAX=MAX_RANK,
        ADD_BASE=True,
    )

    return out


@torch.no_grad()
def pytorch_full_forward(
    model,
    x,
    spans,
    delta_buffer,
    out,
):
    base = model.base(x)

    pytorch_delta_into(
        model,
        x,
        spans,
        delta_buffer,
    )

    torch.add(
        base,
        delta_buffer,
        out=out,
    )

    return out


# =============================================================================
# Benchmark
# =============================================================================

@torch.no_grad()
def benchmark_cuda(fn):
    for _ in range(WARMUP):
        fn()

    torch.cuda.synchronize()

    samples = []

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

        samples.append(
            start.elapsed_time(end)
        )

    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples),
    }


def error_stats(a, b):
    d = (
        a.float()
        - b.float()
    ).abs()

    return (
        d.max().item(),
        d.mean().item(),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    require_cuda()

    device = torch.device("cuda:0")

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    model = build_model(device)

    x, adapter_ids, spans = make_batch(
        device
    )

    A, B, ranks, scales = pack_weights(
        model,
        device,
    )

    M = x.shape[0]

    torch_delta = torch.empty(
        M,
        OUT_FEATURES,
        device=device,
        dtype=DTYPE,
    )

    two_stage_z = torch.empty(
        M,
        MAX_RANK,
        device=device,
        dtype=DTYPE,
    )

    two_stage_out = torch.empty_like(
        torch_delta
    )

    fused_out = torch.empty_like(
        torch_delta
    )

    pytorch_full_out = torch.empty_like(
        torch_delta
    )

    fused_full_out = torch.empty_like(
        torch_delta
    )

    # Pointer is ignored when ADD_BASE=False.
    dummy_base = torch_delta

    print("=" * 80)
    print("MINI-tLoRA PHASE 5C - SINGLE-KERNEL FUSED LoRA + AUTOTUNING")
    print("=" * 80)

    print(f"PyTorch      : {torch.__version__}")
    print(f"Triton       : {triton.__version__}")
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Samples M    : {M}")
    print(f"K / N        : {IN_FEATURES} / {OUT_FEATURES}")
    print(f"Max rank     : {MAX_RANK}")

    # -------------------------------------------------------------------------
    # Correctness
    # -------------------------------------------------------------------------

    print("\nCORRECTNESS")
    print("-" * 80)

    pytorch_delta_into(
        model,
        x,
        spans,
        torch_delta,
    )

    two_stage_delta_into(
        x,
        adapter_ids,
        A,
        B,
        ranks,
        scales,
        two_stage_z,
        two_stage_out,
    )

    fused_delta_into(
        x,
        adapter_ids,
        A,
        B,
        ranks,
        scales,
        dummy_base,
        fused_out,
    )

    torch.cuda.synchronize()

    max_two, mean_two = error_stats(
        torch_delta,
        two_stage_out,
    )

    max_fused, mean_fused = error_stats(
        torch_delta,
        fused_out,
    )

    print(
        f"Two-stage vs PyTorch: "
        f"max={max_two:.6e}, "
        f"mean={mean_two:.6e}"
    )

    print(
        f"Fused vs PyTorch    : "
        f"max={max_fused:.6e}, "
        f"mean={mean_fused:.6e}"
    )

    torch.testing.assert_close(
        fused_out.float(),
        torch_delta.float(),
        rtol=2e-2,
        atol=2e-2,
    )

    print("Fused delta correctness: PASS")

    pytorch_full_forward(
        model,
        x,
        spans,
        torch_delta,
        pytorch_full_out,
    )

    fused_full_forward(
        model,
        x,
        adapter_ids,
        A,
        B,
        ranks,
        scales,
        fused_full_out,
    )

    torch.cuda.synchronize()

    max_full, mean_full = error_stats(
        pytorch_full_out,
        fused_full_out,
    )

    print(
        f"Fused full forward   : "
        f"max={max_full:.6e}, "
        f"mean={mean_full:.6e}"
    )

    torch.testing.assert_close(
        fused_full_out.float(),
        pytorch_full_out.float(),
        rtol=2e-2,
        atol=2e-2,
    )

    print("Fused full-forward correctness: PASS")

    # -------------------------------------------------------------------------
    # Adapter-only benchmark
    # -------------------------------------------------------------------------

    print("\nADAPTER-ONLY BENCHMARK - PREALLOCATED BUFFERS")
    print("-" * 80)

    torch_result = benchmark_cuda(
        lambda: pytorch_delta_into(
            model,
            x,
            spans,
            torch_delta,
        )
    )

    two_stage_result = benchmark_cuda(
        lambda: two_stage_delta_into(
            x,
            adapter_ids,
            A,
            B,
            ranks,
            scales,
            two_stage_z,
            two_stage_out,
        )
    )

    fused_result = benchmark_cuda(
        lambda: fused_delta_into(
            x,
            adapter_ids,
            A,
            B,
            ranks,
            scales,
            dummy_base,
            fused_out,
        )
    )

    print(
        f"{'Path':24s}"
        f"{'Mean ms':>12s}"
        f"{'Median ms':>14s}"
        f"{'Std ms':>12s}"
    )

    for name, r in [
        ("PyTorch", torch_result),
        ("Triton two-stage", two_stage_result),
        ("Triton fused", fused_result),
    ]:
        print(
            f"{name:24s}"
            f"{r['mean']:12.4f}"
            f"{r['median']:14.4f}"
            f"{r['stdev']:12.4f}"
        )

    print(
        "\nPyTorch / fused adapter ratio : "
        f"{torch_result['mean'] / fused_result['mean']:.3f}x"
    )

    print(
        "Two-stage / fused ratio      : "
        f"{two_stage_result['mean'] / fused_result['mean']:.3f}x"
    )

    # -------------------------------------------------------------------------
    # Full-forward benchmark
    # -------------------------------------------------------------------------

    print("\nEND-TO-END SHARED FORWARD")
    print("-" * 80)

    pytorch_full_result = benchmark_cuda(
        lambda: pytorch_full_forward(
            model,
            x,
            spans,
            torch_delta,
            pytorch_full_out,
        )
    )

    fused_full_result = benchmark_cuda(
        lambda: fused_full_forward(
            model,
            x,
            adapter_ids,
            A,
            B,
            ranks,
            scales,
            fused_full_out,
        )
    )

    pytorch_sps = (
        M
        / (
            pytorch_full_result["mean"]
            / 1000.0
        )
    )

    fused_sps = (
        M
        / (
            fused_full_result["mean"]
            / 1000.0
        )
    )

    print(
        f"{'Path':24s}"
        f"{'Mean ms':>12s}"
        f"{'Samples/s':>16s}"
    )

    print(
        f"{'PyTorch Shared SSM':24s}"
        f"{pytorch_full_result['mean']:12.4f}"
        f"{pytorch_sps:16.1f}"
    )

    print(
        f"{'Fused Triton SSM':24s}"
        f"{fused_full_result['mean']:12.4f}"
        f"{fused_sps:16.1f}"
    )

    print(
        "\nPyTorch / fused full ratio : "
        f"{pytorch_full_result['mean'] / fused_full_result['mean']:.3f}x"
    )

    # -------------------------------------------------------------------------
    # Memory traffic avoided
    # -------------------------------------------------------------------------

    z_bytes = (
        M
        * MAX_RANK
        * torch.tensor(
            [],
            dtype=DTYPE,
        ).element_size()
    )

    print("\nFUSION EFFECT")
    print("-" * 80)

    print(
        f"Two-stage global Z buffer : "
        f"{z_bytes / 1024**2:.6f} MB"
    )

    print(
        "Fused global Z buffer     : 0 MB "
        "(compact intermediate remains in registers)"
    )

    print(
        "LoRA kernel launches      : "
        "two-stage=2, fused=1 per packed batch"
    )

    print("\nINTERPRETATION")
    print("-" * 80)

    print(
        "Phase 5C removes the global low-rank intermediate and the boundary\n"
        "between down- and up-projection kernels. It also uses Triton\n"
        "autotuning over several BLOCK_K / BLOCK_N configurations.\n\n"
        "This is materially closer to tLoRA's Kernel Fuser, but it is still\n"
        "a reduced reproduction: the paper additionally integrates training\n"
        "backward kernels, rank-aware nano-batching, distributed overlap,\n"
        "and cluster scheduling."
    )


if __name__ == "__main__":
    main()
