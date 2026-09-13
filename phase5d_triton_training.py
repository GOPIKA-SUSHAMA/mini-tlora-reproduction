import copy
import statistics
import sys

import torch
import torch.nn.functional as F

try:
    import triton
except Exception as exc:
    print("ERROR: Triton import failed:", repr(exc))
    sys.exit(1)

from phase5c_fused_triton import (
    JOBS,
    IN_FEATURES,
    OUT_FEATURES,
    MAX_RANK,
    DTYPE,
    fused_lora_residual_kernel,
)


# =============================================================================
# Purpose
# =============================================================================
#
# Phase 5C proved forward correctness and forward speed.
#
# Phase 5D answers the next research question:
#
#   Can the fused Triton forward participate in TRAINING while preserving
#   the same LoRA gradients as a PyTorch reference?
#
# This phase deliberately uses:
#   Triton forward + explicit PyTorch backward formulas
#
# so we can validate training semantics before writing custom Triton backward
# kernels. This is a bridge toward the paper's full training implementation,
# not the final optimized backward path.
#
# =============================================================================


WARMUP = 10
ITERS = 40
LR = 1e-3


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")


def make_batch(device):
    x_parts = []
    target_parts = []
    aid_parts = []
    spans = {}

    cursor = 0

    for aid, cfg in JOBS.items():
        batch = cfg["batch"]

        x_parts.append(
            torch.randn(
                batch,
                IN_FEATURES,
                device=device,
                dtype=DTYPE,
            )
        )

        target_parts.append(
            torch.randn(
                batch,
                OUT_FEATURES,
                device=device,
                dtype=DTYPE,
            )
        )

        aid_parts.append(
            torch.full(
                (batch,),
                aid,
                device=device,
                dtype=torch.int32,
            )
        )

        spans[aid] = (cursor, cursor + batch)
        cursor += batch

    return (
        torch.cat(x_parts, dim=0).contiguous(),
        torch.cat(target_parts, dim=0).contiguous(),
        torch.cat(aid_parts, dim=0).contiguous(),
        spans,
    )


def make_packed_weights(device):
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

    # Use deterministic LoRA-style initialization:
    # A random, B non-zero small random so both A/B gradients are observable.
    g = torch.Generator(device=device)
    g.manual_seed(2026)

    with torch.no_grad():
        for aid, cfg in JOBS.items():
            rank = cfg["rank"]

            A[aid, :rank].normal_(
                mean=0.0,
                std=0.02,
                generator=g,
            )

            B[aid, :, :rank].normal_(
                mean=0.0,
                std=0.01,
                generator=g,
            )

            ranks[aid] = rank
            scales[aid] = 1.0

    A.requires_grad_(True)
    B.requires_grad_(True)

    return A, B, ranks, scales


def reference_delta(
    x,
    A,
    B,
    ranks,
    scales,
    spans,
):
    """
    Pure PyTorch reference using the same padded A/B tensors.
    Packed job ordering is contiguous, so concatenate per-job outputs.
    """
    parts = []

    for aid in JOBS:
        start, end = spans[aid]
        rank = JOBS[aid]["rank"]

        x_i = x[start:end].float()
        A_i = A[aid, :rank].float()
        B_i = B[aid, :, :rank].float()

        z_i = x_i @ A_i.T
        y_i = (z_i @ B_i.T) * scales[aid]

        # Match the Triton output dtype boundary.
        parts.append(y_i.to(DTYPE))

    return torch.cat(parts, dim=0)


class FusedTritonLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        A,
        B,
        adapter_ids,
        ranks,
        scales,
    ):
        M = x.shape[0]

        out = torch.empty(
            M,
            OUT_FEATURES,
            device=x.device,
            dtype=DTYPE,
        )

        # Pointer ignored because ADD_BASE=False.
        dummy_base = out

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

        ctx.save_for_backward(
            x,
            A,
            B,
            adapter_ids,
            ranks,
            scales,
        )

        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            x,
            A,
            B,
            adapter_ids,
            ranks,
            scales,
        ) = ctx.saved_tensors

        # Compute explicit analytical LoRA gradients in FP32.
        grad_x = torch.zeros_like(
            x,
            dtype=torch.float32,
        )

        grad_A = torch.zeros_like(
            A,
            dtype=torch.float32,
        )

        grad_B = torch.zeros_like(
            B,
            dtype=torch.float32,
        )

        for aid, cfg in JOBS.items():
            rank = cfg["rank"]

            mask = adapter_ids == aid

            x_i = x[mask].float()
            g_i = grad_out[mask].float()

            A_i = A[aid, :rank].float()
            B_i = B[aid, :, :rank].float()
            scale = scales[aid].float()

            # Forward intermediate recomputed for dB.
            z_i = x_i @ A_i.T

            # y = scale * z @ B^T
            #
            # dB = scale * g^T @ z
            grad_B_i = (
                scale
                * (g_i.T @ z_i)
            )

            # dz = scale * g @ B
            grad_z_i = (
                scale
                * (g_i @ B_i)
            )

            # dA = dz^T @ x
            grad_A_i = (
                grad_z_i.T @ x_i
            )

            # dx = dz @ A
            grad_x_i = (
                grad_z_i @ A_i
            )

            grad_B[aid, :, :rank] = grad_B_i
            grad_A[aid, :rank] = grad_A_i
            grad_x[mask] = grad_x_i

        return (
            grad_x.to(x.dtype),
            grad_A.to(A.dtype),
            grad_B.to(B.dtype),
            None,
            None,
            None,
        )


def fused_delta(
    x,
    A,
    B,
    adapter_ids,
    ranks,
    scales,
):
    return FusedTritonLoRAFunction.apply(
        x,
        A,
        B,
        adapter_ids,
        ranks,
        scales,
    )


def max_mean_error(a, b):
    d = (
        a.float()
        - b.float()
    ).abs()

    return d.max().item(), d.mean().item()


def relative_l2_error(a, b):
    da = a.float()
    db = b.float()

    numerator = torch.linalg.vector_norm(
        da - db
    )

    denominator = torch.linalg.vector_norm(
        db
    ).clamp_min(1e-12)

    return (
        numerator / denominator
    ).item()


def active_gradient_stats(
    grad_test,
    grad_ref,
    which,
):
    max_abs = 0.0
    mean_values = []
    rel_values = []

    for aid, cfg in JOBS.items():
        rank = cfg["rank"]

        if which == "A":
            gt = grad_test[aid, :rank]
            gr = grad_ref[aid, :rank]
        else:
            gt = grad_test[aid, :, :rank]
            gr = grad_ref[aid, :, :rank]

        d = (
            gt.float()
            - gr.float()
        ).abs()

        max_abs = max(
            max_abs,
            d.max().item(),
        )

        mean_values.append(
            d.mean().item()
        )

        rel_values.append(
            relative_l2_error(gt, gr)
        )

    return {
        "max_abs": max_abs,
        "mean_abs": statistics.mean(
            mean_values
        ),
        "max_rel_l2": max(
            rel_values
        ),
    }


def run_correctness(
    x_seed,
    target,
    adapter_ids,
    spans,
    A_seed,
    B_seed,
    ranks,
    scales,
):
    # -------------------------------------------------------------------------
    # PyTorch reference
    # -------------------------------------------------------------------------
    x_ref = x_seed.detach().clone().requires_grad_(True)
    A_ref = A_seed.detach().clone().requires_grad_(True)
    B_ref = B_seed.detach().clone().requires_grad_(True)

    y_ref = reference_delta(
        x_ref,
        A_ref,
        B_ref,
        ranks,
        scales,
        spans,
    )

    loss_ref = F.mse_loss(
        y_ref.float(),
        target.float(),
        reduction="mean",
    )

    loss_ref.backward()

    # -------------------------------------------------------------------------
    # Triton-forward training path
    # -------------------------------------------------------------------------
    x_tri = x_seed.detach().clone().requires_grad_(True)
    A_tri = A_seed.detach().clone().requires_grad_(True)
    B_tri = B_seed.detach().clone().requires_grad_(True)

    y_tri = fused_delta(
        x_tri,
        A_tri,
        B_tri,
        adapter_ids,
        ranks,
        scales,
    )

    loss_tri = F.mse_loss(
        y_tri.float(),
        target.float(),
        reduction="mean",
    )

    loss_tri.backward()

    # -------------------------------------------------------------------------
    # Compare
    # -------------------------------------------------------------------------
    fwd_max, fwd_mean = max_mean_error(
        y_tri,
        y_ref,
    )

    x_max, x_mean = max_mean_error(
        x_tri.grad,
        x_ref.grad,
    )

    A_stats = active_gradient_stats(
        A_tri.grad,
        A_ref.grad,
        "A",
    )

    B_stats = active_gradient_stats(
        B_tri.grad,
        B_ref.grad,
        "B",
    )

    print("\nTRAINING CORRECTNESS")
    print("-" * 80)

    print(
        f"Loss PyTorch : {loss_ref.item():.8f}"
    )

    print(
        f"Loss Triton  : {loss_tri.item():.8f}"
    )

    print(
        f"Forward error: max={fwd_max:.6e}, "
        f"mean={fwd_mean:.6e}"
    )

    print(
        f"dX error     : max={x_max:.6e}, "
        f"mean={x_mean:.6e}, "
        f"relL2={relative_l2_error(x_tri.grad, x_ref.grad):.6e}"
    )

    print(
        "dA error     : "
        f"max={A_stats['max_abs']:.6e}, "
        f"mean={A_stats['mean_abs']:.6e}, "
        f"max_relL2={A_stats['max_rel_l2']:.6e}"
    )

    print(
        "dB error     : "
        f"max={B_stats['max_abs']:.6e}, "
        f"mean={B_stats['mean_abs']:.6e}, "
        f"max_relL2={B_stats['max_rel_l2']:.6e}"
    )

    # These are FP16 forward tensors with FP32 math in the backward formulas.
    # Use relative checks for gradients instead of demanding bitwise identity.
    assert fwd_max <= 2e-2
    assert relative_l2_error(
        x_tri.grad,
        x_ref.grad,
    ) <= 5e-2
    assert A_stats["max_rel_l2"] <= 5e-2
    assert B_stats["max_rel_l2"] <= 5e-2

    print("Training gradient equivalence: PASS")

    # -------------------------------------------------------------------------
    # One SGD step equivalence on active LoRA parameters.
    # -------------------------------------------------------------------------
    with torch.no_grad():
        A_ref_next = A_ref - LR * A_ref.grad
        B_ref_next = B_ref - LR * B_ref.grad

        A_tri_next = A_tri - LR * A_tri.grad
        B_tri_next = B_tri - LR * B_tri.grad

    A_step = active_gradient_stats(
        A_tri_next,
        A_ref_next,
        "A",
    )

    B_step = active_gradient_stats(
        B_tri_next,
        B_ref_next,
        "B",
    )

    print("\nONE-STEP SGD EQUIVALENCE")
    print("-" * 80)

    print(
        "A next-step max relative L2: "
        f"{A_step['max_rel_l2']:.6e}"
    )

    print(
        "B next-step max relative L2: "
        f"{B_step['max_rel_l2']:.6e}"
    )

    assert A_step["max_rel_l2"] <= 5e-2
    assert B_step["max_rel_l2"] <= 5e-2

    print("One-step parameter update equivalence: PASS")


def benchmark_training(
    x_seed,
    target,
    adapter_ids,
    spans,
    A_seed,
    B_seed,
    ranks,
    scales,
):
    """
    Compare full forward+backward only.

    Important:
      Triton path uses a custom Triton forward but PyTorch matrix operations
      in the explicit backward. This is a bridge benchmark, not the final
      optimized tLoRA training kernel.
    """

    def make_reference_state():
        return (
            x_seed.detach().clone().requires_grad_(True),
            A_seed.detach().clone().requires_grad_(True),
            B_seed.detach().clone().requires_grad_(True),
        )

    def make_triton_state():
        return (
            x_seed.detach().clone().requires_grad_(True),
            A_seed.detach().clone().requires_grad_(True),
            B_seed.detach().clone().requires_grad_(True),
        )

    def time_one(mode):
        samples = []

        for i in range(WARMUP + ITERS):
            if mode == "reference":
                x, A, B = make_reference_state()
            else:
                x, A, B = make_triton_state()

            torch.cuda.synchronize()

            start = torch.cuda.Event(
                enable_timing=True
            )
            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            if mode == "reference":
                y = reference_delta(
                    x,
                    A,
                    B,
                    ranks,
                    scales,
                    spans,
                )
            else:
                y = fused_delta(
                    x,
                    A,
                    B,
                    adapter_ids,
                    ranks,
                    scales,
                )

            loss = F.mse_loss(
                y.float(),
                target.float(),
                reduction="mean",
            )

            loss.backward()

            end.record()
            end.synchronize()

            if i >= WARMUP:
                samples.append(
                    start.elapsed_time(end)
                )

        return {
            "mean": statistics.mean(samples),
            "median": statistics.median(samples),
            "stdev": statistics.stdev(samples),
        }

    ref = time_one("reference")
    tri = time_one("triton")

    print("\nFORWARD + BACKWARD BRIDGE BENCHMARK")
    print("-" * 80)

    print(
        f"{'Path':28s}"
        f"{'Mean ms':>12s}"
        f"{'Median ms':>14s}"
    )

    print(
        f"{'PyTorch reference':28s}"
        f"{ref['mean']:12.4f}"
        f"{ref['median']:14.4f}"
    )

    print(
        f"{'Triton FWD + PyTorch BWD':28s}"
        f"{tri['mean']:12.4f}"
        f"{tri['median']:14.4f}"
    )

    print(
        "\nReference / bridge ratio: "
        f"{ref['mean'] / tri['mean']:.3f}x"
    )


def main():
    require_cuda()

    device = torch.device("cuda:0")

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    x, target, adapter_ids, spans = make_batch(
        device
    )

    A, B, ranks, scales = make_packed_weights(
        device
    )

    print("=" * 80)
    print("MINI-tLoRA PHASE 5D - TRAINING SEMANTICS WITH TRITON FORWARD")
    print("=" * 80)

    print(f"PyTorch       : {torch.__version__}")
    print(f"Triton        : {triton.__version__}")
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"Samples       : {x.shape[0]}")
    print(f"Ranks         : {[JOBS[a]['rank'] for a in JOBS]}")

    run_correctness(
        x,
        target,
        adapter_ids,
        spans,
        A,
        B,
        ranks,
        scales,
    )

    benchmark_training(
        x,
        target,
        adapter_ids,
        spans,
        A,
        B,
        ranks,
        scales,
    )

    print("\nINTERPRETATION")
    print("-" * 80)

    print(
        "Phase 5D validates that the fused Triton forward can participate in\n"
        "training without changing LoRA learning semantics. Backward is still\n"
        "implemented with explicit PyTorch matrix operations. This intentionally\n"
        "separates correctness from optimization.\n\n"
        "The next optimization stage is a custom heterogeneous backward path\n"
        "for dA, dB and dX, followed by the two-GPU nano-batch/communication\n"
        "overlap experiment."
    )


if __name__ == "__main__":
    main()
