import copy
import statistics
import time

import torch
import torch.nn.functional as F

from tlora.multi_lora_linear import MultiLoRALinear

RANKS = {0: 2, 1: 4, 2: 8, 3: 16}
IN_FEATURES = 512
OUT_FEATURES = 512
BATCH_PER_JOB = 32
WARMUP = 10
ITERS = 50


def bench(fn):
    for _ in range(WARMUP):
        fn()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    return statistics.mean(times), statistics.median(times)


def main():
    torch.manual_seed(2026)

    model = MultiLoRALinear(
        IN_FEATURES,
        OUT_FEATURES,
        RANKS,
    )

    for adapter in model.adapters.values():
        torch.nn.init.normal_(adapter.B, mean=0.0, std=0.01)

    xs = {
        aid: torch.randn(BATCH_PER_JOB, IN_FEATURES)
        for aid in RANKS
    }

    x_all = torch.cat([xs[aid] for aid in RANKS], dim=0)

    adapter_ids = torch.cat([
        torch.full((BATCH_PER_JOB,), aid, dtype=torch.long)
        for aid in RANKS
    ])

    # ---------------------------------------------------------
    # A. Isolate frozen-backbone execution only.
    # ---------------------------------------------------------
    def four_backbone_calls():
        outputs = [model.base(xs[aid]) for aid in RANKS]
        return outputs

    def one_combined_backbone_call():
        return model.base(x_all)

    four = bench(four_backbone_calls)
    one = bench(one_combined_backbone_call)

    # ---------------------------------------------------------
    # B. Compare Phase-1 boolean-mask routing with contiguous
    #    routing. x_all is already grouped by adapter:
    #    [job0 samples][job1 samples][job2 samples][job3 samples]
    # ---------------------------------------------------------
    def boolean_mask_ssm():
        return model(x_all, adapter_ids)

    def contiguous_ssm():
        base = model.base(x_all)
        out = base.clone()

        start = 0
        for aid in RANKS:
            end = start + BATCH_PER_JOB
            out[start:end] = (
                out[start:end]
                + model.adapters[str(aid)](x_all[start:end])
            )
            start = end

        return out

    # Verify that replacing boolean masks by contiguous slices
    # has not changed the result.
    y_mask = boolean_mask_ssm()
    y_contig = contiguous_ssm()

    max_error = (y_mask - y_contig).abs().max().item()
    assert torch.allclose(
        y_mask,
        y_contig,
        atol=1e-6,
        rtol=1e-6,
    )

    mask = bench(boolean_mask_ssm)
    contig = bench(contiguous_ssm)

    print("=" * 72)
    print("MINI-tLoRA PHASE 2B — WHERE IS THE CPU OVERHEAD?")
    print("=" * 72)

    print("\nBACKBONE ONLY")
    print("-" * 72)
    print(
        f"4 separate backbone calls : mean={four[0]*1000:.3f} ms "
        f"median={four[1]*1000:.3f} ms"
    )
    print(
        f"1 combined backbone call  : mean={one[0]*1000:.3f} ms "
        f"median={one[1]*1000:.3f} ms"
    )
    print(
        f"Separate / combined ratio : {four[0]/one[0]:.3f}x"
    )

    print("\nROUTING ONLY (FORWARD)")
    print("-" * 72)
    print(f"Output max error          : {max_error:.3e}")
    print(
        f"Boolean-mask SSM          : mean={mask[0]*1000:.3f} ms "
        f"median={mask[1]*1000:.3f} ms"
    )
    print(
        f"Contiguous-slice SSM      : mean={contig[0]*1000:.3f} ms "
        f"median={contig[1]*1000:.3f} ms"
    )
    print(
        f"Mask / contiguous ratio   : {mask[0]/contig[0]:.3f}x"
    )

    print("\nWHY THIS TEST MATTERS")
    print("-" * 72)
    print(
        "If the combined backbone is faster but the complete mask-based SSM\n"
        "is slower, the lost time is coming from heterogeneous adapter routing\n"
        "rather than from shared-backbone computation itself. That motivates\n"
        "the fused heterogeneous LoRA kernel used later in tLoRA."
    )


if __name__ == "__main__":
    main()
