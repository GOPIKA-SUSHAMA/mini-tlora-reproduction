import copy
import statistics
import sys

import torch
import torch.nn.functional as F

from tlora.multi_lora_linear import IndependentLoRALinear, MultiLoRALinear


# A larger linear layer than the CPU phases so GPU work is measurable,
# while remaining comfortably inside a free 16 GB T4.
IN_FEATURES = 4096
OUT_FEATURES = 4096

JOBS = {
    0: {"rank": 2, "batch": 32},
    1: {"rank": 4, "batch": 64},
    2: {"rank": 8, "batch": 16},
    3: {"rank": 16, "batch": 128},
}

WARMUP = 10
ITERS = 40
DTYPE = torch.float16


def cuda_required():
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not detected.")
        print("Run this phase in a GPU notebook such as Kaggle T4x2.")
        sys.exit(1)


def gpu_info():
    print("=" * 78)
    print("MINI-tLoRA PHASE 5A - CUDA BASELINE")
    print("=" * 78)
    print(f"PyTorch          : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    print(f"CUDA runtime     : {torch.version.cuda}")
    print(f"GPU count        : {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"GPU {i}            : {props.name} "
            f"({props.total_memory / 1024**3:.1f} GiB)"
        )

    print(f"Benchmark device : cuda:0")
    print(f"Benchmark dtype  : {DTYPE}")


def parameter_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def make_seed(device):
    ranks = {aid: cfg["rank"] for aid, cfg in JOBS.items()}

    model = MultiLoRALinear(
        IN_FEATURES,
        OUT_FEATURES,
        ranks,
    ).to(device=device, dtype=DTYPE)

    # Activate non-zero LoRA paths.
    with torch.no_grad():
        for adapter in model.adapters.values():
            torch.nn.init.normal_(
                adapter.B,
                mean=0.0,
                std=0.01,
            )

    return model


def make_data(device):
    xs = {}
    ys = {}
    spans = {}
    x_parts = []
    y_parts = []

    cursor = 0

    for aid, cfg in JOBS.items():
        batch = cfg["batch"]

        x = torch.randn(
            batch,
            IN_FEATURES,
            device=device,
            dtype=DTYPE,
        )

        y = torch.randn(
            batch,
            OUT_FEATURES,
            device=device,
            dtype=DTYPE,
        )

        xs[aid] = x
        ys[aid] = y

        x_parts.append(x)
        y_parts.append(y)

        spans[aid] = (cursor, cursor + batch)
        cursor += batch

    return (
        xs,
        ys,
        torch.cat(x_parts, dim=0),
        torch.cat(y_parts, dim=0),
        spans,
    )


def make_independent(seed, device):
    models = {}

    for aid, cfg in JOBS.items():
        model = IndependentLoRALinear(
            IN_FEATURES,
            OUT_FEATURES,
            cfg["rank"],
        ).to(device=device, dtype=DTYPE)

        model.base.load_state_dict(
            copy.deepcopy(seed.base.state_dict())
        )

        model.adapter.load_state_dict(
            copy.deepcopy(
                seed.adapters[str(aid)].state_dict()
            )
        )

        models[aid] = model

    return models


def make_opts_for_shared(model):
    return {
        aid: torch.optim.SGD(
            model.adapter_parameters(aid),
            lr=1e-5,
        )
        for aid in JOBS
    }


def zero_opts(opts):
    for opt in opts.values():
        opt.zero_grad(set_to_none=True)


def step_opts(opts):
    for opt in opts.values():
        opt.step()


def independent_step(models, opts, xs, ys):
    for aid in JOBS:
        opt = opts[aid]
        opt.zero_grad(set_to_none=True)

        pred = models[aid](xs[aid])
        loss = F.mse_loss(
            pred.float(),
            ys[aid].float(),
            reduction="sum",
        )

        loss.backward()
        opt.step()


def naive_grouped_step(model, opts, xs, ys):
    zero_opts(opts)

    total = None

    for aid in JOBS:
        pred = (
            model.base(xs[aid])
            + model.adapters[str(aid)](xs[aid])
        )

        loss = F.mse_loss(
            pred.float(),
            ys[aid].float(),
            reduction="sum",
        )

        total = loss if total is None else total + loss

    total.backward()
    step_opts(opts)


def shared_contiguous_forward(model, x_all, spans):
    # One shared backbone invocation.
    out = model.base(x_all)
    result = out.clone()

    # Correctness-first contiguous adapter routing.
    for aid in JOBS:
        start, end = spans[aid]

        result[start:end] = (
            result[start:end]
            + model.adapters[str(aid)](
                x_all[start:end]
            )
        )

    return result


def shared_step(model, opts, x_all, y_all, spans):
    zero_opts(opts)

    pred = shared_contiguous_forward(
        model,
        x_all,
        spans,
    )

    loss = F.mse_loss(
        pred.float(),
        y_all.float(),
        reduction="sum",
    )

    loss.backward()
    step_opts(opts)


def cuda_benchmark(fn):
    # Warm-up kernels / caches.
    for _ in range(WARMUP):
        fn()

    torch.cuda.synchronize()

    times_ms = []

    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms)
        if len(times_ms) > 1 else 0.0,
    }


def measure_mode(name, fn, total_samples):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = cuda_benchmark(fn)

    peak_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    throughput = (
        total_samples
        / (result["mean_ms"] / 1000.0)
    )

    return {
        "name": name,
        **result,
        "peak_mb": peak_mb,
        "samples_s": throughput,
    }


def main():
    cuda_required()
    device = torch.device("cuda:0")

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    gpu_info()

    print("\nWORKLOAD")
    print("-" * 78)

    total_samples = 0

    for aid, cfg in JOBS.items():
        print(
            f"Job {aid}: rank={cfg['rank']:2d}, "
            f"batch={cfg['batch']:3d}"
        )
        total_samples += cfg["batch"]

    print(f"Combined batch   : {total_samples}")
    print(
        f"Linear shape     : "
        f"{IN_FEATURES} -> {OUT_FEATURES}"
    )

    seed = make_seed(device)

    xs, ys, x_all, y_all, spans = make_data(
        device
    )

    # ------------------------------------------------------------
    # Correctness: combined shared execution vs per-job formula.
    # ------------------------------------------------------------
    print("\nFORWARD CORRECTNESS")
    print("-" * 78)

    with torch.no_grad():
        shared_out = shared_contiguous_forward(
            seed,
            x_all,
            spans,
        )

        reference = torch.empty_like(shared_out)

        for aid in JOBS:
            start, end = spans[aid]

            reference[start:end] = (
                seed.base(xs[aid])
                + seed.adapters[str(aid)](xs[aid])
            )

        max_error = (
            shared_out.float()
            - reference.float()
        ).abs().max().item()

    print(
        f"Shared vs per-job max_abs_error: "
        f"{max_error:.3e}"
    )

    # fp16 ordering may create small differences.
    assert max_error <= 5e-3

    # ------------------------------------------------------------
    # Parameter storage comparison.
    # ------------------------------------------------------------
    independent = make_independent(
        seed,
        device,
    )

    ind_param_bytes = sum(
        parameter_bytes(m)
        for m in independent.values()
    )

    shared_param_bytes = parameter_bytes(seed)

    reduction = 100.0 * (
        1.0
        - shared_param_bytes / ind_param_bytes
    )

    print("\nPARAMETER STORAGE")
    print("-" * 78)
    print(
        f"Independent jobs : "
        f"{ind_param_bytes / 1024**2:.2f} MB"
    )
    print(
        f"Shared SSM       : "
        f"{shared_param_bytes / 1024**2:.2f} MB"
    )
    print(
        f"Reduction        : "
        f"{reduction:.2f}%"
    )

    # ------------------------------------------------------------
    # Set up equivalent training modes.
    # ------------------------------------------------------------
    independent_opts = {
        aid: torch.optim.SGD(
            independent[aid].adapter.parameters(),
            lr=1e-5,
        )
        for aid in JOBS
    }

    naive = copy.deepcopy(seed)
    shared = copy.deepcopy(seed)

    naive_opts = make_opts_for_shared(naive)
    shared_opts = make_opts_for_shared(shared)

    print("\nCUDA TRAINING-STEP BENCHMARK")
    print("-" * 78)

    results = []

    results.append(
        measure_mode(
            "Independent",
            lambda: independent_step(
                independent,
                independent_opts,
                xs,
                ys,
            ),
            total_samples,
        )
    )

    results.append(
        measure_mode(
            "Naive grouped",
            lambda: naive_grouped_step(
                naive,
                naive_opts,
                xs,
                ys,
            ),
            total_samples,
        )
    )

    results.append(
        measure_mode(
            "Shared SSM",
            lambda: shared_step(
                shared,
                shared_opts,
                x_all,
                y_all,
                spans,
            ),
            total_samples,
        )
    )

    print(
        f"{'Mode':18s}"
        f"{'Mean ms':>12s}"
        f"{'Median ms':>14s}"
        f"{'Samples/s':>14s}"
        f"{'Peak MB':>12s}"
    )

    for r in results:
        print(
            f"{r['name']:18s}"
            f"{r['mean_ms']:12.3f}"
            f"{r['median_ms']:14.3f}"
            f"{r['samples_s']:14.1f}"
            f"{r['peak_mb']:12.1f}"
        )

    ind = results[0]
    naive_r = results[1]
    shared_r = results[2]

    print("\nRELATIVE EXECUTION TIME")
    print("-" * 78)
    print(
        "Independent / Shared : "
        f"{ind['mean_ms']/shared_r['mean_ms']:.3f}x"
    )
    print(
        "Naive / Shared       : "
        f"{naive_r['mean_ms']/shared_r['mean_ms']:.3f}x"
    )

    print("\nIMPORTANT")
    print("-" * 78)
    print(
        "This is the pre-Triton CUDA baseline. Shared SSM still executes\n"
        "each LoRA adapter with separate PyTorch kernels. Phase 5B will\n"
        "replace that adapter path with a custom Triton fused prototype.\n"
        "Do not compare these microbenchmark ratios directly with the\n"
        "paper's end-to-end multi-GPU results."
    )


if __name__ == "__main__":
    main()
