import argparse
import copy
import statistics
import time

import torch
import torch.nn.functional as F

from tlora.multi_lora_linear import IndependentLoRALinear, MultiLoRALinear

RANKS = {0: 2, 1: 4, 2: 8, 3: 16}

def parameter_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def trainable_parameter_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

def mb(n):
    return n / (1024 ** 2)

def make_independent_models(shared, in_features, out_features):
    models = {}
    for aid, rank in RANKS.items():
        m = IndependentLoRALinear(in_features, out_features, rank)
        m.base.load_state_dict(copy.deepcopy(shared.base.state_dict()))
        m.adapter.load_state_dict(copy.deepcopy(shared.adapters[str(aid)].state_dict()))
        models[aid] = m
    return models

def make_data(batch_per_job, in_features, out_features):
    xs, ys = {}, {}
    for aid in RANKS:
        xs[aid] = torch.randn(batch_per_job, in_features)
        ys[aid] = torch.randn(batch_per_job, out_features)
    x_all = torch.cat([xs[aid] for aid in RANKS], dim=0)
    y_all = torch.cat([ys[aid] for aid in RANKS], dim=0)
    adapter_ids = torch.cat([
        torch.full((batch_per_job,), aid, dtype=torch.long)
        for aid in RANKS
    ])
    return xs, ys, x_all, y_all, adapter_ids

def build_opts(model, lr):
    return {
        aid: torch.optim.SGD(model.adapter_parameters(aid), lr=lr)
        for aid in RANKS
    }

def zero_opts(opts):
    for o in opts.values():
        o.zero_grad(set_to_none=True)

def step_opts(opts):
    for o in opts.values():
        o.step()

def independent_step(models, opts, xs, ys):
    for aid in RANKS:
        o = opts[aid]
        o.zero_grad(set_to_none=True)
        pred = models[aid](xs[aid])
        loss = F.mse_loss(pred, ys[aid], reduction="sum")
        loss.backward()
        o.step()

def naive_step(model, opts, xs, ys):
    zero_opts(opts)
    total = None
    for aid in RANKS:
        pred = model.base(xs[aid]) + model.adapters[str(aid)](xs[aid])
        loss = F.mse_loss(pred, ys[aid], reduction="sum")
        total = loss if total is None else total + loss
    total.backward()
    step_opts(opts)

def shared_step(model, opts, x_all, y_all, adapter_ids):
    zero_opts(opts)
    pred = model(x_all, adapter_ids)
    loss = F.mse_loss(pred, y_all, reduction="sum")
    loss.backward()
    step_opts(opts)

def bench(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times), statistics.median(times)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-features", type=int, default=512)
    p.add_argument("--out-features", type=int, default=512)
    p.add_argument("--batch-per-job", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    a = p.parse_args()

    torch.manual_seed(2026)

    seed = MultiLoRALinear(a.in_features, a.out_features, RANKS)
    for adapter in seed.adapters.values():
        torch.nn.init.normal_(adapter.B, mean=0.0, std=0.01)

    independent = make_independent_models(seed, a.in_features, a.out_features)
    naive = copy.deepcopy(seed)
    shared = copy.deepcopy(seed)

    ind_opts = {
        aid: torch.optim.SGD(independent[aid].adapter.parameters(), lr=a.lr)
        for aid in RANKS
    }
    naive_opts = build_opts(naive, a.lr)
    shared_opts = build_opts(shared, a.lr)

    xs, ys, x_all, y_all, adapter_ids = make_data(
        a.batch_per_job, a.in_features, a.out_features
    )
    total_samples = len(RANKS) * a.batch_per_job

    ind_bytes = sum(parameter_bytes(m) for m in independent.values())
    shared_bytes = parameter_bytes(shared)
    savings = 100 * (1 - shared_bytes / ind_bytes)

    print("=" * 68)
    print("MINI-tLoRA PHASE 2 — CPU BASELINE")
    print("=" * 68)
    print(f"Jobs/ranks      : {RANKS}")
    print(f"Batch per job   : {a.batch_per_job}")
    print(f"Total samples   : {total_samples}")
    print(f"Independent mem : {mb(ind_bytes):.3f} MB")
    print(f"Shared SSM mem  : {mb(shared_bytes):.3f} MB")
    print(f"Param reduction : {savings:.2f}%")
    print()

    ind = bench(lambda: independent_step(independent, ind_opts, xs, ys), a.warmup, a.iterations)
    nai = bench(lambda: naive_step(naive, naive_opts, xs, ys), a.warmup, a.iterations)
    sha = bench(lambda: shared_step(shared, shared_opts, x_all, y_all, adapter_ids), a.warmup, a.iterations)

    print(f"{'Mode':18s}{'Mean ms':>12s}{'Median ms':>14s}{'Samples/s':>14s}")
    for name, res in [("Independent", ind), ("Naive grouped", nai), ("Shared SSM", sha)]:
        mean_s, med_s = res
        print(f"{name:18s}{mean_s*1000:12.3f}{med_s*1000:14.3f}{total_samples/mean_s:14.1f}")

    print()
    print(f"Independent / Shared time ratio: {ind[0] / sha[0]:.3f}x")
    print(f"Naive / Shared time ratio      : {nai[0] / sha[0]:.3f}x")
    print()
    print("CPU timings are only a baseline. Do not compare them directly with")
    print("the paper's A100 results; Triton fusion and GPU overlap come later.")

if __name__ == "__main__":
    main()
