from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from dataclasses import dataclass

from phase7a_adapter_scheduler import (
    Job,
    Group,
    GroupPrediction,
    predict_group,
)


# =============================================================================
# Mini-tLoRA Phase 7B
# Hierarchical incremental grouping + binary-cut search + scaling experiment
# =============================================================================
#
# Paper-aligned mechanisms reproduced:
#
#   * bottom-up hierarchical grouping
#   * residual-capacity sort
#   * urgency-first priority
#   * binary-cut search on the right side of the sorted queue
#   * merge / update residual profile / reinsert
#   * bounded-slowdown feasibility checks
#   * O(K log K)-style scheduling operation scaling
#
# Important limitation:
#   Throughput predictions still come from the documented surrogate predictor
#   in Phase 7A. The paper relies on runtime profiling / parallel planning and
#   does not publish a complete closed-form predictor.
#
# The topology levels below are a reduced executable abstraction:
#
#   Tier 1: node-local grouping
#   Tier 2: rack / multi-node grouping
#   Tier 3: cluster-wide grouping
#
# They reproduce the paper's bottom-up topology-aware idea without claiming to
# duplicate the authors' private cluster topology or planner.
# =============================================================================


@dataclass(frozen=True)
class PlacedJob:
    job: Job
    node_id: int
    rack_id: int


@dataclass
class SchedGroup:
    members: tuple[PlacedJob, ...]

    @property
    def jobs(self) -> tuple[Job, ...]:
        return tuple(m.job for m in self.members)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(j.job_id for j in self.jobs)

    @property
    def base_model(self) -> str:
        return self.jobs[0].base_model

    @property
    def urgency(self) -> float:
        return max(j.urgency for j in self.jobs)

    @property
    def residual_score(self) -> float:
        return statistics.mean(
            j.residual_score
            for j in self.jobs
        )

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(sorted({m.node_id for m in self.members}))

    @property
    def racks(self) -> tuple[int, ...]:
        return tuple(sorted({m.rack_id for m in self.members}))


@dataclass
class SearchStats:
    throughput_calls: int = 0
    binary_steps: int = 0
    merges: int = 0
    reinsertions: int = 0

    @property
    def total_ops(self) -> int:
        return (
            self.throughput_calls
            + self.binary_steps
            + self.merges
            + self.reinsertions
        )


# =============================================================================
# Deterministic larger workload
# =============================================================================


def generate_jobs(
    k: int,
    seed: int = 2026,
    nodes: int = 8,
    nodes_per_rack: int = 2,
) -> list[PlacedJob]:
    rng = random.Random(seed)

    placed = []

    for i in range(k):
        # Deliberately create compute-heavy, memory-heavy, and balanced jobs.
        kind = i % 3

        if kind == 0:
            compute = rng.uniform(0.78, 0.95)
            memory = rng.uniform(0.25, 0.48)
        elif kind == 1:
            compute = rng.uniform(0.25, 0.48)
            memory = rng.uniform(0.78, 0.93)
        else:
            compute = rng.uniform(0.52, 0.72)
            memory = rng.uniform(0.50, 0.72)

        throughput = rng.uniform(70.0, 110.0)
        urgency = rng.uniform(0.10, 0.95)
        max_slowdown = rng.uniform(1.12, 1.28)

        node_id = i % nodes
        rack_id = node_id // nodes_per_rack

        job = Job(
            job_id=f"J{i:04d}",
            base_model="Qwen-3-8B",
            standalone_throughput=throughput,
            compute_util=compute,
            memory_util=memory,
            urgency=urgency,
            max_slowdown=max_slowdown,
        )

        placed.append(
            PlacedJob(
                job=job,
                node_id=node_id,
                rack_id=rack_id,
            )
        )

    return placed


# =============================================================================
# Prediction wrapper with instrumentation
# =============================================================================


def measured_predict(
    group: SchedGroup,
    stats: SearchStats,
) -> GroupPrediction:
    stats.throughput_calls += 1
    return predict_group(group.jobs)


def independent_throughput(
    group: SchedGroup,
) -> float:
    return sum(
        j.standalone_throughput
        for j in group.jobs
    )


def merge_groups(
    left: SchedGroup,
    right: SchedGroup,
) -> SchedGroup:
    if left.base_model != right.base_model:
        raise ValueError("base model mismatch")

    return SchedGroup(
        members=left.members + right.members
    )


# =============================================================================
# Tier eligibility
# =============================================================================


def same_domain(
    left: SchedGroup,
    right: SchedGroup,
    tier: str,
) -> bool:
    if left.base_model != right.base_model:
        return False

    if tier == "node":
        # At least one common node and both groups are node-local.
        return (
            len(left.nodes) == 1
            and len(right.nodes) == 1
            and left.nodes == right.nodes
        )

    if tier == "rack":
        # Groups can span nodes but must remain within one rack.
        return (
            len(left.racks) == 1
            and len(right.racks) == 1
            and left.racks == right.racks
        )

    if tier == "cluster":
        return True

    raise ValueError(f"unknown tier: {tier}")


# =============================================================================
# Ordering
# =============================================================================


def queue_key(group: SchedGroup):
    # Urgency descending, residual availability ascending.
    return (
        -group.urgency,
        group.residual_score,
    )


def residual_key(group: SchedGroup):
    return group.residual_score


# =============================================================================
# Binary-cut search
# =============================================================================


def prefix_candidate(
    seed: SchedGroup,
    ordered_right: list[SchedGroup],
    count: int,
) -> SchedGroup:
    merged = seed

    for candidate in ordered_right[:count]:
        merged = merge_groups(
            merged,
            candidate,
        )

    return merged


def efficiency_gain(
    merged: SchedGroup,
    constituents: list[SchedGroup],
    prediction: GroupPrediction,
    stats: SearchStats,
) -> float:
    separate = 0.0

    for g in constituents:
        separate += measured_predict(
            g,
            stats,
        ).total_throughput

    if separate <= 0:
        return -math.inf

    return (
        prediction.total_throughput
        / separate
        - 1.0
    )


def binary_cut_best_prefix(
    seed: SchedGroup,
    candidates: list[SchedGroup],
    stats: SearchStats,
    minimum_gain: float,
):
    """
    Search for the largest beneficial prefix.

    The paper describes a binary-cut search over the right-hand side of the
    residual-sorted queue. We model the cutoff predicate as:

        "Is this prefix feasible and does it improve predicted throughput?"

    Binary search assumes the useful region is approximately prefix-monotonic,
    which is the pruning heuristic that gives the scalable behavior.
    """

    if not candidates:
        return 0, None, None

    lo = 1
    hi = len(candidates)

    best_count = 0
    best_group = None
    best_prediction = None

    while lo <= hi:
        stats.binary_steps += 1

        mid = (lo + hi) // 2

        merged = prefix_candidate(
            seed,
            candidates,
            mid,
        )

        pred = measured_predict(
            merged,
            stats,
        )

        constituents = (
            [seed]
            + candidates[:mid]
        )

        gain = efficiency_gain(
            merged,
            constituents,
            pred,
            stats,
        )

        beneficial = (
            pred.feasible
            and gain >= minimum_gain
        )

        if beneficial:
            best_count = mid
            best_group = merged
            best_prediction = pred
            lo = mid + 1
        else:
            hi = mid - 1

    return (
        best_count,
        best_group,
        best_prediction,
    )


# =============================================================================
# One hierarchical tier
# =============================================================================


def run_tier(
    groups: list[SchedGroup],
    tier: str,
    stats: SearchStats,
    minimum_gain: float,
) -> list[SchedGroup]:
    queue = list(groups)
    finalized = []

    while queue:
        queue.sort(key=queue_key)

        seed = queue.pop(0)

        eligible = [
            g
            for g in queue
            if same_domain(
                seed,
                g,
                tier,
            )
        ]

        # The paper sorts by residual availability and searches on the right.
        eligible.sort(
            key=residual_key
        )

        (
            count,
            merged,
            prediction,
        ) = binary_cut_best_prefix(
            seed,
            eligible,
            stats,
            minimum_gain,
        )

        if count == 0:
            finalized.append(seed)
            continue

        selected_ids = {
            id(g)
            for g in eligible[:count]
        }

        queue = [
            g
            for g in queue
            if id(g) not in selected_ids
        ]

        stats.merges += count
        stats.reinsertions += 1

        queue.append(merged)

    return finalized


# =============================================================================
# Full hierarchy
# =============================================================================


def hierarchical_schedule(
    jobs: list[PlacedJob],
    minimum_gain: float = 0.01,
):
    stats = SearchStats()

    groups = [
        SchedGroup((job,))
        for job in jobs
    ]

    tier_counts = []

    for tier in (
        "node",
        "rack",
        "cluster",
    ):
        before = len(groups)

        groups = run_tier(
            groups,
            tier,
            stats,
            minimum_gain,
        )

        after = len(groups)

        tier_counts.append(
            (tier, before, after)
        )

    return groups, stats, tier_counts


# =============================================================================
# Validation / metrics
# =============================================================================


def schedule_metrics(
    groups: list[SchedGroup],
):
    throughput = 0.0
    violations = 0
    max_slowdown = 1.0

    for group in groups:
        pred = predict_group(
            group.jobs
        )

        throughput += (
            pred.total_throughput
        )

        if not pred.feasible:
            violations += 1

        if pred.per_job:
            max_slowdown = max(
                max_slowdown,
                max(
                    p.slowdown
                    for p in pred.per_job.values()
                ),
            )

    return {
        "throughput": throughput,
        "violations": violations,
        "max_slowdown": max_slowdown,
    }


def standalone_throughput(
    jobs: list[PlacedJob],
):
    return sum(
        j.job.standalone_throughput
        for j in jobs
    )


# =============================================================================
# Scaling benchmark
# =============================================================================


def scaling_experiment(
    sizes: list[int],
    repeats: int,
    minimum_gain: float,
):
    rows = []

    for k in sizes:
        runtimes = []
        operations = []

        for r in range(repeats):
            jobs = generate_jobs(
                k,
                seed=2026 + r,
                nodes=max(
                    4,
                    min(32, k // 4),
                ),
            )

            t0 = time.perf_counter()

            groups, stats, _ = (
                hierarchical_schedule(
                    jobs,
                    minimum_gain=minimum_gain,
                )
            )

            elapsed = (
                time.perf_counter()
                - t0
            )

            metrics = schedule_metrics(
                groups
            )

            assert (
                metrics["violations"] == 0
            )

            runtimes.append(elapsed)
            operations.append(
                stats.total_ops
            )

        median_time = statistics.median(
            runtimes
        )

        median_ops = statistics.median(
            operations
        )

        klogk = (
            k * math.log2(max(2, k))
        )

        rows.append(
            {
                "k": k,
                "median_ms": (
                    median_time * 1000
                ),
                "ops": median_ops,
                "ops_per_klogk": (
                    median_ops / klogk
                ),
            }
        )

    return rows


# =============================================================================
# Reporting
# =============================================================================


def print_demo(
    k: int,
    minimum_gain: float,
):
    jobs = generate_jobs(
        k,
        seed=2026,
        nodes=8,
    )

    t0 = time.perf_counter()

    groups, stats, tiers = (
        hierarchical_schedule(
            jobs,
            minimum_gain=minimum_gain,
        )
    )

    elapsed_ms = (
        time.perf_counter()
        - t0
    ) * 1000

    metrics = schedule_metrics(
        groups
    )

    standalone = standalone_throughput(
        jobs
    )

    print("=" * 92)
    print(
        "MINI-tLoRA PHASE 7B - "
        "HIERARCHICAL BINARY-CUT SCHEDULER"
    )
    print("=" * 92)

    print(
        "\nThe throughput predictor is the same documented surrogate used in\n"
        "Phase 7A. This phase evaluates the hierarchy/search structure."
    )

    print("\nHIERARCHICAL GROUP COUNTS")
    print("-" * 92)

    for tier, before, after in tiers:
        print(
            f"{tier:10s}: "
            f"{before:4d} -> {after:4d} groups"
        )

    print("\nFINAL GROUPS")
    print("-" * 92)

    for idx, group in enumerate(
        groups,
        start=1,
    ):
        pred = predict_group(
            group.jobs
        )

        print(
            f"G{idx:02d}: "
            f"size={len(group.jobs):2d} "
            f"jobs={','.join(group.ids)} "
            f"nodes={group.nodes} "
            f"racks={group.racks} "
            f"throughput={pred.total_throughput:.2f} "
            f"feasible={pred.feasible}"
        )

    print("\nSUMMARY")
    print("-" * 92)

    print(
        f"Jobs                        : {k}"
    )

    print(
        f"Final groups                : "
        f"{len(groups)}"
    )

    print(
        f"Standalone throughput       : "
        f"{standalone:.2f}"
    )

    print(
        f"Grouped surrogate throughput: "
        f"{metrics['throughput']:.2f}"
    )

    print(
        f"Relative throughput         : "
        f"{metrics['throughput'] / standalone:.3f}x"
    )

    print(
        f"Slowdown violations         : "
        f"{metrics['violations']}"
    )

    print(
        f"Worst slowdown              : "
        f"{metrics['max_slowdown']:.3f}x"
    )

    print(
        f"Binary-search steps         : "
        f"{stats.binary_steps}"
    )

    print(
        f"Throughput predictions      : "
        f"{stats.throughput_calls}"
    )

    print(
        f"Merges                      : "
        f"{stats.merges}"
    )

    print(
        f"Reinsertions                : "
        f"{stats.reinsertions}"
    )

    print(
        f"Total instrumented ops      : "
        f"{stats.total_ops}"
    )

    print(
        f"Scheduler wall time         : "
        f"{elapsed_ms:.3f} ms"
    )

    assert (
        metrics["violations"] == 0
    )

    print("\nBOUNDED-SLOWDOWN CHECK: PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demo-jobs",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--minimum-gain",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--scaling-repeats",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    print_demo(
        args.demo_jobs,
        args.minimum_gain,
    )

    print("\nSCALING EXPERIMENT")
    print("-" * 92)

    sizes = [
        32,
        64,
        128,
        256,
        512,
        1024,
    ]

    rows = scaling_experiment(
        sizes,
        repeats=args.scaling_repeats,
        minimum_gain=args.minimum_gain,
    )

    print(
        f"{'K':>8s}"
        f"{'Median ms':>14s}"
        f"{'Ops':>14s}"
        f"{'Ops/(Klog2K)':>18s}"
    )

    for row in rows:
        print(
            f"{row['k']:8d}"
            f"{row['median_ms']:14.3f}"
            f"{row['ops']:14.0f}"
            f"{row['ops_per_klogk']:18.4f}"
        )

    print("\nINTERPRETATION")
    print("-" * 92)

    print(
        "The paper's complexity claim is about the scheduling algorithm, not\n"
        "Python wall-clock constants. The useful signal here is whether the\n"
        "instrumented operation count grows approximately with K log K rather\n"
        "than combinatorially. Ops/(K log2 K) should remain roughly bounded\n"
        "as K increases.\n\n"
        "This remains a reduced reproduction because the throughput predictor\n"
        "is synthetic and the topology tiers are an executable abstraction of\n"
        "the paper's bottom-up hierarchy."
    )


if __name__ == "__main__":
    main()
