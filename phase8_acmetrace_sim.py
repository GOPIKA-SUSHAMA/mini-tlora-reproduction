from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import statistics
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from phase7a_adapter_scheduler import (
    Job,
    predict_group,
)


# =============================================================================
# Mini-tLoRA Phase 8
# Trace-driven online evaluation using PUBLIC ACMETrace arrivals.
# =============================================================================
#
# Paper-aligned methodology reproduced:
#   * real production job arrival / duration / GPU-allocation trace
#   * random LoRA rank from {2,4,8,16}
#   * random batch size from {1,2,4,8}
#   * random base model: Llama-3-8B or Qwen-3-8B
#   * online arrivals / completions
#   * standalone vs FIFO grouping vs tLoRA-inspired scheduling
#   * throughput, JCT, queue delay, slowdown, GPU-allocation utilization
#   * arrival-load scaling
#
# IMPORTANT LIMITATION:
# The tLoRA paper obtains per-job speed profiles from 12 A100 GPUs and feeds
# them into a production-grade distributed simulator. We do not have that
# hardware/profile database. This reduced reproduction therefore uses:
#
#   REAL trace arrivals/durations/GPU allocations
#   + SYNTHETIC LoRA attributes (same ranges as paper)
#   + DOCUMENTED SURROGATE relative throughput model (Phase 7A)
#
# Do not compare the numeric speedups from this script directly with the
# paper's 1.2-1.8x throughput or 2.3-5.4x JCT headline results.
# =============================================================================


RANKS = (2, 4, 8, 16)
BATCHES = (1, 2, 4, 8)
MODELS = ("Llama-3-8B", "Qwen-3-8B")


@dataclass
class TraceJob:
    trace_id: str
    arrival: float
    standalone_duration: float
    gpu_num: int

    rank: int
    batch_size: int
    base_model: str

    profile: Job
    work: float


@dataclass
class Completion:
    finish: float
    counter: int
    job_index: int


@dataclass
class JobResult:
    start: float
    finish: float
    throughput: float


# =============================================================================
# Trace loading
# =============================================================================


def parse_datetime(value: str):
    value = (value or "").strip()

    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_acme_trace(
    path: str,
    limit: int,
    cluster_gpus: int,
    seed: int,
):
    rng = random.Random(seed)

    raw = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "job_id",
            "gpu_num",
            "state",
            "submit_time",
            "duration",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"Trace is missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:
            if (
                row.get("state", "").strip()
                != "COMPLETED"
            ):
                continue

            try:
                gpu_num = int(
                    float(
                        row.get(
                            "gpu_num",
                            "0",
                        )
                    )
                )

                duration = float(
                    row.get(
                        "duration",
                        "0",
                    )
                )
            except ValueError:
                continue

            submit = parse_datetime(
                row.get(
                    "submit_time",
                    "",
                )
            )

            if submit is None:
                continue

            if (
                gpu_num <= 0
                or gpu_num > cluster_gpus
                or duration <= 0
            ):
                continue

            raw.append(
                (
                    row.get(
                        "job_id",
                        f"trace-{len(raw)}",
                    ),
                    submit,
                    duration,
                    gpu_num,
                )
            )

            if len(raw) >= limit:
                break

    if not raw:
        raise RuntimeError(
            "No usable COMPLETED GPU jobs "
            "were found in the trace."
        )

    raw.sort(
        key=lambda x: x[1]
    )

    first_submit = raw[0][1]

    jobs = []

    for idx, (
        trace_id,
        submit,
        duration,
        gpu_num,
    ) in enumerate(raw):
        arrival = (
            submit - first_submit
        ).total_seconds()

        rank = rng.choice(RANKS)
        batch = rng.choice(BATCHES)
        model = rng.choice(MODELS)

        # -------------------------------------------------------------
        # Surrogate standalone throughput.
        #
        # Work is defined so the independent service time is exactly the
        # trace duration. Therefore this absolute formula affects the
        # normalized-work throughput metric, not standalone JCT.
        # -------------------------------------------------------------
        model_factor = (
            1.00
            if model == "Qwen-3-8B"
            else 0.96
        )

        gpu_factor = (
            max(1, gpu_num) ** 0.72
        )

        batch_factor = (
            0.80
            + 0.16 * math.log2(batch + 1)
        )

        rank_penalty = (
            1.0
            / (
                1.0
                + 0.018 * rank
            )
        )

        standalone_tp = (
            24.0
            * model_factor
            * gpu_factor
            * batch_factor
            * rank_penalty
        )

        # -------------------------------------------------------------
        # Surrogate utilization profile.
        #
        # tLoRA's actual scheduler uses measured runtime residuals. The
        # public ACMETrace job table does not contain LoRA-specific compute
        # and memory residuals, so we construct a deterministic profile from
        # sampled LoRA attributes + a small seeded skew to expose
        # complementarity.
        # -------------------------------------------------------------
        skew = rng.uniform(
            -0.16,
            0.16,
        )

        gpu_relief = (
            0.025
            * math.log2(
                max(1, gpu_num)
            )
        )

        compute_util = min(
            0.96,
            max(
                0.22,
                0.34
                + 0.025 * rank
                + 0.030 * batch
                - gpu_relief
                + skew,
            ),
        )

        memory_util = min(
            0.96,
            max(
                0.22,
                0.38
                + 0.055 * batch
                + 0.006 * rank
                - gpu_relief
                - skew,
            ),
        )

        max_slowdown = (
            rng.uniform(
                1.12,
                1.28,
            )
        )

        profile = Job(
            job_id=f"J{idx:05d}",
            base_model=model,
            standalone_throughput=(
                standalone_tp
            ),
            compute_util=compute_util,
            memory_util=memory_util,
            urgency=0.0,
            max_slowdown=max_slowdown,
        )

        jobs.append(
            TraceJob(
                trace_id=str(trace_id),
                arrival=arrival,
                standalone_duration=duration,
                gpu_num=gpu_num,
                rank=rank,
                batch_size=batch,
                base_model=model,
                profile=profile,
                work=(
                    duration
                    * standalone_tp
                ),
            )
        )

    return jobs


# =============================================================================
# Dynamic scheduling helpers
# =============================================================================


def dynamic_profile(
    job: TraceJob,
    now: float,
):
    waiting = max(
        0.0,
        now - job.arrival,
    )

    # Queueing pressure relative to standalone service time.
    urgency = min(
        3.0,
        waiting
        / max(
            1.0,
            job.standalone_duration,
        ),
    )

    return replace(
        job.profile,
        urgency=urgency,
    )


def group_prediction(
    jobs: list[TraceJob],
    now: float,
):
    profiles = [
        dynamic_profile(
            j,
            now,
        )
        for j in jobs
    ]

    return predict_group(
        profiles
    )


def choose_independent(
    waiting,
    jobs,
    available_gpus,
    now,
    max_group,
):
    del now, max_group

    for idx in waiting:
        if (
            jobs[idx].gpu_num
            <= available_gpus
        ):
            return [idx]

    return []


def choose_fifo_group(
    waiting,
    jobs,
    available_gpus,
    now,
    max_group,
):
    del now

    if not waiting:
        return []

    # Earliest waiting job that can fit.
    seed_pos = None

    for pos, idx in enumerate(waiting):
        if (
            jobs[idx].gpu_num
            <= available_gpus
        ):
            seed_pos = pos
            break

    if seed_pos is None:
        return []

    seed_idx = waiting[seed_pos]
    seed = jobs[seed_idx]

    chosen = [seed_idx]
    used = seed.gpu_num

    # FIFO/memory-style heuristic:
    # keep appending same-base-model jobs as long as GPU capacity fits.
    for idx in waiting[seed_pos + 1 :]:
        if len(chosen) >= max_group:
            break

        candidate = jobs[idx]

        if (
            candidate.base_model
            != seed.base_model
        ):
            continue

        if (
            used + candidate.gpu_num
            > available_gpus
        ):
            continue

        chosen.append(idx)
        used += candidate.gpu_num

    return chosen


def choose_tlora_group(
    waiting,
    jobs,
    available_gpus,
    now,
    max_group,
):
    fitting = [
        idx
        for idx in waiting
        if jobs[idx].gpu_num
        <= available_gpus
    ]

    if not fitting:
        return []

    # Paper-inspired ordering:
    # urgency descending, residual availability ascending.
    fitting.sort(
        key=lambda idx: (
            -dynamic_profile(
                jobs[idx],
                now,
            ).urgency,
            dynamic_profile(
                jobs[idx],
                now,
            ).residual_score,
            jobs[idx].arrival,
        )
    )

    seed_idx = fitting[0]
    group = [seed_idx]

    while len(group) < max_group:
        used = sum(
            jobs[idx].gpu_num
            for idx in group
        )

        best_idx = None
        best_gain = 0.0

        current_pred = (
            group_prediction(
                [
                    jobs[idx]
                    for idx in group
                ],
                now,
            )
        )

        for idx in fitting:
            if idx in group:
                continue

            candidate = jobs[idx]

            if (
                candidate.base_model
                != jobs[seed_idx].base_model
            ):
                continue

            if (
                used
                + candidate.gpu_num
                > available_gpus
            ):
                continue

            proposal = (
                group + [idx]
            )

            pred = group_prediction(
                [
                    jobs[j]
                    for j in proposal
                ],
                now,
            )

            if not pred.feasible:
                continue

            separate_tp = (
                current_pred.total_throughput
                + candidate.profile.standalone_throughput
            )

            gain = (
                pred.total_throughput
                / separate_tp
                - 1.0
            )

            if (
                gain >= 0.01
                and gain > best_gain
            ):
                best_gain = gain
                best_idx = idx

        if best_idx is None:
            break

        group.append(
            best_idx
        )

    return group


# =============================================================================
# Event simulator
# =============================================================================


def simulate(
    source_jobs: list[TraceJob],
    policy: str,
    cluster_gpus: int,
    arrival_speedup: float,
    max_group: int,
):
    jobs = []

    for job in source_jobs:
        jobs.append(
            replace(
                job,
                arrival=(
                    job.arrival
                    / arrival_speedup
                ),
            )
        )

    if policy == "independent":
        chooser = choose_independent
    elif policy == "fifo":
        chooser = choose_fifo_group
    elif policy == "tlora":
        chooser = choose_tlora_group
    else:
        raise ValueError(
            f"Unknown policy: {policy}"
        )

    n = len(jobs)

    waiting = []
    active = []
    results = {}

    next_arrival = 0
    now = 0.0
    used_gpus = 0
    completion_counter = 0

    # Integral of allocated GPUs over time.
    gpu_seconds = 0.0
    last_event_time = 0.0

    def advance_time(new_time):
        nonlocal now
        nonlocal gpu_seconds
        nonlocal last_event_time

        if new_time < now:
            raise RuntimeError(
                "time moved backwards"
            )

        gpu_seconds += (
            used_gpus
            * (
                new_time
                - last_event_time
            )
        )

        now = new_time
        last_event_time = new_time

    def add_arrivals():
        nonlocal next_arrival

        while (
            next_arrival < n
            and jobs[
                next_arrival
            ].arrival <= now + 1e-12
        ):
            waiting.append(
                next_arrival
            )

            next_arrival += 1

    def dispatch():
        nonlocal used_gpus
        nonlocal completion_counter

        made_progress = False

        while True:
            available = (
                cluster_gpus
                - used_gpus
            )

            if available <= 0:
                break

            group_indices = chooser(
                waiting,
                jobs,
                available,
                now,
                max_group,
            )

            if not group_indices:
                break

            group_jobs = [
                jobs[idx]
                for idx in group_indices
            ]

            gpu_demand = sum(
                j.gpu_num
                for j in group_jobs
            )

            if gpu_demand > available:
                raise RuntimeError(
                    "scheduler over-allocated GPUs"
                )

            pred = group_prediction(
                group_jobs,
                now,
            )

            # FIFO intentionally ignores feasibility constraints, while tLoRA
            # chooses only feasible beneficial groups.
            for idx, job in zip(
                group_indices,
                group_jobs,
            ):
                p = pred.per_job[
                    job.profile.job_id
                ]

                service = (
                    job.work
                    / p.throughput
                )

                finish = (
                    now + service
                )

                results[idx] = JobResult(
                    start=now,
                    finish=finish,
                    throughput=p.throughput,
                )

                completion_counter += 1

                heapq.heappush(
                    active,
                    (
                        finish,
                        completion_counter,
                        idx,
                    ),
                )

                used_gpus += (
                    job.gpu_num
                )

            selected = set(
                group_indices
            )

            waiting[:] = [
                idx
                for idx in waiting
                if idx not in selected
            ]

            made_progress = True

        return made_progress

    # Start at first arrival.
    if n:
        now = jobs[0].arrival
        last_event_time = now

    while (
        len(results) < n
        or active
    ):
        add_arrivals()
        dispatch()

        next_arrival_time = (
            jobs[
                next_arrival
            ].arrival
            if next_arrival < n
            else math.inf
        )

        next_finish_time = (
            active[0][0]
            if active
            else math.inf
        )

        if (
            next_arrival_time
            == math.inf
            and next_finish_time
            == math.inf
        ):
            break

        next_time = min(
            next_arrival_time,
            next_finish_time,
        )

        advance_time(
            next_time
        )

        # Free all jobs finishing now.
        while (
            active
            and active[0][0]
            <= now + 1e-12
        ):
            _, _, idx = heapq.heappop(
                active
            )

            used_gpus -= (
                jobs[idx].gpu_num
            )

        add_arrivals()

    if len(results) != n:
        raise RuntimeError(
            f"Only completed {len(results)}/{n} jobs"
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    jcts = []
    queues = []
    slowdowns = []

    first_arrival = min(
        j.arrival
        for j in jobs
    )

    last_finish = max(
        r.finish
        for r in results.values()
    )

    makespan = max(
        1e-9,
        last_finish
        - first_arrival
    )

    total_work = sum(
        j.work
        for j in jobs
    )

    for idx, job in enumerate(
        jobs
    ):
        result = results[idx]

        jct = (
            result.finish
            - job.arrival
        )

        queue = (
            result.start
            - job.arrival
        )

        slowdown = (
            jct
            / job.standalone_duration
        )

        jcts.append(jct)
        queues.append(queue)
        slowdowns.append(slowdown)

    return {
        "policy": policy,
        "jobs": n,
        "makespan": makespan,
        "throughput": (
            total_work
            / makespan
        ),
        "jobs_per_hour": (
            n
            / makespan
            * 3600.0
        ),
        "avg_jct": statistics.mean(
            jcts
        ),
        "median_jct": statistics.median(
            jcts
        ),
        "p95_jct": sorted(jcts)[
            max(
                0,
                math.ceil(
                    0.95 * len(jcts)
                )
                - 1,
            )
        ],
        "avg_queue": statistics.mean(
            queues
        ),
        "avg_slowdown": statistics.mean(
            slowdowns
        ),
        "p95_slowdown": sorted(
            slowdowns
        )[
            max(
                0,
                math.ceil(
                    0.95
                    * len(slowdowns)
                )
                - 1,
            )
        ],
        "allocation_util": (
            gpu_seconds
            / (
                cluster_gpus
                * makespan
            )
        ),
    }


# =============================================================================
# Reporting
# =============================================================================


def print_trace_summary(
    jobs,
    path,
):
    print("=" * 102)
    print(
        "MINI-tLoRA PHASE 8 - "
        "ACMETRACE-DRIVEN ONLINE SCHEDULING SIMULATION"
    )
    print("=" * 102)

    print(
        f"Trace file       : {path}"
    )

    print(
        f"Jobs loaded      : {len(jobs)}"
    )

    print(
        f"Arrival span     : "
        f"{jobs[-1].arrival - jobs[0].arrival:.1f} s"
    )

    print(
        "LoRA ranks       : "
        "{2,4,8,16}"
    )

    print(
        "Batch sizes      : "
        "{1,2,4,8}"
    )

    models = sorted(
        {j.base_model for j in jobs}
    )

    print(
        f"Base models      : "
        f"{', '.join(models)}"
    )

    gpu_nums = [
        j.gpu_num
        for j in jobs
    ]

    print(
        f"GPU allocation   : "
        f"min={min(gpu_nums)}, "
        f"median={statistics.median(gpu_nums):.1f}, "
        f"max={max(gpu_nums)}"
    )

    print("\nMETHODOLOGY LIMIT")
    print("-" * 102)

    print(
        "Arrivals, durations and GPU allocations come from public ACMETrace.\n"
        "LoRA attributes use the same sampled ranges described in the tLoRA\n"
        "paper. Relative grouped performance uses our Phase-7 surrogate model,\n"
        "NOT the paper's 12-A100 profiling database / 128-GPU simulator."
    )


def print_results(
    rows,
    load,
):
    print(
        f"\nARRIVAL SPEEDUP = {load:.1f}x"
    )

    print("-" * 102)

    print(
        f"{'Policy':14s}"
        f"{'Norm TP':>12s}"
        f"{'Jobs/hr':>12s}"
        f"{'Avg JCT s':>14s}"
        f"{'P95 JCT s':>14s}"
        f"{'Avg Queue':>14s}"
        f"{'Avg Slow':>12s}"
        f"{'Alloc Util':>13s}"
    )

    baseline = next(
        r
        for r in rows
        if r["policy"]
        == "independent"
    )

    for r in rows:
        print(
            f"{r['policy']:14s}"
            f"{r['throughput']:12.2f}"
            f"{r['jobs_per_hour']:12.2f}"
            f"{r['avg_jct']:14.1f}"
            f"{r['p95_jct']:14.1f}"
            f"{r['avg_queue']:14.1f}"
            f"{r['avg_slowdown']:12.3f}"
            f"{100*r['allocation_util']:12.1f}%"
        )

    print("\nRELATIVE TO INDEPENDENT")
    print("-" * 102)

    for r in rows:
        if r is baseline:
            continue

        tp_ratio = (
            r["throughput"]
            / baseline["throughput"]
        )

        jct_ratio = (
            baseline["avg_jct"]
            / r["avg_jct"]
        )

        queue_ratio = (
            baseline["avg_queue"]
            / max(
                1e-9,
                r["avg_queue"],
            )
        )

        print(
            f"{r['policy']:14s}: "
            f"throughput={tp_ratio:.3f}x, "
            f"JCT improvement={jct_ratio:.3f}x, "
            f"queue improvement={queue_ratio:.3f}x"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trace",
        required=True,
        help=(
            "Path to ACMETrace job CSV, e.g. "
            "../AcmeTrace/data/job_trace/trace_seren.csv"
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--cluster-gpus",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--max-group",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--loads",
        type=float,
        nargs="+",
        default=[
            1.0,
            2.0,
            5.0,
        ],
        help=(
            "Arrival acceleration factors. "
            "2 means jobs arrive twice as quickly."
        ),
    )

    args = parser.parse_args()

    path = Path(
        args.trace
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Trace not found: {path}"
        )

    jobs = load_acme_trace(
        str(path),
        limit=args.jobs,
        cluster_gpus=args.cluster_gpus,
        seed=args.seed,
    )

    print_trace_summary(
        jobs,
        path,
    )

    for load in args.loads:
        rows = []

        for policy in (
            "independent",
            "fifo",
            "tlora",
        ):
            rows.append(
                simulate(
                    jobs,
                    policy=policy,
                    cluster_gpus=args.cluster_gpus,
                    arrival_speedup=load,
                    max_group=args.max_group,
                )
            )

        print_results(
            rows,
            load,
        )

    print("\nWHAT THIS FINAL TECHNICAL PHASE REPRODUCES")
    print("-" * 102)

    print(
        "  - real production arrival/duration/GPU-allocation trace input\n"
        "  - paper-matched LoRA rank/batch/model parameterization ranges\n"
        "  - online queueing and completion events\n"
        "  - independent vs FIFO grouping vs tLoRA-inspired scheduling\n"
        "  - throughput / JCT / queue delay / slowdown / allocation utilization\n"
        "  - 1x / 2x / 5x arrival-load sensitivity\n\n"
        "The remaining work after this script is documentation/consolidation,\n"
        "not another core technical reproduction phase."
    )


if __name__ == "__main__":
    main()
