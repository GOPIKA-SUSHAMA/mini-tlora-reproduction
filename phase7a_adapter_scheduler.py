from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# =============================================================================
# Mini-tLoRA Phase 7A
# Residual-capacity + urgency aware adapter scheduler
# =============================================================================
#
# Paper-aligned ideas reproduced here:
#
#   1. Jobs sharing a base model are candidates for grouping.
#   2. Each job has:
#        - residual resource capacity
#        - urgency / progress pressure
#        - maximum tolerated slowdown
#   3. High-urgency, resource-constrained jobs are considered first.
#   4. Candidate partners are accepted only when:
#        - joint throughput improves
#        - every member satisfies its slowdown bound
#   5. Merged groups are reinserted and may grow further.
#
# IMPORTANT:
# The tLoRA paper does not expose the exact implementation of its runtime
# throughput predictor. This reduced reproduction therefore uses a documented
# SURROGATE predictor based on compute/memory headroom. The scheduling logic,
# constraints, and merge/reinsert behavior are what this phase validates.
# =============================================================================


@dataclass(frozen=True)
class Job:
    job_id: str
    base_model: str

    # Standalone profiling.
    standalone_throughput: float
    compute_util: float
    memory_util: float

    # Progress pressure.
    urgency: float
    max_slowdown: float

    @property
    def residual_compute(self) -> float:
        return max(0.0, 1.0 - self.compute_util)

    @property
    def residual_memory(self) -> float:
        return max(0.0, 1.0 - self.memory_util)

    @property
    def residual_score(self) -> float:
        # Smaller means more resource constrained.
        return self.residual_compute + self.residual_memory


@dataclass
class JobPrediction:
    throughput: float
    slowdown: float


@dataclass
class GroupPrediction:
    total_throughput: float
    per_job: dict[str, JobPrediction]
    feasible: bool


@dataclass
class Group:
    jobs: tuple[Job, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(j.job_id for j in self.jobs)

    @property
    def base_model(self) -> str:
        return self.jobs[0].base_model

    @property
    def urgency(self) -> float:
        # Progress-critical member determines group priority.
        return max(j.urgency for j in self.jobs)

    @property
    def residual_score(self) -> float:
        return sum(
            j.residual_score
            for j in self.jobs
        ) / len(self.jobs)


# =============================================================================
# Deterministic workload
#
# We intentionally include:
#
#   compute-heavy jobs:
#       high compute util, lower memory util
#
#   memory-heavy jobs:
#       lower compute util, high memory util
#
#   balanced/saturated jobs:
#       fewer useful residual resources
#
# This lets the scheduler demonstrate the paper's resource-complementarity idea.
# =============================================================================


JOBS = [
    # Arrival order is intentionally NOT complementarity-friendly so that the
    # FIFO baseline groups similarly saturated jobs, while the tLoRA-inspired
    # scheduler is free to search for better partners.
    Job(
        "J0",
        "Qwen-3-8B",
        standalone_throughput=100.0,
        compute_util=0.90,
        memory_util=0.32,
        urgency=0.90,
        max_slowdown=1.15,
    ),
    Job(
        "J2",
        "Qwen-3-8B",
        standalone_throughput=94.0,
        compute_util=0.82,
        memory_util=0.35,
        urgency=0.55,
        max_slowdown=1.20,
    ),
    Job(
        "J1",
        "Qwen-3-8B",
        standalone_throughput=86.0,
        compute_util=0.28,
        memory_util=0.86,
        urgency=0.35,
        max_slowdown=1.25,
    ),
    Job(
        "J3",
        "Qwen-3-8B",
        standalone_throughput=83.0,
        compute_util=0.34,
        memory_util=0.82,
        urgency=0.70,
        max_slowdown=1.18,
    ),
    Job(
        "J4",
        "Qwen-3-8B",
        standalone_throughput=78.0,
        compute_util=0.64,
        memory_util=0.61,
        urgency=0.20,
        max_slowdown=1.20,
    ),
    Job(
        "J5",
        "Qwen-3-8B",
        standalone_throughput=76.0,
        compute_util=0.66,
        memory_util=0.63,
        urgency=0.45,
        max_slowdown=1.20,
    ),
    Job(
        "J6",
        "Qwen-3-8B",
        standalone_throughput=105.0,
        compute_util=0.93,
        memory_util=0.38,
        urgency=0.80,
        max_slowdown=1.12,
    ),
    Job(
        "J7",
        "Qwen-3-8B",
        standalone_throughput=82.0,
        compute_util=0.31,
        memory_util=0.88,
        urgency=0.60,
        max_slowdown=1.20,
    ),
]


# =============================================================================
# Surrogate throughput predictor
# =============================================================================


def predict_group(
    jobs: Iterable[Job],
) -> GroupPrediction:
    jobs = tuple(jobs)

    if not jobs:
        raise ValueError(
            "group cannot be empty"
        )

    if len(
        {j.base_model for j in jobs}
    ) != 1:
        return GroupPrediction(
            total_throughput=0.0,
            per_job={},
            feasible=False,
        )

    if len(jobs) == 1:
        job = jobs[0]

        return GroupPrediction(
            total_throughput=(
                job.standalone_throughput
            ),
            per_job={
                job.job_id: JobPrediction(
                    throughput=(
                        job.standalone_throughput
                    ),
                    slowdown=1.0,
                )
            },
            feasible=True,
        )

    per_job = {}
    total = 0.0
    feasible = True

    for job in jobs:
        partners = [
            p
            for p in jobs
            if p.job_id != job.job_id
        ]

        # Determine which resource limits this job most.
        compute_dominant = (
            job.compute_util
            >= job.memory_util
        )

        if compute_dominant:
            partner_headroom = sum(
                p.residual_compute
                for p in partners
            ) / len(partners)

            partner_pressure = sum(
                p.compute_util
                for p in partners
            ) / len(partners)
        else:
            partner_headroom = sum(
                p.residual_memory
                for p in partners
            ) / len(partners)

            partner_pressure = sum(
                p.memory_util
                for p in partners
            ) / len(partners)

        # ---------------------------------------------------------------------
        # SURROGATE model:
        #
        # Benefit:
        #   partners with residual capacity in the resource that constrains
        #   this job can lend slack.
        #
        # Penalty:
        #   partners already consuming the same limiting resource create
        #   contention.
        #
        # Group-size penalty:
        #   approximates extra synchronization/coordination cost.
        #
        # None of these coefficients are claimed to come from the paper.
        # They exist only to make the scheduling algorithm executable.
        # ---------------------------------------------------------------------

        help_bonus = (
            0.34 * partner_headroom
        )

        same_resource_contention = (
            0.38
            * max(
                0.0,
                partner_pressure - 0.55,
            )
        )

        coordination_penalty = (
            0.018
            * max(
                0,
                len(jobs) - 2,
            )
        )

        multiplier = (
            1.0
            + help_bonus
            - same_resource_contention
            - coordination_penalty
        )

        multiplier = max(
            0.35,
            multiplier,
        )

        throughput = (
            job.standalone_throughput
            * multiplier
        )

        slowdown = max(
            1.0,
            (
                job.standalone_throughput
                / throughput
            ),
        )

        if slowdown > job.max_slowdown:
            feasible = False

        per_job[job.job_id] = (
            JobPrediction(
                throughput=throughput,
                slowdown=slowdown,
            )
        )

        total += throughput

    return GroupPrediction(
        total_throughput=total,
        per_job=per_job,
        feasible=feasible,
    )


# =============================================================================
# tLoRA-inspired incremental scheduler
# =============================================================================


def queue_sort_key(
    group: Group,
):
    # Paper:
    #   urgency descending,
    #   residual availability ascending.
    return (
        -group.urgency,
        group.residual_score,
    )


def merge_groups(
    left: Group,
    right: Group,
) -> Group:
    if (
        left.base_model
        != right.base_model
    ):
        raise ValueError(
            "Only jobs with the same base model "
            "can form one SSM."
        )

    return Group(
        jobs=left.jobs + right.jobs
    )


def independent_sum(
    group: Group,
) -> float:
    return sum(
        j.standalone_throughput
        for j in group.jobs
    )


def schedule_tlora_inspired(
    jobs: list[Job],
    minimum_gain=0.01,
):
    queue = [
        Group((job,))
        for job in jobs
    ]

    final_groups = []

    while queue:
        queue.sort(
            key=queue_sort_key
        )

        seed = queue.pop(0)

        best_index = None
        best_group = None
        best_prediction = None
        best_gain = 0.0

        seed_prediction = predict_group(
            seed.jobs
        )

        for idx, candidate in enumerate(
            queue
        ):
            if (
                candidate.base_model
                != seed.base_model
            ):
                continue

            merged = merge_groups(
                seed,
                candidate,
            )

            prediction = predict_group(
                merged.jobs
            )

            if not prediction.feasible:
                continue

            separate_throughput = (
                seed_prediction.total_throughput
                + predict_group(
                    candidate.jobs
                ).total_throughput
            )

            gain = (
                prediction.total_throughput
                / separate_throughput
                - 1.0
            )

            if (
                gain >= minimum_gain
                and gain > best_gain
            ):
                best_gain = gain
                best_index = idx
                best_group = merged
                best_prediction = prediction

        if best_group is None:
            final_groups.append(
                seed
            )
            continue

        # Merge/reinsert, matching the paper's incremental grouping idea.
        queue.pop(best_index)
        queue.append(best_group)

    return final_groups


# =============================================================================
# Simple FIFO baseline
# =============================================================================


def schedule_fifo_pairs(
    jobs: list[Job],
):
    """
    Deliberately simple baseline:
      group consecutive arrivals in pairs whenever base models match.

    It does NOT check complementarity or slowdown before grouping.
    This is a reduced FIFO baseline, not a claim to exactly reimplement mLoRA.
    """

    groups = []
    i = 0

    while i < len(jobs):
        if (
            i + 1 < len(jobs)
            and jobs[i].base_model
            == jobs[i + 1].base_model
        ):
            groups.append(
                Group(
                    (
                        jobs[i],
                        jobs[i + 1],
                    )
                )
            )
            i += 2
        else:
            groups.append(
                Group(
                    (jobs[i],)
                )
            )
            i += 1

    return groups


# =============================================================================
# Reporting
# =============================================================================


def report_schedule(
    name: str,
    groups: list[Group],
):
    print(
        f"\n{name}"
    )
    print("-" * 88)

    total_throughput = 0.0
    violations = 0

    print(
        f"{'Group':24s}"
        f"{'Throughput':>14s}"
        f"{'Feasible':>12s}"
        f"{'Worst slowdown':>17s}"
        f"{'Urgency max':>14s}"
    )

    for group in groups:
        pred = predict_group(
            group.jobs
        )

        total_throughput += (
            pred.total_throughput
        )

        if not pred.feasible:
            violations += 1

        worst = max(
            item.slowdown
            for item in pred.per_job.values()
        )

        label = "+".join(
            group.ids
        )

        print(
            f"{label:24s}"
            f"{pred.total_throughput:14.2f}"
            f"{str(pred.feasible):>12s}"
            f"{worst:17.3f}"
            f"{group.urgency:14.2f}"
        )

    print(
        f"\nTotal estimated throughput: "
        f"{total_throughput:.2f}"
    )

    print(
        f"Groups violating slowdown bounds: "
        f"{violations}"
    )

    return total_throughput, violations


def print_job_profiles():
    print(
        f"{'Job':5s}"
        f"{'Throughput':>12s}"
        f"{'Compute':>10s}"
        f"{'Memory':>10s}"
        f"{'Res.C':>10s}"
        f"{'Res.M':>10s}"
        f"{'Urgency':>10s}"
        f"{'Max slow':>11s}"
    )

    for job in JOBS:
        print(
            f"{job.job_id:5s}"
            f"{job.standalone_throughput:12.1f}"
            f"{job.compute_util:10.2f}"
            f"{job.memory_util:10.2f}"
            f"{job.residual_compute:10.2f}"
            f"{job.residual_memory:10.2f}"
            f"{job.urgency:10.2f}"
            f"{job.max_slowdown:11.2f}"
        )


def main():
    print("=" * 88)
    print(
        "MINI-tLoRA PHASE 7A - "
        "RESIDUAL-CAPACITY / URGENCY ADAPTER SCHEDULER"
    )
    print("=" * 88)

    print(
        "\nThis is an algorithmic reproduction of the scheduler logic.\n"
        "The numeric throughput predictor below is a documented surrogate;\n"
        "it is NOT claimed to be the paper authors' private profiling model."
    )

    print("\nJOB PROFILES")
    print("-" * 88)

    print_job_profiles()

    standalone_groups = [
        Group((job,))
        for job in JOBS
    ]

    fifo_groups = schedule_fifo_pairs(
        JOBS
    )

    tlora_groups = (
        schedule_tlora_inspired(
            JOBS,
            minimum_gain=0.01,
        )
    )

    standalone_tp, standalone_bad = (
        report_schedule(
            "STANDALONE",
            standalone_groups,
        )
    )

    fifo_tp, fifo_bad = report_schedule(
        "FIFO PAIR BASELINE",
        fifo_groups,
    )

    tlora_tp, tlora_bad = (
        report_schedule(
            "tLoRA-INSPIRED SCHEDULER",
            tlora_groups,
        )
    )

    print("\nSUMMARY")
    print("-" * 88)

    print(
        f"Standalone throughput : "
        f"{standalone_tp:.2f}"
    )

    print(
        f"FIFO throughput       : "
        f"{fifo_tp:.2f} "
        f"({fifo_tp / standalone_tp:.3f}x standalone)"
    )

    print(
        f"Scheduler throughput  : "
        f"{tlora_tp:.2f} "
        f"({tlora_tp / standalone_tp:.3f}x standalone)"
    )

    print(
        f"FIFO violations       : "
        f"{fifo_bad}"
    )

    print(
        f"Scheduler violations  : "
        f"{tlora_bad}"
    )

    print("\nFINAL tLoRA-INSPIRED GROUPS")
    print("-" * 88)

    for idx, group in enumerate(
        tlora_groups,
        start=1,
    ):
        pred = predict_group(
            group.jobs
        )

        print(
            f"G{idx}: "
            f"{'+'.join(group.ids)} "
            f"| throughput={pred.total_throughput:.2f} "
            f"| urgency={group.urgency:.2f} "
            f"| residual={group.residual_score:.2f}"
        )

    assert tlora_bad == 0, (
        "Scheduler emitted a group that violates "
        "a per-job slowdown bound."
    )

    print("\nSLOWDOWN CONSTRAINT CHECK: PASS")

    print("\nWHAT THIS PHASE DOES / DOES NOT PROVE")
    print("-" * 88)

    print(
        "Reproduced:\n"
        "  - residual-capacity-aware grouping\n"
        "  - urgency-first queue ordering\n"
        "  - joint-throughput-based merge selection\n"
        "  - per-job bounded-slowdown checks\n"
        "  - merge and reinsert behavior\n\n"
        "Not yet reproduced:\n"
        "  - authors' exact runtime throughput profiler\n"
        "  - hierarchical node/rank tiers\n"
        "  - binary-cut optimization / O(K log K) scaling\n"
        "  - production ACMETrace replay"
    )


if __name__ == "__main__":
    main()
