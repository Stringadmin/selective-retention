"""Controlled validation for outcome-backed experience memory.

The workload is deliberately full-information: after each round it observes
the result of both candidate actions. This isolates memory adaptation from
exploration and from LLM reasoning. It is evidence that outcome events can
correct stale external experience, not a claim of general agent autonomy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import math
from pathlib import Path
import random

from .outcome import OutcomeMemory


ACTIONS = ("plan_a", "plan_b")


@dataclass
class _Aggregate:
    correct: int = 0
    total: int = 0

    def add(self, correct: bool) -> None:
        self.correct += int(correct)
        self.total += 1

    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


class _LatestOutcome:
    """Append-only history read using only the most recent observation."""

    def __init__(self):
        self.outcomes: dict[tuple[str, str], bool] = {}

    def record(self, context: str, action: str, succeeded: bool) -> None:
        self.outcomes[(context, action)] = succeeded

    def recommend(self, context: str) -> str:
        return max(ACTIONS, key=lambda action: (self.outcomes.get((context, action), False), action))


class _AllHistoryMean:
    """Non-decayed aggregate baseline; old evidence remains equally strong."""

    def __init__(self):
        self.counts: dict[tuple[str, str], list[int]] = {}

    def record(self, context: str, action: str, succeeded: bool) -> None:
        success, total = self.counts.get((context, action), [0, 0])
        self.counts[(context, action)] = [success + int(succeeded), total + 1]

    def recommend(self, context: str) -> str:
        def rate(action: str) -> tuple[float, int, str]:
            success, total = self.counts.get((context, action), [0, 0])
            return (success / total if total else 0.5, total, action)

        return max(ACTIONS, key=rate)


class _DecayedEvidenceBaseline:
    """Equivalent online statistic without an immutable event archive.

    This is intentionally a strong control: it uses the exact same exponential
    weighting and beta prior as :class:`OutcomeMemory`, but only retains its
    aggregate state. Equal recommendations demonstrate that event logging is
    an audit/interface benefit, not a new prediction algorithm.
    """

    def __init__(self, decay_rate: float, prior_strength: float = 1.0):
        self.decay_rate = decay_rate
        self.prior_strength = prior_strength
        self.states: dict[tuple[str, str], tuple[float, float, float]] = {}

    def record(self, context: str, action: str, succeeded: bool, observed_at: float) -> None:
        success_weight, failure_weight, updated_at = self.states.get(
            (context, action), (0.0, 0.0, observed_at)
        )
        factor = math.exp(-self.decay_rate * (observed_at - updated_at))
        success_weight *= factor
        failure_weight *= factor
        self.states[(context, action)] = (
            success_weight + int(succeeded),
            failure_weight + int(not succeeded),
            observed_at,
        )

    def recommend(self, context: str, at: float) -> str:
        def evidence(action: str) -> tuple[float, float, str]:
            success_weight, failure_weight, updated_at = self.states.get(
                (context, action), (0.0, 0.0, at)
            )
            factor = math.exp(-self.decay_rate * (at - updated_at))
            success_weight *= factor
            failure_weight *= factor
            count = success_weight + failure_weight
            posterior = (self.prior_strength + success_weight) / (
                2 * self.prior_strength + count
            )
            return posterior, count, action

        return max(ACTIONS, key=evidence)


def _sample_outcome(rng: random.Random, action: str, optimal: str, success_probability: float) -> bool:
    p = success_probability if action == optimal else 1.0 - success_probability
    return rng.random() < p


def run(
    seeds: int = 200,
    contexts: int = 32,
    stationary_rounds: int = 12,
    drift_rounds: int = 12,
    success_probability: float = 0.85,
    decay_rate: float = 0.28,
) -> dict:
    """Evaluate whether current evidence corrects stale experiences after drift."""
    if seeds <= 0 or contexts <= 0 or stationary_rounds <= 0 or drift_rounds <= 0:
        raise ValueError("seeds, contexts, and round counts must be positive")
    if not 0.5 < success_probability <= 1.0:
        raise ValueError("success_probability must be in (0.5, 1]")
    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative")

    methods = (
        "latest_outcome",
        "all_history_mean",
        "decayed_evidence_baseline",
        "outcome_memory",
    )
    stationary = {method: _Aggregate() for method in methods}
    drift_by_round = {
        method: [_Aggregate() for _ in range(drift_rounds)]
        for method in methods
    }
    final_event_counts: list[int] = []

    for seed in range(seeds):
        rng = random.Random(seed)
        names = [f"context_{index:02d}" for index in range(contexts)]
        pre_drift_optimal = {
            context: rng.choice(ACTIONS)
            for context in names
        }
        post_drift_optimal = {
            context: ACTIONS[1] if action == ACTIONS[0] else ACTIONS[0]
            for context, action in pre_drift_optimal.items()
        }
        latest = _LatestOutcome()
        all_history = _AllHistoryMean()
        decayed_evidence = _DecayedEvidenceBaseline(decay_rate=decay_rate)
        outcome_memory = OutcomeMemory(decay_rate=decay_rate)
        now = 0.0

        for phase, rounds, optimal_by_context in (
            ("stationary", stationary_rounds, pre_drift_optimal),
            ("drift", drift_rounds, post_drift_optimal),
        ):
            for round_index in range(rounds):
                # One feedback round is one unit of environmental time. Events
                # from different contexts in that round are contemporaneous;
                # unrelated contexts must not make each other look stale.
                now += 1.0
                for context in names:
                    optimal = optimal_by_context[context]
                    for action in ACTIONS:
                        succeeded = _sample_outcome(
                            rng, action, optimal, success_probability
                        )
                        latest.record(context, action, succeeded)
                        all_history.record(context, action, succeeded)
                        decayed_evidence.record(context, action, succeeded, observed_at=now)
                        outcome_memory.record(context, action, succeeded, observed_at=now)

                    recommendations = {
                        "latest_outcome": latest.recommend(context),
                        "all_history_mean": all_history.recommend(context),
                        "decayed_evidence_baseline": decayed_evidence.recommend(context, now),
                        "outcome_memory": outcome_memory.recommend(context, list(ACTIONS)),
                    }
                    if (recommendations["decayed_evidence_baseline"]
                            != recommendations["outcome_memory"]):
                        raise AssertionError("equivalent decayed-evidence control diverged")
                    for method, recommendation in recommendations.items():
                        if phase == "stationary":
                            stationary[method].add(recommendation == optimal)
                        else:
                            drift_by_round[method][round_index].add(recommendation == optimal)
        final_event_counts.append(outcome_memory.event_count)

    def recovery_round(values: list[_Aggregate], threshold: float = 0.8) -> int | None:
        for index, aggregate in enumerate(values, start=1):
            if aggregate.rate >= threshold:
                return index
        return None

    summary = {}
    for method in methods:
        rounds = drift_by_round[method]
        summary[method] = {
            "stationary_accuracy": stationary[method].rate,
            "post_drift_accuracy": [aggregate.rate for aggregate in rounds],
            "post_drift_mean_accuracy": sum(aggregate.rate for aggregate in rounds) / len(rounds),
            "post_drift_final_accuracy": rounds[-1].rate,
            "recovery_round_at_80pct": recovery_round(rounds),
        }

    return {
        "config": {
            "seeds": seeds,
            "contexts_per_seed": contexts,
            "stationary_rounds": stationary_rounds,
            "drift_rounds": drift_rounds,
            "success_probability": success_probability,
            "decay_rate": decay_rate,
            "feedback": "full-information outcomes for both actions per context and round",
        },
        "summary": summary,
        "resources": {
            "outcome_memory_mean_event_records": sum(final_event_counts) / len(final_event_counts),
            "outcome_memory_evidence_keys": contexts * len(ACTIONS),
            "decayed_evidence_baseline_evidence_keys": contexts * len(ACTIONS),
            "decayed_evidence_baseline_event_archive": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--contexts", type=int, default=32)
    parser.add_argument("--stationary-rounds", type=int, default=12)
    parser.add_argument("--drift-rounds", type=int, default=12)
    parser.add_argument("--success-probability", type=float, default=0.85)
    parser.add_argument("--decay-rate", type=float, default=0.28)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "outcome-feedback-drift.json",
    )
    args = parser.parse_args()
    result = run(
        seeds=args.seeds,
        contexts=args.contexts,
        stationary_rounds=args.stationary_rounds,
        drift_rounds=args.drift_rounds,
        success_probability=args.success_probability,
        decay_rate=args.decay_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
