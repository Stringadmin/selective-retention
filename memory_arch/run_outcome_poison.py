"""Partial-feedback experiment with explicitly labeled corrupted outcomes.

The simulator deliberately flips some observed outcomes and labels those
reports as unverified. Source-weighted methods receive that provenance label;
unweighted methods receive the same report but ignore the label. This tests a
conditional safety property, not automatic detection of misinformation.
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


@dataclass(frozen=True)
class _ReportedOutcome:
    true_success: bool
    reported_success: bool
    provenance: str
    evidence_weight: float


@dataclass
class _Aggregate:
    optimal_actions: int = 0
    true_successes: int = 0
    total: int = 0

    def add(self, action_is_optimal: bool, true_success: bool) -> None:
        self.optimal_actions += int(action_is_optimal)
        self.true_successes += int(true_success)
        self.total += 1

    @property
    def action_accuracy(self) -> float:
        return self.optimal_actions / self.total if self.total else 0.0

    @property
    def true_success_rate(self) -> float:
        return self.true_successes / self.total if self.total else 0.0


class _DecayedEvidence:
    """Event-free implementation of the same weighted evidence policy."""

    def __init__(self, decay_rate: float, prior_strength: float = 1.0):
        self.decay_rate = decay_rate
        self.prior_strength = prior_strength
        self.states: dict[tuple[str, str], tuple[float, float, float]] = {}

    def record(
        self,
        context: str,
        action: str,
        succeeded: bool,
        observed_at: float,
        evidence_weight: float = 1.0,
    ) -> None:
        successes, failures, updated_at = self.states.get(
            (context, action), (0.0, 0.0, observed_at)
        )
        factor = math.exp(-self.decay_rate * (observed_at - updated_at))
        self.states[(context, action)] = (
            successes * factor + (evidence_weight if succeeded else 0.0),
            failures * factor + (evidence_weight if not succeeded else 0.0),
            observed_at,
        )

    def recommend(self, context: str, at: float) -> str:
        def score(action: str) -> tuple[float, float, str]:
            successes, failures, updated_at = self.states.get(
                (context, action), (0.0, 0.0, at)
            )
            factor = math.exp(-self.decay_rate * (at - updated_at))
            successes *= factor
            failures *= factor
            count = successes + failures
            posterior = (self.prior_strength + successes) / (
                2 * self.prior_strength + count
            )
            return posterior, count, action

        return max(ACTIONS, key=score)


def _feedback_table(
    rng: random.Random,
    optimal: str,
    success_probability: float,
    corruption_rate: float,
    untrusted_weight: float,
) -> dict[str, _ReportedOutcome]:
    table = {}
    for action in ACTIONS:
        true_success = rng.random() < (
            success_probability if action == optimal else 1.0 - success_probability
        )
        corrupted = rng.random() < corruption_rate
        table[action] = _ReportedOutcome(
            true_success=true_success,
            reported_success=not true_success if corrupted else true_success,
            provenance="unverified_report" if corrupted else "verified_execution",
            evidence_weight=untrusted_weight if corrupted else 1.0,
        )
    return table


def _select(recommendation: str, explore: bool, exploratory_action: str) -> str:
    return exploratory_action if explore else recommendation


def run(
    seeds: int = 400,
    contexts: int = 32,
    stationary_rounds: int = 20,
    drift_rounds: int = 20,
    success_probability: float = 0.85,
    decay_rate: float = 0.28,
    exploration_rate: float = 0.10,
    corruption_rate: float = 0.30,
    untrusted_weight: float = 0.05,
) -> dict:
    """Compare source-weighted and unweighted partial-feedback adaptation."""
    if seeds <= 0 or contexts <= 0 or stationary_rounds <= 0 or drift_rounds <= 0:
        raise ValueError("seeds, contexts, and round counts must be positive")
    if not 0.5 < success_probability <= 1.0:
        raise ValueError("success_probability must be in (0.5, 1]")
    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative")
    if not 0 < exploration_rate < 1:
        raise ValueError("exploration_rate must be in (0, 1)")
    if not 0 <= corruption_rate < 1:
        raise ValueError("corruption_rate must be in [0, 1)")
    if not 0 < untrusted_weight <= 1:
        raise ValueError("untrusted_weight must be in (0, 1]")

    methods = (
        "random_no_memory",
        "unweighted_decayed_evidence",
        "source_weighted_evidence_baseline",
        "source_weighted_outcome_memory",
    )
    stationary = {method: _Aggregate() for method in methods}
    drift_by_round = {
        method: [_Aggregate() for _ in range(drift_rounds)]
        for method in methods
    }
    final_event_counts: list[int] = []
    final_evidence_key_counts: list[int] = []

    for seed in range(seeds):
        rng = random.Random(seed)
        names = [f"context_{index:02d}" for index in range(contexts)]
        before = {context: rng.choice(ACTIONS) for context in names}
        after = {
            context: ACTIONS[1] if action == ACTIONS[0] else ACTIONS[0]
            for context, action in before.items()
        }
        unweighted = _DecayedEvidence(decay_rate=decay_rate)
        weighted_baseline = _DecayedEvidence(decay_rate=decay_rate)
        weighted_memory = OutcomeMemory(decay_rate=decay_rate)
        now = 0.0

        for phase, rounds, optimal_by_context in (
            ("stationary", stationary_rounds, before),
            ("drift", drift_rounds, after),
        ):
            for round_index in range(rounds):
                now += 1.0
                for context in names:
                    optimal = optimal_by_context[context]
                    feedback = _feedback_table(
                        rng, optimal, success_probability, corruption_rate, untrusted_weight
                    )
                    explore = rng.random() < exploration_rate
                    exploratory_action = rng.choice(ACTIONS)
                    recommendations = {
                        "unweighted_decayed_evidence": unweighted.recommend(context, now),
                        "source_weighted_evidence_baseline": weighted_baseline.recommend(context, now),
                        "source_weighted_outcome_memory": weighted_memory.recommend(
                            context, list(ACTIONS), at=now
                        ),
                    }
                    if (recommendations["source_weighted_evidence_baseline"]
                            != recommendations["source_weighted_outcome_memory"]):
                        raise AssertionError("equivalent source-weighted control diverged")
                    selections = {
                        "random_no_memory": rng.choice(ACTIONS),
                        **{
                            method: _select(recommendation, explore, exploratory_action)
                            for method, recommendation in recommendations.items()
                        },
                    }
                    for method, action in selections.items():
                        report = feedback[action]
                        if method == "unweighted_decayed_evidence":
                            unweighted.record(
                                context, action, report.reported_success, now, evidence_weight=1.0
                            )
                        elif method == "source_weighted_evidence_baseline":
                            weighted_baseline.record(
                                context, action, report.reported_success, now,
                                evidence_weight=report.evidence_weight,
                            )
                        elif method == "source_weighted_outcome_memory":
                            weighted_memory.record(
                                context, action, report.reported_success, now,
                                provenance=report.provenance,
                                evidence_weight=report.evidence_weight,
                            )
                        aggregate = (stationary[method] if phase == "stationary"
                                     else drift_by_round[method][round_index])
                        aggregate.add(action == optimal, report.true_success)
        final_event_counts.append(weighted_memory.event_count)
        final_evidence_key_counts.append(weighted_memory.evidence_key_count)

    def recovery_round(values: list[_Aggregate], threshold: float = 0.8) -> int | None:
        for index, aggregate in enumerate(values, start=1):
            if aggregate.action_accuracy >= threshold:
                return index
        return None

    summary = {}
    for method in methods:
        rounds = drift_by_round[method]
        summary[method] = {
            "stationary_action_accuracy": stationary[method].action_accuracy,
            "stationary_true_success_rate": stationary[method].true_success_rate,
            "post_drift_mean_action_accuracy": sum(
                aggregate.action_accuracy for aggregate in rounds
            ) / len(rounds),
            "post_drift_final_action_accuracy": rounds[-1].action_accuracy,
            "post_drift_final_true_success_rate": rounds[-1].true_success_rate,
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
            "exploration_rate": exploration_rate,
            "corruption_rate": corruption_rate,
            "untrusted_weight": untrusted_weight,
            "feedback": (
                "partial; corrupted reports are explicitly labeled unverified, "
                "and only source-weighted methods consume that label"
            ),
        },
        "summary": summary,
        "resources": {
            "source_weighted_outcome_memory_mean_event_records": sum(
                final_event_counts
            ) / len(final_event_counts),
            "source_weighted_outcome_memory_mean_evidence_keys": sum(
                final_evidence_key_counts
            ) / len(final_evidence_key_counts),
            "source_weighted_evidence_baseline_event_archive": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--contexts", type=int, default=32)
    parser.add_argument("--stationary-rounds", type=int, default=20)
    parser.add_argument("--drift-rounds", type=int, default=20)
    parser.add_argument("--success-probability", type=float, default=0.85)
    parser.add_argument("--decay-rate", type=float, default=0.28)
    parser.add_argument("--exploration-rate", type=float, default=0.10)
    parser.add_argument("--corruption-rate", type=float, default=0.30)
    parser.add_argument("--untrusted-weight", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "outcome-feedback-poison.json",
    )
    args = parser.parse_args()
    result = run(
        seeds=args.seeds,
        contexts=args.contexts,
        stationary_rounds=args.stationary_rounds,
        drift_rounds=args.drift_rounds,
        success_probability=args.success_probability,
        decay_rate=args.decay_rate,
        exploration_rate=args.exploration_rate,
        corruption_rate=args.corruption_rate,
        untrusted_weight=args.untrusted_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
