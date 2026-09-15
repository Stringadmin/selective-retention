"""Partial-feedback validation for outcome-backed experience memory.

Unlike ``run_outcome_drift``, each method executes only one action per context
and round, then observes only that action's outcome. This is a small
non-stationary contextual bandit, not an LLM benchmark. It tests whether an
external outcome log can support adaptation when feedback must be earned by
action and limited exploration.
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
    optimal_actions: int = 0
    successes: int = 0
    total: int = 0

    def add(self, action_is_optimal: bool, succeeded: bool) -> None:
        self.optimal_actions += int(action_is_optimal)
        self.successes += int(succeeded)
        self.total += 1

    @property
    def action_accuracy(self) -> float:
        return self.optimal_actions / self.total if self.total else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0


class _AllHistoryMean:
    """Partial-feedback baseline where every old result keeps equal weight."""

    def __init__(self):
        self.counts: dict[tuple[str, str], list[int]] = {}

    def record(self, context: str, action: str, succeeded: bool) -> None:
        success, total = self.counts.get((context, action), [0, 0])
        self.counts[(context, action)] = [success + int(succeeded), total + 1]

    def recommend(self, context: str) -> str:
        def score(action: str) -> tuple[float, int, str]:
            success, total = self.counts.get((context, action), [0, 0])
            return success / total if total else 0.5, total, action

        return max(ACTIONS, key=score)


class _DecayedEvidenceBaseline:
    """Same policy math as ``OutcomeMemory`` without event retention."""

    def __init__(self, decay_rate: float, prior_strength: float = 1.0):
        self.decay_rate = decay_rate
        self.prior_strength = prior_strength
        self.states: dict[tuple[str, str], tuple[float, float, float]] = {}

    def record(self, context: str, action: str, succeeded: bool, observed_at: float) -> None:
        successes, failures, updated_at = self.states.get(
            (context, action), (0.0, 0.0, observed_at)
        )
        factor = math.exp(-self.decay_rate * (observed_at - updated_at))
        self.states[(context, action)] = (
            successes * factor + int(succeeded),
            failures * factor + int(not succeeded),
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


def _reward_table(
    rng: random.Random,
    optimal: str,
    success_probability: float,
) -> dict[str, bool]:
    """Common-random-number reward table; agents only receive their own cell."""
    return {
        action: rng.random() < (success_probability if action == optimal else 1 - success_probability)
        for action in ACTIONS
    }


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
) -> dict:
    """Run an epsilon-greedy, partial-feedback concept-drift experiment."""
    if seeds <= 0 or contexts <= 0 or stationary_rounds <= 0 or drift_rounds <= 0:
        raise ValueError("seeds, contexts, and round counts must be positive")
    if not 0.5 < success_probability <= 1.0:
        raise ValueError("success_probability must be in (0.5, 1]")
    if decay_rate < 0:
        raise ValueError("decay_rate must be non-negative")
    if not 0 < exploration_rate < 1:
        raise ValueError("exploration_rate must be in (0, 1)")

    methods = (
        "random_no_memory",
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
    final_evidence_key_counts: list[int] = []

    for seed in range(seeds):
        rng = random.Random(seed)
        names = [f"context_{index:02d}" for index in range(contexts)]
        before = {context: rng.choice(ACTIONS) for context in names}
        after = {
            context: ACTIONS[1] if action == ACTIONS[0] else ACTIONS[0]
            for context, action in before.items()
        }
        all_history = _AllHistoryMean()
        decayed = _DecayedEvidenceBaseline(decay_rate=decay_rate)
        memory = OutcomeMemory(decay_rate=decay_rate)
        now = 0.0

        for phase, rounds, optimal_by_context in (
            ("stationary", stationary_rounds, before),
            ("drift", drift_rounds, after),
        ):
            for round_index in range(rounds):
                now += 1.0
                for context in names:
                    optimal = optimal_by_context[context]
                    rewards = _reward_table(rng, optimal, success_probability)
                    explore = rng.random() < exploration_rate
                    exploratory_action = rng.choice(ACTIONS)
                    recommendations = {
                        "all_history_mean": all_history.recommend(context),
                        "decayed_evidence_baseline": decayed.recommend(context, now),
                        "outcome_memory": memory.recommend(context, list(ACTIONS), at=now),
                    }
                    if (recommendations["decayed_evidence_baseline"]
                            != recommendations["outcome_memory"]):
                        raise AssertionError("equivalent decayed-evidence control diverged")
                    selections = {
                        "random_no_memory": rng.choice(ACTIONS),
                        **{
                            method: _select(recommendation, explore, exploratory_action)
                            for method, recommendation in recommendations.items()
                        },
                    }
                    for method, action in selections.items():
                        succeeded = rewards[action]
                        if method == "all_history_mean":
                            all_history.record(context, action, succeeded)
                        elif method == "decayed_evidence_baseline":
                            decayed.record(context, action, succeeded, observed_at=now)
                        elif method == "outcome_memory":
                            memory.record(context, action, succeeded, observed_at=now)
                        aggregate = (stationary[method] if phase == "stationary"
                                     else drift_by_round[method][round_index])
                        aggregate.add(action == optimal, succeeded)
        final_event_counts.append(memory.event_count)
        final_evidence_key_counts.append(memory.evidence_key_count)

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
            "stationary_success_rate": stationary[method].success_rate,
            "post_drift_action_accuracy": [aggregate.action_accuracy for aggregate in rounds],
            "post_drift_success_rate": [aggregate.success_rate for aggregate in rounds],
            "post_drift_mean_action_accuracy": sum(
                aggregate.action_accuracy for aggregate in rounds
            ) / len(rounds),
            "post_drift_final_action_accuracy": rounds[-1].action_accuracy,
            "post_drift_final_success_rate": rounds[-1].success_rate,
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
            "feedback": "partial: each policy observes only its executed action's outcome",
        },
        "summary": summary,
        "resources": {
            "outcome_memory_mean_event_records": sum(final_event_counts) / len(final_event_counts),
            "outcome_memory_mean_evidence_keys": sum(final_evidence_key_counts) / len(
                final_evidence_key_counts
            ),
            "decayed_evidence_baseline_event_archive": False,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "outcome-feedback-bandit.json",
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
