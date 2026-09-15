"""Append-only outcome memory for externally learned task experience.

This is deliberately an external learning mechanism: it never changes model
weights.  It records whether a context/action pair succeeded, preserves every
observation for audit, and exposes a time-decayed evidence view for future
recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OutcomeEvent:
    """One immutable observation from executing an action in a context."""

    event_id: int
    context: str
    action: str
    succeeded: bool
    observed_at: float
    provenance: str = "task_outcome"
    evidence_weight: float = 1.0


@dataclass
class _EvidenceState:
    success_weight: float
    failure_weight: float
    updated_at: float


@dataclass(frozen=True)
class EvidenceView:
    """Decayed evidence and its beta-prior posterior at one point in time."""

    success_weight: float
    failure_weight: float
    posterior_mean: float
    effective_observations: float
    observed_at: float


class OutcomeMemory:
    """Outcome-backed experience memory with an append-only audit log.

    A recommendation is the action with the highest posterior success rate in
    the requested context. Evidence decays between observations, which lets a
    repeated recent failure overcome a formerly successful but stale lesson.
    The event archive remains intact, so the update is explainable and can be
    re-scored under another decay policy later.
    """

    def __init__(self, decay_rate: float = 0.0, prior_strength: float = 1.0):
        if decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        self.decay_rate = decay_rate
        self.prior_strength = prior_strength
        self.clock = 0.0
        self.events: list[OutcomeEvent] = []
        self._evidence: dict[tuple[str, str], _EvidenceState] = {}

    def record(
        self,
        context: str,
        action: str,
        succeeded: bool,
        observed_at: float | None = None,
        provenance: str = "task_outcome",
        evidence_weight: float = 1.0,
    ) -> OutcomeEvent:
        """Append one observed outcome and refresh its materialized evidence."""
        if not context or not action:
            raise ValueError("context and action must be non-empty")
        if not 0 < evidence_weight <= 1:
            raise ValueError("evidence_weight must be in (0, 1]")
        timestamp = self.clock + 1.0 if observed_at is None else observed_at
        if timestamp < self.clock:
            raise ValueError("outcome events must be recorded in time order")

        key = (context, action)
        old = self._evidence.get(key)
        if old is None:
            success_weight = failure_weight = 0.0
        else:
            success_weight, failure_weight = self._decay(old, timestamp)
        if succeeded:
            success_weight += evidence_weight
        else:
            failure_weight += evidence_weight
        self._evidence[key] = _EvidenceState(success_weight, failure_weight, timestamp)
        self.clock = timestamp

        event = OutcomeEvent(
            event_id=len(self.events),
            context=context,
            action=action,
            succeeded=succeeded,
            observed_at=timestamp,
            provenance=provenance,
            evidence_weight=evidence_weight,
        )
        self.events.append(event)
        return event

    def evidence(self, context: str, action: str, at: float | None = None) -> EvidenceView:
        """Read the non-destructive, decayed evidence view for one action."""
        timestamp = self.clock if at is None else at
        if timestamp < self.clock:
            raise ValueError("evidence cannot be read before the latest event")
        state = self._evidence.get((context, action))
        if state is None:
            success_weight = failure_weight = 0.0
        else:
            success_weight, failure_weight = self._decay(state, timestamp)
        evidence_count = success_weight + failure_weight
        posterior = (self.prior_strength + success_weight) / (
            2 * self.prior_strength + evidence_count
        )
        return EvidenceView(
            success_weight=success_weight,
            failure_weight=failure_weight,
            posterior_mean=posterior,
            effective_observations=evidence_count,
            observed_at=timestamp,
        )

    def recommend(self, context: str, actions: list[str], at: float | None = None) -> str:
        """Recommend the best-supported action, with deterministic tie-breaking."""
        if not actions:
            raise ValueError("actions must not be empty")
        return max(
            actions,
            key=lambda action: (
                self.evidence(context, action, at).posterior_mean,
                self.evidence(context, action, at).effective_observations,
                action,
            ),
        )

    def history(self, context: str | None = None, action: str | None = None) -> list[OutcomeEvent]:
        """Return immutable observations, optionally scoped to one experience."""
        return [
            event
            for event in self.events
            if (context is None or event.context == context)
            and (action is None or event.action == action)
        ]

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def evidence_key_count(self) -> int:
        return len(self._evidence)

    def _decay(self, state: _EvidenceState, at: float) -> tuple[float, float]:
        if at < state.updated_at:
            raise ValueError("cannot decay evidence backwards in time")
        factor = math.exp(-self.decay_rate * (at - state.updated_at))
        return state.success_weight * factor, state.failure_weight * factor
