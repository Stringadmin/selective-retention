"""Semantic context routing used to connect retrieval with outcome memory."""

from __future__ import annotations

from dataclasses import dataclass

from . import cosine


@dataclass(frozen=True)
class ContextRoute:
    context_id: str
    similarity: float
    runner_up_context_id: str | None = None
    runner_up_similarity: float | None = None

    @property
    def margin(self) -> float | None:
        if self.runner_up_similarity is None:
            return None
        return self.similarity - self.runner_up_similarity


class SemanticContextRouter:
    """Route a task description to the closest canonical experience context.

    The router owns no facts or outcomes. It only converts a new surface form
    into a stable context ID, so Versioned/Outcome memory can apply their own
    state, evidence, and provenance rules after retrieval.
    """

    def __init__(self, embedder, prototypes: dict[str, str]):
        if not prototypes:
            raise ValueError("prototypes must not be empty")
        self.embedder = embedder
        self.prototypes = dict(prototypes)
        self._vectors = {
            context_id: self.embedder.embed(text)
            for context_id, text in self.prototypes.items()
        }

    def route(self, text: str) -> ContextRoute:
        if not text:
            raise ValueError("text must not be empty")
        query = self.embedder.embed(text)
        ranked = sorted(
            ((cosine(query, vector), context_id) for context_id, vector in self._vectors.items()),
            reverse=True,
        )
        similarity, context_id = ranked[0]
        runner_up_similarity, runner_up_context_id = (
            ranked[1] if len(ranked) > 1 else (None, None)
        )
        return ContextRoute(
            context_id=context_id,
            similarity=similarity,
            runner_up_context_id=runner_up_context_id,
            runner_up_similarity=runner_up_similarity,
        )
