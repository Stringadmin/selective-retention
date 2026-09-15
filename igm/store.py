"""IGM memory store: an event archive plus a current-state view.

The store is the *write side* of memory.  Two behaviours distinguish it from a
plain vector store:

- **Slot supersede**: facts carry an optional ``slot`` (an attribute key such
  as "住址").  Writing a new value to an existing slot closes the previous
  event's validity interval instead of deleting it, so a fact that is updated
  never leaves a stale copy in the *current view* while the history stays
  queryable.  ``supersede="delete"`` restores the old destructive behaviour.
- **Forgetting lifecycle**: each memory has a consolidation strength that
  decays over time unless it is reused.  Low-value memories fade instead of
  accumulating forever.

The archive is the storage cost; ``len(store)`` and retrieval describe the
current projection, which is what a reader sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .embedders import Embedder, HashEmbedder, cosine

SUPERSEDE_ARCHIVE = "archive"
SUPERSEDE_DELETE = "delete"


@dataclass
class MemoryItem:
    text: str
    embedding: list[float]
    importance: float = 1.0
    reuse_count: int = 0
    created_at: float = 0.0
    last_used: float = 0.0
    slot: str | None = None
    metadata: dict = field(default_factory=dict)
    event_id: int = -1
    valid_to: float | None = None
    supersedes: int | None = None

    @property
    def is_current(self) -> bool:
        return self.valid_to is None


class MemoryStore:
    """Vector store with slot supersede + ACT-R-style forgetting.

    This is the reference backend.  It holds everything in a Python list, which
    is fine for thousands of memories; swap in an external vector DB (Qdrant,
    pgvector, ...) for larger scale by subclassing or wrapping.
    """

    def __init__(self, embedder: Embedder | None = None, decay: float = 0.0,
                 supersede: str = SUPERSEDE_ARCHIVE):
        if supersede not in (SUPERSEDE_ARCHIVE, SUPERSEDE_DELETE):
            raise ValueError(
                f"supersede must be {SUPERSEDE_ARCHIVE!r} or {SUPERSEDE_DELETE!r}, got {supersede!r}")
        self.embedder = embedder or HashEmbedder()
        self.decay = decay
        self.supersede = supersede
        self.items: list[MemoryItem] = []
        self.clock = 0.0
        self._next_event_id = 0

    # ------------------------------------------------------------------ write
    def write(self, text: str, importance: float = 1.0,
              slot: str | None = None, metadata: dict | None = None) -> MemoryItem:
        """Store a memory.  A non-None ``slot`` supersedes the memory currently
        holding that attribute (knowledge update).

        Under the default ``archive`` policy the previous event is closed at the
        current clock rather than removed, so ``history(slot)`` still answers
        "what was it before?".
        """
        previous = None
        if slot is not None:
            if self.supersede == SUPERSEDE_DELETE:
                self.items = [it for it in self.items if it.slot != slot]
            else:
                previous = next(
                    (it for it in self.items if it.slot == slot and it.valid_to is None), None)
                if previous is not None:
                    previous.valid_to = self.clock
        item = MemoryItem(
            text=text,
            embedding=self.embedder.embed(text),
            importance=importance,
            created_at=self.clock,
            last_used=self.clock,
            slot=slot,
            metadata=metadata or {},
            event_id=self._next_event_id,
            supersedes=previous.event_id if previous is not None else None,
        )
        self._next_event_id += 1
        self.items.append(item)
        return item

    # --------------------------------------------------------------- versions
    def current(self, slot: str | None = None) -> list[MemoryItem]:
        """The current-state projection, optionally routed to one slot."""
        if slot is not None:
            return [it for it in self.items if it.slot == slot and it.valid_to is None]
        return [it for it in self.items if it.valid_to is None]

    def history(self, slot: str) -> list[MemoryItem]:
        """Every event ever written for ``slot``, oldest first."""
        return [it for it in self.items if it.slot == slot]

    def previous(self, slot: str) -> MemoryItem | None:
        """The value a slot held before its current one."""
        versions = self.history(slot)
        return versions[-2] if len(versions) >= 2 else None

    @property
    def event_count(self) -> int:
        return len(self.items)

    @property
    def active_count(self) -> int:
        return sum(1 for it in self.items if it.valid_to is None)

    # --------------------------------------------------------------- strength
    def _effective_strength(self, item: MemoryItem) -> float:
        age = self.clock - item.last_used
        return item.importance * math.exp(-self.decay * age) * (1 + 0.1 * item.reuse_count)

    # --------------------------------------------------------------- retrieve
    def retrieve(self, query: str, top_k: int = 5, min_strength: float = 0.0,
                 slot: str | None = None,
                 include_history: bool = False) -> list[MemoryItem]:
        """Return the top-k memories for a query.

        Archived versions are excluded by default: retrieving them would put a
        superseded value back in front of the reader, which is the failure mode
        slot supersede exists to prevent.  Pass ``include_history=True`` to
        search the archive as well.

        ``slot`` (optional) gives an embedding-independent boost to memories
        whose slot exactly matches — a precise routing signal that pure
        similarity search lacks.
        """
        q = self.embedder.embed(query)
        scored = []
        for item in self.items:
            if item.valid_to is not None and not include_history:
                continue
            strength = self._effective_strength(item)
            if strength < min_strength:
                continue
            sim = cosine(q, item.embedding)
            if slot is not None and item.slot == slot:
                sim += 1.0
            scored.append((sim * strength, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def mark_used(self, item: MemoryItem) -> None:
        """Record a reuse: boosts consolidation and resets the decay clock."""
        item.reuse_count += 1
        item.last_used = self.clock
        item.importance = min(item.importance * 1.05, 5.0)

    # --------------------------------------------------------------- lifecycle
    def tick(self, amount: float = 1.0) -> None:
        """Advance the store clock (call once per turn / session)."""
        self.clock += amount

    def prune(self, threshold: float) -> int:
        """Drop memories below an effective-strength threshold.  Returns count removed.

        Archived versions decay like live ones, so consolidation eventually
        compacts the history instead of growing it without bound.
        """
        before = len(self.items)
        self.items = [it for it in self.items if self._effective_strength(it) >= threshold]
        return before - len(self.items)

    def __len__(self) -> int:
        """Size of the current projection — what a reader sees."""
        return self.active_count
