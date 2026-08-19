"""IGM memory store: vector memory with slot supersede and forgetting.

The store is the *write side* of memory.  Two behaviours distinguish it from a
plain vector store:

- **Slot supersede**: facts carry an optional ``slot`` (an attribute key such
  as "住址").  Writing a new value to an existing slot removes the old value,
  so a fact that is updated never leaves a stale copy behind.  This is what
  solves knowledge update / contradiction resolution.
- **Forgetting lifecycle**: each memory has a consolidation strength that
  decays over time unless it is reused.  Low-value memories fade instead of
  accumulating forever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .embedders import Embedder, HashEmbedder, cosine


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


class MemoryStore:
    """In-memory vector store with slot supersede + ACT-R-style forgetting.

    This is the reference backend.  It holds everything in a Python list, which
    is fine for thousands of memories; swap in an external vector DB (Qdrant,
    pgvector, ...) for larger scale by subclassing or wrapping.
    """

    def __init__(self, embedder: Embedder | None = None, decay: float = 0.0):
        self.embedder = embedder or HashEmbedder()
        self.decay = decay
        self.items: list[MemoryItem] = []
        self.clock = 0.0

    # ------------------------------------------------------------------ write
    def write(self, text: str, importance: float = 1.0,
              slot: str | None = None, metadata: dict | None = None) -> MemoryItem:
        """Store a memory.  A non-None ``slot`` supersedes any existing memory
        with the same slot (knowledge update)."""
        if slot is not None:
            self.items = [it for it in self.items if it.slot != slot]
        item = MemoryItem(
            text=text,
            embedding=self.embedder.embed(text),
            importance=importance,
            created_at=self.clock,
            last_used=self.clock,
            slot=slot,
            metadata=metadata or {},
        )
        self.items.append(item)
        return item

    # --------------------------------------------------------------- strength
    def _effective_strength(self, item: MemoryItem) -> float:
        age = self.clock - item.last_used
        return item.importance * math.exp(-self.decay * age) * (1 + 0.1 * item.reuse_count)

    # --------------------------------------------------------------- retrieve
    def retrieve(self, query: str, top_k: int = 5, min_strength: float = 0.0,
                 slot: str | None = None) -> list[MemoryItem]:
        """Return the top-k memories for a query.

        ``slot`` (optional) gives an embedding-independent boost to memories
        whose slot exactly matches — a precise routing signal that pure
        similarity search lacks.
        """
        q = self.embedder.embed(query)
        scored = []
        for item in self.items:
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
        """Drop memories below an effective-strength threshold.  Returns count removed."""
        before = len(self.items)
        self.items = [it for it in self.items if self._effective_strength(it) >= threshold]
        return before - len(self.items)

    def __len__(self) -> int:
        return len(self.items)
