"""IGM: an importance-gated write layer for RAG memory.

IGM sits in front of a vector store and decides **what gets written** and
**how updates are applied** — the two things plain RAG does not handle.

    from igm import Memory

    mem = Memory()                                  # zero-dependency defaults
    mem.add("我的住址是北京。")
    mem.add("更新一下，我的住址现在是深圳了。")        # supersedes the old value
    mem.query("我现在的住址是什么？")                  # -> the Shenzhen fact only

To use a real embedding model:

    from igm import Memory, SentenceTransformerEmbedder
    mem = Memory(embedder=SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5"))
"""

from __future__ import annotations

from .embedders import Embedder, HashEmbedder, cosine
from .gate import Scorer, WriteGate
from .store import MemoryItem, MemoryStore


class Memory:
    """Importance-gated memory.  The single entry point for users."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        scorer: Scorer | None = None,
        write_threshold: float = 0.6,
        decay: float = 0.0,
        prune_threshold: float = 0.1,
        top_k: int = 5,
    ):
        self.store = MemoryStore(embedder or HashEmbedder(), decay=decay)
        self.gate = WriteGate(scorer, threshold=write_threshold)
        self.prune_threshold = prune_threshold
        self.top_k = top_k
        self.n_considered = 0   # candidates seen
        self.n_written = 0      # candidates actually stored

    # ------------------------------------------------------------------ write
    def add(self, text: str, metadata: dict | None = None) -> MemoryItem | None:
        """Consider a candidate memory.  Returns the stored item, or None if
        the gate rejected it.  Facts about the same attribute supersede."""
        self.n_considered += 1
        self.store.tick()
        max_sim = self._max_sim(text)
        ok, score = self.gate.should_write(text, max_sim)
        if not ok:
            return None
        slot = self.gate.slot_for(text)
        item = self.store.write(text, importance=score, slot=slot, metadata=metadata)
        self.n_written += 1
        return item

    def _max_sim(self, text: str) -> float:
        if not self.store.items:
            return 0.0
        emb = self.store.embedder.embed(text)
        return max(cosine(emb, it.embedding) for it in self.store.items[-200:])

    # --------------------------------------------------------------- retrieve
    def query(self, text: str, top_k: int | None = None) -> list[MemoryItem]:
        """Retrieve relevant memories.  Slot-aware: an attribute in the query
        routes to the memory holding its current value."""
        slot = self.gate.slot_for(text)
        items = self.store.retrieve(
            text, top_k=top_k or self.top_k,
            min_strength=self.prune_threshold, slot=slot,
        )
        for it in items:
            self.store.mark_used(it)
        return items

    def query_texts(self, text: str, top_k: int | None = None) -> list[str]:
        return [it.text for it in self.query(text, top_k)]

    # --------------------------------------------------------------- lifecycle
    def consolidate(self) -> int:
        """Forget low-value memories.  Call at a session/day boundary."""
        return self.store.prune(self.prune_threshold)

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        return {
            "stored": len(self.store),
            "considered": self.n_considered,
            "written": self.n_written,
            "selectivity": (self.n_written / self.n_considered) if self.n_considered else 0.0,
        }

    def __len__(self) -> int:
        return len(self.store)
