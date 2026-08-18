"""Importance-Gated Memory (IGM) M0 baseline framework.

A modular memory-pipeline harness for evaluating agent memory strategies on
multi-session conversational data.  All methods share the same base LLM
(qwen3:4b via llama-cpp) and the same data; they differ only in how they
write, store, and retrieve memories.

Methods (the M0 baseline matrix):
  - full_context : stuff all history into the context window (no memory layer)
  - naive_rag    : write every turn, retrieve top-k by embedding similarity
  - mem0_style   : heuristic salience extraction (an LLM decides what's a fact)
  - igm          : importance-gated writing (learned scorer decides whether to
                   store) + consolidation/forgetting lifecycle

Data: LongMemEval format (sessions of {role, content} turns + questions with
answers).  A small synthetic generator is included for offline pipeline
bring-up until the real LongMemEval data is obtainable (its GitHub/HF sources
are network-blocked in this environment).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class Turn:
    role: str          # "user" | "assistant"
    content: str
    session: int = 0
    timestamp: float = 0.0


@dataclass
class Question:
    id: str
    category: str      # e.g. single_session_user, knowledge_update, temporal_reasoning
    question: str
    answer: str


@dataclass
class Conversation:
    conv_id: str
    turns: list[Turn]
    questions: list[Question] = field(default_factory=list)


# --------------------------------------------------------------------------
# Memory store: simple embedding-based vector memory
# --------------------------------------------------------------------------


class Embedder:
    """Pluggable text embedder.

    Two backends:
      - "bge": a real sentence-embedding model (BGE-small-zh) loaded via
        sentence-transformers from a local path (ModelScope download).  Much
        better semantic discrimination than hashing.
      - "hash" (default fallback): a zero-dependency hashing bag-of-words, so
        the pipeline still runs without any external embedding model.
    """

    def __init__(self, dim: int = 256, backend: str = "hash", model_path: str | None = None):
        self.dim = dim
        self.backend = backend
        self._st_model = None
        if backend == "bge":
            if not model_path:
                raise ValueError("bge backend requires model_path")
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_path)

    def embed(self, text: str) -> list[float]:
        if self._st_model is not None:
            vec = self._st_model.encode(text, normalize_embeddings=True)
            return [float(v) for v in vec]
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            h = hash(tok) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class MemoryItem:
    text: str
    embedding: list[float]
    importance: float = 1.0      # consolidation strength
    reuse_count: int = 0          # how often it was retrieved & used
    created_at: float = 0.0
    last_used: float = 0.0
    slot: str | None = None       # attribute key (e.g. "住址"); same-slot writes supersede


class MemoryStore:
    """Vector memory with optional importance-based forgetting."""

    def __init__(self, embedder: Embedder, decay: float = 0.0):
        self.embedder = embedder
        self.decay = decay        # forgetting rate per "time unit" (0 = never forget)
        self.items: list[MemoryItem] = []
        self.clock = 0.0

    def write(self, text: str, importance: float = 1.0, slot: str | None = None) -> None:
        # Knowledge-update semantics: a new fact for an existing slot
        # supersedes the old one (the outdated value is forgotten).
        if slot is not None:
            self.items = [it for it in self.items if it.slot != slot]
        self.items.append(MemoryItem(
            text=text,
            embedding=self.embedder.embed(text),
            importance=importance,
            created_at=self.clock,
            last_used=self.clock,
            slot=slot,
        ))

    def _effective_strength(self, item: MemoryItem) -> float:
        """ACT-R-style: consolidation strength decays unless reused."""
        age = self.clock - item.last_used
        return item.importance * math.exp(-self.decay * age) * (1 + 0.1 * item.reuse_count)

    def retrieve(self, query: str, top_k: int = 5, min_strength: float = 0.0,
                 slot: str | None = None) -> list[MemoryItem]:
        q = self.embedder.embed(query)
        scored = []
        for item in self.items:
            strength = self._effective_strength(item)
            if strength < min_strength:
                continue  # forgotten
            sim = cosine(q, item.embedding)
            # Slot-aware routing: an exact attribute match is a strong,
            # embedding-independent signal (fixes bag-of-words noise).
            if slot is not None and item.slot == slot:
                sim += 1.0
            scored.append((sim * strength, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def mark_used(self, item: MemoryItem) -> None:
        item.reuse_count += 1
        item.last_used = self.clock
        item.importance = min(item.importance * 1.05, 5.0)  # consolidation

    def prune(self, threshold: float) -> int:
        """Drop memories whose effective strength fell below threshold.  Returns count removed."""
        before = len(self.items)
        self.items = [it for it in self.items if self._effective_strength(it) >= threshold]
        return before - len(self.items)

    def __len__(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------
# LLM wrapper (llama-cpp, local qwen3:4b)
# --------------------------------------------------------------------------


class LocalLLM:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = 0):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_gpu_layers=n_gpu_layers, n_batch=512, verbose=False)

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0) -> str:
        # qwen3 no-think switch: answer directly, no long reasoning preface.
        msgs = [dict(m) for m in messages]
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["content"] = "/no_think\n" + m["content"]
                break
        out = self.llm.create_chat_completion(
            messages=msgs, max_tokens=max_tokens, temperature=temperature,
            stop=["</think>"],
        )
        text = out["choices"][0]["message"]["content"].strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return text


# --------------------------------------------------------------------------
# Memory method strategies
# --------------------------------------------------------------------------


class BaseMethod:
    name = "base"

    def __init__(self, llm: LocalLLM, embedder: Embedder):
        self.llm = llm
        self.embedder = embedder
        self.store = MemoryStore(embedder)
        self.n_writes = 0

    def ingest(self, conv: Conversation) -> None:
        raise NotImplementedError

    def answer(self, question: str) -> str:
        raise NotImplementedError


class FullContextMethod(BaseMethod):
    name = "full_context"

    def __init__(self, llm, embedder):
        super().__init__(llm, embedder)
        self.history: list[Turn] = []

    def ingest(self, conv: Conversation) -> None:
        self.history.extend(conv.turns)
        self.n_writes += len(conv.turns)

    def answer(self, question: str) -> str:
        # Truncate history to fit context window.
        text = "\n".join(f"{t.role}: {t.content}" for t in self.history)
        text = text[-6000:]  # crude char truncation for ctx budget
        msgs = [
            {"role": "system", "content": "Based on the conversation history, answer the question concisely."},
            {"role": "user", "content": f"History:\n{text}\n\nQuestion: {question}"},
        ]
        return self.llm.chat(msgs, max_tokens=400)


class NaiveRAGMethod(BaseMethod):
    name = "naive_rag"

    def ingest(self, conv: Conversation) -> None:
        for t in conv.turns:
            self.store.write(f"{t.role}: {t.content}")
            self.n_writes += 1

    def answer(self, question: str) -> str:
        items = self.store.retrieve(question, top_k=5)
        ctx = "\n".join(it.text for it in items)
        msgs = [
            {"role": "system", "content": "Answer concisely using the retrieved memories."},
            {"role": "user", "content": f"Memories:\n{ctx}\n\nQuestion: {question}"},
        ]
        return self.llm.chat(msgs, max_tokens=400)


class Mem0StyleMethod(BaseMethod):
    """Heuristic extraction: an LLM extracts 'salient facts' from each session."""
    name = "mem0_style"

    def ingest(self, conv: Conversation) -> None:
        # Batch per session: ask the LLM to extract facts worth remembering.
        by_session: dict[int, list[Turn]] = {}
        for t in conv.turns:
            by_session.setdefault(t.session, []).append(t)
        for session, turns in by_session.items():
            text = "\n".join(f"{t.role}: {t.content}" for t in turns)
            msgs = [
                {"role": "system", "content":
                 "Extract key facts about the user worth remembering long-term, "
                 "as a short bullet list. If none, output 'NONE'."},
                {"role": "user", "content": text},
            ]
            facts = self.llm.chat(msgs, max_tokens=200)
            for line in facts.splitlines():
                line = line.strip().lstrip("-*• ").strip()
                if line and line.upper() != "NONE" and len(line) > 3:
                    self.store.write(line)
                    self.n_writes += 1

    def answer(self, question: str) -> str:
        items = self.store.retrieve(question, top_k=5)
        ctx = "\n".join(it.text for it in items)
        msgs = [
            {"role": "system", "content": "Answer concisely using the stored facts."},
            {"role": "user", "content": f"Facts:\n{ctx}\n\nQuestion: {question}"},
        ]
        return self.llm.chat(msgs, max_tokens=400)


class IGMMethod(BaseMethod):
    """Importance-Gated Memory: score each candidate memory, write only the
    important ones, and apply a consolidation/forgetting lifecycle."""
    name = "igm"

    def __init__(self, llm, embedder, write_threshold: float = 0.5,
                 decay: float = 0.02, prune_threshold: float = 0.1,
                 scorer=None):
        super().__init__(llm, embedder)
        self.store = MemoryStore(embedder, decay=decay)
        self.write_threshold = write_threshold
        self.prune_threshold = prune_threshold
        self._surprise_history: list[float] = []
        # Optional learned write-gate (M1).  When provided it replaces the
        # hand-tuned heuristic in ``_importance``.
        self.scorer = scorer

    # Linguistic markers of a durable, self-referential fact about the user.
    _FACT_MARKERS = ("我", "我的", "喜欢", "是", "在", "去过", "住", "工作", "现在")

    # Temporal/status prefixes that modify but are not part of the attribute.
    _SLOT_PREFIXES = ("现在的", "目前的", "新的", "原来的", "以前的", "当前的")

    @staticmethod
    def _extract_slot(text: str) -> str | None:
        """Extract the attribute key from a self-referential fact of the form
        '...我的{ATTR}是...' or '...我的{ATTR}现在是...'.  Temporal prefixes
        like '现在的' are stripped so a query '我现在的住址' and a fact
        '我的住址是...' map to the same slot.  Returns None if not an
        attribute-style statement/question."""
        for anchor in ("我的", "我"):
            if anchor in text:
                rest = text.split(anchor, 1)[1]
                for stop in ("现在是", "是什么", "是", "了", "。", "，", ",", "？", "?"):
                    if stop in rest:
                        attr = rest.split(stop, 1)[0].strip()
                        for pref in IGMMethod._SLOT_PREFIXES:
                            if attr.startswith(pref):
                                attr = attr[len(pref):]
                        if 0 < len(attr) <= 6:
                            return attr
        return None

    def _surprise(self, text: str) -> float:
        """Novelty vs already-stored memories (1 = completely novel)."""
        emb = self.embedder.embed(text)
        if not self.store.items:
            return 1.0
        sims = [cosine(emb, it.embedding) for it in self.store.items[-200:]]
        return 1.0 - max(sims)

    def _importance(self, text: str) -> float:
        """Importance score.  Uses the learned scorer when available (M1);
        otherwise falls back to the hand-tuned heuristic (M0)."""
        surprise = self._surprise(text)
        if self.scorer is not None:
            # The scorer expects max_sim (similarity), so pass 1 - surprise.
            return self.scorer.score(text, max_sim_to_store=1.0 - surprise)
        # Heuristic fallback (M0): fact markers + surprise + density.
        marker_hits = sum(1 for mk in self._FACT_MARKERS if mk in text)
        fact_score = min(marker_hits / 3.0, 1.0)
        density = min(len(text.split()) / 20.0, 1.0)
        return 0.6 * fact_score + 0.3 * surprise + 0.1 * density

    def ingest(self, conv: Conversation) -> None:
        for t in conv.turns:
            self.store.clock += 1.0
            text = f"{t.role}: {t.content}"
            score = self._importance(text)
            self._surprise_history.append(score)
            if score >= self.write_threshold:
                slot = self._extract_slot(text)
                self.store.write(text, importance=score, slot=slot)
                self.n_writes += 1
        # End-of-conversation consolidation pass: forget low-value memories.
        self.store.prune(self.prune_threshold)

    def answer(self, question: str) -> str:
        slot = self._extract_slot(question)
        items = self.store.retrieve(question, top_k=5, min_strength=self.prune_threshold, slot=slot)
        for it in items:
            self.store.mark_used(it)
        ctx = "\n".join(it.text for it in items)
        msgs = [
            {"role": "system", "content":
             "Answer the question using the stored memories. Give the final answer "
             "clearly at the end. If a memory was updated, use the most recent value."},
            {"role": "user", "content": f"Memories:\n{ctx}\n\nQuestion: {question}"},
        ]
        # Generous token budget: qwen3's thinking preface cannot be disabled on
        # this GGUF, so let it finish and rely on containment scoring.
        return self.llm.chat(msgs, max_tokens=400)


METHODS = {
    "full_context": FullContextMethod,
    "naive_rag": NaiveRAGMethod,
    "mem0_style": Mem0StyleMethod,
    "igm": IGMMethod,
}
