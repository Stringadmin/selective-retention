"""Write gate: decide what is worth storing, and how to slot it.

The write gate is IGM's core.  Two responsibilities:

1. **Importance gating** — score a candidate text and decide whether to store
   it.  Uses a pluggable scorer; a heuristic and a learned logistic scorer are
   provided.
2. **Slot extraction** — pull the attribute key out of a self-referential fact
   ("我的{ATTR}是...") so the store can supersede it on update.
"""

from __future__ import annotations

import json
import math
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    def score(self, text: str, max_sim_to_store: float = 0.0) -> float:
        ...


# ---------------------------------------------------------------------------
# Slot extraction (knowledge-update routing)
# ---------------------------------------------------------------------------

_SLOT_PREFIXES = ("现在的", "目前的", "新的", "原来的", "以前的", "当前的")
_SLOT_ANCHORS = ("我的", "我")
_SLOT_STOPS = ("现在是", "是什么", "是", "了", "。", "，", ",", "？", "?")


def extract_slot(text: str, max_len: int = 6) -> str | None:
    """Extract the attribute key from a fact or question of the form
    '我的{ATTR}是...' / '我现在的{ATTR}是什么'.

    Temporal/status prefixes ("现在的") are stripped so a query and its fact
    map to the same slot.  Returns None for non-attribute text.
    """
    for anchor in _SLOT_ANCHORS:
        if anchor not in text:
            continue
        rest = text.split(anchor, 1)[1]
        for stop in _SLOT_STOPS:
            if stop in rest:
                attr = rest.split(stop, 1)[0].strip()
                for pref in _SLOT_PREFIXES:
                    if attr.startswith(pref):
                        attr = attr[len(pref):]
                if 0 < len(attr) <= max_len:
                    return attr
    return None


# ---------------------------------------------------------------------------
# Importance scorers
# ---------------------------------------------------------------------------

_FACT_MARKERS = ("我", "我的", "喜欢", "是", "在", "去过", "住", "工作", "现在",
                 "叫", "名字", "来自", "出生", "毕业", "擅长")
_QUESTION_MARKERS = ("什么", "吗", "？", "?", "哪", "怎么", "如何", "为什么")


def _featurize(text: str, max_sim_to_store: float) -> list[float]:
    marker_hits = sum(1 for mk in _FACT_MARKERS if mk in text)
    fact_density = min(marker_hits / 3.0, 1.0)
    is_q = 1.0 if any(q in text for q in _QUESTION_MARKERS) else 0.0
    surprise = 1.0 - max_sim_to_store
    density = min(len(text.split()) / 20.0, 1.0)
    first_person = 1.0 if "我" in text else 0.0
    return [1.0, fact_density, is_q, surprise, density, first_person]


class HeuristicScorer:
    """Hand-tuned importance scorer (no training data needed).

    Questions are never memories, so an interrogative hard-caps the score low;
    durable self-referential facts score high; filler stays below threshold.
    """

    def score(self, text: str, max_sim_to_store: float = 0.0) -> float:
        # Questions ask for memories; they are not themselves memories.
        if any(q in text for q in _QUESTION_MARKERS):
            return 0.0
        marker_hits = sum(1 for mk in _FACT_MARKERS if mk in text)
        fact_score = min(marker_hits / 3.0, 1.0)
        surprise = 1.0 - max_sim_to_store
        density = min(len(text.split()) / 20.0, 1.0)
        return 0.6 * fact_score + 0.3 * surprise + 0.1 * density


class LearnedScorer:
    """Logistic-regression scorer with interpretable learned weights."""

    FEATURE_NAMES = ["bias", "fact_density", "is_question", "surprise", "density", "first_person"]

    def __init__(self, weights: list[float] | None = None):
        self.weights = list(weights) if weights is not None else [0.0] * 6

    def score(self, text: str, max_sim_to_store: float = 0.0) -> float:
        x = _featurize(text, max_sim_to_store)
        z = sum(w * xi for w, xi in zip(self.weights, x))
        z = max(min(z, 30), -30)
        return 1.0 / (1.0 + math.exp(-z))

    @classmethod
    def load(cls, path: str) -> "LearnedScorer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(weights=d["weights"])


# ---------------------------------------------------------------------------
# Write gate
# ---------------------------------------------------------------------------


class WriteGate:
    """Decides whether a candidate text is written, and under which slot."""

    def __init__(self, scorer: Scorer | None = None, threshold: float = 0.6):
        self.scorer = scorer or HeuristicScorer()
        self.threshold = threshold

    def should_write(self, text: str, max_sim_to_store: float = 0.0) -> tuple[bool, float]:
        s = self.scorer.score(text, max_sim_to_store)
        return (s >= self.threshold), s

    def slot_for(self, text: str) -> str | None:
        return extract_slot(text)
