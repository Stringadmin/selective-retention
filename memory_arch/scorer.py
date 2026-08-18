"""Learnable importance scorer for IGM (M1).

Replaces the hand-tuned heuristic weights (0.6/0.3/0.1) with a logistic
regression trained on weakly-supervised labels.  Labels are auto-generated:
self-referential durable facts ("我的X是Y") are positive, filler chit-chat is
negative.  This is the "learned write-gate" that distinguishes IGM from
heuristic extraction (Mem0) and LLM-self-judgment (Letta).

No sklearn dependency: logistic regression is implemented in numpy so the
whole scorer stays inside the project's existing runtime.
"""

from __future__ import annotations

import json
import math
import os

# Feature extraction is deliberately transparent (and matches the heuristic's
# signals) so the learned weights are interpretable — a selling point for the
# mechanism-analysis angle of the paper.
FACT_MARKERS = ("我", "我的", "喜欢", "是", "在", "去过", "住", "工作", "现在",
                "叫", "名字", "来自", "出生", "毕业", "擅长")
QUESTION_MARKERS = ("什么", "吗", "？", "?", "哪", "怎么", "如何", "为什么")


def featurize(text: str, max_sim_to_store: float = 0.0) -> list[float]:
    """Return a feature vector for a candidate memory text.

    Features:
      0: bias (1.0)
      1: self-referential fact-marker density (0..1)
      2: is-question flag (1 if interrogative, facts are declarative)
      3: surprise (1 - max similarity to already-stored memories)
      4: information density (length-normalized, capped)
      5: has first-person ("我") flag
    """
    marker_hits = sum(1 for mk in FACT_MARKERS if mk in text)
    fact_density = min(marker_hits / 3.0, 1.0)
    is_q = 1.0 if any(q in text for q in QUESTION_MARKERS) else 0.0
    surprise = 1.0 - max_sim_to_store
    density = min(len(text.split()) / 20.0, 1.0)
    first_person = 1.0 if ("我" in text) else 0.0
    return [1.0, fact_density, is_q, surprise, density, first_person]


class ImportanceScorer:
    """Logistic-regression write-gate.  score(text) -> probability in [0,1]."""

    FEATURE_NAMES = ["bias", "fact_density", "is_question", "surprise", "density", "first_person"]

    def __init__(self, weights: list[float] | None = None):
        # Default: untrained zero weights (score = 0.5 for everything).
        self.weights = list(weights) if weights is not None else [0.0] * 6

    def score(self, text: str, max_sim_to_store: float = 0.0) -> float:
        x = featurize(text, max_sim_to_store)
        z = sum(w * xi for w, xi in zip(self.weights, x))
        return 1.0 / (1.0 + math.exp(-z))

    def train(self, examples: list[tuple[str, int, float]], lr: float = 0.5,
              epochs: int = 300, l2: float = 1e-3) -> dict:
        """Batch gradient descent on (text, label, max_sim) examples.

        label: 1 = worth remembering, 0 = not.  max_sim: precomputed surprise
        context (0 for isolated training).  Returns training metrics.
        """
        X = [featurize(t, ms) for t, _, ms in examples]
        y = [lbl for _, lbl, _ in examples]
        w = [0.0] * len(X[0])
        n = len(X)
        history = []
        for ep in range(epochs):
            grad = [0.0] * len(w)
            loss = 0.0
            for xi, yi in zip(X, y):
                z = sum(wj * xj for wj, xj in zip(w, xi))
                p = 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))
                loss += -(yi * math.log(p + 1e-9) + (1 - yi) * math.log(1 - p + 1e-9))
                for j in range(len(w)):
                    grad[j] += (p - yi) * xi[j]
            for j in range(len(w)):
                # L2 on all but the bias term.
                reg = l2 * w[j] if j > 0 else 0.0
                w[j] -= lr * (grad[j] / n + reg)
            if ep % 50 == 0:
                history.append(loss / n)
        self.weights = w
        # Final accuracy on the training set.
        correct = 0
        for xi, yi in zip(X, y):
            p = 1.0 / (1.0 + math.exp(-sum(wj * xj for wj, xj in zip(w, xi))))
            correct += int((p >= 0.5) == bool(yi))
        return {"train_accuracy": correct / n, "n_examples": n,
                "loss_history": history, "weights": dict(zip(self.FEATURE_NAMES, w))}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"weights": self.weights, "features": self.FEATURE_NAMES}, f)

    @classmethod
    def load(cls, path: str) -> "ImportanceScorer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(weights=d["weights"])
