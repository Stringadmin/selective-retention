"""Pluggable embedding backends for IGM.

IGM needs embeddings only for similarity search.  Any callable
``(str) -> list[float]`` works as a backend; three are provided out of the box:

- :class:`HashEmbedder` — zero-dependency hashing bag-of-words (default).
- :class:`SentenceTransformerEmbedder` — a real model (e.g. BGE) via
  ``sentence-transformers`` (optional dependency).
- :class:`CallableEmbedder` — wrap your own function (OpenAI, Cohere, ...).
"""

from __future__ import annotations

import math
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class HashEmbedder:
    """Deterministic hashing bag-of-words embedding.  Zero dependencies.

    Good enough for tests and small stores; use a real model for production.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            vec[hash(tok) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class CallableEmbedder:
    """Wrap any ``(str) -> list[float]`` callable as an embedder."""

    def __init__(self, fn: Callable[[str], list[float]]):
        self._fn = fn

    def embed(self, text: str) -> list[float]:
        return list(self._fn(text))


class SentenceTransformerEmbedder:
    """Real sentence embeddings via sentence-transformers (optional dep).

    Example:
        SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5")
        SentenceTransformerEmbedder("/path/to/local/bge")  # offline
    """

    def __init__(self, model: str, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires `sentence-transformers`. "
                "Install it with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model, device=device)

    def embed(self, text: str) -> list[float]:
        return [float(v) for v in self._model.encode(text, normalize_embeddings=True)]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
