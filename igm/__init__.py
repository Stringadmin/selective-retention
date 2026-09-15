"""IGM — Importance-Gated Memory: a write layer for RAG.

Public API:
    Memory                       the entry point
    WriteGate / Scorer           the write-side gate
    MemoryStore / MemoryItem     event archive + current view: slot supersede, history, forgetting
    Embedder backends            HashEmbedder / SentenceTransformerEmbedder / CallableEmbedder
    LearnedScorer / HeuristicScorer
"""

from .embedders import (
    CallableEmbedder,
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
    cosine,
)
from .gate import HeuristicScorer, LearnedScorer, Scorer, WriteGate, extract_slot
from .memory import Memory
from .store import MemoryItem, MemoryStore

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "MemoryItem",
    "MemoryStore",
    "WriteGate",
    "Scorer",
    "HeuristicScorer",
    "LearnedScorer",
    "extract_slot",
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "CallableEmbedder",
    "cosine",
    "__version__",
]
