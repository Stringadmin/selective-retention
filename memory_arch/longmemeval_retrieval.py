"""Offline evidence-retrieval evaluation for LongMemEval-S.

This module evaluates only whether a retriever brings the benchmark's labeled
evidence sessions into its top-k. It deliberately does not claim answer
correctness: LongMemEval's QA score needs a reader and its official evaluator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from . import Embedder, cosine


DEFAULT_TOP_KS = (1, 3, 5, 10, 30, 50)

# Frozen from the internal development partition only.  These phrases refer
# specifically to a prior assistant response; a generic "you" does not.
ASSISTANT_HISTORY_PATTERNS = (
    r"\bprevious (?:chat|conversation)\b",
    r"\blast time\b",
    r"\byou (?:recommended|told me|mentioned|said|created)\b",
    r"\bdid you (?:say|tell|mention|recommend)\b",
    r"\bwe (?:outlined|discussed)\b",
)
_ASSISTANT_HISTORY_RE = re.compile("(?:" + "|".join(ASSISTANT_HISTORY_PATTERNS) + ")", re.IGNORECASE)


def needs_assistant_history(question: str) -> bool:
    """Return whether wording explicitly asks about a prior assistant reply.

    This is a deterministic routing heuristic, not a learned classifier.  It
    deliberately uses text alone and never consults LongMemEval's question
    type, which is available only to the evaluator.
    """
    return bool(_ASSISTANT_HISTORY_RE.search(question))


def session_documents(entry: dict[str, Any], include_assistant: bool = True) -> list[tuple[str, str]]:
    """Create one retrieval document for every timestamped history session."""
    documents: list[tuple[str, str]] = []
    for session_id, turns in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        text = "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in turns
            if include_assistant or turn["role"] == "user"
        )
        # Match LongMemEval's flat session index: retain a session ID even
        # when user-only projection has no text for an assistant-only session.
        documents.append((session_id, text))
    return documents


def _dcg(relevances: Sequence[int], k: int) -> float:
    """Match LongMemEval's published retrieval evaluator discount formula."""
    values = relevances[:k]
    if not values:
        return 0.0
    score = float(values[0])
    for rank, relevance in enumerate(values[1:], start=2):
        # The official implementation uses log2(2) for the second item.
        import math
        score += relevance / math.log2(rank)
    return score


def retrieval_metrics(
    ranked_ids: Sequence[str],
    answer_session_ids: Iterable[str],
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, float]:
    """Compute session-level recall and nDCG from official evidence labels."""
    targets = set(answer_session_ids)
    if not targets:
        raise ValueError("answer_session_ids must not be empty for retrieval scoring")
    relevances = [int(session_id in targets) for session_id in ranked_ids]
    ideal = sorted(relevances, reverse=True)
    metrics: dict[str, float] = {}
    for k in top_ks:
        recalled = set(ranked_ids[:k])
        metrics[f"recall_any@{k}"] = float(bool(recalled & targets))
        metrics[f"recall_all@{k}"] = float(targets <= recalled)
        ideal_dcg = _dcg(ideal, k)
        metrics[f"ndcg@{k}"] = _dcg(relevances, k) / ideal_dcg if ideal_dcg else 0.0
    return metrics


def rank_sessions(
    question: str,
    documents: Sequence[tuple[str, str]],
    embedder: Any,
    recency_weight: float = 0.0,
) -> list[str]:
    """Rank sessions by semantic similarity plus an optional recency prior."""
    if recency_weight < 0:
        raise ValueError("recency_weight must be non-negative")
    texts = [question, *(text for _, text in documents)]
    if hasattr(embedder, "embed_many"):
        vectors = embedder.embed_many(texts)
    else:
        vectors = [embedder.embed(text) for text in texts]
    query, document_vectors = vectors[0], vectors[1:]
    denominator = max(len(documents) - 1, 1)
    scored = [
        (
            cosine(query, vector) + recency_weight * (position / denominator),
            position,
            session_id,
        )
        for position, ((session_id, _), vector) in enumerate(zip(documents, document_vectors))
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [session_id for _, _, session_id in scored]


def select_partition(entries: Iterable[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    """Make a stable 20% development / 80% test split from question IDs.

    LongMemEval-S ships as one evaluation set. This split is only for selecting
    an internal diagnostic parameter; it is not an official train/test split.
    """
    if partition not in {"all", "development", "test"}:
        raise ValueError("partition must be 'all', 'development', or 'test'")
    selected: list[dict[str, Any]] = []
    for entry in entries:
        bucket = hashlib.sha256(entry["question_id"].encode("utf-8")).digest()[0] % 5
        if partition == "all" or (partition == "development" and bucket == 0) or (
            partition == "test" and bucket != 0
        ):
            selected.append(entry)
    return selected


def evaluate_entries(
    entries: Iterable[dict[str, Any]],
    embedder: Any,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    include_assistant: bool = True,
    role_aware: bool = False,
    recency_weight: float = 0.0,
) -> dict[str, Any]:
    """Evaluate a retriever against LongMemEval session labels.

    The official benchmark omits abstention questions from retrieval metrics,
    because they deliberately have no evidence session. This function mirrors
    that rule and reports the count explicitly.
    """
    all_metrics: list[dict[str, float]] = []
    per_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    sampled_failures: list[dict[str, Any]] = []
    total = 0
    skipped_abstention = 0
    skipped_without_documents = 0
    assistant_history_routes = 0
    user_history_routes = 0

    for entry in entries:
        total += 1
        if entry["question_id"].endswith("_abs"):
            skipped_abstention += 1
            continue
        use_assistant_history = include_assistant
        if role_aware:
            use_assistant_history = needs_assistant_history(entry["question"])
            if use_assistant_history:
                assistant_history_routes += 1
            else:
                user_history_routes += 1
        documents = session_documents(entry, include_assistant=use_assistant_history)
        if not documents:
            skipped_without_documents += 1
            continue
        ranked_ids = rank_sessions(
            entry["question"],
            documents,
            embedder,
            recency_weight=recency_weight,
        )
        metrics = retrieval_metrics(ranked_ids, entry["answer_session_ids"], top_ks=top_ks)
        all_metrics.append(metrics)
        per_type[entry["question_type"]].append(metrics)
        if metrics["recall_any@1"] == 0.0 and len(sampled_failures) < 20:
            sampled_failures.append({
                "question_id": entry["question_id"],
                "question_type": entry["question_type"],
                "question": entry["question"],
                "answer_session_ids": entry["answer_session_ids"],
                "top_3_session_ids": ranked_ids[:3],
            })

    if not all_metrics:
        raise ValueError("no non-abstention entries with retrieval documents")

    def mean(rows: Sequence[dict[str, float]]) -> dict[str, float]:
        return {
            name: sum(row[name] for row in rows) / len(rows)
            for name in rows[0]
        }

    return {
        "scope": {
            "instances_total": total,
            "instances_scored": len(all_metrics),
            "skipped_abstention": skipped_abstention,
            "skipped_without_documents": skipped_without_documents,
            "document_granularity": "session",
            "retrieval_mode": (
                "role-aware" if role_aware else "all-turns" if include_assistant else "user-only"
            ),
            "include_assistant": include_assistant,
            "role_aware": role_aware,
            "assistant_history_routes": assistant_history_routes if role_aware else None,
            "user_history_routes": user_history_routes if role_aware else None,
            "recency_weight": recency_weight,
            "top_ks": list(top_ks),
        },
        "overall": mean(all_metrics),
        "by_question_type": {
            question_type: mean(rows)
            for question_type, rows in sorted(per_type.items())
        },
        "sampled_top1_failures": sampled_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embed-model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--user-only", action="store_true")
    parser.add_argument(
        "--role-aware",
        action="store_true",
        help="Use all turns only for explicit prior-assistant-history questions.",
    )
    parser.add_argument("--recency-weight", type=float, default=0.0)
    parser.add_argument(
        "--partition",
        choices=("all", "development", "test"),
        default="all",
        help="Deterministic internal split for diagnostic parameter selection.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.user_only and args.role_aware:
        raise ValueError("--user-only and --role-aware cannot be combined")
    entries = json.loads(args.input.read_text(encoding="utf-8"))
    entries = select_partition(entries, args.partition)
    if args.limit is not None:
        entries = entries[:args.limit]
    result = evaluate_entries(
        entries,
        Embedder(backend="bge", model_path=args.embed_model),
        include_assistant=not args.user_only,
        role_aware=args.role_aware,
        recency_weight=args.recency_weight,
    )
    result["config"] = {
        "input": str(args.input),
        "instances_loaded": len(entries),
        "embed_backend": "bge",
        "partition": args.partition,
        "retrieval_mode": (
            "role-aware" if args.role_aware else "all-turns" if not args.user_only else "user-only"
        ),
        "recency_weight": args.recency_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scope": result["scope"], "overall": result["overall"]}, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
