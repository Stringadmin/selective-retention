"""Stale-value leakage on LongMemEval-S knowledge-update questions.

``longmemeval_retrieval.py`` scores whether the labeled evidence sessions reach
top-k.  It does not care about the ORDER inside that set, so it cannot see the
failure this module measures: a reader that consumes top-k in rank order meets
the superseded value before the current one.

LongMemEval labels two evidence sessions for a knowledge-update question: the
one where the value was corrected (new) and the earlier mention (old).  This
module identifies which is which, ranks sessions with the same retriever as the
retrieval report, and reports per top-k:

  stale_above_new : the old session outranks the new one
  new_missing     : the new session is not in top-k at all
  leak            : either of the above

That number is the real-data counterpart of the synthetic decisive load, which
made a stale value look indistinguishable from the current one by construction.
It is still an evidence-level measurement: it says what a reader would meet
first, not what it would answer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
from typing import Any, Sequence

from . import Embedder
from .longmemeval_retrieval import rank_sessions, select_partition, session_documents


DEFAULT_TOP_KS = (1, 3, 5, 10)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def identify_new_session(entry: dict[str, Any]) -> tuple[str | None, str, list[str]]:
    """Return ``(new_session_id, how, gold_ids)`` for one knowledge-update entry.

    The session that states the corrected value is the one containing the gold
    answer.  When the answer text cannot discriminate (booleans, counts stated
    in both sessions, or an empty match), fall back to the later date, and
    report which path was taken so the leakage number stays auditable.
    """
    gold_ids = list(entry.get("answer_session_ids") or [])
    if len(gold_ids) < 2:
        return None, "single_evidence", gold_ids

    answer = _normalize(entry.get("answer", ""))
    sessions = dict(zip(entry["haystack_session_ids"], entry["haystack_sessions"]))
    matched: list[str] = []
    if answer:
        for session_id in gold_ids:
            text = " ".join(
                f"{turn['role']}: {turn['content']}" for turn in sessions.get(session_id, [])
            )
            if answer in _normalize(text):
                matched.append(session_id)
    if len(matched) == 1:
        return matched[0], "answer_text", gold_ids

    dates = dict(zip(entry["haystack_session_ids"], entry["haystack_dates"]))
    ordered = sorted(gold_ids, key=lambda session_id: dates.get(session_id, ""))
    return ordered[-1], "date_order", gold_ids


def leak_metrics(
    ranked_ids: Sequence[str],
    new_id: str,
    old_id: str,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, float]:
    """Rank-order leakage of the superseded value against the current one."""
    position = {session_id: index + 1 for index, session_id in enumerate(ranked_ids)}
    new_pos = position.get(new_id)
    old_pos = position.get(old_id)
    metrics: dict[str, float] = {"mrr": 1.0 / new_pos if new_pos else 0.0}
    for k in top_ks:
        new_in = new_pos is not None and new_pos <= k
        old_in = old_pos is not None and old_pos <= k
        stale_above = float(old_in and old_pos < (new_pos or len(ranked_ids) + 1))
        missing = float(not new_in)
        metrics[f"stale_above_new@{k}"] = stale_above
        metrics[f"new_missing@{k}"] = missing
        metrics[f"leak@{k}"] = float(bool(stale_above or missing))
    return metrics


def run(
    entries: list[dict[str, Any]],
    embedder: Any,
    weights: Sequence[float],
    include_assistant: bool = False,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for weight in weights:
        rows: list[tuple[dict[str, float], str]] = []
        ident = defaultdict(int)
        per_type: dict[str, list[dict[str, float]]] = defaultdict(list)
        for entry in entries:
            if entry["question_id"].endswith("_abs"):
                continue
            new_id, how, gold_ids = identify_new_session(entry)
            ident[how] += 1
            if new_id is None:
                continue
            old_ids = [session_id for session_id in gold_ids if session_id != new_id]
            documents = session_documents(entry, include_assistant=include_assistant)
            ranked = rank_sessions(entry["question"], documents, embedder, recency_weight=weight)
            # The benchmark can label more than two sessions; leakage is scored
            # against the closest predecessor, i.e. the other earliest mention.
            dates = dict(zip(entry["haystack_session_ids"], entry["haystack_dates"]))
            old_id = min(old_ids, key=lambda session_id: dates.get(session_id, ""))
            metrics = leak_metrics(ranked, new_id, old_id, top_ks=top_ks)
            rows.append((metrics, how))
            per_type[entry["question_type"]].append(metrics)
        if not rows:
            raise ValueError("no knowledge-update entries with two evidence sessions")
        # answer_text pairs do not rely on the "later date means updated" assumption,
        # so they are reported separately as the robustness check.
        by_identification: dict[str, list[dict[str, float]]] = defaultdict(list)
        for metrics, how in rows:
            by_identification[how].append(metrics)
        results[str(weight)] = {
            "overall": {
                key: sum(metrics[key] for metrics, _ in rows) / len(rows)
                for key in rows[0][0]
            },
            "n_scored": len(rows),
            "identification": dict(sorted(ident.items())),
            "by_identification": {
                how: {
                    key: sum(row[key] for row in group) / len(group) for key in group[0]
                }
                for how, group in sorted(by_identification.items())
            },
            "by_question_type": {
                question_type: {
                    key: sum(row[key] for row in type_rows) / len(type_rows)
                    for key in type_rows[0]
                }
                for question_type, type_rows in sorted(per_type.items())
            },
        }
    return {
        "config": {
            "document_mode": "with assistant" if include_assistant else "user-only session",
            "candidate_recency_weights": list(weights),
            "top_ks": list(top_ks),
            "metric_note": (
                "stale_above_new: superseded session ranks above the corrected one; "
                "new_missing: corrected session outside top-k; leak: either"
            ),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="longmemeval_s_cleaned.json")
    parser.add_argument("--output", default="reports/longmemeval-s-stale-leak.json")
    parser.add_argument("--embed-model", required=True)
    parser.add_argument("--partition", choices=["all", "development", "test"], default="all")
    parser.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.02])
    parser.add_argument("--include-assistant", action="store_true")
    args = parser.parse_args()

    entries = json.loads(Path(args.input).read_text(encoding="utf-8"))
    entries = [e for e in select_partition(entries, args.partition)
               if e["question_type"] == "knowledge-update"]
    embedder = Embedder(backend="bge", model_path=args.embed_model)

    class CachedEmbedder:
        """Preserve vectors across recency weights within one process."""

        def __init__(self, delegate):
            self.delegate = delegate
            self.cache: dict[str, Any] = {}

        def embed(self, text: str):
            if text not in self.cache:
                self.cache[text] = self.delegate.embed(text)
            return self.cache[text]

    report = run(entries, CachedEmbedder(embedder), args.weights,
                 include_assistant=args.include_assistant)
    report["config"].update({"input": args.input, "partition": args.partition})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for weight, result in report["results"].items():
        overall = result["overall"]
        print(f"recency_weight={weight}  n={result['n_scored']}  "
              f"leak@1={overall['leak@1']:.3f} leak@10={overall['leak@10']:.3f}  "
              f"stale_above_new@10={overall['stale_above_new@10']:.3f}  "
              f"new_missing@10={overall['new_missing@10']:.3f}")
        print(f"  identification: {result['identification']}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
