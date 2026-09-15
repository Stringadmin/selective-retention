"""Select a simple temporal retrieval prior on a LongMemEval-S dev split.

This is a diagnostic ablation, not an official LongMemEval leaderboard run.
The public release has one evaluation split, so this runner deterministically
reserves 20% of question IDs for parameter selection and leaves 80% untouched
for one follow-up evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import Embedder
from .longmemeval_retrieval import evaluate_entries, select_partition


SELECTION_METRIC = "recall_all@10"


def select_weight(results: dict[float, dict[str, Any]]) -> float:
    """Choose the best development score, preferring the smaller prior on ties."""
    return max(
        results,
        key=lambda weight: (results[weight]["overall"][SELECTION_METRIC], -weight),
    )


def run(
    entries: list[dict[str, Any]],
    embedder: Any,
    weights: Sequence[float],
) -> dict[str, Any]:
    if not weights or any(weight < 0 for weight in weights):
        raise ValueError("weights must contain non-negative values")
    development = select_partition(entries, "development")
    results = {
        weight: evaluate_entries(
            development,
            embedder,
            include_assistant=False,
            recency_weight=weight,
        )
        for weight in weights
    }
    selected_weight = select_weight(results)
    return {
        "config": {
            "partition": "development",
            "partitioning": "sha256(question_id)[0] % 5 == 0",
            "document_mode": "user-only session",
            "selection_metric": SELECTION_METRIC,
            "candidate_recency_weights": list(weights),
        },
        "selected_recency_weight": selected_weight,
        "results": {str(weight): result for weight, result in results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embed-model", required=True)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2),
    )
    args = parser.parse_args()

    entries = json.loads(args.input.read_text(encoding="utf-8"))
    result = run(entries, Embedder(backend="bge", model_path=args.embed_model), args.weights)
    result["config"]["input"] = str(args.input)
    result["config"]["instances_loaded"] = len(entries)
    result["config"]["embed_backend"] = "bge"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        weight: report["overall"][SELECTION_METRIC]
        for weight, report in result["results"].items()
    }
    print(json.dumps({"selection_metric": SELECTION_METRIC, "scores": summary,
                      "selected_recency_weight": result["selected_recency_weight"]}, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
