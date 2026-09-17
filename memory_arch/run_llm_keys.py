"""Can an LLM assign the write-time attribute keys that regex could not?

The slot rule reads the key out of the sentence itself and was measured at 0/72
key reuse on natural updates (``run_write_coverage``).  The E3b ``current_only``
arm shows what perfect keys would be worth (58.8% vs 41.3% for similarity-ordered
evidence), so the open question is whether any practical key source reaches that
ceiling.

This runner uses the honest write-time protocol: one user turn at a time, no
access to the question and no access to the other mention of the value.  For
every knowledge-update pair it then asks whether the two mentions share a key,
which is the property supersede needs.

Key comparison is reported twice, because a real system would canonicalise:

  exact  : normalised string equality
  fuzzy  : cosine of the two keys under the same embedder, above a threshold

Cost is reported per call and extrapolated, since the write path would pay it
per stored turn.

Prompt provenance: the first version asked for the key of a message and declined
when no durable fact was present.  A two-instance smoke test showed it classified
messages that mix a fact with a request ("I recently ran 27:12. Any tips?") as
NONE, which silently drops the older mention of an updated value.  The rule and
example for mixed messages were added before the full run, and this is disclosed
rather than presented as a tuned classifier; nothing else was adjusted after
seeing results.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from . import Embedder, cosine
from .longmemeval_retrieval import select_partition
from .run_reader_eval import TransformersReader, user_turns
from .run_stale_leak import identify_new_session

SYSTEM_PROMPT = (
    "You read one user message from a long conversation and name the attribute it "
    "states about the user, if any.\n"
    "Rules:\n"
    "- Output a short attribute key (1-5 lowercase English words) naming WHAT the "
    "fact is about, never its value.\n"
    "- A message often contains a fact AND a request (\"... I recently ran 27:12. "
    "Any tips?\"). Name the fact; do not answer the request.\n"
    "- Output NONE only if the message states no durable fact about the user.\n"
    "- Output only the key or NONE, nothing else.\n"
    "Examples:\n"
    "message: My favourite colour is teal. -> favourite colour\n"
    "message: I just moved to Shenzhen. -> home city\n"
    "message: I set a personal best of 27:12 last month. Any tips to improve? -> personal best\n"
    "message: Could you summarise the article for me? -> NONE\n"
    "message: Thanks, that's really helpful! -> NONE\n"
)

GRANULAR_SUFFIX = (
    "- The key must be specific enough that a later message changing the same thing "
    "would receive the same key. Avoid broad topics: prefer \"yoga frequency\" over "
    "\"self-care\", prefer \"5k personal best\" over \"fitness\".\n"
)

PROMPT_VARIANTS = ("plain", "granular")


def system_prompt(variant: str) -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant!r}")
    if variant == "plain":
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT.replace("Rules:\n", "Rules:\n" + GRANULAR_SUFFIX)


NONE_TOKENS = {"none", "none.", "n/a", "na"}
FUZZY_THRESHOLD = 0.85


def normalize_key(raw: str) -> str | None:
    """Clean one model output into a key, or None when it declines."""
    text = re.sub(r"\s+", " ", str(raw).strip().strip('"\'`.,')).lower()
    if not text or text in NONE_TOKENS or text.startswith("none"):
        return None
    # The model sometimes answers with a full sentence; keep it auditable by
    # truncating rather than guessing an extraction out of it.
    return text[:60]


def extract_keys(reader: Any, texts: Sequence[str], max_new_tokens: int = 24,
                 thinking: bool = False, variant: str = "plain") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt = system_prompt(variant)
    for index, text in enumerate(texts):
        started = time.time()
        raw, answer_tokens = reader.answer(prompt, text,
                                          max_new_tokens=max_new_tokens,
                                          thinking=thinking)
        rows.append({
            "turn_index": index,
            "text": text,
            "raw": raw,
            "key": normalize_key(raw),
            "answer_tokens": answer_tokens,
            "seconds": time.time() - started,
        })
    return rows


def key_sets(rows: Sequence[dict[str, Any]], embedder: Any) -> tuple[set[str], dict[str, list[float]]]:
    keys = {row["key"] for row in rows if row["key"]}
    return keys, {key: embedder.embed(key) for key in keys}


def pair_overlap(old_keys: set[str], new_keys: set[str],
                 vectors: dict[str, list[float]],
                 threshold: float = FUZZY_THRESHOLD) -> dict[str, Any]:
    exact = sorted(old_keys & new_keys)
    if exact:
        return {"exact": exact[:1], "fuzzy": exact[:1], "best_cosine": 1.0}
    best: tuple[float, str, str] = (0.0, "", "")
    for old in old_keys:
        for new in new_keys:
            similarity = cosine(vectors[old], vectors[new])
            if similarity > best[0]:
                best = (similarity, old, new)
    fuzzy = [f"{best[1]} ~ {best[2]}"] if best[0] >= threshold else []
    return {"exact": [], "fuzzy": fuzzy, "best_cosine": round(best[0], 3)}


def run(entries: list[dict[str, Any]], reader: Any, embedder: Any,
        limit: int | None = None, thinking: bool = False,
        variant: str = "plain") -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    for entry in entries:
        new_id, how, gold_ids = identify_new_session(entry)
        if new_id is None:
            continue
        old_ids = [session_id for session_id in gold_ids if session_id != new_id]
        old_texts = [text for session_id in old_ids for text in user_turns(entry, session_id)]
        new_texts = user_turns(entry, new_id)
        old_rows = extract_keys(reader, old_texts, thinking=thinking, variant=variant)
        new_rows = extract_keys(reader, new_texts, thinking=thinking, variant=variant)
        for row in old_rows:
            turn_rows.append({"question_id": entry["question_id"], "side": "old", **row})
        for row in new_rows:
            turn_rows.append({"question_id": entry["question_id"], "side": "new", **row})
        old_keys, old_vectors = key_sets(old_rows, embedder)
        new_keys, new_vectors = key_sets(new_rows, embedder)
        overlap = pair_overlap(old_keys, new_keys, {**old_vectors, **new_vectors})
        pairs.append({
            "question_id": entry["question_id"],
            "identification": how,
            "old_keys": sorted(old_keys),
            "new_keys": sorted(new_keys),
            "exact_overlap": overlap["exact"],
            "fuzzy_overlap": overlap["fuzzy"],
            "best_cosine": overlap["best_cosine"],
        })
        print(f"  {entry['question_id'][:14]} exact={bool(overlap['exact'])} "
              f"fuzzy={bool(overlap['fuzzy'])} best={overlap['best_cosine']}", flush=True)
        if limit is not None and len(pairs) >= limit:
            break

    n = len(pairs)
    exact = sum(1 for pair in pairs if pair["exact_overlap"])
    fuzzy = sum(1 for pair in pairs if pair["exact_overlap"] or pair["fuzzy_overlap"])
    seconds = [row["seconds"] for row in turn_rows]
    fired = sum(1 for row in turn_rows if row["key"])
    return {
        "config": {"fuzzy_threshold": FUZZY_THRESHOLD, "thinking": thinking,
                   "limit": limit, "n_pairs": n, "prompt_variant": variant},
        "key_stability": {
            "n_pairs": n,
            "exact_pairs": exact,
            "fuzzy_pairs": fuzzy,
            "exact_rate": exact / n if n else 0.0,
            "fuzzy_rate": fuzzy / n if n else 0.0,
        },
        "fire_rate": {
            "user_turns": len(turn_rows),
            "turns_with_key": fired,
            "rate": fired / len(turn_rows) if turn_rows else 0.0,
        },
        "cost": {
            "calls": len(turn_rows),
            "seconds_mean": statistics.mean(seconds) if seconds else 0.0,
            "seconds_total": sum(seconds),
            "answer_tokens_mean": statistics.mean(row["answer_tokens"] for row in turn_rows) if turn_rows else 0.0,
        },
        "pairs": pairs,
        "turns": turn_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/longmemeval-s-llm-keys.json")
    parser.add_argument("--model", required=True, help="HF checkpoint of the extractor")
    parser.add_argument("--embed-model", required=True, help="embedder for fuzzy key matching")
    parser.add_argument("--partition", choices=["all", "development", "test"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="plain")
    args = parser.parse_args()

    entries = json.loads(Path(args.input).read_text(encoding="utf-8"))
    entries = [e for e in select_partition(entries, args.partition)
               if e["question_type"] == "knowledge-update"
               and not e["question_id"].endswith("_abs")]
    reader = TransformersReader(args.model)
    embedder = Embedder(backend="bge", model_path=args.embed_model)
    report = run(entries, reader, embedder, limit=args.limit,
                 thinking=args.thinking, variant=args.prompt_variant)
    report["config"].update({"input": args.input, "extractor_model": args.model,
                             "embed_model": args.embed_model, "partition": args.partition})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    stability = report["key_stability"]
    print(json.dumps({
        "key_stability": f"{stability['exact_pairs']}/{stability['n_pairs']} exact, "
                         f"{stability['fuzzy_pairs']}/{stability['n_pairs']} fuzzy",
        "fire_rate": f"{report['fire_rate']['rate']:.1%} of {report['fire_rate']['user_turns']} turns",
        "cost_per_call_s": round(report["cost"]["seconds_mean"], 2),
        "saved": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
