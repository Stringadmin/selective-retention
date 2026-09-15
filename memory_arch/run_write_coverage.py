"""Coverage of surface-syntax write keys on natural conversation text.

The slot rule in ``igm/gate.py`` reads the attribute key out of the sentence
itself ("我的{ATTR}是...").  The decisive synthetic load was generated from
templates shaped exactly like that, so extraction was never the bottleneck
there.  This module asks the prior question on real data: how often does a
natural conversation contain an extractable attribute key at all, and when it
does, is the key stable across the two mentions of an updated value?

The English pattern set below is a literal translation of the relations the
Chinese rule implements ("my X is now Y" / "I changed my X to Y" / naming).
It was written down BEFORE measuring and is deliberately not tuned afterwards:
a pattern set iterated against the same corpus it is scored on would report an
overfit number.  Whatever it measures, the number is a weak upper bound on what
this family of rule can reach, because a natural update frequently carries no
surface key at all ("I've tried four different ones so far").
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from igm.gate import extract_slot


# Literal English analogues of the Chinese relational patterns. Frozen before
# measurement; see the module docstring for why they are not tuned.
ENGLISH_PATTERNS: tuple[re.Pattern[str], ...] = (
    # I changed/switched/upgraded my X to Y
    re.compile(r"\bI (?:changed|switched|upgraded|moved) my (?P<attr>[A-Za-z][A-Za-z '\-]{1,30}?) (?:to|over to)\b"),
    # my X is now Y / my new X is Y / my current X is Y
    re.compile(r"\bmy (?:new |current |updated )?(?P<attr>[A-Za-z][A-Za-z '\-]{1,30}?) is (?:now )?"
               r"(?=[A-Z0-9$])"),
    # the X I use is Y / the X I'm using is Y
    re.compile(r"\bthe (?P<attr>[A-Za-z][A-Za-z '\-]{1,30}?) I(?:'m| am)? us(?:e|ing) is\b"),
    # I set a/ my X to Y
    re.compile(r"\bI (?:set|reset) (?:a |my |the )?(?P<attr>[A-Za-z][A-Za-z '\-]{1,30}?) to\b"),
)

_STOP_KEYS = {
    "name", "question", "point", "problem", "thing", "way", "idea", "guess", "opinion",
    "plan", "schedule", "goal", "dream", "favorite", "first", "last", "best", "fault",
}


def extract_slot_en(text: str) -> str | None:
    """Return the English surface key, or None. Mirrors the Chinese contract."""
    if not isinstance(text, str) or not text.strip():
        return None
    for pattern in ENGLISH_PATTERNS:
        match = pattern.search(text)
        if match:
            key = re.sub(r"\s+", " ", match.group("attr")).strip().lower()
            if key and key not in _STOP_KEYS:
                return key
    return None


def _turns(entry: dict[str, Any], session_id: str, role: str = "user") -> list[str]:
    sessions = dict(zip(entry["haystack_session_ids"], entry["haystack_sessions"]))
    return [t["content"] for t in sessions.get(session_id, []) if t["role"] == role]


def _keys(texts: list[str]) -> set[str]:
    return {key for key in (extract_slot_en(text) for text in texts) if key}


def key_stability(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """How often an updated value carries the same surface key twice.

    Reuses the old/new session identification from ``run_stale_leak`` so the
    corpus and labels are identical to the leakage measurement.
    """
    from .run_stale_leak import identify_new_session

    rows = []
    for entry in entries:
        if entry["question_id"].endswith("_abs"):
            continue
        new_id, how, gold_ids = identify_new_session(entry)
        if new_id is None:
            continue
        old_ids = [session_id for session_id in gold_ids if session_id != new_id]
        old_keys: set[str] = set()
        for session_id in old_ids:
            old_keys |= _keys(_turns(entry, session_id))
        new_keys = _keys(_turns(entry, new_id))
        shared = old_keys & new_keys
        rows.append({
            "question_id": entry["question_id"],
            "identification": how,
            "old_keys": sorted(old_keys),
            "new_keys": sorted(new_keys),
            "shared_key": sorted(shared)[0] if shared else None,
        })
    total = len(rows)
    shared = [row for row in rows if row["shared_key"]]
    return {
        "n_pairs": total,
        "n_shared_key": len(shared),
        "key_stability_rate": len(shared) / total if total else 0.0,
        "n_new_side_only": sum(1 for row in rows if not row["shared_key"] and row["new_keys"]),
        "examples": rows[:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/write-coverage-natural.json")
    parser.add_argument("--audit-sample", type=int, default=40)
    args = parser.parse_args()

    entries = json.loads(Path(args.input).read_text(encoding="utf-8"))

    # 1. Fire rate of the shipped rule and of the literal English translation.
    turns = 0
    fired_zh = 0
    fired_en = 0
    en_hits: Counter[str] = Counter()
    for entry in entries:
        for session in entry["haystack_sessions"]:
            for turn in session:
                if turn["role"] != "user":
                    continue
                turns += 1
                if extract_slot(turn["content"]):
                    fired_zh += 1
                key = extract_slot_en(turn["content"])
                if key:
                    fired_en += 1
                    en_hits[key] += 1

    # 2. Key stability across the two mentions of an updated value.
    ku = [entry for entry in entries if entry["question_type"] == "knowledge-update"]
    stability = key_stability(ku)

    # 3. Deterministic audit sample: the most frequent keys, so the reviewer
    #    sees the rule's typical output rather than the tail.
    audit = [{"key": key, "count": count} for key, count in en_hits.most_common(args.audit_sample)]

    report = {
        "config": {"input": args.input, "patterns": [p.pattern for p in ENGLISH_PATTERNS]},
        "fire_rate": {
            "user_turns": turns,
            "shipped_rule_fired": fired_zh,
            "shipped_rule_rate": fired_zh / turns if turns else 0.0,
            "english_translation_fired": fired_en,
            "english_translation_rate": fired_en / turns if turns else 0.0,
        },
        "key_stability_knowledge_update": stability,
        "audit_top_keys": audit,
        "distinct_keys": len(en_hits),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fire = report["fire_rate"]
    print(f"user turns: {fire['user_turns']}")
    print(f"shipped rule fired: {fire['shipped_rule_fired']} ({fire['shipped_rule_rate']:.4%})")
    print(f"english translation fired: {fire['english_translation_fired']} "
          f"({fire['english_translation_rate']:.4%})")
    print(f"distinct english keys: {report['distinct_keys']}")
    print(f"key stability on knowledge-update pairs: {stability['n_shared_key']}/{stability['n_pairs']} "
          f"({stability['key_stability_rate']:.1%})")
    print(f"top keys: {[item['key'] for item in audit[:12]]}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
