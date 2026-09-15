"""Measure slot extraction on out-of-template phrasing.

Reports the labelled-corpus scores for both Python copies of the rule, plus the
store-level consequences (updates that fail to supersede, unrelated memories
masked by a colliding slot).  See memory_arch/slot_ood.py for why the synthetic
loads cannot surface any of this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from igm import Memory
from igm.gate import extract_slot
from memory_arch import IGMMethod
from memory_arch.slot_ood import (
    COLLISION_PAIRS,
    NAMED_ATTRIBUTE,
    NOT_ATTRIBUTE,
    UPDATE_CHAINS,
    evaluate,
)


def _stored_items(old: str, new: str):
    mem = Memory()
    mem.add(old)
    mem.add(new)
    # The current projection: what a reader sees.  Under archiving supersede the
    # archive (`mem.store.items`) additionally holds the masked predecessor.
    return mem.store.current()


def _supersede_outcome(old: str, new: str) -> dict:
    mem = Memory()
    mem.add(old)
    mem.add(new)
    archived = [item.text for item in mem.store.items if item.valid_to is not None]
    return {
        "old": old,
        "new": new,
        "stored": len(mem.store.current()),
        "events": mem.store.event_count,
        "archived": archived,
        "slots": [item.slot for item in mem.store.current()],
        "kept": [item.text for item in mem.store.current()],
    }


def build_oracle() -> dict:
    """Ground truth plus labels, keyed by exact utterance text.

    dsh-igm-memory/test/test_igm_plugin.mjs reads this to hold the third copy of
    the rule (lib/index.js) to the same standard.  Carrying the labels as well
    means the JS suite needs no copy of the corpus and cannot drift from it.
    """
    # An utterance can appear in two categories (a collision sentence is also a
    # non-attribute case), so de-duplicate before counting.
    texts = list(dict.fromkeys(
        [c.text for c in NOT_ATTRIBUTE] + [c.text for c in NAMED_ATTRIBUTE]
        + [t for chain in UPDATE_CHAINS for t in (chain.old, chain.new)]
        + [t for pair in COLLISION_PAIRS for t in (pair.first, pair.second)]
    ))
    return {
        "slots": {text: extract_slot(text) for text in texts},
        "not_attribute": [c.text for c in NOT_ATTRIBUTE],
        "named_attribute": {c.text: c.expect for c in NAMED_ATTRIBUTE},
        "update_chains": [
            {"old": c.old, "new": c.new, "attribute": c.attribute, "defect": c.defect}
            for c in UPDATE_CHAINS
        ],
        "collisions": [{"first": p.first, "second": p.second, "shared_slot": p.shared_slot}
                       for p in COLLISION_PAIRS],
        "counts": {
            "slots": len(texts),
            "not_attribute": len(NOT_ATTRIBUTE),
            "named_attribute": len(NAMED_ATTRIBUTE),
            "update_chains": len(UPDATE_CHAINS),
            "collisions": len(COLLISION_PAIRS),
        },
        "write_pairs": {
            f"{a}||{b}": [item.text for item in _stored_items(a, b)]
            for a, b in [(c.old, c.new) for c in UPDATE_CHAINS]
            + [(p.first, p.second) for p in COLLISION_PAIRS]
        },
    }


def build_report() -> dict:
    return {
        "summary": {
            "igm_gate": evaluate(extract_slot),
            "memory_arch_igm_method": evaluate(IGMMethod._extract_slot),
            "python_implementations_agree": (
                evaluate(extract_slot) == evaluate(IGMMethod._extract_slot)
            ),
        },
        "oracle": build_oracle(),
        "update_chains": [
            {"attribute": chain.attribute, "defect": chain.defect,
             **_supersede_outcome(chain.old, chain.new)}
            for chain in UPDATE_CHAINS
        ],
        "collisions": [
            {"shared_slot": pair.shared_slot, **_supersede_outcome(pair.first, pair.second)}
            for pair in COLLISION_PAIRS
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/slot-ood-baseline.json")
    args = parser.parse_args()

    report = build_report()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = report["summary"]["igm_gate"]
    print(f"非属性句误抽率   {gate['false_extraction_rate']:.1%} "
          f"({gate['not_attribute_extracted']}/{gate['not_attribute_total']})")
    print(f"命名属性键正确率 {gate['named_key_accuracy']:.1%} "
          f"({gate['named_exact']}/{gate['named_total']})")
    masked = [c for c in report["collisions"] if c["stored"] < 2]
    destroyed = [c for c in masked if c["events"] < 2]
    stale = [c for c in report["update_chains"] if c["stored"] > 1 and c["defect"]]
    print(f"无关记忆被遮蔽   {len(masked)}/{len(report['collisions'])}"
          f"（其中不可恢复 {len(destroyed)}）")
    print(f"更新未覆盖旧值   {len(stale)}/{len(report['update_chains']) - 1}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
