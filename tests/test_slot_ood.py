"""Slot extraction on phrasing outside the templates the rule was written for.

``reports/slot-ood-baseline.json`` holds the measured numbers; regenerate with
``python -m memory_arch.run_slot_ood``.

The former ``xfail(strict=True)`` marks were defect pins.  The shared extractor
now runs these cases as ordinary regression assertions; future OOD additions
should be independently labelled rather than hidden behind a permissive
fallback.
"""

import json
from pathlib import Path

import pytest

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

# A non-None slot arms supersede, so speech-act text must not create a guessed
# key that masks an unrelated memory.
FALSE_EXTRACTION_PINS = set()

# Attributes named in the sentence, including natural update and naming verbs.
WRONG_KEY_PINS = set()

# Attributes whose natural update phrasing leaves the old value stored.
UNCOVERED_UPDATE_PINS = set()

COLLIDING_SLOT_PINS = set()


def _labelled_cases(cases, pinned, reason, key=lambda case: case.text,
                    id_of=lambda case: case.text[:16]):
    return [
        pytest.param(
            case,
            id=id_of(case),
            marks=pytest.mark.xfail(strict=True, reason=reason) if key(case) in pinned else (),
        )
        for case in cases
    ]


@pytest.mark.parametrize("case", _labelled_cases(
    NOT_ATTRIBUTE, FALSE_EXTRACTION_PINS,
    '误抽：感叹语和引述句是 "我的X是" 形状，被当成属性'))
def test_non_attribute_utterance_gets_no_slot(case):
    assert extract_slot(case.text) is None


@pytest.mark.parametrize("case", _labelled_cases(
    NAMED_ATTRIBUTE, WRONG_KEY_PINS,
    '漏抽或键不稳定，见 case.why：规则依赖系词"是"，且只在键首剥离时态修饰'))
def test_named_attribute_maps_to_its_key(case):
    assert extract_slot(case.text) == case.expect


def _stored_after(old: str, new: str):
    mem = Memory()
    mem.add(old)
    mem.add(new)
    # The current projection, not the event archive: what a reader can see.
    return mem.store.current()


@pytest.mark.parametrize("chain", _labelled_cases(
    UPDATE_CHAINS, UNCOVERED_UPDATE_PINS,
    '更新未覆盖旧值，见 chain.defect：当前状态里同一属性留两份',
    key=lambda chain: chain.attribute, id_of=lambda chain: chain.attribute))
def test_update_leaves_exactly_one_current_value(chain):
    items = _stored_after(chain.old, chain.new)
    assert [item.text for item in items] == [chain.new]


def test_a_superseded_value_is_masked_not_destroyed():
    # Any two unrelated facts that share an extracted slot used to lose the
    # first one silently.  Extraction now refuses the known collision phrasing,
    # but the storage layer must still keep the predecessor recoverable.
    mem = Memory()
    mem.add("我的住址是北京。")
    mem.add("我的住址是深圳。")
    assert [item.text for item in mem.store.current()] == ["我的住址是深圳。"]
    assert [item.text for item in mem.store.history("住址")] == [
        "我的住址是北京。",
        "我的住址是深圳。",
    ]
    assert mem.previous("住址").text == "我的住址是北京。"


@pytest.mark.parametrize("pair", _labelled_cases(
    COLLISION_PAIRS, COLLIDING_SLOT_PINS,
    '无关记忆共用一个 slot，后写的那条把前一条遮蔽掉',
    key=lambda pair: pair.shared_slot, id_of=lambda pair: pair.shared_slot))
def test_colliding_slot_keeps_both_unrelated_memories(pair):
    items = _stored_after(pair.first, pair.second)
    assert [item.text for item in items] == [pair.first, pair.second]


def test_the_two_python_copies_of_the_rule_stay_in_lockstep():
    # IGMMethod delegates to igm.gate.extract_slot; the DSH plugin has a second
    # copy in lib/index.js guarded by its own parity suite.
    assert evaluate(extract_slot) == evaluate(IGMMethod._extract_slot)


def test_reported_baseline_still_matches_the_rule():
    path = Path(__file__).resolve().parents[1] / "reports" / "slot-ood-baseline.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["summary"]["igm_gate"] == evaluate(extract_slot), (
        "抽取行为已变化：跑 python -m memory_arch.run_slot_ood 重生成基线，"
        "并同步更新本文件的 defect pins"
    )


def test_oracle_covers_every_corpus_text():
    # dsh-igm-memory/test/test_igm_plugin.mjs reads report["oracle"] as its only
    # copy of the corpus.  A partial oracle would silently shrink the JS suite.
    path = Path(__file__).resolve().parents[1] / "reports" / "slot-ood-baseline.json"
    oracle = json.loads(path.read_text(encoding="utf-8"))["oracle"]
    # One sentence can legitimately belong to two categories, so compare sets.
    expected = {c.text for c in NOT_ATTRIBUTE} | {c.text for c in NAMED_ATTRIBUTE}
    expected |= {t for c in UPDATE_CHAINS for t in (c.old, c.new)}
    expected |= {t for p in COLLISION_PAIRS for t in (p.first, p.second)}
    assert set(oracle["slots"]) == expected
    assert set(oracle["write_pairs"]) == {
        f"{c.old}||{c.new}" for c in UPDATE_CHAINS
    } | {f"{p.first}||{p.second}" for p in COLLISION_PAIRS}
    assert oracle["not_attribute"] == [c.text for c in NOT_ATTRIBUTE]
    assert oracle["named_attribute"] == {c.text: c.expect for c in NAMED_ATTRIBUTE}
    # Sizes are stated explicitly so the JS suite can reject a truncated oracle
    # instead of comparing a silently smaller sample.
    assert oracle["counts"] == {
        "slots": len(expected),  # de-duplicated: categories overlap
        "not_attribute": len(NOT_ATTRIBUTE),
        "named_attribute": len(NAMED_ATTRIBUTE),
        "update_chains": len(UPDATE_CHAINS),
        "collisions": len(COLLISION_PAIRS),
    }


def test_every_pin_points_at_a_corpus_entry():
    assert FALSE_EXTRACTION_PINS <= {case.text for case in NOT_ATTRIBUTE}
    assert WRONG_KEY_PINS <= {case.text for case in NAMED_ATTRIBUTE}
    assert UNCOVERED_UPDATE_PINS <= {chain.attribute for chain in UPDATE_CHAINS}
    assert COLLIDING_SLOT_PINS <= {pair.shared_slot for pair in COLLISION_PAIRS}
