"""Unit tests for the igm library — the write layer for RAG.

These exercise the mechanism without any LLM: gating, slot supersede,
forgetting, and slot-aware retrieval.
"""

import pytest

from igm import (
    CallableEmbedder,
    HashEmbedder,
    HeuristicScorer,
    LearnedScorer,
    Memory,
    MemoryStore,
    WriteGate,
    extract_slot,
)


# ---------------------------------------------------------------------------
# slot extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("我的住址是北京。", "住址"),
    ("更新一下，我的住址现在是深圳了。", "住址"),
    ("我现在的住址是什么？", "住址"),
    ("我现在的最喜欢的食物是什么？", "最喜欢的食物"),
    ("我的宠物是什么？", "宠物"),
    ("user: 我想告诉你，我的最喜欢的颜色是蓝色。", "最喜欢的颜色"),
])
def test_extract_slot_matches_fact_and_query(text, expected):
    assert extract_slot(text) == expected


@pytest.mark.parametrize("text", [
    "今天天气不错。",
    "这几天在整理房间。",
    "下午要开个会。",
])
def test_extract_slot_none_for_non_facts(text):
    assert extract_slot(text) is None


# ---------------------------------------------------------------------------
# write gate
# ---------------------------------------------------------------------------

def test_gate_writes_facts_rejects_filler():
    gate = WriteGate(HeuristicScorer(), threshold=0.5)
    ok_fact, s_fact = gate.should_write("我的职业是软件工程师。")
    ok_filler, s_filler = gate.should_write("今天天气不错。")
    assert ok_fact and not ok_filler
    assert s_fact > s_filler


def test_learned_scorer_prefers_facts():
    # Trained weights from memory_arch/train_scorer.py (held-out ~0.956).
    scorer = LearnedScorer(weights=[-2.279, 3.589, -4.234, -1.976, -0.099, 2.610])
    assert scorer.score("我的爱好是摄影。") > 0.5
    assert scorer.score("最近在看一本书，挺有意思。") < 0.5
    assert scorer.score("我的爱好是什么？") < 0.5   # questions are not memories


# ---------------------------------------------------------------------------
# slot supersede (the headline behaviour)
# ---------------------------------------------------------------------------

def test_slot_supersede_replaces_old_value():
    store = MemoryStore(HashEmbedder())
    store.write("user: 我的住址是北京。", slot="住址")
    store.write("user: 我的宠物是一只猫。", slot="宠物")
    store.write("user: 我的住址现在是深圳了。", slot="住址")
    assert len(store) == 2
    addr = [it for it in store.items if it.slot == "住址"]
    assert len(addr) == 1 and "深圳" in addr[0].text and "北京" not in addr[0].text


def test_memory_add_supersedes_via_gate():
    mem = Memory(embedder=HashEmbedder())
    mem.add("我的住址是北京。")
    mem.add("更新一下，我的住址现在是深圳了。")
    results = mem.query_texts("我现在的住址是什么？")
    assert any("深圳" in r for r in results)
    assert not any("北京" in r for r in results)


# ---------------------------------------------------------------------------
# forgetting lifecycle
# ---------------------------------------------------------------------------

def test_prune_drops_decayed_memories():
    store = MemoryStore(HashEmbedder(), decay=1.0)
    store.write("user: 我的住址是北京。", slot="住址", importance=0.5)
    for _ in range(50):
        store.tick()
    removed = store.prune(threshold=0.1)
    assert removed == 1 and len(store) == 0


def test_reuse_protects_from_decay():
    store = MemoryStore(HashEmbedder(), decay=0.5)
    item = store.write("user: 我的宠物是一只猫。", slot="宠物", importance=0.5)
    for _ in range(20):
        store.tick()
        store.mark_used(item)   # reused repeatedly -> stays consolidated
    assert len(store) == 1
    store.prune(threshold=0.1)
    assert len(store) == 1


# ---------------------------------------------------------------------------
# pluggable embedder
# ---------------------------------------------------------------------------

def test_callable_embedder():
    mem = Memory(embedder=CallableEmbedder(lambda t: [float(len(t)), 1.0]))
    mem.add("我的职业是软件工程师。")
    assert len(mem) == 1


def test_selectivity_stat():
    mem = Memory(embedder=HashEmbedder())
    mem.add("我的职业是软件工程师。")
    mem.add("今天天气不错。")
    s = mem.stats()
    assert s["considered"] == 2
    assert 0.0 < s["selectivity"] <= 1.0
