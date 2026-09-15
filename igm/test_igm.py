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
    ("我之前的住址是什么？", "住址"),
    ("我上一次的常用语言是什么", "常用语言"),
    ("我上次的工位是什么？", "工位"),
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

def test_slot_supersede_archives_the_predecessor():
    store = MemoryStore(HashEmbedder())
    store.write("user: 我的住址是北京。", slot="住址")
    store.write("user: 我的宠物是一只猫。", slot="宠物")
    store.write("user: 我的住址现在是深圳了。", slot="住址")
    assert len(store) == 2
    # The current view carries only the new value...
    addr = store.current("住址")
    assert len(addr) == 1 and "深圳" in addr[0].text and "北京" not in addr[0].text
    # ...but the superseded value was closed, not destroyed.
    assert [it.text for it in store.history("住址")] == [
        "user: 我的住址是北京。",
        "user: 我的住址现在是深圳了。",
    ]
    assert store.event_count == 3 and store.active_count == 2


def test_closed_event_keeps_its_validity_interval():
    store = MemoryStore(HashEmbedder())
    store.clock = 1.0
    first = store.write("user: 我的住址是北京。", slot="住址")
    store.clock = 2.0
    second = store.write("user: 我的住址现在是深圳了。", slot="住址")
    assert first.valid_to == 2.0 and not first.is_current
    assert second.valid_to is None and second.supersedes == first.event_id
    assert store.previous("住址") is first


def test_retrieve_hides_archived_versions_by_default():
    store = MemoryStore(HashEmbedder())
    store.write("user: 我的住址是北京。", slot="住址")
    store.write("user: 我的住址现在是深圳了。", slot="住址")
    assert len(store.retrieve("我现在的住址是什么？", top_k=5, slot="住址")) == 1
    assert len(store.retrieve("我现在的住址是什么？", top_k=5, include_history=True)) == 2


def test_delete_supersede_stays_destructive():
    store = MemoryStore(HashEmbedder(), supersede="delete")
    store.write("user: 我的住址是北京。", slot="住址")
    store.write("user: 我的住址现在是深圳了。", slot="住址")
    assert store.event_count == 1 and store.history("住址")[0].valid_to is None


def test_unknown_supersede_policy_is_rejected():
    with pytest.raises(ValueError):
        MemoryStore(HashEmbedder(), supersede="merge")


def test_memory_add_supersedes_via_gate():
    mem = Memory(embedder=HashEmbedder())
    mem.add("我的住址是北京。")
    mem.add("更新一下，我的住址现在是深圳了。")
    results = mem.query_texts("我现在的住址是什么？")
    assert any("深圳" in r for r in results)
    assert not any("北京" in r for r in results)


def test_memory_history_answers_the_previous_value():
    mem = Memory(embedder=HashEmbedder())
    mem.add("我的住址是北京。")
    mem.add("更新一下，我的住址现在是深圳了。")
    assert len(mem) == 1                      # current projection
    assert mem.stats()["archived"] == 1       # cost of keeping the version
    assert "北京" in mem.previous("住址").text


def test_repeated_updates_do_not_gate_out_the_third_value():
    # Archived predecessors must not count as "already known" when scoring the
    # next update, or a chain of updates would stop after the second one.
    mem = Memory(embedder=HashEmbedder())
    mem.add("我的住址是北京。")
    mem.add("更新一下，我的住址现在是深圳了。")
    mem.add("再改一次，我的住址现在是杭州了。")
    assert "杭州" in mem.store.current("住址")[0].text
    assert len(mem.store.history("住址")) == 3


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
