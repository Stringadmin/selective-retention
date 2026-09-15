"""Controlled update-memory experiment with fair versioning ablations.

The benchmark isolates one narrow question: when a user's attribute changes,
what is gained and lost by compacting its old versions at write time?  It keeps
a deliberately strong version-aware full-store baseline.  Therefore a 100%
score for both that baseline and IGM is an expected, informative outcome -- it
means explicit update semantics, rather than IGM branding, solves the current
value question.

``--random-trials`` turns the five readable hand-authored chains into many
seeded held-out combinations.  This improves stability checks, but it is
still synthetic data; it is not a substitute for LongMemEval or real logs.

Run (the answer step is rule-based to isolate the memory layer):
    python -m memory_arch.run_decisive
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from . import Conversation, Question, Turn
from .scorer import ImportanceScorer
from . import Embedder, MemoryItem, MemoryStore, cosine, IGMMethod


# Attribute value pools: latest value is semantically CLOSE to an older one,
# so embedding similarity cannot separate "current" from "stale".
UPDATE_CHAINS = {
    "住址": ["北京", "上海", "广州", "深圳", "珠海"],   # all cities, highly confusable
    "工作": ["教师", "助教", "讲师", "教授", "医生"],     # 教师/助教/讲师/教授 very close
    "最喜欢的颜色": ["蓝色", "青色", "蓝绿色", "绿色", "紫色"],  # shades of blue/green
    "座驾": ["自行车", "电动车", "摩托车", "汽车", "越野车"],
    "最喜欢的食物": ["火锅", "麻辣烫", "串串香", "冒菜", "寿司"],
}

# Larger pools support long, randomized update chains.  The first five values
# deliberately remain the hand-authored chains above, so the default report is
# stable and easy to inspect.
UPDATE_VALUE_POOLS = {
    "住址": ["北京", "上海", "广州", "深圳", "珠海", "苏州", "成都", "厦门", "武汉", "南京",
             "天津", "重庆", "杭州", "宁波", "青岛", "大连", "福州", "昆明", "西安", "长沙"],
    "工作": ["教师", "助教", "讲师", "教授", "医生", "工程师", "设计师", "产品经理", "会计",
             "律师", "研究员", "记者", "摄影师", "厨师", "护士", "翻译", "咨询师", "销售", "编辑", "分析师"],
    "最喜欢的颜色": ["蓝色", "青色", "蓝绿色", "绿色", "紫色", "红色", "橙色", "黄色", "粉色",
                 "黑色", "白色", "灰色", "银色", "金色", "棕色", "米色", "靛蓝色", "湖蓝色", "墨绿色", "酒红色"],
    "座驾": ["自行车", "电动车", "摩托车", "汽车", "越野车", "轿车", "旅行车", "皮卡", "面包车",
           "公交车", "地铁", "高铁", "滑板车", "三轮车", "房车", "跑车", "敞篷车", "SUV", "MPV", "出租车"],
    "最喜欢的食物": ["火锅", "麻辣烫", "串串香", "冒菜", "寿司", "拉面", "披萨", "汉堡", "饺子",
             "烤鸭", "炒饭", "面条", "牛排", "沙拉", "咖喱饭", "煎饼", "米线", "螺蛳粉", "烤鱼", "蛋糕"],
}

FILLER = [
    "今天天气不错。", "最近有点忙。", "周末想出去走走。", "这本书挺有意思。",
    "晚上打算早点休息。", "最近在学习新东西。", "楼下新开了家店。",
]

# Durable, non-conflicting facts used to test IGM under a wider resident-memory
# load. They are injected into both methods; none shares a slot with the five
# update chains or with the three fixed facts in ``make_chain_conv``.
BACKGROUND_FACTS = [
    ("常用语言", "Python"), ("常用编辑器", "VS Code"),
    ("操作系统", "Linux"), ("代码仓库", "GitHub"),
    ("显示器", "双屏"), ("键盘", "机械键盘"),
    ("鼠标", "无线鼠标"), ("浏览器", "Firefox"),
    ("时区", "东八区"), ("办公城市", "杭州"),
    ("学历", "本科"), ("毕业年份", "2020"),
    ("常用终端", "PowerShell"), ("包管理器", "npm"),
    ("编程语言", "TypeScript"), ("数据库", "PostgreSQL"),
    ("云平台", "阿里云"), ("部署方式", "Docker"),
    ("测试框架", "pytest"), ("版本控制", "Git"),
    ("代码风格", "Prettier"), ("常用模型", "Qwen"),
    ("工作节奏", "早十晚六"), ("联系渠道", "邮件"),
]


class CachedEmbedder:
    """Share deterministic embeddings across stores and repeated queries.

    It preserves the requested backend while making a hundreds-of-trials BGE
    run practical.  The cache changes no vectors and is deliberately scoped to
    one ``run`` call.
    """

    def __init__(self, delegate: Embedder):
        self.delegate = delegate
        self.cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.delegate.embed(text)
        return self.cache[text]


def make_trial_specs(rng: random.Random, random_trials: int,
                     updates_per_chain: int) -> list[tuple[str, str, list[str]]]:
    """Return ``(id, attribute, update_chain)`` cases for a benchmark run."""
    if not 2 <= updates_per_chain <= min(len(v) for v in UPDATE_VALUE_POOLS.values()):
        raise ValueError("updates_per_chain must be between 2 and 20")
    if random_trials < 0:
        raise ValueError("random_trials must be non-negative")
    if random_trials == 0:
        if updates_per_chain == len(next(iter(UPDATE_CHAINS.values()))):
            return [(f"chain-{attr}", attr, chain) for attr, chain in UPDATE_CHAINS.items()]
        return [
            (f"chain-{attr}-{updates_per_chain}", attr,
             UPDATE_VALUE_POOLS[attr][:updates_per_chain])
            for attr in UPDATE_CHAINS
        ]

    attrs = list(UPDATE_VALUE_POOLS)
    return [
        (f"random-{trial:04d}-{attr}", attr,
         rng.sample(UPDATE_VALUE_POOLS[attr], updates_per_chain))
        for trial in range(random_trials)
        for attr in [rng.choice(attrs)]
    ]


def make_chain_conv(rng, conv_id, attr, chain, n_filler=4, background_facts=0):
    """One conversation where `attr` is updated through `chain` (oldest->newest),
    each update in its own session, interleaved with filler and other facts.

    ``background_facts`` adds durable facts with distinct slots to both stores.
    It is a resident-memory-load control, not an IGM-only advantage.
    """
    if not 0 <= background_facts <= len(BACKGROUND_FACTS):
        raise ValueError(f"background_facts must be between 0 and {len(BACKGROUND_FACTS)}")
    turns: list[Turn] = []
    # Plant a few unrelated durable facts to add retrieval noise.
    other_facts = [("宠物", "一只猫"), ("家乡", "杭州"), ("爱好", "摄影")]
    for k, v in other_facts:
        turns.append(Turn(role="user", content=f"我的{k}是{v}。", session=0, timestamp=0.0))
    for k, v in BACKGROUND_FACTS[:background_facts]:
        turns.append(Turn(role="user", content=f"我的{k}是{v}。", session=0, timestamp=0.0))
    t = 10.0
    for session, value in enumerate(chain):
        for _ in range(n_filler):
            turns.append(Turn(role="user", content=rng.choice(FILLER),
                              session=session, timestamp=t)); t += 1
        if session == 0:
            text = f"我的{attr}是{value}。"
        else:
            text = f"更新一下，我的{attr}现在是{value}了。"
        turns.append(Turn(role="user", content=text, session=session, timestamp=t)); t += 1
    current = chain[-1]
    q = Question(id=f"{conv_id}-q", category="knowledge_update_chain",
                 question=f"我现在的{attr}是什么？", answer=current)
    return Conversation(conv_id=conv_id, turns=turns, questions=[q])


# --- Retrieval-then-answer (rule-based) -------------------------------------
# We deliberately do NOT call an LLM to answer.  The thing under test is the
# MEMORY layer's ability to surface the correct current value.  A rule-based
# reader isolates that from LLM reasoning/thinking noise.


def answer_full_storage(store: MemoryStore, attr: str, top_k: int = 3) -> str:
    """Full-storage baseline: retrieve top-k by embedding, return the value in
    the single best-matching memory (naive reader: trust the top hit)."""
    items = store.retrieve(f"我现在的{attr}是什么？", top_k=top_k)
    # Return the value in the top-1 retrieved memory (naive RAG trusts rank-1).
    return _value_of(items[0].text, attr) if items else ""


def answer_full_storage_recency(store: MemoryStore, attr: str, top_k: int = 3) -> str:
    """A stronger full-storage variant: retrieve top-k, then pick the one with
    the newest timestamp.  (A fair chance: full storage CAN sort by time.)"""
    items = store.retrieve(f"我现在的{attr}是什么？", top_k=top_k)
    if not items:
        return ""
    newest = max(items, key=lambda it: it.created_at)
    return _value_of(newest.text, attr)


def answer_full_storage_slot_recency(store: MemoryStore, attr: str) -> str:
    """Strong full-storage baseline with explicit attribute metadata.

    It keeps every historical record, filters to the requested attribute, then
    returns the newest value. This shows that IGM's accuracy gain is not magic:
    a full store with explicit update-aware filtering can match it, at the cost
    of retaining stale versions.
    """
    slot = IGMMethod._extract_slot(f"我现在的{attr}是什么？")
    matches = [item for item in store.items if IGMMethod._extract_slot(item.text) == slot]
    if not matches:
        return ""
    newest = max(matches, key=lambda item: item.created_at)
    return _value_of(newest.text, attr)


def answer_full_storage_slot_history(store: MemoryStore, attr: str) -> str:
    """Return the immediately preceding value for a version-history query.

    IGM intentionally cannot answer this query after superseding a slot.  It
    is reported to make that product tradeoff visible rather than hiding it
    behind a current-value-only score.
    """
    # The extractor intentionally supports current-value phrasing only, so a
    # history query routes from its already-known attribute name.
    matches = [item for item in store.items if IGMMethod._extract_slot(item.text) == attr]
    if len(matches) < 2:
        return ""
    previous = sorted(matches, key=lambda item: item.created_at)[-2]
    return _value_of(previous.text, attr)


def write_full_storage(store: MemoryStore, text: str, capacity: int | None = None) -> None:
    """Append a raw record and optionally enforce a FIFO item budget."""
    store.write(text)
    if capacity is not None and len(store.items) > capacity:
        del store.items[:-capacity]


def append_fact(store: MemoryStore, text: str, importance: float, slot: str | None,
                capacity: int | None = None) -> None:
    """Append a gated fact while retaining its slot/version metadata.

    ``MemoryStore.write(slot=...)`` implements supersede, so the gate-only
    ablation appends ``MemoryItem`` directly.  It has the identical extractor
    and importance decision as IGM; only version retention differs.
    """
    store.items.append(MemoryItem(
        text=text,
        embedding=store.embedder.embed(text),
        importance=importance,
        created_at=store.clock,
        last_used=store.clock,
        slot=slot,
    ))
    if capacity is not None and len(store.items) > capacity:
        del store.items[:-capacity]


def importance_of(store: MemoryStore, text: str, scorer: ImportanceScorer | None) -> float:
    """The IGM write gate, factored out so all ablations share it exactly."""
    if scorer is not None:
        emb = store.embedder.embed(text)
        max_sim = max([cosine(emb, it.embedding) for it in store.current()[-200:]] or [0.0])
        return scorer.score(text, max_sim_to_store=max_sim)
    marker_hits = sum(1 for marker in IGMMethod._FACT_MARKERS if marker in text)
    return min(marker_hits / 3.0, 1.0)


def answer_igm(store: MemoryStore, attr: str) -> str:
    """IGM: retrieve with slot routing, read the (single) current value."""
    slot = IGMMethod._extract_slot(f"我现在的{attr}是什么？")
    items = store.retrieve(f"我现在的{attr}是什么？", top_k=3, slot=slot)
    for it in items:
        if it.slot == attr:
            return _value_of(it.text, attr)
    return _value_of(items[0].text, attr) if items else ""


def answer_igm_history(store: MemoryStore, attr: str) -> str:
    """IGM stores current state, so an overwritten slot has no prior version."""
    del store, attr
    return ""


def answer_versioned_igm(store: MemoryStore, attr: str) -> str:
    """Current query: route straight to the current-state projection."""
    events = store.current(attr)
    return _value_of(events[0].text, attr) if events else ""


def answer_versioned_igm_history(store: MemoryStore, attr: str) -> str:
    """History query: read the archived predecessor of the current event."""
    event = store.previous(attr)
    return _value_of(event.text, attr) if event is not None else ""


def _value_of(text: str, attr: str) -> str:
    """Extract the value from a stored fact text ('...我的{attr}现在是{V}了。')."""
    for stop in ("现在是", "是"):
        if stop in text and attr in text:
            rest = text.split(attr, 1)[1]
            if stop in rest:
                v = rest.split(stop, 1)[1]
                for tail in ("了", "。", "，", ","):
                    v = v.split(tail)[0]
                return v.strip()
    return ""


def run(seed: int = 1337, embed_backend: str = "bge",
        embed_model: str | None = None, scorer_path: str | None = None,
        background_facts: int = 0, full_capacity: int | None = None,
        random_trials: int = 0, updates_per_chain: int = 5) -> dict:
    if full_capacity is not None and full_capacity <= 0:
        raise ValueError("full_capacity must be positive")
    rng = random.Random(seed)
    # The cache preserves vectors while avoiding duplicate BGE calls across
    # stores and repeated questions in the randomized workload.
    embedder = CachedEmbedder(Embedder(backend=embed_backend, model_path=embed_model))
    scorer = ImportanceScorer.load(scorer_path) if scorer_path else None

    results = {
        "full_top1": [],
        "full_recency": [],
        "full_slot_recency": [],
        "gate_top1": [],
        "gate_recency": [],
        "gate_slot_recency": [],
        "igm": [],
        "versioned_igm": [],
        "full_slot_history": [],
        "gate_slot_history": [],
        "igm_history": [],
        "versioned_igm_history": [],
    }
    if background_facts:
        results.update({
            "background_full_top1": [],
            "background_full_slot_recency": [],
            "background_gate_top1": [],
            "background_gate_slot_recency": [],
            "background_igm": [],
            "background_versioned_igm": [],
        })
    if full_capacity is not None:
        results.update({
            "full_fifo_top1": [],
            "full_fifo_recency": [],
            "full_fifo_slot_recency": [],
            "gate_fifo_top1": [],
            "gate_fifo_recency": [],
            "gate_fifo_slot_recency": [],
        })
        if background_facts:
            results.update({
                "background_full_fifo_top1": [],
                "background_full_fifo_slot_recency": [],
                "background_gate_fifo_top1": [],
                "background_gate_fifo_slot_recency": [],
            })
    details = []
    trial_specs = make_trial_specs(rng, random_trials, updates_per_chain)
    for trial, (conv_id, attr, chain) in enumerate(trial_specs):
        conv = make_chain_conv(rng, conv_id, attr, chain,
                               background_facts=background_facts)

        # Full storage writes each raw turn. Gate-only makes exactly IGM's
        # write decision but keeps every accepted version. Therefore any
        # difference between gate-only and IGM comes from supersede, not a
        # favorable change in extraction or salience filtering.
        full_store = MemoryStore(embedder)
        full_fifo_store = MemoryStore(embedder) if full_capacity is not None else None
        gate_store = MemoryStore(embedder)
        gate_fifo_store = MemoryStore(embedder) if full_capacity is not None else None
        # Destructive supersede: the archived predecessor is physically removed,
        # which is what makes this arm unable to answer history queries.
        igm_store = MemoryStore(embedder, supersede="delete")
        # Archiving supersede is igm's shipped default: same gate, same current
        # view, but the predecessor keeps a validity interval.
        versioned_igm_store = MemoryStore(embedder)
        full_writes = 0
        gate_writes = 0
        igm_writes = 0
        for t in conv.turns:
            text = f"{t.role}: {t.content}"
            full_store.clock += 1.0
            write_full_storage(full_store, text)
            full_writes += 1
            if full_fifo_store is not None:
                full_fifo_store.clock += 1.0
                write_full_storage(full_fifo_store, text, full_capacity)
            gate_store.clock += 1.0
            if gate_fifo_store is not None:
                gate_fifo_store.clock += 1.0
            igm_store.clock += 1.0
            versioned_igm_store.clock += 1.0
            # Score against IGM's own pre-write state, then reuse that exact
            # decision in gate-only.  This makes the ablation differ only in
            # whether same-slot versions are superseded.
            score = importance_of(igm_store, text, scorer)
            if score >= 0.5:
                slot = IGMMethod._extract_slot(text)
                append_fact(gate_store, text, importance=score, slot=slot)
                if gate_fifo_store is not None:
                    append_fact(gate_fifo_store, text, importance=score, slot=slot,
                                capacity=full_capacity)
                igm_store.write(text, importance=score, slot=slot)
                versioned_igm_store.write(text, importance=score, slot=slot)
                gate_writes += 1
                igm_writes += 1

        gold = conv.questions[0].answer
        history_gold = chain[-2]
        a_full = answer_full_storage(full_store, attr)
        a_rec = answer_full_storage_recency(full_store, attr)
        a_slot_rec = answer_full_storage_slot_recency(full_store, attr)
        a_gate = answer_full_storage(gate_store, attr)
        a_gate_rec = answer_full_storage_recency(gate_store, attr)
        a_gate_slot_rec = answer_full_storage_slot_recency(gate_store, attr)
        if full_fifo_store is not None:
            a_fifo = answer_full_storage(full_fifo_store, attr)
            a_fifo_rec = answer_full_storage_recency(full_fifo_store, attr)
            a_fifo_slot_rec = answer_full_storage_slot_recency(full_fifo_store, attr)
            a_gate_fifo = answer_full_storage(gate_fifo_store, attr)
            a_gate_fifo_rec = answer_full_storage_recency(gate_fifo_store, attr)
            a_gate_fifo_slot_rec = answer_full_storage_slot_recency(gate_fifo_store, attr)
        a_igm = answer_igm(igm_store, attr)
        a_versioned_igm = answer_versioned_igm(versioned_igm_store, attr)
        results["full_top1"].append(a_full == gold)
        results["full_recency"].append(a_rec == gold)
        results["full_slot_recency"].append(a_slot_rec == gold)
        results["gate_top1"].append(a_gate == gold)
        results["gate_recency"].append(a_gate_rec == gold)
        results["gate_slot_recency"].append(a_gate_slot_rec == gold)
        if full_fifo_store is not None:
            results["full_fifo_top1"].append(a_fifo == gold)
            results["full_fifo_recency"].append(a_fifo_rec == gold)
            results["full_fifo_slot_recency"].append(a_fifo_slot_rec == gold)
            results["gate_fifo_top1"].append(a_gate_fifo == gold)
            results["gate_fifo_recency"].append(a_gate_fifo_rec == gold)
            results["gate_fifo_slot_recency"].append(a_gate_fifo_slot_rec == gold)
        results["igm"].append(a_igm == gold)
        results["versioned_igm"].append(a_versioned_igm == gold)
        results["full_slot_history"].append(
            answer_full_storage_slot_history(full_store, attr) == history_gold)
        results["gate_slot_history"].append(
            answer_full_storage_slot_history(gate_store, attr) == history_gold)
        # A current-state store has deliberately discarded the preceding value.
        results["igm_history"].append(answer_igm_history(igm_store, attr) == history_gold)
        results["versioned_igm_history"].append(
            answer_versioned_igm_history(versioned_igm_store, attr) == history_gold)

        # A bounded store may improve a current-value answer simply by evicting
        # history. Check whether it also retains the durable resident facts.
        background_scores = {
            "full_top1": [], "full_slot_recency": [],
            "gate_top1": [], "gate_slot_recency": [], "igm": [], "versioned_igm": [],
        }
        if full_fifo_store is not None:
            background_scores.update({
                "full_fifo_top1": [], "full_fifo_slot_recency": [],
                "gate_fifo_top1": [], "gate_fifo_slot_recency": [],
            })
        for background_attr, background_value in BACKGROUND_FACTS[:background_facts]:
            background_scores["full_top1"].append(
                answer_full_storage(full_store, background_attr) == background_value)
            background_scores["full_slot_recency"].append(
                answer_full_storage_slot_recency(full_store, background_attr) == background_value)
            background_scores["gate_top1"].append(
                answer_full_storage(gate_store, background_attr) == background_value)
            background_scores["gate_slot_recency"].append(
                answer_full_storage_slot_recency(gate_store, background_attr) == background_value)
            background_scores["igm"].append(
                answer_igm(igm_store, background_attr) == background_value)
            background_scores["versioned_igm"].append(
                answer_versioned_igm(versioned_igm_store, background_attr) == background_value)
            if full_fifo_store is not None:
                background_scores["full_fifo_top1"].append(
                    answer_full_storage(full_fifo_store, background_attr) == background_value)
                background_scores["full_fifo_slot_recency"].append(
                    answer_full_storage_slot_recency(full_fifo_store, background_attr) == background_value)
                background_scores["gate_fifo_top1"].append(
                    answer_full_storage(gate_fifo_store, background_attr) == background_value)
                background_scores["gate_fifo_slot_recency"].append(
                    answer_full_storage_slot_recency(gate_fifo_store, background_attr) == background_value)
        if background_facts:
            results["background_full_top1"].extend(background_scores["full_top1"])
            results["background_full_slot_recency"].extend(background_scores["full_slot_recency"])
            results["background_gate_top1"].extend(background_scores["gate_top1"])
            results["background_gate_slot_recency"].extend(background_scores["gate_slot_recency"])
            results["background_igm"].extend(background_scores["igm"])
            results["background_versioned_igm"].extend(background_scores["versioned_igm"])
            if full_fifo_store is not None:
                results["background_full_fifo_top1"].extend(background_scores["full_fifo_top1"])
                results["background_full_fifo_slot_recency"].extend(background_scores["full_fifo_slot_recency"])
                results["background_gate_fifo_top1"].extend(background_scores["gate_fifo_top1"])
                results["background_gate_fifo_slot_recency"].extend(background_scores["gate_fifo_slot_recency"])
        detail = {
            "id": conv_id, "attr": attr, "chain": chain, "gold": gold,
            "full_top1": a_full, "full_recency": a_rec,
            "full_slot_recency": a_slot_rec,
            "gate_top1": a_gate, "gate_recency": a_gate_rec,
            "gate_slot_recency": a_gate_slot_rec,
            "igm": a_igm, "versioned_igm": a_versioned_igm,
            "history_gold": history_gold,
            "full_slot_history": answer_full_storage_slot_history(full_store, attr),
            "gate_slot_history": answer_full_storage_slot_history(gate_store, attr),
            "igm_history": answer_igm_history(igm_store, attr),
            "versioned_igm_history": answer_versioned_igm_history(versioned_igm_store, attr),
            "full_mem": len(full_store), "full_writes": full_writes,
            "gate_mem": len(gate_store), "gate_writes": gate_writes,
            "igm_mem": len(igm_store), "igm_writes": igm_writes,
            "versioned_igm_events": versioned_igm_store.event_count,
            "versioned_igm_active": versioned_igm_store.active_count,
            "versioned_igm_current_candidates": len(versioned_igm_store.current(attr)),
            "versioned_igm_history_candidates": len(versioned_igm_store.history(attr)),
        }
        if full_fifo_store is not None:
            detail.update({
                "full_fifo_top1": a_fifo,
                "full_fifo_recency": a_fifo_rec,
                "full_fifo_slot_recency": a_fifo_slot_rec,
                "full_fifo_mem": len(full_fifo_store),
                "gate_fifo_top1": a_gate_fifo,
                "gate_fifo_recency": a_gate_fifo_rec,
                "gate_fifo_slot_recency": a_gate_fifo_slot_rec,
                "gate_fifo_mem": len(gate_fifo_store),
            })
        if background_facts:
            detail["background_accuracy"] = {
                name: sum(scores) / len(scores)
                for name, scores in background_scores.items()
            }
        details.append(detail)
        if len(trial_specs) <= 10 or trial < 10 or trial == len(trial_specs) - 1:
            print(f"[{attr:10s}] gold={gold} | full_top1={a_full} gate_top1={a_gate} "
                  f"gate_slot_recency={a_gate_slot_rec} igm={a_igm} vigm={a_versioned_igm} | "
                  f"mem full={len(full_store)} gate={len(gate_store)} igm={len(igm_store)} "
                  f"vigm events={versioned_igm_store.event_count}/active={versioned_igm_store.active_count}"
                  + (f" full_fifo={len(full_fifo_store)} gate_fifo={len(gate_fifo_store)}"
                     if full_fifo_store is not None else ""), flush=True)
        elif trial == 10:
            print("... individual trial output suppressed ...", flush=True)

    summary = {k: sum(v) / len(v) for k, v in results.items()}
    metric_stats = {name: _metric_stats(values) for name, values in results.items()}
    resources = {
        "full_storage": {
            "mean_writes": sum(d["full_writes"] for d in details) / len(details),
            "mean_resident_items": sum(d["full_mem"] for d in details) / len(details),
        },
        "gate_only": {
            "mean_writes": sum(d["gate_writes"] for d in details) / len(details),
            "mean_resident_items": sum(d["gate_mem"] for d in details) / len(details),
        },
        "igm": {
            "mean_writes": sum(d["igm_writes"] for d in details) / len(details),
            "mean_resident_items": sum(d["igm_mem"] for d in details) / len(details),
        },
        "versioned_igm": {
            "mean_writes": sum(d["igm_writes"] for d in details) / len(details),
            "mean_event_records": sum(d["versioned_igm_events"] for d in details) / len(details),
            "mean_active_projection_items": sum(
                d["versioned_igm_active"] for d in details) / len(details),
            "mean_current_query_candidates": sum(
                d["versioned_igm_current_candidates"] for d in details) / len(details),
            "mean_history_query_candidates": sum(
                d["versioned_igm_history_candidates"] for d in details) / len(details),
        },
    }
    if full_capacity is not None:
        resources["full_fifo"] = {
            "mean_resident_items": sum(d["full_fifo_mem"] for d in details) / len(details),
        }
        resources["gate_fifo"] = {
            "mean_resident_items": sum(d["gate_fifo_mem"] for d in details) / len(details),
        }
    return {"summary": summary, "metric_stats": metric_stats,
            "resources": resources, "details": details,
            "config": {
                "embed_backend": embed_backend,
                "scorer": bool(scorer),
                "seed": seed,
                "background_facts": background_facts,
                "full_capacity": full_capacity,
                "random_trials": random_trials,
                "updates_per_chain": updates_per_chain,
                "embedding_cache_entries": len(embedder.cache),
            }}


def _metric_stats(values: list[bool]) -> dict[str, float | int | list[float]]:
    """Exact count plus Wilson 95% interval, for descriptive reporting only."""
    n = len(values)
    successes = sum(values)
    rate = successes / n
    z = 1.96
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return {
        "successes": successes,
        "n": n,
        "rate": rate,
        "wilson_95": [max(0.0, centre - margin), min(1.0, centre + margin)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--embed-backend", choices=["hash", "bge"], default="bge")
    ap.add_argument("--embed-model", default=("/home/omnichat/.cache/modelscope/models/"
                                              "BAAI--bge-small-zh-v1.5/snapshots/master"))
    ap.add_argument("--scorer", default=str(Path(__file__).parent / "scorer_weights.json"))
    ap.add_argument("--background-facts", type=int, default=0,
                    help="Number of shared durable background facts (0-%(choices)s).",
                    choices=range(len(BACKGROUND_FACTS) + 1))
    ap.add_argument("--full-capacity", type=int,
                    help="Optional FIFO item cap for an additional full-storage baseline.")
    ap.add_argument("--random-trials", type=int, default=0,
                    help="Seeded randomized held-out chains; 0 keeps the five readable cases.")
    ap.add_argument("--updates-per-chain", type=int, default=5,
                    help="Number of distinct updates per chain (2-20).")
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                    / "reports" / "memory-decisive.json")
    args = ap.parse_args()

    scorer_path = args.scorer if Path(args.scorer).is_file() else None
    out = run(seed=args.seed, embed_backend=args.embed_backend,
              embed_model=args.embed_model if args.embed_backend == "bge" else None,
              scorer_path=scorer_path,
              background_facts=args.background_facts,
              full_capacity=args.full_capacity,
              random_trials=args.random_trials,
              updates_per_chain=args.updates_per_chain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (accuracy) ===")
    for k, v in out["summary"].items():
        print(f"  {k:14s} {v:.2f}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
