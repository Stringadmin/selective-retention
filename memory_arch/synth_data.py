"""Synthetic multi-session memory test-set generator.

Builds controlled conversations in the LongMemEval spirit so the memory
pipeline can be validated offline (real LongMemEval sources are network
blocked).  Covers the categories that matter most for the IGM hypotheses:

  - single_session_recall : a fact stated once, asked later
  - knowledge_update      : a fact stated, then CHANGED later (tests forgetting)
  - temporal_reasoning    : facts with time references, asked about recency

Each conversation embeds facts in a stream of filler turns, then asks
questions whose correct answer requires the RIGHT memory behavior (not just
the latest text).  This is the analog of the FIP synthetic-task philosophy:
fully controlled, reproducible, targeted at the mechanism under test.
"""

from __future__ import annotations

import random

from . import Conversation, Question, Turn

FILLER = [
    "今天天气不错，适合出去散步。",
    "最近在看一本关于历史的书，挺有意思的。",
    "周末打算去超市买点东西。",
    "这道菜的做法我研究了一下，还挺简单的。",
    "工作上最近项目比较多，有点忙。",
    "昨晚看了一部电影，剧情一般。",
    "我打算开始学习一门新技能。",
    "最近睡眠质量不太好，总是做梦。",
    "楼下新开了一家咖啡店，改天去试试。",
    "这几天在整理房间，发现了不少旧东西。",
]


def _filler_turns(rng: random.Random, n: int, session: int, t0: float) -> list[Turn]:
    turns = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = rng.choice(FILLER)
        turns.append(Turn(role=role, content=content, session=session, timestamp=t0 + i))
    return turns


def make_recall_conv(rng: random.Random, conv_id: str, fact_key: str, fact_val: str) -> Conversation:
    """A single fact stated in session 0, asked at the end."""
    t0 = 0.0
    turns = _filler_turns(rng, 6, session=0, t0=t0)
    # Insert the fact.
    turns.insert(2, Turn(role="user", content=f"我想告诉你，我的{fact_key}是{fact_val}。",
                         session=0, timestamp=t0 + 1.5))
    q = Question(id=f"{conv_id}-q1", category="single_session_recall",
                 question=f"我的{fact_key}是什么？", answer=fact_val)
    return Conversation(conv_id=conv_id, turns=turns, questions=[q])


def make_update_conv(rng: random.Random, conv_id: str, fact_key: str,
                     old_val: str, new_val: str) -> Conversation:
    """Fact stated in session 0, UPDATED in session 1.  Correct answer = new_val.
    A memory system that never forgets the old value will answer old_val."""
    turns: list[Turn] = []
    # Session 0: old value.
    turns += _filler_turns(rng, 4, session=0, t0=0.0)
    turns.insert(1, Turn(role="user", content=f"我的{fact_key}是{old_val}。",
                         session=0, timestamp=0.5))
    # Session 1: updated value.
    base = 100.0
    turns += _filler_turns(rng, 4, session=1, t0=base)
    turns.insert(len(turns) - 2, Turn(
        role="user", content=f"更新一下，我的{fact_key}现在是{new_val}了。",
        session=1, timestamp=base + 1.5))
    q = Question(id=f"{conv_id}-q1", category="knowledge_update",
                 question=f"我现在的{fact_key}是什么？", answer=new_val)
    return Conversation(conv_id=conv_id, turns=turns, questions=[q])


def make_temporal_conv(rng: random.Random, conv_id: str) -> Conversation:
    """Two events with explicit times; ask which is more recent."""
    turns: list[Turn] = []
    turns.append(Turn(role="user", content="我三月份去了北京出差。", session=0, timestamp=0.0))
    turns += _filler_turns(rng, 3, session=0, t0=1.0)
    turns.append(Turn(role="user", content="我七月份去了上海旅游。", session=1, timestamp=100.0))
    turns += _filler_turns(rng, 3, session=1, t0=101.0)
    q = Question(id=f"{conv_id}-q1", category="temporal_reasoning",
                 question="我最近一次出行是去哪里？", answer="上海")
    return Conversation(conv_id=conv_id, turns=turns, questions=[q])


def make_distractor_update_conv(rng: random.Random, conv_id: str, fact_key: str,
                                old_val: str, new_val: str,
                                other_facts: list[tuple[str, str]]) -> Conversation:
    """Knowledge-update conversation embedded among MANY other facts.

    The distractor facts share surface vocabulary with the target, so a
    full-storage method retrieves noisy/related memories and can answer with
    a stale or wrong value, while a slot-supersede method isolates the latest
    value of the exact attribute.  This is where selective writing should win.
    """
    turns: list[Turn] = []
    # Session 0: old value + several distractor facts.
    turns += _filler_turns(rng, 3, session=0, t0=0.0)
    turns.insert(1, Turn(role="user", content=f"我的{fact_key}是{old_val}。",
                         session=0, timestamp=0.5))
    for i, (k, v) in enumerate(other_facts):
        turns.append(Turn(role="user", content=f"另外，我的{k}是{v}。",
                          session=0, timestamp=2.0 + i))
        turns += _filler_turns(rng, 2, session=0, t0=3.0 + i)
    # Session 1: update the target fact among more filler.
    base = 100.0
    turns += _filler_turns(rng, 3, session=1, t0=base)
    turns.insert(len(turns) - 1, Turn(
        role="user", content=f"更新一下，我的{fact_key}现在是{new_val}了。",
        session=1, timestamp=base + 1.5))
    q = Question(id=f"{conv_id}-q1", category="knowledge_update",
                 question=f"我现在的{fact_key}是什么？", answer=new_val)
    return Conversation(conv_id=conv_id, turns=turns, questions=[q])


# Larger, more confusable fact pools for the hard dataset.
_HARD_FACTS = [
    ("最喜欢的颜色", ["蓝色", "红色", "绿色", "紫色", "青色"]),
    ("宠物", ["一只猫", "一只狗", "两只猫", "一只鹦鹉", "一只仓鼠"]),
    ("家乡", ["杭州", "南京", "苏州", "无锡", "宁波"]),
    ("职业", ["软件工程师", "硬件工程师", "测试工程师", "架构师", "产品经理"]),
    ("爱好", ["摄影", "摄像", "绘画", "书法", "篆刻"]),
    ("最喜欢的食物", ["火锅", "寿司", "刺身", "烤肉", "烧烤"]),
    ("住址", ["北京", "上海", "深圳", "广州", "成都"]),
    ("座驾", ["自行车", "电动车", "摩托车", "汽车"]),
]


def build_dataset(seed: int = 1337, hard: bool = False,
                  n_recall: int = 5, n_update: int = 3, n_temporal: int = 3) -> list[Conversation]:
    rng = random.Random(seed)
    convs: list[Conversation] = []
    if not hard:
        facts = [("最喜欢的颜色", "蓝色"), ("宠物", "一只猫"), ("家乡", "杭州"),
                 ("职业", "软件工程师"), ("爱好", "摄影")]
        for i, (k, v) in enumerate(facts):
            convs.append(make_recall_conv(rng, f"recall-{i}", k, v))
        updates = [("住址", "北京", "深圳"), ("工作", "教师", "医生"),
                   ("最喜欢的食物", "火锅", "寿司")]
        for i, (k, old, new) in enumerate(updates):
            convs.append(make_update_conv(rng, f"update-{i}", k, old, new))
        for i in range(3):
            convs.append(make_temporal_conv(rng, f"temporal-{i}"))
        return convs

    # Hard mode: distractor facts create retrieval noise; selective writing
    # and slot-supersede become decisive.
    for i in range(n_recall):
        k, vals = _HARD_FACTS[i % len(_HARD_FACTS)]
        v = rng.choice(vals)
        convs.append(make_recall_conv(rng, f"recall-{i}", k, v))
    for i in range(n_update):
        k, vals = _HARD_FACTS[(i + 2) % len(_HARD_FACTS)]
        old, new = rng.sample(vals, 2)
        # Distractors: other attributes with confusable values.
        others = []
        for j in range(3):
            ok, ovals = _HARD_FACTS[(i + j + 4) % len(_HARD_FACTS)]
            others.append((ok, rng.choice(ovals)))
        convs.append(make_distractor_update_conv(rng, f"update-{i}", k, old, new, others))
    for i in range(n_temporal):
        convs.append(make_temporal_conv(rng, f"temporal-{i}"))
    return convs


if __name__ == "__main__":
    ds = build_dataset()
    cats = {}
    for c in ds:
        for q in c.questions:
            cats[q.category] = cats.get(q.category, 0) + 1
    print(f"conversations: {len(ds)}, questions by category: {cats}")
    for c in ds[:2]:
        print(f"--- {c.conv_id} ({len(c.turns)} turns) ---")
        for t in c.turns[:3]:
            print(f"  [{t.role}] {t.content}")
        for q in c.questions:
            print(f"  Q({q.category}): {q.question} -> {q.answer}")
