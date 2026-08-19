"""Minimal IGM usage example — no LLM, no external deps.

Run from the repo root: python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igm import Memory

mem = Memory()  # zero-dependency: hash embedder + heuristic gate

# A user conversation stream.  IGM decides what is worth storing.
stream = [
    "今天天气不错，适合出去散步。",          # filler — filtered out
    "我的住址是北京。",                        # fact — stored (slot: 住址)
    "最近在看一本书，挺有意思的。",            # filler — filtered out
    "我的职业是软件工程师。",                  # fact — stored (slot: 职业)
    "更新一下，我的住址现在是深圳了。",        # update — supersedes 北京
]

for text in stream:
    kept = mem.add(text)
    print(f"{'存' if kept else '滤'}  {text}")

print()
print("库存记忆:")
for it in mem.store.items:
    print(f"  [{it.slot or '-':6}] {it.text}")

print()
print("查询: 我现在的住址是什么？")
print(" ->", mem.query_texts("我现在的住址是什么？"))

print()
print("统计:", mem.stats())
