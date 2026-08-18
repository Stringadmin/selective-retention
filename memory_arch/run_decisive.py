"""Decisive experiment: IGM slot-supersede vs full-storage on repeated updates.

The one scenario where a full-storage memory (naive RAG over a strong
embedder) MUST fail: an attribute is updated many times, and the question
asks for the CURRENT value.  Full storage keeps every historical value; the
latest and some older values are semantically near-identical, so embedding
retrieval cannot reliably pick the newest — it returns a stale value.

IGM's slot supersede keeps only the latest value per attribute, so it is
correct by construction.

This is the "headshot" experiment for the write-up: a baseline the reader
cannot dismiss, on a task (knowledge update / contradiction resolution) that
the 2026 memory literature explicitly flags as the hardest open problem.

Run (any Python; the answer step is rule-based to avoid LLM noise — this
isolates the MEMORY mechanism, which is the thing under test):
    python -m memory_arch.run_decisive
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import Conversation, Question, Turn
from .scorer import ImportanceScorer
from . import Embedder, MemoryStore, cosine, IGMMethod


# Attribute value pools: latest value is semantically CLOSE to an older one,
# so embedding similarity cannot separate "current" from "stale".
UPDATE_CHAINS = {
    "住址": ["北京", "上海", "广州", "深圳", "珠海"],   # all cities, highly confusable
    "工作": ["教师", "助教", "讲师", "教授", "医生"],     # 教师/助教/讲师/教授 very close
    "最喜欢的颜色": ["蓝色", "青色", "蓝绿色", "绿色", "紫色"],  # shades of blue/green
    "座驾": ["自行车", "电动车", "摩托车", "汽车", "越野车"],
    "最喜欢的食物": ["火锅", "麻辣烫", "串串香", "冒菜", "寿司"],
}

FILLER = [
    "今天天气不错。", "最近有点忙。", "周末想出去走走。", "这本书挺有意思。",
    "晚上打算早点休息。", "最近在学习新东西。", "楼下新开了家店。",
]


def make_chain_conv(rng, conv_id, attr, chain, n_filler=4):
    """One conversation where `attr` is updated through `chain` (oldest->newest),
    each update in its own session, interleaved with filler and OTHER facts."""
    turns: list[Turn] = []
    # Plant a few unrelated durable facts to add retrieval noise.
    other_facts = [("宠物", "一只猫"), ("家乡", "杭州"), ("爱好", "摄影")]
    for k, v in other_facts:
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


def answer_igm(store: MemoryStore, attr: str) -> str:
    """IGM: retrieve with slot routing, read the (single) current value."""
    slot = IGMMethod._extract_slot(f"我现在的{attr}是什么？")
    items = store.retrieve(f"我现在的{attr}是什么？", top_k=3, slot=slot)
    for it in items:
        if it.slot == attr:
            return _value_of(it.text, attr)
    return _value_of(items[0].text, attr) if items else ""


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
        embed_model: str | None = None, scorer_path: str | None = None) -> dict:
    rng = random.Random(seed)
    embedder = Embedder(backend=embed_backend, model_path=embed_model)
    scorer = ImportanceScorer.load(scorer_path) if scorer_path else None

    results = {"full_top1": [], "full_recency": [], "igm": []}
    details = []
    attrs = list(UPDATE_CHAINS.keys())
    for trial in range(len(attrs)):
        attr = attrs[trial]
        chain = UPDATE_CHAINS[attr]
        conv = make_chain_conv(rng, f"chain-{attr}", attr, chain)

        # --- Full-storage memory: write every turn, no supersede.
        full_store = MemoryStore(Embedder(backend=embed_backend, model_path=embed_model))
        for t in conv.turns:
            full_store.clock += 1.0
            full_store.write(f"{t.role}: {t.content}")
        # --- IGM memory: importance gate + slot supersede (no LLM in the loop).
        igm_store = MemoryStore(Embedder(backend=embed_backend, model_path=embed_model))
        igm_gate = IGMMethod.__new__(IGMMethod)  # use its static/logic without LLM
        # Manually replicate IGM ingest without needing an LLM instance.
        for t in conv.turns:
            igm_store.clock += 1.0
            text = f"{t.role}: {t.content}"
            # importance via learned scorer or the slot/marker heuristic
            if scorer is not None:
                emb = igm_store.embedder.embed(text)
                max_sim = max([cosine(emb, it.embedding) for it in igm_store.items[-200:]] or [0.0])
                score = scorer.score(text, max_sim_to_store=max_sim)
            else:
                marker_hits = sum(1 for mk in IGMMethod._FACT_MARKERS if mk in text)
                score = min(marker_hits / 3.0, 1.0)
            if score >= 0.5:
                slot = IGMMethod._extract_slot(text)
                igm_store.write(text, importance=score, slot=slot)

        gold = conv.questions[0].answer
        a_full = answer_full_storage(full_store, attr)
        a_rec = answer_full_storage_recency(full_store, attr)
        a_igm = answer_igm(igm_store, attr)
        results["full_top1"].append(a_full == gold)
        results["full_recency"].append(a_rec == gold)
        results["igm"].append(a_igm == gold)
        details.append({
            "attr": attr, "chain": chain, "gold": gold,
            "full_top1": a_full, "full_recency": a_rec, "igm": a_igm,
            "full_mem": len(full_store), "igm_mem": len(igm_store),
        })
        print(f"[{attr:10s}] gold={gold} | full_top1={a_full} full_recency={a_rec} "
              f"igm={a_igm} | mem full={len(full_store)} igm={len(igm_store)}", flush=True)

    summary = {k: sum(v) / len(v) for k, v in results.items()}
    return {"summary": summary, "details": details,
            "config": {"embed_backend": embed_backend, "scorer": bool(scorer), "seed": seed}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--embed-backend", choices=["hash", "bge"], default="bge")
    ap.add_argument("--embed-model", default=("/home/omnichat/.cache/modelscope/models/"
                                              "BAAI--bge-small-zh-v1.5/snapshots/master"))
    ap.add_argument("--scorer", default=str(Path(__file__).parent / "scorer_weights.json"))
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                    / "reports" / "memory-decisive.json")
    args = ap.parse_args()

    scorer_path = args.scorer if Path(args.scorer).is_file() else None
    out = run(seed=args.seed, embed_backend=args.embed_backend,
              embed_model=args.embed_model if args.embed_backend == "bge" else None,
              scorer_path=scorer_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY (accuracy) ===")
    for k, v in out["summary"].items():
        print(f"  {k:14s} {v:.2f}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
