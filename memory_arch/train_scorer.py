"""Train the IGM importance scorer on weakly-supervised synthetic labels.

Labels are auto-generated (no human annotation):
  - positive (1): self-referential durable facts ("我的X是Y", "我喜欢Z")
  - negative (0): filler chit-chat, questions, transient statements

This mirrors the FIP philosophy: a fully controlled training distribution so
the learned weights are attributable to the intended signal, not data noise.
The scorer is then evaluated on a held-out synthetic split.

Usage (any Python with numpy; no LLM needed):
    python -m memory_arch.train_scorer --output memory_arch/scorer_weights.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .scorer import ImportanceScorer

# --- Positive templates: durable, self-referential facts -------------------
ATTR_VALUES = {
    "住址": ["北京", "上海", "深圳", "杭州", "成都"],
    "职业": ["软件工程师", "医生", "教师", "设计师", "产品经理"],
    "最喜欢的颜色": ["蓝色", "红色", "绿色", "黑色"],
    "最喜欢的食物": ["火锅", "寿司", "饺子", "面条"],
    "宠物": ["一只猫", "一只狗", "两只猫"],
    "家乡": ["杭州", "南京", "武汉", "西安"],
    "爱好": ["摄影", "跑步", "游泳", "画画"],
    "名字": ["张伟", "李娜", "王芳"],
}
POS_TEMPLATES = [
    "我的{a}是{v}。",
    "我想告诉你，我的{a}是{v}。",
    "更新一下，我的{a}现在是{v}了。",
    "其实我的{a}一直是{v}。",
    "顺便说下，我的{a}是{v}。",
]

# --- Negative templates: filler / question / transient ---------------------
NEG_FILLER = [
    "今天天气不错，适合出去散步。",
    "最近在看一本书，挺有意思的。",
    "周末打算去超市买点东西。",
    "这道菜的做法还挺简单的。",
    "工作上项目比较多，有点忙。",
    "昨晚看了一部电影，剧情一般。",
    "最近睡眠质量不太好。",
    "楼下新开了一家咖啡店。",
    "这几天在整理房间。",
    "时间过得真快啊。",
]
NEG_QUESTIONS = [
    "我的{a}是什么？",
    "你知道我的{a}吗？",
    "我现在的{a}是什么？",
    "最近有什么推荐的电影吗？",
    "你能帮我查一下天气吗？",
]
NEG_TRANSIENT = [
    "我现在有点饿。",
    "等下我要去开会。",
    "刚刚那个笑话真好笑。",
    "今天路上有点堵。",
]


def build_training_set(seed: int = 1337) -> tuple[list, list]:
    rng = random.Random(seed)
    pos, neg = [], []
    attrs = list(ATTR_VALUES.keys())
    # Positives.
    for _ in range(400):
        a = rng.choice(attrs)
        v = rng.choice(ATTR_VALUES[a])
        t = rng.choice(POS_TEMPLATES).format(a=a, v=v)
        pos.append((t, 1, 0.0))
    # Negatives: filler + questions + transient.
    for _ in range(250):
        neg.append((rng.choice(NEG_FILLER), 0, 0.0))
    for _ in range(150):
        a = rng.choice(attrs)
        neg.append((rng.choice(NEG_QUESTIONS).format(a=a), 0, 0.0))
    for _ in range(100):
        neg.append((rng.choice(NEG_TRANSIENT), 0, 0.0))
    data = pos + neg
    rng.shuffle(data)
    # 80/20 split.
    k = int(0.8 * len(data))
    return data[:k], data[k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "scorer_weights.json")
    args = ap.parse_args()

    train, test = build_training_set(args.seed)
    print(f"train={len(train)} test={len(test)}")
    scorer = ImportanceScorer()
    metrics = scorer.train(train, lr=args.lr, epochs=args.epochs)
    print(f"train_accuracy={metrics['train_accuracy']:.3f}")
    print("learned weights:")
    for name, w in metrics["weights"].items():
        print(f"  {name:14s} {w:+.3f}")

    # Held-out evaluation.
    correct = 0
    for text, label, ms in test:
        p = scorer.score(text, ms)
        correct += int((p >= 0.5) == bool(label))
    print(f"held_out_accuracy={correct/len(test):.3f} (n={len(test)})")

    scorer.save(str(args.output))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
