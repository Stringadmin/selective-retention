"""M0 baseline evaluation runner for the IGM memory architecture.

Runs the four memory methods (full_context / naive_rag / mem0_style / igm)
over the synthetic multi-session dataset and reports, per method:
  - accuracy overall and by category (exact-match + containment)
  - number of memory writes (storage footprint)
  - wall-clock time

Usage (in WSL, model on local disk):
    ~/fip-venv/bin/python -m memory_arch.run_m0 \
        --model /home/omnichat/models/.../sha256-... \
        --methods naive_rag mem0_style igm \
        --output reports/memory-m0.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_arch import Embedder, LocalLLM, METHODS
from memory_arch.synth_data import build_dataset
from memory_arch.scorer import ImportanceScorer

DEFAULT_MODEL = ("/home/omnichat/models/ollama-archive/blobs/"
                 "sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f")
DEFAULT_BGE = ("/home/omnichat/.cache/modelscope/models/"
               "BAAI--bge-small-zh-v1.5/snapshots/master")
DEFAULT_SCORER = str(Path(__file__).parent / "scorer_weights.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--methods", nargs="+", default=["full_context", "naive_rag", "mem0_style", "igm"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--n-gpu-layers", type=int, default=0)
    p.add_argument("--max-answer-tokens", type=int, default=120)
    p.add_argument("--embed-backend", choices=["hash", "bge"], default="hash")
    p.add_argument("--embed-model", default=DEFAULT_BGE)
    p.add_argument("--scorer", default=None,
                   help="path to learned scorer weights; if set, IGM uses the learned write-gate")
    p.add_argument("--use-learned-scorer", action="store_true",
                   help=f"load default scorer weights from {DEFAULT_SCORER}")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "memory-m0.json")
    p.add_argument("--hard", action="store_true",
                   help="use the harder distractor dataset (retrieval noise)")
    p.add_argument("--n-recall", type=int, default=8)
    p.add_argument("--n-update", type=int, default=6)
    p.add_argument("--n-temporal", type=int, default=3)
    return p.parse_args()


def score(pred: str, gold: str) -> bool:
    """Containment scoring: correct if the gold answer appears in the prediction."""
    p = pred.replace(" ", "").lower()
    g = gold.replace(" ", "").lower()
    return g in p


def main():
    args = parse_args()
    convs = build_dataset(args.seed, hard=args.hard,
                          n_recall=args.n_recall, n_update=args.n_update,
                          n_temporal=args.n_temporal)
    n_q = sum(len(c.questions) for c in convs)
    print(f"dataset: {len(convs)} conversations, {n_q} questions", flush=True)

    embedder = Embedder(backend=args.embed_backend,
                        model_path=args.embed_model if args.embed_backend == "bge" else None)
    print(f"embedder backend: {args.embed_backend}", flush=True)

    scorer = None
    scorer_path = args.scorer or (DEFAULT_SCORER if args.use_learned_scorer else None)
    if scorer_path and Path(scorer_path).is_file():
        scorer = ImportanceScorer.load(scorer_path)
        print(f"loaded learned scorer from {scorer_path}", flush=True)

    print("loading LLM...", flush=True)
    llm = LocalLLM(args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers)
    print("LLM ready", flush=True)

    results = []
    for method_name in args.methods:
        cls = METHODS[method_name]
        # Only IGM accepts a scorer kwarg; others take (llm, embedder).
        if method_name == "igm":
            method = cls(llm, embedder, scorer=scorer)
        else:
            method = cls(llm, embedder)
        t0 = time.time()
        # Ingest all conversations.
        for conv in convs:
            method.ingest(conv)
        # Answer all questions.
        per_cat: dict[str, list[bool]] = {}
        details = []
        for conv in convs:
            for q in conv.questions:
                pred = method.answer(q.question)
                ok = score(pred, q.answer)
                per_cat.setdefault(q.category, []).append(ok)
                details.append({
                    "conv": conv.conv_id, "category": q.category,
                    "question": q.question, "gold": q.answer,
                    "pred": pred, "correct": ok,
                })
        elapsed = time.time() - t0
        all_ok = [ok for oks in per_cat.values() for ok in oks]
        acc = sum(all_ok) / len(all_ok) if all_ok else 0.0
        cat_acc = {c: (sum(oks) / len(oks) if oks else 0.0) for c, oks in per_cat.items()}
        n_mem = len(method.store) if hasattr(method, "store") else method.n_writes
        results.append({
            "method": method_name,
            "accuracy": acc,
            "category_accuracy": cat_acc,
            "memory_items": n_mem,
            "n_writes": method.n_writes,
            "elapsed_seconds": elapsed,
        })
        print(f"[{method_name:13s}] acc={acc:.3f} cat={ {k: round(v,2) for k,v in cat_acc.items()} } "
              f"mem={n_mem} writes={method.n_writes} ({elapsed:.0f}s)", flush=True)
        # Save per-method details alongside.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / f"memory-m0-{method_name}-details.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "kind": "memory_m0_baseline",
        "dataset": {"conversations": len(convs), "questions": n_q, "seed": args.seed},
        "model": args.model,
        "config": {
            "embed_backend": args.embed_backend,
            "embed_model": args.embed_model if args.embed_backend == "bge" else None,
            "learned_scorer": scorer_path,
            "n_gpu_layers": args.n_gpu_layers,
            "hard": args.hard,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "summary": {r["method"]: round(r["accuracy"], 3) for r in results}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
