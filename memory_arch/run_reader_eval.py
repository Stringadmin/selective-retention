"""Reader-in-the-loop evaluation on LongMemEval-S.

The retrieval report measures whether labeled evidence sessions reach top-k and
the leakage report measures rank order inside that set.  Neither says what a
reader *answers*.  This runner closes that gap on knowledge-update questions,
where the failure mode has a name: the reader meets the superseded value first.

Arms (identical evidence, one factor changed at a time):

  ranked_k8 : top-8 user turns by similarity, presented in RANK order, undated
  chrono_k8 : the SAME 8 turns, presented in chronological order
  oracle    : user turns from the labeled evidence sessions, chronological

``ranked_k8`` vs ``chrono_k8`` isolates ORDER with the retrieved set held
constant; ``oracle`` is the reader's ceiling under perfect retrieval.  A
control sample of other question types under ``ranked_k8`` shows the reader is
not simply broken.  Answers are scored by containment against the benchmark
answer, mirroring ``run_m0.score``; this is not the official LongMemEval
evaluator and it does not use an LLM judge.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
import statistics
from pathlib import Path
from typing import Any, Sequence

from . import Embedder, cosine
from .longmemeval_retrieval import select_partition
from .run_stale_leak import identify_new_session

MAX_TURN_CHARS = 500
ARMS = ("ranked_k8", "chrono_k8", "oracle", "reverse_k8", "current_only", "stale_only")
CONTROL_ARMS = ("ranked_k8",)

SYSTEM_PROMPT = (
    "You answer questions about a user using only the provided excerpts of past "
    "conversations. If the excerpts disagree, use the statement that reflects the "
    "user's latest situation. Answer with the value only, no explanation."
)


def score(pred: str, gold: Any) -> bool:
    """Containment scoring on alphanumerics only.

    Deliberately more forgiving than ``run_m0.score``: LongMemEval answers
    appear as numbers (``400000``) or formatted strings (``$400,000``) depending
    on the instance, and punctuation-only differences would otherwise be scored
    as wrong answers and add noise to arm comparisons.  No LLM judge is used.
    """
    normalize = lambda value: re.sub(r"[^0-9a-z]+", "", str(value).lower())
    return normalize(gold) in normalize(pred)


class TransformersReader:
    """Qwen3-4B reader with thinking disabled by the chat template.

    The local GGUF build ignores ``/no_think`` and ``LocalLLM.chat`` truncates at
    ``</think>``, so a llama.cpp reader spends ~50s per question on a reasoning
    preface and then loses the answer.  The HF checkpoint honours
    ``enable_thinking=False``, which answers directly and runs an order of
    magnitude faster on the same GPU.
    """

    def __init__(self, path: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.integrations import hub_kernels

        # The FP8 checkpoint loads its kernel lazily from a hub kernel repo on
        # the first forward pass, and transformers gates that behind a module
        # global that no public argument reaches.  This is a local research run
        # against the official Qwen checkpoint, so allow it explicitly here.
        hub_kernels.ALLOW_ALL_KERNELS = True
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def answer(self, system: str, user: str, max_new_tokens: int = 64,
               thinking: bool = False) -> tuple[str, int]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if thinking and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return text, int(new_tokens.shape[0])


def is_truncated(pred: str) -> bool:
    """A generation that ran out of budget before producing an answer.

    In thinking mode a too-small budget yields an unclosed ``<think>`` preface
    with no answer at all; such rows must not be silently scored as wrong
    answers, because thinking length correlates with context length and would
    bias the arms differently.
    """
    text = str(pred).strip()
    return not text or "<think>" in text


class LlamaCppReader:
    """GGUF reader via llama.cpp, fully offloaded to the GPU.

    Used for the stronger-reader robustness run: a Q6_K quant of the same
    model family fits the 12 GB card in full, which bf16 weights do not.
    Qwen3's GGUF chat template has no enable_thinking switch, so the thinking
    preface is always generated and stripped here; ``always_thinks`` tells the
    runner to budget for it.
    """

    always_thinks = True

    def __init__(self, path: str, n_ctx: int = 8192, n_gpu_layers: int = -1):
        from llama_cpp import Llama

        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                         n_batch=512, verbose=False)

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def answer(self, system: str, user: str, max_new_tokens: int = 768,
               thinking: bool = True) -> tuple[str, int]:
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        raw = out["choices"][0]["message"]["content"].strip()
        if "</think>" in raw:
            answer = raw.split("</think>", 1)[1].strip()
        else:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            answer = lines[-1] if lines else ""
        return answer, len(self.llm.tokenize(raw.encode("utf-8"), add_bos=False))


def user_turns(entry: dict[str, Any], session_id: str) -> list[str]:
    sessions = dict(zip(entry["haystack_session_ids"], entry["haystack_sessions"]))
    return [t["content"] for t in sessions.get(session_id, []) if t["role"] == "user"]


def turn_documents(entry: dict[str, Any]) -> list[tuple[str, int, str]]:
    """One document per user turn: ``(session_id, turn_index, text)``."""
    documents: list[tuple[str, int, str]] = []
    for session_id, turns in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        for index, turn in enumerate(turns):
            if turn["role"] == "user":
                documents.append((session_id, index, turn["content"]))
    return documents


def rank_turns(question: str, documents: Sequence[tuple[str, int, str]],
               embedder: Any) -> list[tuple[str, int, str]]:
    query = embedder.embed(question)
    vectors = [embedder.embed(text) for _, _, text in documents]
    scored = sorted(
        zip(documents, vectors),
        key=lambda pair: -cosine(query, pair[1]),
    )
    return [document for document, _ in scored]


def chronological(entry: dict[str, Any], selected: Sequence[tuple[str, int, str]]):
    dates = dict(zip(entry["haystack_session_ids"], entry["haystack_dates"]))
    return sorted(selected, key=lambda item: (dates.get(item[0], ""), item[1]))


def build_context(turns: Sequence[tuple[str, int, str]]) -> str:
    blocks = []
    for number, (_, _, text) in enumerate(turns, start=1):
        excerpt = text[:MAX_TURN_CHARS] + ("…" if len(text) > MAX_TURN_CHARS else "")
        blocks.append(f"[{number}] {excerpt}")
    return "\n".join(blocks)


def select_evidence(entry: dict[str, Any], arm: str, ranked: Sequence[tuple[str, int, str]],
                    k: int, new_id: str | None = None,
                    old_ids: Sequence[str] = ()) -> list[tuple[str, int, str]]:
    def session_turns(session_ids: Sequence[str]) -> list[tuple[str, int, str]]:
        turns: list[tuple[str, int, str]] = []
        for session_id in session_ids:
            for index, text in enumerate(user_turns(entry, session_id)):
                turns.append((session_id, index, text))
        return chronological(entry, turns)

    if arm == "oracle":
        return session_turns(entry["answer_session_ids"])
    if arm == "current_only":
        return session_turns([new_id]) if new_id else []
    if arm == "stale_only":
        return session_turns(list(old_ids))
    selected = list(ranked[:k])
    if arm == "ranked_k8":
        return selected
    ordered = chronological(entry, selected)
    # reverse_k8 holds the retrieved set constant and only flips the direction,
    # so "chronological" can be separated from "the newest statement last".
    return list(reversed(ordered)) if arm == "reverse_k8" else ordered


def run(entries: list[dict[str, Any]], embedder: Any, reader: Any, k: int = 8,
        control_limit: int = 60, limit: int | None = None,
        thinking: bool = False, arms: Sequence[str] = ARMS,
        run_control: bool = True) -> dict[str, Any]:
    def ask(prompt: str) -> tuple[str, int]:
        budget = 768 if (thinking or getattr(reader, "always_thinks", False)) else 64
        return reader.answer(SYSTEM_PROMPT, prompt, max_new_tokens=budget,
                             thinking=thinking)

    knowledge_update = [e for e in entries if e["question_type"] == "knowledge-update"
                        and not e["question_id"].endswith("_abs")]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry["question_type"] != "knowledge-update" and not entry["question_id"].endswith("_abs"):
            by_type[entry["question_type"]].append(entry)
    per_type = max(1, control_limit // max(len(by_type), 1))
    control = ([entry for question_type in sorted(by_type)
                for entry in by_type[question_type][:per_type]]
               if run_control else [])
    if limit is not None:
        knowledge_update = knowledge_update[:limit]

    rows: list[dict[str, Any]] = []
    for entry in knowledge_update:
        new_id, how, gold_ids = identify_new_session(entry)
        old_ids = [s for s in gold_ids if s != new_id] if new_id else []
        ranked = rank_turns(entry["question"], turn_documents(entry), embedder)
        for arm in arms:
            if arm in ("current_only", "stale_only") and not old_ids:
                continue
            turns = select_evidence(entry, arm, ranked, k, new_id=new_id, old_ids=old_ids)
            context = build_context(turns)
            prompt = f"Excerpts:\n{context}\n\nQuestion: {entry['question']}"
            pred, answer_tokens = ask(prompt)
            shown_sessions = {session_id for session_id, _, _ in turns}
            rows.append({
                "question_id": entry["question_id"],
                "question_type": entry["question_type"],
                "arm": arm,
                "identification": how,
                "leak_in_context": bool(shown_sessions & set(old_ids)),
                "n_turns": len(turns),
                "context_chars": len(context),
                "context_tokens": reader.count_tokens(context),
                "answer_tokens": answer_tokens,
                "gold": entry["answer"],
                "pred": pred,
                "truncated": is_truncated(pred),
                "correct": score(pred, entry["answer"]),
            })
            print(f"  [{arm}] {entry['question_id']} leak={rows[-1]['leak_in_context']} "
                  f"ok={rows[-1]['correct']} pred={pred[:60]!r}", flush=True)

    control_rows: list[dict[str, Any]] = []
    for entry in control:
        ranked = rank_turns(entry["question"], turn_documents(entry), embedder)
        turns = select_evidence(entry, "ranked_k8", ranked, k)
        context = build_context(turns)
        prompt = f"Excerpts:\n{context}\n\nQuestion: {entry['question']}"
        pred, answer_tokens = ask(prompt)
        control_rows.append({
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "arm": "ranked_k8",
            "n_turns": len(turns),
            "context_chars": len(context),
            "context_tokens": reader.count_tokens(context),
            "answer_tokens": answer_tokens,
            "gold": entry["answer"],
            "pred": pred,
            "truncated": is_truncated(pred),
            "correct": score(pred, entry["answer"]),
        })
        print(f"  [control] {entry['question_id']} ok={control_rows[-1]['correct']}", flush=True)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {}
        tokens = [item["context_tokens"] for item in items]
        answered = [item for item in items if not item.get("truncated")]
        return {
            "n": len(items),
            "accuracy": sum(item["correct"] for item in items) / len(items),
            # Rows that never produced an answer are excluded here and their
            # share is reported, so a too-small generation budget cannot hide
            # inside the accuracy.
            "accuracy_answered": (sum(item["correct"] for item in answered) / len(answered)
                                  if answered else 0.0),
            "truncated_rate": 1.0 - len(answered) / len(items),
            "context_tokens_mean": statistics.mean(tokens),
            "context_tokens_p95": sorted(tokens)[int(0.95 * (len(tokens) - 1))],
            "context_chars_mean": statistics.mean(item["context_chars"] for item in items),
            "answer_tokens_mean": statistics.mean(item.get("answer_tokens", 0) for item in items),
        }

    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in arms}
    leak_split = {}
    for arm in arms:
        group = by_arm[arm]
        leaked = [row for row in group if row["leak_in_context"]]
        clean = [row for row in group if not row["leak_in_context"]]
        leak_split[arm] = {
            "with_stale_evidence": summarize(leaked),
            "without_stale_evidence": summarize(clean),
        }

    return {
        "config": {"k": k, "control_limit": control_limit, "limit": limit,
                   "max_turn_chars": MAX_TURN_CHARS, "arms": list(arms),
                   "reader_thinking": thinking},
        "knowledge_update": {
            "overall": {arm: summarize(by_arm[arm]) for arm in arms},
            "by_arm_question_type": {
                arm: {
                    question_type: summarize([row for row in by_arm[arm]
                                              if row["question_type"] == question_type])
                    for question_type in sorted({row["question_type"] for row in by_arm[arm]})
                }
                for arm in arms
            },
            "leak_split": leak_split,
        },
        "control": {
            "by_arm": {"ranked_k8": summarize(control_rows)} if control_rows else {},
            "by_question_type": {
                question_type: summarize([row for row in control_rows
                                          if row["question_type"] == question_type])
                for question_type in sorted({row["question_type"] for row in control_rows})
            },
        },
        "errors": {
            "knowledge_update": [row for row in rows if not row["correct"]],
            "control": [row for row in control_rows if not row["correct"]],
        },
    }


def analyze_discourse_order(entries: list[dict[str, Any]], embedder: Any, report: dict[str, Any],
                            arms: Sequence[str], k: int = 8) -> dict[str, Any]:
    """Does the reader treat discourse order as temporal order?

    No dates are shown to the reader, so the only ordering signal is position:
    in normal prose the later-mentioned statement is the more recent one.  This
    analysis asks whether the shown order agrees with the true temporal order of
    the two conflicting excerpts, and cross-tabs accuracy against that.

    Contexts are rebuilt deterministically (same code path that produced the
    report); a row is correct iff it is absent from ``report["errors"]``.
    """
    failures = {(row["question_id"], row["arm"]) for row in report["errors"]["knowledge_update"]}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        new_id, how, gold_ids = identify_new_session(entry)
        if new_id is None:
            continue
        old_ids = [session_id for session_id in gold_ids if session_id != new_id]
        ranked = rank_turns(entry["question"], turn_documents(entry), embedder)
        for arm in arms:
            if arm in ("current_only", "stale_only") and not old_ids:
                continue
            turns = select_evidence(entry, arm, ranked, k, new_id=new_id, old_ids=old_ids)
            new_positions = [i for i, (sid, _, _) in enumerate(turns) if sid == new_id]
            old_positions = [i for i, (sid, _, _) in enumerate(turns) if sid in old_ids]
            if not new_positions or not old_positions:
                continue  # only one side is visible; the case cannot discriminate
            rows.append({
                "arm": arm,
                "identification": how,
                "new_after_old": max(new_positions) > min(old_positions),
                "gap": max(new_positions) - min(old_positions),
                "correct": (entry["question_id"], arm) not in failures,
            })

    def rate(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {"n": len(group), "accuracy": (sum(r["correct"] for r in group) / len(group)
                                              if group else 0.0)}

    def split(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "new_after_old": rate([r for r in group if r["new_after_old"]]),
            "new_before_old": rate([r for r in group if not r["new_after_old"]]),
        }

    answer_text_only = [r for r in rows if r["identification"] == "answer_text"]
    return {
        "n_rows_both_visible": len(rows),
        "pooled": split(rows),
        "pooled_answer_text_only": split(answer_text_only),
        "per_arm": {arm: split([r for r in rows if r["arm"] == arm]) for arm in arms},
    }


def analyze_gold_position(entries: list[dict[str, Any]], embedder: Any, report: dict[str, Any],
                          arms: Sequence[str], k: int = 8) -> dict[str, Any]:
    """Accuracy as a function of where the answer-bearing excerpt sits.

    Position 1 = first excerpt shown, k = last.  Falling accuracy with position
    means primacy anchoring; rising accuracy means recency.  Only pairs whose
    gold string is verbatim present in some excerpt can be scored, so this uses
    the ``answer_text`` subset of the knowledge-update questions.
    """
    failures = {(row["question_id"], row["arm"]) for row in report["errors"]["knowledge_update"]}
    normalize = lambda value: re.sub(r"[^0-9a-z]+", "", str(value).lower())
    buckets: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        new_id, _, gold_ids = identify_new_session(entry)
        if new_id is None:
            continue
        old_ids = [session_id for session_id in gold_ids if session_id != new_id]
        gold = normalize(entry["answer"])
        if not gold:
            continue
        ranked = rank_turns(entry["question"], turn_documents(entry), embedder)
        for arm in arms:
            if arm in ("current_only", "stale_only") and not old_ids:
                continue
            turns = select_evidence(entry, arm, ranked, k, new_id=new_id, old_ids=old_ids)
            positions = [i + 1 for i, (_, _, text) in enumerate(turns) if gold in normalize(text)]
            if not positions:
                continue
            buckets[arm][min(positions)].append(
                (entry["question_id"], arm) not in failures)

    def summarize(by_position: dict[int, list[bool]]) -> dict[str, Any]:
        total = [value for values in by_position.values() for value in values]
        return {
            "n": len(total),
            "accuracy": sum(total) / len(total) if total else 0.0,
            "by_position": {
                str(position): {"n": len(values), "accuracy": sum(values) / len(values)}
                for position, values in sorted(by_position.items())
            },
        }

    return {arm: summarize(buckets[arm]) for arm in arms if buckets[arm]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/longmemeval-s-reader-eval.json")
    parser.add_argument("--embed-model", required=True)
    parser.add_argument("--model", required=False, default=None,
                        help="HF checkpoint directory of the reader (generation mode)")
    parser.add_argument("--analyze-report", default=None,
                        help="skip generation: rebuild this report's contexts and print the "
                             "mechanism analyses")
    parser.add_argument("--analysis", choices=("discourse", "gold-position"), default="discourse")
    parser.add_argument("--partition", choices=["all", "development", "test"], default="all")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--control-limit", type=int, default=60)
    parser.add_argument("--thinking", action="store_true",
                        help="let the reader reason before answering (slower, stronger)")
    parser.add_argument("--limit", type=int, default=None,
                        help="knowledge-update questions to run (smoke testing)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                        help="subset of arms to run (default: all)")
    parser.add_argument("--reader-backend", choices=("hf", "llama"), default="hf",
                        help="hf = HF checkpoint (TransformersReader); llama = GGUF file (LlamaCppReader)")
    parser.add_argument("--no-control", action="store_true",
                        help="skip the other-question-type control group")
    args = parser.parse_args()

    entries = json.loads(Path(args.input).read_text(encoding="utf-8"))
    entries = list(select_partition(entries, args.partition))
    embedder = Embedder(backend="bge", model_path=args.embed_model)

    if args.analyze_report:
        report = json.loads(Path(args.analyze_report).read_text(encoding="utf-8"))
        knowledge_update = [e for e in entries
                            if e["question_type"] == "knowledge-update"
                            and not e["question_id"].endswith("_abs")]
        analyze = (analyze_discourse_order if args.analysis == "discourse"
                   else analyze_gold_position)
        result = analyze(knowledge_update, embedder, report,
                         report["config"]["arms"], k=report["config"]["k"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not args.model:
        parser.error("--model is required to generate; use --analyze-report to analyse an existing report")
    print("loading reader...", flush=True)
    reader = (TransformersReader(args.model) if args.reader_backend == "hf"
              else LlamaCppReader(args.model))
    print(f"reader ready ({args.reader_backend})", flush=True)

    report = run(entries, embedder, reader, k=args.k,
                 control_limit=args.control_limit, limit=args.limit, thinking=args.thinking,
                 arms=args.arms, run_control=not args.no_control)
    report["config"].update({"input": args.input, "partition": args.partition,
                             "reader_model": args.model, "embed_model": args.embed_model})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "knowledge_update": {arm: round(summary["accuracy"], 3)
                             for arm, summary in report["knowledge_update"]["overall"].items()},
        "control": (round(report["control"]["by_arm"]["ranked_k8"]["accuracy"], 3)
                    if report["control"]["by_arm"] else None),
        "saved": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
