"""Fast end-to-end smoke for the controlled FIP Phase 0 pipeline.

This is intentionally *not* the frozen Phase 0 experiment.  It checks that
all three currently implemented groups (dense A, sparse-only E, FIP F) use the
same stream, token budget, steps, initial seed, evaluation protocol, report
format, and optimizer wiring before a costly five-seed A--F matrix is built.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.synthetic_tasks import PROMPT_LENGTH, VOCAB_SIZE, SyntheticTask, SyntheticTaskSuite
from fip import FeatureMaskedAdamW, FIPTransformer, ModelConfig, PlasticityController


IMPLEMENTED_GROUPS = ("A", "E", "F")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--groups", nargs="+", default=list(IMPLEMENTED_GROUPS))
    parser.add_argument("--tasks", nargs="+", default=SyntheticTaskSuite().names)
    parser.add_argument("--steps-per-task", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports" / "phase0-smoke.json"
    )
    args = parser.parse_args()
    invalid = set(args.groups) - set(IMPLEMENTED_GROUPS)
    if invalid:
        parser.error(
            f"smoke implements only {IMPLEMENTED_GROUPS}; B/C/D require their own "
            f"frozen baseline implementations, not a placeholder ({sorted(invalid)})"
        )
    if args.steps_per_task <= 0 or args.batch_size <= 0 or args.eval_batches <= 0:
        parser.error("steps-per-task, batch-size, and eval-batches must be positive")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(args: argparse.Namespace, group: str, device: torch.device) -> FIPTransformer:
    use_fip = group in ("E", "F")
    cfg = ModelConfig(
        vocab_size=VOCAB_SIZE,
        n_layer=args.layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_head=4,
        context=PROMPT_LENGTH,
        top_k=args.top_k,
        free_ratio=0.2,
        use_fip=use_fip,
    )
    return FIPTransformer(cfg).to(device)


def train_task(
    model: FIPTransformer,
    task: SyntheticTask,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    last_loss = float("nan")
    for _ in range(steps):
        inputs, targets = task.sample(batch_size, generator=generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


@torch.no_grad()
def evaluate_task(
    model: FIPTransformer,
    task: SyntheticTask,
    *,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    correct = 0
    total = 0
    model.eval()
    for _ in range(batches):
        inputs, targets = task.sample(batch_size, generator=generator, device=device)
        prediction = model(inputs)[:, -1, :].argmax(dim=-1)
        correct += int((prediction == targets).sum().item())
        total += targets.numel()
    return correct / total


def feature_summary(model: FIPTransformer) -> dict[str, Any] | None:
    if not model.cfg.use_fip:
        return None
    return {
        str(layer_id): counts
        for layer_id, counts in model.registry.state_counts().items()
    }


def run_group(
    group: str,
    tasks: list[SyntheticTask],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    # Resetting the seed makes the A/E/F initialization reproducible.  Dense
    # and sparse models have equal FFN parameter shapes; only the Top-K path
    # and FIP controller distinguish the three smoke groups.
    seed_everything(args.seed)
    model = make_model(args, group, device)
    controller = PlasticityController(model, model.registry) if group == "F" else None
    if model.cfg.use_fip:
        optimizer: torch.optim.Optimizer = FeatureMaskedAdamW(
            model.parameters(), model=model, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        if controller is not None:
            controller.begin_task()
        loss = train_task(
            model, task, optimizer,
            steps=args.steps_per_task,
            batch_size=args.batch_size,
            seed=args.seed + task_index,
            device=device,
        )
        if controller is not None:
            controller.end_task()

        scores = {
            previous.name: evaluate_task(
                model, previous,
                batches=args.eval_batches,
                batch_size=args.batch_size,
                seed=args.seed + 10_000 + previous_index,
                device=device,
            )
            for previous_index, previous in enumerate(tasks[: task_index + 1])
        }
        history.append(
            {
                "trained_task": task.name,
                "last_train_loss": loss,
                "accuracy_after_task": scores,
                "feature_states": feature_summary(model),
            }
        )

    obsolete_tasks = {
        task_name for task in tasks for task_name in task.supersedes
    }
    per_task_forgetting: dict[str, float | None] = {}
    for task in tasks:
        observed = [entry["accuracy_after_task"].get(task.name) for entry in history]
        observed = [value for value in observed if value is not None]
        per_task_forgetting[task.name] = (
            None if task.name in obsolete_tasks else max(observed) - observed[-1]
        )
    retention_forgetting = [
        value for value in per_task_forgetting.values() if value is not None
    ]
    elapsed = time.perf_counter() - started
    return {
        "group": group,
        "parameter_count": model.num_params(),
        "elapsed_seconds": elapsed,
        "tokens_trained": len(tasks) * args.steps_per_task * args.batch_size * PROMPT_LENGTH,
        "history": history,
        "final_accuracy": history[-1]["accuracy_after_task"],
        "obsolete_tasks": sorted(obsolete_tasks),
        "per_task_forgetting": per_task_forgetting,
        "average_retention_forgetting": sum(retention_forgetting) / len(retention_forgetting),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    suite = SyntheticTaskSuite()
    tasks = suite.ordered(args.tasks)
    if args.top_k > args.d_ff:
        raise ValueError("top-k cannot exceed d-ff")

    report = {
        "kind": "phase0_pipeline_smoke",
        "scientific_result": False,
        "reason": "small configuration and partial A/E/F matrix; not the frozen five-seed Phase 0",
        "torch": torch.__version__,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "tasks": [task.name for task in tasks],
        "fairness": {
            "same_task_order": True,
            "same_steps_per_task": args.steps_per_task,
            "same_batch_size": args.batch_size,
            "same_tokens_per_group": len(tasks) * args.steps_per_task * args.batch_size * PROMPT_LENGTH,
            "same_learning_rate": args.lr,
            "same_weight_decay": args.weight_decay,
            "same_initial_seed": args.seed,
        },
        "groups": [run_group(group, tasks, args, device) for group in args.groups],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "groups": [group["group"] for group in report["groups"]],
        "device": report["device"],
        "scientific_result": report["scientific_result"],
    }, indent=2))


if __name__ == "__main__":
    main()
