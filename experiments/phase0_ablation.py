"""S1 attribution ablation runner (diagnostic, non-comparable).

Answers the Phase 0 gating question: is FIP's *intelligent feature
allocation* itself effective, or is the F group's failure caused elsewhere?

Groups (all single-seed, diagnostic only):

- ``F``: full FIP reproduction (importance allocation, stable freeze, Top-K).
  Must match the frozen group F result (forgetting ~0.4915).
- ``F-random`` (protocol ablation #5): same state-distribution sizes, but
  stable/shared promotion slots are chosen uniformly at random instead of by
  importance ranking.
- ``F-nofreeze`` (protocol ablation #2): stable features stay trainable
  (stable_scale = 1.0), isolating the contribution of the freeze.
- ``F-notopk`` (protocol ablation #1): Top-K = d_ff, i.e. dense activation
  inside the FIP FFN, isolating the contribution of sparsity.
- ``E``: sparse full-FT reference (no feature plasticity management), run via
  ``experiments.phase0.train_one_group`` for exact parity with the frozen
  runner.

The ablation reuses the exact Phase 0 task suite, token budget, steps,
batch size, seed schedule, and evaluation protocol.  It never modifies
``experiments/phase0.py``, protecting the frozen runner's source hashes.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.synthetic_tasks import PROMPT_LENGTH, VOCAB_SIZE, SyntheticTaskSuite
from fip import (
    FeatureMaskedAdamW,
    FIPTransformer,
    ModelConfig,
    PlasticityController,
)
from experiments.phase0 import (
    evaluate,
    feature_set,
    feature_state_summary,
    module_drift,
    resolve_device,
    seed_everything,
    snapshot_parameters,
    train_one_group,
)


ABLATION_GROUPS = ("F", "F-random", "F-nofreeze", "F-notopk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--groups", nargs="+", default=("F", "F-random", "F-nofreeze", "F-notopk", "E"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps-per-task", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "phase0-ablation-s1")
    return parser.parse_args()


def abl_config(group: str) -> ModelConfig:
    """FIP configuration for the F-style ablation groups."""
    base = dict(
        vocab_size=VOCAB_SIZE,
        n_layer=6,
        d_model=256,
        d_ff=1024,
        n_head=4,
        context=256,
        top_k=64,
        free_ratio=0.2,
        importance_ema=0.99,
        stable_quantile=0.5,
        shared_min_tasks=2,
        free_activate_quantile=0.25,
        task_sig_quantile=0.75,
        use_fip=True,
    )
    if group == "F-random":
        base["allocation_mode"] = "random"
    elif group == "F-nofreeze":
        base["stable_scale"] = 1.0
    elif group == "F-notopk":
        base["top_k"] = base["d_ff"]  # dense activation, FIP allocation intact
    return ModelConfig(**base)


def train_abl_group(
    group: str,
    *,
    seed: int,
    tasks,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """FIP training loop mirroring experiments.phase0 group-F logic."""
    seed_everything(seed)
    model = FIPTransformer(abl_config(group)).to(device)
    controller = PlasticityController(model, model.registry)
    optimizer = FeatureMaskedAdamW(
        model.parameters(), model=model, lr=args.lr, weight_decay=args.weight_decay
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    task_history: list[dict[str, Any]] = []
    active_sets: list[set[tuple[int, int]]] = []
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        controller.begin_task()
        generator = torch.Generator(device="cpu").manual_seed(seed + 100 * task_index)
        final_loss = float("nan")
        version_before = {
            layer_id: reg.version.clone() for layer_id, reg in model.registry.layers.items()
        }
        param_snapshot = snapshot_parameters(model)
        model.train()
        for _ in range(args.steps_per_task):
            inputs, targets = task.sample(args.batch_size, generator=generator, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), args.grad_clip
            )
            optimizer.step()
            final_loss = float(loss.detach().cpu())

        active_sets.append(feature_set(model))
        drift = module_drift(model, param_snapshot, model.registry)
        controller.end_task()
        accuracy_after = {
            previous.name: evaluate(
                model, previous,
                seed=seed + 10_000 + previous_index,
                batch_size=args.batch_size,
                batches=args.eval_batches,
                device=device,
            )
            for previous_index, previous in enumerate(tasks[: task_index + 1])
        }
        changed = total = 0
        for layer_id, reg in model.registry.layers.items():
            changed += int((reg.version != version_before[layer_id]).sum().item())
            total += reg.n_features
        task_history.append({
            "trained_task": task.name,
            "final_train_loss": final_loss,
            "accuracy_after_task": accuracy_after,
            "feature_states": feature_state_summary(model),
            "active_feature_count": len(active_sets[-1]),
            "significant_feature_fraction": (
                len(active_sets[-1]) / sum(reg.n_features for reg in model.registry.layers.values())
            ),
            "changed_feature_ratio": changed / total if total else None,
            "module_drift": drift,
        })

    obsolete = {name for task in tasks for name in task.supersedes}
    forgetting: dict[str, float | None] = {}
    for task in tasks:
        observed = [item["accuracy_after_task"].get(task.name) for item in task_history]
        observed = [score for score in observed if score is not None]
        forgetting[task.name] = None if task.name in obsolete else max(observed) - observed[-1]
    retention_values = [value for value in forgetting.values() if value is not None]
    elapsed = time.perf_counter() - started
    jaccard: dict[str, float] = {}
    for later_index, later in enumerate(active_sets):
        for earlier_index, earlier in enumerate(active_sets[:later_index]):
            union = later | earlier
            key = f"{tasks[earlier_index].name}__{tasks[later_index].name}"
            jaccard[key] = len(later & earlier) / len(union) if union else 1.0
    return {
        "group": group,
        "ablated_component": {
            "F": None,
            "F-random": "allocation: importance -> random",
            "F-nofreeze": "stable_scale: 0.0 -> 1.0",
            "F-notopk": "top_k: 64 -> 1024 (dense)",
        }[group],
        "base_parameter_count": model.num_params(),
        "tokens_trained": len(tasks) * args.steps_per_task * args.batch_size * PROMPT_LENGTH,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (
            len(tasks) * args.steps_per_task * args.batch_size * PROMPT_LENGTH
        ) / elapsed,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "task_history": task_history,
        "final_accuracy": task_history[-1]["accuracy_after_task"],
        "obsolete_tasks": sorted(obsolete),
        "per_task_forgetting": forgetting,
        "average_retention_forgetting": sum(retention_values) / len(retention_values),
        "feature_jaccard": jaccard,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    suite = SyntheticTaskSuite()
    tasks = suite.ordered()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for group in args.groups:
        if group == "E":
            results.append(train_one_group(
                "E", seed=args.seed, tasks=tasks, args=args, device=device
            ))
        else:
            results.append(train_abl_group(
                group, seed=args.seed, tasks=tasks, args=args, device=device
            ))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "kind": "fip_phase0_ablation_s1",
        "diagnostic": "non-comparable; single seed; attribution of FIP feature allocation",
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "fairness": {
            "same_task_order": True,
            "same_steps_per_task": args.steps_per_task,
            "same_batch_size": args.batch_size,
            "same_token_budget": True,
            "same_lr": args.lr,
            "same_weight_decay": args.weight_decay,
            "e_group_via_frozen_runner": True,
            "f_group_must_match_0_4915": True,
        },
        "results": results,
    }
    output = args.output_dir / "phase0-results.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
