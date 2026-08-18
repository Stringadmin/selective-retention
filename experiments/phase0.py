"""Protocol-driven Phase 0 continual-learning experiment runner.

The default configuration matches ``configs/phase0.yaml`` and the frozen
research protocol: six controlled tasks, fixed seeds, equal current-task token
budgets and optimizer steps.  It is intentionally separate from
``phase0_smoke``; a smoke report is never a scientific result.

Implemented comparable groups are A (dense FT), B (online EWC), C (5% replay),
E (Top-K sparse FT), and F (FIP).  D's mergeable LoRA mechanism is implemented
and unit-tested, but this runner refuses it by default because its temporary
adapter parameters violate the protocol's strict maximum-parameter equality
until a width-matching calibration is frozen.

Diagnostic groups:
- G (FIP + frozen non-FFN after task 1): measures the FFN-only isolation
  boundary.  Non-comparable.
- H (FIP + progressive non-FFN protection): importance-driven update scaling
  for embedding/head, attention, and norms.  Non-comparable.
- I (FIP + sticky non-FFN protection): same as H but with cumulative
  protection that prevents protected-set turnover.  Non-comparable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from experiments.baselines import OnlineEWC, ReplayBuffer
from fip import (
    FeatureMaskedAdamW,
    FeatureState,
    FIPTransformer,
    ModelConfig,
    NonFFNProtector,
    PlasticityController,
)


COMPARABLE_GROUPS = ("A", "B", "C", "E", "F")
ALL_GROUPS = ("A", "B", "C", "D", "E", "F", "G", "H", "I")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1337, 42, 7, 2024, 314159))
    parser.add_argument("--groups", nargs="+", default=COMPARABLE_GROUPS, choices=ALL_GROUPS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps-per-task", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ewc-coefficient", type=float, default=100.0)
    parser.add_argument("--ewc-decay", type=float, default=1.0)
    parser.add_argument("--replay-fraction", type=float, default=0.05)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--protection-quantile", type=float, default=0.5)
    parser.add_argument("--protected-scale", type=float, default=0.1)
    parser.add_argument(
        "--per-task-budget",
        type=float,
        default=None,
        help="group I: per-task admission budget for sticky protection "
             "(capacity grows with task count); None uses the fixed cap",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "phase0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-noncomparable-lora",
        action="store_true",
        help="allow D despite its temporary adapter parameter overhead (diagnostic only)",
    )
    args = parser.parse_args()
    if args.steps_per_task <= 0 or args.batch_size <= 1 or args.eval_batches <= 0:
        parser.error("steps-per-task/eval-batches must be positive and batch-size must exceed 1")
    if "D" in args.groups and not args.allow_noncomparable_lora:
        parser.error(
            "D is blocked: temporary LoRA parameters break frozen max-parameter parity. "
            "Calibrate a matched-width D variant first, or use the explicit diagnostic override."
        )
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(group: str, device: torch.device) -> FIPTransformer:
    # The fixed Phase 0 architecture in the frozen protocol/config.
    cfg = ModelConfig(
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
        use_fip=group in ("E", "F", "G", "H", "I"),
    )
    model = FIPTransformer(cfg).to(device)
    if group == "D":
        model.enable_lora(rank=8, alpha=8)
    return model


def evaluate(
    model: FIPTransformer,
    task: SyntheticTask,
    *,
    seed: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = task.sample(batch_size, generator=generator, device=device)
            predicted = model(inputs)[:, -1, :].argmax(dim=-1)
            correct += int((predicted == targets).sum().item())
            total += targets.numel()
    return correct / total


def feature_set(model: FIPTransformer) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for layer_id, reg in model.registry.layers.items():
        for index in model.registry.significant_indices(layer_id).flatten().tolist():
            result.add((layer_id, index))
    return result


def feature_state_summary(model: FIPTransformer) -> dict[str, Any] | None:
    if not model.cfg.use_fip:
        return None
    return {str(layer_id): counts for layer_id, counts in model.registry.state_counts().items()}


def snapshot_parameters(model: FIPTransformer) -> dict[str, torch.Tensor]:
    """Clone all non-LoRA parameters for drift measurement."""
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if "lora_" not in name
    }


def module_drift(
    model: FIPTransformer,
    snapshot: dict[str, torch.Tensor],
    registry: FeatureRegistry | None,
) -> dict[str, float]:
    """Compute per-module-category L2 drift from a parameter snapshot.

    Categories: embedding_head, attention, norms, ffn_stable, ffn_shared,
    ffn_plastic_free.  FFN drift is broken down by the *current* feature state
    (i.e. the state that was active during training, before ``end_task``
    promotes states for the next task).
    """
    drift_sq: dict[str, float] = {
        "embedding_head": 0.0,
        "attention": 0.0,
        "norms": 0.0,
        "ffn_stable": 0.0,
        "ffn_shared": 0.0,
        "ffn_plastic_free": 0.0,
    }
    for name, p in model.named_parameters():
        if "lora_" in name:
            continue
        ref = snapshot.get(name)
        if ref is None:
            continue
        delta = p.detach() - ref
        if "tok_emb" in name or "head" in name:
            drift_sq["embedding_head"] += delta.float().square().sum().item()
        elif "attn" in name:
            drift_sq["attention"] += delta.float().square().sum().item()
        elif "norm" in name:
            drift_sq["norms"] += delta.float().square().sum().item()
        elif any(k in name for k in ("w_gate", "w_up", "w_down")):
            if registry is None:
                continue
            parts = name.split(".")
            layer_id = int(parts[1])
            reg = registry.layers[layer_id]
            state = reg.state
            is_down = "w_down" in name
            for label, mask in (
                ("ffn_stable", state == int(FeatureState.STABLE)),
                ("ffn_shared", state == int(FeatureState.SHARED)),
                (
                    "ffn_plastic_free",
                    (state == int(FeatureState.PLASTIC)) | (state == int(FeatureState.FREE)),
                ),
            ):
                if not mask.any():
                    continue
                slice_delta = delta[mask] if not is_down else delta[:, mask]
                drift_sq[label] += slice_delta.float().square().sum().item()
    return {label: value ** 0.5 for label, value in drift_sq.items()}


def freeze_non_ffn_params(model: FIPTransformer) -> int:
    """Freeze all parameters except FIP-FFN feature weights. Returns count frozen."""
    frozen = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            continue
        is_ffn = any(k in name for k in ("w_gate", "w_up", "w_down"))
        if not is_ffn and p.requires_grad:
            p.requires_grad_(False)
            frozen += 1
    return frozen


def train_one_group(
    group: str,
    *,
    seed: int,
    tasks: list[SyntheticTask],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    model = make_model(group, device)
    adapter_parameter_count = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name
    )
    base_parameter_count = model.num_params() - adapter_parameter_count
    controller = PlasticityController(model, model.registry) if group in ("F", "G", "H", "I") else None
    ewc = OnlineEWC(model, args.ewc_coefficient, args.ewc_decay) if group == "B" else None
    replay = ReplayBuffer(args.replay_capacity, args.replay_fraction, seed) if group == "C" else None
    non_ffn_protector = (
        NonFFNProtector(
            model,
            protection_quantile=args.protection_quantile,
            protected_scale=args.protected_scale,
            sticky=(group == "I"),
            max_protected_fraction=0.5,
            per_task_budget=args.per_task_budget,
            token_ownership=(group == "I"),
        )
        if group in ("H", "I") else None
    )

    if group in ("E", "F", "G", "H", "I"):
        optimizer: torch.optim.Optimizer | None = FeatureMaskedAdamW(
            model.parameters(), model=model, lr=args.lr, weight_decay=args.weight_decay,
            non_ffn_protector=non_ffn_protector,
        )
    elif group == "D":
        optimizer = None  # fresh adapter optimizer at each merged task boundary
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    task_history: list[dict[str, Any]] = []
    active_sets: list[set[tuple[int, int]]] = []
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        if controller is not None:
            controller.begin_task()
        if non_ffn_protector is not None:
            non_ffn_protector.begin_task()
        if replay is not None:
            replay.begin_task()
        if group == "D":
            optimizer = torch.optim.AdamW(
                model.lora_parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        assert optimizer is not None
        generator = torch.Generator(device="cpu").manual_seed(seed + 100 * task_index)
        replay_examples = 0
        final_loss = float("nan")
        version_before = {
            layer_id: reg.version.clone() for layer_id, reg in model.registry.layers.items()
        }
        param_snapshot = snapshot_parameters(model) if group in ("F", "G", "H", "I") else None
        model.train()
        for _ in range(args.steps_per_task):
            if replay is not None:
                replay_count = replay.next_replay_count(args.batch_size)
                inputs, targets = task.sample(
                    args.batch_size - replay_count, generator=generator, device=device
                )
                batch = replay.mix(
                    inputs, targets, total_batch_size=args.batch_size, device=device
                )
                replay.add(inputs, targets)
                inputs, targets = batch.inputs, batch.targets
                replay_examples += batch.replay_examples
            else:
                inputs, targets = task.sample(args.batch_size, generator=generator, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            task_loss = F.cross_entropy(logits[:, -1, :], targets)
            if ewc is not None:
                # Fisher comes from current-task likelihood gradients only; the
                # EWC regularizer then contributes its own gradient to the step.
                # The Fisher accumulator reads the resulting gradients, not
                # the task-loss graph.  Retaining that graph would leak one
                # full autograd graph per training step in the EWC baseline.
                task_loss.backward()
                ewc.accumulate_current_gradients()
                regularizer = ewc.penalty()
                loss = task_loss + regularizer
                if regularizer.requires_grad:
                    regularizer.backward()
            else:
                loss = task_loss
                loss.backward()
            if non_ffn_protector is not None:
                non_ffn_protector.update_importance()
                non_ffn_protector.record_task_tokens(inputs)
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                args.grad_clip,
            )
            optimizer.step()
            final_loss = float(loss.detach().cpu())

        active = feature_set(model) if group in ("F", "G", "H", "I") else set()
        active_sets.append(active)
        drift = (
            module_drift(model, param_snapshot, model.registry)
            if param_snapshot is not None else None
        )
        if controller is not None:
            controller.end_task()
        if non_ffn_protector is not None:
            non_ffn_protector.end_task()
        protected_frac = (
            non_ffn_protector.protected_fraction()
            if non_ffn_protector is not None else None
        )
        protector_turnover = (
            non_ffn_protector.turnover()
            if non_ffn_protector is not None else None
        )
        protector_summary = (
            non_ffn_protector.protected_set_summary()
            if non_ffn_protector is not None else None
        )
        if ewc is not None:
            ewc.end_task()
        if group == "D":
            model.merge_lora_()
        if group == "G" and task_index == 0:
            freeze_non_ffn_params(model)

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
        changed_feature_ratio = None
        if group in ("F", "G", "H", "I"):
            changed = 0
            total_features = 0
            for layer_id, reg in model.registry.layers.items():
                changed += int((reg.version != version_before[layer_id]).sum().item())
                total_features += reg.n_features
            changed_feature_ratio = changed / total_features
        task_history.append({
            "trained_task": task.name,
            "final_train_loss": final_loss,
            "accuracy_after_task": accuracy_after,
            "replay_examples": replay_examples,
            "feature_states": feature_state_summary(model),
            "active_feature_count": len(active),
            "significant_feature_fraction": (
                len(active) / sum(reg.n_features for reg in model.registry.layers.values())
                if group in ("F", "G", "H", "I") else None
            ),
            "changed_feature_ratio": changed_feature_ratio,
            "module_drift": drift,
            "non_ffn_protected_fraction": protected_frac,
            "non_ffn_turnover": protector_turnover,
            "non_ffn_protected_set": protector_summary,
        })

    obsolete = {name for task in tasks for name in task.supersedes}
    forgetting: dict[str, float | None] = {}
    for task in tasks:
        observed = [item["accuracy_after_task"].get(task.name) for item in task_history]
        observed = [score for score in observed if score is not None]
        forgetting[task.name] = None if task.name in obsolete else max(observed) - observed[-1]
    retention_values = [value for value in forgetting.values() if value is not None]
    elapsed = time.perf_counter() - started
    result = {
        "group": group,
        "base_parameter_count": base_parameter_count,
        "total_parameter_count": model.num_params(),
        "temporary_adapter_parameter_count": adapter_parameter_count,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "tokens_trained": len(tasks) * args.steps_per_task * args.batch_size * PROMPT_LENGTH,
        "current_task_tokens": (
            len(tasks) * args.steps_per_task * args.batch_size * PROMPT_LENGTH
            - sum(item["replay_examples"] for item in task_history) * PROMPT_LENGTH
        ),
        "replay_tokens": sum(item["replay_examples"] for item in task_history) * PROMPT_LENGTH,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (len(tasks) * args.steps_per_task * args.batch_size * PROMPT_LENGTH) / elapsed,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "task_history": task_history,
        "final_accuracy": task_history[-1]["accuracy_after_task"],
        "obsolete_tasks": sorted(obsolete),
        "per_task_forgetting": forgetting,
        "average_retention_forgetting": sum(retention_values) / len(retention_values),
        "ewc_state_mb": ewc.bytes_used() / 2**20 if ewc is not None else 0.0,
        "replay_state_mb": replay.bytes_used() / 2**20 if replay is not None else 0.0,
    }
    if group in ("F", "G", "H", "I"):
        jaccard: dict[str, float] = {}
        for later_index, later in enumerate(active_sets):
            for earlier_index, earlier in enumerate(active_sets[:later_index]):
                union = later | earlier
                key = f"{tasks[earlier_index].name}__{tasks[later_index].name}"
                jaccard[key] = len(later & earlier) / len(union) if union else 1.0
        result["feature_jaccard"] = jaccard
    return result


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    suite = SyntheticTaskSuite()
    tasks = suite.ordered()
    preflight = {
        "groups": list(args.groups),
        "seeds": list(args.seeds),
        "tasks": [task.name for task in tasks],
        "device": str(device),
        "tokens_per_group_per_seed": len(tasks) * args.steps_per_task * args.batch_size * PROMPT_LENGTH,
        "steps_per_group_per_seed": len(tasks) * args.steps_per_task,
        "parameter_parity_groups": list(COMPARABLE_GROUPS),
        "diagnostic_groups": [g for g in args.groups if g not in COMPARABLE_GROUPS],
        "fip_registry": {
            "free_ratio": 0.2,
            "importance_ema": 0.99,
            "stable_quantile": 0.5,
            "shared_min_tasks": 2,
            "free_activate_quantile": 0.25,
            "task_sig_quantile": 0.75,
        },
        "non_ffn_protection": {
            "protection_quantile": args.protection_quantile,
            "protected_scale": args.protected_scale,
            "sticky_group_I": True,
            "per_task_budget_group_I": args.per_task_budget,
        },
    }
    if args.dry_run:
        print(json.dumps(preflight, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        for group in args.groups:
            results.append(train_one_group(
                group, seed=seed, tasks=tasks, args=args, device=device
            ))
            # FIP registers tensor hooks and several baselines retain sizable
            # optimizer state.  Release any reference cycles and cached blocks
            # between independent groups/seeds so a five-seed run is bounded by
            # its largest run rather than cumulative allocator cache growth.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report = {
        "kind": "fip_phase0",
        "protocol": "docs/FIP_RESEARCH_PROTOCOL.md",
        "preflight": preflight,
        "fairness": {
            "same_task_order": True,
            "same_steps_per_task": args.steps_per_task,
            "same_batch_size": args.batch_size,
            "same_current_token_budget": True,
            "same_learning_rate": args.lr,
            "same_weight_decay": args.weight_decay,
            "same_seed_list": list(args.seeds),
            "d_blocked_without_explicit_noncomparable_override": True,
            "g_is_diagnostic_non_comparable": True,
            "h_is_diagnostic_non_comparable": True,
            "i_is_diagnostic_non_comparable": True,
        },
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "source_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                PROJECT_ROOT / "data" / "synthetic_tasks.py",
                PROJECT_ROOT / "experiments" / "baselines.py",
                PROJECT_ROOT / "experiments" / "phase0.py",
                PROJECT_ROOT / "fip" / "model.py",
                PROJECT_ROOT / "fip" / "feature_registry.py",
                PROJECT_ROOT / "fip" / "rollback.py",
                PROJECT_ROOT / "fip" / "non_ffn_protection.py",
                PROJECT_ROOT / "fip" / "optimizer.py",
            )
        },
        "results": results,
    }
    output = args.output_dir / "phase0-results.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(results), **preflight}, indent=2))


if __name__ == "__main__":
    main()
