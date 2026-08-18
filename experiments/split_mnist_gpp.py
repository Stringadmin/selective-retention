"""Split MNIST continual-learning benchmark: GPP vs EWC vs Dense.

The missing "standard benchmark" piece for the GPP project
(``docs/GPP_M0_DISJOINT_SUPPLEMENT.md`` §5).  Real MNIST, five disjoint
binary tasks, class-IL single shared 10-way head, no inference-time task id.

Methods (identical MLP, optimizer budget, steps, batch size, seeds):

- ``dense``: full fine-tune, ordinary AdamW.
- ``ewc``: online EWC (soft Fisher regularization), ordinary AdamW.
- ``gpp``: NonFFNProtector (sticky, per-task budget, include all MLP rows) +
  FeatureMaskedAdamW delta projection.  No Fisher, no replay, no task id.

Metrics per method/seed: average accuracy over all tasks after the last task,
and average forgetting (max accuracy drop per task).  Reported so the GPP
result can be compared against the synthetic disjoint-suite numbers directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.split_mnist import SplitMNIST
from experiments.baselines import OnlineEWC
from fip import FeatureMaskedAdamW, FeatureRegistry, NonFFNProtector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[1337, 42, 7])
    p.add_argument("--methods", nargs="+", default=["dense", "ewc", "gpp"])
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--layers", type=int, default=2, help="number of hidden layers")
    p.add_argument("--steps-per-task", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--ewc-coefficient", type=float, default=400.0)
    p.add_argument("--gpp-quantile", type=float, default=0.5)
    p.add_argument("--gpp-scale", type=float, default=0.0)
    p.add_argument("--gpp-budget", type=float, default=0.2)
    p.add_argument("--gpp-max-frac", type=float, default=0.9)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "split-mnist-gpp.json")
    return p.parse_args()


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_layers: int, n_classes: int = 10):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device, bs: int = 2048) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, x.numel() // x.shape[-1] if x.dim() == 1 else x.shape[0], bs):
            xb = x[i:i + bs].to(device)
            yb = y[i:i + bs].to(device)
            pred = model(xb).argmax(dim=-1)
            correct += int((pred == yb).sum().item())
    return correct / x.shape[0]


def train_one(method: str, seed: int, data: SplitMNIST, args, device) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = MLP(data.input_dim(), args.hidden, args.layers).to(device)

    protector = None
    ewc = None
    if method == "gpp":
        protector = NonFFNProtector(
            model,
            protection_quantile=args.gpp_quantile,
            protected_scale=args.gpp_scale,
            sticky=True,
            max_protected_fraction=args.gpp_max_frac,
            per_task_budget=args.gpp_budget,
            include_ffn=True,  # MLP has no FFN markers; protect ALL linear rows
        )
        optimizer = FeatureMaskedAdamW(
            model.parameters(),
            model=model,
            registry=FeatureRegistry(),  # empty; no FIPFFN layers in an MLP
            non_ffn_protector=protector,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif method == "ewc":
        ewc = OnlineEWC(model, args.ewc_coefficient, decay=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_tasks = data.n_tasks()
    # acc_matrix[t][j] = accuracy on task j measured right after training task t
    acc_matrix: list[list[float | None]] = []
    started = time.perf_counter()

    for t in range(n_tasks):
        task = data.tasks[t]
        if protector is not None:
            protector.begin_task()
        n = task.x_train.shape[0]
        model.train()
        for step in range(args.steps_per_task):
            idx = torch.randint(0, n, (args.batch_size,), generator=gen)
            xb = task.x_train[idx].to(device)
            yb = task.y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if ewc is not None:
                loss.backward()
                ewc.accumulate_current_gradients()
                reg = ewc.penalty()
                if reg.requires_grad:
                    reg.backward()
            else:
                loss.backward()
            if protector is not None:
                protector.update_importance()
            optimizer.step()
        if protector is not None:
            protector.end_task()
        if ewc is not None:
            ewc.end_task()
        row: list[float | None] = []
        for j in range(n_tasks):
            if j <= t:
                row.append(evaluate(model, data.tasks[j].x_test, data.tasks[j].y_test, device))
            else:
                row.append(None)
        acc_matrix.append(row)

    # Final average accuracy over all tasks.
    final = acc_matrix[-1]
    final_accs = [a for a in final if a is not None]
    avg_acc = sum(final_accs) / len(final_accs)
    # Average forgetting: per task, max historical acc minus final acc.
    forgetting_vals = []
    for j in range(n_tasks - 1):  # last task has no forgetting
        hist = [acc_matrix[t][j] for t in range(n_tasks) if acc_matrix[t][j] is not None]
        forgetting_vals.append(max(hist) - hist[-1])
    avg_forgetting = sum(forgetting_vals) / len(forgetting_vals)
    elapsed = time.perf_counter() - started
    protected_frac = protector.protected_fraction() if protector is not None else None
    return {
        "method": method,
        "seed": seed,
        "avg_accuracy": avg_acc,
        "avg_forgetting": avg_forgetting,
        "final_per_task": final,
        "per_task_forgetting": forgetting_vals,
        "acc_matrix": acc_matrix,
        "protected_fraction": protected_frac,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    data = SplitMNIST()
    results = []
    for seed in args.seeds:
        for method in args.methods:
            r = train_one(method, seed, data, args, device)
            results.append(r)
            print(
                f"[{method:5s} seed={seed}] avg_acc={r['avg_accuracy']:.4f} "
                f"avg_forg={r['avg_forgetting']:.4f} prot={r['protected_fraction']} "
                f"({r['elapsed_seconds']:.0f}s)",
                flush=True,
            )
    # Aggregate by method.
    agg: dict[str, dict[str, Any]] = {}
    for r in results:
        m = agg.setdefault(r["method"], {"acc": [], "forg": [], "prot": []})
        m["acc"].append(r["avg_accuracy"])
        m["forg"].append(r["avg_forgetting"])
        if r["protected_fraction"] is not None:
            m["prot"].append(r["protected_fraction"])
    summary = {
        m: {
            "avg_accuracy_mean": sum(v["acc"]) / len(v["acc"]),
            "avg_forgetting_mean": sum(v["forg"]) / len(v["forg"]),
            "avg_accuracy_seeds": v["acc"],
            "avg_forgetting_seeds": v["forg"],
            "protected_fraction_mean": (sum(v["prot"]) / len(v["prot"])) if v["prot"] else None,
        }
        for m, v in agg.items()
    }
    report = {
        "kind": "split_mnist_gpp",
        "benchmark": "Split MNIST (real IDX), 5 disjoint binary tasks, class-IL single head",
        "config": {
            "hidden": args.hidden, "layers": args.layers, "steps_per_task": args.steps_per_task,
            "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
            "ewc_coefficient": args.ewc_coefficient,
            "gpp": {"quantile": args.gpp_quantile, "scale": args.gpp_scale,
                    "budget": args.gpp_budget, "max_frac": args.gpp_max_frac},
            "seeds": args.seeds, "device": str(device), "torch": torch.__version__,
        },
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
