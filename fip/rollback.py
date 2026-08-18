"""Rollback manager: snapshot and restore feature increments per task.

Semantics: ``checkpoint_before(task_id)`` records the model + registry state
immediately *before* a task is trained. ``rollback(task_id)`` restores that
snapshot, which exactly undoes the task's updates (logits recovery < 1e-6).

v0 stores full per-layer weight clones for correctness. The interface is
structured so the storage can later be swapped to per-feature deltas without
changing call sites.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch

from .feature_registry import FeatureRegistry
from .sparse_ffn import FIPFFN


class RollbackManager:
    def __init__(self, model: torch.nn.Module, registry: FeatureRegistry) -> None:
        self.model = model
        self.registry = registry
        self.ffn_layers: List[FIPFFN] = [
            m for m in model.modules() if isinstance(m, FIPFFN)
        ]
        self._snapshots: Dict[int, dict] = {}

    @torch.no_grad()
    def checkpoint_before(
        self, task_id: int, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """Snapshot full model, registry, and optionally optimizer state.

        v0 stores full clones for exact recovery (<1e-6). The interface allows
        later swapping to per-feature deltas without changing call sites.
        """
        snap: dict = {
            "model": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "registry": {},
            "current_task": self.registry.current_task,
        }
        if optimizer is not None:
            snap["optimizer"] = copy.deepcopy(optimizer.state_dict())
        for lid, reg in self.registry.layers.items():
            snap["registry"][lid] = {
                "state": reg.state.detach().clone(),
                "importance": reg.importance.detach().clone(),
                "usage_count": reg.usage_count.detach().clone(),
                "version": reg.version.detach().clone(),
                "task_contrib": reg.task_contrib.detach().clone(),
            }
        self._snapshots[task_id] = snap

    @torch.no_grad()
    def rollback(
        self,
        task_id: int,
        controller,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Restore the pre-task snapshot, undoing that task's updates.

        Pass the same optimizer used for training whenever the task is going
        to continue after rollback.  Its momentum and variance estimates are
        then restored along with the model, preventing hidden post-rollback
        traces of the reverted task.
        """
        if task_id not in self._snapshots:
            raise KeyError(f"no checkpoint for task {task_id}")
        snap = self._snapshots[task_id]
        self.model.load_state_dict(snap["model"])
        for lid, tensors in snap["registry"].items():
            reg = self.registry.layers[lid]
            reg.state.copy_(tensors["state"])
            reg.importance.copy_(tensors["importance"])
            reg.usage_count.copy_(tensors["usage_count"])
            reg.version.copy_(tensors["version"])
            reg.task_contrib.copy_(tensors["task_contrib"])
        self.registry.current_task = snap["current_task"]
        if optimizer is not None and "optimizer" in snap:
            optimizer.load_state_dict(snap["optimizer"])
        # Rebuild masks so frozen/restored states take effect.
        controller._rebuild_all_masks()

    @torch.no_grad()
    def recovery_error(self, forward_fn, logits_ref: torch.Tensor) -> float:
        """Max abs error between current model output and reference logits.

        ``forward_fn`` is a zero-arg callable returning the model logits on the
        same inputs used to produce ``logits_ref``. Call after ``rollback``.
        """
        logits = forward_fn()
        return (logits - logits_ref).abs().max().item()
