"""FIP feature registry: per-layer feature state machine, importance EMA, and masks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import torch


class FeatureState(IntEnum):
    FREE = 0      # reserved capacity, writable, never meaningfully used yet
    PLASTIC = 1   # currently writable by the active task
    SHARED = 2    # cross-task common ability, lightly updatable
    STABLE = 3    # carries important old knowledge, frozen (grad -> 0)


# Gradient multiplier per state.
STATE_GRAD_SCALE = {
    FeatureState.FREE: 1.0,
    FeatureState.PLASTIC: 1.0,
    FeatureState.SHARED: 0.1,   # lr_shared relative scale
    FeatureState.STABLE: 0.0,
}


@dataclass
class LayerRegistry:
    """State for the feature slots of a single FIP-FFN layer."""

    n_features: int
    state: torch.Tensor            # [F] int64, FeatureState
    importance: torch.Tensor       # [F] float32, EMA(|act * grad|)
    usage_count: torch.Tensor      # [F] int64, #tasks with significant use
    version: torch.Tensor          # [F] int64, #times weights were committed
    task_contrib: torch.Tensor     # [F] float32, accumulated |act*grad| this task
    stable_scale: float = 0.0      # ablation: update scale of stable features

    def grad_mask(self) -> torch.Tensor:
        """Return the [F] update-scale mask on the registry's device."""
        mask = torch.ones(
            self.n_features,
            dtype=torch.float32,
            device=self.state.device,
        )
        for st, scale in STATE_GRAD_SCALE.items():
            if st == FeatureState.STABLE:
                scale = self.stable_scale
            mask[self.state == int(st)] = scale
        return mask

    def state_counts(self) -> Dict[str, int]:
        out = {}
        for st in FeatureState:
            out[st.name.lower()] = int((self.state == int(st)).sum().item())
        return out


class FeatureRegistry:
    """Tracks feature states across all FIP-FFN layers in the model."""

    def __init__(
        self,
        free_ratio: float = 0.2,
        importance_ema: float = 0.99,
        stable_quantile: float = 0.5,
        shared_min_tasks: int = 2,
        free_activate_quantile: float = 0.25,
        task_sig_quantile: float = 0.75,
        stable_scale: float = 0.0,
        allocation_mode: str = "importance",
    ) -> None:
        for name, value in {
            "free_ratio": free_ratio,
            "importance_ema": importance_ema,
            "stable_quantile": stable_quantile,
            "free_activate_quantile": free_activate_quantile,
            "task_sig_quantile": task_sig_quantile,
            "stable_scale": stable_scale,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if shared_min_tasks < 1:
            raise ValueError("shared_min_tasks must be >= 1")
        if allocation_mode not in ("importance", "random"):
            raise ValueError(
                f"allocation_mode must be 'importance' or 'random', got {allocation_mode}"
            )
        self.free_ratio = free_ratio
        self.importance_ema = importance_ema
        self.stable_quantile = stable_quantile
        self.shared_min_tasks = shared_min_tasks
        self.free_activate_quantile = free_activate_quantile
        self.task_sig_quantile = task_sig_quantile
        self.stable_scale = stable_scale
        self.allocation_mode = allocation_mode
        self.layers: Dict[int, LayerRegistry] = {}
        self.current_task: int = -1

    # ------------------------------------------------------------------ setup
    def register_layer(self, layer_id: int, n_features: int, device, dtype) -> LayerRegistry:
        n_free = int(round(n_features * self.free_ratio))
        state = torch.full((n_features,), int(FeatureState.PLASTIC), dtype=torch.int64, device=device)
        state[:n_free] = int(FeatureState.FREE)
        reg = LayerRegistry(
            n_features=n_features,
            state=state,
            importance=torch.zeros(n_features, dtype=torch.float32, device=device),
            usage_count=torch.zeros(n_features, dtype=torch.int64, device=device),
            version=torch.zeros(n_features, dtype=torch.int64, device=device),
            task_contrib=torch.zeros(n_features, dtype=torch.float32, device=device),
            stable_scale=self.stable_scale,
        )
        self.layers[layer_id] = reg
        return reg

    def _apply(self, fn) -> "FeatureRegistry":
        """Apply a ``nn.Module._apply``-style transform to registry tensors.

        The registry deliberately is not an ``nn.Module``: it is bookkeeping
        shared by several FFN modules and must not appear in a model state
        dict.  It still has to follow the model when ``model.to(device)`` or a
        dtype conversion is used, otherwise mask construction can cross
        devices during a training step.
        """
        for reg in self.layers.values():
            reg.state = fn(reg.state)
            reg.importance = fn(reg.importance)
            reg.usage_count = fn(reg.usage_count)
            reg.version = fn(reg.version)
            reg.task_contrib = fn(reg.task_contrib)
        return self

    def to(self, *args, **kwargs) -> "FeatureRegistry":
        """Move registry tensors using the same semantics as ``Tensor.to``."""
        return self._apply(lambda tensor: tensor.to(*args, **kwargs))

    # --------------------------------------------------------------- training
    def begin_task(self) -> int:
        """Start a new task. Returns the new task id."""
        self.current_task += 1
        for reg in self.layers.values():
            reg.task_contrib.zero_()
        return self.current_task

    def update_importance(self, layer_id: int, contrib: torch.Tensor) -> None:
        """EMA update of importance from |act * grad| averaged over batch/time."""
        reg = self.layers[layer_id]
        contrib = contrib.to(reg.importance.device, dtype=torch.float32)
        reg.importance.mul_(self.importance_ema).add_(contrib * (1.0 - self.importance_ema))
        reg.task_contrib.add_(contrib)

    def _significant_mask(self, reg: LayerRegistry) -> torch.Tensor:
        """Boolean mask of features significantly used in the current task.

        Only features whose per-task accumulated contribution exceeds the
        ``task_sig_quantile`` quantile among non-zero contributors are
        considered *significantly used*.  This replaces the old
        "activated at least once" rule, which over 2 000 Top-K steps marked
        nearly every slot as touched and promoted almost everything to
        ``shared``.
        """
        tc = reg.task_contrib
        nz = tc > 0
        if not nz.any():
            return torch.zeros_like(tc, dtype=torch.bool)
        thr = torch.quantile(tc[nz], self.task_sig_quantile)
        return nz & (tc >= thr)

    def _random_choice(
        self, reg: LayerRegistry, pool: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Uniformly pick ``n`` slots from ``pool`` (no replacement)."""
        idx = torch.nonzero(pool, as_tuple=False).squeeze(-1)
        if n >= idx.numel():
            return pool
        perm = torch.randperm(idx.numel(), device=reg.state.device)
        chosen = idx[perm[:n]]
        out = torch.zeros_like(pool)
        out[chosen] = True
        return out

    def end_task(self) -> None:
        """Promote feature states at task boundary based on accumulated importance."""
        for reg in self.layers.values():
            significant = self._significant_mask(reg)
            reg.usage_count[significant] += 1

            imp = reg.importance
            nz = imp > 0
            if nz.any():
                thr_stable = torch.quantile(imp[nz], self.stable_quantile)
            else:
                thr_stable = float("inf")
            if nz.any():
                thr_free_act = torch.quantile(imp[nz], self.free_activate_quantile)
            else:
                thr_free_act = float("inf")

            new_state = reg.state.clone()
            # High importance -> stable (task-specific knowledge), fully frozen.
            # Stability takes precedence over sharing: a feature that carries
            # important old knowledge must not be demoted to "shared" (lightly
            # updatable) just because a later task happens to touch it, or that
            # later task would slowly overwrite the earlier knowledge.
            high = (imp >= thr_stable) & (imp > 0)
            # Ablation #5 (allocation_mode="random"): keep the exact same
            # number of promoted slots as the importance ranking would choose,
            # but pick them uniformly at random.  This isolates the value of
            # *intelligent* feature allocation while holding the state
            # distribution fixed.
            if self.allocation_mode == "random":
                high = self._random_choice(reg, nz, int(high.sum().item()))
            new_state[high] = int(FeatureState.STABLE)
            # A feature once frozen as stable stays stable: the importance EMA
            # decays over later tasks, so re-evaluating it would demote an old
            # feature and let a later task overwrite it.  Stability is sticky.
            already_stable = reg.state == int(FeatureState.STABLE)
            new_state[already_stable] = int(FeatureState.STABLE)
            # Low-importance features reused across >= min_tasks -> shared
            # (cross-task common ability, lightly updatable).
            reused = reg.usage_count >= self.shared_min_tasks
            shared = reused & (~high) & (~already_stable)
            new_state[shared] = int(FeatureState.SHARED)
            # Free features that started being used -> promote to plastic.
            promoted = (reg.state == int(FeatureState.FREE)) & (imp >= thr_free_act) & (~high) & (~already_stable) & (~shared)
            if self.allocation_mode == "random":
                free_pool = (reg.state == int(FeatureState.FREE)) & (~high) & (~already_stable) & (~shared)
                promoted = self._random_choice(reg, free_pool, int(promoted.sum().item()))
            new_state[promoted] = int(FeatureState.PLASTIC)
            reg.state.copy_(new_state)
            reg.task_contrib.zero_()

    def commit_version(self, layer_id: int, changed_mask: torch.Tensor) -> None:
        """Increment version for features whose weights changed this step."""
        reg = self.layers[layer_id]
        reg.version[changed_mask] += 1

    # ----------------------------------------------------------------- query
    def grad_mask(self, layer_id: int) -> torch.Tensor:
        return self.layers[layer_id].grad_mask()

    def state_counts(self) -> Dict[int, Dict[str, int]]:
        return {lid: reg.state_counts() for lid, reg in self.layers.items()}

    def significant_indices(self, layer_id: int) -> torch.Tensor:
        """Indices of features significantly used in the current (not yet ended) task."""
        reg = self.layers[layer_id]
        return torch.nonzero(self._significant_mask(reg), as_tuple=False).squeeze(-1)

    def all_stable_indices(self, layer_id: int) -> torch.Tensor:
        reg = self.layers[layer_id]
        return torch.nonzero(reg.state == int(FeatureState.STABLE), as_tuple=False).squeeze(-1)

    def dead_feature_ratio(self, layer_id: int, threshold: float = 0.0) -> float:
        """Fraction of features with negligible lifetime importance."""
        reg = self.layers[layer_id]
        dead = (reg.importance <= threshold).sum().item()
        return dead / reg.n_features
