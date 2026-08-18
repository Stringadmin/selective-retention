"""Minimal mergeable LoRA delta used by the frozen Phase 0 D baseline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRADelta(nn.Module):
    """A rank-``r`` additive matrix delta ``scale * B @ A``.

    ``B`` starts at zero, so attaching an adapter cannot change a model's
    output.  At a task boundary ``merge_into_`` commits the learned delta into
    the base matrix and resets the adapter, yielding one task-ID-free model for
    subsequent training and inference.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        alpha: float | None = None,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        factory = {"device": device, "dtype": dtype}
        self.A = nn.Parameter(torch.empty(rank, in_features, **factory))
        self.B = nn.Parameter(torch.empty(out_features, rank, **factory))
        self.reset_parameters()

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def delta(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale

    @torch.no_grad()
    def merge_into_(self, base_weight: torch.Tensor) -> None:
        if tuple(base_weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"base shape {tuple(base_weight.shape)} does not match LoRA "
                f"shape {(self.out_features, self.in_features)}"
            )
        base_weight.add_(self.delta().to(base_weight))
        self.reset_parameters()
