"""Plasticity controller: coordinates task boundaries and gradient isolation."""

from __future__ import annotations

import torch

from .feature_registry import FeatureRegistry
from .sparse_ffn import FIPFFN


class PlasticityController:
    """Wraps a model's FIP-FFN layers and the shared FeatureRegistry.

    The training loop calls begin_task()/end_task() at task boundaries. After
    every state change the controller rebuilds the per-layer gradient masks so
    that the registered grad hooks enforce isolation automatically.
    """

    def __init__(self, model: torch.nn.Module, registry: FeatureRegistry) -> None:
        self.model = model
        self.registry = registry
        self.ffn_layers: list[FIPFFN] = [
            m for m in model.modules() if isinstance(m, FIPFFN)
        ]

    def begin_task(self) -> int:
        tid = self.registry.begin_task()
        self._rebuild_all_masks()
        return tid

    def end_task(self) -> None:
        self.registry.end_task()
        self._rebuild_all_masks()

    def _rebuild_all_masks(self) -> None:
        for ffn in self.ffn_layers:
            ffn.rebuild_mask()

    @torch.no_grad()
    def freeze_features(self, layer_id: int, indices: torch.Tensor) -> None:
        """Manually force features to stable (used by tests / ablations)."""
        reg = self.registry.layers[layer_id]
        reg.state[indices] = 3  # FeatureState.STABLE
        for ffn in self.ffn_layers:
            if ffn.layer_id == layer_id:
                ffn.rebuild_mask()

    def summary(self) -> dict:
        return {
            "task": self.registry.current_task,
            "layers": self.registry.state_counts(),
        }
