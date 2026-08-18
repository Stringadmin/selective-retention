"""Optimizer-level feature isolation for FIP-FFN.

Gradient hooks are necessary for importance accounting, but they cannot make a
feature immutable under optimizers such as AdamW: decoupled weight decay and
old momentum can still change a slice whose current gradient is zero.  This
optimizer enforces the state mask on the *actual parameter delta* after the
base AdamW update.

When a ``NonFFNProtector`` is supplied, the same post-step delta projection
is applied to non-FFN parameters (embedding, attention, norms) using the
protector's per-row importance scales.  This extends FIP's graded protection
philosophy beyond the FFN without gradient hooks, so weight decay and Adam
momentum are also constrained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .feature_registry import FeatureRegistry, FeatureState
from .non_ffn_protection import NonFFNProtector
from .sparse_ffn import FIPFFN


@dataclass(frozen=True)
class _FeatureParameter:
    """How a FIP feature axis is laid out in one parameter tensor."""

    layer_id: int
    feature_dim: int


class FeatureMaskedAdamW(torch.optim.AdamW):
    """AdamW that makes FIP feature states constrain real updates.

    ``stable`` slots retain their exact pre-step values, including against
    decoupled weight decay and historic Adam momentum.  ``shared`` slots keep
    exactly 0.1 of the AdamW delta; ``plastic`` and ``free`` slots keep all of
    it.  Non-FIP parameters use ordinary AdamW behavior unchanged, unless a
    ``non_ffn_protector`` is supplied, in which case their deltas are also
    projected onto the protector's per-row scales.

    Parameters
    ----------
    params:
        Standard AdamW parameter iterable.
    model:
        The module containing FIPFFN layers.  It is inspected once during
        construction to identify their three feature-axis parameters.
    registry:
        Optional explicit registry.  Defaults to ``model.registry``.
    non_ffn_protector:
        Optional ``NonFFNProtector`` for graded protection of embedding,
        attention, and norm parameters.

    The post-step delta projection keeps a full clone of each FIP parameter
    for one optimizer step.  That is intentional for Phase 0 correctness; a
    slice-only implementation can replace it after the mechanism gates pass.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        model: torch.nn.Module,
        registry: Optional[FeatureRegistry] = None,
        non_ffn_protector: Optional[NonFFNProtector] = None,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.registry = registry if registry is not None else getattr(model, "registry", None)
        if self.registry is None:
            raise ValueError("FeatureMaskedAdamW needs a FeatureRegistry")

        managed_parameters = {
            parameter for group in self.param_groups for parameter in group["params"]
        }
        self._feature_params: dict[torch.nn.Parameter, _FeatureParameter] = {}
        for ffn in model.modules():
            if not isinstance(ffn, FIPFFN):
                continue
            if ffn.w_gate in managed_parameters:
                self._feature_params[ffn.w_gate] = _FeatureParameter(ffn.layer_id, 0)
            if ffn.w_up in managed_parameters:
                self._feature_params[ffn.w_up] = _FeatureParameter(ffn.layer_id, 0)
            if ffn.w_down in managed_parameters:
                self._feature_params[ffn.w_down] = _FeatureParameter(ffn.layer_id, 1)

        if not self._feature_params and non_ffn_protector is None:
            raise ValueError(
                "FeatureMaskedAdamW requires at least one FIPFFN layer or a "
                "non_ffn_protector"
            )

        self.non_ffn_protector = non_ffn_protector
        self._non_ffn_params: list[torch.nn.Parameter] = []
        if non_ffn_protector is not None:
            self._non_ffn_params = [
                p for p in managed_parameters
                if p in non_ffn_protector._param_names
            ]

    def _expanded_mask(
        self, parameter: torch.nn.Parameter, spec: _FeatureParameter
    ) -> torch.Tensor:
        mask = self.registry.grad_mask(spec.layer_id).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return mask.unsqueeze(-1) if spec.feature_dim == 0 else mask.unsqueeze(0)

    def _clear_stable_optimizer_state(
        self,
        parameter: torch.nn.Parameter,
        spec: _FeatureParameter,
    ) -> None:
        """Drop momentum for frozen slots so a later unfreeze cannot jump.

        Skipped when ``stable_scale > 0`` (ablation "no stable freeze"):
        the slots are trainable, so clearing their moments would corrupt
        ordinary AdamW behavior instead of isolating it.
        """
        if self.registry.stable_scale > 0:
            return
        stable = (
            self.registry.layers[spec.layer_id].state.to(parameter.device)
            == int(FeatureState.STABLE)
        )
        if not stable.any():
            return
        stable = stable.unsqueeze(-1) if spec.feature_dim == 0 else stable.unsqueeze(0)
        for value in self.state[parameter].values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value.masked_fill_(stable, 0)

    @torch.no_grad()
    def step(self, closure=None):
        """Run AdamW, then project FIP and non-FFN updates onto their scales."""
        before = {
            parameter: parameter.detach().clone(memory_format=torch.preserve_format)
            for parameter in self._feature_params
        }
        non_ffn_before = {
            parameter: parameter.detach().clone(memory_format=torch.preserve_format)
            for parameter in self._non_ffn_params
        } if self._non_ffn_params else {}
        loss = super().step(closure)
        changed_by_layer: dict[int, torch.Tensor] = {}
        for parameter, spec in self._feature_params.items():
            mask = self._expanded_mask(parameter, spec)
            parameter.copy_(before[parameter] + mask * (parameter - before[parameter]))
            delta = parameter - before[parameter]
            changed = delta.ne(0).any(dim=1 - spec.feature_dim)
            if spec.layer_id not in changed_by_layer:
                changed_by_layer[spec.layer_id] = torch.zeros_like(changed, dtype=torch.bool)
            changed_by_layer[spec.layer_id].logical_or_(changed)
            self._clear_stable_optimizer_state(parameter, spec)
        for layer_id, changed in changed_by_layer.items():
            self.registry.commit_version(layer_id, changed)
        for parameter in self._non_ffn_params:
            scale = self.non_ffn_protector.get_scale(parameter)
            if scale is None:
                continue
            scale = scale.to(parameter.device, dtype=parameter.dtype)
            # Broadcast the per-out-row scale over all trailing dimensions:
            # [out] -> [out, 1] for linear, [out, 1, 1, 1] for conv2d, etc.
            if parameter.dim() > 1:
                scale = scale.reshape(scale.shape[0], *([1] * (parameter.dim() - 1)))
            parameter.copy_(
                non_ffn_before[parameter] + scale * (parameter - non_ffn_before[parameter])
            )
        return loss
