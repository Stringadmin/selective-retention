"""FIP sparse FFN with Top-K activation and feature-level gradient isolation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_registry import FeatureRegistry
from .lora import LoRADelta


class DenseFFN(nn.Module):
    """Standard SwiGLU FFN (group A baseline)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_gate = nn.Parameter(torch.empty(d_ff, d_model))
        self.w_up = nn.Parameter(torch.empty(d_ff, d_model))
        self.w_down = nn.Parameter(torch.empty(d_model, d_ff))
        self.lora_gate: LoRADelta | None = None
        self.lora_up: LoRADelta | None = None
        self.lora_down: LoRADelta | None = None
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.w_gate, std=std)
        nn.init.normal_(self.w_up, std=std)
        nn.init.normal_(self.w_down, std=std)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w_gate = self.w_gate if self.lora_gate is None else self.w_gate + self.lora_gate.delta()
        w_up = self.w_up if self.lora_up is None else self.w_up + self.lora_up.delta()
        w_down = self.w_down if self.lora_down is None else self.w_down + self.lora_down.delta()
        gate = F.silu(h @ w_gate.t())
        up = h @ w_up.t()
        act = gate * up
        return act @ w_down.t()

    def enable_lora(self, rank: int, alpha: float | None = None) -> None:
        """Attach zero-output adapters to all three FFN feature matrices."""
        if self.lora_gate is not None:
            raise RuntimeError("LoRA is already enabled for this FFN")
        factory = {"device": self.w_gate.device, "dtype": self.w_gate.dtype}
        self.lora_gate = LoRADelta(self.d_ff, self.d_model, rank, alpha, **factory)
        self.lora_up = LoRADelta(self.d_ff, self.d_model, rank, alpha, **factory)
        self.lora_down = LoRADelta(self.d_model, self.d_ff, rank, alpha, **factory)

    @torch.no_grad()
    def merge_lora_(self) -> None:
        """Merge task adapters into base weights and reset them for the next task."""
        if self.lora_gate is None:
            return
        self.lora_gate.merge_into_(self.w_gate)
        self.lora_up.merge_into_(self.w_up)
        self.lora_down.merge_into_(self.w_down)


class FIPFFN(nn.Module):
    """Top-K sparse SwiGLU FFN with per-feature plasticity masking.

    Each feature slot j corresponds to row j of w_gate/w_up and column j of
    w_down. Freezing feature j zeroes the gradient on all three weight slices.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        top_k: int,
        registry: FeatureRegistry,
        layer_id: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.registry = registry
        self.layer_id = layer_id

        factory = {"device": device, "dtype": dtype}
        self.w_gate = nn.Parameter(torch.empty(d_ff, d_model, **factory))
        self.w_up = nn.Parameter(torch.empty(d_ff, d_model, **factory))
        self.w_down = nn.Parameter(torch.empty(d_model, d_ff, **factory))
        self._init_weights()

        self.registry.register_layer(layer_id, d_ff, device, dtype)
        self._mask = self.registry.grad_mask(layer_id)
        self._register_grad_hooks()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.w_gate, std=std)
        nn.init.normal_(self.w_up, std=std)
        nn.init.normal_(self.w_down, std=std)

    def rebuild_mask(self) -> torch.Tensor:
        self._mask = self.registry.grad_mask(self.layer_id)
        return self._mask

    def _register_grad_hooks(self):
        """Zero/scale gradients for stable/shared feature slots after backward.

        Stable features are fully frozen (scale 0); shared features are lightly
        updatable (scale lr_shared). Plastic and free features are fully
        trainable. The mask is [F] and applied along the feature dimension.
        """

        def gate_hook(_p):
            if self.w_gate.grad is not None:
                self.w_gate.grad.mul_(self._mask.to(self.w_gate.grad).unsqueeze(-1))

        def up_hook(_p):
            if self.w_up.grad is not None:
                self.w_up.grad.mul_(self._mask.to(self.w_up.grad).unsqueeze(-1))

        def down_hook(_p):
            if self.w_down.grad is not None:
                self.w_down.grad.mul_(self._mask.to(self.w_down.grad).unsqueeze(0))

        self.w_gate.register_post_accumulate_grad_hook(gate_hook)
        self.w_up.register_post_accumulate_grad_hook(up_hook)
        self.w_down.register_post_accumulate_grad_hook(down_hook)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gate = F.silu(h @ self.w_gate.t())          # [B, T, F]
        up = h @ self.w_up.t()                       # [B, T, F]
        act = gate * up                              # [B, T, F]

        # Top-K sparse activation (keeps K largest features per position).
        vals, idx = act.topk(self.top_k, dim=-1)
        sparse = torch.zeros_like(act).scatter(-1, idx, vals)

        # Importance EMA: capture |act * grad| during backward (training only).
        if self.training and act.requires_grad:
            # Do not close over ``act`` itself: a Tensor -> hook -> closure ->
            # Tensor cycle retains its entire autograd graph across optimizer
            # steps and eventually exhausts GPU memory.  The detached values
            # are all the importance statistic needs.
            act_abs = act.detach().abs()

            def imp_hook(grad):
                with torch.no_grad():
                    contrib = (act_abs * grad.detach().abs())
                    contrib = contrib.float().mean(dim=tuple(range(contrib.dim() - 1)))
                    self.registry.update_importance(self.layer_id, contrib)

            act.register_hook(imp_hook)

        out = sparse @ self.w_down.t()             # [B, T, F] @ [F, D] -> [B, T, D]
        return out

    def weight_slices(self, j: int):
        """Return the three weight slices for feature j (for inspection/rollback)."""
        return self.w_gate.data[j], self.w_up.data[j], self.w_down.data[:, j]
