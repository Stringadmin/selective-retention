"""Minimal decoder-only Transformer for FIP Phase 0.

Uses RMSNorm, RoPE, causal self-attention, and either DenseFFN or FIPFFN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_ffn import DenseFFN, FIPFFN
from .feature_registry import FeatureRegistry
from .lora import LoRADelta


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    d_model: int = 256
    d_ff: int = 1024
    n_head: int = 4
    context: int = 256
    top_k: int = 64
    free_ratio: float = 0.2
    importance_ema: float = 0.99
    stable_quantile: float = 0.5
    shared_min_tasks: int = 2
    free_activate_quantile: float = 0.25
    task_sig_quantile: float = 0.75
    stable_scale: float = 0.0        # ablation: update scale of stable features
    allocation_mode: str = "importance"  # ablation: "importance" | "random"
    use_fip: bool = True
    dtype: torch.dtype = torch.float32


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B, H, T, Dh]; cos/sin: [T, Dh]
    q2 = (q * cos) + (rotate_half(q) * sin)
    k2 = (k * cos) + (rotate_half(k) * sin)
    return q2, k2


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                       # [T, D/2]
        cos = freqs.cos().repeat_interleave(2, dim=-1)         # [T, D]
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:T].to(device), self.sin[:T].to(device)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.lora_qkv: LoRADelta | None = None
        self.lora_proj: LoRADelta | None = None
        self.rope = RoPE(self.head_dim, cfg.context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv_weight = self.qkv.weight if self.lora_qkv is None else self.qkv.weight + self.lora_qkv.delta()
        qkv = F.linear(x, qkv_weight, self.qkv.bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(T, x.device)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        proj_weight = self.proj.weight if self.lora_proj is None else self.proj.weight + self.lora_proj.delta()
        return F.linear(out, proj_weight, self.proj.bias)

    def enable_lora(self, rank: int, alpha: float | None = None) -> None:
        if self.lora_qkv is not None:
            raise RuntimeError("LoRA is already enabled for this attention block")
        self.lora_qkv = LoRADelta(
            self.qkv.out_features, self.qkv.in_features, rank, alpha,
            device=self.qkv.weight.device, dtype=self.qkv.weight.dtype,
        )
        self.lora_proj = LoRADelta(
            self.proj.out_features, self.proj.in_features, rank, alpha,
            device=self.proj.weight.device, dtype=self.proj.weight.dtype,
        )

    @torch.no_grad()
    def merge_lora_(self) -> None:
        if self.lora_qkv is None:
            return
        self.lora_qkv.merge_into_(self.qkv.weight)
        self.lora_proj.merge_into_(self.proj.weight)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, registry: FeatureRegistry, layer_id: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        if cfg.use_fip:
            self.ffn = FIPFFN(cfg.d_model, cfg.d_ff, cfg.top_k, registry, layer_id,
                              dtype=cfg.dtype)
        else:
            self.ffn = DenseFFN(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FIPTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.registry = FeatureRegistry(
            free_ratio=cfg.free_ratio,
            importance_ema=cfg.importance_ema,
            stable_quantile=cfg.stable_quantile,
            shared_min_tasks=cfg.shared_min_tasks,
            free_activate_quantile=cfg.free_activate_quantile,
            task_sig_quantile=cfg.task_sig_quantile,
            stable_scale=cfg.stable_scale,
            allocation_mode=cfg.allocation_mode,
        )
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, self.registry, i) for i in range(cfg.n_layer)]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._lora_enabled = False
        # weight tying
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.head(x)

    def enable_lora(self, rank: int, alpha: float | None = None) -> None:
        """Enable task-local adapters for the dense, cumulative-LoRA baseline.

        FIP has its own feature-level update mechanism and is intentionally not
        combined with LoRA in the frozen Phase 0 matrix.
        """
        if self.cfg.use_fip:
            raise ValueError("cumulative LoRA baseline requires use_fip=False")
        if self._lora_enabled:
            raise RuntimeError("LoRA is already enabled for this model")
        for block in self.blocks:
            block.attn.enable_lora(rank, alpha)
            assert isinstance(block.ffn, DenseFFN)
            block.ffn.enable_lora(rank, alpha)
        # ``head.weight`` is tied to ``tok_emb.weight``.  Merging an output-head
        # adapter into it would also change every input embedding and break the
        # merge-equivalence invariant, so the cumulative baseline adapts
        # attention and FFN projections only.
        self._lora_enabled = True
        for name, parameter in self.named_parameters():
            parameter.requires_grad_("lora_" in name)

    def lora_parameters(self):
        """Yield only adapter parameters after ``enable_lora``."""
        if not self._lora_enabled:
            raise RuntimeError("LoRA is not enabled")
        return (parameter for name, parameter in self.named_parameters() if "lora_" in name)

    @torch.no_grad()
    def merge_lora_(self) -> None:
        """Commit each task adapter, then reset it without changing outputs."""
        if not self._lora_enabled:
            raise RuntimeError("LoRA is not enabled")
        for block in self.blocks:
            block.attn.merge_lora_()
            assert isinstance(block.ffn, DenseFFN)
            block.ffn.merge_lora_()

    def _apply(self, fn):
        """Move registry tensors and cached FIP masks with the model."""
        super()._apply(fn)
        self.registry._apply(fn)
        for block in self.blocks:
            if isinstance(block.ffn, FIPFFN):
                block.ffn.rebuild_mask()
        return self

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n
