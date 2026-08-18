"""Importance-driven update scaling for non-FFN parameters.

The FFN-boundary diagnostic (group G) showed that freezing all non-FFN
parameters after task 1 reduces forgetting by 38.8% but drops new-task
plasticity to 86.6% of F.  This module provides *graded* protection instead
of a binary freeze: parameters with high accumulated importance receive a
reduced update scale (default 0.1, like FIP's shared features), while
unimportant parameters remain fully plastic.

Two protection modes are supported:

- **Non-sticky** (default): protection scales are recomputed from scratch at
  each ``end_task()``.  A row that was important in an earlier task can lose
  its protection if its importance EMA decays relative to newer tasks.
- **Sticky** (group I): the protected set is cumulative across tasks — once a
  row is marked protected it stays protected.  Two capacity mechanisms are
  available:

  - *Fixed cap* (``per_task_budget is None``): ``max_protected_fraction`` is a
    global admission limit.  The first task that fills it blocks all later
    admissions, which was shown to preserve the wrong rows (diagnostic I).
  - *Per-task budget* (``per_task_budget`` set): each task may admit at most
    ``per_task_budget`` *fraction* of the parameter's rows as *newly*
    protected rows, selected from the rows that row was important for during
    that task (per-task contribution, not lifetime EMA).  Capacity grows with
    task count as ``min(max_protected_fraction, per_task_budget * n_tasks)``,
    so every task gets a chance to protect its own knowledge without the
    first task monopolizing the whole budget.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class NonFFNProtector:
    """Per-row importance tracking and protection-scale computation for non-FFN params.

    Parameters are identified by name: anything that is *not* ``w_gate``,
    ``w_up``, ``w_down``, or ``lora_`` is tracked.  Importance is an EMA of
    the mean absolute gradient per output row (dimension 0 of the weight
    matrix, or the full vector for 1-D parameters like norms).

    Call ``update_importance()`` after ``loss.backward()`` and before
    ``optimizer.step()``.  Call ``end_task()`` at the task boundary to
    recompute protection scales for the next task.

    When ``sticky=True``, the protected set accumulates across tasks and is
    capped by ``max_protected_fraction`` (fixed) or by a per-task admission
    budget.  A row that was important in an earlier task remains protected
    even if its EMA decays, preventing protected-set turnover.
    """

    def __init__(
        self,
        model: nn.Module,
        protection_quantile: float = 0.5,
        protected_scale: float = 0.1,
        importance_ema: float = 0.99,
        sticky: bool = False,
        max_protected_fraction: float = 0.5,
        per_task_budget: Optional[float] = None,
        token_ownership: bool = False,
        include_ffn: bool = False,
    ) -> None:
        if not 0.0 <= protection_quantile <= 1.0:
            raise ValueError("protection_quantile must be in [0, 1]")
        if not 0.0 <= protected_scale <= 1.0:
            raise ValueError("protected_scale must be in [0, 1]")
        if not 0.0 <= importance_ema <= 1.0:
            raise ValueError("importance_ema must be in [0, 1]")
        if not 0.0 <= max_protected_fraction <= 1.0:
            raise ValueError("max_protected_fraction must be in [0, 1]")
        if per_task_budget is not None and not 0.0 <= per_task_budget <= 1.0:
            raise ValueError("per_task_budget must be in [0, 1]")
        self.model = model
        self.protection_quantile = protection_quantile
        self.protected_scale = protected_scale
        self.importance_ema = importance_ema
        self.sticky = sticky
        self.max_protected_fraction = max_protected_fraction
        self.per_task_budget = per_task_budget
        self.token_ownership = token_ownership
        self.include_ffn = include_ffn
        self.importance: Dict[str, torch.Tensor] = {}
        self.task_contrib: Dict[str, torch.Tensor] = {}
        self.scales: Dict[str, torch.Tensor] = {}
        self.protected_mask: Dict[str, torch.Tensor] = {}
        self._prev_protected: Dict[str, torch.Tensor] = {}
        self._param_names: Dict[torch.nn.Parameter, str] = {}
        self._is_1d: Dict[str, bool] = {}
        self._n_tasks = 0
        self._register(model)
        # Token-ownership bookkeeping (embedding rows owned by exactly one task).
        self._task_tokens: set[int] = set()
        self._token_task_count: Dict[int, int] = {}
        self._frozen_tokens: set[int] = set()
        self._emb_param_name: Optional[str] = (
            "tok_emb.weight" if "tok_emb.weight" in self.importance else None
        )

    # ------------------------------------------------------------------ setup
    def _register(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if "lora_" in name:
                continue
            if not self.include_ffn and any(k in name for k in ("w_gate", "w_up", "w_down")):
                continue
            if not p.requires_grad:
                continue
            out_dim = p.shape[0]
            self.importance[name] = torch.zeros(out_dim, device=p.device, dtype=torch.float32)
            self.task_contrib[name] = torch.zeros(out_dim, device=p.device, dtype=torch.float32)
            self.scales[name] = torch.ones(out_dim, device=p.device, dtype=p.dtype)
            self.protected_mask[name] = torch.zeros(out_dim, dtype=torch.bool, device=p.device)
            self._prev_protected[name] = torch.zeros(out_dim, dtype=torch.bool, device=p.device)
            self._param_names[p] = name
            self._is_1d[name] = p.dim() == 1

    def to(self, *args, **kwargs) -> "NonFFNProtector":
        for name in self.importance:
            self.importance[name] = self.importance[name].to(*args, **kwargs)
            self.task_contrib[name] = self.task_contrib[name].to(*args, **kwargs)
            self.scales[name] = self.scales[name].to(*args, **kwargs)
            self.protected_mask[name] = self.protected_mask[name].to(*args, **kwargs)
            self._prev_protected[name] = self._prev_protected[name].to(*args, **kwargs)
        return self

    # --------------------------------------------------------------- training
    def begin_task(self) -> None:
        """Save current protected mask for turnover measurement."""
        for name in self.protected_mask:
            self._prev_protected[name] = self.protected_mask[name].clone()
            self.task_contrib[name].zero_()
        if self.token_ownership:
            self._task_tokens = set()

    def record_task_tokens(self, inputs: torch.Tensor) -> None:
        """Record the token ids seen by the current task (for ownership).

        Call once per batch (or at least once per task) before ``end_task``.
        Tokens that appear in exactly one task own their embedding row.
        """
        if not self.token_ownership or self._emb_param_name is None:
            return
        tokens = inputs.detach().cpu().reshape(-1).unique().tolist()
        self._task_tokens.update(tokens)

    def update_importance(self) -> None:
        """EMA-update importance from current gradients.

        Call after ``loss.backward()``, before ``optimizer.step()``.
        """
        for p, name in self._param_names.items():
            if p.grad is None:
                continue
            grad_abs = p.grad.detach().abs()
            if p.dim() > 1:
                contrib = grad_abs.mean(dim=tuple(range(1, p.dim())))
            else:
                contrib = grad_abs
            imp = self.importance[name]
            imp.mul_(self.importance_ema).add_(contrib.float() * (1.0 - self.importance_ema))
            self.task_contrib[name].add_(contrib.float())

    def end_task(self) -> None:
        """Recompute protection scales at the task boundary.

        In sticky mode, newly-important rows are admitted into the cumulative
        ``protected_mask`` only while capacity remains.  Existing protected
        rows are never evicted, so the mask is genuinely monotonic.  In
        non-sticky mode, the mask is replaced.

        Two sticky capacity policies:
        - ``per_task_budget=None``: a global ``max_protected_fraction`` cap.
        - ``per_task_budget`` set: capacity grows as
          ``min(max_protected_fraction, per_task_budget * n_tasks)`` and new
          admissions are chosen from rows significant in the *current* task.
        """
        self._n_tasks += 1
        for name, imp in self.importance.items():
            nz = imp > 0
            if nz.any():
                threshold = torch.quantile(imp[nz], self.protection_quantile)
                new_protected = nz & (imp >= threshold)
            else:
                new_protected = torch.zeros_like(imp, dtype=torch.bool)

            if self.sticky and self._is_1d[name]:
                # 1-D parameters (norms) are global per-channel scalers: every
                # channel feeds every task, so leaving any channel fully
                # plastic lets a later task rescale the whole representation
                # and break all earlier tasks.  Once a norm carries any
                # knowledge it gets a uniform reduced scale for its WHOLE
                # vector (sticky): no partial per-channel admission.
                if imp.sum() > 0 or self.task_contrib[name].sum() > 0:
                    self.protected_mask[name].fill_(True)
                self.scales[name] = torch.full_like(
                    self.scales[name], self.protected_scale,
                    dtype=self.scales[name].dtype,
                )
                continue
            elif self.sticky:
                max_n = int(self.protected_mask[name].numel() * self.max_protected_fraction)
                current = self.protected_mask[name]
                n_protected = int(current.sum().item())

                if self.per_task_budget is not None:
                    # Capacity grows with the number of tasks so a later task
                    # is not locked out by an early one that filled the cap.
                    budget_frac = min(self.max_protected_fraction,
                                      self.per_task_budget * self._n_tasks)
                    max_n = max(max_n, int(self.protected_mask[name].numel() * budget_frac))
                    # New admissions come from rows significant in the current
                    # task (per-task contribution), not the lifetime EMA which
                    # is dominated by the first task.
                    tc = self.task_contrib[name]
                    tc_nz = tc > 0
                    if tc_nz.any():
                        tc_thr = torch.quantile(tc[tc_nz], self.protection_quantile)
                        new_protected = tc_nz & (tc >= tc_thr)

                if max_n <= 0:
                    # A zero-capacity sticky protector cannot admit any rows.
                    current.zero_()
                elif n_protected > max_n:
                    # Do not silently evict rows if capacity is changed after
                    # admission: that would violate the sticky contract.
                    raise RuntimeError(
                        f"sticky protected set for {name} exceeds capacity "
                        f"({n_protected} > {max_n})"
                    )
                else:
                    candidates = new_protected & ~current
                    remaining = max_n - n_protected
                    n_candidates = int(candidates.sum().item())
                    if n_candidates <= remaining:
                        current.logical_or_(candidates)
                    elif remaining > 0:
                        # Rank only new candidates. Existing protected rows
                        # are never eligible for replacement.
                        scores = imp.masked_fill(~candidates, float("-inf"))
                        _, top_idx = scores.topk(remaining)
                        current[top_idx] = True
            else:
                self.protected_mask[name] = new_protected

            self.scales[name] = torch.where(
                self.protected_mask[name],
                torch.full_like(imp, self.protected_scale, dtype=self.scales[name].dtype),
                torch.ones_like(self.scales[name]),
            )

        if self.token_ownership and self._emb_param_name is not None:
            # Embedding rows owned by exactly one task are fully frozen (scale
            # 0) so a later task can never overwrite a token's meaning.  This
            # closes the leak where tasks sharing surface tokens (e.g. string
            # vs unrelated both using SOURCE_BASE tokens) rewrite each other's
            # input embeddings.  Shared tokens (used by multiple tasks) keep
            # their graded protection and remain adaptively useful.  Once a
            # token is frozen it stays frozen even if a later task reuses it.
            for tok in self._task_tokens:
                self._token_task_count[tok] = self._token_task_count.get(tok, 0) + 1
            emb = self.importance[self._emb_param_name]
            n_emb = emb.numel()
            exclusive = [
                tok for tok, count in self._token_task_count.items()
                if count == 1 and 0 <= tok < n_emb
            ]
            self._frozen_tokens.update(exclusive)
            if self._frozen_tokens:
                frozen = torch.zeros_like(emb, dtype=torch.bool)
                frozen[list(self._frozen_tokens)] = True
                emb_mask = self.protected_mask[self._emb_param_name]
                emb_mask.logical_or_(frozen)
                scale = self.scales[self._emb_param_name]
                scale.masked_fill_(frozen, 0.0)

    # ----------------------------------------------------------------- query
    def get_scale(self, parameter: torch.nn.Parameter) -> torch.Tensor | None:
        """Return the [out_dim] protection scale for a parameter, or None."""
        name = self._param_names.get(parameter)
        if name is None:
            return None
        return self.scales[name]

    def protected_fraction(self) -> float:
        """Fraction of tracked rows currently under protection."""
        total = 0
        protected = 0
        for name, mask in self.protected_mask.items():
            total += mask.numel()
            protected += int(mask.sum().item())
        return protected / total if total else 0.0

    def turnover(self) -> float:
        """Fraction of previously-protected rows that lost protection this task.

        Computed as ``1 - |prev ∩ curr| / |prev|`` averaged over all tracked
        parameters.  Call after ``end_task()``.
        """
        total_prev = 0
        total_lost = 0
        for name in self.protected_mask:
            prev = self._prev_protected[name]
            curr = self.protected_mask[name]
            n_prev = int(prev.sum().item())
            n_kept = int((prev & curr).sum().item())
            total_prev += n_prev
            total_lost += n_prev - n_kept
        return total_lost / total_prev if total_prev else 0.0

    def protected_set_summary(self) -> dict[str, float]:
        """Per-module-type protected fraction for diagnostics."""
        summary: dict[str, float] = {"embedding_head": 0.0, "attention": 0.0, "norms": 0.0}
        counts: dict[str, int] = {"embedding_head": 0, "attention": 0, "norms": 0}
        for name, mask in self.protected_mask.items():
            if "tok_emb" in name or "head" in name:
                key = "embedding_head"
            elif "attn" in name:
                key = "attention"
            elif "norm" in name:
                key = "norms"
            else:
                continue
            summary[key] += float(mask.sum().item())
            counts[key] += mask.numel()
        for key in summary:
            if counts[key] > 0:
                summary[key] /= counts[key]
        return summary
