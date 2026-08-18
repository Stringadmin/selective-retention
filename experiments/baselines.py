"""Baseline mechanisms for the frozen Phase 0 continual-learning matrix.

These classes are intentionally small and explicit so their data/compute costs
can be reported beside FIP rather than hidden in a training framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class OnlineEWC:
    """Online EWC using squared training gradients, with no extra data passes.

    The diagonal Fisher estimate is accumulated from the same backward passes
    already required by the task loss.  This preserves the configured number of
    examples and optimizer steps; its additional memory and arithmetic are
    reported by the runner.
    """

    def __init__(self, model: torch.nn.Module, coefficient: float, decay: float = 1.0) -> None:
        if coefficient < 0:
            raise ValueError("EWC coefficient must be non-negative")
        self.model = model
        self.coefficient = coefficient
        self.decay = decay
        self._fisher: dict[str, torch.Tensor] = {}
        self._anchor: dict[str, torch.Tensor] = {}
        self._task_sum: dict[str, torch.Tensor] = {}
        self._steps = 0

    def penalty(self) -> torch.Tensor:
        if not self._fisher:
            return next(self.model.parameters()).new_zeros(())
        penalty = next(self.model.parameters()).new_zeros(())
        for name, parameter in self.model.named_parameters():
            if name in self._fisher:
                penalty = penalty + (self._fisher[name] * (parameter - self._anchor[name]).square()).sum()
        return penalty * (0.5 * self.coefficient)

    @torch.no_grad()
    def accumulate_current_gradients(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None or not parameter.requires_grad:
                continue
            squared = parameter.grad.detach().float().square()
            if name not in self._task_sum:
                self._task_sum[name] = squared.clone()
            else:
                self._task_sum[name].add_(squared)
        self._steps += 1

    @torch.no_grad()
    def end_task(self) -> None:
        if self._steps == 0:
            raise RuntimeError("cannot finalize EWC without training gradients")
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or name not in self._task_sum:
                continue
            current = self._task_sum[name] / self._steps
            if name in self._fisher:
                self._fisher[name].mul_(self.decay).add_(current)
            else:
                self._fisher[name] = current.clone()
            self._anchor[name] = parameter.detach().clone()
        self._task_sum.clear()
        self._steps = 0

    def bytes_used(self) -> int:
        tensors = list(self._fisher.values()) + list(self._anchor.values())
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass
class ReplayBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    replay_examples: int


class ReplayBuffer:
    """Fixed-capacity reservoir buffer with a fixed replay fraction per batch."""

    def __init__(self, capacity: int, replay_fraction: float, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        if not 0.0 <= replay_fraction < 1.0:
            raise ValueError("replay fraction must be in [0, 1)")
        self.capacity = capacity
        self.replay_fraction = replay_fraction
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.inputs = torch.empty((0, 0), dtype=torch.long)
        self.targets = torch.empty((0,), dtype=torch.long)
        self.seen = 0
        self._fractional_replay = 0.0
        self._eligible_size = 0

    @property
    def size(self) -> int:
        return int(self.targets.numel())

    def begin_task(self) -> None:
        """Freeze the old-data sampling boundary for one task.

        Examples added while the task is being trained only become eligible on
        the *next* task.  Otherwise a replay baseline can replay its current
        task within a few steps, which is neither a rehearsal baseline nor the
        protocol's intended 5% old-data comparison.
        """
        self._fractional_replay = 0.0
        self._eligible_size = self.size

    def next_replay_count(self, batch_size: int) -> int:
        """Return a count whose long-run share is exactly ``replay_fraction``.

        Per-batch rounding turns a 5% budget into 6.25% for batch size 32, or
        silently into 0% for batch size 8.  Carrying the fractional remainder
        makes the aggregate count faithful to the frozen 5% rule and still
        keeps every optimizer step's batch size fixed.
        """
        if self._eligible_size == 0:
            return 0
        desired = batch_size * self.replay_fraction + self._fractional_replay
        count = min(int(desired), self._eligible_size)
        self._fractional_replay = desired - count
        return count

    @torch.no_grad()
    def add(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        inputs = inputs.detach().to("cpu", dtype=torch.long)
        targets = targets.detach().to("cpu", dtype=torch.long)
        if self.size == 0:
            self.inputs = torch.empty((0, inputs.shape[1]), dtype=torch.long)
        for row, target in zip(inputs, targets):
            self.seen += 1
            if self.size < self.capacity:
                self.inputs = torch.cat((self.inputs, row.unsqueeze(0)), dim=0)
                self.targets = torch.cat((self.targets, target.reshape(1)), dim=0)
                continue
            replacement = int(torch.randint(0, self.seen, (1,), generator=self.generator))
            if replacement < self.capacity:
                self.inputs[replacement].copy_(row)
                self.targets[replacement].copy_(target)

    def mix(
        self,
        current_inputs: torch.Tensor,
        current_targets: torch.Tensor,
        *,
        total_batch_size: int,
        device: torch.device,
    ) -> ReplayBatch:
        replay_count = total_batch_size - current_targets.numel()
        if replay_count < 0 or replay_count > self._eligible_size:
            raise ValueError("current/replay batch sizes are incompatible with the buffer")
        if replay_count == 0:
            return ReplayBatch(current_inputs, current_targets, 0)
        indices = torch.randint(0, self._eligible_size, (replay_count,), generator=self.generator)
        replay_inputs = self.inputs[indices].to(device)
        replay_targets = self.targets[indices].to(device)
        inputs = torch.cat((current_inputs, replay_inputs), dim=0)
        targets = torch.cat((current_targets, replay_targets), dim=0)
        order = torch.randperm(targets.numel(), generator=self.generator).to(device)
        return ReplayBatch(inputs[order], targets[order], replay_count)

    def bytes_used(self) -> int:
        return self.inputs.numel() * self.inputs.element_size() + self.targets.numel() * self.targets.element_size()
