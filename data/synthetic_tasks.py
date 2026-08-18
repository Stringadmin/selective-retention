"""Controlled synthetic tasks for the FIP Phase 0 learning stream.

The suite deliberately uses a tiny fixed vocabulary and a final-token
classification objective.  Every example is generated from a deterministic
rule, so no hidden corpus, tokenizer choice, or sampling shuffle can explain a
between-group result.  The Phase 0 smoke only checks pipeline integrity; the
frozen protocol still requires the configured 6-layer, five-seed experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch


VOCAB_SIZE = 256
PROMPT_LENGTH = 4

_BOS = 1
_FACT_QUERY = 2
_MOD_QUERY = 3
_STRING_QUERY = 4
_SHARED_QUERY = 5
_UNRELATED_QUERY = 6
_ATTRIBUTE = 7
_STRING_OP = 8

_ENTITY_BASE = 32
_NUMBER_BASE = 64
_VALUE_BASE = 96
_SOURCE_BASE = 128
_STRING_VALUE_BASE = 160
_WIDTH = 16


@dataclass(frozen=True)
class SyntheticTask:
    """One infinite, deterministic distribution in the continual stream."""

    name: str
    kind: str
    supersedes: tuple[str, ...] = ()

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([B, 4] prompt tokens, [B] answer token)``.

        Random draws occur on CPU for generator portability.  The final tensors
        are then moved to the requested training device.
        """
        first = torch.randint(0, _WIDTH, (batch_size,), generator=generator)
        second = torch.randint(0, _WIDTH, (batch_size,), generator=generator)
        bos = torch.full((batch_size,), _BOS, dtype=torch.long)

        if self.kind == "facts":
            prompt = torch.stack(
                (bos, torch.full_like(bos, _FACT_QUERY), _ENTITY_BASE + first,
                 torch.full_like(bos, _ATTRIBUTE)),
                dim=-1,
            )
            answer = _VALUE_BASE + ((first * 7 + 3) % _WIDTH)
        elif self.kind == "conflict":
            # Same fact query and entities as ``facts``, deliberately revised.
            prompt = torch.stack(
                (bos, torch.full_like(bos, _FACT_QUERY), _ENTITY_BASE + first,
                 torch.full_like(bos, _ATTRIBUTE)),
                dim=-1,
            )
            answer = _VALUE_BASE + ((first * 7 + 8) % _WIDTH)
        elif self.kind == "modular":
            prompt = torch.stack(
                (bos, torch.full_like(bos, _MOD_QUERY), _NUMBER_BASE + first,
                 _NUMBER_BASE + second),
                dim=-1,
            )
            answer = _NUMBER_BASE + ((first + second) % _WIDTH)
        elif self.kind == "string":
            prompt = torch.stack(
                (bos, torch.full_like(bos, _STRING_QUERY), _SOURCE_BASE + first,
                 torch.full_like(bos, _STRING_OP)),
                dim=-1,
            )
            answer = _STRING_VALUE_BASE + ((first * 5 + 1) % _WIDTH)
        elif self.kind == "shared":
            # Different surface marker, same compositional addition rule.
            prompt = torch.stack(
                (bos, torch.full_like(bos, _SHARED_QUERY), _NUMBER_BASE + first,
                 _NUMBER_BASE + second),
                dim=-1,
            )
            answer = _NUMBER_BASE + ((first + second) % _WIDTH)
        elif self.kind == "unrelated":
            prompt = torch.stack(
                (bos, torch.full_like(bos, _UNRELATED_QUERY), _SOURCE_BASE + first,
                 torch.full_like(bos, _ATTRIBUTE)),
                dim=-1,
            )
            answer = _VALUE_BASE + ((first * 11 + 4) % _WIDTH)
        else:
            raise ValueError(f"unsupported synthetic task kind: {self.kind}")

        return prompt.to(device), answer.to(device)


class SyntheticTaskSuite:
    """Named, ordered task stream specified by the frozen Phase 0 protocol."""

    def __init__(self) -> None:
        self._tasks: Dict[str, SyntheticTask] = {
            "facts": SyntheticTask("facts", "facts"),
            "conflict": SyntheticTask("conflict", "conflict", supersedes=("facts",)),
            "modular": SyntheticTask("modular", "modular"),
            "string": SyntheticTask("string", "string"),
            "shared": SyntheticTask("shared", "shared"),
            "unrelated": SyntheticTask("unrelated", "unrelated"),
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def get(self, name: str) -> SyntheticTask:
        try:
            return self._tasks[name]
        except KeyError as exc:
            raise KeyError(f"unknown task {name!r}; expected one of {self.names}") from exc

    def ordered(self, names: Iterable[str] | None = None) -> list[SyntheticTask]:
        names = self.names if names is None else names
        return [self.get(name) for name in names]
