"""Split MNIST continual-learning benchmark loader (real IDX data).

Five binary tasks over disjoint digit pairs (0-1, 2-3, 4-5, 6-7, 8-9), the
standard Split MNIST protocol.  Uses the IDX files already downloaded into
this directory (no network access at run time).

Two evaluation protocols are supported:

- ``task_il``: each task has its own 2-way head (needs a task id at test time,
  the easier protocol).
- ``class_il``: a single shared 10-way head, no task id (the harder, and the
  one relevant to GPP/FIP, which forbid inference-time task ids).
"""

from __future__ import annotations

import gzip
import os
import struct
from dataclasses import dataclass

import torch

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist")

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def _read_images(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    assert magic == 0x803, f"bad image magic {magic:#x} in {path}"
    arr = torch.frombuffer(bytearray(data[16:]), dtype=torch.uint8)
    return arr.reshape(n, rows * cols).float() / 255.0


def _read_labels(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n = struct.unpack(">II", data[:8])
    assert magic == 0x801, f"bad label magic {magic:#x} in {path}"
    return torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8).long()


@dataclass
class TaskData:
    """One Split MNIST task's tensors."""

    x_train: torch.Tensor
    y_train: torch.Tensor   # original digit labels (class-IL target space)
    x_test: torch.Tensor
    y_test: torch.Tensor
    digits: tuple[int, int]


class SplitMNIST:
    """Loads and splits the real MNIST IDX files into five disjoint tasks."""

    def __init__(self, directory: str | None = None) -> None:
        d = directory or _DIR
        xtr = _read_images(os.path.join(d, "train-images-idx3-ubyte.gz"))
        ytr = _read_labels(os.path.join(d, "train-labels-idx1-ubyte.gz"))
        xte = _read_images(os.path.join(d, "t10k-images-idx3-ubyte.gz"))
        yte = _read_labels(os.path.join(d, "t10k-labels-idx1-ubyte.gz"))
        self.tasks: list[TaskData] = []
        for a, b in TASKS:
            tr = (ytr == a) | (ytr == b)
            te = (yte == a) | (yte == b)
            self.tasks.append(TaskData(xtr[tr], ytr[tr], xte[te], yte[te], (a, b)))

    def n_tasks(self) -> int:
        return len(self.tasks)

    def input_dim(self) -> int:
        return 28 * 28


if __name__ == "__main__":
    sm = SplitMNIST()
    for i, t in enumerate(sm.tasks):
        print(
            f"task {i} digits={t.digits} train={t.x_train.shape[0]} "
            f"test={t.x_test.shape[0]} label_range=({int(t.y_train.min())},{int(t.y_train.max())})"
        )
