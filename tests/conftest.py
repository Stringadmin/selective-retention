"""Pytest config: make the project importable and pick a device."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    os.environ.setdefault("FIP_DEVICE", "cuda")


@pytest.fixture(scope="session")
def device():
    import torch
    d = "cuda" if torch.cuda.is_available() else "cpu"
    return d
