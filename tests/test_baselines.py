"""Unit tests for Phase 0 comparison baselines."""

import torch

from experiments.baselines import OnlineEWC, ReplayBuffer


def test_online_ewc_penalizes_departure_from_task_snapshot():
    model = torch.nn.Linear(3, 2, bias=False)
    ewc = OnlineEWC(model, coefficient=10.0)
    loss = model(torch.ones(4, 3)).square().mean()
    loss.backward()
    ewc.accumulate_current_gradients()
    ewc.end_task()
    assert ewc.penalty().item() == 0.0
    with torch.no_grad():
        model.weight.add_(0.1)
    assert ewc.penalty().item() > 0.0
    assert ewc.bytes_used() > 0


def test_replay_buffer_has_fixed_capacity_and_fixed_mixed_batch_size():
    buffer = ReplayBuffer(capacity=5, replay_fraction=0.25, seed=4)
    inputs = torch.arange(24).view(6, 4)
    targets = torch.arange(6)
    buffer.add(inputs, targets)
    assert buffer.size == 5
    buffer.begin_task()
    assert [buffer.next_replay_count(8) for _ in range(5)] == [2, 2, 2, 2, 2]
    fractional = ReplayBuffer(capacity=5, replay_fraction=0.05, seed=4)
    fractional.add(inputs, targets)
    fractional.begin_task()
    assert [fractional.next_replay_count(8) for _ in range(5)] == [0, 0, 1, 0, 1]
    current_inputs = torch.full((6, 4), 99, dtype=torch.long)
    current_targets = torch.full((6,), 99, dtype=torch.long)
    mixed = buffer.mix(
        current_inputs, current_targets, total_batch_size=8, device=torch.device("cpu")
    )
    assert mixed.inputs.shape == (8, 4)
    assert mixed.targets.shape == (8,)
    assert mixed.replay_examples == 2


def test_replay_does_not_use_examples_added_during_the_current_task():
    buffer = ReplayBuffer(capacity=10, replay_fraction=0.5, seed=0)
    buffer.begin_task()
    buffer.add(torch.ones(4, 3, dtype=torch.long), torch.ones(4, dtype=torch.long))
    assert buffer.next_replay_count(4) == 0
    buffer.begin_task()
    assert buffer.next_replay_count(4) == 2
