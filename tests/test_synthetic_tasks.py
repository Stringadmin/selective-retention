"""Tests for the controlled Phase 0 synthetic task source."""

import torch

from data.synthetic_tasks import PROMPT_LENGTH, VOCAB_SIZE, SyntheticTaskSuite


def test_task_suite_matches_frozen_phase0_stream():
    assert SyntheticTaskSuite().names == (
        "facts", "conflict", "modular", "string", "shared", "unrelated"
    )


def test_sampling_is_reproducible_for_a_fixed_seed():
    task = SyntheticTaskSuite().get("modular")
    first = task.sample(32, generator=torch.Generator().manual_seed(42))
    second = task.sample(32, generator=torch.Generator().manual_seed(42))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_every_task_emits_valid_fixed_length_token_examples():
    suite = SyntheticTaskSuite()
    for index, task in enumerate(suite.ordered()):
        prompt, answer = task.sample(19, generator=torch.Generator().manual_seed(index))
        assert prompt.shape == (19, PROMPT_LENGTH)
        assert answer.shape == (19,)
        assert int(prompt.min()) >= 0 and int(prompt.max()) < VOCAB_SIZE
        assert int(answer.min()) >= 0 and int(answer.max()) < VOCAB_SIZE


def test_conflict_task_revises_the_same_fact_prompt():
    suite = SyntheticTaskSuite()
    seed = 99
    fact_prompt, fact_answer = suite.get("facts").sample(
        64, generator=torch.Generator().manual_seed(seed)
    )
    conflict_prompt, conflict_answer = suite.get("conflict").sample(
        64, generator=torch.Generator().manual_seed(seed)
    )
    assert torch.equal(fact_prompt, conflict_prompt)
    assert not torch.equal(fact_answer, conflict_answer)
