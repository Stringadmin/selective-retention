from memory_arch.run_decisive import make_trial_specs, run
import random


def test_decisive_runner_records_capacity_and_background_tradeoffs():
    result = run(
        embed_backend="hash",
        scorer_path=None,
        background_facts=2,
        full_capacity=6,
    )

    assert result["config"]["background_facts"] == 2
    assert result["config"]["full_capacity"] == 6
    assert all(detail["igm_mem"] == 6 for detail in result["details"])
    assert all(detail["gate_mem"] == 10 for detail in result["details"])
    assert all(detail["full_fifo_mem"] == 6 for detail in result["details"])
    assert result["summary"]["igm"] == 1.0
    assert result["summary"]["full_slot_recency"] == 1.0
    assert result["summary"]["gate_slot_recency"] == 1.0
    assert result["summary"]["versioned_igm"] == 1.0
    assert result["summary"]["full_slot_history"] == 1.0
    assert result["summary"]["gate_slot_history"] == 1.0
    assert result["summary"]["igm_history"] == 0.0
    assert result["summary"]["versioned_igm_history"] == 1.0
    assert result["metric_stats"]["igm"]["n"] == 5
    assert all(detail["versioned_igm_events"] == 10 for detail in result["details"])
    assert all(detail["versioned_igm_active"] == 6 for detail in result["details"])
    assert all(detail["versioned_igm_current_candidates"] == 1 for detail in result["details"])
    assert result["summary"]["background_igm"] == 1.0
    assert result["summary"]["background_versioned_igm"] == 1.0
    assert result["summary"]["background_full_slot_recency"] == 1.0


def test_decisive_randomized_cases_are_seeded_and_held_out_from_default_chains():
    specs_a = make_trial_specs(random.Random(42), random_trials=12, updates_per_chain=7)
    specs_b = make_trial_specs(random.Random(42), random_trials=12, updates_per_chain=7)

    assert specs_a == specs_b
    assert len(specs_a) == 12
    assert all(len(chain) == 7 for _, _, chain in specs_a)
    assert all(len(set(chain)) == 7 for _, _, chain in specs_a)
