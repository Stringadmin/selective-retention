import pytest

from memory_arch.outcome import OutcomeMemory
from memory_arch.context_router import ContextRoute, SemanticContextRouter
from memory_arch.run_outcome_bandit import run as run_bandit
from memory_arch.run_outcome_drift import run
from memory_arch.run_outcome_poison import run as run_poison
from memory_arch.run_semantic_transfer import EVALUATION_QUERIES, _queries_for, _selective_routing
from memory_arch.longmemeval_retrieval import (
    evaluate_entries,
    needs_assistant_history,
    rank_sessions,
    retrieval_metrics,
    select_partition,
    session_documents,
)
from memory_arch import Embedder
from memory_arch.run_longmemeval_recency_sweep import select_weight


def test_outcome_memory_keeps_an_audit_log_and_updates_evidence():
    memory = OutcomeMemory(decay_rate=0.0)
    first = memory.record("deploy", "canary", True, observed_at=1.0)
    second = memory.record("deploy", "rollback", False, observed_at=2.0)
    third = memory.record("deploy", "canary", False, observed_at=3.0)

    canary = memory.evidence("deploy", "canary")
    rollback = memory.evidence("deploy", "rollback")
    assert memory.event_count == 3
    assert [event.event_id for event in memory.history("deploy")] == [
        first.event_id,
        second.event_id,
        third.event_id,
    ]
    assert canary.success_weight == 1.0
    assert canary.failure_weight == 1.0
    assert rollback.posterior_mean < canary.posterior_mean
    assert memory.recommend("deploy", ["canary", "rollback"]) == "canary"


def test_semantic_context_router_separates_routing_from_outcome_storage():
    class ToyEmbedder:
        vectors = {
            "发布故障": [1.0, 0.0],
            "数据库迁移": [0.0, 1.0],
            "部署以后服务报错": [0.9, 0.1],
        }

        def embed(self, text):
            return self.vectors[text]

    router = SemanticContextRouter(
        ToyEmbedder(), {"release": "发布故障", "database": "数据库迁移"}
    )
    route = router.route("部署以后服务报错")

    assert route.context_id == "release"
    assert route.similarity > 0.8
    assert route.runner_up_context_id == "database"
    assert route.margin is not None and route.margin > 0.7


def test_selective_routing_reports_coverage_and_rejected_misroutes():
    routes = {
        "clear": ContextRoute("release", 0.90, "database", 0.50),
        "ambiguous-and-wrong": ContextRoute("database", 0.61, "release", 0.60),
        "ambiguous-but-right": ContextRoute("release", 0.70, "database", 0.69),
    }
    expected = {
        "clear": "release",
        "ambiguous-and-wrong": "release",
        "ambiguous-but-right": "release",
    }

    summary = _selective_routing(routes, expected, min_margin=0.02)

    assert summary["accepted"] == 1
    assert summary["abstained"] == 2
    assert summary["automatic_coverage"] == pytest.approx(1 / 3)
    assert summary["accepted_precision"] == 1.0
    assert summary["abstained_incorrect"] == 1
    assert summary["abstained_examples"][0]["correct"] is False


def test_semantic_evaluation_probes_are_disjoint_from_calibration_probes():
    calibration = _queries_for("calibration")
    evaluation = _queries_for("evaluation")

    assert evaluation == EVALUATION_QUERIES
    assert set(calibration) == set(evaluation)
    assert all(set(calibration[key]).isdisjoint(evaluation[key]) for key in calibration)


def test_longmemeval_retrieval_uses_labeled_sessions_and_skips_abstention():
    class ToyEmbedder:
        def embed(self, text):
            return [float("target" in text), float("other" in text)]

    answer = {
        "question_id": "q1",
        "question_type": "knowledge-update",
        "question": "target question",
        "answer_session_ids": ["answer_session"],
        "haystack_session_ids": ["distractor", "answer_session"],
        "haystack_sessions": [
            [{"role": "user", "content": "other fact"}],
            [{"role": "assistant", "content": "target fact"}],
        ],
    }
    abstention = {
        "question_id": "q2_abs",
        "question_type": "single-session-user",
        "question": "missing question",
        "answer_session_ids": [],
        "haystack_session_ids": ["distractor"],
        "haystack_sessions": [[{"role": "user", "content": "other fact"}]],
    }

    documents = session_documents(answer)
    result = evaluate_entries([answer, abstention], ToyEmbedder(), top_ks=(1,))

    assert documents[1] == ("answer_session", "assistant: target fact")
    assert result["scope"]["instances_scored"] == 1
    assert result["scope"]["skipped_abstention"] == 1
    assert result["scope"]["recency_weight"] == 0.0
    assert result["overall"]["recall_any@1"] == 1.0
    assert result["by_question_type"]["knowledge-update"]["recall_all@1"] == 1.0


def test_user_only_longmemeval_documents_keep_assistant_only_session_ids():
    entry = {
        "haystack_session_ids": ["assistant_only"],
        "haystack_sessions": [[{"role": "assistant", "content": "reply only"}]],
    }

    assert session_documents(entry, include_assistant=False) == [("assistant_only", "")]


def test_assistant_history_router_requires_explicit_prior_assistant_wording():
    assert needs_assistant_history("What did you recommend last time?")
    assert needs_assistant_history("We outlined this in our previous conversation.")
    assert not needs_assistant_history("What is my current preference for coffee?")
    assert not needs_assistant_history("What was my previous marathon time?")


def test_role_aware_retrieval_reads_assistant_turns_only_for_routed_question():
    class ToyEmbedder:
        def embed(self, text):
            return [
                float("recommend" in text or "assistant fact" in text),
                float("preference" in text or "user fact" in text),
            ]

    assistant_question = {
        "question_id": "assistant-question",
        "question_type": "single-session-assistant",
        "question": "What did you recommend last time?",
        "answer_session_ids": ["assistant-session"],
        "haystack_session_ids": ["assistant-session", "user-session"],
        "haystack_sessions": [
            [{"role": "assistant", "content": "assistant fact"}],
            [{"role": "user", "content": "user fact"}],
        ],
    }
    user_question = {
        **assistant_question,
        "question_id": "user-question",
        "question_type": "single-session-user",
        "question": "What is my preference?",
        "answer_session_ids": ["user-session"],
    }

    result = evaluate_entries(
        [assistant_question, user_question],
        ToyEmbedder(),
        top_ks=(1,),
        role_aware=True,
    )

    assert result["scope"]["assistant_history_routes"] == 1
    assert result["scope"]["user_history_routes"] == 1
    assert result["scope"]["retrieval_mode"] == "role-aware"
    assert result["overall"]["recall_all@1"] == 1.0


def test_longmemeval_retrieval_metrics_require_all_evidence_sessions_for_recall_all():
    metrics = retrieval_metrics(
        ["answer_one", "distractor", "answer_two"],
        ["answer_one", "answer_two"],
        top_ks=(1, 3),
    )

    assert metrics["recall_any@1"] == 1.0
    assert metrics["recall_all@1"] == 0.0
    assert metrics["recall_all@3"] == 1.0


def test_hash_embedder_batch_matches_scalar_embeddings():
    embedder = Embedder(dim=8)
    texts = ["alpha beta", "beta gamma"]

    assert embedder.embed_many(texts) == [embedder.embed(text) for text in texts]


def test_recency_prior_breaks_semantic_ties_without_deleting_older_sessions():
    class FlatEmbedder:
        def embed(self, _):
            return [1.0]

    documents = [("old", "old fact"), ("recent", "recent fact")]

    assert rank_sessions("question", documents, FlatEmbedder()) == ["old", "recent"]
    assert rank_sessions("question", documents, FlatEmbedder(), recency_weight=0.1) == [
        "recent", "old"
    ]


def test_longmemeval_partition_is_stable_and_disjoint():
    entries = [{"question_id": f"q{i}"} for i in range(30)]
    development = select_partition(entries, "development")
    test = select_partition(entries, "test")

    assert {entry["question_id"] for entry in development}.isdisjoint(
        entry["question_id"] for entry in test
    )
    assert development + test != entries  # Partition order differs from source order.
    assert {entry["question_id"] for entry in development + test} == {
        entry["question_id"] for entry in entries
    }


def test_recency_sweep_prefers_the_smallest_weight_when_scores_tie():
    reports = {
        0.0: {"overall": {"recall_all@10": 0.70}},
        0.02: {"overall": {"recall_all@10": 0.75}},
        0.05: {"overall": {"recall_all@10": 0.75}},
    }

    assert select_weight(reports) == 0.02


def test_recent_feedback_can_overcome_stale_evidence_without_deleting_history():
    memory = OutcomeMemory(decay_rate=1.0)
    memory.record("release", "fast", True, observed_at=1.0)
    memory.record("release", "fast", True, observed_at=2.0)
    memory.record("release", "safe", False, observed_at=3.0)
    memory.record("release", "fast", False, observed_at=10.0)
    memory.record("release", "safe", True, observed_at=11.0)

    assert memory.recommend("release", ["fast", "safe"]) == "safe"
    assert len(memory.history("release")) == 5
    assert memory.history("release", "fast")[0].succeeded is True


def test_outcome_memory_rejects_out_of_order_events():
    memory = OutcomeMemory()
    memory.record("task", "a", True, observed_at=2.0)
    with pytest.raises(ValueError, match="time order"):
        memory.record("task", "b", True, observed_at=1.0)


def test_low_trust_feedback_is_retained_but_has_less_effect_on_evidence():
    memory = OutcomeMemory(decay_rate=0.0)
    memory.record("deploy", "canary", True, observed_at=1.0, provenance="test", evidence_weight=1.0)
    low_trust = memory.record(
        "deploy", "canary", False, observed_at=2.0,
        provenance="unverified_report", evidence_weight=0.1,
    )

    evidence = memory.evidence("deploy", "canary")
    assert low_trust.provenance == "unverified_report"
    assert low_trust.evidence_weight == 0.1
    assert evidence.success_weight == 1.0
    assert evidence.failure_weight == 0.1
    with pytest.raises(ValueError, match="evidence_weight"):
        memory.record("deploy", "canary", True, evidence_weight=0.0)


def test_drift_benchmark_is_seeded_and_shows_adaptation_tradeoffs():
    result = run(seeds=40, contexts=12, stationary_rounds=8, drift_rounds=8)
    summary = result["summary"]

    assert summary["outcome_memory"]["stationary_accuracy"] > 0.8
    assert summary["outcome_memory"]["post_drift_final_accuracy"] > 0.8
    assert summary["outcome_memory"]["post_drift_final_accuracy"] > (
        summary["all_history_mean"]["post_drift_final_accuracy"] + 0.2
    )
    assert summary["outcome_memory"] == summary["decayed_evidence_baseline"]
    assert result["resources"]["outcome_memory_mean_event_records"] == 12 * 2 * 16


def test_partial_feedback_bandit_keeps_equivalence_and_adapts_after_drift():
    result = run_bandit(
        seeds=80,
        contexts=12,
        stationary_rounds=12,
        drift_rounds=16,
        exploration_rate=0.10,
    )
    summary = result["summary"]

    assert summary["outcome_memory"] == summary["decayed_evidence_baseline"]
    assert summary["outcome_memory"]["stationary_action_accuracy"] > 0.7
    assert summary["outcome_memory"]["post_drift_final_action_accuracy"] > 0.7
    assert summary["outcome_memory"]["post_drift_final_action_accuracy"] > (
        summary["all_history_mean"]["post_drift_final_action_accuracy"] + 0.2
    )
    assert result["resources"]["outcome_memory_mean_event_records"] == 12 * 28


def test_source_weighted_evidence_resists_labeled_corrupted_feedback():
    result = run_poison(
        seeds=80,
        contexts=12,
        stationary_rounds=12,
        drift_rounds=16,
        corruption_rate=0.30,
        untrusted_weight=0.05,
    )
    summary = result["summary"]

    assert (summary["source_weighted_outcome_memory"]
            == summary["source_weighted_evidence_baseline"])
    assert summary["source_weighted_outcome_memory"]["post_drift_final_action_accuracy"] > 0.7
    assert summary["source_weighted_outcome_memory"]["post_drift_final_action_accuracy"] > (
        summary["unweighted_decayed_evidence"]["post_drift_final_action_accuracy"] + 0.1
    )
    assert result["resources"]["source_weighted_outcome_memory_mean_event_records"] == 12 * 28
