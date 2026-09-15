from memory_arch.run_reader_eval import (
    MAX_TURN_CHARS,
    build_context,
    chronological,
    score,
    select_evidence,
)


def _entry():
    return {
        "question_id": "q1",
        "question_type": "knowledge-update",
        "question": "What is my current X?",
        "answer": "B",
        "answer_session_ids": ["s_old", "s_new"],
        "haystack_dates": ["2023/01/01 (Sun) 10:00", "2023/02/01 (Wed) 10:00", "2023/03/01 (Wed) 10:00"],
        "haystack_session_ids": ["s_old", "s_new", "s_other"],
        "haystack_sessions": [
            [{"role": "user", "content": "my X is A"}, {"role": "assistant", "content": "ok"}],
            [{"role": "user", "content": "my X is now B"}],
            [{"role": "user", "content": "unrelated"}],
        ],
    }


def test_ranked_and_chrono_arms_share_the_evidence_but_not_the_order():
    entry = _entry()
    ranked = [("s_new", 0, "my X is now B"), ("s_old", 0, "my X is A")]
    assert select_evidence(entry, "ranked_k8", ranked, k=2) == ranked
    assert select_evidence(entry, "chrono_k8", ranked, k=2) == [
        ("s_old", 0, "my X is A"), ("s_new", 0, "my X is now B")]


def test_oracle_arm_reads_the_labeled_sessions_in_date_order():
    turns = select_evidence(_entry(), "oracle", ranked=[], k=8)
    assert [session_id for session_id, _, _ in turns] == ["s_old", "s_new"]
    assert [text for _, _, text in turns] == ["my X is A", "my X is now B"]


def test_reverse_arm_flips_the_same_evidence_set():
    entry = _entry()
    ranked = [("s_new", 0, "my X is now B"), ("s_old", 0, "my X is A")]
    assert select_evidence(entry, "reverse_k8", ranked, k=2) == [
        ("s_new", 0, "my X is now B"), ("s_old", 0, "my X is A")]
    assert set(select_evidence(entry, "reverse_k8", ranked, k=2)) == set(
        select_evidence(entry, "chrono_k8", ranked, k=2))


def test_current_and_stale_arms_isolate_one_side_of_the_update():
    entry = _entry()
    current = select_evidence(entry, "current_only", ranked=[], k=8, new_id="s_new",
                              old_ids=["s_old"])
    stale = select_evidence(entry, "stale_only", ranked=[], k=8, new_id="s_new",
                            old_ids=["s_old"])
    assert [text for _, _, text in current] == ["my X is now B"]
    assert [text for _, _, text in stale] == ["my X is A"]


def test_assistant_turns_are_not_evidence():
    assert all("ok" not in text for _, _, text in select_evidence(_entry(), "oracle", [], 8))


def test_chronological_uses_session_date_then_turn_index():
    entry = _entry()
    turns = [("s_other", 0, "c"), ("s_new", 0, "b"), ("s_old", 1, "a2"), ("s_old", 0, "a1")]
    assert chronological(entry, turns) == [
        ("s_old", 0, "a1"), ("s_old", 1, "a2"), ("s_new", 0, "b"), ("s_other", 0, "c")]


def test_build_context_numbers_and_truncates_turns():
    long_turn = "x" * (MAX_TURN_CHARS + 50)
    context = build_context([("s", 0, long_turn), ("s", 1, "short")])
    lines = context.splitlines()
    assert lines[0].startswith("[1] ") and lines[0].endswith("…")
    assert len(lines[0]) == len("[1] ") + MAX_TURN_CHARS + 1
    assert lines[1] == "[2] short"


def test_containment_scoring_coerces_numeric_gold():
    assert score("The amount was $400,000.", 400000)
    assert score("400000", 400000)
    assert not score("$350,000", 400000)
