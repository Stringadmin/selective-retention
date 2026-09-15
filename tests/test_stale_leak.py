from memory_arch.run_stale_leak import identify_new_session, leak_metrics


def _entry(answer="25:50", dates=("2023/01/01 (Sun) 10:00", "2023/02/01 (Wed) 10:00")):
    return {
        "question_id": "q1",
        "question_type": "knowledge-update",
        "question": "What was my personal best time in the charity 5K run?",
        "answer": answer,
        "answer_session_ids": ["s_old", "s_new"],
        "haystack_dates": list(dates),
        "haystack_session_ids": ["s_old", "s_new", "s_other"],
        "haystack_sessions": [
            [{"role": "user", "content": "I set a personal best time of 27:12."}],
            [{"role": "user", "content": "I hope to beat my personal best time of 25:50."}],
            [{"role": "user", "content": "Unrelated chat."}],
        ],
    }


def test_new_session_is_the_one_containing_the_answer():
    new_id, how, gold = identify_new_session(_entry())
    assert (new_id, how) == ("s_new", "answer_text")
    assert gold == ["s_old", "s_new"]


def test_undiscriminating_answer_falls_back_to_the_later_date():
    new_id, how, _ = identify_new_session(_entry(answer="Yes."))
    assert (new_id, how) == ("s_new", "date_order")


def test_single_evidence_entry_cannot_be_scored():
    entry = _entry()
    entry["answer_session_ids"] = ["s_new"]
    assert identify_new_session(entry) == (None, "single_evidence", ["s_new"])


def test_leak_is_the_old_session_outranking_the_new_one():
    assert leak_metrics(["s_old", "s_new"], "s_new", "s_old")["leak@1"] == 1.0
    assert leak_metrics(["s_new", "s_old"], "s_new", "s_old")["leak@1"] == 0.0
    ordered = leak_metrics(["s_new", "s_old"], "s_new", "s_old")
    assert ordered["stale_above_new@1"] == 0.0 and ordered["mrr"] == 1.0
    # A missing current value leaks regardless of where the old one sits.
    missing = leak_metrics(["s_other", "s_old"], "s_new", "s_old")
    assert missing["new_missing@1"] == 1.0 and missing["leak@10"] == 1.0
