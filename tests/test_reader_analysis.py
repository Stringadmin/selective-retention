from memory_arch.run_reader_eval import analyze_discourse_order, analyze_gold_position


class FakeEmbedder:
    """Deterministic stand-in: similarity follows token overlap with the question."""

    def embed(self, text):
        words = set(text.lower().split())
        return [float(word in words) for word in ("question", "alpha", "beta", "gold")] or [0.0]


def _entry():
    return {
        "question_id": "q1",
        "question_type": "knowledge-update",
        "question": "question alpha",
        "answer": "gold",
        "answer_session_ids": ["s_old", "s_new"],
        "haystack_dates": ["2023/01/01 (Sun) 10:00", "2023/02/01 (Wed) 10:00"],
        "haystack_session_ids": ["s_old", "s_new", "s_other"],
        "haystack_sessions": [
            [{"role": "user", "content": "alpha beta"}],
            [{"role": "user", "content": "alpha gold"}],
            [{"role": "user", "content": "beta"}],
        ],
    }


def _report(errors):
    return {"errors": {"knowledge_update": [{"question_id": q, "arm": a} for q, a in errors]},
            "config": {"arms": ["chrono_k8"], "k": 2}}


def _analyze(analyzer, errors=()):
    return analyzer([_entry()], FakeEmbedder(), _report(errors), ["chrono_k8"], k=2)


def test_gold_position_buckets_by_first_answer_bearing_excerpt():
    result = _analyze(analyze_gold_position, errors=[("q1", "chrono_k8")])
    # chrono_k8 shows the two evidence turns oldest-first, so the gold-bearing
    # turn ("alpha gold") lands at position 2.
    assert result["chrono_k8"]["by_position"]["2"]["n"] == 1
    assert result["chrono_k8"]["accuracy"] == 0.0  # the report marks the row failed


def test_gold_position_counts_a_correct_row_when_absent_from_errors():
    result = _analyze(analyze_gold_position, errors=())
    assert result["chrono_k8"]["accuracy"] == 1.0


def test_discourse_order_detects_that_the_update_is_shown_after_the_stale_one():
    result = _analyze(analyze_discourse_order)
    assert result["pooled"]["new_after_old"]["n"] == 1
    assert result["pooled"]["new_before_old"]["n"] == 0


def test_discourse_order_skips_rows_where_only_one_side_is_visible():
    # current_only shows just the update, so it cannot discriminate and must not
    # be counted.
    entry = _entry()
    report = {"errors": {"knowledge_update": []}, "config": {"arms": ["current_only"], "k": 2}}
    result = analyze_discourse_order([entry], FakeEmbedder(), report, ["current_only"], k=2)
    assert result["n_rows_both_visible"] == 0
