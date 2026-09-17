import pytest

from memory_arch.run_llm_keys import normalize_key, pair_overlap, system_prompt


def test_normalize_key_declines_and_cleans():
    assert normalize_key("NONE") is None
    assert normalize_key(" none. ") is None
    assert normalize_key("") is None
    assert normalize_key("Personal  Best") == "personal best"
    assert normalize_key('"home city".') == "home city"


def test_normalize_key_truncates_instead_of_guessing():
    long_answer = "the user's personal best time in the charity five k run last spring"
    assert len(normalize_key(long_answer)) == 60


def test_prompt_variants_differ_and_reject_unknown_names():
    plain, granular = system_prompt("plain"), system_prompt("granular")
    assert plain != granular
    # Both must carry the mixed message rule found necessary by the smoke test.
    assert "Any tips?" in plain and "Any tips?" in granular
    assert "yoga frequency" in granular and "yoga frequency" not in plain
    with pytest.raises(ValueError):
        system_prompt("tuned-by-hand")


def test_exact_overlap_short_circuits_fuzzy():
    result = pair_overlap({"home city"}, {"home city"}, {})
    assert result["exact"] == ["home city"] and result["best_cosine"] == 1.0


def test_fuzzy_overlap_uses_embedding_similarity():
    vectors = {"home city": [1.0, 0.0], "home town": [0.99, 0.141]}
    result = pair_overlap({"home city"}, {"home town"}, vectors)
    assert result["exact"] == [] and result["fuzzy"] == ["home city ~ home town"]
    assert result["best_cosine"] >= 0.85


def test_disjoint_keys_stay_unmatched():
    vectors = {"home city": [1.0, 0.0], "yoga frequency": [0.0, 1.0]}
    result = pair_overlap({"home city"}, {"yoga frequency"}, vectors)
    assert result["exact"] == [] and result["fuzzy"] == []
    assert result["best_cosine"] == 0.0
