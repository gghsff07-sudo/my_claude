"""Tests for the distinct-n diversity metric in fable_lm.eval."""

from fable_lm.eval import distinct_n


def test_distinct_1_all_unique():
    assert distinct_n("a b c d e", 1) == 1.0


def test_distinct_1_repetitive():
    assert abs(distinct_n("a a a a", 1) - 0.25) < 1e-9


def test_distinct_2_mixed():
    # grams: (a,b) (b,a) (a,b) -> 2 unique of 3
    assert abs(distinct_n("a b a b", 2) - 2 / 3) < 1e-9


def test_distinct_n_short_text_is_zero():
    assert distinct_n("hello", 2) == 0.0  # fewer tokens than n
    assert distinct_n("", 1) == 0.0


def test_distinct_n_longer_text():
    text = "the cat sat on the mat the cat sat"
    toks = text.split()
    grams = [tuple(toks[i : i + 2]) for i in range(len(toks) - 1)]
    expected = len(set(grams)) / len(grams)
    assert abs(distinct_n(text, 2) - expected) < 1e-9
