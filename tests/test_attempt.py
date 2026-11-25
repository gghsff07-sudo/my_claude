"""Tests for fable_lm/attempt.py — parsing only; the actual 30B construction
attempt is a measurement (it deliberately runs until the OOM killer), not a
unit test."""

from fable_lm.attempt import LINE_RE, parse_child_progress

SAMPLE = """constructing LADDER['30b']: 49 layers x d7168, target 30,233,097,216 params (30.2331 B)
machine: 4.36 GB total, 3.19 GB available at start
embeddings allocated | rss=180 MB | peak=180 MB | params_built=18,350,080 (0.06%)
layer 1/49 allocated | rss=2700 MB | peak=2700 MB | params_built=634,977,280 (2.10%)
layer 2/49 allocated | rss=3800 MB | peak=3800 MB | params_built=1,251,604,480 (4.14%)
ALLOC-FAILED: cannot allocate memory
"""


def test_parse_child_progress_extracts_layers_and_rss():
    records = parse_child_progress(SAMPLE)
    assert [r["event"] for r in records] == [
        "embeddings allocated", "layer 1/49 allocated", "layer 2/49 allocated"]
    assert records[1]["layer"] == 1
    assert records[2]["params_built"] == 1_251_604_480
    assert records[1]["rss_mb"] == 2700
    assert records[0]["layer"] is None


def test_parse_ignores_non_progress_lines():
    records = parse_child_progress(
        "random stderr noise\nparams_built=99\nlayer 7/49 allocated | "
        "rss=1 MB | peak=2 MB | params_built=4,315,613,440 (14.27%)")
    assert len(records) == 1
    assert records[0]["layer"] == 7
    assert records[0]["peak_mb"] == 2


def test_line_re_rejects_malformed():
    assert LINE_RE.match("nothing here") is None
    assert LINE_RE.match("layer 3/49 done") is None  # 'allocated' is required
