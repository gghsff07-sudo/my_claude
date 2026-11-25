"""Tests for the scale-up automation (fable_lm/scale.py).

The 30B claim is arithmetic: ModelConfig.param_count() is cross-checked
against real instantiation on small configs, and the 30B preset is verified
by the same formula without ever being built.
"""

import pytest

from fable_lm.model import FableLM, ModelConfig
from fable_lm.scale import (
    LADDER,
    TARGET_30B,
    build_ladder,
    fit_batch,
    parse_target,
    plan_rungs,
    training_peak_bytes,
)


@pytest.mark.parametrize(
    "cfg",
    [
        ModelConfig(n_layers=1, d_model=16, n_heads=2, d_ff=16),
        ModelConfig(n_layers=2, d_model=32, n_heads=4, d_ff=32),
        ModelConfig(n_layers=4, d_model=256, n_heads=4, d_ff=768),  # shipped miniature
        ModelConfig(n_layers=8, d_model=768, n_heads=12, d_ff=3072),
    ],
)
def test_param_count_matches_instantiation(cfg):
    """param_count() arithmetic == sum(p.numel()) on a real model. This is the
    formula the 30B figure rests on, so it must hold wherever it is checkable."""
    model = FableLM(cfg)
    assert cfg.param_count() == model.num_params()


def test_thirty_b_preset_is_really_30b():
    cfg = LADDER["30b"]
    n = cfg.param_count()
    assert n >= TARGET_30B, f"30B preset is only {n / 1e9:.3f}B"
    assert n < 31 * 1e9, f"30B preset overshoots: {n / 1e9:.3f}B"
    # architecture constraints must hold even though the model is never built here
    assert cfg.d_model % cfg.n_heads == 0
    assert cfg.d_ff == 4 * cfg.d_model


def test_ladder_is_monotonic_and_reaches_target():
    rungs = build_ladder(TARGET_30B)
    params = [r.params for r in rungs]
    assert params == sorted(params) and len(set(params)) == len(params)
    assert params[0] == LADDER["3.3m"].param_count()  # starts at the shipped model
    assert params[-1] >= TARGET_30B
    assert rungs[-1].name == "30b"


def test_ladder_smaller_target_stops_early():
    rungs = build_ladder(100_000_000)
    assert rungs[-1].params >= 100_000_000
    assert all(r.params < 1e9 for r in rungs)


def test_parse_target():
    assert parse_target("30b") == 30_000_000_000
    assert parse_target("15M") == 15_000_000
    assert parse_target("3e9") == 3_000_000_000
    assert parse_target("3000000000") == 3_000_000_000
    with pytest.raises(ValueError):
        parse_target("bogus")


def test_training_peak_bytes_scales_and_gates():
    small = training_peak_bytes(LADDER["3.3m"], batch=8)
    big = training_peak_bytes(LADDER["30b"], batch=8)
    assert big > 400e9  # 30B AdamW fp32 training is hundreds of GB, not GB
    assert big > small * 1000
    # smaller batch reduces the estimate monotonically
    b8 = training_peak_bytes(LADDER["59m"], batch=8)
    b1 = training_peak_bytes(LADDER["59m"], batch=1)
    assert b1 < b8


def test_fit_batch_returns_none_when_impossible():
    assert fit_batch(LADDER["30b"], mem_limit=2_300_000_000) is None


def test_plan_rungs_dry_never_instantiates():
    """Planning the whole 30B ladder must work on a 4 GB machine — no model
    is built, so no allocation can fail."""
    plan = plan_rungs(TARGET_30B, steps=10, mem_limit=2_300_000_000)
    assert [r["name"] for r in plan] == [r.name for r in build_ladder(TARGET_30B)]
    assert plan[-1]["params"] >= TARGET_30B
    assert plan[-1]["feasible"] is False
    feasible = [r for r in plan if r["feasible"]]
    assert feasible and all(r["batch"] in (1, 2, 4, 8) for r in feasible)


def test_head_dims_divide():
    """Every preset must satisfy d_model % n_heads == 0 (FableLM asserts it)."""
    for name, cfg in LADDER.items():
        assert cfg.d_model % cfg.n_heads == 0, name
