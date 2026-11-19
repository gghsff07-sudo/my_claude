import pytest
import torch

from fable_lm.model import Block, FableLM, ModelConfig


def tiny_cfg(**kw):
    base = dict(vocab_size=64, context_len=32, n_layers=2, d_model=32, n_heads=4, d_ff=64)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    model = FableLM(tiny_cfg())
    B, T = 3, 16
    idx = torch.randint(0, 64, (B, T))
    targets = torch.randint(0, 64, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, 64)
    assert loss is not None and loss.dim() == 0
    assert torch.isfinite(loss)
    # loss of an untrained model should be near ln(V) = ln(64)
    assert 3.0 < loss.item() < 5.0


def test_causality_future_tokens_do_not_affect_past():
    torch.manual_seed(0)
    model = FableLM(tiny_cfg())
    model.eval()
    B, T = 2, 12
    idx = torch.randint(0, 64, (B, T))
    # perturb the LAST token and check logits at earlier positions are unchanged
    idx2 = idx.clone()
    idx2[:, -1] = (idx2[:, -1] + 1) % 64
    with torch.no_grad():
        l1, _ = model(idx)
        l2, _ = model(idx2)
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-6), "causality violated"


def test_gradient_flows():
    torch.manual_seed(0)
    model = FableLM(tiny_cfg())
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    _, loss = model(idx, targets)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    # tied head: tok_emb gradient must exist (shared with head weight)
    assert model.tok_emb.weight.grad is not None


def test_tied_weights_and_param_count():
    cfg = tiny_cfg()
    model = FableLM(cfg)
    assert model.head.weight is model.tok_emb.weight
    expected = cfg.param_count()
    assert model.num_params() == expected, (model.num_params(), expected)


def test_context_len_guard():
    model = FableLM(tiny_cfg(context_len=8))
    with pytest.raises(AssertionError):
        model(torch.randint(0, 64, (1, 9)))


def test_block_residual_shape():
    blk = Block(tiny_cfg())
    x = torch.randn(2, 10, 32)
    y = blk(x)
    assert y.shape == x.shape


def test_default_config_param_budget():
    """DESIGN.md: shipped default must land in the 'few million' range."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "small_cfg", Path(__file__).resolve().parents[1] / "configs" / "small.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n = mod.config.param_count()
    assert 1_000_000 < n < 5_000_000, f"default config out of budget: {n}"
