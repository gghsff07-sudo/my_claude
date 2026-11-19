"""Training config for the default ~2.9M model (DESIGN.md §1)."""
from fable_lm.model import ModelConfig

config = ModelConfig(
    vocab_size=2048,
    context_len=512,
    n_layers=4,
    d_model=256,
    n_heads=4,
    d_ff=768,
)
