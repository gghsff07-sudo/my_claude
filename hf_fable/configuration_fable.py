"""FableConfig — Hugging Face configuration for the `fable` miniature.

Mirrors `fable_lm.model.ModelConfig` (vocab_size=2048, context_len=512, n_layers=4,
d_model=256, n_heads=4, d_ff=768) using standard HF configuration names so the
model is loadable with `AutoConfig` / `trust_remote_code`.

This file is intentionally self-contained (no `fable_lm` import): it is copied
verbatim into the Hub repo, where only the files next to `config.json` exist.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class FableConfig(PretrainedConfig):
    model_type = "fable"

    def __init__(
        self,
        vocab_size: int = 2048,
        max_position_embeddings: int = 512,
        num_hidden_layers: int = 4,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        intermediate_size: int = 768,
        dropout: float = 0.0,
        pad_token_id: int = 256,   # <|pad|>
        bos_token_id: int = 258,   # <|doc|> doubles as BOS (document separator)
        eos_token_id: int = 257,   # <|eos|>
        **kwargs,
    ):
        # architecture hyperparameters (names follow HF conventions)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        # token embeddings are tied into the LM head (FableLM: head.weight = tok_emb.weight)
        self.tie_word_embeddings = True
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    # ---- read-only aliases for the original repo-style names ----------------
    # Properties (not attributes) so they never appear in to_dict()/config.json;
    # serialization always uses the canonical HF names above.

    @property
    def context_len(self) -> int:
        return self.max_position_embeddings

    @property
    def n_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def n_heads(self) -> int:
        return self.num_attention_heads

    @property
    def d_ff(self) -> int:
        return self.intermediate_size


try:
    # Local (non-remote-code) use: `import hf_fable` makes AutoConfig aware of
    # "fable" without trust_remote_code. exist_ok: the module may also be
    # imported under a second name (hf_fable.configuration_fable vs a plain
    # configuration_fable copied next to config.json).
    from transformers import AutoConfig

    AutoConfig.register("fable", FableConfig, exist_ok=True)
except Exception:  # pragma: no cover - registration must never break import
    pass
