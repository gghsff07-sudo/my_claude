"""hf_fable — Hugging Face compatible wrapper for the Fable miniature.

Self-contained: these files are copied verbatim into the Hub repo (they must
work with only config.json/model.safetensors/tokenizer.json next to them, no
`fable_lm` on the import path). Parity with the original implementation is
enforced by tests/test_hf_compat.py.
"""

from .configuration_fable import FableConfig
from .modeling_fable import FableLMHeadModel, FablePreTrainedModel

__all__ = ["FableConfig", "FableLMHeadModel", "FablePreTrainedModel"]
