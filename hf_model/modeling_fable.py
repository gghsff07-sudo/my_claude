"""FableLMHeadModel — Hugging Face wrapper for the Fable miniature (model_type "fable").

Self-contained copy of the `fable_lm.FableLM` architecture (HF remote-code files
must not import the repo package):
  pre-LN decoder-only transformer, learned absolute positions, GELU FF,
  fused bias-free QKV, tied input/output embeddings.
Identical module tree and tensor names, so a checkpoint state dict loads
unchanged:
  tok_emb.weight, pos_emb.weight,
  blocks.{i}.ln1.{weight,bias}, blocks.{i}.attn.qkv.weight, blocks.{i}.attn.proj.weight,
  blocks.{i}.ln2.{weight,bias}, blocks.{i}.ff.fc.{weight,bias}, blocks.{i}.ff.proj.{weight,bias},
  ln_f.{weight,bias}, head.weight (tied to tok_emb.weight)

forward(input_ids, attention_mask, labels) -> CausalLMOutputWithPast, with an
optional DynamicCache KV cache for generate(). Supports temperature, top_p,
top_k, max_new_tokens and repetition_penalty through GenerationMixin.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

# transformers re-exports its own nn.init wrappers (they take a `generator`
# kwarg and advance the global RNG differently when it is explicitly None).
# Importing torch's originals by module path guarantees the same init draws
# as fable_lm.model.FableLM, so same-seed constructors are bit-identical.
_nn_init_normal = torch.nn.init.normal_
_nn_init_zeros = torch.nn.init.zeros_

from .configuration_fable import FableConfig


class CausalSelfAttention(nn.Module):
    """Fused-QKV multi-head self-attention, bias-free, with KV-cache support."""

    def __init__(self, cfg: FableConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_attention_heads == 0
        self.n_heads = cfg.num_attention_heads
        self.d_head = cfg.hidden_size // cfg.num_attention_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, attn_mask_4d, past_key_values=None, layer_idx=0, use_cache=True):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, nh, T, kv)
        if attn_mask_4d is not None:
            att = att.masked_fill(~attn_mask_4d, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, cfg: FableConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-LN block: x + attn(ln1(x)), then x + ff(ln2(x)) — as in fable_lm."""

    def __init__(self, cfg: FableConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.ff = FeedForward(cfg)

    def forward(self, x, attn_mask_4d, past_key_values=None, layer_idx=0, use_cache=True):
        x = x + self.attn(self.ln1(x), attn_mask_4d, past_key_values, layer_idx, use_cache)
        x = x + self.ff(self.ln2(x))
        return x


class FablePreTrainedModel(PreTrainedModel):
    config_class = FableConfig
    base_model_prefix = "fable"
    supports_gradient_checkpointing = False
    _tied_weights_keys = {"head.weight": "tok_emb.weight"}  # 5.x: {target: source}

    def _init_weights(self, module):
        """Match fable_lm.FableLM._init: normal(0, 0.02), zero biases.

        Uses torch's original nn.init functions (not transformers' re-exports)
        so the random stream matches fable_lm exactly for a given seed.
        """
        if isinstance(module, nn.Linear):
            _nn_init_normal(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                _nn_init_zeros(module.bias)
        elif isinstance(module, nn.Embedding):
            _nn_init_normal(module.weight, mean=0.0, std=0.02)


class FableLMHeadModel(FablePreTrainedModel, GenerationMixin):
    def __init__(self, config: FableConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        # tied output head: logits = ln_f(x) @ tok_emb.weight.T
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tie BEFORE post_init, exactly like fable_lm.FableLM ties before its
        # apply(_init): then the init draw for `head` overwrites the shared
        # tok_emb storage, and same-seed construction is bit-identical to the
        # original. post_init()'s tie_weights() then re-ties a no-op.
        self.head.weight = self.tok_emb.weight
        self.post_init()

    # ---- embeddings plumbing (needed by PreTrainedModel.tie_weights) ----

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    # ---- mask helper ----

    @staticmethod
    def _make_attention_mask(
        batch_size: int,
        q_len: int,
        kv_len: int,
        past_len: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Combined causal + padding mask as a bool (B, 1, T, kv) keep-mask.

        None (no padding) reduces to the pure causal mask, identical to
        fable_lm's `masked_fill(triu(diagonal=1), -inf)` when past_len == 0.
        """
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=past_len)
        keep = causal[None, None, :, :]  # (1, 1, T, kv)
        if attention_mask is not None:
            pad = attention_mask.to(device=device, dtype=torch.bool)
            if pad.ndim == 2:  # (B, kv) -> (B, 1, 1, kv)
                pad = pad[:, None, None, :]
            keep = keep & pad
        elif past_len == 0 and q_len == kv_len:
            # exactly the original mask (triu diagonal=1 inverted) — skip the
            # extra AND for bit-identical parity with fable_lm
            return keep
        return keep

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Causal LM forward. `labels` are shifted internally (HF convention):
        loss = CE(logits[:, :-1], labels[:, 1:])."""
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds required")
        use_cache = True if use_cache is None else use_cache

        if inputs_embeds is not None:
            x = inputs_embeds
            B, T = x.shape[:2]
        else:
            x = self.tok_emb(input_ids)
            B, T = input_ids.shape

        past_len = past_key_values.get_seq_length() if isinstance(past_key_values, Cache) else 0
        if T + past_len > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {T + past_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=x.device)
        x = x + self.pos_emb(position_ids)  # (B, T, C); pos_emb(ids) broadcasts from (T, C)

        attn_mask_4d = self._make_attention_mask(
            B, T, T + past_len, past_len, attention_mask, x.device
        )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask_4d, past_key_values, i, use_cache)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    def num_params(self, non_embedding: bool = False) -> int:
        """fable_lm-compatible param count (tied head counted once)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n


try:
    # Local (non-remote-code) use: registering with the Auto classes makes
    # from_pretrained work on a local hf_model/ dir without trust_remote_code.
    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.register(FableConfig, FableLMHeadModel, exist_ok=True)
except Exception:  # pragma: no cover
    pass
