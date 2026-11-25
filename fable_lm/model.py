"""Decoder-only transformer LM, implemented from scratch in PyTorch.

DESIGN.md §1: pre-LN blocks, causal self-attention, learned absolute positions,
GELU FF, tied input/output embeddings. Configurable size so the same code runs a
tiny smoke-test model and the ~2.9M default.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    context_len: int = 512
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 768
    dropout: float = 0.0  # off by default: tiny corpus, CPU training

    def param_count(self) -> int:
        """Exact parameter count for this config (embedding-tied)."""
        d, L, V, C, F = self.d_model, self.n_layers, self.vocab_size, self.context_len, self.d_ff
        tok = V * d                # token embedding (shared with output head)
        pos = C * d                # learned positions
        per_block = (
            2 * d                  # ln1 weight+bias
            + 2 * d                # ln2 weight+bias
            + 3 * d * d + d * d    # qkv (no bias) + proj (no bias)
            + 2 * d * F + F + d    # fc weight+bias, proj weight+bias
        )
        return tok + pos + L * per_block + 2 * d  # final LN


class FableKVCache:
    """Per-layer key/value cache for incremental generation.

    Each layer stores its (k, v) tensors of shape (B, nh, T_kv, d_head).
    update() concatenates the new keys/values onto the stored ones and returns
    the full tensors, so attention over the new tokens sees the whole past.
    """

    def __init__(self, n_layers: int):
        self.kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    def seq_len(self) -> int:
        first = self.kv[0]
        return 0 if first is None else first[0].size(2)

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev = self.kv[layer_idx]
        if prev is not None:
            pk, pv = prev
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        self.kv[layer_idx] = (k, v)
        return k, v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide into n_heads"
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cache: FableKVCache | None = None, layer_idx: int = 0
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)
        # (B, nh, T, dh)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)  # (B, nh, T_kv, dh)
            past = k.size(2) - T
            # token i (absolute past+i) may see j <= past+i, i.e. j > i+past masked
            mask = torch.triu(
                torch.ones(T, k.size(2), dtype=torch.bool, device=x.device),
                diagonal=1 + past,
            )
        else:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, nh, T, T_kv)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(ln(x)), then x + ff(ln(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, cache: FableKVCache | None = None, layer_idx: int = 0
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache, layer_idx)
        x = x + self.ff(self.ln2(x))
        return x


class FableLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # tied output head: logits = ln_f(x) @ tok_emb.weight.T
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) int64. Returns (logits (B,T,V), loss) — loss None w/o targets."""
        B, T = idx.shape
        assert T <= self.cfg.context_len, f"T={T} > context_len={self.cfg.context_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    def forward_with_cache(
        self, idx: torch.Tensor, cache: FableKVCache | None = None
    ) -> tuple[torch.Tensor, FableKVCache]:
        """Cached forward for incremental generation.

        idx: (B, T) int64 of NEW tokens only; cache carries the past. Positions
        continue from the cache length, so prefill + step tokens see exactly the
        positions an uncached forward over the full sequence would use. Returns
        (logits (B,T,V), cache) — pass the returned cache back in for the next
        chunk. Reuses the same blocks; only the attention mask and position
        offset differ. When cache is None this is a prefill over idx.
        """
        B, T = idx.shape
        past = cache.seq_len() if cache is not None else 0
        assert T + past <= self.cfg.context_len, (
            f"T+past={T + past} > context_len={self.cfg.context_len}; "
            "crop the input or restart the cache"
        )
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        if cache is None:
            cache = FableKVCache(self.cfg.n_layers)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cache, i)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, cache

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()  # tok_emb is tied into head, keep it
        return n

    def config_dict(self) -> dict:
        return asdict(self.cfg)
