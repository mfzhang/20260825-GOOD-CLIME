"""
transformer.py — Transformer Encoder with RoPE.

Linear(67,128) → RoPE → 2×(MHA(4heads)+Res+LN+FFN(512)+Res+LN)
→ last token → FC → score
"""

import math

import torch
import torch.nn as nn

from src.data.dataset import L, D


class RotaryPositionalEncoding(nn.Module):
    """RoPE: 对 Q, K 的每对相邻维度施加旋转位置编码。"""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # [max_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)         # [max_len, dim]
        self.register_buffer("cos_cached", emb.cos())   # [max_len, dim]
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        T = x.size(1)
        half = x.size(-1) // 2
        cos = self.cos_cached[:T, :half]  # [T, half]
        sin = self.sin_cached[:T, :half]

        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class AttentionPooling(nn.Module):
    """加性注意力池化：学习每个时间步的重要性权重，替代 last-token 聚合."""

    def __init__(self, d_model: int = 256, d_attn: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1, bias=False),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, L, d_model]
        scores = self.attn(h)                # [B, L, 1]
        weights = torch.softmax(scores, dim=1)  # [B, L, 1]
        return (weights * h).sum(dim=1)      # [B, d_model]


class TransformerPredictor(nn.Module):
    def __init__(self, seq_len: int = L, feat_dim: int = D, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 4,
                 ffn_hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.rope = RotaryPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.attn_pool = AttentionPooling(d_model, d_attn=128)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x = self.input_proj(x)     # [B, L, d_model]
        x = self.rope(x)
        for block in self.blocks:
            x = block(x)
        # attention pooling over time dimension
        x = self.attn_pool(x)       # [B, d_model]
        return self.head(x).squeeze(-1)  # [B]
