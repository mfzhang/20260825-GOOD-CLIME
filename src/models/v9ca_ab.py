"""
V9CA_AB: Combined Scaled + Gated Injection — V9CA improvement A+B.

Combines the two orthogonal improvements:
  A: Unit-normalized offset with learnable bounded scale
     → Cuts gradient shortcut, stabilizes training
  B: Per-stock gate from market_ctx
     → Respects stock heterogeneity in market sensitivity

Together:
  h' = h + gate * tanh(s) * (offset / ||offset||) * sqrt(d_model)

The gate controls WHICH stocks get market influence, the scale controls
HOW MUCH market influence is allowed globally. Both are learnable and
initialized to small values so backbone dominates early training.
"""

import math
import torch
import torch.nn as nn
from .transformer import TransformerPredictor

PEER_DYN_DIM = 24
D_MODEL = 256
MARKET_CTX_INDICES = [34, 37, 39, 43, 46, 47, 48, 62]
MARKET_CTX_DIM = len(MARKET_CTX_INDICES)  # 8


class ScaledGatedEncoder(nn.Module):
    """Combined A+B encoder: unit-norm offset + per-stock gate + learnable scale.

    Three components:
      1. offset_net: peer_dyn + market_ctx → raw_offset [B, d_model]
         → unit-normalized to produce direction (Scheme A)
      2. gate_net: market_ctx → gate [B, 1]
         → per-stock gating (Scheme B)
      3. logit_scale: learnable scalar (bounded by tanh) × sqrt(d_model)
         → global magnitude control (Scheme A)
    """

    def __init__(self, input_dim: int = PEER_DYN_DIM + MARKET_CTX_DIM,
                 d_model: int = D_MODEL, n_layers: int = 3,
                 gate_hidden: int = 64, init_scale: float = 0.1):
        super().__init__()

        # Offset network
        offset_layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            offset_layers.extend([
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            ])
            in_dim = d_model
        offset_layers.append(nn.Linear(d_model, d_model))
        self.offset_net = nn.Sequential(*offset_layers)

        nn.init.normal_(self.offset_net[-1].weight, std=0.001)
        nn.init.zeros_(self.offset_net[-1].bias)

        # Gate network: small MLP, output ∈ [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(MARKET_CTX_DIM, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.gate_net[-2].weight, std=0.01)
        nn.init.zeros_(self.gate_net[-2].bias)

        # Learnable global scale
        self.logit_scale = nn.Parameter(torch.tensor(init_scale))
        self.d_model = d_model

    def forward(self, peer_dyn: torch.Tensor,
                market_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            offset: [B, d_model] — normalized + gated + scaled offset
            gate:   [B, 1]      — per-stock gate (for monitoring)
        """
        x = torch.cat([peer_dyn, market_ctx], dim=-1)
        raw_offset = self.offset_net(x)                    # [B, d_model]

        # Scheme A: unit-normalize (discard magnitude, keep direction)
        norm = raw_offset.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        direction = raw_offset / norm                       # [B, d_model]

        # Scheme B: per-stock gate
        gate = self.gate_net(market_ctx)                   # [B, 1]

        # Scheme A: bounded global scale
        scale = torch.tanh(self.logit_scale) * math.sqrt(self.d_model)

        return direction * gate * scale, gate

    def get_gate_stats(self, peer_dyn, market_ctx):
        """For monitoring: return mean, std of gate values."""
        with torch.no_grad():
            gate = self.gate_net(market_ctx)
            return gate.mean().item(), gate.std().item()


class V9CA_ABModel(nn.Module):
    """V9CA_AB: V9CA with Scaled Additive + Gated Reference (combined).

    peer_dyn[24] + market_ctx[8] → ScaledGatedEncoder → offset[256] + gate[B,1]
    x[67] → input_proj → h[256]
    h_mod = h + offset  (unit-normed, gated, globally scaled)
    h_mod → rope → blocks → last_token → head → pred
    """

    def __init__(self, backbone_ckpt_path: str,
                 peer_dim: int = PEER_DYN_DIM,
                 market_ctx_dim: int = MARKET_CTX_DIM,
                 n_encoder_layers: int = 3,
                 init_scale: float = 0.1,
                 gate_hidden: int = 64):
        super().__init__()

        self.backbone = TransformerPredictor(
            seq_len=40, feat_dim=67, d_model=D_MODEL,
            n_heads=8, n_layers=4, ffn_hidden=1024,
        )
        ckpt = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
        bb_sd = ckpt.get("model_state_dict", ckpt)
        missing, _ = self.backbone.load_state_dict(bb_sd, strict=False)
        if missing:
            print(f"  [V9CA_AB] Backbone: {len(missing)} missing keys")

        self.encoder = ScaledGatedEncoder(
            input_dim=peer_dim + market_ctx_dim,
            d_model=D_MODEL, n_layers=n_encoder_layers,
            init_scale=init_scale, gate_hidden=gate_hidden,
        )
        self.market_ctx_indices = MARKET_CTX_INDICES

        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.BatchNorm1d(D_MODEL),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL, 1),
        )

        self._last_gate_mean = 0.0
        self._last_gate_std = 0.0

    def forward(self, x: torch.Tensor, peer_dyn: torch.Tensor) -> torch.Tensor:
        """x: [B, L, 67], peer_dyn: [B, 24] → pred: [B]"""
        h = self.backbone.input_proj(x)                    # [B, L, d_model]
        market_ctx = x[:, -1, self.market_ctx_indices]     # [B, 8]

        offset, gate = self.encoder(peer_dyn, market_ctx)  # [B, d_model], [B, 1]

        self._last_gate_mean = gate.mean().item()
        self._last_gate_std = gate.std().item()

        h_mod = h + offset[:, None, :]                     # [B, L, d_model]

        h_mod = self.backbone.rope(h_mod)
        for block in self.backbone.blocks:
            h_mod = block(h_mod)
        last = h_mod[:, -1, :]

        return self.head(last).squeeze(-1)

    def get_offset_norm(self) -> float:
        """Return current effective scale."""
        return torch.tanh(self.encoder.logit_scale).item() * math.sqrt(self.encoder.d_model)

    def get_raw_encoder_norm(self) -> float:
        """Monitor encoder output layer weight norm."""
        w = self.encoder.offset_net[-1].weight.data
        return w.norm().item()
