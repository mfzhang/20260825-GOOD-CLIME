"""
CLIME 完整模型：ScaledGatedEncoder + Transformer Backbone。

对应报告：
  - Section 2.3.2  ScaledGatedEncoder（市场信息加法注入）
  - Section 2.3.3  加法注入与设计原则
  - Section 2.4.2  Stage 2: Three-Phase Curriculum Training

核心公式（报告 Eq. 1-2）：
  h' = h + offset,  offset = gate * tanh(s) * (û / ||û||) * sqrt(d_model)

设计原则：
  A: Unit-normalized offset with learnable bounded scale
     → 方向归一化 + 全局可学习幅度，切断梯度短路
  B: Per-stock gate from market_ctx
     → 逐股票门控，尊重个股对市场敏感度的异质性

加性注入（而非 FiLM 乘性调制）确保 Backbone 的原始判断不被覆盖，
市场信息仅作为受控增量修正。详见报告 Section 2.3.3。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from .transformer import TransformerPredictor

PEER_DYN_DIM = 24
D_MODEL = 256
# market_ctx 的 8 维：从 67 维特征最后时间步提取，对应报告附录 A 特征表：
#   idx 34 → feat_35 sh_ret_1d              (上证指数日收益)
#   idx 37 → feat_38 hs300_ret_1d           (沪深300日收益)
#   idx 39 → feat_40 cyb_ret_1d             (创业板指日收益)
#   idx 43 → feat_44 hs300_drawdown_20d     (沪深300 20日回撤)
#   idx 46 → feat_47 market_advancing_ratio  (市场上涨比例)
#   idx 47 → feat_48 market_cross_section_vol (市场截面波动率)
#   idx 48 → feat_49 market_top_bottom_spread (市场顶部-底部价差)
#   idx 62 → feat_63 relative_industry_ret_1d (个股相对行业收益)
MARKET_CTX_INDICES = [34, 37, 39, 43, 46, 47, 48, 62]
MARKET_CTX_DIM = len(MARKET_CTX_INDICES)  # 8


class ScaledGatedEncoder(nn.Module):
    """市场信息加法注入编码器。对应报告 Section 2.3.2。

    三个子模块协同工作：
      1. offset_net:  peer_dyn(24) + market_ctx(8) → raw_offset [B, 256]
         → L2 归一化为单位方向向量（控制方向，不控制幅度）
      2. gate_net:    market_ctx(8) → gate ∈ [0, 1]
         → 逐股票选择性门控（不同股票接收不同程度的市场上下文）
      3. logit_scale: 全局可学习标量 s → tanh(s) * sqrt(d_model)
         → 全局幅度约束（报告 Table 2 消融实验验证 init_scale=0.3 最优）

    最终 offset（报告 Eq. 1）：
      offset = gate * tanh(logit_scale) * (raw_offset / ||raw_offset||) * sqrt(d_model)
    """

    def __init__(self, input_dim: int = PEER_DYN_DIM + MARKET_CTX_DIM,
                 d_model: int = D_MODEL, n_layers: int = 3,
                 gate_hidden: int = 64, init_scale: float = 0.1):
        super().__init__()

        # offset_net: 将 peer+market 拼接映射为 256 维方向向量
        # 最后层用小权重初始化，确保训练初期 market offset ≈ 0
        # 对应报告 Section 2.3.2 / 附录 Table 超参
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

        # gate_net: 逐股票选择性门控 g ∈ [0, 1]
        # 对应报告 Section 2.3.2 gate_net 描述
        self.gate_net = nn.Sequential(
            nn.Linear(MARKET_CTX_DIM, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.gate_net[-2].weight, std=0.01)
        nn.init.zeros_(self.gate_net[-2].bias)

        # logit_scale: 全局可学习幅度，经 tanh 约束
        # 对应报告 Section 2.3.2 / 报告 Table 2 消融
        # 最优 init_scale = 0.3（通过 grid search 确定，见报告附录 Table 版本演进）
        self.logit_scale = nn.Parameter(torch.tensor(init_scale))
        self.d_model = d_model

    def forward(self, peer_dyn: torch.Tensor,
                market_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            offset: [B, d_model] — 归一化 + 门控 + 幅度约束后的加法注入向量
            gate:   [B, 1]      — 逐股票门控值（用于监控，对应报告 RQ3）
        """
        x = torch.cat([peer_dyn, market_ctx], dim=-1)
        raw_offset = self.offset_net(x)                    # [B, d_model]

        # L2 归一化 → 只保留方向信息（报告 Eq. 1）
        norm = raw_offset.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        direction = raw_offset / norm                       # [B, d_model]

        # 逐股票选择性门控 g ∈ [0, 1]（报告 Eq. 1）
        gate = self.gate_net(market_ctx)                   # [B, 1]

        # 全局幅度 tanh 约束后乘以 sqrt(d_model)（报告 Eq. 1）
        scale = torch.tanh(self.logit_scale) * math.sqrt(self.d_model)

        return direction * gate * scale, gate

    def get_gate_stats(self, peer_dyn, market_ctx):
        """For monitoring: return mean, std of gate values."""
        with torch.no_grad():
            gate = self.gate_net(market_ctx)
            return gate.mean().item(), gate.std().item()


class CLIMEModel(nn.Module):
    """CLIME（Cross-Modal Injection via Learned Market Encoding）完整模型。

    对应报告 Figure 1（架构图）+ Section 2.3（Model Architecture）。

    双流设计：
      peer_dyn[24] + market_ctx[8] → ScaledGatedEncoder → offset[256]
      x[67] → input_proj → h[256]
      h_mod = h + offset  （加法注入，广播到全部 40 个时间步）
      h_mod → RoPE → 4×TransformerBlock → last-token → head → score

    Stage1 的 backbone 中 attn_pool 和 head 在此不被使用；CLIME 手动拆开
    backbone，在表征空间注入 market offset 后用自己的 head 做预测。
    详见报告 Section 2.3.2–2.3.3 和 Section 2.4.2。
    """

    def __init__(self, backbone_ckpt_path: str,
                 peer_dim: int = PEER_DYN_DIM,
                 market_ctx_dim: int = MARKET_CTX_DIM,
                 n_encoder_layers: int = 3,
                 init_scale: float = 0.1,
                 gate_hidden: int = 64):
        super().__init__()

        # 加载 Stage1 backbone 权重（报告 Section 2.4.1 → 2.4.2）
        self.backbone = TransformerPredictor(
            seq_len=40, feat_dim=67, d_model=D_MODEL,
            n_heads=8, n_layers=4, ffn_hidden=1024,
        )
        ckpt = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
        bb_sd = ckpt.get("model_state_dict", ckpt)
        missing, _ = self.backbone.load_state_dict(bb_sd, strict=False)
        if missing:
            print(f"  [CLIME] Backbone: {len(missing)} missing keys"
                  f" (expected: attn_pool + head — unused in CLIME)")

        # ScaledGatedEncoder: 市场信息加法注入模块（报告 Section 2.3.2）
        self.encoder = ScaledGatedEncoder(
            input_dim=peer_dim + market_ctx_dim,
            d_model=D_MODEL, n_layers=n_encoder_layers,
            init_scale=init_scale, gate_hidden=gate_hidden,
        )
        self.market_ctx_indices = MARKET_CTX_INDICES

        # Stage 2 独立预测头（替代 backbone 内置 head）
        # 比 backbone.head 多了 BatchNorm + Dropout 以增强泛化
        # 对应报告 Section 2.4.2 Phase 1-3
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
        """CLIME 前向传播。对应报告 Figure 1 架构图。

        x: [B, L=40, D=67]  个股特征序列
        peer_dyn: [B, 24]    同业动态特征（报告 Section 2.2.2）
        → pred: [B]          alpha 分数（报告 Section 2.1）
        """
        # 1. Backbone input projection（报告 Section 2.3.1）
        h = self.backbone.input_proj(x)                    # [B, L, d_model=256]

        # 2. 从最后一帧提取 market_ctx（8 维，见文件头注释）
        market_ctx = x[:, -1, self.market_ctx_indices]     # [B, 8]

        # 3. ScaledGatedEncoder: 生成受控 market offset（报告 Eq. 1 + Section 2.3.2）
        offset, gate = self.encoder(peer_dyn, market_ctx)  # [B, d_model], [B, 1]

        self._last_gate_mean = gate.mean().item()
        self._last_gate_std = gate.std().item()

        # 4. 加法注入到表征空间（报告 Eq. 2 + Section 2.3.3）
        h_mod = h + offset[:, None, :]                     # [B, L, d_model]

        # 5. RoPE + Transformer 层（报告 Section 2.3.1）
        h_mod = self.backbone.rope(h_mod)
        for block in self.backbone.blocks:
            h_mod = block(h_mod)

        # 6. Last-token extraction + 独立 head → 标量分数（报告 Section 2.3.1）
        last = h_mod[:, -1, :]
        return self.head(last).squeeze(-1)

    def get_offset_norm(self) -> float:
        """当前全局有效 scale 值 = tanh(logit_scale) * sqrt(d_model)。
        用于监控市场信息注入幅度（报告 Section 2.3.2 中 logit_scale 的作用）。"""
        return torch.tanh(self.encoder.logit_scale).item() * math.sqrt(self.encoder.d_model)

    def get_raw_encoder_norm(self) -> float:
        """监控 offset_net 输出层权重范数（诊断工具）。"""
        w = self.encoder.offset_net[-1].weight.data
        return w.norm().item()


# 向后兼容别名（旧脚本中仍可使用 V9CA_ABModel）
V9CA_ABModel = CLIMEModel
