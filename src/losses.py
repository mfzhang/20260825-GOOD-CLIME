"""
CLIME 损失函数。对应报告：
  - pairwise_ranking_loss:       Section 2.4.1 (Stage 1)
  - directional_regression_loss: Section 2.4.3 (Directional Regression Loss)
"""

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    s_i: torch.Tensor,
    s_j: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Pairwise logistic ranking loss（报告 Eq. 3, Section 2.4.1）。

    L = log(1 + exp(-y * (s_i - s_j))) = softplus(-y*(s_i - s_j))

    用于 Stage 1 Backbone 预训练：直接优化相对排序，模型只需学会判断
    "谁更好"，而不需要精确回归收益幅度。
    """
    s_i = s_i.squeeze()
    s_j = s_j.squeeze()
    return F.softplus(-y * (s_i - s_j)).mean()


def directional_regression_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    alpha: float = 3.0,
    beta: float = 0.5,
    delta: float = 0.01,
) -> torch.Tensor:
    """Directional Regression Loss（报告 Eq. 4-6, Section 2.4.3）。

    底层 Huber(δ=0.01) 对抗涨跌停极端值，上层非对称权重对齐选股目标：
      - 方向错误 (sign mismatch):                alpha × huber  (默认 3.0, 重罚)
      - 保守悲观 (同向同负且 pred < true):        beta  × huber  (默认 0.5, 降权)
      - 正常 (otherwise):                         1.0   × huber

    三档权重设计逻辑（报告 Section 2.4.3 末段）：
      方向错误: 对 top-20 排序影响最大，梯度 ×3
      保守悲观: 方向正确但预测偏低，该股票通常不会进 top-20, 梯度 ×0.5
      正常:     方向正确且不满足保守悲观条件，标准 Huber 惩罚

    Parameters
    ----------
    pred, true : [B]    模型预测的 alpha 分数和真实次日收益率
    alpha      : float  方向错误惩罚系数（报告默认 3.0）
    beta       : float  保守悲观降权系数（报告默认 0.5）
    delta      : float  Huber loss δ（报告默认 0.01）
    """
    pred = pred.float()
    true = true.float()

    huber = F.huber_loss(pred, true, delta=delta, reduction="none")

    # 三档非对称权重（报告 Eq. 5 + Section 2.4.3）
    same_sign = (torch.sign(pred) == torch.sign(true))
    diff_sign = ~same_sign                              # 方向错误 → 权重 alpha
    conservative = (true < 0) & (pred < true)            # 保守悲观 → 权重 beta

    weight = torch.ones_like(huber)                      # 正常 → 权重 1.0
    weight[diff_sign] = alpha                            # 方向错误 → 权重 3.0
    weight[conservative] = beta                          # 保守悲观 → 权重 0.5

    return (weight * huber).mean()
