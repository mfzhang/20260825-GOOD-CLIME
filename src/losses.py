"""
losses.py — Pairwise logistic ranking loss + Directional regression loss.
"""

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    s_i: torch.Tensor,
    s_j: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Pairwise logistic ranking loss.

    L = log(1 + exp(-y * (s_i - s_j))) = softplus(-y*(s_i - s_j))
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
    """Asymmetric directional regression loss for stock return prediction.

    Rules:
      - Sign mismatch (pred pos, true neg or vice versa): alpha * huber  (heavy penalty)
      - Both negative & pred < true (conservative):       beta  * huber  (reward)
      - Otherwise:                                        1.0   * huber  (normal)

    Anchored on Huber loss for robustness to extreme returns.

    Parameters
    ----------
    pred, true : [B]    预测值和真实值
    alpha      : float  sign-mismatch penalty multiplier (default 3.0)
    beta       : float  conservative negative prediction reward (default 0.5)
    delta      : float  huber loss delta (default 0.01)
    """
    pred = pred.float()
    true = true.float()

    huber = F.huber_loss(pred, true, delta=delta, reduction="none")

    same_sign = (torch.sign(pred) == torch.sign(true))  # zero is treated as own sign
    diff_sign = ~same_sign                              # sign mismatch
    conservative = (true < 0) & (pred < true)            # negative stock, predicted even worse

    weight = torch.ones_like(huber)
    weight[diff_sign] = alpha
    weight[conservative] = beta

    return (weight * huber).mean()
