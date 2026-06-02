"""
backtest.py — V9CA_AB 回测。

模拟真实交易:
  - 每日按模型预测分数排序，选 top-N 股票等权持仓
  - 计算累计收益、夏普比率、最大回撤、日胜率
  - 对比等权市场基准

用法:
  python backtest.py --v9ca-ab
  python backtest.py --v9ca-ab --split val_v5 --n 20
  python backtest.py --v9ca-ab --split both --n 20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output" / "transformer_v5"

from src.data.dataset import _cache_path_for
from src.models.v9ca_ab import V9CA_ABModel

RISK_FREE_RATE = 0.015

V5_SPLITS = {
    "val_v5":     ("20250918", "20251204"),
    "holdout_v5": ("20251205", "20260429"),
}

V9C_SPLITS = {
    "val_v5":     ("20250918", "20251204"),
    "holdout_v5": ("20251205", "20260511"),
}


# ---- 数据加载 ----

def load_v5_data(split_name: str):
    step = 1 if "holdout" in split_name or "val" in split_name else 10
    cache_path = _cache_path_for(split_name, 40, step)
    print(f"Loading {cache_path.name}...")
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    seqs = data["sequences"]
    returns = data["returns"]
    codes = data["codes"]
    dates = sorted(seqs.keys())
    return seqs, returns, codes, dates


# ---- 指标计算 ----

def compute_metrics(daily_rets: np.ndarray, daily_values: np.ndarray, n_dates: int):
    cum_ret = float(daily_values[-1] - 1)
    ann_ret = float((1 + cum_ret) ** (252 / max(n_dates, 1)) - 1)
    excess = daily_rets - RISK_FREE_RATE / 252
    sharpe = float(excess.mean() / (daily_rets.std() + 1e-8) * np.sqrt(252))
    peak = np.maximum.accumulate(daily_values)
    drawdowns = (peak - daily_values) / (peak + 1e-8)
    max_dd = float(drawdowns.max())
    win_rate = float((daily_rets > 0).mean())
    avg_daily_bps = float(daily_rets.mean() * 10000)
    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "daily_win_rate": win_rate,
        "avg_daily_return_bps": avg_daily_bps,
    }


# ---- V9CA_AB 回测 ----

def _load_v9cx_ckpt(model_cls, ckpt_path: str, stage1_path: str):
    model = model_cls(stage1_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _backtest_split_v9cx(split_name: str, model: torch.nn.Module,
                          device: torch.device, top_n: int, label: str):
    start_date, end_date = V9C_SPLITS[split_name]
    print(f"\n{'='*70}")
    print(f"  {label} Backtest: {split_name}  ({start_date} ~ {end_date})  Top-N={top_n}")
    print(f"{'='*70}")

    seqs, returns, codes, dates = load_v5_data(split_name)
    print(f"  {len(dates)} trading days")

    # Slice from 87-dim to 67-dim
    v9_feat_idx = list(range(66)) + [86]
    for d in dates:
        if seqs[d].shape[-1] > 67:
            seqs[d] = seqs[d][:, :, v9_feat_idx]
    print(f"  Features: {seqs[dates[0]].shape[-1]}")

    step = 1
    dyn_path = CACHE_DIR / f"v7_peer_dynamics_{split_name}_L40_step{step}.pt"
    peer_dyn = torch.load(dyn_path, map_location="cpu", weights_only=False) if dyn_path.exists() else {}
    print(f"  Peer dynamics: {len(peer_dyn)} dates")

    model.to(device)

    daily_rets = []
    daily_values = [1.0]

    for date in tqdm(dates, desc="  Backtesting", unit="day"):
        day_seqs = seqs[date]
        day_ret = returns[date]
        N = day_seqs.shape[0]

        if N < top_n:
            daily_rets.append(0.0)
            daily_values.append(daily_values[-1])
            continue

        dyn = peer_dyn.get(date)
        if dyn is None:
            daily_rets.append(0.0)
            daily_values.append(daily_values[-1])
            continue

        x = torch.as_tensor(day_seqs, dtype=torch.float32, device=device)
        dyn_t = torch.as_tensor(dyn, dtype=torch.float32, device=device)

        with torch.no_grad():
            out = model(x, dyn_t)
            if isinstance(out, tuple):
                scores = out[0].cpu().numpy()
            else:
                scores = out.cpu().numpy()

        top_idx = np.argsort(scores)[-top_n:]
        top_ret = float(day_ret[top_idx].mean())
        daily_rets.append(top_ret)
        daily_values.append(daily_values[-1] * (1.0 + top_ret))

    daily_rets = np.array(daily_rets)
    daily_values = np.array(daily_values)
    metrics = compute_metrics(daily_rets, daily_values, len(dates))

    del model
    torch.cuda.empty_cache()

    market_rets = [float(returns[d].mean()) for d in dates]
    market_values = [1.0]
    for r in market_rets:
        market_values.append(market_values[-1] * (1.0 + r))
    market_metrics = compute_metrics(
        np.array(market_rets), np.array(market_values), len(dates))

    exc = metrics["cumulative_return"] - market_metrics["cumulative_return"]
    print(f"  {label} cumulative return: {metrics['cumulative_return']:.4%}")
    print(f"  Market cumulative return: {market_metrics['cumulative_return']:.4%}")
    print(f"  Excess vs Market: {exc:.4%}")

    return {
        "split": split_name,
        "date_range": f"{start_date} ~ {end_date}",
        "n_dates": len(dates),
        "top_n": top_n,
        "market": market_metrics,
        label.lower(): metrics,
        f"excess_{label.lower()}_vs_market": exc,
    }


def _run_v9cx_backtest(label: str, model_cls, ckpt_path: str, stage1_path: str,
                        device: torch.device, top_n: int, output_dir: Path):
    all_results = {}
    for split in ["val_v5", "holdout_v5"]:
        m = _load_v9cx_ckpt(model_cls, ckpt_path, stage1_path)
        all_results[split] = _backtest_split_v9cx(split, m, device, top_n, label)
    output_path = output_dir / f"backtest_{label.lower()}_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return all_results


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="V9CA_AB Backtest")
    parser.add_argument("--split", type=str, default="both",
                        choices=["val_v5", "holdout_v5", "both"])
    parser.add_argument("--v9ca-ab", action="store_true",
                        help="Run V9CA_AB backtest")
    parser.add_argument("--v9ca-ab-ckpt", type=str, default=None,
                        help="V9CA_AB checkpoint path")
    parser.add_argument("--stage1", type=str, default=None,
                        help="Stage 1 checkpoint path")
    parser.add_argument("--n", type=int, default=20,
                        help="持仓股票数 (default: 20)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stage1_path = args.stage1 or str(OUTPUT_DIR / "stage1_best.pt")

    if args.v9ca_ab:
        ckpt_path = args.v9ca_ab_ckpt or str(OUTPUT_DIR / "transformer_v9ca_ab_best.pt")
        all_results = _run_v9cx_backtest(
            "V9CA_AB", V9CA_ABModel, ckpt_path, stage1_path, device, args.n, OUTPUT_DIR)
        return

    print("Usage: python backtest.py --v9ca-ab [--split val_v5|holdout_v5|both]")
    sys.exit(1)


if __name__ == "__main__":
    main()
