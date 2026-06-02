"""
build_v7_peer_data.py — 构建 V7 genuine peer mapping + 24-dim peer dynamics。

Peer 选择: 多维相似度 (industry, act_name, area, market, act_ent_type, return corr)
Peer dynamics: 24 个预计算统计特征

输出:
  cache/v7_peer_mapping_{split}_L40_step{step}.pt
  cache/v7_peer_dynamics_{split}_L40_step{step}.pt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
CACHE_DIR = PROJECT_ROOT / "cache"

from src.data.dataset import _cache_path_for, PEER_FEAT_INDICES

# 特征索引 (0-based)
IDX_RET_1D = 0
IDX_RET_5D = 2
IDX_RET_20D = 4
IDX_VOLUME_LOG = 10
IDX_AMOUNT_LOG = 11
IDX_TURNOVER = 22
IDX_RSI = 19
IDX_MACD = 20
IDX_BOLLINGER = 21
IDX_PB = 24
IDX_LOG_MV = 27

DYNAMICS_FEAT_INDICES = [
    IDX_RET_1D, IDX_RET_5D, IDX_RET_20D,
    IDX_VOLUME_LOG, IDX_AMOUNT_LOG, IDX_TURNOVER,
    IDX_RSI, IDX_MACD, IDX_BOLLINGER,
    IDX_PB, IDX_LOG_MV,
]
N_DYN_FEATS = len(DYNAMICS_FEAT_INDICES)  # 11

# 相似度权重
W_INDUSTRY = 0.30
W_ACT_NAME = 0.25
W_AREA = 0.10
W_MARKET = 0.10
W_ENT_TYPE = 0.10
W_RET_CORR = 0.15

N_PEERS = 10
SPLITS = {
    "train_v5": ("20160104", "20250917", 10),
    "val_v5": ("20250918", "20251204", 1),
    "holdout_v5": ("20251205", "20260429", 1),
}


def load_basic_info():
    """加载 basic.csv，返回 {ts_code: {industry, act_name, area, market, act_ent_type}}。"""
    df = pd.read_csv(PROJECT_ROOT / "data" / "basic.csv")
    info = {}
    for _, row in df.iterrows():
        info[row["ts_code"]] = {
            "industry": str(row.get("industry", "")),
            "act_name": str(row.get("act_name", "")),
            "area": str(row.get("area", "")),
            "market": str(row.get("market", "")),
            "act_ent_type": str(row.get("act_ent_type", "")),
        }
    print(f"[basic] {len(info)} stocks loaded")
    return info


def compute_genuine_peers(codes, day_seqs, basic_info):
    """为一个交易日计算 genuine peer mapping。

    Returns:
        peer_indices: List[List[int]], 每个股票 top-K 个 peer 索引
    """
    N = len(codes)
    if N < N_PEERS + 1:
        # 股票太少，返回最近邻
        result = []
        for i in range(N):
            peers = [j for j in range(N) if j != i][:N_PEERS]
            result.append(peers)
        return result

    # 构建标签数组
    ind_labels = np.array([basic_info.get(c, {}).get("industry", "") for c in codes])
    act_labels = np.array([basic_info.get(c, {}).get("act_name", "") for c in codes])
    area_labels = np.array([basic_info.get(c, {}).get("area", "") for c in codes])
    mkt_labels = np.array([basic_info.get(c, {}).get("market", "") for c in codes])
    ent_labels = np.array([basic_info.get(c, {}).get("act_ent_type", "") for c in codes])

    # 分类相似度 (向量化)
    sim = np.zeros((N, N), dtype=np.float32)
    sim += W_INDUSTRY * (ind_labels[:, None] == ind_labels[None, :])
    sim += W_ACT_NAME * (act_labels[:, None] == act_labels[None, :])
    sim += W_AREA * (area_labels[:, None] == area_labels[None, :])
    sim += W_MARKET * (mkt_labels[:, None] == mkt_labels[None, :])
    sim += W_ENT_TYPE * (ent_labels[:, None] == ent_labels[None, :])

    # 收益相关性相似度
    rets = day_seqs[:, -20:, IDX_RET_1D].astype(np.float64)  # [N, 20]
    rets_c = rets - rets.mean(axis=1, keepdims=True)
    rets_n = rets_c / (np.linalg.norm(rets_c, axis=1, keepdims=True) + 1e-8)
    ret_corr = rets_n @ rets_n.T  # [N, N]
    sim += W_RET_CORR * np.maximum(0, ret_corr.astype(np.float32))

    # 排除自身
    np.fill_diagonal(sim, -np.inf)

    # Top-K peers
    peer_indices = []
    for i in range(N):
        top_k = np.argpartition(-sim[i], N_PEERS)[:N_PEERS]
        top_k = top_k[np.argsort(-sim[i, top_k])]  # 按相似度降序
        peer_indices.append(top_k.tolist())

    return peer_indices


def compute_peer_dynamics(day_seqs, peer_indices):
    """为一天的所有股票计算 24-dim peer dynamics。

    Args:
        day_seqs: [N, L, 67]
        peer_indices: List[List[int]], 每只股票的 peer 索引

    Returns:
        dynamics: [N, 24]
    """
    N = len(peer_indices)
    last_feats = day_seqs[:, -1, :]  # [N, 67]

    # 预提取 peer dynamics 需要的特征
    stock_f = last_feats[:, DYNAMICS_FEAT_INDICES].astype(np.float32)  # [N, 11]

    dyn = np.zeros((N, 24), dtype=np.float32)

    for i in range(N):
        p_idx = peer_indices[i]
        if not p_idx:
            continue
        peer_f = stock_f[p_idx]  # [K, 11]
        K = len(p_idx)

        # Group 1: Peer return stats (7)
        dyn[i, 0] = peer_f[:, 0].mean()   # peer_mean_ret_1d
        dyn[i, 1] = peer_f[:, 1].mean()   # peer_mean_ret_5d
        dyn[i, 2] = peer_f[:, 2].mean()   # peer_mean_ret_20d
        dyn[i, 3] = peer_f[:, 0].std()    # peer_std_ret_1d
        dyn[i, 4] = peer_f[:, 1].std()    # peer_std_ret_5d
        dyn[i, 5] = peer_f[:, 2].std()    # peer_std_ret_20d
        dyn[i, 6] = peer_f[:, 0].max() - peer_f[:, 0].min()  # spread

        # Group 2: Volume/activity (5)
        dyn[i, 7] = peer_f[:, 3].mean()   # peer_mean_volume_log
        dyn[i, 8] = peer_f[:, 3].std()    # peer_std_volume_log
        dyn[i, 9] = peer_f[:, 5].mean()   # peer_mean_turnover
        dyn[i, 10] = peer_f[:, 5].std()   # peer_std_turnover
        dyn[i, 11] = peer_f[:, 4].mean()  # peer_mean_amount_log

        # Group 3: Technical (4)
        dyn[i, 12] = peer_f[:, 6].mean()  # peer_mean_rsi
        dyn[i, 13] = peer_f[:, 7].mean()  # peer_mean_macd
        dyn[i, 14] = peer_f[:, 7].std()   # peer_std_macd
        dyn[i, 15] = peer_f[:, 8].mean()  # peer_mean_bollinger

        # Group 4: Fundamental (2)
        dyn[i, 16] = peer_f[:, 9].mean()  # peer_mean_pb
        dyn[i, 17] = peer_f[:, 10].mean() # peer_mean_log_mv

        # Group 5: Stock vs peer (6)
        dyn[i, 18] = stock_f[i, 0] - peer_f[:, 0].mean()  # rel_ret_1d
        dyn[i, 19] = stock_f[i, 1] - peer_f[:, 1].mean()  # rel_ret_5d
        dyn[i, 20] = stock_f[i, 2] - peer_f[:, 2].mean()  # rel_ret_20d
        dyn[i, 21] = stock_f[i, 3] - peer_f[:, 3].mean()  # rel_volume
        dyn[i, 22] = stock_f[i, 5] - peer_f[:, 5].mean()  # rel_turnover
        dyn[i, 23] = stock_f[i, 6] - peer_f[:, 6].mean()  # rel_rsi

    return dyn


def build_split(split_name, basic_info):
    """为单个 split 构建 peer mapping + dynamics。"""
    start, end, step = SPLITS[split_name]
    cache_path = _cache_path_for(split_name, 40, step)
    if not cache_path.exists():
        print(f"[skip] Cache not found: {cache_path}")
        return

    print(f"\n{'='*60}")
    print(f"Building V7 peer data: {split_name} (step={step})")
    print(f"{'='*60}")

    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    seqs = data["sequences"]
    codes = data["codes"]
    dates = sorted(seqs.keys())
    print(f"  {len(dates)} trading days")

    mapping_out = CACHE_DIR / f"v7_peer_mapping_{split_name}_L40_step{step}.pt"
    dynamics_out = CACHE_DIR / f"v7_peer_dynamics_{split_name}_L40_step{step}.pt"

    all_mapping = {}
    all_dynamics = {}

    for date in tqdm(dates, desc=f"  {split_name}", unit="day"):
        day_seqs = seqs[date]
        day_codes = codes[date]

        # Genuine peer selection
        peer_idx = compute_genuine_peers(day_codes, day_seqs, basic_info)

        # Peer dynamics
        dyn = compute_peer_dynamics(day_seqs, peer_idx)

        all_mapping[date] = peer_idx
        all_dynamics[date] = dyn

    torch.save(all_mapping, mapping_out)
    torch.save(all_dynamics, dynamics_out)
    print(f"  Peer mapping -> {mapping_out}")
    print(f"  Peer dynamics -> {dynamics_out}  ({all_dynamics[dates[0]].shape[1]} dims)")


def main():
    print("Loading basic.csv...")
    basic_info = load_basic_info()

    for split in ["train_v5", "val_v5", "holdout_v5"]:
        build_split(split, basic_info)

    print("\nDone. V7 peer data ready.")


if __name__ == "__main__":
    main()
