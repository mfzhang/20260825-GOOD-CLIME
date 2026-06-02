"""
daily_inference.py — V9CA_AB is0.3 每日推理脚本

用法:
    # 单日推理
    python daily_inference.py --date 20260529

    # 指定资金量，自动计算持仓手数
    python daily_inference.py --date 20260529 --capital 1000000

    # 多日模拟
    python daily_inference.py --dates 20260512,20260513,20260514

    # 输出 CSV 用于提交
    python daily_inference.py --date 20260529 --output result.csv

工作流程:
    1. 可选: 重建 raw_panel + normalized_features（仅新数据到来时需要）
    2. 构建目标日期的推理序列 [N, 40, 67] + peer_dyn [N, 24]
    3. 计算 risk_score [N]
    4. V9CA_AB 模型推理 → alpha_score
    5. 两阶段: top-10% alpha → z(alpha) - 0.3*z(risk) → top-20
    6. Softmax 权重分配
    7. 输出推荐列表（代码 + 名称 + 权重）
"""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output" / "transformer_v5"
DATA_DIR = PROJECT_ROOT / "data"

V9_FEATURE_INDICES = list(range(66)) + [86]  # 67 dims used by V9
L_SEQ = 40
BATCH_SIZE = 4096
LAMBDA_RISK = 0.3
TOP_PCT = 0.10
TOP_N = 20

STAGE1_PATH = str(OUTPUT_DIR / "stage1_best.pt")
DEFAULT_V9CA_AB_PATH = str(OUTPUT_DIR / "transformer_v9ca_ab_is0p3_best.pt")

# 用于 softmax 的 temperature（越小越集中，越大越平均）
WEIGHT_TEMPERATURE = 0.5

# 股票名称缓存
_STOCK_NAMES: Dict[str, str] = {}


def load_stock_names() -> Dict[str, str]:
    """加载股票代码 → 名称映射。"""
    global _STOCK_NAMES
    if _STOCK_NAMES:
        return _STOCK_NAMES
    basic_path = DATA_DIR / "basic.csv"
    if basic_path.exists():
        df = pd.read_csv(basic_path)
        _STOCK_NAMES = dict(zip(df["ts_code"], df["name"]))
    return _STOCK_NAMES


# ===========================================================================
# 数据更新
# ===========================================================================

def rebuild_raw_panel(force: bool = False) -> pd.DataFrame:
    """重建 raw_panel.pkl。如果缓存存在且不强制，直接加载。"""
    cache_path = CACHE_DIR / "raw_panel.pkl"

    if not force and cache_path.exists():
        print("[panel] Loading cached raw_panel.pkl...")
        with open(cache_path, "rb") as f:
            panel = pickle.load(f)
        print(f"  Shape: {panel.shape}, dates: {panel['trade_date'].min()} ~ {panel['trade_date'].max()}")
        return panel

    print("[panel] Rebuilding raw_panel from CSV files...")
    from src.data.loader import load_daily_panel

    if cache_path.exists():
        cache_path.unlink()
        print("  Deleted stale cache")

    panel = load_daily_panel(start="20160104", end="20261231")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(panel, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved → {cache_path} ({panel.shape})")
    return panel


def rebuild_features(panel: pd.DataFrame) -> pd.DataFrame:
    """重建 normalized_features.parquet。"""
    from src.data.loader import load_market, load_basic, load_st_dates
    from src.data.features import compute_features
    from src.data.preprocess import preprocess

    print("[features] Computing features...")
    market = load_market("20160104", "20261231")
    basic = load_basic()
    st = load_st_dates("20160104", "20261231")

    feat_df = compute_features(panel, market, basic, st, label_horizon=1)
    print(f"  Feature table: {feat_df.shape}")

    print("[features] Preprocessing (filter + normalize)...")
    clean = preprocess(feat_df, filter_universe=True)
    print(f"  Clean table: {clean.shape}")

    clean_cache = CACHE_DIR / "normalized_features.parquet"
    clean.to_parquet(clean_cache, index=False)
    print(f"  Saved → {clean_cache}")

    return clean


# ===========================================================================
# 单日序列构建
# ===========================================================================

def build_single_day_sequence(date_str: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """为单个日期构建推理序列。

    Returns
    -------
    seqs: np.ndarray [N, L, 87]    全特征序列（87 = 86 feat + lag_norm）
    returns: np.ndarray [N]         future return（用于回测验证）
    codes: List[str]               股票代码
    """
    parquet_path = CACHE_DIR / "normalized_features.parquet"
    clean_df = pd.read_parquet(parquet_path)

    feat_cols = [c for c in clean_df.columns if c.startswith("feat_")]
    n_feat = len(feat_cols)

    # 筛选在 date_str 有数据的股票
    day_mask = clean_df["trade_date"] == date_str
    day_codes = set(clean_df.loc[day_mask, "ts_code"].unique())
    if len(day_codes) == 0:
        raise ValueError(f"No stocks on date {date_str}")

    # 为每只股票构建 [L, n_feat+1] 序列
    seqs_list = []
    returns_list = []
    codes_list = []

    grouped = clean_df.groupby("ts_code")
    lag_norm = np.arange(L_SEQ, dtype=np.float32) / (L_SEQ - 1)

    for code, grp in grouped:
        if code not in day_codes:
            continue
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        # 找到 date_str 的位置
        date_positions = grp["trade_date"].values == date_str
        if not date_positions.any():
            continue
        idx = int(np.where(date_positions)[0][0])

        # 需要 L_SEQ 天历史数据
        if idx < L_SEQ - 1:
            continue
        start = idx - L_SEQ + 1
        window = grp[feat_cols].values[start:idx + 1].astype(np.float32)  # [L, F]
        window = np.nan_to_num(window, nan=0.0)

        # 添加 lag_norm
        lag = lag_norm.reshape(L_SEQ, 1)
        seq = np.concatenate([window, lag], axis=1)  # [L, F+1]

        ret = grp["future_return"].values[idx]

        seqs_list.append(seq)
        returns_list.append(ret)
        codes_list.append(code)

    seqs = np.stack(seqs_list, axis=0)  # [N, L, F+1]
    returns = np.array(returns_list, dtype=np.float32)
    codes = codes_list

    print(f"  Built sequence for {date_str}: {seqs.shape}, codes={len(codes)}")
    return seqs, returns, codes


# ===========================================================================
# Peer Dynamics (单日)
# ===========================================================================

def compute_peer_for_single_date(
    day_codes: List[str],
    day_seqs_raw: np.ndarray,
) -> np.ndarray:
    """为单日计算 24 维 peer dynamics。

    复用 build_v7_peer_data.py 中的 compute_genuine_peers + compute_peer_dynamics。
    """
    from build_peer_dynamics import load_basic_info, compute_genuine_peers, compute_peer_dynamics

    basic_info = load_basic_info()
    peer_idx = compute_genuine_peers(day_codes, day_seqs_raw, basic_info)
    dyn = compute_peer_dynamics(day_seqs_raw, peer_idx)
    return dyn.astype(np.float32)


# ===========================================================================
# Risk (单日, 使用预计算缓存或实时计算)
# ===========================================================================

def compute_risk_for_single_date(
    date_str: str,
    codes: List[str],
) -> np.ndarray:
    """为单日计算风险分数。优先使用 risk_cache，否则实时计算。"""
    risk_cache_path = CACHE_DIR / "risk_cache.pkl"

    # 尝试从缓存加载
    if risk_cache_path.exists():
        with open(risk_cache_path, "rb") as f:
            risk_cache = pickle.load(f)
        if date_str in risk_cache:
            risk_data = risk_cache[date_str]
            code_to_score = dict(zip(risk_data["codes"], risk_data["risk_score"]))
            scores = np.array(
                [code_to_score.get(c, 0.0) for c in codes],
                dtype=np.float32,
            )
            return scores

    # 实时计算
    from src.risk import RiskEstimator
    panel_path = CACHE_DIR / "raw_panel.pkl"
    with open(panel_path, "rb") as f:
        panel = pickle.load(f)

    estimator = RiskEstimator()
    estimator.fit(panel)
    scores = estimator.compute_risk(date_str, codes)
    return scores


# ===========================================================================
# 模型推理
# ===========================================================================

def load_model(path: str = None, init_scale: float = 0.3):
    """加载 V9CA_AB 模型（支持 is0.2/0.3/0.5 等变体）。"""
    if path is None:
        path = DEFAULT_V9CA_AB_PATH

    from src.models.v9ca_ab import V9CA_ABModel

    model = V9CA_ABModel(STAGE1_PATH, init_scale=init_scale)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def run_inference(
    day_seqs: np.ndarray,      # [N, L, 87]
    peer_dyn: np.ndarray,       # [N, 24]
    risk_scores: np.ndarray,    # [N]
    model,
    device: torch.device,
    v9_feat_indices: List[int] = V9_FEATURE_INDICES,
    lambda_risk: float = LAMBDA_RISK,
    top_pct: float = TOP_PCT,
    top_n: int = TOP_N,
    temperature: float = WEIGHT_TEMPERATURE,
) -> Dict:
    """运行一次推理，返回 top-N 结果及权重。"""
    N = day_seqs.shape[0]

    # 切片到 V9 用到的 67 维
    x_full = day_seqs[:, :, v9_feat_indices]  # [N, L, 67]

    # 批量推理
    alpha_scores = np.zeros(N, dtype=np.float32)
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        x = torch.from_numpy(x_full[start:end]).float().to(device)
        d = torch.from_numpy(peer_dyn[start:end]).float().to(device)
        with torch.no_grad():
            s = model(x, d)
        alpha_scores[start:end] = s.cpu().numpy()

    # 两阶段选择
    pool_size = max(int(N * top_pct), top_n)
    pool_idx = np.argpartition(alpha_scores, -pool_size)[-pool_size:]

    alpha_pool = alpha_scores[pool_idx]
    risk_pool = risk_scores[pool_idx]

    def safe_zscore(arr):
        std = np.std(arr)
        if std < 1e-12:
            return np.zeros_like(arr)
        return np.clip((arr - np.mean(arr)) / std, -3, 3)

    z_alpha = safe_zscore(alpha_pool)
    z_risk = safe_zscore(risk_pool)
    final = z_alpha - lambda_risk * z_risk

    # 选出 top-N
    top_n_in_pool = np.argpartition(final, -top_n)[-top_n:]
    # 按 final score 降序排列
    sort_order = np.argsort(final[top_n_in_pool])[::-1]
    top_n_in_pool = top_n_in_pool[sort_order]
    final_indices = pool_idx[top_n_in_pool]

    # 计算权重：softmax on final scores within top-N
    top_final = final[top_n_in_pool]
    weights = np.exp(top_final / temperature)
    weights = weights / weights.sum()

    return {
        "indices": final_indices,              # original indices in [N]
        "alpha_scores": alpha_scores[final_indices],
        "risk_scores": risk_scores[final_indices],
        "final_scores": top_final,
        "weights": weights,
        "pool_size": pool_size,
    }


def print_top_n(
    result: Dict,
    codes: List[str],
    returns: np.ndarray,
    date_str: str,
    top_n: int = TOP_N,
    lambda_risk: float = LAMBDA_RISK,
    top_pct: float = TOP_PCT,
    capital: float = None,
):
    """打印 top-N 推荐结果，包含股票名称和权重。"""
    indices = result["indices"]
    alpha = result["alpha_scores"]
    risk = result["risk_scores"]
    final = result["final_scores"]
    weights = result["weights"]

    names = load_stock_names()

    print(f"\n{'='*80}")
    print(f"  每日推理结果 — {date_str}")
    print(f"  参数: lambda_risk={lambda_risk}, top_pct={top_pct}, N={len(codes)}")
    if capital:
        print(f"  总资金: {capital:,.0f} 元")
    print(f"{'='*80}")
    header = (f"  {'排名':<5} {'代码':<12} {'名称':<10} "
              f"{'Alpha':>8} {'Risk':>8} {'Final':>8} {'权重':>8}")
    if capital:
        header += f" {'金额':>12}  {'手数':>8}"
    print(header)
    print(f"  {'─'*75}")

    avg_ret = 0.0
    total_weight = 0.0
    for rank, (orig_idx, w) in enumerate(zip(indices, weights), 1):
        code = codes[orig_idx]
        name = names.get(code, "")[:8]
        a = alpha[rank - 1]
        r = risk[rank - 1]
        f = final[rank - 1]
        ret = returns[orig_idx]
        avg_ret += ret * w
        total_weight += w
        line = (f"  {rank:<5} {code:<12} {name:<10} "
                f"{a:8.3f} {r:8.3f} {f:8.3f} {w*100:7.2f}%")
        if capital:
            amount = capital * w
            lots = int(amount / 100)  # A股: 1手=100股, 简化用金额÷100
            line += f" {amount:>12,.0f}  {lots:>6}"
        print(line)

    print(f"  {'─'*75}")
    print(f"  总权重: {total_weight*100:.1f}%")
    print(f"  加权平均收益: {avg_ret:+.6f}")
    print(f"  等权平均收益: {float(returns[indices].mean()):+.6f}")
    print(f"{'='*80}")
    return avg_ret


# ===========================================================================
# 主流程
# ===========================================================================

def daily_pipeline(
    date_str: str,
    rebuild: bool = False,
    device: torch.device = None,
    model_path: str = None,
    lambda_risk: float = LAMBDA_RISK,
    top_n: int = TOP_N,
    top_pct: float = TOP_PCT,
    capital: float = None,
    output_csv: str = None,
    temperature: float = WEIGHT_TEMPERATURE,
) -> Dict:
    """完整的日度推理流程（V9CA_AB is0.3）。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = DEFAULT_V9CA_AB_PATH

    print(f"\n{'#'*70}")
    print(f"# Daily Inference Pipeline: {date_str}")
    print(f"# Model: V9CA_AB is0.3  |  Path: {model_path}")
    print(f"{'#'*70}")

    # Step 1: 更新 raw_panel（如果需要）
    print(f"\n[Step 1/6] Panel data...")
    panel = rebuild_raw_panel(force=rebuild)
    panel_dates = sorted(panel["trade_date"].unique())
    print(f"  Latest date in panel: {panel_dates[-1]}")

    # 检查日期是否在面板中
    if date_str not in set(panel_dates):
        # 尝试增量加载
        print(f"  Date {date_str} not in panel, trying incremental load...")
        from src.data.loader import load_daily_panel
        panel = load_daily_panel(start="20160104", end="20261231")
        with open(CACHE_DIR / "raw_panel.pkl", "wb") as f:
            pickle.dump(panel, f, protocol=pickle.HIGHEST_PROTOCOL)
        panel_dates = sorted(panel["trade_date"].unique())
        if date_str not in set(panel_dates):
            raise ValueError(f"Date {date_str} still not found in panel even after full reload")

    # Step 2: 更新 features（如果需要）
    print(f"\n[Step 2/6] Features...")
    norm_path = CACHE_DIR / "normalized_features.parquet"
    if rebuild or not norm_path.exists():
        rebuild_features(panel)
    else:
        # 检查是否需要更新
        norm_df = pd.read_parquet(norm_path)
        norm_dates = sorted(norm_df["trade_date"].unique())
        if date_str not in set(norm_dates) or norm_dates[-1] < panel_dates[-1]:
            print(f"  normalized_features needs update (latest={norm_dates[-1]}, panel={panel_dates[-1]})")
            rebuild_features(panel)
        else:
            print(f"  normalized_features up to date (latest={norm_dates[-1]})")

    # Step 3: 构建推理序列
    print(f"\n[Step 3/6] Building sequence for {date_str}...")
    seqs, returns, codes = build_single_day_sequence(date_str)
    N = len(codes)
    print(f"  {N} stocks ready for inference")

    # Step 4: Peer dynamics
    print(f"\n[Step 4/6] Computing peer dynamics...")
    peer_dyn = compute_peer_for_single_date(codes, seqs)
    print(f"  Peer dynamics: {peer_dyn.shape}")

    # Step 5: Risk
    print(f"\n[Step 5/6] Computing risk scores...")
    risk_scores = compute_risk_for_single_date(date_str, codes)
    print(f"  Risk scores: mean={risk_scores.mean():.3f}, std={risk_scores.std():.3f}")

    # Step 6: Inference
    print(f"\n[Step 6/6] Running V9CA_AB inference...")
    model = load_model(model_path)
    model.to(device)

    result = run_inference(seqs, peer_dyn, risk_scores, model, device,
                           lambda_risk=lambda_risk, top_n=top_n, top_pct=top_pct,
                           temperature=temperature)
    avg_ret = print_top_n(result, codes, returns, date_str,
                         top_n=top_n, lambda_risk=lambda_risk, top_pct=top_pct,
                         capital=capital)

    # 可选输出 CSV
    if output_csv:
        names = load_stock_names()
        rows = []
        for i, (orig_idx, w) in enumerate(zip(result["indices"], result["weights"]), 1):
            code = codes[orig_idx]
            rows.append({
                "rank": i,
                "ts_code": code,
                "name": names.get(code, ""),
                "alpha": result["alpha_scores"][i - 1],
                "risk": result["risk_scores"][i - 1],
                "final_score": result["final_scores"][i - 1],
                "weight": w,
            })
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n  [CSV] Saved → {output_csv}")

    del model
    torch.cuda.empty_cache()

    return {
        "date": date_str,
        "n_stocks": N,
        "avg_return": avg_ret,
        "market_return": float(np.mean(returns)),
        "top_codes": [codes[i] for i in result["indices"]],
        "top_returns": [float(returns[i]) for i in result["indices"]],
        "weights": result["weights"].tolist(),
    }


def simulate_dates(
    dates: List[str],
    rebuild_first: bool = True,
    device: torch.device = None,
    model_path: str = None,
    lambda_risk: float = LAMBDA_RISK,
    top_n: int = TOP_N,
    top_pct: float = TOP_PCT,
    capital: float = None,
    output_csv: str = None,
    temperature: float = WEIGHT_TEMPERATURE,
):
    """在多个日期上依次推理并模拟收益（V9CA_AB is0.3）。

    rebuild_first=True: 只在第一个日期前做一次数据重建。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*70}")
    print(f"# Simulation: {len(dates)} dates  |  Model: V9CA_AB is0.3")
    print(f"# Dates: {dates}")
    print(f"{'#'*70}")

    # 只在第一个日期前重建
    if rebuild_first:
        print("\n[Setup] Rebuilding data once for all dates...")
        panel = rebuild_raw_panel(force=True)
        rebuild_features(panel)

    results = []
    daily_rets = []

    for i, date_str in enumerate(dates):
        r = daily_pipeline(
            date_str,
            rebuild=False,  # 后续日期不需要重建
            device=device,
            model_path=model_path,
            lambda_risk=lambda_risk,
            top_n=top_n,
            top_pct=top_pct,
            capital=capital, temperature=temperature,
        )
        results.append(r)
        daily_rets.append(r["avg_return"])

    # 汇总
    print(f"\n{'='*80}")
    print(f"  Simulation Summary")
    print(f"{'='*80}")
    print(f"  {'Date':<12} {'N Stocks':>10} {'Top-20 Avg Ret':>16} {'Cum Return':>14}")
    print(f"  {'─'*60}")

    cum = 1.0
    for i, (date_str, ret) in enumerate(zip(dates, daily_rets)):
        cum *= (1 + ret)
        cum_ret = cum - 1
        print(f"  {date_str:<12} {results[i]['n_stocks']:>10} {ret:+.8f}      {cum_ret:+.6f}")

    # Market benchmark (equal weight all stocks)
    cum_mkt = 1.0
    for r in results:
        # The full universe's avg return for that date is approximated by
        # the mean of future returns in the sequence
        avg_mkt_ret = float(np.mean([ret for ret in r.get("all_rets", [])]))
        cum_mkt *= (1 + avg_mkt_ret) if abs(avg_mkt_ret) > 1e-9 else 1.0

    # Market benchmark (equal weight all stocks)
    market_rets = [r["market_return"] for r in results]
    cum_mkt = 1.0
    cum_mkt_vals = [1.0]
    for mr in market_rets:
        cum_mkt *= (1 + mr)
        cum_mkt_vals.append(cum_mkt)

    print(f"  {'─'*60}")
    total_ret = cum - 1
    ann_ret = (cum ** (252 / len(dates))) - 1
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    mkt_total = cum_mkt - 1
    excess = total_ret - mkt_total
    print(f"  Total Return:       {total_ret:+.6f}")
    print(f"  Market (EW) Return:  {mkt_total:+.6f}")
    print(f"  Excess over Market:  {excess:+.6f}")
    print(f"  Annualized Return:   {ann_ret:+.6f}")
    print(f"  Sharpe:              {sharpe:+.4f}")
    print(f"{'='*80}")

    return results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Daily inference for stock selection")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYYMMDD) for inference")
    parser.add_argument("--dates", type=str, default=None,
                        help="Comma-separated dates for simulation, e.g. 20260512,20260513,20260514")
    parser.add_argument("--simulate", action="store_true",
                        help="Run simulation mode: track returns across dates")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of raw_panel + normalized_features")
    parser.add_argument("--model-path", type=str, default=None,
                        help="模型 checkpoint 路径（默认：transformer_v9ca_ab_is0p3_best.pt）")
    parser.add_argument("--lambda-risk", type=float, default=LAMBDA_RISK)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--top-pct", type=float, default=TOP_PCT)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--capital", type=float, default=None,
                        help="总资金量（元），用于计算每只股票的持仓金额和手数")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径")
    parser.add_argument("--temperature", type=float, default=WEIGHT_TEMPERATURE,
                        help=f"Softmax temperature for weight allocation (default: {WEIGHT_TEMPERATURE})")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    lambda_risk = args.lambda_risk
    top_n = args.top_n
    top_pct = args.top_pct

    temperature = args.temperature

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",")]
        simulate_dates(dates, rebuild_first=args.rebuild, device=device,
                       model_path=args.model_path,
                       lambda_risk=lambda_risk, top_n=top_n, top_pct=top_pct,
                       capital=args.capital, output_csv=args.output,
                       temperature=temperature)
    elif args.date:
        daily_pipeline(args.date, rebuild=args.rebuild, device=device,
                       model_path=args.model_path,
                       lambda_risk=lambda_risk, top_n=top_n, top_pct=top_pct,
                       capital=args.capital, output_csv=args.output,
                       temperature=temperature)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
