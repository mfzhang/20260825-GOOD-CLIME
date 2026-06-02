"""
risk.py — Risk Estimator: 基于 raw_panel 历史统计量估算每只股票的风险。

风险维度（4 核心）：
  1. realized_vol_20d   — 过去 20 日收益率标准差
  2. max_drawdown_20d    — 过去 20 日最大回撤
  3. amplitude_20d       — 过去 20 日平均 (high-low)/close
  4. liquidity_amount_20d — -log1p(mean(amount)) 成交额不足风险

组合方式：截面上 z-score 标准化后等权相加。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class RiskEstimator:
    """风险估计器。

    对每个交易日，基于 raw_panel 中过去 20 个交易日的数据，计算每只股票的
    4 维风险指标，截面上 z-score 标准化后等权相加得到 risk_score。

    Parameters
    ----------
    vol_weight, dd_weight, amp_weight, liq_weight : float
        各风险指标的权重，默认均为 1.0。
    lookback : int
        计算风险指标的回看窗口（交易日数），默认 20。
    """

    def __init__(
        self,
        vol_weight: float = 1.0,
        dd_weight: float = 1.0,
        amp_weight: float = 1.0,
        liq_weight: float = 1.0,
        lookback: int = 20,
    ):
        self.vol_weight = vol_weight
        self.dd_weight = dd_weight
        self.amp_weight = amp_weight
        self.liq_weight = liq_weight
        self.lookback = lookback

        # 预计算缓存：raw_panel 按 (ts_code, trade_date) 索引
        self._panel: Optional[pd.DataFrame] = None
        self._panel_index: Optional[pd.DataFrame] = None  # pivoted data for fast lookup
        self._all_dates: Optional[np.ndarray] = None
        self._all_codes: Optional[np.ndarray] = None

    def fit(self, panel: pd.DataFrame) -> "RiskEstimator":
        """用 raw_panel 初始化，构建快速查找结构。

        Parameters
        ----------
        panel : pd.DataFrame
            raw_panel，列包含 ts_code, trade_date, open, high, low, close,
            pct_chg, amount 等。
        """
        self._panel = panel.copy()
        self._panel["trade_date"] = pd.to_datetime(
            self._panel["trade_date"], format="%Y%m%d"
        )
        self._all_dates = np.array(sorted(self._panel["trade_date"].unique()))
        self._all_codes = np.array(sorted(self._panel["ts_code"].unique()))

        # 为每只股票构建 date → row 的快速查找
        # 使用 pivot 或 groupby 构建 (ts_code, trade_date) 索引
        self._panel_index = self._panel.set_index(["ts_code", "trade_date"])
        # 排序以加速切片
        self._panel_index = self._panel_index.sort_index()

        return self

    def _get_history(self, code: str, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """获取单只股票在 date 之前（含）的最近 lookback 个交易日数据。"""
        try:
            stock_data = self._panel_index.loc[code]
        except KeyError:
            return None
        if isinstance(stock_data, pd.Series):
            return None
        # 筛选 date 之前的数据
        mask = stock_data.index <= date
        recent = stock_data[mask].tail(self.lookback)
        if len(recent) < self.lookback // 2:  # 至少需要一半数据
            return None
        return recent

    def _compute_stock_risk(
        self, code: str, date: pd.Timestamp
    ) -> Tuple[float, float, float, float]:
        """计算单只股票的 4 维风险指标。"""
        hist = self._get_history(code, date)
        if hist is None or len(hist) < 5:
            return 0.0, 0.0, 0.0, 0.0

        # 1. realized volatility: std of daily returns
        if "pct_chg" in hist.columns:
            rets = hist["pct_chg"].values / 100.0  # pct_chg 是百分比
        else:
            rets = hist["close"].pct_change().dropna().values
        vol = float(np.std(rets)) if len(rets) > 1 else 0.0

        # 2. max drawdown
        closes = hist["close"].values.astype(float)
        running_max = np.maximum.accumulate(closes)
        drawdowns = (running_max - closes) / running_max
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # 3. amplitude: mean((high - low) / close)
        highs = hist["high"].values.astype(float)
        lows = hist["low"].values.astype(float)
        amplitude = float(np.mean((highs - lows) / closes))

        # 4. liquidity: -log1p(mean(amount))
        amounts = hist["amount"].values.astype(float)
        mean_amount = float(np.mean(amounts))
        liq = float(-np.log1p(mean_amount))

        return vol, max_dd, amplitude, liq

    def compute_risk(
        self,
        date: str,
        codes: List[str],
        scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """计算给定日期的所有股票的风险分数。

        Parameters
        ----------
        date : str
            交易日期，格式 "YYYYMMDD"。
        codes : List[str]
            股票代码列表，如 ["000001.SZ", "000002.SZ", ...]。
        scores : np.ndarray or None
            alpha_strength_score（当前未使用，对齐接口）。

        Returns
        -------
        risk_scores : np.ndarray, shape (N,)
            每只股票的综合风险分数（越高 = 越大风险）。
            以及 risk_components: np.ndarray, shape (N, 4)
            各维度的原始风险指标（z-score 之前）。
        """
        date_ts = pd.to_datetime(date, format="%Y%m%d")
        N = len(codes)

        vol_arr = np.zeros(N, dtype=np.float32)
        dd_arr = np.zeros(N, dtype=np.float32)
        amp_arr = np.zeros(N, dtype=np.float32)
        liq_arr = np.zeros(N, dtype=np.float32)

        for i, code in enumerate(codes):
            vol, dd, amp, liq = self._compute_stock_risk(code, date_ts)
            vol_arr[i] = vol
            dd_arr[i] = dd
            amp_arr[i] = amp
            liq_arr[i] = liq

        # 截面上 z-score（处理常数情况）
        def safe_zscore(arr: np.ndarray) -> np.ndarray:
            std = np.std(arr)
            if std < 1e-12:
                return np.zeros_like(arr)
            return (arr - np.mean(arr)) / std

        z_vol = safe_zscore(vol_arr)
        z_dd = safe_zscore(dd_arr)
        z_amp = safe_zscore(amp_arr)
        z_liq = safe_zscore(liq_arr)

        risk_score = (
            self.vol_weight * z_vol
            + self.dd_weight * z_dd
            + self.amp_weight * z_amp
            + self.liq_weight * z_liq
        )

        return risk_score.astype(np.float32)

    def compute_risk_components(
        self, date: str, codes: List[str]
    ) -> Dict[str, np.ndarray]:
        """返回风险各维度的原始值 + z-score + 综合分数，用于诊断。

        Returns
        -------
        dict with keys:
            vol_raw, dd_raw, amp_raw, liq_raw  — 原始指标
            vol_z, dd_z, amp_z, liq_z          — z-score
            risk_score                           — 综合风险分数
        """
        date_ts = pd.to_datetime(date, format="%Y%m%d")
        N = len(codes)

        vol_raw = np.zeros(N, dtype=np.float32)
        dd_raw = np.zeros(N, dtype=np.float32)
        amp_raw = np.zeros(N, dtype=np.float32)
        liq_raw = np.zeros(N, dtype=np.float32)

        for i, code in enumerate(codes):
            vol, dd, amp, liq = self._compute_stock_risk(code, date_ts)
            vol_raw[i] = vol
            dd_raw[i] = dd
            amp_raw[i] = amp
            liq_raw[i] = liq

        def safe_zscore(arr: np.ndarray) -> np.ndarray:
            std = np.std(arr)
            if std < 1e-12:
                return np.zeros_like(arr)
            return (arr - np.mean(arr)) / std

        risk_score = (
            self.vol_weight * safe_zscore(vol_raw)
            + self.dd_weight * safe_zscore(dd_raw)
            + self.amp_weight * safe_zscore(amp_raw)
            + self.liq_weight * safe_zscore(liq_raw)
        )

        return {
            "vol_raw": vol_raw,
            "dd_raw": dd_raw,
            "amp_raw": amp_raw,
            "liq_raw": liq_raw,
            "vol_z": safe_zscore(vol_raw),
            "dd_z": safe_zscore(dd_raw),
            "amp_z": safe_zscore(amp_raw),
            "liq_z": safe_zscore(liq_raw),
            "risk_score": risk_score.astype(np.float32),
        }
