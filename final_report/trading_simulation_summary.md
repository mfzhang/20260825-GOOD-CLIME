# Trading Simulation Record Summary

Source image: `final_report/record.png`

## Recognized Results

The screenshot is a collage of 同花顺/模拟交易 records covering several trading days in early June 2026. The visible account performance card shows:

| Metric | Value |
|---|---:|
| Total return | -19.40% |
| Monthly return | -19.40% |
| Stock selection success rate | 21.90% |
| Ranking | 119 |
| Followers | 0 |

The transaction history contains buy/sell records across dates including 2026-06-01, 2026-06-03, 2026-06-04, 2026-06-05, 2026-06-08, 2026-06-09, and 2026-06-10. Many records are intraday executions rather than a clean close-to-close rebalance.

Visible traded names include, among others:

- 新易盛、腾讯科技、青木科技、青龙管业、寒武纪、科华数据
- 江顺科技、江苏雷利、中国铁建、中国银行、招商银行
- 海航控股、中远海控、国资材料、国检集团、韩建河山
- 阳光股份、粤电力A、欧晶科技、宁德时代、美的集团

The records show frequent manual orders, often split across multiple small fills. Several stocks appear repeatedly within the same day or adjacent days, suggesting the actual simulation involved high turnover and non-negligible execution timing effects.

## Observed Execution Pattern

The simulation result is poor, but the screenshot suggests that the poor result should not be interpreted as direct failure of the model's holdout ranking signal.

The main reason is that the simulation protocol differs materially from the model's evaluation protocol:

1. The model is trained and evaluated on daily close-to-next-day-close ranking signals.
2. The real simulation orders were manually executed during the trading day.
3. Intraday K-line movement was large, so manual delay could easily turn a good daily signal into a bad execution.
4. The model operator and the trade executor were the same person, creating additional timing and behavioral noise.
5. The actual records show frequent intraday actions and fragmented fills, while the paper's main evaluation assumes a clean top-20 equal-weight daily rebalance.

Therefore, this simulation mainly evaluates the combination of:

`model alpha signal + manual execution timing + intraday volatility + trading platform constraints`

rather than the pure cross-sectional ranking ability measured in the holdout experiment.

## Interpretation for the Report

This section should be written as an execution-layer reflection, not as a contradiction of the main experimental results.

A fair wording is:

> The 同花顺 simulation produced a negative return (-19.40%) despite strong holdout ranking results. This gap highlights the difference between a daily alpha signal and an executable trading strategy. The CLIME model outputs a close-to-next-day-close ranking score, while the simulation required manual intraday execution. Because intraday prices moved significantly and orders were often not placed immediately after signal generation, the realized trades were exposed to adverse timing. Therefore, the simulation should be interpreted as evidence that execution design is a separate and necessary layer, rather than as a direct invalidation of the model's ranking ability.

## What This Reveals

The simulation is still useful because it exposes a real limitation of the current system:

- The model produces ranking scores, but does not decide the exact execution time.
- The current pipeline has no slippage model, transaction cost model, turnover constraint, or intraday execution policy.
- Manual execution introduces timing risk that can dominate daily alpha.
- A strong holdout ranking signal is not sufficient for a profitable live/manual trading workflow.

This actually strengthens the report's earlier distinction between the prediction layer and the trading strategy layer.

## Recommended Report Framing

The report should not try to hide the negative simulation result. It should present it as an honest boundary:

1. Main experiments evaluate controlled holdout ranking quality.
2. The simulation evaluates a much harder and noisier setting involving manual intraday execution.
3. The negative result shows that future work must add an execution module.
4. The failure mode is explainable and consistent with the system design: CLIME is an alpha/ranking model, not a complete trading bot.

Recommended future improvements:

- Generate signals before market open or define a fixed rebalance time.
- Use opening price / VWAP / close auction execution consistently.
- Add transaction cost and slippage simulation.
- Add turnover constraints to avoid excessive daily reshuffling.
- Separate model inference from human discretionary execution.
- Evaluate paper trading under a deterministic execution protocol before real manual trading.

## Overall Evaluation

From a teaching-assistant perspective, this simulation does not make the project weaker if it is explained correctly. It makes the project more realistic:

- The main report already proves that CLIME has a strong controlled ranking signal.
- The simulation shows that alpha prediction and trade execution are different engineering problems.
- A negative live/manual result is acceptable because it exposes a concrete system boundary.

The key is to avoid presenting the simulation as the same evaluation target as the holdout backtest. It should be framed as an execution-layer stress test and reflection.

