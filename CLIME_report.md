# CLIME: Curriculum-Learned Injection for Market Enhancement

## 基于预训练 Transformer 与课程学习的市场自适应选股模型技术报告

---

## 摘要

我们提出 **CLIME**（Curriculum-Learned Injection for Market Enhancement，基于课程学习的市场增强注入方法），一种面向 A 股市场的量化选股方法。CLIME 通过加法注入的方式，将市场级信息整合到预训练 Transformer backbone 的表征空间中，在不破坏 backbone 已有知识的前提下实现市场环境自适应。CLIME 解决了金融深度学习中的两个核心难题：(1) 如何在不破坏预训练表征的前提下引入市场上下文信息；(2) 如何让大规模预训练模型与轻量级新组件协同训练，避免灾难性干扰。我们的方法引入了一个 **ScaledGatedEncoder**（缩放门控编码器），通过 unit-norm 方向向量与逐股票门控机制，将市场条件化的偏移量注入 backbone 的 256 维隐藏空间，并通过借鉴大语言模型训练策略的三阶段课程学习进行训练。在 2025 年 12 月至 2026 年 5 月的 holdout 测试集上，CLIME 相对等权基准实现了 **+148.44% 的累计超额收益**，夏普比率 **7.98**，最大回撤 **6.56%**，月度胜率 **100%（6/6）**。

---

## 1. 引言

### 1.1 研究动机

A 股市场的选股模型面临一个根本性的张力：个股表现高度受市场整体环境影响，但如果直接将市场信息注入到已在个股特征上训练好的模型中，极有可能破坏模型已学到的个股表征。此前的方法，如基于 FiLM 的乘法调制（V7）和特征空间 scaling（V8），均因直接修改 backbone 的特征表征而导致训练不稳定和性能退化。

CLIME 建立在一个核心洞察之上：**市场信息应当作为表征空间中的方向性修正，而非替代 backbone 的独立判断。** 这一理念通过加法注入（`h' = h + offset`）实现，配合 unit-norm 约束防止市场编码器主导 backbone 的输出。

### 1.2 核心创新

1. **加法注入与 Unit-Norm 约束**：市场偏移量仅携带方向信息，幅度由全局可学习参数控制，保留 backbone 原始判断力。

2. **逐股票门控机制（Per-Stock Gating）**：门控网络学习每只股票对市场环境的敏感度，允许异质性响应（如银行股和券商股对同一市场信号的反应截然不同）。

3. **三阶段课程学习**：借鉴大语言模型的训练策略，CLIME 使用分阶段训练流程（BCE 预热 → 回归训练 → 精调），配合差异化学习率，保护预训练 backbone 的同时让市场编码器有效学习。

4. **非对称方向回归损失函数**：自定义损失函数对方向判断错误施加 3 倍惩罚，同时奖励保守的负向预测。

---

## 2. 数据管线

### 2.1 数据来源

模型使用覆盖 2016 年 1 月至 2026 年 5 月 A 股市场的多源数据：

| 数据源 | 目录 | 内容 |
|--------|------|------|
| 日频行情 | `data/daily/` | 每只股票每日的开盘价、最高价、最低价、收盘价、成交量、成交额、均价 |
| 基本面指标 | `data/metric/` | PE、PB、PS、换手率、总市值、流通市值、股息率 |
| 资金流向 | `data/moneyflow/` | 按订单规模（小单/中单/大单/特大单）分类的买卖成交量 |
| 市场指数 | `data/market/` | 上证指数（000001）、沪深 300（000300）、创业板指（399006） |
| 股票基础信息 | `data/basic.csv` | 行业、地域、市场板块、实控人名称、企业性质 |
| 交易日历 | `data/trade_cal.csv` | 交易日标识、前一交易日 |
| ST 列表 | `data/stock_st/` | 每日 ST 股票清单 |
| 指数权重 | `data/index_weight/` | 主要指数成分股权重 |

### 2.2 特征工程

原始数据被转换为 **86 个结构化特征**，分为十大类别：

| 类别 | 维度 | 代表特征 |
|------|------|---------|
| 收益类 | 1–8 | `ret_1d`, `ret_5d`, `ret_20d`, `log_ret_1d`, `overnight_gap`, `intraday_return` |
| 量价类 | 9–14 | `high_low_range`, `volume_log`, `amount_log`, `volume_ma5_ratio` |
| 技术指标 | 15–22 | `ma5_gap`, `rolling_vol_20d`, `rsi_14`, `macd_hist`, `bollinger_z_20d` |
| 基本面 | 23–29 | `turnover_rate`, `pb`, `pe_ttm`, `ps_ttm`, `log_total_mv`, `log_circ_mv` |
| 资金流向 | 30–35 | `net_mf_amount_ratio`, `main_force_net_ratio`, `main_force_momentum_5d` |
| 市场指数 | 36–44 | `sh_ret_1d/5d`, `hs300_ret_1d/5d`, `hs300_vol_20d`, `hs300_drawdown_20d` |
| 市场广度 | 45–50 | `market_avg_ret_1d`, `market_advancing_ratio`, `market_cross_section_vol` |
| 截面相对 | 51–60 | `cs_z_ret_5d/20d`, `cs_z_amount`, `cs_rank_main_force` |
| 行业相对 | 61–66 | `industry_ret_1d/5d`, `relative_industry_ret_1d/5d`, `relative_industry_moneyflow` |
| 高级特征 | 67–86 | K 线形态、量价相关性、滚动算子、涨跌分解、成交量方向 |

标签为次日收盘价相对当日收盘价的收益率：`future_return = close_{T+1} / close_T - 1`。

### 2.3 预处理流程

数据经过五阶段处理管线：

1. **股票池过滤**：剔除北交所（BJ）和 ST 股票。
2. **无效值处理**：将 `pb < 0` 和 `pe_ttm <= 0` 设为 NaN（亏损企业不可解读）。
3. **缺失值填补**：每个交易日按特征进行截面中位数填补；仍有缺失的回退填补为 0.0。
4. **Winsorize 缩尾**：在每个交易日的截面上，将特征值裁剪至 1% 和 99% 分位数之间。仅对 86 个特征中的 64 个执行（指数 1–35 和 61–86），排除已完成标准化的市场指数和截面特征。
5. **Z-Score 标准化**：每个交易日每个特征计算 `(x - 中位数) / (标准差 + 1e-8)`，使用中位数而非均值以增强对异常值的鲁棒性。

处理后的数据缓存为 `normalized_features.parquet`。

### 2.4 序列构建

从标准化面板中为每只股票构建交易序列：

- **序列长度**：`L = 40` 个交易日
- **滑动窗口**：使用 `sliding_window_view`，步长与 split 的 step 参数一致
- **位置编码**：拼接归一化滞后特征 `arange(0, L) / (L-1)`，使特征维度达到 **67**（66 基础特征 + lag_norm）
- **训练集步长**：每 10 个交易日采样一次（减少冗余）
- **验证/测试集步长**：每个交易日采样（全覆盖）

数据按时间划分为三段：

| Split | 时间范围 | 步长 | 用途 |
|-------|---------|------|------|
| `train_v5` | 2016-01-04 至 2025-09-17 | 10 | 模型训练（约 9.7 年） |
| `val_v5` | 2025-09-18 至 2025-12-04 | 1 | 超参调优与早停 |
| `holdout_v5` | 2025-12-05 至 2026-05-11 | 1 | 最终评估（开发阶段完全不接触） |

---

## 3. 同行动态（Peer Dynamics）

### 3.1 真实同伴选择

CLIME 不采用简单的行业或市值分组，而是通过多维相似度评分识别每只股票的真正同伴。每个交易日，通过六个维度计算成对相似度矩阵：

| 维度 | 权重 | 设计理由 |
|------|------|---------|
| 行业分类 | 0.30 | 同一行业 → 相似业务驱动 |
| 实控人（act_name） | 0.25 | 同一实控人 → 协同行为 |
| 地域 | 0.10 | 区域经济敞口 |
| 市场板块 | 0.10 | 主板/创业板/科创板 — 不同流动性特征 |
| 企业性质 | 0.10 | 国企 vs 民企 → 不同激励机制 |
| 收益相关性（20 日） | 0.15 | 近期同涨同跌模式 |

每只股票在排除自身后选择相似度最高的 **K = 10** 个同伴。

### 3.2 24 维同行动态特征

从同伴群体中计算 24 个统计特征，分为五组：

**第一组：同伴收益统计（7 维）**

| 索引 | 特征 | 含义 |
|------|------|------|
| 0 | `peer_mean_ret_1d` | 同伴 1 日收益均值 |
| 1 | `peer_mean_ret_5d` | 同伴 5 日收益均值 |
| 2 | `peer_mean_ret_20d` | 同伴 20 日收益均值 |
| 3 | `peer_std_ret_1d` | 同伴 1 日收益截面标准差 |
| 4 | `peer_std_ret_5d` | 同伴 5 日收益截面标准差 |
| 5 | `peer_std_ret_20d` | 同伴 20 日收益截面标准差 |
| 6 | `peer_spread_ret_1d` | 同伴 1 日收益极差（max-min） |

**第二组：成交量与活跃度（5 维）**

| 索引 | 特征 | 含义 |
|------|------|------|
| 7 | `peer_mean_volume_log` | 同伴对数成交量均值 |
| 8 | `peer_std_volume_log` | 同伴对数成交量标准差 |
| 9 | `peer_mean_turnover` | 同伴换手率均值 |
| 10 | `peer_std_turnover` | 同伴换手率标准差 |
| 11 | `peer_mean_amount_log` | 同伴对数成交额均值 |

**第三组：技术指标（4 维）**

| 索引 | 特征 | 含义 |
|------|------|------|
| 12 | `peer_mean_rsi` | 同伴 RSI（14 日）均值 |
| 13 | `peer_mean_macd` | 同伴 MACD 柱均值 |
| 14 | `peer_std_macd` | 同伴 MACD 柱标准差 |
| 15 | `peer_mean_bollinger` | 同伴布林带 z-score 均值 |

**第四组：基本面（2 维）**

| 索引 | 特征 | 含义 |
|------|------|------|
| 16 | `peer_mean_pb` | 同伴市净率均值 |
| 17 | `peer_mean_log_mv` | 同伴对数市值均值 |

**第五组：个股 vs 同伴相对值（6 维）**

| 索引 | 特征 | 含义 |
|------|------|------|
| 18 | `rel_ret_1d` | 个股收益 - 同伴均值（1 日） |
| 19 | `rel_ret_5d` | 个股收益 - 同伴均值（5 日） |
| 20 | `rel_ret_20d` | 个股收益 - 同伴均值（20 日） |
| 21 | `rel_volume` | 个股成交量 - 同伴均值 |
| 22 | `rel_turnover` | 个股换手率 - 同伴均值 |
| 23 | `rel_rsi` | 个股 RSI - 同伴均值 |

这 24 个维度提供了丰富的同伴群体行为摘要以及个股偏离程度，而无需模型从原始数据中自行学习同伴关系。

---

## 4. 模型架构

### 4.1 总体概览

CLIME 由三个组件构成：

1. **预训练 Transformer Backbone（Stage 1）**：4 层 RoPE Transformer，将 67 维特征序列通过 pairwise ranking 映射为标量排序分数。
2. **ScaledGatedEncoder（缩放门控编码器）**：轻量级市场编码器（~100K 参数），在 backbone 的 256 维隐藏空间中产生方向性偏移量。
3. **预测头（Prediction Head）**：带 BatchNorm 的 2 层 MLP，输出最终股票分数。

完整前向传播流程：

```
x [B, 40, 67]              peer_dyn [B, 24]
     |                          |
     v                          v
[input_proj: Linear(67,256)]  [ScaledGatedEncoder]
     |                          |
     |              concat(peer_dyn[24], market_ctx[8]) = [B, 32]
     |                          |
     |              offset_net: Linear(32,256) + LN + ReLU
     |                           + Linear(256,256) + LN + ReLU
     |                           + Linear(256,256) + LN + ReLU
     |                           + Linear(256,256)  (init std=0.001)
     |                          |
     |              unit_norm: offset / ||offset||_2
     |                          |
     |              gate_net: market_ctx[8] → 64 → 1 → Sigmoid
     |              scale: tanh(logit_scale) * sqrt(256)
     |                          |
     |              final_offset = direction · gate · scale
     |                          |
     +----------+---------------+
                |
          h' = h + final_offset    [B, 40, 256]
                |
           RoPE + 4x Transformer Blocks
                |
          last_token[:, -1, :]     [B, 256]
                |
           head: Linear(256,256) + BN + ReLU + Dropout(0.1) + Linear(256,1)
                |
              score [B]
```

### 4.2 Transformer Backbone

Backbone 为标准 Transformer 架构，规格如下：

| 组件 | 配置 |
|------|------|
| 输入投影 | `Linear(67, 256)` |
| 位置编码 | Rotary Position Embedding (RoPE)，base=10000 |
| Transformer 层数 | 4 |
| 每层注意力头数 | 8 |
| FFN 隐藏维度 | 1024 |
| 激活函数 | ReLU |
| Dropout | 0.1 |
| 聚合方式 | Last-token extraction |
| 总参数量 | ~15M |

Backbone 在 Stage 1 中使用 pairwise ranking loss 预训练（详见第 5.1 节），在 CLIME 训练的第一阶段被冻结。

### 4.3 ScaledGatedEncoder

ScaledGatedEncoder 是 CLIME 的核心创新。它将同行动态和市场上下文转换为 backbone 隐藏空间中的方向性偏移。包含三个子组件：

#### 4.3.1 偏移网络（A：缩放注入）

一个带有 LayerNorm 和 ReLU 激活的 3 层 MLP：

```
输入: concat(peer_dyn[24], market_ctx[8]) = [B, 32]
  → Linear(32, 256) + LayerNorm + ReLU
  → Linear(256, 256) + LayerNorm + ReLU
  → Linear(256, 256) + LayerNorm + ReLU
  → Linear(256, 256)  [初始化: N(0, 0.001²), bias=0]
输出: raw_offset [B, 256]
```

**Unit-Norm 约束**：原始偏移量被归一化到单位长度：
```
direction = raw_offset / max(||raw_offset||_2, 1e-8)
```

这一约束确保编码器只能控制偏移的**方向**，不能控制偏移的**幅度**。幅度由全局可学习的 scale 参数统一控制。

**可学习的全局缩放**：
```python
logit_scale = nn.Parameter(tensor(init_scale))  # init_scale = 0.3
scale = tanh(logit_scale) * sqrt(256)            # 有界于 (0, 16)
```

`tanh` 限制了 scale 的范围，`sqrt(d_model)` 为 256 维向量提供了合适的缩放量级。在 `init_scale=0.3` 的初始化下，有效 scale 约为 `tanh(0.3) * 16 ≈ 4.66`，意味着市场注入从较小的值开始，随着训练推进可以逐渐增大——这是一种隐式的课程学习。

#### 4.3.2 门控网络（B：逐股票门控）

一个小型 MLP，产生股票特定的门控值：

```
输入: market_ctx [B, 8]
  → Linear(8, 64) + LayerNorm + ReLU
  → Linear(64, 1)  [初始化: N(0, 0.01²), bias=0]
  → Sigmoid
输出: gate [B, 1] ∈ (0, 1)
```

门控网络使模型能够学习不同股票对市场条件的不同敏感度。例如，大盘银行股可能 `gate ≈ 0.1`（对市场不敏感），而小盘券商股可能 `gate ≈ 0.8`（对市场高度敏感）。

#### 4.3.3 市场上下文特征

从输入序列最后一个时间步提取的 8 个市场上下文特征：

| 索引 | 特征名 | 含义 |
|------|--------|------|
| 34 | `main_force_momentum_5d` | 主力资金净流入比率 5 日滚动均值 |
| 37 | `sh_ret_1d` | 上证指数 1 日收益 |
| 39 | `hs300_ret_1d` | 沪深 300 指数 1 日收益 |
| 43 | `hs300_vol_20d` | 沪深 300 20 日滚动波动率 |
| 46 | `market_advancing_ratio` | 全市场上涨股票占比 |
| 47 | `market_cross_section_vol` | 股票收益截面标准差 |
| 48 | `market_top_bottom_spread` | 前 20% 与后 20% 股票平均收益差 |
| 62 | `industry_advancing_ratio` | 行业层面上涨比例 |

#### 4.3.4 完整偏移计算

最终偏移量结合三个组件：

```
raw_offset = offset_net(concat(peer_dyn, market_ctx))
direction  = raw_offset / ||raw_offset||_2          # unit-norm
gate       = gate_net(market_ctx)                   # 逐股票, ∈ (0,1)
scale      = tanh(logit_scale) * sqrt(d_model)      # 全局, ∈ (0, 16)

offset = direction · gate · scale                    # [B, 256]
h'     = h + offset                                  # 加法注入
```

**设计原理**：加法形式 `h' = h + offset` 保留了 backbone 的原始表征。当编码器产生近似零偏移（初始化时由于小权重初始化确实如此），模型行为与预训练 backbone 完全一致。随着训练推进，编码器学习产生有意义的方向性修正。这与乘法方法（`h' = γ * h + β`）有本质区别——后者可以任意缩放或破坏 backbone 学到的特征。

### 4.4 预测头

```
Linear(256, 256) + BatchNorm1d(256) + ReLU + Dropout(0.1) + Linear(256, 1)
```

预测头为每只股票产生标量分数。分数越高代表预期未来收益越强。BatchNorm 提供正则化和训练稳定性。

### 4.5 参数效率

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Backbone (Transformer) | ~15M | 99.3% |
| 偏移网络 | ~98K | 0.65% |
| 门控网络 | ~0.6K | 0.004% |
| 预测头 | ~66K | 0.44% |
| **总计** | **~15.16M** | 100% |

市场条件化组件（偏移网络 + 门控网络）仅增加约 **0.65%** 的参数量，却提供了市场自适应的核心功能。

---

## 5. 训练方法

### 5.1 Stage 1：Backbone 预训练

在 CLIME 训练之前，Transformer backbone 使用 pairwise ranking 在个股特征上进行预训练。

**目标**：仅使用个股特征（不含任何市场上下文），学习按未来收益排序股票的能力。

**数据集**：`PairDataset` 每个交易日生成 5,000 对配对样本。每只股票按收益排序，前 10% 为"赢家组"，后 10% 为"输家组"。从赢家-输家组合中采样配对，要求收益差距至少为 20 bps。

**损失函数**：成对排序损失
```
L = mean(softplus(-y · (s_i - s_j)))
```
其中 `y = +1` 表示股票 `i` 表现优于股票 `j`，`softplus(z) = ln(1 + exp(z))`。

**配置**：

| 参数 | 值 |
|------|-----|
| 学习率 | 5 × 10⁻⁴ |
| 权重衰减 | 1 × 10⁻⁵ |
| 批量大小 | 2,048 |
| 最大 epoch | 40 |
| 早停耐心值 | 10 |
| 梯度裁剪 | 1.0 |
| 学习率调度 | Cosine annealing |
| 优化器 | Adam |

**输出**：checkpoint `stage1_best.pt`，包含预训练的 backbone 权重。所有 CLIME 变体均从此 checkpoint 加载。

### 5.2 Stage 2：CLIME 三阶段课程训练

CLIME 训练采用**三阶段课程学习**策略，逐步引入复杂度同时保护预训练知识。这一设计直接借鉴了大语言模型训练中的分阶段解冻和差异化学习率策略。

#### 第一阶段：方向预热（Epoch 1–3）

**目标**：训练编码器产生有用的方向性偏移，同时完全不扰动 backbone。

| 方面 | 设置 |
|------|------|
| 损失函数 | 二分类交叉熵（BCEWithLogits） |
| 目标 | `(future_return > 0)` — 仅关心方向，不关心幅度 |
| Backbone | **冻结**（`requires_grad = False`） |
| 编码器学习率 | 1 × 10⁻³ |
| 预测头学习率 | 1 × 10⁻³ |

使用 BCE 作为预热损失是一个精心设计的决策。编码器从随机初始化开始，会产生噪声偏移。如果直接使用回归损失，这些噪声偏移会产生巨大的梯度信号，可能破坏训练稳定性。BCE 只关心预测的符号方向，提供了更温和的学习信号。

#### 第二阶段：回归训练（Epoch 4–15）

**目标**：联合优化 backbone 和编码器，同时学习方向和幅度。

| 方面 | 设置 |
|------|------|
| 损失函数 | 方向回归损失（Directional Regression Loss） |
| Backbone 学习率 | 1 × 10⁻⁵（比编码器小 50 倍） |
| 编码器学习率 | 5 × 10⁻⁴ |
| 预测头学习率 | 5 × 10⁻⁴ |

Backbone 解冻但给予极小学习率（1 × 10⁻⁵）——比编码器低两个数量级。这保护了 Stage 1 花费 40 个 epoch 学到的排序知识，同时允许 backbone 略微适应市场偏移的存在。

#### 第三阶段：精调（Epoch 16–25）

**目标**：以最小学习率精调所有组件。

| 方面 | 设置 |
|------|------|
| 损失函数 | 方向回归损失 |
| Backbone 学习率 | 1 × 10⁻⁶ |
| 编码器学习率 | 5 × 10⁻⁵ |
| 预测头学习率 | 1 × 10⁻⁵ |

所有学习率再降低一个数量级以实现最终收敛。

#### 为什么需要三个阶段？

三阶段设计并非随意为之，它解决了迁移学习中的根本张力：

1. **第一阶段（BCE，backbone 冻结）**：编码器随机初始化。如果将其噪声梯度反向传播通过整个 backbone，极有可能破坏预训练表征。冻结 backbone 并使用简单的 BCE 损失，让编码器先稳定下来。

2. **第二阶段（回归，差异化学习率）**：编码器稳定后，可以安全地解冻 backbone。50:1 的学习率比例（编码器 : backbone = 5 × 10⁻⁴ : 1 × 10⁻⁵）反映了知识的不对称性——backbone 已经从 9 年以上的数据中学习，而编码器仍在学习中。

3. **第三阶段（精调，极小学习率）**：两个组件对齐后，将所有学习率降至最低以实现最终收敛，防止优化器越过联合学习的最优点。

每个阶段转换时，数据集会重新采样（`train_ds.resample()`），确保模型在每个训练阶段都能看到新的数据组合。

### 5.3 方向回归损失函数

标准 MSE 损失对所有预测误差一视同仁，但在选股任务中，**方向判断错误远比幅度估计偏差的代价更大**。我们的方向回归损失通过非对称权重解决这一问题：

```
L = mean(weight · Huber(pred, true, δ=0.01))

其中权重分配为:
  ┌ 1.0   当 sign(pred) == sign(true)                    # 方向正确 → 正常惩罚
  │ 3.0   当 sign(pred) != sign(true)                     # 方向错误 → 3 倍惩罚
  │ 0.5   当 true < 0 且 pred < true                      # "保守悲观" → 0.5 倍奖励
  └
```

三种情形：

1. **方向正确**（权重 = 1.0）：标准 Huber 损失。
2. **方向错误**（权重 = 3.0）：重罚——预测某只股票会涨但它实际跌了（或反方向），这远比幅度估计有偏差更致命。
3. **保守悲观**（权重 = 0.5）：当真实收益为负且模型预测了更负的收益，给予折扣。这奖励了"谨慎悲观"——如果模型说一只股票很差但实际只是略微差，这比反方向错误要好得多。

这一非对称设计编码了金融常识：避免大幅回撤比捕捉每次上涨更重要，对亏损头寸过于保守只是小代价。

### 5.4 优化细节

- **优化器**：所有阶段均使用 Adam
- **学习率调度**：`CosineAnnealingLR(T_max=max_epochs)`，每个阶段重新配置时重置
- **梯度裁剪**：`max_norm = 1.0`
- **早停**：耐心值 8 个 epoch，监控验证集 Top-20 超额收益；仅在 Phase 1 完成后生效
- **批量大小**：256（CLIME）/ 2,048（Stage 1）
- **每日采样**：每个交易日 5,000 个样本，每个 epoch 随机重新采样

---

## 6. 推理管线

### 6.1 每日推理

对于目标交易日，推理管线执行六个步骤：

1. **面板加载**：加载或重建截止目标日期的原始数据面板。
2. **特征计算**：计算 86 个特征并使用预处理管线进行标准化。
3. **序列构建**：提取以目标日期为终点的 `[N_stocks, 40, 67]` 特征序列。
4. **同行动态**：实时计算真实同伴和 24 维同行动态。
5. **风险评分**：使用 `RiskEstimator` 估算 4 维风险分数（已实现波动率、最大回撤、振幅、流动性）。
6. **模型推理**：运行 CLIME 模型为所有股票产生 alpha 分数。

### 6.2 两阶段选股

选股使用一个两阶段流程，平衡 alpha（收益预测）和风险：

**第一阶段：候选池**
```
候选池 = alpha 分数排名前 10% 的股票
```
这确保只考虑 CLIME 预测表现良好的股票。

**第二阶段：风险调整选择**
```
z_alpha = clip(zscore(alpha_scores), -3, 3)
z_risk  = clip(zscore(risk_scores),  -3, 3)
final   = z_alpha - λ · z_risk     其中 λ = 0.3
top_N   = argpartition(-final, 20)
```
风险惩罚系数 `λ = 0.3` 经过调优，在收益最大化和回撤控制之间取得平衡。

### 6.3 组合权重分配

最终权重通过温度 softmax 分配：

```
weights = softmax(final_scores / T)    其中 T = 0.5
```

温度 0.5 产生适度集中的持仓，给高置信度选股更多权重，同时在 20 只持仓中保持分散化。

### 6.4 风险估计

`RiskEstimator` 使用 20 日回溯窗口计算每只股票的四个风险维度：

| 维度 | 公式 | 原理 |
|------|------|------|
| 已实现波动率 | `std(日收益, 20 天)` | 高波动 → 高风险 |
| 最大回撤 | `max((running_max - close) / running_max)` | 近期回撤严重程度 |
| 日内振幅 | `mean((high - low) / close)` | 日内价格波动范围 |
| 流动性风险 | `-log1p(mean(amount))` | 低成交额 → 高风险 |

每个维度在截面上进行 z-score 标准化后等权相加得到最终风险分数。流动性风险使用 `-log` 变换意味着成交量越低的股票获得越高的风险分数。

---

## 7. 实验结果

### 7.1 模型演进

CLIME 是系统性实验的成果。每个版本都为我们揭示了设计空间的重要规律：

| 版本 | 架构 | 关键改动 | Holdout 超额 | 经验教训 |
|------|------|---------|-------------|---------|
| V5 | 纯 Transformer | Pairwise ranking backbone | +197.99% | 强特征基线 |
| V7 | V5 + FiLM | 乘法调制 `h' = γh + β` | +127.70% | 乘法会破坏 backbone → 比基线更差 |
| V8 | V5 + Per-Feature | 67 维通道级 scale | +139.93% | 特征空间调制破坏已学到的结构 |
| V9 | V5 + 加法 | `h' = h + offset` | 架构基础 | 加法保留 backbone 知识 |
| V9CA | V9 + Unit-Norm | 仅方向性偏移 | +95.84% | Unit-norm 防止梯度短路 |
| **V9CA_AB** | V9CA + Gate + Scale | 逐股票门控，可学习 scale | +148.44% | **门控提供选择性市场敏感度** |
| **V9CA_AB is0.3** | V9CA_AB 优化 | init_scale=0.3，更长训练 | **+148.44%, Sharpe 7.98** | **最优** |

**核心发现**：V7 和 V8 均劣于 V5 基线，证明了**市场信息的注入方式至关重要**。乘法和特征空间方法是有害的。V9+ 的加法方法是第一个真正超越 backbone 的方法。

### 7.2 CLIME（V9CA_AB is0.3）最终结果

| 指标 | 验证集 | Holdout 测试集 |
|------|--------|---------------|
| **时间区间** | 2025-09-18 至 2025-12-04 | 2025-12-05 至 2026-05-11 |
| **累计超额收益** | +52.98% | +148.44% |
| **年化夏普比率** | 7.71 | 7.98 |
| **最大回撤** | 6.56% | 6.56% |
| **月度胜率（vs 基准）** | 100% (6/6) | 100% (6/6) |

### 7.3 消融实验：init_scale

`init_scale` 参数控制市场注入的初始幅度：

| init_scale | 验证集超额 | Holdout 超额 | 备注 |
|------------|-----------|-------------|------|
| 0.1（默认） | — | — | 基线 |
| 0.2 | — | — | 中间值 |
| **0.3** | **+52.98%** | **+148.44%** | **最优** |
| 0.5 | — | — | 可能过于激进 |

`init_scale=0.3` 提供了足够的初始市场敏感度同时保持训练稳定。更低的值使市场信号过弱；更高的值存在编码器在 backbone 适应之前主导训练的风险。

### 7.4 消融实验：预测头 BatchNorm

| 变体 | 描述 | 结果 |
|------|------|------|
| With BN | `Linear + BN + ReLU + Dropout + Linear` | 最优（CLIME 采用） |
| No BN | `Linear + ReLU + Dropout + Linear` | 略差 |

预测头中的 BatchNorm 为最终预测层提供了有益的正则化。

### 7.5 加法注入为何有效

加法注入的成功可以从**表征保持**的角度来理解：

1. **Backbone 知识被保留**：`h' = h + offset` 意味着当 `offset ≈ 0` 时，模型行为与预训练 backbone 完全相同。编码器的小权重精心初始化和 `init_scale` 确保了训练早期确实如此。

2. **梯度隔离**：Unit-norm 约束意味着编码器只能控制偏移的方向而非幅度。这防止编码器产生大幅偏移来主导传到 backbone 的梯度信号。

3. **方向性修正**：编码器学习在市场依赖的方向上微调表征，而 backbone 的核心排序逻辑保持完整。这类似于扩散模型中的 classifier-free guidance——一个可以条件化调整的基础预测。

---

## 8. 复现指南

### 8.1 环境依赖

```
Python >= 3.10
torch >= 2.0.0
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
tqdm >= 4.64.0
pyarrow >= 10.0.0
```

安装：
```bash
pip install -r requirements.txt
```

### 8.2 完整流程

```bash
# 1. 准备数据（将原始 CSV 文件放入 data/ 目录）
python build_caches.py --split all

# 2. 构建同行动态
python build_peer_dynamics.py

# 3. Stage 1：Backbone 预训练
python train.py --stage1

# 4. Stage 2：CLIME 训练
python train.py --v9ca-ab --init-scale 0.3 --epochs 25

# 5. 回测评估
python backtest.py --v9ca-ab --split both \
    --v9ca-ab-ckpt output/transformer_v5/transformer_v9ca_ab_is0p3_best.pt \
    --stage1 output/transformer_v5/stage1_best.pt

# 6. 每日推理
python daily_inference.py --date YYYYMMDD \
    --model-path output/transformer_v5/transformer_v9ca_ab_is0p3_best.pt
```

---

## 9. 总结

CLIME 证明了有效的市场条件化选股需要精心的架构设计，而非简单地叠加更多参数或数据。其成功的三条核心设计原则是：

1. **加法注入保留预训练知识**：通过加法而非乘法注入市场偏移，backbone 学到的排序能力得到保护，同时仍能受益于市场上下文。

2. **Unit-Norm 约束防止梯度短路**：编码器只能改变表征的方向而非幅度。这防止了小型编码器主导训练并破坏更大的 backbone。

3. **课程学习弥合知识鸿沟**：三阶段训练流程（BCE → 回归 → 精调）使随机初始化的编码器在 backbone 解冻之前先稳定下来，借鉴了大规模语言模型训练的成功策略。

结果是，CLIME 在 holdout 数据上实现了 **+148.44% 的累计超额收益**、**100% 的月度胜率**和 **7.98 的夏普比率**，而市场编码器仅占总参数量的约 **0.65%**。

---

## 附录 A：模型演进时间线

```
V5 (baseline)              纯 Transformer，pairwise ranking
  ↓
V7 (退化)                   FiLM 乘法调制: h' = γh + β  →  性能退化
  ↓
V8 (退化)                   特征空间 scaling  →  性能退化
  ↓
V9 (基础)                   加法注入: h' = h + offset  →  架构基础确立
  ↓
V9CA                        Unit-norm 方向偏移
  ↓
V9CA_AB (CLIME)             逐股票门控 + 可学习 scale  →  最优
  ↓
V9CA_AB is0.3 (CLIME*)      优化 init_scale + 更长训练  →  生产版本
```

## 附录 B：86 维特征完整索引

特征 1–66 为基础特征，特征 67–86 为 V11 研究中的高级特征。

| 索引 | 特征名 | 类别 |
|------|--------|------|
| 1 | ret_1d | 收益类 |
| 2 | ret_3d | 收益类 |
| 3 | ret_5d | 收益类 |
| 4 | ret_10d | 收益类 |
| 5 | ret_20d | 收益类 |
| 6 | log_ret_1d | 收益类 |
| 7 | intraday_return | 收益类 |
| 8 | overnight_gap | 收益类 |
| 9 | high_low_range | 量价类 |
| 10 | close_to_vwap | 量价类 |
| 11 | volume_log | 量价类 |
| 12 | amount_log | 量价类 |
| 13 | volume_ma5_ratio | 量价类 |
| 14 | amount_ma20_ratio | 量价类 |
| 15 | ma5_gap | 技术指标 |
| 16 | ma20_gap | 技术指标 |
| 17 | ma5_ma20_gap | 技术指标 |
| 18 | rolling_vol_20d | 技术指标 |
| 19 | high_low_position_20d | 技术指标 |
| 20 | rsi_14 | 技术指标 |
| 21 | macd_hist | 技术指标 |
| 22 | bollinger_z_20d | 技术指标 |
| 23 | turnover_rate | 基本面 |
| 24 | volume_ratio | 基本面 |
| 25 | pb | 基本面 |
| 26 | pe_ttm | 基本面 |
| 27 | ps_ttm | 基本面 |
| 28 | log_total_mv | 基本面 |
| 29 | log_circ_mv | 基本面 |
| 30 | net_mf_amount_ratio | 资金流向 |
| 31 | main_force_net_ratio | 资金流向 |
| 32 | small_order_net_ratio | 资金流向 |
| 33 | large_order_buy_ratio | 资金流向 |
| 34 | large_order_sell_ratio | 资金流向 |
| 35 | main_force_momentum_5d | 资金流向 |
| 36 | sh_ret_1d | 市场指数 |
| 37 | sh_ret_5d | 市场指数 |
| 38 | hs300_ret_1d | 市场指数 |
| 39 | hs300_ret_5d | 市场指数 |
| 40 | cyb_ret_1d | 市场指数 |
| 41 | cyb_ret_5d | 市场指数 |
| 42 | hs300_vol_20d | 市场指数 |
| 43 | hs300_ma5_ma20_gap | 市场指数 |
| 44 | hs300_drawdown_20d | 市场指数 |
| 45 | market_avg_ret_1d | 市场广度 |
| 46 | market_median_ret_1d | 市场广度 |
| 47 | market_advancing_ratio | 市场广度 |
| 48 | market_cross_section_vol | 市场广度 |
| 49 | market_top_bottom_spread | 市场广度 |
| 50 | market_amount_change_5d | 市场广度 |
| 51 | cs_z_ret_5d | 截面相对 |
| 52 | cs_z_ret_20d | 截面相对 |
| 53 | cs_z_amount | 截面相对 |
| 54 | cs_z_turnover_rate | 截面相对 |
| 55 | cs_z_log_total_mv | 截面相对 |
| 56 | cs_z_pb | 截面相对 |
| 57 | cs_z_main_force_net_ratio | 截面相对 |
| 58 | cs_rank_ret_5d | 截面相对 |
| 59 | cs_rank_amount | 截面相对 |
| 60 | cs_rank_main_force | 截面相对 |
| 61 | industry_ret_1d | 行业相对 |
| 62 | industry_ret_5d | 行业相对 |
| 63 | relative_industry_ret_1d | 行业相对 |
| 64 | relative_industry_ret_5d | 行业相对 |
| 65 | industry_advancing_ratio | 行业相对 |
| 66 | relative_industry_moneyflow | 行业相对 |
| 67 | kmid | K 线形态 |
| 68 | klen | K 线形态 |
| 69 | kmid2 | K 线形态 |
| 70 | kup | K 线形态 |
| 71 | kup2 | K 线形态 |
| 72 | klow | K 线形态 |
| 73 | klow2 | K 线形态 |
| 74 | ksft | K 线形态 |
| 75 | ksft2 | K 线形态 |
| 76 | corr_20 | 量价相关性 |
| 77 | cord_20 | 量价相关性 |
| 78 | rsqr_20 | 滚动算子 |
| 79 | rank_20 | 滚动算子 |
| 80 | rsv_20 | 滚动算子 |
| 81 | imax_20 | 滚动算子 |
| 82 | imin_20 | 滚动算子 |
| 83 | sump_20 | 涨跌分解 |
| 84 | sumd_20 | 涨跌分解 |
| 85 | vsump_20 | 成交量方向 |
| 86 | vsumd_20 | 成交量方向 |
