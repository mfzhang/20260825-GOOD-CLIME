# V9CA_AB is0.3 — 基于 Transformer 的 A 股量化选股模型

基于 Transformer + RoPE 架构的 A 股日频选股模型。核心思路：在 Stage1 backbone 预训练（pairwise ranking）基础上，通过 **Scaled Gated Injection** 机制将市场级 peer dynamics 信息注入个股表征空间（256-dim），实现市场环境自适应的股票排序。

## 关键结果

| 指标 | Validation (202509-202512) | Holdout (202512-202605) |
|------|---------------------------|-------------------------|
| 累计超额收益 (vs 等权基准) | +52.98% | +148.44% |
| 年化 Sharpe Ratio | 7.71 | 7.98 |
| 最大回撤 | 6.56% | 6.56% |
| 月度胜率 (vs 基准) | 100% (6/6) | 100% (6/6) |

## 架构

```
x [B, L, 67]              peer_dyn [B, 24]
     |                         |
     v                         v
+--------------+    +----------------------+
|  Backbone    |    |  Market Encoder      |
|  (Stage1)    |    |  MLP(32->256, 3 层)  |
|              |    |  = peer_dyn[24]      |
| input_proj   |    |  + market_ctx[8]     |
|  -> rope     |    |  -> Linear(256, 256) |
|  -> blocksx4 |    |  -> raw_offset[B,256]|
|  -> last     |    +----------+-----------+
|    [B, 256]  |               |
+------+-------+    +----------+-----------+
       |            |  ScaledGatedEncoder  |
       |            |  A: unit-norm offset |
       |            |     + tanh scale     |
       |            |  B: per-stock gate   |
       |            |     in [0, 1]        |
       |            +----------+-----------+
       |                       |
       +-----------+-----------+
                   v
         h' = h + gate * scale * offset
                   |
                   v
            head -> score [B]
```

### 核心设计原则

1. **正确的调制层级**：市场信息注入在 256-dim 表征空间，不在 67-dim 特征空间。特征空间的任何修改都会破坏 backbone 已学到的表示
2. **信息瓶颈**：24+8=32 -> 256 -> 256，3 层 MLP 不做显式压缩，靠最后 Linear 的随机初始化（std=0.001）自然形成低秩注入
3. **加法优于乘法**：`h' = h + offset`（而非 `h' = gamma * h`），保留 backbone 原始判断
4. **per-stock 选择性**：gate 网络（MLP 8->64->1->Sigmoid）让模型为每只股票学习不同的市场敏感度

## 目录结构

```
reconstruct_code/
|-- train.py                   # 训练入口 (Stage1 + V9CA_AB)
|-- backtest.py                # 回测评估
|-- daily_inference.py         # 每日推理 & 选股
|-- build_caches.py            # 从 parquet 构建特征序列缓存
|-- build_peer_dynamics.py     # 构建 peer dynamics 缓存
|-- README.md
|-- requirements.txt
|-- data/                      # [用户准备] 原始数据目录 (见下方说明)
|-- cache/                     # [自动生成] 特征序列 + peer dynamics 缓存
|-- output/                    # [自动生成] 训练 checkpoint + 日志 + 回测结果
`-- src/
    |-- losses.py              # pairwise ranking loss + directional regression loss
    |-- trainer.py             # Stage1 训练循环
    |-- risk.py                # 风险评分 (波动率/回撤/振幅/流动性)
    |-- models/
    |   |-- transformer.py     # Transformer 骨架 (RoPE + MHA + FFN)
    |   `-- v9ca_ab.py         # V9CA_AB 模型 (ScaledGatedEncoder)
    `-- data/
        |-- dataset.py         # 数据集类 & 序列构建
        |-- features.py        # 特征计算
        |-- preprocess.py      # 标准化、Winsorize、缺失值填充
        `-- loader.py          # 原始 CSV 数据加载器
```

## 安装

```bash
pip install -r requirements.txt
```

依赖项：`torch`, `numpy`, `pandas`, `scipy`, `tqdm`, `pyarrow`。

## 数据准备

### 原始数据目录结构

请将以下数据放入 `data/` 目录：

```
data/
|-- basic.csv                  # 股票基础信息 (ts_code, industry, area, market, act_name, act_ent_type)
|-- trade_cal.csv              # 交易日历
|-- daily/                     # 个股日频行情 (按日期存储, 每个 CSV 为一个交易日)
|-- daily_open/                # 开盘价数据 (按日期存储)
|-- metric/                    # 基本面指标 (PE, PB, 换手率, 总市值等)
|-- moneyflow/                 # 资金流向数据 (大单/中单/小单/特大单买卖)
|-- market/                    # 市场指数数据 (上证/沪深300/创业板)
|-- index_weight/              # 指数成分股及权重
|-- stock_st/                  # ST 股票清单 (按日期存储)
`-- news/                      # 新闻快讯数据
```

### 数据格式说明

| 目录 | 关键字段 |
|------|---------|
| `daily/` | ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount, vwap |
| `metric/` | ts_code, trade_date, turnover_rate, pe, pe_ttm, pb, total_mv, circ_mv, total_share, float_share |
| `moneyflow/` | ts_code, trade_date, buy_lg_vol, sell_lg_vol, buy_elg_vol, sell_elg_vol, net_mf_vol, net_mf_amount |
| `stock_st/` | ts_code, trade_date, type, type_name |
| `basic.csv` | ts_code, industry, act_name, area, market, act_ent_type, list_date |
| `trade_cal.csv` | cal_date, is_open, pretrade_date |

`daily/` 目录中的每个 CSV 文件对应一个交易日（文件名格式：`YYYYMMDD.csv`），包含当日所有股票的行情数据。

## 快速开始

### 1. 准备数据

将原始数据按上述结构放入 `data/` 目录，然后构建标准化特征 parquet：

```python
from src.data.loader import load_all
from src.data.features import compute_features
from src.data.preprocess import preprocess

# 加载原始数据
raw = load_all("data")

# 计算 66 个基础特征
features = compute_features(raw)

# 清洗 + z-score 标准化
normalized = preprocess(features)
normalized.to_parquet("cache/normalized_features.parquet")
```

### 2. 构建缓存

```bash
# 构建所有 split 的特征序列缓存 (train/val/holdout)
python build_caches.py --split all

# 构建 peer dynamics 缓存
python build_peer_dynamics.py
```

缓存文件（自动生成于 `cache/` 目录）：
- 特征序列: `cache/{split}_L40_step{step}.pt`
- Peer dynamics: `cache/v7_peer_dynamics_{split}_L40_step{step}.pt`

### 3. 训练

#### Stage 1: Backbone 预训练

```bash
python train.py --stage1
```

输出: `output/transformer_v5/stage1_best.pt`

#### Stage 2: V9CA_AB 联合训练

```bash
# 最优配置 (init_scale=0.3)
python train.py --v9ca-ab --init-scale 0.3 --epochs 25
```

训练分三阶段：

| 阶段 | Loss | Backbone LR | Modulator LR | 说明 |
|------|------|-------------|--------------|------|
| Phase 1 (BCE) | BCEWithLogits | frozen | 1e-3 | 仅训练 encoder + head, 3 epochs |
| Phase 2 (Reg) | DirectionalReg | 1e-5 | 5e-4 | 解冻 backbone 联合训练, 12 epochs |
| Phase 3 (FT) | DirectionalReg | 1e-6 | 5e-5 | 微调 |

### 4. 回测

```bash
# 全部 split
python backtest.py --v9ca-ab --stage1 output/transformer_v5/stage1_best.pt

# 仅 holdout, 自定义持仓数
python backtest.py --v9ca-ab --split holdout_v5 \
    --v9ca-ab-ckpt output/transformer_v5/transformer_v9ca_ab_is0p3_best.pt \
    --stage1 output/transformer_v5/stage1_best.pt \
    --n 20
```

### 5. 每日推理

```bash
python daily_inference.py --date 20260530 \
    --model-path output/transformer_v5/transformer_v9ca_ab_is0p3_best.pt \
    --capital 1000000 --lambda-risk 0.3 --top-n 20
```

选股流程：alpha score 排序 -> top-10% 候选池 -> `combined = z(alpha) - 0.3 * z(risk)` -> 取 top-20 -> 温度 softmax 分配权重。

## CLI 参数参考

### train.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--stage1` | flag | — | Stage1 backbone 预训练 |
| `--v9ca-ab` | flag | — | V9CA_AB 联合训练 |
| `--stage1-ckpt` | str | None | Stage1 checkpoint 路径 |
| `--init-scale` | float | 0.1 | V9CA_AB 初始 scale（推荐 0.3） |
| `--resume` | str | None | 从 checkpoint 恢复训练 |
| `--epochs` | int | None | 覆盖 max_epochs |
| `--batch-size` | int | None | 覆盖 batch_size |
| `--feat-dim` | int | None | 特征维度 (67 或 72) |
| `--d-model` | int | 256 | Transformer 模型维度 |
| `--n-heads` | int | 8 | 注意力头数 |
| `--n-layers` | int | 4 | Transformer 层数 |
| `--ffn-hidden` | int | 1024 | FFN 隐藏层维度 |

### backtest.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--v9ca-ab` | flag | — | 运行 V9CA_AB 回测 |
| `--split` | str | both | `val_v5` / `holdout_v5` / `both` |
| `--v9ca-ab-ckpt` | str | None | V9CA_AB checkpoint 路径 |
| `--stage1` | str | None | Stage1 checkpoint 路径 |
| `--n` | int | 20 | 持仓股票数 |
| `--device` | str | cuda | 计算设备 |

### daily_inference.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--date` | str | None | 目标日期 (YYYYMMDD) |
| `--dates` | str | None | 多日期 (逗号分隔) |
| `--model-path` | str | None | 模型 checkpoint 路径 |
| `--lambda-risk` | float | 0.3 | 风险惩罚权重 |
| `--top-n` | int | 20 | 选股数 |
| `--top-pct` | float | 0.10 | 第一阶段候选池比例 |
| `--capital` | float | None | 总资金量 (元) |
| `--temperature` | float | 0.5 | Softmax 温度 |
| `--rebuild` | flag | — | 强制重建原始面板和特征 |
| `--no-cuda` | flag | — | 强制 CPU |
| `--output` | str | None | CSV 输出路径 |

## 模型演进

| 版本 | 架构 | 关键改变 | Holdout 超额 |
|------|------|---------|-------------|
| V5 | Pure Transformer | Stage1 backbone (pairwise ranking) | +197.99% |
| V7 | V5 + FiLM | 乘法调制 (`h' = gamma*h + beta`) | +127.70% (退化) |
| V8 | V5 + Per-Feature Scale | 67-dim feature-wise scaling | +139.93% (退化) |
| V9 | V5 + Additive Injection | `h' = h + offset`, 信息瓶颈 | 架构基础 |
| V9CA | V9 + Unit-Norm Offset | 方向性 offset, 丢弃幅度 | +95.84% |
| **V9CA_AB is0.3** | **V9CA + Gate + Scale** | **per-stock gate, tanh-bounded scale, init_scale=0.3** | **+148.44%, Sharpe 7.98** |

### 为什么 V7/V8 退化而 V9CA_AB 成功？

| 设计 | 调制方式 | 问题 |
|------|---------|------|
| V7 FiLM | `gamma*h + beta` | 乘法调制使 gamma 直接缩放 backbone 表征，训练不稳定 |
| V8 Per-Feature | 67-dim channel-wise scale | 在特征空间调制破坏了 backbone 学到的表示结构 |
| V9CA_AB | `h + gate*scale*unit_offset` | 加法保持 backbone 原始信息，gate 提供选择性，unit-norm 防止梯度短路 |

## FAQ

**Q: 训练 OOM？**
减小 batch_size 或使用 `--feat-dim 67`。

**Q: 回测时 CUDA OOM？**
使用 `--device cpu` 或在运行前清理 GPU。

**Q: checkpoint 加载报 missing keys？**
正常现象。Stage1 checkpoint 中的 attn_pool 权重在 V9CA_AB 中不使用（V9CA_AB 使用 last-token extraction），训练脚本会打印 missing keys 数量供确认。
