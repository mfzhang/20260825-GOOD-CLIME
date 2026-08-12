# Literature Survey: Motivation for CLIME

This document surveys the literature supporting four key motivations behind CLIME (V9CA_AB), organized to directly inform the Introduction's narrative: **Problem → Gap → Our Approach**.

---

## Part 1: Current Deep Learning Methods for Stock Selection (The "Gap")

### 1.1 Dominant Architectures

| Architecture | Papers | How They Handle Market Context |
|---|---|---|
| **GNN / Hypergraph** | H-GAT (2306.15526), STHAN-SR (AAAI 2021), THGNN (CIKM 2022), MGAR (Info.Sci. 2023), RT-GCN (ICDE 2023) | Encode market structure into stock-relation graphs (industry, supply chain, correlation). GNN message-passing propagates market context. |
| **Cross-Attention** | MCI-GRU (2410.20679), X-Trend (2310.10500) | Cross-attention from stock features to market-state embeddings. Condition predictions on market regime. |
| **Transformer** | StockFormer (2401.06139), Quantformer (2404.00424), SERT (2505.01575) | Self-attention over historical sequences. Market context via multi-task learning or transfer learning. |
| **End-to-End Factor** | E2EAI (ICAIF 2023) | End-to-end pipeline from factor selection to portfolio construction. |

### 1.2 Summary of Current Approaches

**How market information is incorporated (methods used by existing work):**
1. **Graph-based** (most common): Stock inter-relations become edges; message-passing propagates context
2. **Cross-attention conditioning**: Learn latent market state, attend to it from stock features
3. **Feature concatenation**: Simply concatenate market features into input vector
4. **Transfer learning**: Pre-train on large-scale data to capture market structure

**Loss functions used:**
1. Listwise ranking (STHAN-SR, RT-GCN, MGAR, H-GAT)
2. MSE regression (DL Long-Short, MCI-GRU, SERT)
3. Multi-task (StockFormer, E2EAI)
4. Meta-learning (X-Trend)

**Training strategies:**
1. Single-stage supervised (most papers)
2. Transfer learning / pre-training (Quantformer, SERT)
3. Few-shot meta-learning (X-Trend)

### 1.3 What's Missing — Our Gap

| Dimension | Existing Methods | Our CLIME |
|-----------|-----------------|-----------|
| **Market injection** | Graph edges, cross-attention, concatenation | **Constrained additive injection** (unit-norm + gate + scale) |
| **Backbone preservation** | Not addressed — fine-tuning can distort | **Explicit preservation** via additive design + near-zero init |
| **Loss function** | MSE or standard ranking losses | **Directional Regression Loss**: asymmetric weighting (3x sign-error, 0.5x conservative) |
| **Training strategy** | Single-stage or transfer | **Two-stage curriculum**: pairwise pretraining → directional regression |
| **Theoretical grounding** | Empirically motivated | Grounded in: LTR theory + curriculum learning + RLHF preference paradigm |

---

## Part 2: Why Pairwise is an Easier Task (Motivation for Stage 1)

This chain establishes that **pairwise ranking is a cognitively and statistically simpler learning task than pointwise regression**, supporting our two-stage curriculum design.

### 2.1 Cognitive Foundation: Humans Are Comparative Thinkers

**Kahneman & Tversky (1979)** — "Prospect Theory: An Analysis of Decision under Risk"
- Econometrica, 47(2), pp. 263-291. DOI: 10.2307/1914185
- **Key insight**: Humans evaluate changes **relative to a reference point**, not absolute states. The "carrier of utility" is relative change, not absolute value. The Bradley-Terry model formalizes: $P(i \succ j) = \sigma(r_i - r_j)$.

**Hsee (1996)** — "The Evaluability Hypothesis"
- Organizational Behavior and Human Decision Processes, 67(3), pp. 247-257. DOI: 10.1006/obhd.1996.0077
- **Key insight**: Joint (comparative) evaluation is more accurate than separate (absolute) evaluation. Comparative judgment provides an **external reference standard**, reducing noise and systematic bias.

### 2.2 Learning to Rank: Pairwise Losses Beat Pointwise

**Joachims (2002)** — "Optimizing Search Engines Using Clickthrough Data"
- KDD 2002, pp. 133-142. DOI: 10.1145/775047.775067. **KDD 2015 Test of Time Award**
- **Key insight**: Pairwise preference constraints (clicked > not-clicked) are more abundant and better aligned with ranking quality than absolute relevance labels. Ranking SVM learns from pairwise constraints directly.

**Burges et al. (2005)** — "Learning to Rank Using Gradient Descent" (RankNet)
- ICML 2005, pp. 89-96. DOI: 10.1145/1102351.1102363
- **Key insight**: Ranking should be treated as predicting **relative order** $P_{ij} = \sigma(s_i - s_j)$, not absolute values. The pairwise cross-entropy loss directly optimizes for document ordering while avoiding the "rank boundary" problem of ordinal regression.

**Rendle et al. (2009)** — "BPR: Bayesian Personalized Ranking from Implicit Feedback"
- UAI 2009, pp. 452-461. DOI: 10.5555/1795114.1795167
- **Key insight**: Pairwise optimization (BPR-Opt) **significantly outperforms pointwise optimization** on ranking metrics. Optimizing for the right criterion (ranking) matters more than model architecture.

### 2.3 RLHF: Pairwise Preferences at Scale

**Christiano et al. (2017)** — "Deep Reinforcement Learning from Human Preferences"
- NeurIPS 2017. arXiv: 1706.03741
- **Key insight**: Complex RL tasks can be solved using only **pairwise human preferences**, without any hand-crafted reward function. Pairwise comparison is practical and scalable for non-expert labelers. Bradley-Terry formulation: $\hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum \hat{r}(o_t^1)}{\exp \sum \hat{r}(o_t^1) + \exp \sum \hat{r}(o_t^2)}$.

**Ouyang et al. (2022)** — "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
- NeurIPS 2022. arXiv: 2203.02155
- **Key insight**: A 1.3B model trained with pairwise human preference feedback is **preferred over a 175B GPT-3** in human evaluations. The quality of the preference signal matters more than model scale. Pairwise ranking is the effective paradigm for aligning AI systems.

### 2.4 Curriculum Learning: Easy-to-Hard Training

**Bengio et al. (2009)** — "Curriculum Learning"
- ICML 2009, pp. 41-48. DOI: 10.1145/1553374.1553380
- **Key insight**: Training on **easier examples first** and gradually increasing difficulty acts as a **continuation method** that guides optimization toward better local minima and faster convergence. The curriculum is formalized as a sequence of training distributions with increasing entropy.

**→ Our motivation**: If pairwise ranking is an easier task, then a **curriculum** of pairwise pretraining (Stage 1) → directional regression (Stage 2) should yield better convergence than training on regression from scratch.

---

## Part 3: Why Additive Injection Preserves Representations (Motivation for ScaledGatedEncoder)

This chain establishes that **additive conditioning with zero-initialization** is the proven strategy for preserving pretrained backbone representations, while multiplicative methods risk destruction.

### 3.1 The Proven Approach: Additive Injection from Zero

**Houlsby et al. (2019)** — "Adapter: Parameter-Efficient Transfer Learning via Additive Bottlenecks"
- ICML 2019. arXiv: 1902.00751
- **Key insight**: $h' = h + \text{Adapter}(h)$ with near-zero initialization preserves pretrained behavior at t=0. Only trains 3.6% of parameters but achieves within 0.4% of full fine-tuning on GLUE. The additive design bounds the damage — the adapter can only add a correction, not multiply/scale the backbone's features.

**Hu et al. (2022)** — "LoRA: Low-Rank Adaptation"
- ICLR 2022. arXiv: 2106.09685
- **Key insight**: $W = W_0 + BA$, where $B$ is initialized to **zero**. The model is initially identical to the pretrained version. Low-rank constraint acts as implicit regularizer — updates can only reinforce directions already present in $W_0$. Biderman et al. (2024) confirms: LoRA sits on the "learn less, forget less" end of the learning-forgetting Pareto frontier.

**Zhang et al. (2023)** — "ControlNet: Additive Conditioning that Preserves the Backbone"
- ICCV 2023. arXiv: 2302.05543
- **Key insight**: Adds conditioning to a **frozen** backbone via zero-convolutions (initialized to zero). The authors' term "sudden convergence" describes how parameters grow progressively from zero. This is the strongest evidence for additive conditioning as a backbone-preserving strategy.

### 3.2 The Risk: Multiplicative Modulation Destroys Representations

**Perez et al. (2018)** — "FiLM: Visual Reasoning with a General Conditioning Layer"
- AAAI 2018. arXiv: 1709.07871
- **Key insight**: $y = \gamma * x + \beta$ provides fine-grained control but can significantly alter backbone feature distributions. When $\gamma$ deviates from 1, features are amplified or suppressed. **Powerful but risky for preserving pretrained representations.**

**Dauphin et al. (2017)** — "Gated Linear Units"
- ICML 2017. arXiv: 1612.08083
- **Key insight**: Purely multiplicative gating ($x \odot \sigma(g(x))$). When gate outputs 0, information is permanently lost. **Not representation-preserving.** Valuable for internal computation but unsuitable for external conditioning.

**Srivastava et al. (2015)** — "Highway Networks"
- NeurIPS 2015. arXiv: 1505.00387
- **Key insight**: $y = H(x) \cdot T(x) + x \cdot (1 - T(x))$ provides a **guaranteed additive carry path**. Bias initialization (T near 0) ensures the additive path dominates early training. This directly generalizes to our design: the backbone's original representation always has an additive identity path.

### 3.3 Our Position

| Approach | Mechanism | Preserves Backbone? | Risk |
|----------|-----------|:---:|------|
| FiLM | $\gamma \odot h + \beta$ | No | High — $\gamma$ distorts features |
| GLU | $x \odot \sigma(g(x))$ | No | High — gate kills information |
| Adapter/LoRA/ControlNet | $h + \Delta h$ (additive) | **Yes** | Low — only adds correction |
| **CLIME (Ours)** | $h + \text{gate} \cdot \tanh(s) \cdot \frac{\hat{\mathbf{u}}}{\|\hat{\mathbf{u}}\|} \cdot \sqrt{256}$ | **Yes** | **Bounded** — unit-norm + tanh + gate triple constraint |

Our design goes beyond simple additive injection by adding **three constraints**:
1. Unit-norm direction (encoder controls **which way**, not **how much**)
2. Per-stock gate (selective injection for stocks that need market context)
3. Global tanh-bounded scale (overall magnitude can never explode)

---

## Part 4: Direction-Aware Loss Functions (Motivation for Directional Regression Loss)

This chain establishes that **ranking-aware, direction-sensitive losses** consistently outperform symmetric regression for stock prediction tasks.

### 4.1 Direction-Aware Losses in Finance

**Michankow et al. (2023)** — "Mean Absolute Directional Loss (MADL)"
- Journal of Computational Science, 2024. arXiv: 2309.10546
- **Key insight**: $\text{MADL} = \frac{1}{N}\sum (-1) \times \text{sign}(R_i \cdot \hat{R}_i) \times |R_i|$. Loss is negative (reward) when direction matches, positive (penalty) when direction is wrong. **Evaluates predictions based on whether the trade would have made money, not prediction error magnitude.** Outperforms MAE on Bitcoin and Crude Oil.

**Michankow et al. (2024)** — "Generalized MADL (GMADL)"
- arXiv: 2412.18405
- **Key insight**: Replaces non-differentiable sign() with smooth sigmoid: $\text{GMADL} = \frac{1}{N}\sum (-1) \times (\frac{1}{1+\exp(-a R_i \hat{R}_i)} - 0.5) \times |R_i|^b$. Consistently outperforms MSE across Transformer, LSTM, and MLP architectures.

### 4.2 Ranking-Aware Losses

**Lin et al. (2026)** — "LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction"
- arXiv: 2605.00501
- **Key insight**: Derives closed-form lambda gradients that directly optimize Spearman Rank IC (the evaluation metric quants actually use). Outperforms both MSE and NDCG-oriented ranking methods on real market data.

**Kwiatkowski & Chudziak (2025)** — "On Evaluating Loss Functions for Stock Ranking"
- arXiv: 2510.14156
- **Key insight**: Comprehensive benchmarking of pointwise (MSE), pairwise (Hinge, BPR, RankNet), and listwise (ListNet) losses for Transformer stock ranking on S&P 500. **Pairwise/listwise losses consistently outperformed MSE in all portfolio metrics.** Key finding: better ranking metrics did NOT always translate to better portfolio metrics, highlighting the gap between ranking quality and investment performance.

**Poh et al. (2020)** — "Building Cross-Sectional Systematic Strategies By Learning to Rank"
- Journal of Financial Data Science, 2024. arXiv: 2012.07149
- **Key insight**: Foundational paper bridging LTR with quantitative finance. LambdaMART achieved Sharpe ~2.16, roughly **3x improvement over classical regression methods**. Listwise losses (ListNet, ListMLE) had strong top-k performance.

### 4.3 Asymmetric Loss Functions

**Taggart (2022)** — "Generalized Huber Loss as Asymmetric Forecasting Objective"
- Electronic Journal of Statistics, 2022. arXiv: 2108.12426
- **Key insight**: Asymmetric Huber loss bridges quantile loss ($\alpha \to \infty$) and expectile loss ($\alpha \to 0$). The asymmetry parameter $\tau$ directly maps to a decision-maker's risk aversion. Quadratic region provides noise robustness; linear tails handle heavy-tailed return distributions.

### 4.4 Our Position

| Loss | Direction-Sensitive | Ranking-Aware | Asymmetric Weight | Trainable with GD |
|------|:---:|:---:|:---:|:---:|
| MSE | No | No | No | Yes |
| MADL | Yes | No | No | No (non-differentiable) |
| GMADL | Yes | No | No | Yes |
| LambdaRankIC | No | Yes | No | Yes |
| BPR | No | Yes | No | Yes |
| **DirectionalReg (Ours)** | **Yes** | **Yes** | **Yes** (3x / 0.5x / 1x) | **Yes** |

**Our Directional Regression Loss is unique in combining all three properties**:
1. **Direction sensitivity** (like MADL/GMADL): sign mismatch → 3x penalty
2. **Ranking awareness** (like BPR/LambdaRank): conservative pessimism → 0.5x relief (won't select it anyway)
3. **Asymmetric weighting** (like Generalized Huber): different errors have different consequences for portfolio construction

---

## Summary: The Four-Link Motivation Chain

```
Link 1: Current methods have gaps
  └─ No constrained additive market injection
  └─ No backbone-preserving conditioning
  └─ No direction-aware asymmetric losses

Link 2: Pairwise is an easier learning task
  └─ Cognitive science (Kahneman & Tversky, Hsee)
  └─ LTR theory (Joachims, Burges, Rendle)
  └─ RLHF at scale (Christiano, Ouyang)
  └─ Curriculum learning (Bengio)
  → Stage 1: Pairwise pretraining

Link 3: Additive injection preserves representations
  └─ Adapter (Houlsby), LoRA (Hu), ControlNet (Zhang)
  └─ FiLM/GLU risk destroying backbone (Perez, Dauphin)
  → ScaledGatedEncoder: h' = h + offset (constrained)

Link 4: Direction-aware loss aligns with the task
  └─ MADL/GMADL (Michankow): direction matters
  └─ LambdaRankIC (Lin): rank IC optimization
  └─ Generalized Huber (Taggart): asymmetric weighting
  → Directional Regression Loss: 3x / 0.5x / 1x
```

---

## Full Paper Reference List

| # | Title | Authors | Year | Venue | arXiv / DOI |
|---|-------|--------|------|-------|-------------|
| 1 | 101 Formulaic Alphas | Kakushadze, Z. | 2016 | — | arXiv:1601.00991 |
| 2 | Prospect Theory | Kahneman, D., Tversky, A. | 1979 | Econometrica | DOI:10.2307/1914185 |
| 3 | The Evaluability Hypothesis | Hsee, C.K. | 1996 | OBHDP | DOI:10.1006/obhd.1996.0077 |
| 4 | Optimizing Search Engines Using Clickthrough Data | Joachims, T. | 2002 | KDD | DOI:10.1145/775047.775067 |
| 5 | Learning to Rank Using Gradient Descent | Burges, C., et al. | 2005 | ICML | DOI:10.1145/1102351.1102363 |
| 6 | Curriculum Learning | Bengio, Y., et al. | 2009 | ICML | DOI:10.1145/1553374.1553380 |
| 7 | BPR: Bayesian Personalized Ranking | Rendle, S., et al. | 2009 | UAI | DOI:10.5555/1795114.1795167 |
| 8 | Highway Networks | Srivastava, R.K., et al. | 2015 | NeurIPS | arXiv:1505.00387 |
| 9 | Dynamic Filter Networks | De Brabandere, B., et al. | 2016 | NeurIPS | arXiv:1605.09673 |
| 10 | Conditional Instance Normalization | Dumoulin, V., et al. | 2017 | ICLR | arXiv:1610.07629 |
| 11 | Deep RL from Human Preferences | Christiano, P., et al. | 2017 | NeurIPS | arXiv:1706.03741 |
| 12 | Gated Linear Units | Dauphin, Y.N., et al. | 2017 | ICML | arXiv:1612.08083 |
| 13 | FiLM: Visual Reasoning with a General Conditioning Layer | Perez, E., et al. | 2018 | AAAI | arXiv:1709.07871 |
| 14 | Adapter: Parameter-Efficient Transfer Learning | Houlsby, N., et al. | 2019 | ICML | arXiv:1902.00751 |
| 15 | Building Cross-Sectional Strategies By Learning to Rank | Poh, D., et al. | 2020 | J.Fin.DataSci. | arXiv:2012.07149 |
| 16 | STHAN-SR: Stock Selection via Spatiotemporal Hypergraph Attention | — | 2021 | AAAI | DOI:10.1609/aaai.v35i1.16127 |
| 17 | Training Language Models to Follow Instructions | Ouyang, L., et al. | 2022 | NeurIPS | arXiv:2203.02155 |
| 18 | LoRA: Low-Rank Adaptation | Hu, E.J., et al. | 2022 | ICLR | arXiv:2106.09685 |
| 19 | Generalized Huber Loss | Taggart, R.J. | 2022 | EJS | arXiv:2108.12426 |
| 20 | ControlNet | Zhang, L., et al. | 2023 | ICCV | arXiv:2302.05543 |
| 21 | H-GAT: Higher-order GAT for Stock Selection | — | 2023 | — | arXiv:2306.15526 |
| 22 | THGNN: Temporal and Heterogeneous GNN | — | 2023 | CIKM 2022 | arXiv:2305.08740 |
| 23 | MGAR: Multi-relational Graph Attention Ranking | — | 2023 | Info.Sci. | DOI:10.1016/j.ins.2023.119236 |
| 24 | RT-GCN: Relational Temporal GCN for Ranking-Based Stock Prediction | — | 2023 | ICDE | DOI:10.1109/ICDE55515.2023.00017 |
| 25 | E2EAI: End-to-End Deep Learning for Active Investing | — | 2023 | ICAIF | — |
| 26 | X-Trend: Few-Shot Learning in Financial Time-Series | — | 2023 | J.Fin.DataSci. | arXiv:2310.10500 |
| 27 | MADL: Mean Absolute Directional Loss | Michankow, J., et al. | 2023 | J.Comput.Sci. | arXiv:2309.10546 |
| 28 | StockFormer: Wavelet Transform + Multi-Task Self-Attention | — | 2024 | ESWA | arXiv:2401.06139 |
| 29 | Quantformer: From Attention to Profit | — | 2024 | ESWA | arXiv:2404.00424 |
| 30 | DL in Long-Short Stock Portfolio Allocation | — | 2024 | — | arXiv:2411.13555 |
| 31 | MCI-GRU: Multi-Head Cross-Attention + GRU | — | 2024 | — | arXiv:2410.20679 |
| 32 | GMADL: Generalized Mean Absolute Directional Loss | Michankow, J., et al. | 2024 | — | arXiv:2412.18405 |
| 33 | SERT: Asset Pricing in Pre-trained Transformer | — | 2025 | — | arXiv:2505.01575 |
| 34 | Loss Functions for Stock Ranking (Empirical Analysis) | Kwiatkowski, J., Chudziak, J.A. | 2025 | — | arXiv:2510.14156 |
| 35 | LambdaRankIC: Directly Optimizing Rank IC | Lin, Y., et al. | 2026 | — | arXiv:2605.00501 |
| 36 | Error and Optimism Bias Regularization | Sohaee, N. | 2023 | J.BigData | DOI:10.1186/s40537-023-00685-9 |
