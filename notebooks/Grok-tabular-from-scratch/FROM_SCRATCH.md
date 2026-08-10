# Tabular ML + Time Series From Scratch

> 类似 **LLM from scratch**：一条可跑通的实验链，不是概念清单。  
> 每一步：**概念 → 最小实现 → 真实输入 → 可观察输出 → 与上一步对比**。  
> 领域：表格分类 · 表格回归 · 时间序列预测。

**执行**：`scripts/run_tabular_fs_ladder.py`（Kaggle T4×2）→ `results/fs*/` + `SUMMARY.json`。

---

## 0. 全领域实验地图

```text
从零实现              经典方法                 关键突破                现代系统                 前沿
────────────────────────────────────────────────────────────────────────────────────────────────
【表格】
FS00 协议：指标/切分/泄漏
FS01 常数基线（众数/均值）
FS02 线性 LogReg/Ridge        ← 广义线性
FS03 特征工程：分箱+交互
FS04 决策树                    ← CART
FS05 随机森林                  ← Bagging
FS06 从零 GBDT                 ← Friedman 提升
FS07 HistGBDT 工业实现         ← XGB/LGBM 思想
FS08 Target Encoding           ← 高基数类别
FS09 表格 MLP
FS10 类别 Embedding
FS11 FT-Transformer 迷你       ← Gorishniy 等
FS12 Stacking 集成
【时序】
FS13 时序协议 walk-forward
FS14 朴素/季节朴素/MA
FS15 指数平滑 Holt-Winters     ← ETS
FS16 滞后特征 + 树             ← 时序表格化
FS17 线性 AR                   ← ARIMA 核心
FS18 LSTM 序列模型
FS19 TCN / N-BEATS 迷你
FS20 PatchTransformer          ← PatchTST 思想
FS21 全谱排行榜与闭环
```

**主线一句话**  
表格：基线 → 线性 → 特征工程 → 树 → 提升 → 类别编码 → 表格 DL → 集成。  
时序：禁止打乱 → 朴素基线 → 平滑 → 滞后树 → 序列 DL → Patch Transformer。

---

## 1. 步骤速查

| 阶段 | 你应感知到 |
|------|------------|
| FS00 | 尺子先锁死 |
| FS01 | 打不过基线=实现有病 |
| FS02 | 线性吃可加信号 |
| FS03 | FE 把非线性翻译成列 |
| FS04 | 树自动切交互、高方差 |
| FS05 | Bagging 换稳定 |
| FS06 | Boosting 拟合残差 |
| FS07 | 工业 GBDT=表格默认武器 |
| FS08 | 类别编码常常 > 换模型 |
| FS09–11 | 小数据 DL 未必赢 GBDT；FTT 是现代骨干 |
| FS12 | 多样性 stacking 抠点 |
| FS13 | 打乱时间=偷看答案 |
| FS14 | 季节朴素意外地强 |
| FS15 | 水平/趋势/季节递归态 |
| FS16 | 时序→表格是工业捷径 |
| FS17 | AR 可解释、非线性弱 |
| FS18–20 | 序列/卷积/Patch 端到端 |
| FS21 | 一张图看懂能力如何叠上去 |

## 2. 数据

- 分类/回归：可控合成（数值+类别+非线性规则）
- 时序：趋势 + 周季节 + 年季节 + 脉冲 + 噪声
- 全部离线可复现（`SEED=42`）

## 3. 运行

```bash
# Kaggle
python scripts/kaggle_run.py notebooks/Grok-tabular-from-scratch --accelerator NvidiaTeslaT4

# 本地仅调试（不推荐训练）
TABULAR_FS_OUT=./results python notebooks/Grok-tabular-from-scratch/scripts/run_tabular_fs_ladder.py
```
