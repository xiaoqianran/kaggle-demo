# PROGRESS — Tabular From Scratch

**Status: FS00–FS21 全部验收通过（Kaggle T4×2 COMPLETE）**  
Kernel: https://www.kaggle.com/code/yunianan/grok-tabular-from-scratch  
Elapsed: 31.73s · stages=22 · ACCEPTANCE 78/78 ✅

## 修复记录（本轮）

- 修复 notebook `main()` 双执行 → stage_order 44 的问题；单次 22 阶段
- 修复 Stacking 用训练集预测做 meta 的**泄漏** → 3-fold OOF
- 加强 RF（更多树/特征）避免回归显著劣于单树
- 增加硬验收 `ACCEPTANCE.json`（文件/PNG/指标/OOF）
- 重置全局 STAGE_ORDER/SUMMARY，避免状态污染

| Stage | Status | Note |
|-------|--------|------|
| fs00_protocol | ✅ | 尺子先锁死，否则后面任何提升都不可信 |
| fs01_baselines | ✅ | 任何模型若打不过基线，实现或泄漏有问题 |
| fs02_linear | ✅ | 线性模型吃掉可加性信号；非线性交互仍吃不掉 |
| fs03_feature_engineering | ✅ | 特征工程=把非线性变成线性模型可读的列 |
| fs04_trees | ✅ | 树自动切交互，但单棵高方差、边界锯齿 |
| fs05_random_forest | ✅ | 集成用平均换稳定；RF 是表格经典甜点 |
| fs06_gbdt_scratch | ✅ | Boosting 用偏差换精度；表格王座的核心机制 |
| fs07_hist_gbdt | ✅ | 工业 GBDT：更快、更稳、Kaggle 表格默认武器 |
| fs08_target_encoding | ✅ | 类别列的正确编码往往 > 换模型；但泄漏会虚高 |
| fs09_mlp | ✅ | 纯 MLP 在中小表格常输给 GBDT——这是领域事实 |
| fs10_embeddings | ✅ | Embedding 是 DL 处理类别的正确姿势 |
| fs11_ft_transformer | ✅ | 特征即 token：注意力建模特征交互 |
| fs12_stacking | ✅ | Stacking 必须 OOF；用训练集预测做 meta = 泄漏虚高 |
| fs13_ts_protocol | ✅ | 打乱时间=考试偷看答案；必须 walk-forward |
| fs14_naive_forecasts | ✅ | 季节朴素经常意外地强 |
| fs15_exp_smoothing | ✅ | 平滑法用递归状态捕捉水平/趋势/季节 |
| fs16_lag_tree | ✅ | 时序→表格化是工业最强捷径之一 |
| fs17_ar_linear | ✅ | 线性 AR 解释性强，非线性季节/冲击时吃力 |
| fs18_lstm | ✅ | LSTM 自动学记忆，但小数据易过拟合、训练更重 |
| fs19_tcn_nbeats | ✅ | 现代时序 DL：卷积感受野与可解释 stack 块 |
| fs20_patch_transformer | ✅ | 前沿时序：把时间切 patch 当 token |
| fs21_leaderboard | ✅ | 表格主线=基线→线性→树→提升→编码→DL/Transformer→集成；时序=协议→朴素→平... |

## Leaderboard

### Classification acc
```
maj      0.5560
linear   0.7240
FE       0.7240
tree     0.7060
RF       0.7240
GBDTs    0.7260
HistGB   0.7160
TE       0.7100
MLP      0.7040
Emb      0.6980
FTT      0.7180
Stack    0.7420
```

### Regression RMSE
```
mean     3.9028
ridge    1.0731
FE       0.4941
tree     2.1015
RF       2.4790
GBDTs    1.3780
HistGB   0.8411
MLP      0.7299
FTT      1.1036
```

### Time series RMSE
```
seas_naive 0.9118
HW         1.0477
lag_tree   1.0396
AR         0.8779
LSTM       2.0997
TCN/NB     1.1100
Patch      1.6206
```

- Best cls: `Stack` · Best reg: `FE` · Best ts: `AR`
