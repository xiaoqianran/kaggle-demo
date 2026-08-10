# PROGRESS — Graph ML From Scratch

**Status: FS00–FS16 全部验收通过（Kaggle T4×2 COMPLETE）**  
Kernel: https://www.kaggle.com/code/yunianan/grok-graph-from-scratch  
Elapsed: 33.68s · stages=17 · ACCEPTANCE 61/61 ✅

| Stage | Status | Note |
|-------|--------|------|
| fs00_protocol | ✅ | 先锁图协议与同质性；后面对比才有意义 |
| fs01_features_only | ✅ | 无结构基线：后面 GNN 必须打过它才算用上图 |
| fs02_label_propagation | ✅ | 边=标签高速公路；同质图上 LP 极强且无需特征 |
| fs03_spectral | ✅ | 谱嵌入把连通结构变成几何坐标（GCN 的谱根源） |
| fs04_deepwalk | ✅ | 游走把高阶邻近变成词向量；无特征也能嵌入节点 |
| fs05_mean_agg | ✅ | 邻居平均=最简消息传递；GNN 从这里开始 |
| fs06_gcn | ✅ | 可学习的谱近似消息传递；GNN 经典基石 |
| fs07_gat | ✅ | 注意力让重要邻居权重大；异构邻域时更有用 |
| fs08_gin | ✅ | 求和+MLP 逼近单射聚合；表达力理论关键突破 |
| fs09_link_pred | ✅ | 链接预测是推荐/知识图谱核心任务 |
| fs10_graph_clf | ✅ | 分子/社交图分类依赖 readout 聚合 |
| fs11_hetero | ✅ | 推荐/知识图谱天然异构；边类型即归纳偏置 |
| fs12_graph_transformer | ✅ | 现代图模型：Transformer 全局注意力 + 结构偏置 |
| fs13_contrastive | ✅ | 自监督图学习：少标签时靠增强与对比 |
| fs14_over_smoothing | ✅ | 过平滑是深层 GNN 经典病；残差/注意力/扩散缓解 |
| fs15_residual_gcn | ✅ | 残差是治过平滑/深层训练的实用解药之一 |
| fs16_leaderboard | ✅ | 图能力链：特征→标签传播→谱/游走→消息传递→注意力/表达力→任务族→Transforme... |

## Node classification (test acc)
```
feat       0.5250
LP         1.0000
spectral   1.0000
DeepWalk   1.0000
meanAgg    0.9750
GCN        1.0000
GAT        1.0000
GIN        1.0000
GraphTR    0.9500
SSL        1.0000
ResGCN     1.0000
```

- Link AUC: 0.8576
- Graph clf: 0.8333
- Hetero: 1.0000
- Best node: `LP`
