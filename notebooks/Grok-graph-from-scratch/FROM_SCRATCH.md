# Graph Machine Learning From Scratch

> 类似 **LLM from scratch**：一条可跑通的实验链。  
> 每一步：**概念 → 最小实现 → 真实输入 → 可观察输出 → 与上一步对比**。

**执行状态：FS00–FS16 已全部验收通过**（Kaggle T4×2 · `PROGRESS.md` · `results/ACCEPTANCE.json`）。

---

## 0. 全领域实验地图

```text
从零实现              经典方法                 关键突破                现代系统                 前沿
────────────────────────────────────────────────────────────────────────────────────────────────
FS00 图协议：A/X/Y、度、同质性
FS01 仅特征 Softmax（无视边）
FS02 标签传播 LP                 ← 图半监督经典
FS03 谱嵌入 Laplacian            ← 谱图理论
FS04 DeepWalk 游走+PPMI          ← 2014 网络嵌入
FS05 邻居均值聚合                ← 消息传递原型
FS06 GCN                         ← Kipf & Welling 2017
FS07 GAT 注意力                  ← Veličković 2018
FS08 GIN 表达力                  ← Xu et al. 2019 / WL
FS09 链接预测                    ← 边级任务
FS10 图分类 + readout
FS11 异构/二部图
FS12 Graph Transformer           ← 结构偏置注意力
FS13 对比学习 GraphCL 精神       ← 自监督
FS14 过平滑诊断
FS15 残差深层 GCN                ← 治过平滑
FS16 全谱排行榜
```

**主线一句话**  
无结构特征 → 标签/谱/游走经典图学习 → 消息传递 GNN → 注意力与表达力 → 多任务与异构 → Transformer/SSL 与深层病症治理。
