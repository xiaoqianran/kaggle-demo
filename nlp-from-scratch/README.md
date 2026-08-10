# NLP From Scratch — 自然语言处理全路线实验

**领域**：NLP  
**任务**：文本分类 · 词元分类 · 表格问答 · 问答 · 零次射击分类 · 翻译 · 总结 · 特征提取 · 文本生成 · 填充掩膜 · 句子相似度 · 文本排名

## 地图

见 [ROADMAP.md](ROADMAP.md) · 进度 [PROGRESS.md](PROGRESS.md)

```
从零实现(S00–S02) → 经典方法(S03–S05) → 神经突破(S06–S10)
        → 现代系统(S11–S14) → 前沿统一(S15–S16)
```

## 怎么跑

### 经典阶段（本地秒级，纯 NumPy）

```bash
source /workspace/.venv-kaggle/bin/activate   # or any py3.11+ with numpy
export PYTHONPATH=nlp-from-scratch
python nlp-from-scratch/stages/S00_bow_naive_bayes.py
# … S01 … S05
```

### 神经 / 现代阶段（Kaggle T4×2，禁止弱 CPU 训练）

```bash
source scripts/kaggle-env.sh
./scripts/auto-fix-run.sh notebooks/Grok-nlp-neural-from-scratch
./scripts/auto-fix-run.sh notebooks/Grok-nlp-modern-frontier
```

| Notebook | Stages | Kaggle |
|----------|--------|--------|
| `Grok-nlp-neural-from-scratch` | S06–S10 | https://www.kaggle.com/code/zhengyingxiong/grok-nlp-neural-from-scratch |
| `Grok-nlp-modern-frontier` | S11–S16 | https://www.kaggle.com/code/zhengyingxiong/grok-nlp-modern-frontier |

结果 JSON：`nlp-from-scratch/results/S*.json`

## 每一步契约

**概念 → 最小实现 → 真实输入 → 可观察输出 → 与上一步对比 → 新增能力**
