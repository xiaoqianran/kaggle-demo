# Multimodal From Scratch

> 类似 **LLM from scratch** 的完整多模态实验链。  
> 每步：**概念 → 最小实现 → 真实（合成可控）输入 → 可观察输出 → 与上步对比**。  
> 算力：全部在 **Kaggle `NvidiaTeslaT4`（T4×2）** 后台执行。

配套：[PROGRESS.md](./PROGRESS.md) · [LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md) · 结果 `results/` · 图 `figures/`

---

## 0. 全领域实验地图

```text
从零实现              经典方法                 关键突破              现代系统              前沿
────────────────────────────────────────────────────────────────────────────────────────────
FS00 模态/对齐地图
FS01 颜色袋伪字幕
FS02 CNN→类别词           ← ImageNet 骨干直觉
FS03 Show&Tell            ← Vinyals 2015
FS04 Show Attend Tell     ← Xu 2015
FS05 CLIP 双塔对比         ← Radford 2021 ★
FS06 VQA 融合答题          ← Antol / 经典 fusion
FS07 OCR+TF-IDF DocQA
FS08 页补丁 MaxSim 检索    ← ColPali 直觉
FS09 帧CNN+GRU 视频→文
FS10 log-mel 音频→文
FS11 条件迷你 DDPM 生图    ← Ho 2020 直觉
FS12 条件短视频生成
FS13 图像前缀+Transformer LM ← LLaVA 式接口
FS14 Any-to-Any 路由器     ← 统一多路由 API
```

**主线：** 感官→张量 → 对齐语言 → 融合答题/检索 → 条件生成媒体 → 统一任意路由。

---

## 1. 阶段说明（摘要）

### FS00 模态与对齐
- **概念**：图像/文本作为张量；对齐=可比较分数  
- **实现**：合成色块图 + 手工 color/shape 对齐分  
- **感知**：匹配对=2，错配<2  

### FS01 颜色袋字幕
- **概念**：最笨的 image→text I/O 契约  
- **vs FS00**：从打分到**生成句子**  

### FS02 迷你 CNN
- **概念**：可学习卷积替代规则  
- **vs FS01**：闭集类别词，但是**学出来的**  

### FS03 Show&Tell
- **概念**：CNN 全局向量 + LSTM 解码字幕  
- **vs FS02**：从单标签到**词序列**  

### FS04 Attend
- **概念**：每个词 soft-attend 空间特征图  
- **vs FS03**：可可视化「模型在看哪」  

### FS05 CLIP 双塔
- **概念**：InfoNCE 共享嵌入空间  
- **vs FS03/04**：检索对齐，不必生成  

### FS06 VQA
- **概念**：图特征 ⊕ 问句嵌入 → 答案分类  
- **vs FS05**：针对**单图自由问题**  

### FS07 DocQA
- **概念**：页面渲染 → OCR 文本 → TF-IDF 检索 → 跨度答案  
- **vs FS06**：多页文档证据  

### FS08 视觉文档检索
- **概念**：页补丁嵌入 + MaxSim（ColPali 直觉）  
- **vs FS07**：不依赖完美 OCR，从**像素页**检索  

### FS09 视频→文本
- **概念**：帧编码 + GRU 时序池化  
- **vs 图像阶段**：建模**运动**  

### FS10 音频→文本
- **概念**：波形 → log-mel → CNN/BiGRU  
- **vs FS09**：时频通路  

### FS11 条件生图
- **概念**：迷你 DDPM，类别条件去噪  
- **vs 理解阶段**：首次**合成像素**  

### FS12 条件生视频
- **概念**：潜变量+类别 → 帧序列  
- **vs FS11**：时间维  

### FS13 多模态 LM
- **概念**：图像 token 前缀 + 文本 token，统一自回归  
- **vs FS06**：LLaVA 式**单一接口**  

### FS14 Any-to-Any
- **概念**：共享 trunk + 模态适配器，路由 in×out  
- **vs FS13**：多条通路（图↔文↔音…）  

---

## 2. 如何复现

```bash
export KAGGLE_API_TOKEN=KGAT_...
python scripts/kaggle_run.py notebooks/Grok-multimodal-fs00-fs02-foundations --accelerator NvidiaTeslaT4
# 其余 kernel 同理
```

Notebook 命名：`Grok-multimodal-{阶段slug}`（符合 `Grok-{领域}-{任务}`）。
