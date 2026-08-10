# kaggle-demo

在 **Kaggle T4×2** 后台跑 notebook 的最小可复现仓库。  
本地只做：鉴权、推送、拉日志/产物；**训练与 CUDA 一律在 Kaggle 上执行**。

## 命名规范

Notebook：`Grok-{领域}-{任务}`

| 示例 | 领域 | 任务 | Kaggle |
|------|------|------|--------|
| `Grok-ml-t4x2-smoke` | ml | t4x2-smoke | [qiaojiajin/grok-ml-t4x2-smoke](https://www.kaggle.com/code/qiaojiajin/grok-ml-t4x2-smoke) |
| `Grok-tabular-from-scratch` | tabular | from-scratch FS00–21 | [yunianan/grok-tabular-from-scratch](https://www.kaggle.com/code/yunianan/grok-tabular-from-scratch) ✅ |
| `Grok-ml-gpu-smoke` | ml | gpu-smoke | [zhengyingxiong/grok-ml-gpu-smoke](https://www.kaggle.com/code/zhengyingxiong/grok-ml-gpu-smoke) |
| `Grok-gpu-t4x2-smoke` | gpu | t4x2-smoke | [zhengyingxiong/grok-gpu-t4x2-smoke](https://www.kaggle.com/code/zhengyingxiong/grok-gpu-t4x2-smoke) |
| `Grok-ML-gpu-smoke` | ML | gpu-smoke | [yunianan/grok-ml-gpu-smoke](https://www.kaggle.com/code/yunianan/grok-ml-gpu-smoke) ✅ |
| `Grok-GPU-T4x2-Smoke` | GPU | T4x2-Smoke | [xiaosuhuaer/grok-gpu-t4x2-smoke](https://www.kaggle.com/code/xiaosuhuaer/grok-gpu-t4x2-smoke) ✅ |
| `Grok-infra-t4x2-smoke` | infra | t4x2-smoke | [shuhuaqqq/grok-infra-t4x2-smoke](https://www.kaggle.com/code/shuhuaqqq/grok-infra-t4x2-smoke) ✅ |
| `Grok-Audio-from-scratch` | Audio | from-scratch | [xiaosuhuaer/grok-audio-from-scratch](https://www.kaggle.com/code/xiaosuhuaer/grok-audio-from-scratch) ✅ |

目录：`notebooks/<Name>/`，内含：

- `<Name>.ipynb`
- `kernel-metadata.json`（`machine_shape: NvidiaTeslaT4` = **T4×2**）

## 鉴权

```bash
export KAGGLE_API_TOKEN=KGAT_...
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
echo -n "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

CLI 需要 **Python ≥ 3.11** 的 `kaggle>=2`：

```bash
uv venv .venv-kaggle --python 3.11   # 或 /opt/kaggle-venv + 3.12
source .venv-kaggle/bin/activate
uv pip install 'git+https://github.com/Kaggle/kaggle-cli.git'   # 或: uv pip install kaggle
source scripts/kaggle-env.sh
kaggle quota
```

## Skills / MCP

- CLI skill：[`skills/SKILL.md`](skills/SKILL.md) + [`skills/references/`](skills/references/)
- Official CLI skill mirror：[`skills/kaggle-cli/`](skills/kaggle-cli/)
- 镜像：[`docs/skills/`](docs/skills/)（write-kaggle-benchmarks / SAE / hackathon-judging）
- MCP：[`.grok/config.toml`](.grok/config.toml) → `https://www.kaggle.com/mcp`（`${KAGGLE_API_TOKEN}`）

## 在 Kaggle 后台跑（T4×2）

```bash
source scripts/kaggle-env.sh

# Python runner：推送 + 轮询 + 失败自动重试/轻量修复 + 下载产物
python scripts/kaggle_run.py notebooks/Grok-ml-t4x2-smoke --accelerator NvidiaTeslaT4

# 本仓库新增：push_and_wait.py（解析 KernelWorkerStatus.*）
python scripts/push_and_wait.py -p notebooks/Grok-GPU-T4x2-Smoke --accelerator NvidiaTeslaT4

# Shell：auto-fix 循环
./scripts/auto-fix-run.sh notebooks/Grok-gpu-t4x2-smoke
./scripts/auto_fix_loop.sh notebooks/Grok-infra-t4x2-smoke

# 轻量 shell 轮询
./scripts/push-and-wait.sh notebooks/Grok-gpu-t4x2-smoke
./scripts/push_and_wait.sh notebooks/Grok-ML-gpu-smoke
./scripts/run_on_kaggle.sh notebooks/Grok-ml-t4x2-smoke
./scripts/run_on_kaggle_poll.sh notebooks/Grok-infra-t4x2-smoke
```

成功标志：

- `kaggle kernels status …` → `KernelWorkerStatus.COMPLETE`
- 产物含 `results*.json` / `SUCCESS` / `SMOKE PASS` / `SMOKE_OK`
- `device_count == 2` 且设备名为 Tesla T4

已验证结果快照：

- `results/Grok-ml-t4x2-smoke.json`
- `results/Grok-gpu-t4x2-smoke.json`（`dual_gpu=true`, 2×Tesla T4）
- `results/Grok-ML-gpu-smoke-yunianan.json`（`device_count=2`, DataParallel, `SMOKE_OK`）
- `results/Grok-GPU-T4x2-Smoke.json`（`device_count=2`, DataParallel, `SMOKE PASS`，账号 `xiaosuhuaer`）
- `results/Grok-infra-t4x2-smoke.json`（`device_count=2`, 2×Tesla T4, loss 2.31→0.52, 账号 `shuhuaqqq`）
- `results/audio_from_scratch/` · Speech/Audio From Scratch S00–S10 全通过
- `PROGRESS.md` · 实验进度与能力阶梯

## 常用命令

```bash
kaggle quota
kaggle kernels status shuhuaqqq/grok-infra-t4x2-smoke
kaggle kernels logs shuhuaqqq/grok-infra-t4x2-smoke
kaggle kernels output shuhuaqqq/grok-infra-t4x2-smoke -p artifacts/out -o
```

## 约定

- **禁止**在本地弱 CPU 上跑训练/大推理
- 密钥不入库（见 `.gitignore`）
- Commit 使用 Conventional Commits（阿里规范）：`feat(scope): …` / `fix: …` / `docs: …`
- 加速器：`NvidiaTeslaT4`（平台 **T4×2**；勿用 P100 + 默认 cu128 镜像）

Repo: https://github.com/xiaoqianran/kaggle-demo

## RL + Robotics From Scratch

完整实验地图：[`docs/rl-robotics-from-scratch/ROADMAP.md`](docs/rl-robotics-from-scratch/ROADMAP.md)  
进度：[`docs/rl-robotics-from-scratch/PROGRESS.md`](docs/rl-robotics-from-scratch/PROGRESS.md)

全部在 **Kaggle T4×2** 执行。Notebook 命名 `Grok-rl-*` / `Grok-robotics-*`。
## NLP From Scratch

完整 NLP 实验路线（S00→S16）：经典 IR/统计 → 神经 from-scratch → 预训练现代系统。

- 目录：[`nlp-from-scratch/`](nlp-from-scratch/) · 进度：[`nlp-from-scratch/PROGRESS.md`](nlp-from-scratch/PROGRESS.md)
- Kaggle T4×2：
  - [Grok-nlp-neural-from-scratch](https://www.kaggle.com/code/zhengyingxiong/grok-nlp-neural-from-scratch) (S06–S10)
  - [Grok-nlp-modern-frontier](https://www.kaggle.com/code/zhengyingxiong/grok-nlp-modern-frontier) (S11–S16，全任务验收通过)
