# kaggle-demo

在 **Kaggle T4×2** 后台跑 notebook 的最小可复现仓库。  
本地只做：鉴权、推送、拉日志/产物；**训练与 CUDA 一律在 Kaggle 上执行**。

## 命名规范

Notebook：`Grok-{领域}-{任务}`

| 示例 | 领域 | 任务 | Kaggle |
|------|------|------|--------|
| `Grok-ml-t4x2-smoke` | ml | t4x2-smoke | [qiaojiajin/grok-ml-t4x2-smoke](https://www.kaggle.com/code/qiaojiajin/grok-ml-t4x2-smoke) |
| `Grok-ml-gpu-smoke` | ml | gpu-smoke | [zhengyingxiong/grok-ml-gpu-smoke](https://www.kaggle.com/code/zhengyingxiong/grok-ml-gpu-smoke) |
| `Grok-gpu-t4x2-smoke` | gpu | t4x2-smoke | [zhengyingxiong/grok-gpu-t4x2-smoke](https://www.kaggle.com/code/zhengyingxiong/grok-gpu-t4x2-smoke) |

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
- 镜像：[`docs/skills/`](docs/skills/)（write-kaggle-benchmarks / SAE / hackathon-judging）
- MCP：[`.grok/config.toml`](.grok/config.toml) → `https://www.kaggle.com/mcp`（`${KAGGLE_API_TOKEN}`）

## 在 Kaggle 后台跑（T4×2）

```bash
source scripts/kaggle-env.sh

# Python runner：推送 + 轮询 + 失败自动重试/轻量修复 + 下载产物
python scripts/kaggle_run.py notebooks/Grok-ml-t4x2-smoke --accelerator NvidiaTeslaT4

# Shell：auto-fix 循环
./scripts/auto-fix-run.sh notebooks/Grok-gpu-t4x2-smoke

# 轻量 shell 轮询
./scripts/push-and-wait.sh notebooks/Grok-gpu-t4x2-smoke
./scripts/push_and_wait.sh notebooks/Grok-ml-gpu-smoke
./scripts/run_on_kaggle.sh notebooks/Grok-ml-t4x2-smoke
```

成功标志：

- `kaggle kernels status …` → `KernelWorkerStatus.COMPLETE`
- 产物含 `results*.json` / `SUCCESS`
- `device_count == 2` 且设备名为 Tesla T4

已验证结果快照：

- `results/Grok-ml-t4x2-smoke.json`
- `results/Grok-gpu-t4x2-smoke.json`（`dual_gpu=true`, 2×Tesla T4）

## 常用命令

```bash
kaggle quota
kaggle kernels status zhengyingxiong/grok-gpu-t4x2-smoke
kaggle kernels logs zhengyingxiong/grok-gpu-t4x2-smoke
kaggle kernels output zhengyingxiong/grok-gpu-t4x2-smoke -p artifacts/out -o
```

## 约定

- **禁止**在本地弱 CPU 上跑训练/大推理
- 密钥不入库（见 `.gitignore`）
- Commit 使用 Conventional Commits：`feat(scope): …` / `fix: …` / `docs: …`
- 加速器：`NvidiaTeslaT4`（平台 **T4×2**；勿用 P100 + 默认 cu128 镜像）

Repo: https://github.com/xiaoqianran/kaggle-demo
