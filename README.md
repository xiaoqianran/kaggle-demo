# kaggle-demo

在 **Kaggle T4×2** 后台跑 notebook 的最小可复现仓库。  
本地只做：鉴权、推送、拉日志/产物；**训练与 CUDA 一律在 Kaggle 上执行**。

## 命名规范

Notebook：`Grok-{领域}-{任务}`

| 示例 | 领域 | 任务 | Kaggle |
|------|------|------|--------|
| `Grok-ml-t4x2-smoke` | ml | t4x2-smoke | [qiaojiajin/grok-ml-t4x2-smoke](https://www.kaggle.com/code/qiaojiajin/grok-ml-t4x2-smoke) |
| `Grok-ml-gpu-smoke` | ml | gpu-smoke | 同加速器 T4×2 备用冒烟 |

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
uv venv /opt/kaggle-venv --python 3.12
source /opt/kaggle-venv/bin/activate
uv pip install kaggle
kaggle quota
```

## Skills / MCP

- CLI skill：[`skills/SKILL.md`](skills/SKILL.md) + [`skills/references/`](skills/references/)
- 官方：`write-kaggle-benchmarks`、`kaggle-standardized-agent-exam`、`hackathon-judging`
- MCP：[`.grok/config.toml`](.grok/config.toml) → `https://www.kaggle.com/mcp`（`${KAGGLE_API_TOKEN}`）

## 在 Kaggle 后台跑（T4×2）

```bash
source scripts/kaggle-env.sh

# 推荐：推送 + 轮询 + 失败自动重试/轻量修复 + 下载产物
python scripts/kaggle_run.py notebooks/Grok-ml-t4x2-smoke --accelerator NvidiaTeslaT4

# 轻量 shell 轮询
./scripts/push_and_wait.sh notebooks/Grok-ml-gpu-smoke
# 或
./scripts/run_on_kaggle.sh notebooks/Grok-ml-t4x2-smoke
```

成功标志：

- `kaggle kernels status …` → `KernelWorkerStatus.COMPLETE`
- 产物含 `results.json` / `SUCCESS`
- `device_count == 2` 且设备名为 Tesla T4

## 常用命令

```bash
kaggle quota
kaggle kernels status qiaojiajin/grok-ml-t4x2-smoke
kaggle kernels logs qiaojiajin/grok-ml-t4x2-smoke
kaggle kernels output qiaojiajin/grok-ml-t4x2-smoke -p artifacts/out -o
```

## 约定

- **禁止**在本地弱 CPU 上跑训练/大推理
- 密钥不入库（见 `.gitignore`）
- Commit 使用 Conventional Commits：`feat(scope): …` / `fix: …` / `docs: …`
