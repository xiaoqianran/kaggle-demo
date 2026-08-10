# kaggle-demo

在 **Kaggle T4×2** 后台跑 notebook 的最小可复现仓库。  
本地只做：鉴权、推送、拉日志/产物；**训练与 CUDA 一律在 Kaggle 上执行**。

## 命名规范

Notebook：`Grok-{领域}-{任务}`

| 示例 | 领域 | 任务 |
|------|------|------|
| `Grok-ml-t4x2-smoke` | ml | t4x2-smoke |

目录：`notebooks/<Name>/`，内含：

- `<Name>.ipynb`
- `kernel-metadata.json`（`machine_shape: NvidiaTeslaT4` = T4×2）

## 鉴权

```bash
# 推荐：API token（KGAT_…）
export KAGGLE_API_TOKEN=KGAT_...
# 或写入
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
echo -n "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

CLI 需要 **Python ≥ 3.11** 的 `kaggle>=2`（含 `kaggle auth` / `kaggle quota` / `kernels logs`）。

```bash
# 本机示例（uv）
uv venv /opt/kaggle-venv --python 3.12
source /opt/kaggle-venv/bin/activate
uv pip install kaggle
kaggle quota
```

## Skills / MCP

- CLI skill：[`skills/SKILL.md`](skills/SKILL.md) + [`skills/references/`](skills/references/)
- 官方 skills：`write-kaggle-benchmarks`、`kaggle-standardized-agent-exam`、`hackathon-judging`
- MCP：[`.grok/config.toml`](.grok/config.toml) → `https://www.kaggle.com/mcp`，Header 用 `${KAGGLE_API_TOKEN}`

## 在 Kaggle 后台跑（T4×2）

```bash
source scripts/kaggle-env.sh
# 推送 + 轮询 + 失败自动重试/轻量修复
python scripts/kaggle_run.py notebooks/Grok-ml-t4x2-smoke --accelerator NvidiaTeslaT4
# 或
./scripts/run_on_kaggle.sh notebooks/Grok-ml-t4x2-smoke
```

成功后：

- 状态 `complete`
- 产物下载到 `artifacts/qiaojiajin__grok-ml-t4x2-smoke/`
- 应含 `results.json` 与 `SUCCESS`

## 常用命令

```bash
kaggle quota
kaggle kernels status qiaojiajin/grok-ml-t4x2-smoke
kaggle kernels logs qiaojiajin/grok-ml-t4x2-smoke --follow
kaggle kernels output qiaojiajin/grok-ml-t4x2-smoke -p artifacts/out -o
```

## 约定

- **禁止**在本地弱 CPU 上跑训练/大推理
- 密钥不入库（见 `.gitignore`）
- Commit 用 Conventional Commits：`feat(scope): …` / `fix: …` / `docs: …`
