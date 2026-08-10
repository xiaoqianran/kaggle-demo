# Setup notes

1. Install Python 3.12 + `kaggle` CLI 2.x
2. Place `KAGGLE_API_TOKEN` in env or `~/.kaggle/access_token`
3. `kaggle quota` should show weekly GPU hours
4. Push kernels with `machine_shape` / `--accelerator NvidiaTeslaT4` for **T4×2**
5. Prefer `scripts/kaggle_run.py` for push → poll → logs → download → autofix loop
