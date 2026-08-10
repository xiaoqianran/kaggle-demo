# kaggle-demo

Remote GPU workflows on Kaggle (T4×2), orchestrated from CLI and synced to GitHub.

## Naming

Notebooks: `Grok-{domain}-{task}`  
Example: `Grok-ml-gpu-smoke`

## Auth

- GitHub: `gh auth login`
- Kaggle: place access token in `~/.kaggle/access_token` (KGAT_…)

## Run on Kaggle (not local CPU)

```bash
source /opt/kaggle-venv/bin/activate
./scripts/push_and_wait.sh notebooks/Grok-ml-gpu-smoke NvidiaTeslaT4
```

## Layout

```
notebooks/
  Grok-ml-gpu-smoke/     # kernel source + kernel-metadata.json
scripts/
  push_and_wait.sh       # push → poll → pull output; exit non-zero on fail
```
