# Contributing

## Commit message (Alibaba / Conventional Commits)

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

| type | when |
|------|------|
| feat | new notebook / capability |
| fix | bugfix (incl. Kaggle run fix) |
| docs | README only |
| refactor | no behavior change |
| test | smoke / verification |
| chore | tooling, deps, scripts |
| perf | performance |
| ci | automation |

Rules:
- subject: imperative mood, <= 72 chars, no trailing period
- scope: notebook domain or `scripts` / `infra`
- one logical change per commit
- never commit `KAGGLE_API_TOKEN`, `ghp_*`, or `~/.kaggle/*`

Example:

```
feat(infra): add Grok-infra-t4x2-smoke notebook

Run multi-GPU matmul + tiny train on Kaggle T4 x2.
```
