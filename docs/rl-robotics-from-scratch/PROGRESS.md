# PROGRESS — RL + Robotics From Scratch

| Stage | Status | Kaggle | Key Metric | Notes |
|-------|--------|--------|------------|-------|
| 00 Roadmap | DONE | — | full map | ROADMAP.md |
| 01 Bandits | **DONE** | [qixiaer/grok-rl-01-bandits](https://www.kaggle.com/code/qixiaer/grok-rl-01-bandits) | Thompson regret≈83 vs random≈1398 | T4×2, STAGE01_OK |
| 02 MDP+DP | RUNNING | — | — | — |
| 03 MC/TD/QL | PENDING | | | |
| 04 DQN | PENDING | | | |
| 05 PG→PPO | PENDING | | | |
| 06 Classical Robotics | PENDING | | | |
| 07 SAC Continuous | PENDING | | | |
| 08 MBRL+MPC | PENDING | | | |
| 09 Imitation+Offline | PENDING | | | |
| 10 Frontier DT | PENDING | | | |

## Stage 01 结果摘要
- best: **Thompson Sampling** final_avg_reward≈0.531, opt_rate≈93%
- vs random: regret ratio ~16.8× better
- GPU: 2× Tesla T4 confirmed

Last updated: after stage 01 COMPLETE
