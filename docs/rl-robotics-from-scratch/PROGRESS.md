# PROGRESS — RL + Robotics From Scratch

**状态：10/10 全量验收通过（含 Stage04/05 质量加固）**

| Stage | Status | Key metric |
|-------|--------|------------|
| 01 Bandits | DONE | Thompson best, regret ratio≈16.8 |
| 02 MDP+DP | DONE | V*≫random, VI≡PI |
| 03 MC/TD/QL | DONE | QL/SARSA learn cliff |
| 04 DQN | DONE✓ | DQN last50≈**288** ≫ frozen≈9 |
| 05 PG→PPO | DONE✓ | PPO≈220; **EMA baseline 460** (std 17 vs RF 166) |
| 06 Classical | DONE | PD err≈0.02 |
| 07 SAC | DONE | SAC≫random |
| 08 MBRL+MPC | DONE | Dyna early+late; MPC≫random |
| 09 Offline | DONE | BC=expert; CQL≫naive |
| 10 DT | DONE | RC G-cond; SeqBC=-6 |

## Latest harden
- Stage04: 3 seeds, stronger DQN (mean≈288, best≈398)
- Stage05: classic EMA return baseline multi-seed — higher mean + much lower seed variance than plain REINFORCE

Gates: `python scripts/rl_accept.py` · `python scripts/accept_all.py`
