# PROGRESS — RL + Robotics From Scratch

**状态：10/10 质量验收通过（含 Stage10 MiniDT/SeqBC 最终修复）**

| Stage | Status | Kaggle | Quality bar |
|-------|--------|--------|-------------|
| 01 Bandits | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-01-bandits) | Thompson ≪ random regret |
| 02 MDP+DP | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-02-mdp-dp) | V* ≫ V_random；VI≡PI |
| 03 MC/TD/QL | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-03-mc-td-qlearning) | SARSA/QL 学会避悬崖 |
| 04 DQN | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-04-dqn) | DQN≫frozen-target |
| 05 PG→PPO | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-05-pg-ppo) | PPO last50≈253 |
| 06 Classical | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-robotics-06-classical) | PD err≈0.02 |
| 07 SAC | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-07-continuous-sac) | SAC≫random |
| 08 MBRL+MPC | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-robotics-08-mbrl-mpc) | Dyna early+late；MPC≫random |
| 09 Offline | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-09-imitation-offline) | BC≈专家；CQL≫naive offline |
| 10 DT | DONE✓final | [link](https://www.kaggle.com/code/qixiaer/grok-rl-10-frontier-dt) | RC G=-6→-6 / G=-100→-100；SeqBC=-6 |

## Final fix (this pass)
- Stage10 MiniDT：去掉 pad=真实 state0 的致命 bug；改为 **无 padding 变长历史 SeqBC**，专家回报 -6.0
- RC-BC 继续负责回报条件化证据
- `scripts/rl_accept.py` 要求 SeqBC 也达标
- `scripts/accept_cv.py` live 检查改用 PATH 上的 `kaggle` CLI（不再硬编码 python3.11 -m kaggle）
- `scripts/accept_all.py` 调用 `rl_accept.py` 质量门而非仅数文件

验收：`python scripts/rl_accept.py` → ACCEPTANCE PASSED  
Master：`python scripts/accept_all.py`
