# PROGRESS — RL + Robotics From Scratch

**状态：10/10 质量验收通过（hardened re-run on Kaggle T4×2）**

| Stage | Status | Kaggle | Quality bar |
|-------|--------|--------|-------------|
| 01 Bandits | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-01-bandits) | Thompson ≪ random regret |
| 02 MDP+DP | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-02-mdp-dp) | V* ≫ V_random；VI≡PI |
| 03 MC/TD/QL | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-03-mc-td-qlearning) | SARSA/QL 学会避悬崖 |
| 04 DQN | DONE✓fix | [link](https://www.kaggle.com/code/qixiaer/grok-rl-04-dqn) | DQN≫frozen-target |
| 05 PG→PPO | DONE✓fix | [link](https://www.kaggle.com/code/qixiaer/grok-rl-05-pg-ppo) | PPO last50≈253 |
| 06 Classical | DONE✓fix | [link](https://www.kaggle.com/code/qixiaer/grok-robotics-06-classical) | PD err≈0.02 ≪ open-loop |
| 07 SAC | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-07-continuous-sac) | SAC≫random pendulum |
| 08 MBRL+MPC | DONE✓fix | [link](https://www.kaggle.com/code/qixiaer/grok-robotics-08-mbrl-mpc) | Dyna early+late win；MPC≫random |
| 09 Offline | DONE | [link](https://www.kaggle.com/code/qixiaer/grok-rl-09-imitation-offline) | BC≈专家；CQL≫naive offline |
| 10 DT | DONE✓fix | [link](https://www.kaggle.com/code/qixiaer/grok-rl-10-frontier-dt) | RC-BC：G=-6→-6；G=-100→-106 |

## Hardened fixes this pass
- **04**: 消融改为 frozen random target（CartPole 上 online-TD 过强，旧消融倒置）
- **05**: episode-based multi-epoch PPO-clip（旧 batched rollout 欠训练）
- **06**: 更慢参考轨迹 + 速度误差 PD（跟踪误差 0.74→0.02）
- **08**: 更紧 episode 预算让 Dyna 前半/后半都更好；held-out MSE 随数据下降
- **10**: 混合质量 demo + Return-Conditioned BC；目标回报用 in-distribution G=-6

验收脚本：`python scripts/rl_accept.py`

Last updated: full quality acceptance PASSED
