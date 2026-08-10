# PROGRESS — RL + Robotics From Scratch

全部阶段在 **Kaggle T4×2**（`NvidiaTeslaT4`）执行并通过。

| Stage | Status | Kaggle | 关键可观察结果 |
|-------|--------|--------|----------------|
| 00 Roadmap | DONE | — | [ROADMAP.md](ROADMAP.md) |
| 01 Bandits | **DONE** | [grok-rl-01-bandits](https://www.kaggle.com/code/qixiaer/grok-rl-01-bandits) | Thompson regret≈83 ≪ random≈1398 |
| 02 MDP+DP | **DONE** | [grok-rl-02-mdp-dp](https://www.kaggle.com/code/qixiaer/grok-rl-02-mdp-dp) | V*(start)≫V_random；VI/PI 策略一致 |
| 03 MC/TD/QL | **DONE** | [grok-rl-03-mc-td-qlearning](https://www.kaggle.com/code/qixiaer/grok-rl-03-mc-td-qlearning) | SARSA / Q-Learning / MC 悬崖策略对比 |
| 04 DQN | **DONE** | [grok-rl-04-dqn](https://www.kaggle.com/code/qixiaer/grok-rl-04-dqn) | CartPole 连续状态神经 Q |
| 05 PG→PPO | **DONE** | [grok-rl-05-pg-ppo](https://www.kaggle.com/code/qixiaer/grok-rl-05-pg-ppo) | REINFORCE / baseline / PPO |
| 06 Classical Robotics | **DONE** | [grok-robotics-06-classical](https://www.kaggle.com/code/qixiaer/grok-robotics-06-classical) | 2R 臂 FK/IK + PD 跟踪误差更低 |
| 07 Continuous SAC | **DONE** | [grok-rl-07-continuous-sac](https://www.kaggle.com/code/qixiaer/grok-rl-07-continuous-sac) | SAC eval≈-152 ≫ random≈-1225 |
| 08 MBRL+MPC | **DONE** | [grok-robotics-08-mbrl-mpc](https://www.kaggle.com/code/qixiaer/grok-robotics-08-mbrl-mpc) | Dyna 早期更快；真模型 MPC≫random |
| 09 Imitation+Offline | **DONE** | [grok-rl-09-imitation-offline](https://www.kaggle.com/code/qixiaer/grok-rl-09-imitation-offline) | BC≈专家；朴素 offline DQN 坠崖；CQL 稳健 |
| 10 Frontier DT | **DONE** | [grok-rl-10-frontier-dt](https://www.kaggle.com/code/qixiaer/grok-rl-10-frontier-dt) | 高 RTG 回报 > 低 RTG（条件生成策略） |

## 能力阶梯（一句话）

```
探索臂 → 有模型规划 → 无模型采样控制 → 深度价值 → 策略梯度/PPO
→ 机器人运动学+PD → 连续力矩 SAC → 模型复用/MPC → 离线/模仿 → DT 序列决策
```

## 产物位置

- Notebooks: `notebooks/Grok-rl-*`, `notebooks/Grok-robotics-*`
- 结果 JSON/图: `results/rl-robotics/`
- 运行器: `scripts/kaggle_run.py`（T4×2 push + 轮询 + 下载）

Last updated: full curriculum COMPLETE on Kaggle T4×2
