# Reinforcement Learning + Robotics — From Scratch 实验地图

领域：强化学习（RL）× 机器人技术（Robotics）  
执行环境：**Kaggle NvidiaTeslaT4（T4×2）**，本地只做编排  
命名：`Grok-{domain}-{task}`

## 全路线（从零 → 前沿）

| # | 阶段 | Notebook | 解决的问题 | 关键新增能力 |
|---|------|----------|------------|--------------|
| 01 | 从零：试错与探索 | `Grok-rl-01-bandits` | 不知道哪只手臂最好 | ε-greedy / UCB / Thompson；遗憾(regret) |
| 02 | 序列决策基础 | `Grok-rl-02-mdp-dp` | 有模型时如何最优规划 | MDP、Bellman、VI/PI |
| 03 | 无模型控制 | `Grok-rl-03-mc-td-qlearning` | 无模型时如何学策略 | MC、TD、SARSA、Q-Learning |
| 04 | 函数逼近 + 深度价值 | `Grok-rl-04-dqn` | 状态空间爆炸 | Linear FA → DQN（replay+target） |
| 05 | 策略梯度族 | `Grok-rl-05-pg-ppo` | 直接优化策略 | REINFORCE → A2C → **PPO** |
| 06 | 机器人经典控制 | `Grok-robotics-06-classical` | 不用 RL 先能动起来 | 运动学、PD/PID、轨迹跟踪 |
| 07 | 连续控制深度 RL | `Grok-rl-07-continuous-sac` | 连续动作机器人 | Actor-Critic 连续；**SAC** |
| 08 | 模型基 + 规划 | `Grok-robotics-08-mbrl-mpc` | 样本效率 / 可控规划 | Dyna、随机射击 **MPC** |
| 09 | 模仿 + 离线 | `Grok-rl-09-imitation-offline` | 不会探索/无在线交互 | BC、Offline RL（CQL-lite） |
| 10 | 前沿：序列决策 Transformer | `Grok-rl-10-frontier-dt` | 把 RL 当条件生成 | **Decision Transformer** lite |

## 每阶段验收标准

1. **概念**：一句话说清“这一步解决什么痛点”
2. **最小实现**：可运行代码（非伪代码）
3. **真实输入**：可改 seed / 环境参数 / 算法超参
4. **可观察输出**：曲线、表格、策略可视化、JSON 指标
5. **与上一步对比**：量化说明新增能力
6. **产物**：`/kaggle/working/results_*.json` + 图 PNG

## 依赖原则

- 环境尽量 **from scratch（numpy 物理/网格）**，不依赖黑盒 gym（阶段后期可用 torch）
- 训练长度以保证 **Kaggle 上可在 ~10–20 min 内完成** 为上限，同时保证曲线可辨
- 失败自动修：重推 notebook，直到 `COMPLETE` 且 `ok=true`
