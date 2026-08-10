#!/usr/bin/env python3
"""Quality-bar acceptance for RL+Robotics curriculum results."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "rl-robotics"

def load(n):
    return json.loads((RES / f"results_stage{n:02d}.json").read_text())

def main():
    fails = []
    # 01
    d = load(1)
    if not d.get("ok"): fails.append("01 ok")
    if d.get("best_agent") != "Thompson": fails.append("01 best_agent")
    if d.get("improvement_vs_random_regret_ratio", 0) < 5: fails.append("01 regret ratio")
    if d.get("gpu", {}).get("device_count", 0) != 2: fails.append("01 gpu")

    # 02
    d = load(2)
    if not d.get("ok"): fails.append("02 ok")
    if d.get("V_start_optimal", -1e9) <= d.get("V_start_random", 0): fails.append("02 V*")
    if not d.get("policies_match"): fails.append("02 VI/PI match")

    # 03
    d = load(3)
    m = d["metrics"]
    if max(m.values()) <= -50: fails.append("03 learning")
    if m["qlearning_last100"] > 0 and m["sarsa_last100"] > 0: pass

    # 04
    d = load(4)
    m = d["metrics"]
    dqn = m.get("dqn_last50_mean")
    if dqn is None: fails.append("04 missing dqn")
    else:
        weak = m.get("frozen_target_last50_mean", m.get("no_replay_last50_mean", m.get("no_target_last50_mean", -1e9)))
        if dqn < 120: fails.append(f"04 dqn weak {dqn}")
        if dqn <= weak + 40: fails.append(f"04 dqn not>weak {dqn} vs {weak}")

    # 05
    d = load(5)
    m = d["metrics"]
    ppo = m.get("ppo_last50", m.get("ppo_last40", m.get("ppo_last30", 0)))
    rf = m.get("reinforce_last50", m.get("reinforce_last40", m.get("reinforce_last30", 0)))
    if ppo < 150: fails.append(f"05 ppo weak {ppo}")
    if ppo + 50 < min(rf, 300) and ppo < rf - 80: fails.append(f"05 ppo<<rf {ppo} vs {rf}")

    # 06
    d = load(6)
    m = d["metrics"]
    if not (m["pd_mean_err"] < m["weak_pd_mean_err"] < m["openloop_mean_err"]):
        fails.append("06 order")
    if m["pd_mean_err"] > 0.25: fails.append(f"06 pd err {m['pd_mean_err']}")

    # 07
    d = load(7)
    m = d["metrics"]
    if m["sac_eval"] <= m["random_eval"] + 50: fails.append("07 sac")

    # 08
    d = load(8)
    m = d["metrics"]
    if m["early_dyna"] <= m["early_q"]: fails.append("08 early dyna")
    if m["true_mpc_pendulum"] <= m["random_pendulum"] + 100: fails.append("08 mpc")
    curve = m["model_heldout_curve"]
    if curve[-1]["heldout_mse"] >= curve[0]["heldout_mse"] * 0.5: fails.append("08 mse")

    # 09
    d = load(9)
    m = d["metrics"]
    if m["bc"] < -30: fails.append("09 bc")
    if m["offline_dqn"] >= m["cql"]: fails.append("09 cql should beat naive offline")

    # 10
    d = load(10)
    m = d["metrics"]
    if m["return_rtg_high"] <= m["return_rtg_low"] + 10: fails.append("10 rtg")
    if m["return_rtg_high"] < -25: fails.append(f"10 high weak {m['return_rtg_high']}")

    # gpu all
    for i in range(1, 11):
        g = load(i).get("gpu", {})
        if g.get("device_count") != 2:
            fails.append(f"{i:02d} not T4x2")

    if fails:
        print("ACCEPTANCE FAILED:")
        for f in fails: print(" -", f)
        return 1
    print("ACCEPTANCE PASSED: all 10 stages meet quality bar")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
