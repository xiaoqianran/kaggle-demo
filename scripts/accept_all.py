#!/usr/bin/env python3
"""Master acceptance gate for kaggle-demo domains."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)

def main() -> int:
    fails: list[str] = []
    print("== CV ==")
    cp = run([sys.executable, "scripts/accept_cv.py", "--live"])
    print(cp.stdout, end="")
    if cp.returncode != 0:
        fails.append("cv")
    print("== NLP ==")
    cp = run([sys.executable, "nlp-from-scratch/scripts_accept.py"])
    out = (cp.stdout or "") + (cp.stderr or "")
    print(out[-400:])
    if "ALL ACCEPTANCE CHECKS PASSED" not in out:
        fails.append("nlp")
    print("== JSON domains ==")
    paths = [
        ROOT / "results/audio_from_scratch/ACCEPTANCE.json",
        ROOT / "results/Grok-graph-from-scratch-ACCEPTANCE.json",
        ROOT / "results/Grok-tabular-from-scratch-ACCEPTANCE.json",
        ROOT / "results/multimodal-from-scratch/results/ACCEPTANCE_MASTER.json",
    ]
    for p in paths:
        d = json.loads(p.read_text())
        ok = d.get("pass") is True or d.get("ok") is True
        if not ok or d.get("failures"):
            print(f"FAIL {p}")
            fails.append(str(p.relative_to(ROOT)))
        else:
            print(f"PASS {p.relative_to(ROOT)}")
    # RL stages present
    rl = ROOT / "results/rl-robotics"
    if rl.is_dir():
        stages = sorted(rl.glob("results_stage*.json"))
        if len(stages) < 10:
            fails.append(f"rl stages count={len(stages)}")
            print(f"FAIL rl stages count={len(stages)}")
        else:
            print(f"PASS rl stages={len(stages)}")
    print("---")
    if fails:
        print("MASTER_FAIL", fails)
        return 1
    print("MASTER_ALL_PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
