#!/usr/bin/env bash
set -uo pipefail
cd /workspace/kaggle-demo
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(cat /root/.kaggle/access_token)}"
export PATH="/usr/local/bin:$PATH"
LOG=results/cv/remaining.log
mkdir -p results/cv

TASKS=(
  video-classification
  text-to-video
  zero-shot-image-classification
  mask-generation
  zero-shot-object-detection
  text-to-3d
  image-to-3d
  image-feature-extraction
  keypoint-detection
  video-to-video
)

for t in "${TASKS[@]}"; do
  if [[ -f "results/Grok-cv-$t.json" ]]; then
    echo "SKIP $t already ok" | tee -a "$LOG"
    continue
  fi
  echo "==== RUN $t $(date -Is) ====" | tee -a "$LOG"
  # max 4 attempts inside python
  python3.11 scripts/run_cv_batch.py "$t" >>"$LOG" 2>&1
  rc=$?
  echo "==== DONE $t rc=$rc $(date -Is) ====" | tee -a "$LOG"
done

python3.11 - <<'PY' | tee -a results/cv/remaining.log
import json
from pathlib import Path
all_tasks = [
"depth-estimation","image-classification","object-detection","image-segmentation",
"text-to-image","image-to-text","image-to-image","image-to-video",
"unconditional-image-generation","video-classification","text-to-video",
"zero-shot-image-classification","mask-generation","zero-shot-object-detection",
"text-to-3d","image-to-3d","image-feature-extraction","keypoint-detection","video-to-video"
]
s={}
for t in all_tasks:
    p=Path(f"results/Grok-cv-{t}.json")
    s[t]="ok" if p.exists() else "missing"
print(json.dumps(s, indent=2))
Path("results/cv/summary.json").write_text(json.dumps(s, indent=2))
print("OK", sum(1 for v in s.values() if v=="ok"), "/", len(s))
PY
echo ALL_FINISHED >> "$LOG"
