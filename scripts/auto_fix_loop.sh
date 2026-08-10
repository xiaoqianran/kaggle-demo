#!/usr/bin/env bash
# Push/run on Kaggle T4x2; on failure apply heuristics and retry until success.
set -euo pipefail
FOLDER="${1:-notebooks/Grok-infra-t4x2-smoke}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh 2>/dev/null || true

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  echo "######## attempt $attempt / $MAX_ATTEMPTS ########"
  if ./scripts/run_on_kaggle_poll.sh "$FOLDER"; then
    echo "SUCCESS on attempt $attempt"
    exit 0
  fi
  echo "attempt $attempt failed — applying heuristics..."
  LOG="outputs/$(basename "$FOLDER")/kernel.log"
  META="$FOLDER/kernel-metadata.json"

  python3.11 - << PY
import json
from pathlib import Path
p = Path("$META")
m = json.loads(p.read_text())
m["enable_gpu"] = True
m["enable_tpu"] = False
m["enable_internet"] = True
m["machine_shape"] = "NvidiaTeslaT4"
p.write_text(json.dumps(m, indent=2) + "\n")
print("patched metadata: GPU + NvidiaTeslaT4")
PY

  if [[ -f "$LOG" ]] && grep -qiE "OutOfMemory|CUDA out of memory|OOM" "$LOG"; then
    python3.11 - << PY
import json
from pathlib import Path
folder = Path("$FOLDER")
ipynb = next(folder.glob("*.ipynb"))
nb = json.loads(ipynb.read_text())
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "4096, 4096" in src or "bs=256" in src:
        src = src.replace("4096, 4096", "2048, 2048").replace("bs=256", "bs=128")
        lines = src.split("\n")
        cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
        print("reduced workload for OOM")
ipynb.write_text(json.dumps(nb, indent=1) + "\n")
PY
  fi

  if [[ -f "$LOG" ]] && grep -qiE "loss did not|AssertionError" "$LOG"; then
    python3.11 - << PY
import json
from pathlib import Path
folder = Path("$FOLDER")
ipynb = next(folder.glob("*.ipynb"))
nb = json.loads(ipynb.read_text())
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "assert losses" in src:
        src = src.replace("range(80)", "range(150)").replace("lr=1e-3", "lr=3e-3")
        lines = src.split("\n")
        cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
        print("increased train steps / lr")
ipynb.write_text(json.dumps(nb, indent=1) + "\n")
PY
  fi

  attempt=$((attempt + 1))
  sleep 8
done
echo "FAILED after $MAX_ATTEMPTS attempts" >&2
exit 1
