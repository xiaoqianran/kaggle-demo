#!/usr/bin/env bash
# Push → wait → on failure, pull logs, apply known fixes, re-push. Max N attempts.
# Usage: scripts/auto-fix-run.sh [notebook_folder] [max_attempts]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/kaggle-env.sh"

FOLDER="${1:-$ROOT/notebooks/Grok-gpu-t4x2-smoke}"
MAX="${2:-5}"
KERNEL_ID="$(python3 -c "import json; print(json.load(open('$FOLDER/kernel-metadata.json'))['id'])")"
LOGDIR="$ROOT/artifacts/logs"
mkdir -p "$LOGDIR"

attempt=1
while (( attempt <= MAX )); do
  echo ""
  echo "======== attempt $attempt / $MAX ========"
  if "$ROOT/scripts/push-and-wait.sh" "$FOLDER" 1800; then
    echo "Run succeeded on attempt $attempt"
    exit 0
  fi
  rc=$?
  logf="$LOGDIR/attempt-${attempt}.log"
  kaggle kernels logs "$KERNEL_ID" >"$logf" 2>&1 || true
  echo "Saved logs → $logf (rc=$rc)"

  # Known auto-fixes
  if grep -qiE 'cudaErrorNoKernelImageForDevice|no kernel image' "$logf"; then
    echo "FIX: P100/Pascal incompatibility → force NvidiaTeslaT4 in metadata + env"
    python3 - <<PY
import json
from pathlib import Path
p = Path("$FOLDER") / "kernel-metadata.json"
meta = json.loads(p.read_text())
meta["machine_shape"] = "NvidiaTeslaT4"
meta["enable_gpu"] = "true"
p.write_text(json.dumps(meta, indent=2) + "\n")
print("updated", p)
PY
    export ACCELERATOR=NvidiaTeslaT4
  elif grep -qiE 'CUDA not available|assert.*cuda' "$logf"; then
    echo "FIX: GPU not attached → re-enable GPU metadata"
    python3 - <<PY
import json
from pathlib import Path
p = Path("$FOLDER") / "kernel-metadata.json"
meta = json.loads(p.read_text())
meta["enable_gpu"] = "true"
meta["machine_shape"] = meta.get("machine_shape") or "NvidiaTeslaT4"
p.write_text(json.dumps(meta, indent=2) + "\n")
print("updated", p)
PY
    export ACCELERATOR=NvidiaTeslaT4
  elif grep -qiE 'ModuleNotFoundError|No module named' "$logf"; then
    echo "FIX: missing module — notebook should use stock Kaggle image only (torch)"
    # No internet install preferred; log for manual inspection
    grep -iE 'ModuleNotFoundError|No module named' "$logf" | tail -20 || true
  else
    echo "No automatic fix matched; re-pushing same code"
    tail -40 "$logf" || true
  fi

  attempt=$((attempt + 1))
  sleep 15
done

echo "ERROR: exhausted $MAX attempts" >&2
exit 1
