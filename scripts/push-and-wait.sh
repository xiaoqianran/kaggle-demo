#!/usr/bin/env bash
# Push a kernel folder to Kaggle and wait until the run completes.
# Usage: scripts/push-and-wait.sh notebooks/Grok-gpu-t4x2-smoke [timeout_sec]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/kaggle-env.sh"

FOLDER="${1:-$ROOT/notebooks/Grok-gpu-t4x2-smoke}"
TIMEOUT="${2:-1800}"
ACCELERATOR="${ACCELERATOR:-NvidiaTeslaT4}"
POLL="${POLL:-20}"

if [[ ! -f "$FOLDER/kernel-metadata.json" ]]; then
  echo "ERROR: no kernel-metadata.json in $FOLDER" >&2
  exit 1
fi

KERNEL_ID="$(python3 -c "import json; print(json.load(open('$FOLDER/kernel-metadata.json'))['id'])")"
echo "==> kernel: $KERNEL_ID"
echo "==> accelerator: $ACCELERATOR"
echo "==> push path: $FOLDER"

kaggle kernels push -p "$FOLDER" --accelerator "$ACCELERATOR" -t "$TIMEOUT"

echo "==> waiting for status (timeout ${TIMEOUT}s)..."
start=$(date +%s)
last=""
while true; do
  now=$(date +%s)
  elapsed=$((now - start))
  if (( elapsed > TIMEOUT + 120 )); then
    echo "ERROR: overall wait exceeded" >&2
    kaggle kernels status "$KERNEL_ID" || true
    kaggle kernels logs "$KERNEL_ID" 2>&1 | tail -80 || true
    exit 2
  fi
  status_raw="$(kaggle kernels status "$KERNEL_ID" 2>&1 || true)"
  # Normalize: COMPLETE / ERROR / CANCELLED / RUNNING / QUEUED / ...
  status="$(echo "$status_raw" | tr '[:upper:]' '[:lower:]')"
  if [[ "$status_raw" != "$last" ]]; then
    echo "[$elapsed s] $status_raw"
    last="$status_raw"
  fi
  if echo "$status" | grep -Eq 'complete|has run|success'; then
    echo "==> SUCCESS"
    mkdir -p "$ROOT/artifacts/$(basename "$FOLDER")"
    kaggle kernels output "$KERNEL_ID" -p "$ROOT/artifacts/$(basename "$FOLDER")" -o || true
    exit 0
  fi
  if echo "$status" | grep -Eq 'error|fail|cancel'; then
    echo "==> FAILED status: $status_raw" >&2
    echo "==> logs (tail):" >&2
    kaggle kernels logs "$KERNEL_ID" 2>&1 | tail -120 || true
    exit 3
  fi
  sleep "$POLL"
done
