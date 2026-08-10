#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/kaggle-env.sh"
FOLDER="${1:-notebooks/Grok-ml-t4x2-smoke}"
exec python3 "$ROOT/scripts/kaggle_run.py" "$FOLDER" "${@:2}"
