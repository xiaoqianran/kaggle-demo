#!/usr/bin/env bash
# Source this: source scripts/kaggle-env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HOME/.kaggle/access_token" && -z "${KAGGLE_API_TOKEN:-}" ]]; then
  export KAGGLE_API_TOKEN="$(tr -d '\n' < "$HOME/.kaggle/access_token")"
fi
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; source "$ROOT/.env"; set +a
fi
if [[ -x /opt/kaggle-venv/bin/kaggle ]]; then
  export PATH="/opt/kaggle-venv/bin:$PATH"
fi
command -v kaggle >/dev/null || { echo "kaggle CLI not found"; exit 1; }
