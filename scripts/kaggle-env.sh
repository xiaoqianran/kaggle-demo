#!/usr/bin/env bash
# Source this: source scripts/kaggle-env.sh
# Injects KGAT token + prefers modern kaggle CLI on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$HOME/.kaggle/access_token" && -z "${KAGGLE_API_TOKEN:-}" ]]; then
  export KAGGLE_API_TOKEN="$(tr -d '\n' < "$HOME/.kaggle/access_token")"
fi

if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; source "$ROOT/.env"; set +a
fi

# Prefer project/sandbox venv, then /opt, then whatever is on PATH
for candidate in \
  "$ROOT/.venv-kaggle/bin" \
  /workspace/.venv-kaggle/bin \
  /opt/kaggle-venv/bin
do
  if [[ -x "$candidate/kaggle" ]]; then
    export PATH="$candidate:$PATH"
    break
  fi
done

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "ERROR: set KAGGLE_API_TOKEN or put token in ~/.kaggle/access_token" >&2
  return 1 2>/dev/null || exit 1
fi

command -v kaggle >/dev/null || {
  echo "kaggle CLI not found (need Python ≥3.11 kaggle>=2)" >&2
  return 1 2>/dev/null || exit 1
}
