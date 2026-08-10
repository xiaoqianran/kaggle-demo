#!/usr/bin/env bash
set -euo pipefail
# Usage: push_and_wait.sh <kernel_folder> [accelerator]
FOLDER="${1:?kernel folder required}"
ACC="${2:-NvidiaTeslaT4}"
MAX_WAIT="${MAX_WAIT:-1800}"
POLL_SEC="${POLL_SEC:-15}"

if [[ -f /opt/kaggle-venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /opt/kaggle-venv/bin/activate
fi

if [[ ! -f "$FOLDER/kernel-metadata.json" ]]; then
  echo "missing kernel-metadata.json in $FOLDER" >&2
  exit 2
fi

SLUG=$(python3 -c "import json; print(json.load(open('${FOLDER}/kernel-metadata.json'))['id'])")
echo "==> Pushing ${SLUG} with accelerator=${ACC}"
kaggle kernels push -p "$FOLDER" --accelerator "$ACC"

echo "==> Polling ${SLUG} (max ${MAX_WAIT}s)"
python3 - "$SLUG" "$MAX_WAIT" "$POLL_SEC" "$FOLDER" <<'PY'
import sys, time, os
from kaggle.api.kaggle_api_extended import KaggleApi

slug, max_wait, poll, folder = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
api = KaggleApi(); api.authenticate()
start = time.time()
while True:
    st = api.kernels_status(slug)
    status = str(st.status).upper()
    fail = getattr(st, "failure_message", None) or ""
    elapsed = int(time.time() - start)
    print(f"[{elapsed}s] status={st.status} failure={fail!r}", flush=True)
    if "COMPLETE" in status:
        out = os.path.join(folder, "output")
        os.makedirs(out, exist_ok=True)
        try:
            api.kernels_output(slug, out)
        except Exception as e:
            print("output pull warning:", e, file=sys.stderr)
        if fail:
            print("COMPLETE with failure message:", fail, file=sys.stderr)
            raise SystemExit(1)
        print("SUCCESS")
        raise SystemExit(0)
    if any(x in status for x in ("ERROR", "FAILED", "CANCEL")):
        out = os.path.join(folder, "output")
        os.makedirs(out, exist_ok=True)
        try:
            api.kernels_output(slug, out)
        except Exception:
            pass
        print("FAILED", st.to_json(), file=sys.stderr)
        raise SystemExit(1)
    if elapsed > max_wait:
        print("TIMEOUT", file=sys.stderr)
        raise SystemExit(2)
    time.sleep(poll)
PY
