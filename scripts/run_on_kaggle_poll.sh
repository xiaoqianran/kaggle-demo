#!/usr/bin/env bash
# Push notebook folder to Kaggle (T4 x2), poll until COMPLETE, pull logs/output.
set -euo pipefail

FOLDER="${1:-}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-1800}"
POLL_SEC="${POLL_SEC:-15}"

if [[ -z "$FOLDER" || ! -d "$FOLDER" ]]; then
  echo "usage: $0 <notebook-folder>" >&2
  exit 2
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "${HOME}/.kaggle/access_token" ]]; then
  export KAGGLE_API_TOKEN="$(cat "${HOME}/.kaggle/access_token")"
fi
if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN missing (or ~/.kaggle/access_token)" >&2
  exit 2
fi

META="$FOLDER/kernel-metadata.json"
[[ -f "$META" ]] || { echo "missing $META" >&2; exit 2; }

KERNEL_ID="$(python3.11 -c "import json;print(json.load(open('$META'))['id'])")"
echo "==> push $KERNEL_ID from $FOLDER (accelerator=NvidiaTeslaT4 / T4 x2)"
python3.11 -m kaggle kernels push -p "$FOLDER" --acc NvidiaTeslaT4

echo "==> poll status (max ${MAX_WAIT_SEC}s)"
python3.11 - "$KERNEL_ID" "$MAX_WAIT_SEC" "$POLL_SEC" <<'PY'
import sys, time
from kaggle.api.kaggle_api_extended import KaggleApi

kid, max_wait, poll = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
api = KaggleApi(); api.authenticate()
t0 = time.time()
while True:
    st = api.kernels_status(kid)
    status = getattr(st, "status", None)
    status_s = str(status).split(".")[-1].upper() if status is not None else ""
    fail = getattr(st, "failure_message", None) or getattr(st, "failureMessage", None) or ""
    elapsed = int(time.time() - t0)
    print(f"[{elapsed}s] status={status_s} failure={fail!r}", flush=True)
    if status_s in ("COMPLETE", "COMPLETED", "SUCCESS"):
        if fail:
            raise SystemExit(f"COMPLETE with failure: {fail}")
        print("KERNEL_SUCCESS", flush=True)
        sys.exit(0)
    if status_s in ("ERROR", "FAILED", "CANCELLED", "CANCEL_REQUESTED", "CANCELREQUESTED"):
        raise SystemExit(f"kernel failed: status={status_s} failure={fail}")
    if elapsed > max_wait:
        raise SystemExit(f"timeout after {max_wait}s last={status_s}")
    time.sleep(poll)
PY

OUT_DIR="outputs/$(basename "$FOLDER")"
mkdir -p "$OUT_DIR"
echo "==> pull logs + output -> $OUT_DIR"
python3.11 -m kaggle kernels logs "$KERNEL_ID" > "$OUT_DIR/kernel.raw.jsonl" 2>&1 || true
python3.11 -m kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR" -o 2>&1 || true

python3.11 - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
raw_path = out / "kernel.raw.jsonl"
text = ""
if raw_path.exists():
    raw = raw_path.read_text(errors="replace")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            text = "".join(ev.get("data", "") for ev in data if isinstance(ev, dict))
        else:
            text = raw
    except json.JSONDecodeError:
        text = raw
    (out / "kernel.log").write_text(text)

ok = False
if (out / "result.json").exists():
    try:
        r = json.loads((out / "result.json").read_text())
        ok = bool(r.get("ok"))
        print("result.json:", json.dumps(r, indent=2)[:800])
    except Exception as e:
        print("result parse err", e)
if "SMOKE_OK" in text:
    ok = True
if "Traceback (most recent call last)" in text and "SMOKE_OK" not in text:
    print(text[-1500:])
    raise SystemExit("traceback in kernel log")
if not ok:
    print("warning: no result.json/SMOKE_OK; treating as soft-success if no traceback")
print("artifacts:", sorted(p.name for p in out.iterdir()))
print("RUN_OK")
PY
