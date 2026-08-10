#!/usr/bin/env python3
"""Push a notebook folder to Kaggle and wait for COMPLETE. Auto-retry on error.

Usage:
  python scripts/kaggle_run.py notebooks/Grok-ml-t4x2-smoke
  python scripts/kaggle_run.py notebooks/Grok-ml-t4x2-smoke --max-retries 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_env() -> None:
    token_path = Path.home() / ".kaggle" / "access_token"
    if not os.environ.get("KAGGLE_API_TOKEN") and token_path.is_file():
        os.environ["KAGGLE_API_TOKEN"] = token_path.read_text().strip()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit("KAGGLE_API_TOKEN missing. export it or write ~/.kaggle/access_token")
    # Prefer modern CLI venv
    venv_kaggle = Path("/opt/kaggle-venv/bin")
    if venv_kaggle.is_dir():
        os.environ["PATH"] = f"{venv_kaggle}:{os.environ.get('PATH', '')}"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=True, check=False if not check else False)


def kernel_ref_from_meta(folder: Path) -> str:
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    return meta["id"]


def push(folder: Path, accelerator: str, timeout: int | None) -> None:
    cmd = ["kaggle", "kernels", "push", "-p", str(folder), "--accelerator", accelerator]
    if timeout:
        cmd += ["-t", str(timeout)]
    cp = run(cmd)
    print(cp.stdout)
    if cp.returncode != 0:
        print(cp.stderr, file=sys.stderr)
        raise RuntimeError(f"push failed rc={cp.returncode}")


def status(ref: str) -> str:
    cp = run(["kaggle", "kernels", "status", ref])
    out = (cp.stdout or "") + (cp.stderr or "")
    print(out.strip())
    # status lines often: "has status \"complete\"" or similar
    # e.g. has status "KernelWorkerStatus.COMPLETE"
    m = re.search(r'KernelWorkerStatus\.([A-Za-z_]+)', out, re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r'status\s+"?([A-Za-z_\.]+)"?', out, re.I)
    if m:
        token = m.group(1).split(".")[-1].lower()
        return token
    low = out.lower()
    for s in ("complete", "error", "cancel", "running", "queued", "pending"):
        if s in low:
            return s
    return "unknown"


def logs(ref: str) -> str:
    cp = run(["kaggle", "kernels", "logs", ref])
    text = (cp.stdout or "") + (cp.stderr or "")
    print(text[-4000:])
    return text


def download_output(ref: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cp = run(["kaggle", "kernels", "output", ref, "-p", str(dest), "-o"])
    print(cp.stdout)
    if cp.returncode != 0:
        print(cp.stderr, file=sys.stderr)


def wait_done(ref: str, poll: int, max_wait: int) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_wait:
        st = status(ref)
        last = st
        if st in {"complete", "error", "cancel", "cancelled", "canceled"}:
            return st
        time.sleep(poll)
    return last or "timeout"


def maybe_autofix(folder: Path, log_text: str, attempt: int) -> bool:
    """Best-effort local autofix for common failures. Returns True if files changed."""
    ipynb = next(folder.glob("*.ipynb"), None)
    if not ipynb:
        return False
    changed = False
    text = ipynb.read_text()
    low = log_text.lower()

    # If assertion about dual GPU fails hard somewhere, soften already handled.
    # Memory errors → smaller GEMM
    if "out of memory" in low or "cuda out of memory" in low:
        if "N = 4096" in text:
            text = text.replace("N = 4096", "N = 2048")
            changed = True
            print("autofix: reduced GEMM N to 2048")
    # Module not found (should not happen for torch-only)
    if "modulenotfounderror" in low:
        print("autofix: dependency missing — enable_internet + pip not auto-applied for offline smoke")
    # DataParallel device issues
    if "output 0 of cudnn" in low or "cudnn" in low and "error" in low:
        if "epochs = 5" in text:
            text = text.replace("epochs = 5", "epochs = 3")
            changed = True
    if changed:
        ipynb.write_text(text)
        note = folder / f"autofix-attempt-{attempt}.md"
        note.write_text(f"Applied autofix on attempt {attempt}\n\nLog tail:\n```\n{log_text[-2000:]}\n```\n")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="kernel folder with kernel-metadata.json + ipynb")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4", help="T4×2 package id")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--max-wait", type=int, default=3600)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    ensure_env()
    folder = args.folder if args.folder.is_absolute() else (ROOT / args.folder)
    folder = folder.resolve()
    if not (folder / "kernel-metadata.json").is_file():
        sys.exit(f"missing kernel-metadata.json in {folder}")

    ref = kernel_ref_from_meta(folder)
    print(f"kernel ref: {ref}")
    print(f"accelerator: {args.accelerator} (Kaggle T4×2)")

    for attempt in range(1, args.max_retries + 1):
        print(f"\n======== ATTEMPT {attempt}/{args.max_retries} ========")
        push(folder, args.accelerator, args.timeout)
        # give Kaggle a moment to queue
        time.sleep(5)
        st = wait_done(ref, args.poll, args.max_wait)
        print("final status:", st)
        log_text = logs(ref)
        if st == "complete":
            if not args.no_download:
                dest = ROOT / "artifacts" / ref.replace("/", "__")
                download_output(ref, dest)
                print("outputs in", dest)
            print("SUCCESS on Kaggle")
            return 0
        # failed — try autofix then retry
        fixed = maybe_autofix(folder, log_text, attempt)
        if attempt == args.max_retries:
            print("FAILED after max retries", file=sys.stderr)
            return 1
        if not fixed:
            print("no autofix applied; retrying same code after backoff")
            time.sleep(15 * attempt)
        else:
            print("autofix applied; re-pushing")
            time.sleep(5)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
