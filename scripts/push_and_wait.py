#!/usr/bin/env python3
"""Push a Kaggle kernel and poll until complete. Exit 0 on success."""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from pathlib import Path

def run(cmd, check=True):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.stdout:
        print(p.stdout, end="" if p.stdout.endswith("\n") else "\n", flush=True)
    if p.stderr:
        print(p.stderr, end="" if p.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p

def status(kernel_id: str) -> str:
    p = run(["kaggle", "kernels", "status", kernel_id], check=False)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    # e.g. has status "KernelWorkerStatus.COMPLETE"
    m = re.search(r'KernelWorkerStatus\.([A-Za-z]+)', out)
    if m:
        return m.group(1).lower()
    m = re.search(r'status\s+"?([A-Za-z_.]+)"?', out, re.I)
    if m:
        token = m.group(1).lower().split(".")[-1]
        return token
    return out.lower() or "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    args = ap.parse_args()

    path = Path(args.path)
    meta = json.loads((path / "kernel-metadata.json").read_text())
    kernel_id = meta["id"]

    run([
        "kaggle", "kernels", "push",
        "-p", str(path),
        "-t", str(args.timeout),
        "--accelerator", args.accelerator,
    ])

    deadline = time.time() + args.timeout + 300
    last = ""
    while time.time() < deadline:
        st = status(kernel_id)
        if st != last:
            print(f"[status] {kernel_id}: {st}", flush=True)
            last = st
        if st in ("complete", "completed", "success"):
            out_dir = path / "kaggle_output"
            out_dir.mkdir(exist_ok=True)
            run(["kaggle", "kernels", "output", kernel_id, "-p", str(out_dir), "-o"], check=False)
            run(["kaggle", "kernels", "logs", kernel_id], check=False)
            print("SUCCESS", kernel_id)
            return 0
        if st in ("error", "failed", "cancel", "cancelled", "canceled"):
            run(["kaggle", "kernels", "logs", kernel_id], check=False)
            print("FAILED", kernel_id, st, file=sys.stderr)
            return 2
        time.sleep(args.poll)

    print("TIMEOUT waiting for", kernel_id, file=sys.stderr)
    return 3

if __name__ == "__main__":
    raise SystemExit(main())
