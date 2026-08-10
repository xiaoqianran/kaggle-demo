#!/usr/bin/env python3
"""Run all Grok-cv-* notebooks on Kaggle T4x2 sequentially with auto-fix."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "cv"
RESULTS.mkdir(parents=True, exist_ok=True)

TASKS = [
    "depth-estimation",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-text",
    "image-to-image",
    "image-to-video",
    "unconditional-image-generation",
    "video-classification",
    "text-to-video",
    "zero-shot-image-classification",
    "mask-generation",
    "zero-shot-object-detection",
    "text-to-3d",
    "image-to-3d",
    "image-feature-extraction",
    "keypoint-detection",
    "video-to-video",
]


def ensure_env() -> None:
    token = Path.home() / ".kaggle" / "access_token"
    if not os.environ.get("KAGGLE_API_TOKEN") and token.is_file():
        os.environ["KAGGLE_API_TOKEN"] = token.read_text().strip()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit("KAGGLE_API_TOKEN missing")
    # ensure python3.11 kaggle on path
    os.environ["PATH"] = f"/usr/local/bin:{os.environ.get('PATH','')}"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=True)


def kaggle(*args: str) -> subprocess.CompletedProcess:
    return run(["python3.11", "-m", "kaggle", *args])


def status(ref: str) -> tuple[str, str]:
    cp = kaggle("kernels", "status", ref)
    out = (cp.stdout or "") + (cp.stderr or "")
    print(out.strip(), flush=True)
    m = re.search(r"KernelWorkerStatus\.([A-Za-z_]+)", out)
    if m:
        return m.group(1).upper(), out
    low = out.lower()
    for s in ("COMPLETE", "ERROR", "RUNNING", "QUEUED", "CANCEL"):
        if s.lower() in low:
            return s, out
    return "UNKNOWN", out


def wait_complete(ref: str, max_wait: int = 2400, poll: int = 15) -> str:
    t0 = time.time()
    while True:
        st, raw = status(ref)
        if st in ("COMPLETE", "COMPLETED", "SUCCESS"):
            return "COMPLETE"
        if st in ("ERROR", "FAILED") or "ERROR" in st:
            return "ERROR"
        if "CANCEL" in st:
            return "CANCEL"
        if time.time() - t0 > max_wait:
            return "TIMEOUT"
        time.sleep(poll)


def pull(ref: str, out_dir: Path) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "logs", ref)
    # save logs
    cp = kaggle("kernels", "logs", ref)
    (out_dir / "kernel.raw").write_text((cp.stdout or "") + (cp.stderr or ""))
    cp2 = kaggle("kernels", "output", ref, "-p", str(out_dir), "-o")
    print(cp2.stdout, cp2.stderr, flush=True)
    result = out_dir / "result.json"
    if result.exists():
        return json.loads(result.read_text())
    # parse SMOKE_OK from logs
    raw = (out_dir / "kernel.raw").read_text(errors="replace")
    if "SMOKE_OK" in raw:
        return {"ok": True, "from_logs": True}
    return None


def apply_fix(folder: Path, attempt: int, log_text: str) -> None:
    """Heuristic notebook fixes."""
    meta_p = folder / "kernel-metadata.json"
    meta = json.loads(meta_p.read_text())
    meta["enable_gpu"] = True
    meta["enable_internet"] = True
    meta["machine_shape"] = "NvidiaTeslaT4"
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")

    ipynb = folder / meta["code_file"]
    nb = json.loads(ipynb.read_text())
    changed = False
    src_all = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

    # OOM: reduce sizes
    if re.search(r"OutOfMemory|CUDA out of memory", log_text, re.I):
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            s = "".join(cell.get("source", []))
            s2 = s
            s2 = s2.replace("num_inference_steps=8", "num_inference_steps=4")
            s2 = s2.replace("num_inference_steps=6", "num_inference_steps=4")
            s2 = s2.replace("num_inference_steps=25", "num_inference_steps=12")
            s2 = s2.replace("height=256, width=256", "height=192, width=192")
            s2 = s2.replace("(512, 512)", "(384, 384)")
            if s2 != s:
                cell["source"] = [l + "\n" for l in s2.split("\n")[:-1]] + ([s2.split("\n")[-1] + "\n"] if s2 else [])
                changed = True
        print("fix: reduced compute for OOM", flush=True)

    # missing package imageio
    if "No module named 'imageio'" in log_text or "imageio" in log_text and "ModuleNotFound" in log_text:
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            s = "".join(cell.get("source", []))
            if "import imageio" in s and "pip install" not in s:
                install = "import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'imageio', 'imageio-ffmpeg'])\n"
                s2 = install + s
                cell["source"] = [l + "\n" for l in s2.split("\n")[:-1]] + ([s2.split("\n")[-1] + "\n"] if s2 else [])
                changed = True
                print("fix: pip install imageio", flush=True)
                break

    # model download / not found → switch to more common models
    if re.search(r"404|Not Found|Repository Not Found|is not a valid model", log_text, re.I):
        replacements = {
            "nota-ai/bk-sdm-tiny": "hf-internal-testing/tiny-stable-diffusion-pipe",
            "Intel/dpt-hybrid-midas": "Intel/dpt-large",
            "MCG-NJU/videomae-base-finetuned-kinetics": "MCG-NJU/videomae-base",
        }
        # tiny-stable-diffusion-pipe may not work the same — better use runwayml
        replacements["nota-ai/bk-sdm-tiny"] = "CompVis/stable-diffusion-v1-4"
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            s = "".join(cell.get("source", []))
            s2 = s
            for a, b in replacements.items():
                s2 = s2.replace(a, b)
            # fewer steps if full SD
            if "stable-diffusion-v1-4" in s2:
                s2 = s2.replace("num_inference_steps=8", "num_inference_steps=4")
            if s2 != s:
                cell["source"] = [l + "\n" for l in s2.split("\n")[:-1]] + ([s2.split("\n")[-1] + "\n"] if s2 else [])
                changed = True
        print("fix: model id replacements", flush=True)

    # safety_checker issues
    if "safety_checker" in log_text and "unexpected" in log_text.lower():
        pass

    if changed:
        ipynb.write_text(json.dumps(nb, indent=1))


def run_one(task: str, max_attempts: int = 5) -> bool:
    name = f"Grok-cv-{task}"
    folder = ROOT / "notebooks" / name
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    ref = meta["id"]
    out_dir = RESULTS / name

    for attempt in range(1, max_attempts + 1):
        print(f"\n######## {name} attempt {attempt}/{max_attempts} ########", flush=True)
        # wait if prior GPU sessions still running
        for _ in range(40):
            cp = kaggle("kernels", "push", "-p", str(folder), "--acc", "NvidiaTeslaT4")
            out = (cp.stdout or "") + (cp.stderr or "")
            print(out, flush=True)
            if cp.returncode == 0:
                break
            if "Maximum batch GPU" in out or "session count" in out:
                print("GPU session full — sleep 30s", flush=True)
                time.sleep(30)
                continue
            if "500 Server Error" in out or "Internal Server Error" in out:
                print("500 — sleep 15s", flush=True)
                time.sleep(15)
                continue
            apply_fix(folder, attempt, out)
            time.sleep(8)
            break
        else:
            continue
        if cp.returncode != 0:
            continue

        st = wait_complete(ref)
        result = pull(ref, out_dir)
        log_text = (out_dir / "kernel.raw").read_text(errors="replace") if (out_dir / "kernel.raw").exists() else ""

        if st == "COMPLETE" and result and result.get("ok", True) is not False and (
            result.get("ok") is True or "SMOKE_OK" in log_text or result.get("from_logs")
        ):
            # require ok True if present
            if result.get("ok") is False:
                print("result.ok=false", result, flush=True)
            else:
                dest = ROOT / "results" / f"{name}.json"
                if (out_dir / "result.json").exists():
                    shutil.copy(out_dir / "result.json", dest)
                else:
                    dest.write_text(json.dumps(result, indent=2))
                print(f"SUCCESS {name}", flush=True)
                return True

        print(f"FAIL {name} status={st}", flush=True)
        # dump tail of logs
        print(log_text[-2500:], flush=True)
        apply_fix(folder, attempt, log_text)
        time.sleep(8)

    return False


def main():
    ensure_env()
    only = sys.argv[1:]  # optional task filter
    tasks = only if only else TASKS
    summary = {}
    for task in tasks:
        # allow passing full name or task slug
        task = task.replace("Grok-cv-", "").strip("/")
        ok = run_one(task)
        summary[task] = "ok" if ok else "fail"
        (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    failed = [k for k, v in summary.items() if v != "ok"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
