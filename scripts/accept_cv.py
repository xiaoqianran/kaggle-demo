#!/usr/bin/env python3
"""Acceptance gate for Grok-cv-* notebooks (local snapshots + optional live status)."""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    "depth-estimation","image-classification","object-detection","image-segmentation",
    "text-to-image","image-to-text","image-to-image","image-to-video",
    "unconditional-image-generation","video-classification","text-to-video",
    "zero-shot-image-classification","mask-generation","zero-shot-object-detection",
    "text-to-3d","image-to-3d","image-feature-extraction","keypoint-detection","video-to-video",
]

def check_result(t: str, r: dict) -> list[str]:
    errs = []
    if r.get("ok") is not True:
        errs.append(f"ok={r.get('ok')} error={r.get('error')}")
    gpu = r.get("gpu") or {}
    if gpu.get("device_count") != 2:
        errs.append(f"device_count={gpu.get('device_count')}")
    devs = gpu.get("devices") or []
    if len(devs) != 2 or not all("T4" in d for d in devs):
        errs.append(f"devices={devs}")
    # task fields
    need = {
        "depth-estimation": ["depth_shape"],
        "image-classification": ["label"],
        "object-detection": ["num_detections"],
        "image-segmentation": ["num_segments"],
        "text-to-image": ["image_shape"],
        "image-to-text": ["caption"],
        "image-to-image": ["image_shape"],
        "image-to-video": ["num_frames"],
        "unconditional-image-generation": ["image_shape"],
        "video-classification": ["label"],
        "text-to-video": ["num_frames"],
        "zero-shot-image-classification": ["best_label"],
        "mask-generation": ["mask_shape", "mask_coverage"],
        "zero-shot-object-detection": ["num_detections"],
        "text-to-3d": ["num_points"],
        "image-to-3d": ["num_points"],
        "image-feature-extraction": ["feature_dim"],
        "keypoint-detection": ["keypoints_shape"],
        "video-to-video": ["num_frames"],
    }
    for k in need.get(t, []):
        if r.get(k) in (None, "", [], {}):
            errs.append(f"missing {k}")
    # functional thresholds
    if t == "object-detection" and (r.get("num_detections") or 0) < 1:
        errs.append("num_detections < 1")
    if t == "zero-shot-object-detection" and (r.get("num_detections") or 0) < 1:
        errs.append("num_detections < 1")
    if t == "mask-generation" and (r.get("mask_coverage") or 0) <= 0.01:
        errs.append("empty mask")
    if t == "image-segmentation" and (r.get("num_segments") or 0) < 1:
        errs.append("no segments")
    if t in ("text-to-3d", "image-to-3d") and (r.get("num_points") or 0) < 100:
        errs.append("too few 3d points")
    if t == "image-feature-extraction" and (r.get("feature_dim") or 0) < 64:
        errs.append("feature_dim too small")
    return errs

def main() -> int:
    live = "--live" in sys.argv
    bad = []
    for t in TASKS:
        name = f"Grok-cv-{t}"
        folder = ROOT / "notebooks" / name
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        if meta.get("machine_shape") != "NvidiaTeslaT4" or not meta.get("enable_gpu"):
            bad.append((t, "metadata GPU/T4 invalid"))
            continue
        res = ROOT / "results" / f"{name}.json"
        if not res.exists():
            bad.append((t, "missing results snapshot"))
            continue
        r = json.loads(res.read_text())
        errs = check_result(t, r)
        if live:
            cp = subprocess.run(
                ["kaggle", "kernels", "status", meta["id"]],
                text=True, capture_output=True,
            )
            out = (cp.stdout or "") + (cp.stderr or "")
            if "COMPLETE" not in out:
                errs.append(f"live status not COMPLETE: {out.strip()[:100]}")
        if errs:
            bad.append((t, "; ".join(errs)))
        else:
            print(f"PASS {t}")
    print("---")
    if bad:
        print(f"FAIL {len(bad)}/{len(TASKS)}")
        for t, m in bad:
            print(f"  {t}: {m}")
        return 1
    print(f"ALL_PASS {len(TASKS)}/{len(TASKS)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
