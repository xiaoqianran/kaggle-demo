"""Save stage results as JSON + markdown snippets."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
VIZ = ROOT / "viz"
RESULTS.mkdir(parents=True, exist_ok=True)
VIZ.mkdir(parents=True, exist_ok=True)


def save_result(stage_id: str, payload: dict[str, Any]) -> Path:
    payload = {
        **payload,
        "stage": stage_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = RESULTS / f"{stage_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {path}")
    return path


def print_io(title: str, pairs: list[tuple[str, str]]) -> None:
    print(f"\n=== {title} ===")
    for inp, out in pairs:
        print(f"  IN : {inp}")
        print(f"  OUT: {out}")
        print()
