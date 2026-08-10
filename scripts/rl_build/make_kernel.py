#!/usr/bin/env python3
"""Build a Kaggle kernel folder from markdown+code cell specs."""
from __future__ import annotations
import argparse, json, uuid, re
from pathlib import Path

def slugify_title(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")

def make_nb(cells_spec: list[dict]) -> dict:
    cells = []
    for spec in cells_spec:
        cid = uuid.uuid4().hex[:12]
        if spec["type"] == "md":
            src = spec["source"]
            if not src.endswith("\n"):
                src += "\n"
            cells.append({
                "cell_type": "markdown",
                "id": cid,
                "metadata": {},
                "source": [line + "\n" for line in src.split("\n")[:-1]] + ([src.split("\n")[-1] + "\n"] if src.split("\n")[-1] != "" else []),
            })
            # simplify source as single list of lines
            cells[-1]["source"] = [l + "\n" for l in spec["source"].split("\n")]
            if cells[-1]["source"] and cells[-1]["source"][-1] == "\n":
                cells[-1]["source"][-1] = ""
            # better: keep clean
            text = spec["source"]
            if not text.endswith("\n"):
                text += "\n"
            cells[-1]["source"] = [text]
        else:
            text = spec["source"]
            if not text.endswith("\n"):
                text += "\n"
            cells.append({
                "cell_type": "code",
                "id": cid,
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [text],
            })
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }

def write_kernel(root: Path, title: str, username: str, cells: list[dict], internet: bool = False, private: bool = True):
    folder = root / "notebooks" / title
    folder.mkdir(parents=True, exist_ok=True)
    nb_name = f"{title}.ipynb"
    nb = make_nb(cells)
    (folder / nb_name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    slug = slugify_title(title)
    meta = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true" if private else "false",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true" if internet else "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {folder} -> {meta['id']}")
    return folder

if __name__ == "__main__":
    print("import make_kernel helpers from other stage scripts")
