#!/usr/bin/env python3
"""S04 — TextRank extractive summarization (from scratch).

Concept: sentences as graph nodes; edge weight = content overlap; PageRank → central sentences.
Task: summarization (extractive).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import DOCS
from common.report import print_io, save_result


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def sentence_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    return len(A & B) / (math_log(len(A)) + math_log(len(B)) + 1e-8)


def math_log(x: int) -> float:
    import math

    return math.log(max(x, 1))


def textrank(sentences: list[str], damping: float = 0.85, iters: int = 50) -> np.ndarray:
    n = len(sentences)
    toks = [tokenize(s) for s in sentences]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            W[i, j] = sentence_overlap(toks[i], toks[j])
    # normalize rows
    row_sum = W.sum(axis=1, keepdims=True) + 1e-8
    M = W / row_sum
    scores = np.ones(n) / n
    for _ in range(iters):
        scores = (1 - damping) / n + damping * M.T @ scores
    return scores


def summarize(text: str, k: int = 2) -> dict:
    sents = split_sentences(text)
    if len(sents) <= k:
        return {"sentences": sents, "summary": " ".join(sents), "scores": [1.0] * len(sents)}
    scores = textrank(sents)
    # pick top-k by score, restore original order
    top_idx = sorted(np.argsort(-scores)[:k])
    summary = " ".join(sents[i] for i in top_idx)
    return {
        "sentences": sents,
        "scores": [float(s) for s in scores],
        "picked": [int(i) for i in top_idx],
        "summary": summary,
    }


def lead_baseline(text: str, k: int = 2) -> str:
    sents = split_sentences(text)
    return " ".join(sents[:k])


def main() -> None:
    print("=" * 60)
    print("S04 · TextRank Extractive Summarization")
    print("Concept: graph centrality of sentences = summary worthiness")
    print("=" * 60)

    demos = []
    results = []
    for doc in DOCS:
        out = summarize(doc["text"], k=2)
        lead = lead_baseline(doc["text"], k=2)
        print(f"\n--- {doc['title']} ---")
        print("Original sentences:")
        for i, (s, sc) in enumerate(zip(out["sentences"], out["scores"])):
            mark = "*" if i in out.get("picked", []) else " "
            print(f"  {mark} [{sc:.3f}] {s}")
        print(f"SUMMARY(TextRank): {out['summary']}")
        print(f"SUMMARY(Lead-2)  : {lead}")
        demos.append((f"[{doc['title']}] {doc['text'][:80]}...", out["summary"]))
        results.append(
            {
                "title": doc["title"],
                "textrank": out["summary"],
                "lead2": lead,
                "scores": out["scores"],
                "picked": out.get("picked", []),
            }
        )

    print_io("Document → extractive summary", demos)
    save_result(
        "S04_textrank_summarize",
        {
            "concept": "TextRank (PageRank on sentence similarity graph)",
            "results": results,
            "demos": [{"input": a, "output": b} for a, b in demos],
            "new_capability": "Unsupervised extractive summarization without labels",
            "previous": "S03 ranks docs; S04 ranks sentences inside a doc",
        },
    )
    print("\n[S04 DONE] New capability: extractive summarization.")


if __name__ == "__main__":
    main()
