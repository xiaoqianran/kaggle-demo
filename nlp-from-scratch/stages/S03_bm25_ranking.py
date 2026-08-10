#!/usr/bin/env python3
"""S03 — BM25 text ranking (from scratch).

Concept: probabilistic IR ranking — term frequency saturates, long docs penalized, IDF boosts rare terms.
Task: text ranking (query → ordered documents).
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import DOCS
from common.report import print_io, save_result


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, docs: list[dict]) -> "BM25":
        self.docs = docs
        self.tok_docs = [tokenize(d["title"] + " " + d["text"]) for d in docs]
        self.N = len(docs)
        self.doc_len = [len(t) for t in self.tok_docs]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        df = Counter()
        for toks in self.tok_docs:
            df.update(set(toks))
        self.df = df
        self.idf = {
            w: math.log(1 + (self.N - df[w] + 0.5) / (df[w] + 0.5)) for w in df
        }
        return self

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        tf = Counter(self.tok_docs[idx])
        dl = self.doc_len[idx]
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            freq = tf[t]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (freq * (self.k1 + 1)) / denom
        return s

    def rank(self, query: str, topk: int = 4) -> list[tuple[str, float, str]]:
        scored = [(self.docs[i]["id"], self.score(query, i), self.docs[i]["title"]) for i in range(self.N)]
        scored.sort(key=lambda x: -x[1])
        return scored[:topk]


def main() -> None:
    print("=" * 60)
    print("S03 · BM25 Ranking")
    print("Concept: rank documents for a query with TF saturation + length norm + IDF")
    print("=" * 60)

    bm25 = BM25().fit(DOCS)
    queries = [
        "how do plants make food from sunlight",
        "force mass acceleration law",
        "learning patterns from labeled data",
        "1789 liberty equality revolution",
        "oxygen produced by chlorophyll",
    ]

    demos = []
    rankings = {}
    for q in queries:
        ranked = bm25.rank(q)
        rankings[q] = [{"id": i, "score": s, "title": t} for i, s, t in ranked]
        out = " | ".join(f"{t}({s:.2f})" for _, s, t in ranked)
        demos.append((q, out))
        print(f"\nQ: {q}")
        for i, s, t in ranked:
            print(f"  {s:7.3f}  {i}  {t}")

    # Contrast: plain term-count ranking (no IDF/length)
    def count_rank(q):
        qt = set(tokenize(q))
        scored = []
        for d, toks in zip(DOCS, bm25.tok_docs):
            scored.append((d["title"], len(qt & set(toks))))
        scored.sort(key=lambda x: -x[1])
        return scored

    print("\n=== Contrast BM25 vs raw overlap count ===")
    contrasts = []
    for q in queries[:3]:
        b = [t for _, _, t in bm25.rank(q)]
        c = [t for t, _ in count_rank(q)]
        contrasts.append({"query": q, "bm25": b, "count": c})
        print(f"  Q: {q}")
        print(f"    BM25 : {b}")
        print(f"    Count: {c}")

    print_io("Query → ranked titles", demos)
    save_result(
        "S03_bm25_ranking",
        {
            "concept": "Okapi BM25 classical IR ranking",
            "rankings": rankings,
            "contrast_vs_count": contrasts,
            "demos": [{"input": a, "output": b} for a, b in demos],
            "new_capability": "Principled multi-document ranking for search-like queries",
            "previous": "S02 similarity was pairwise vectors; BM25 is IR ranking",
        },
    )
    print("\n[S03 DONE] New capability: text ranking with BM25.")


if __name__ == "__main__":
    main()
