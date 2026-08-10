#!/usr/bin/env python3
"""S02 — Skip-gram style embeddings (SGNS via PPMI-SVD) + sentence similarity.

Concept (distributional hypothesis): words in similar contexts get similar vectors.
On tiny data we use PPMI + SVD (classic cheap SGNS approximation).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import CLS_TRAIN, DOCS, MT_PAIRS
from common.report import print_io, save_result


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


# Expand corpus slightly with paraphrases so co-occurrence has signal
EXTRA = [
    "I love this fantastic amazing wonderful movie film cinema",
    "I hate this terrible awful horrible boring movie film",
    "investors buy shares stock market profits earnings growth company",
    "team won championship final sports athlete coach victory",
    "photosynthesis plants sunlight chlorophyll leaves oxygen glucose energy",
    "newton force mass acceleration motion reaction laws physics",
    "machine learning algorithms supervised labeled examples neural networks data",
    "french revolution liberty equality fraternity 1789 bastille politics",
]


def build_embeddings(sentences: list[list[str]], window: int = 3, dim: int = 32):
    vocab = sorted({w for s in sentences for w in s})
    v2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    co = np.zeros((V, V), dtype=np.float64)
    for s in sentences:
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if i == j:
                    continue
                co[v2i[w], v2i[s[j]]] += 1.0
    # add small self to stabilize
    co += np.eye(V) * 0.1
    row = co.sum(axis=1, keepdims=True) + 1e-8
    col = co.sum(axis=0, keepdims=True) + 1e-8
    total = co.sum() + 1e-8
    pmi = np.log((co * total) / (row @ col) + 1e-12)
    ppmi = np.maximum(pmi, 0.0)
    # SVD
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    k = min(dim, U.shape[1])
    emb = U[:, :k] * np.sqrt(S[:k])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return vocab, v2i, emb


def sent_vec(tokens, v2i, emb):
    idxs = [v2i[t] for t in tokens if t in v2i]
    if not idxs:
        return np.zeros(emb.shape[1])
    v = emb[idxs].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a, b):
    return float(a @ b)


def jaccard(a, b):
    A, B = set(tokenize(a)), set(tokenize(b))
    return len(A & B) / max(1, len(A | B))


def main() -> None:
    print("=" * 60)
    print("S02 · PPMI-SVD Embeddings + Sentence Similarity")
    print("Concept: similar contexts → similar vectors; mean-pool for sentences")
    print("=" * 60)

    corpus = (
        [tokenize(t) for t, _ in CLS_TRAIN]
        + [tokenize(d["text"]) for d in DOCS]
        + [tokenize(a) + tokenize(b) for a, b in MT_PAIRS]
        + [tokenize(x) for x in EXTRA]
    )
    vocab, v2i, emb = build_embeddings(corpus, dim=40)
    print(f"vocab={len(vocab)} emb_dim={emb.shape[1]}")

    def nn(word, k=5):
        if word not in v2i:
            return []
        v = emb[v2i[word]]
        sims = emb @ v
        order = np.argsort(-sims)
        out = []
        for i in order:
            if vocab[i] == word:
                continue
            out.append((vocab[i], float(sims[i])))
            if len(out) >= k:
                break
        return out

    probes = ["movie", "love", "team", "market", "photosynthesis", "force"]
    word_nn = {}
    print("=== Word nearest neighbors ===")
    for w in probes:
        word_nn[w] = nn(w)
        print(f"  {w}: {word_nn[w]}")

    pairs = [
        ("I love this amazing movie", "Fantastic film highly recommend"),
        ("I love this amazing movie", "The team won the championship final"),
        ("Investors buy shares amid growth", "Company profits beat estimates"),
        ("Plants convert sunlight into energy", "Photosynthesis produces oxygen and glucose"),
        ("Newton force mass acceleration", "Machine learning algorithms learn patterns"),
    ]
    demos = []
    contrasts = []
    print("\n=== Sentence similarity: embedding cos vs Jaccard ===")
    for a, b in pairs:
        ea = cosine(sent_vec(tokenize(a), v2i, emb), sent_vec(tokenize(b), v2i, emb))
        jac = jaccard(a, b)
        contrasts.append({"a": a, "b": b, "emb_cos": ea, "jaccard": jac})
        demos.append((f"A={a!r} || B={b!r}", f"emb_cos={ea:.3f}  jaccard={jac:.3f}"))
        print(f"  emb={ea:.3f} jac={jac:.3f} | {a[:36]!r} vs {b[:36]!r}")

    # Expect related pair (plants/photosynthesis) > unrelated (movie/sports)
    rel = contrasts[3]["emb_cos"]
    unrel = contrasts[1]["emb_cos"]
    print(f"\nRelated(plants↔photo)={rel:.3f}  Unrelated(movie↔sports)={unrel:.3f}")

    save_result(
        "S02_embeddings_similarity",
        {
            "concept": "PPMI + SVD dense embeddings (SGNS approximation)",
            "vocab_size": len(vocab),
            "dim": int(emb.shape[1]),
            "word_nn": word_nn,
            "sentence_pairs": contrasts,
            "demos": [{"input": a, "output": b} for a, b in demos],
            "new_capability": "Dense feature extraction & soft semantic similarity",
            "previous": "S01 sparse TF-IDF",
        },
    )
    print("\n[S02 DONE] New capability: dense features + sentence similarity.")


if __name__ == "__main__":
    main()
