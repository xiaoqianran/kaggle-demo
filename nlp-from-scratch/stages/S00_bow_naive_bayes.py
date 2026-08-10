#!/usr/bin/env python3
"""S00 — Bag-of-Words + Multinomial Naive Bayes (from scratch).

Concept: classify documents by word counts under class-conditional multinomials.
Solves: first workable text classification without neural nets.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import CLS_TEST, CLS_TRAIN
from common.report import print_io, save_result


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class BowNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes: list[str] = []
        self.class_logprior: dict[str, float] = {}
        self.class_loglik: dict[str, dict[str, float]] = {}
        self.vocab: set[str] = set()
        self.word_counts: dict[str, Counter] = {}
        self.class_total_tokens: dict[str, int] = {}

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.classes = sorted(set(labels))
        n = len(labels)
        self.word_counts = {c: Counter() for c in self.classes}
        class_docs = Counter(labels)
        for text, y in zip(texts, labels):
            toks = tokenize(text)
            self.word_counts[y].update(toks)
            self.vocab.update(toks)
        V = len(self.vocab)
        self.class_total_tokens = {c: sum(self.word_counts[c].values()) for c in self.classes}
        self.class_logprior = {c: math.log(class_docs[c] / n) for c in self.classes}
        self.class_loglik = {}
        for c in self.classes:
            total = self.class_total_tokens[c]
            self.class_loglik[c] = {
                w: math.log((self.word_counts[c][w] + self.alpha) / (total + self.alpha * V))
                for w in self.vocab
            }
            # OOV fallback
            self.class_loglik[c]["__OOV__"] = math.log(self.alpha / (total + self.alpha * V))

    def predict_proba(self, text: str) -> dict[str, float]:
        toks = tokenize(text)
        scores = {}
        for c in self.classes:
            s = self.class_logprior[c]
            ll = self.class_loglik[c]
            for t in toks:
                s += ll.get(t, ll["__OOV__"])
            scores[c] = s
        m = max(scores.values())
        exps = {c: math.exp(v - m) for c, v in scores.items()}
        z = sum(exps.values())
        return {c: exps[c] / z for c in self.classes}

    def predict(self, text: str) -> str:
        p = self.predict_proba(text)
        return max(p, key=p.get)


def accuracy(model, pairs) -> float:
    ok = sum(1 for x, y in pairs if model.predict(x) == y)
    return ok / len(pairs)


def main() -> None:
    print("=" * 60)
    print("S00 · Bag-of-Words + Naive Bayes")
    print("Concept: P(class|words) ∝ P(class) Π P(word|class) with add-α smoothing")
    print("=" * 60)

    Xtr = [t for t, _ in CLS_TRAIN]
    ytr = [y for _, y in CLS_TRAIN]
    model = BowNaiveBayes(alpha=1.0)
    model.fit(Xtr, ytr)

    train_acc = accuracy(model, CLS_TRAIN)
    test_acc = accuracy(model, CLS_TEST)
    print(f"vocab_size={len(model.vocab)} classes={model.classes}")
    print(f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    demos = []
    for text, gold in CLS_TEST:
        pred = model.predict(text)
        proba = model.predict_proba(text)
        top = ", ".join(f"{c}:{proba[c]:.2f}" for c in sorted(proba, key=proba.get, reverse=True)[:3])
        demos.append((f"{text!r}  [gold={gold}]", f"{pred}  ({top})"))
    print_io("Live predictions on held-out text", demos)

    # Ablation: remove a key word → see probability shift
    base = "I love this amazing movie"
    ablated = "I this movie"
    print("\n=== Contrast: keyword ablation ===")
    print(f"  '{base}' → {model.predict(base)} {model.predict_proba(base)}")
    print(f"  '{ablated}' → {model.predict(ablated)} {model.predict_proba(ablated)}")

    save_result(
        "S00_bow_naive_bayes",
        {
            "concept": "Bag-of-words multinomial Naive Bayes",
            "metric": {"train_acc": train_acc, "test_acc": test_acc},
            "vocab_size": len(model.vocab),
            "classes": model.classes,
            "demos": [{"input": a, "output": b} for a, b in demos],
            "new_capability": "First end-to-end text classifier from raw strings",
            "previous": None,
        },
    )
    print("\n[S00 DONE] New capability: classify short documents by word counts.")


if __name__ == "__main__":
    main()
