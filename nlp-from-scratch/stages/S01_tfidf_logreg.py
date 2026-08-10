#!/usr/bin/env python3
"""S01 — TF-IDF features + Logistic Regression (from scratch NumPy).

Concept: rare-but-discriminative words get higher weight than raw counts.
Compare: should beat or match S00 NB, and expose linear feature importances.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import CLS_TEST, CLS_TRAIN
from common.report import print_io, save_result
from stages.S00_bow_naive_bayes import BowNaiveBayes


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class TfidfVectorizer:
    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        docs = [tokenize(t) for t in texts]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.vocab = sorted(df.keys())
        self.v2i = {w: i for i, w in enumerate(self.vocab)}
        n = len(texts)
        self.idf = np.array([math.log((n + 1) / (df[w] + 1)) + 1.0 for w in self.vocab])
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for i, t in enumerate(texts):
            counts = Counter(tokenize(t))
            if not counts:
                continue
            for w, c in counts.items():
                j = self.v2i.get(w)
                if j is not None:
                    X[i, j] = c
            row = X[i] * self.idf
            nrm = np.linalg.norm(row)
            X[i] = row / nrm if nrm > 0 else row
        return X


class SoftmaxRegression:
    def __init__(self, lr=0.5, epochs=400, l2=1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X: np.ndarray, y: list[str]) -> "SoftmaxRegression":
        self.classes = sorted(set(y))
        c2i = {c: i for i, c in enumerate(self.classes)}
        Y = np.zeros((len(y), len(self.classes)))
        for i, lab in enumerate(y):
            Y[i, c2i[lab]] = 1.0
        n, d = X.shape
        k = len(self.classes)
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        for _ in range(self.epochs):
            logits = X @ self.W + self.b
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            P = exp / exp.sum(axis=1, keepdims=True)
            grad = (P - Y) / n
            self.W -= self.lr * (X.T @ grad + self.l2 * self.W)
            self.b -= self.lr * grad.sum(axis=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> list[str]:
        P = self.predict_proba(X)
        idx = P.argmax(axis=1)
        return [self.classes[i] for i in idx]

    def top_features(self, vocab: list[str], class_name: str, k: int = 5) -> list[tuple[str, float]]:
        j = self.classes.index(class_name)
        w = self.W[:, j]
        order = np.argsort(-w)[:k]
        return [(vocab[i], float(w[i])) for i in order]


def main() -> None:
    print("=" * 60)
    print("S01 · TF-IDF + Softmax Logistic Regression")
    print("Concept: weight words by TF×IDF, learn linear decision boundaries")
    print("=" * 60)

    Xtr_txt = [t for t, _ in CLS_TRAIN]
    ytr = [y for _, y in CLS_TRAIN]
    Xte_txt = [t for t, _ in CLS_TEST]
    yte = [y for _, y in CLS_TEST]

    vec = TfidfVectorizer().fit(Xtr_txt)
    Xtr = vec.transform(Xtr_txt)
    Xte = vec.transform(Xte_txt)
    clf = SoftmaxRegression().fit(Xtr, ytr)

    tr_pred = clf.predict(Xtr)
    te_pred = clf.predict(Xte)
    train_acc = sum(a == b for a, b in zip(tr_pred, ytr)) / len(ytr)
    test_acc = sum(a == b for a, b in zip(te_pred, yte)) / len(yte)

    nb = BowNaiveBayes()
    nb.fit(Xtr_txt, ytr)
    nb_test = sum(1 for x, y in CLS_TEST if nb.predict(x) == y) / len(CLS_TEST)

    print(f"TFIDF dim={Xtr.shape[1]} train_acc={train_acc:.3f} test_acc={test_acc:.3f}")
    print(f"S00 NB test_acc={nb_test:.3f}  →  Δ(test)={test_acc - nb_test:+.3f}")

    demos = []
    for text, gold in CLS_TEST:
        x = vec.transform([text])
        pred = clf.predict(x)[0]
        p = clf.predict_proba(x)[0]
        top = ", ".join(f"{c}:{p[i]:.2f}" for i, c in enumerate(clf.classes))
        demos.append((f"{text!r} [gold={gold}]", f"{pred} ({top})"))
    print_io("Live predictions", demos)

    print("=== Top linear features (what the model 'looks at') ===")
    feat_map = {}
    for c in clf.classes:
        tops = clf.top_features(vec.vocab, c, 5)
        feat_map[c] = tops
        print(f"  {c}: {tops}")

    save_result(
        "S01_tfidf_logreg",
        {
            "concept": "TF-IDF weighted bag features + multinomial logistic regression",
            "metric": {"train_acc": train_acc, "test_acc": test_acc, "s00_nb_test_acc": nb_test},
            "delta_vs_s00": test_acc - nb_test,
            "top_features": feat_map,
            "demos": [{"input": a, "output": b} for a, b in demos],
            "new_capability": "Interpretable linear weights; rare terms up-weighted via IDF",
            "previous": "S00",
        },
    )
    print("\n[S01 DONE] New capability: TF-IDF + linear model with feature importances.")


if __name__ == "__main__":
    main()
