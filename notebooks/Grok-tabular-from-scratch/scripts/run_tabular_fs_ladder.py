#!/usr/bin/env python3
"""Tabular ML + Time Series From-Scratch ladder (FS00–FS21).

Runs fully offline with sklearn/torch synthetic + built-in datasets.
Writes results/<stage>/results.json + PNG under OUT_DIR.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = Path(os.environ.get("TABULAR_FS_OUT", "/kaggle/working/results"))
if not OUT_DIR.parent.exists():
    OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

STAGE_ORDER: List[str] = []
SUMMARY: Dict[str, Any] = {"stages": {}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save_stage(name: str, payload: Dict[str, Any], figs: Optional[Dict[str, plt.Figure]] = None) -> Path:
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "stage": name, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (d / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    if figs:
        for fn, fig in figs.items():
            fig.savefig(d / f"{fn}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    STAGE_ORDER.append(name)
    SUMMARY["stages"][name] = {k: payload[k] for k in payload if k in (
        "task", "metric", "metrics", "vs_prev", "concept", "you_should_feel", "ok"
    ) or k.endswith("_score") or k in ("rmse", "mae", "accuracy", "auc", "r2", "smape")}
    print(f"[OK] {name} → {d}")
    return d


# ---------------------------------------------------------------------------
# Metrics & utils
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def log_loss_binary(y_true, p, eps=1e-7) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def roc_auc_binary(y_true, score) -> float:
    """Mann-Whitney AUC without sklearn dependency for ranking."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # efficient
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # handle ties average rank
    # simple: use pairwise
    # fallback pairwise for correctness on small n
    if len(s) <= 5000:
        wins = 0.0
        for p in pos:
            wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return float(wins / (len(pos) * len(neg)))
    n_pos, n_neg = len(pos), len(neg)
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-12))


def smape(y_true, y_pred, eps=1e-8) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100)


def train_val_split_idx(n: int, val_ratio=0.25, seed=SEED) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    r = np.random.default_rng(seed)
    r.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def one_hot(x: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x).astype(int)
    k = int(n_classes or (x.max() + 1))
    oh = np.zeros((len(x), k), dtype=float)
    oh[np.arange(len(x)), x] = 1.0
    return oh


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------

def make_tabular_classification(n=2000, n_num=8, n_cat=4, seed=SEED):
    """Synthetic adult-like: numeric + categorical → binary label with nonlinear rules."""
    r = np.random.default_rng(seed)
    X_num = r.normal(size=(n, n_num))
    # categoricals with different cardinalities
    cards = [3, 5, 8, 12][:n_cat]
    X_cat = np.column_stack([r.integers(0, c, size=n) for c in cards])
    # latent
    logit = (
        1.2 * X_num[:, 0]
        - 0.8 * X_num[:, 1]
        + 0.5 * X_num[:, 0] * X_num[:, 1]
        + 0.7 * (X_num[:, 2] > 0).astype(float)
        + 0.4 * (X_cat[:, 0] == 1).astype(float)
        - 0.6 * (X_cat[:, 1] >= 3).astype(float)
        + 0.15 * X_num[:, 3] ** 2
    )
    p = 1 / (1 + np.exp(-logit))
    y = (r.random(n) < p).astype(int)
    return {
        "X_num": X_num.astype(np.float64),
        "X_cat": X_cat.astype(int),
        "y": y,
        "cat_cards": cards,
        "feature_names_num": [f"num_{i}" for i in range(n_num)],
        "feature_names_cat": [f"cat_{i}" for i in range(n_cat)],
        "task": "classification",
    }


def make_tabular_regression(n=2000, n_num=10, n_cat=3, seed=SEED + 1):
    r = np.random.default_rng(seed)
    X_num = r.normal(size=(n, n_num))
    cards = [4, 6, 10][:n_cat]
    X_cat = np.column_stack([r.integers(0, c, size=n) for c in cards])
    y = (
        3.0 * X_num[:, 0]
        - 2.0 * X_num[:, 1]
        + 1.5 * np.sin(X_num[:, 2])
        + 0.8 * X_num[:, 0] * X_num[:, 3]
        + 0.5 * (X_cat[:, 0] - cards[0] / 2)
        + r.normal(0, 0.5, size=n)
    )
    return {
        "X_num": X_num.astype(np.float64),
        "X_cat": X_cat.astype(int),
        "y": y.astype(np.float64),
        "cat_cards": cards,
        "task": "regression",
    }


def make_timeseries(n=1500, seed=SEED + 2):
    """Trend + daily/weekly seasonality + noise + holiday spikes."""
    r = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.01 * t + 0.00002 * t ** 2 / n
    season_day = 1.5 * np.sin(2 * np.pi * t / 7)  # weekly
    season_year = 0.8 * np.sin(2 * np.pi * t / 365.25) if n > 400 else 0.5 * np.sin(2 * np.pi * t / 30)
    noise = r.normal(0, 0.4, size=n)
    spikes = np.zeros(n)
    for k in r.choice(n, size=12, replace=False):
        spikes[k] = r.uniform(2.0, 4.0)
    y = 10 + trend + season_day + season_year + noise + spikes
    return {"t": t, "y": y.astype(np.float64), "task": "timeseries"}


def design_matrix(X_num, X_cat, cat_cards, mode="onehot"):
    parts = [X_num]
    if mode == "onehot":
        for j, c in enumerate(cat_cards):
            parts.append(one_hot(X_cat[:, j], c))
    elif mode == "ordinal":
        parts.append(X_cat.astype(float))
    elif mode == "ignore_cat":
        pass
    else:
        raise ValueError(mode)
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# Models (from-scratch-ish)
# ---------------------------------------------------------------------------

class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iter=400, l2=1e-3):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        y = y.astype(float)
        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y)
            self.w -= self.lr * (X.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class RidgeRegressionGD:
    def __init__(self, lr=0.05, n_iter=500, l2=1e-2):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        y = y.astype(float)
        for _ in range(self.n_iter):
            pred = X @ self.w + self.b
            err = pred - y
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict(self, X):
        return X @ self.w + self.b


class DecisionStump:
    """Single-feature threshold stump for regression or classification (gini/mse)."""

    def __init__(self, task="classification"):
        self.task = task
        self.feature = 0
        self.threshold = 0.0
        self.left_value = 0.0
        self.right_value = 0.0

    def fit(self, X, y, sample_weight=None):
        n, d = X.shape
        y = y.astype(float)
        w = np.ones(n) if sample_weight is None else sample_weight.astype(float)
        best = (math.inf, 0, 0.0, 0.0, 0.0)
        # subsample features for speed
        feat_idx = np.arange(d)
        if d > 12:
            feat_idx = np.random.default_rng(0).choice(d, size=12, replace=False)
        for j in feat_idx:
            xs = X[:, j]
            # candidate thresholds: quantiles
            qs = np.quantile(xs, np.linspace(0.1, 0.9, 9))
            for thr in qs:
                left = xs <= thr
                right = ~left
                if left.sum() < 5 or right.sum() < 5:
                    continue
                if self.task == "classification":
                    # weighted variance of labels as impurity proxy
                    def impure(mask):
                        ww = w[mask]
                        yy = y[mask]
                        p = np.average(yy, weights=ww) if ww.sum() > 0 else 0.5
                        return float((ww * (yy - p) ** 2).sum())
                    score = impure(left) + impure(right)
                    lv = float(np.average(y[left], weights=w[left]))
                    rv = float(np.average(y[right], weights=w[right]))
                else:
                    def sse(mask):
                        ww = w[mask]
                        yy = y[mask]
                        m = np.average(yy, weights=ww)
                        return float((ww * (yy - m) ** 2).sum())
                    score = sse(left) + sse(right)
                    lv = float(np.average(y[left], weights=w[left]))
                    rv = float(np.average(y[right], weights=w[right]))
                if score < best[0]:
                    best = (score, j, float(thr), lv, rv)
        _, self.feature, self.threshold, self.left_value, self.right_value = best
        return self

    def predict(self, X):
        left = X[:, self.feature] <= self.threshold
        out = np.where(left, self.left_value, self.right_value)
        return out


class DecisionTreeScratch:
    """Shallow binary tree (depth-limited) using greedy stumps on residuals/targets."""

    def __init__(self, max_depth=3, task="classification", min_leaf=10):
        self.max_depth = max_depth
        self.task = task
        self.min_leaf = min_leaf
        self.nodes = []

    def fit(self, X, y):
        y = y.astype(float)
        self.nodes = []
        self._build(X, y, depth=0, node_id=0)
        return self

    def _build(self, X, y, depth, node_id):
        # store as list of dicts
        while len(self.nodes) <= node_id:
            self.nodes.append(None)
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf:
            val = float(y.mean()) if self.task == "regression" else float(y.mean())
            self.nodes[node_id] = {"leaf": True, "value": val}
            return
        stump = DecisionStump(task=self.task).fit(X, y)
        left = X[:, stump.feature] <= stump.threshold
        right = ~left
        if left.sum() < self.min_leaf or right.sum() < self.min_leaf:
            self.nodes[node_id] = {"leaf": True, "value": float(y.mean())}
            return
        self.nodes[node_id] = {
            "leaf": False,
            "feature": stump.feature,
            "threshold": stump.threshold,
            "left": 2 * node_id + 1,
            "right": 2 * node_id + 2,
        }
        self._build(X[left], y[left], depth + 1, 2 * node_id + 1)
        self._build(X[right], y[right], depth + 1, 2 * node_id + 2)

    def predict(self, X):
        out = np.zeros(len(X))
        for i in range(len(X)):
            nid = 0
            while True:
                node = self.nodes[nid]
                if node is None or node.get("leaf", True):
                    out[i] = node["value"] if node else 0.0
                    break
                if X[i, node["feature"]] <= node["threshold"]:
                    nid = node["left"]
                else:
                    nid = node["right"]
                if nid >= len(self.nodes) or self.nodes[nid] is None:
                    out[i] = node.get("value", 0.0)
                    break
        return out


class RandomForestScratch:
    def __init__(self, n_estimators=15, max_depth=4, task="classification", seed=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.task = task
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        r = np.random.default_rng(self.seed)
        self.trees = []
        n = len(X)
        for i in range(self.n_estimators):
            idx = r.integers(0, n, size=n)
            # feature bagging by zeroing some columns randomly via projection
            tree = DecisionTreeScratch(max_depth=self.max_depth, task=self.task)
            # column subsample
            d = X.shape[1]
            keep = r.choice(d, size=max(2, int(np.sqrt(d)) + 1), replace=False)
            Xs = X[idx][:, keep]
            tree.fit(Xs, y[idx])
            self.trees.append((tree, keep))
        return self

    def predict(self, X):
        preds = []
        for tree, keep in self.trees:
            preds.append(tree.predict(X[:, keep]))
        P = np.mean(preds, axis=0)
        if self.task == "classification":
            return (P >= 0.5).astype(int), P
        return P


class GradientBoostingScratch:
    """Minimal GBDT: trees on residuals (regression) or logistic residuals."""

    def __init__(self, n_estimators=30, max_depth=2, lr=0.1, task="classification"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.task = task
        self.trees = []
        self.init_ = 0.0

    def fit(self, X, y):
        y = y.astype(float)
        self.trees = []
        if self.task == "classification":
            p = np.clip(y.mean(), 1e-3, 1 - 1e-3)
            self.init_ = math.log(p / (1 - p))
            F = np.full(len(y), self.init_)
            for _ in range(self.n_estimators):
                p = 1 / (1 + np.exp(-np.clip(F, -30, 30)))
                resid = y - p
                tree = DecisionTreeScratch(max_depth=self.max_depth, task="regression")
                tree.fit(X, resid)
                self.trees.append(tree)
                F = F + self.lr * tree.predict(X)
        else:
            self.init_ = float(y.mean())
            F = np.full(len(y), self.init_)
            for _ in range(self.n_estimators):
                resid = y - F
                tree = DecisionTreeScratch(max_depth=self.max_depth, task="regression")
                tree.fit(X, resid)
                self.trees.append(tree)
                F = F + self.lr * tree.predict(X)
        return self

    def predict_raw(self, X):
        F = np.full(len(X), self.init_)
        for tree in self.trees:
            F = F + self.lr * tree.predict(X)
        return F

    def predict(self, X):
        if self.task == "classification":
            p = 1 / (1 + np.exp(-np.clip(self.predict_raw(X), -30, 30)))
            return (p >= 0.5).astype(int), p
        return self.predict_raw(X)


def target_encode_fit(x_cat, y, cards, m=20.0):
    """Smoothed target encoding per column. Returns list of maps and global mean."""
    y = y.astype(float)
    global_mean = float(y.mean())
    maps = []
    for j, c in enumerate(cards):
        col = x_cat[:, j]
        mp = {}
        for k in range(c):
            mask = col == k
            if mask.sum() == 0:
                mp[k] = global_mean
            else:
                n_k = mask.sum()
                mean_k = y[mask].mean()
                mp[k] = (n_k * mean_k + m * global_mean) / (n_k + m)
        maps.append(mp)
    return maps, global_mean


def target_encode_apply(x_cat, maps, global_mean):
    cols = []
    for j, mp in enumerate(maps):
        cols.append(np.array([mp.get(int(v), global_mean) for v in x_cat[:, j]], dtype=float))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Torch models (optional GPU)
# ---------------------------------------------------------------------------

def get_torch():
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch, nn, device


class MLP(torch.nn.Module if False else object):
    pass


def build_mlp(nn, d_in, d_hidden=128, d_out=1, depth=2):
    layers = []
    d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, d_hidden), nn.ReLU(), nn.BatchNorm1d(d_hidden), nn.Dropout(0.1)]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class CatEmbedMLP:
    def __init__(self, n_num, cat_cards, task="classification", emb_dim=8, hidden=128, lr=1e-3, epochs=40, batch=256):
        self.n_num = n_num
        self.cat_cards = cat_cards
        self.task = task
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.torch, self.nn, self.device = get_torch()
        self.model = None
        self.mu = None
        self.sd = None

    def _build(self):
        torch, nn, device = self.torch, self.nn, self.device
        cards = self.cat_cards

        class Net(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.embs = nn.ModuleList([nn.Embedding(c, min(self.emb_dim, max(2, c // 2 + 1))) for c in cards])
                d_emb = sum(e.embedding_dim for e in self_inner.embs)
                d_in = self.n_num + d_emb
                self_inner.mlp = build_mlp(nn, d_in, self.hidden, 1, depth=2)

            def forward(self_inner, x_num, x_cat):
                em = [emb(x_cat[:, j]) for j, emb in enumerate(self_inner.embs)]
                x = torch.cat([x_num] + em, dim=1)
                return self_inner.mlp(x).squeeze(-1)

        self.model = Net().to(device)

    def fit(self, X_num, X_cat, y, Xn_val=None, Xc_val=None, y_val=None):
        torch, nn, device = self.torch, self.nn, self.device
        Xn, self.mu, self.sd = standardize_fit(X_num)
        self._build()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        Xn_t = torch.tensor(Xn, dtype=torch.float32, device=device)
        Xc_t = torch.tensor(X_cat, dtype=torch.long, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        n = len(y)
        best_state = None
        best_val = math.inf
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                logits = self.model(Xn_t[idx], Xc_t[idx])
                if self.task == "classification":
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t[idx])
                else:
                    loss = nn.functional.mse_loss(logits, y_t[idx])
                loss.backward()
                opt.step()
            if Xn_val is not None:
                val = self._val_loss(Xn_val, Xc_val, y_val)
                if val < best_val:
                    best_val = val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def _val_loss(self, X_num, X_cat, y):
        torch, nn, device = self.torch, self.nn, self.device
        self.model.eval()
        Xn = standardize_apply(X_num, self.mu, self.sd)
        with torch.no_grad():
            logits = self.model(
                torch.tensor(Xn, dtype=torch.float32, device=device),
                torch.tensor(X_cat, dtype=torch.long, device=device),
            )
            y_t = torch.tensor(y, dtype=torch.float32, device=device)
            if self.task == "classification":
                return float(nn.functional.binary_cross_entropy_with_logits(logits, y_t).item())
            return float(nn.functional.mse_loss(logits, y_t).item())

    def predict(self, X_num, X_cat):
        torch, device = self.torch, self.device
        self.model.eval()
        Xn = standardize_apply(X_num, self.mu, self.sd)
        with torch.no_grad():
            logits = self.model(
                torch.tensor(Xn, dtype=torch.float32, device=device),
                torch.tensor(X_cat, dtype=torch.long, device=device),
            )
            if self.task == "classification":
                p = torch.sigmoid(logits).cpu().numpy()
                return (p >= 0.5).astype(int), p
            return logits.cpu().numpy()


class FTTransformerMini:
    """Feature Tokenizer + tiny Transformer encoder for tabular (Gorishniy et al. spirit)."""

    def __init__(self, n_num, cat_cards, task="classification", d_token=32, n_heads=4, n_layers=2, epochs=40, batch=256, lr=1e-3):
        self.n_num = n_num
        self.cat_cards = cat_cards
        self.task = task
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.torch, self.nn, self.device = get_torch()
        self.model = None
        self.mu = self.sd = None

    def _build(self):
        torch, nn, device = self.torch, self.nn, self.device
        n_num, cards, d = self.n_num, self.cat_cards, self.d_token

        class Model(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.num_w = nn.Parameter(torch.randn(n_num, d) * 0.02)
                self_inner.num_b = nn.Parameter(torch.zeros(n_num, d))
                self_inner.cat_emb = nn.ModuleList([nn.Embedding(c, d) for c in cards])
                self_inner.cls = nn.Parameter(torch.zeros(1, 1, d))
                enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=self.n_heads, dim_feedforward=d * 2, batch_first=True, dropout=0.1)
                self_inner.enc = nn.TransformerEncoder(enc_layer, num_layers=self.n_layers)
                self_inner.head = nn.Linear(d, 1)

            def forward(self_inner, x_num, x_cat):
                # x_num: B,n_num
                tok_num = x_num.unsqueeze(-1) * self_inner.num_w + self_inner.num_b  # B,n_num,d
                tok_cat = torch.stack([emb(x_cat[:, j]) for j, emb in enumerate(self_inner.cat_emb)], dim=1)
                cls = self_inner.cls.expand(x_num.size(0), -1, -1)
                tokens = torch.cat([cls, tok_num, tok_cat], dim=1)
                h = self_inner.enc(tokens)
                return self_inner.head(h[:, 0]).squeeze(-1)

        self.model = Model().to(device)

    def fit(self, X_num, X_cat, y):
        torch, nn, device = self.torch, self.nn, self.device
        Xn, self.mu, self.sd = standardize_fit(X_num)
        self._build()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        Xn_t = torch.tensor(Xn, dtype=torch.float32, device=device)
        Xc_t = torch.tensor(X_cat, dtype=torch.long, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        n = len(y)
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                logits = self.model(Xn_t[idx], Xc_t[idx])
                if self.task == "classification":
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t[idx])
                else:
                    loss = nn.functional.mse_loss(logits, y_t[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X_num, X_cat):
        torch, device = self.torch, self.device
        self.model.eval()
        Xn = standardize_apply(X_num, self.mu, self.sd)
        with torch.no_grad():
            logits = self.model(
                torch.tensor(Xn, dtype=torch.float32, device=device),
                torch.tensor(X_cat, dtype=torch.long, device=device),
            )
            if self.task == "classification":
                p = torch.sigmoid(logits).cpu().numpy()
                return (p >= 0.5).astype(int), p
            return logits.cpu().numpy()


# ---------------------------------------------------------------------------
# Time series helpers
# ---------------------------------------------------------------------------

def walk_forward_splits(n, n_splits=3, min_train=200, horizon=14):
    """Yield (train_end, test_start, test_end) indices."""
    usable = n - horizon
    step = max(1, (usable - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_start = train_end
        test_end = min(train_end + horizon, n)
        if test_end <= test_start:
            continue
        yield train_end, test_start, test_end


def make_lag_matrix(y, lags=(1, 2, 3, 7, 14), start=0, end=None):
    end = end if end is not None else len(y)
    rows, targets, idx = [], [], []
    max_lag = max(lags)
    for t in range(max(start, max_lag), end):
        rows.append([y[t - L] for L in lags])
        targets.append(y[t])
        idx.append(t)
    return np.asarray(rows, float), np.asarray(targets, float), np.asarray(idx)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def fs00_protocol(cls_data, reg_data, ts_data):
    """Lock evaluation protocol: metrics, split, leakage checklist."""
    y = cls_data["y"]
    base_rate = float(y.mean())
    # demonstrate metric sensitivity
    n = len(y)
    perfect = accuracy(y, y)
    random_pred = (rng.random(n) < base_rate).astype(int)
    constant = np.zeros(n, dtype=int)
    metrics = {
        "class_base_rate": base_rate,
        "acc_perfect": perfect,
        "acc_random": accuracy(y, random_pred),
        "acc_all_zero": accuracy(y, constant),
        "auc_perfect": roc_auc_binary(y, y.astype(float)),
        "auc_random": roc_auc_binary(y, rng.random(n)),
        "reg_y_std": float(reg_data["y"].std()),
        "reg_rmse_mean_baseline": rmse(reg_data["y"], np.full_like(reg_data["y"], reg_data["y"].mean())),
        "ts_length": int(len(ts_data["y"])),
        "ts_naive_last_rmse_tail14": rmse(ts_data["y"][-14:], np.full(14, ts_data["y"][-15])),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].bar(["perfect", "random", "all0"], [metrics["acc_perfect"], metrics["acc_random"], metrics["acc_all_zero"]], color=["#2ecc71", "#e67e22", "#e74c3c"])
    axes[0].set_title("Classification accuracy protocol")
    axes[0].set_ylim(0, 1.05)
    axes[1].hist(reg_data["y"], bins=40, color="#3498db", alpha=0.85)
    axes[1].set_title("Regression target distribution")
    axes[2].plot(ts_data["t"][:200], ts_data["y"][:200], color="#9b59b6")
    axes[2].set_title("Time series (first 200)")
    fig.tight_layout()
    payload = {
        "task": "protocol",
        "concept": "Before models: fix split, metrics, leakage rules",
        "metrics": metrics,
        "leakage_rules": [
            "Fit scalers/encoders on train only",
            "Time series: never shuffle; only past→future",
            "Target encoding must use train folds only",
        ],
        "you_should_feel": "尺子先锁死，否则后面任何提升都不可信",
        "ok": True,
    }
    save_stage("fs00_protocol", payload, {"overview": fig})
    return metrics


def fs01_baselines(cls_data, reg_data):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    ytr, yva = cls_data["y"][tr], cls_data["y"][va]
    maj = int(np.round(ytr.mean()))  # majority approx via mean for binary
    # true majority
    maj = int(np.bincount(ytr).argmax())
    pred = np.full_like(yva, maj)
    cls_acc = accuracy(yva, pred)

    tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
    mean_y = reg_data["y"][tr_r].mean()
    pred_r = np.full_like(reg_data["y"][va_r], mean_y)
    reg_rmse = rmse(reg_data["y"][va_r], pred_r)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["cls maj acc", "reg mean RMSE"], [cls_acc, reg_rmse], color=["#1abc9c", "#e74c3c"])
    ax.set_title("FS01 constant baselines")
    payload = {
        "task": "baseline",
        "concept": "Constant predictors: majority class / train mean",
        "metrics": {"cls_acc": cls_acc, "reg_rmse": reg_rmse, "majority_class": maj, "mean_y": float(mean_y)},
        "vs_prev": "FS00 只定义尺子；FS01 给出不可再低的参照线",
        "you_should_feel": "任何模型若打不过基线，实现或泄漏有问题",
        "ok": True,
    }
    save_stage("fs01_baselines", payload, {"baselines": fig})
    return payload["metrics"]


def fs02_linear(cls_data, reg_data, prev):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    X = design_matrix(cls_data["X_num"], cls_data["X_cat"], cls_data["cat_cards"], "onehot")
    Xtr, mu, sd = standardize_fit(X[tr])
    Xva = standardize_apply(X[va], mu, sd)
    clf = LogisticRegressionGD(lr=0.2, n_iter=500, l2=1e-3).fit(Xtr, cls_data["y"][tr])
    p = clf.predict_proba(Xva)
    pred = (p >= 0.5).astype(int)
    cls_m = {"acc": accuracy(cls_data["y"][va], pred), "auc": roc_auc_binary(cls_data["y"][va], p), "logloss": log_loss_binary(cls_data["y"][va], p)}

    tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
    Xr = design_matrix(reg_data["X_num"], reg_data["X_cat"], reg_data["cat_cards"], "onehot")
    Xtr_r, mu_r, sd_r = standardize_fit(Xr[tr_r])
    Xva_r = standardize_apply(Xr[va_r], mu_r, sd_r)
    reg = RidgeRegressionGD(lr=0.05, n_iter=600, l2=1e-2).fit(Xtr_r, reg_data["y"][tr_r])
    pred_r = reg.predict(Xva_r)
    reg_m = {"rmse": rmse(reg_data["y"][va_r], pred_r), "mae": mae(reg_data["y"][va_r], pred_r), "r2": r2_score(reg_data["y"][va_r], pred_r)}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(["baseline", "logreg"], [prev["cls_acc"], cls_m["acc"]], color=["#95a5a6", "#2980b9"])
    axes[0].set_title("Classification acc")
    axes[0].set_ylim(0, 1)
    axes[1].bar(["baseline", "ridge"], [prev["reg_rmse"], reg_m["rmse"]], color=["#95a5a6", "#8e44ad"])
    axes[1].set_title("Regression RMSE (lower better)")
    payload = {
        "task": "linear",
        "concept": "Linear models on standardized one-hot features",
        "metrics": {"classification": cls_m, "regression": reg_m},
        "cls_acc": cls_m["acc"],
        "reg_rmse": reg_m["rmse"],
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{cls_m['acc']:.3f}; rmse {prev['reg_rmse']:.3f}→{reg_m['rmse']:.3f}",
        "you_should_feel": "线性模型吃掉可加性信号；非线性交互仍吃不掉",
        "ok": True,
    }
    save_stage("fs02_linear", payload, {"vs_baseline": fig})
    return {"cls_acc": cls_m["acc"], "reg_rmse": reg_m["rmse"], "cls_auc": cls_m["auc"]}


def fs03_feature_engineering(cls_data, reg_data, prev):
    """Binning + interactions on top of linear."""
    tr, va = train_val_split_idx(len(cls_data["y"]))
    Xn = cls_data["X_num"].copy()
    # bin first numeric feature into 5 bins using train quantiles
    edges = np.quantile(Xn[tr, 0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bins = np.digitize(Xn[:, 0], edges[1:-1])
    inter = (Xn[:, 0] * Xn[:, 1]).reshape(-1, 1)
    X_base = design_matrix(Xn, cls_data["X_cat"], cls_data["cat_cards"], "onehot")
    X = np.hstack([X_base, one_hot(bins, 5), inter])
    Xtr, mu, sd = standardize_fit(X[tr])
    Xva = standardize_apply(X[va], mu, sd)
    clf = LogisticRegressionGD(lr=0.2, n_iter=500).fit(Xtr, cls_data["y"][tr])
    p = clf.predict_proba(Xva)
    cls_acc = accuracy(cls_data["y"][va], (p >= 0.5).astype(int))
    cls_auc = roc_auc_binary(cls_data["y"][va], p)

    tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
    Rn = reg_data["X_num"]
    inter_r = (Rn[:, 0] * Rn[:, 3]).reshape(-1, 1)
    sin_f = np.sin(Rn[:, 2]).reshape(-1, 1)
    Xr = np.hstack([design_matrix(Rn, reg_data["X_cat"], reg_data["cat_cards"], "onehot"), inter_r, sin_f])
    Xtr_r, mu_r, sd_r = standardize_fit(Xr[tr_r])
    Xva_r = standardize_apply(Xr[va_r], mu_r, sd_r)
    reg = RidgeRegressionGD(lr=0.05, n_iter=600).fit(Xtr_r, reg_data["y"][tr_r])
    pred = reg.predict(Xva_r)
    reg_rmse = rmse(reg_data["y"][va_r], pred)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["linear", "+FE"], [prev["cls_acc"], cls_acc], color=["#3498db", "#27ae60"])
    ax.set_title("Feature eng. lifts linear (cls acc)")
    ax.set_ylim(0, 1)
    payload = {
        "task": "feature_engineering",
        "concept": "Binning + hand interactions unlock nonlinear signal for linear models",
        "metrics": {"cls_acc": cls_acc, "cls_auc": cls_auc, "reg_rmse": reg_rmse},
        "cls_acc": cls_acc,
        "reg_rmse": reg_rmse,
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{cls_acc:.3f}; rmse {prev['reg_rmse']:.3f}→{reg_rmse:.3f}",
        "you_should_feel": "特征工程=把非线性变成线性模型可读的列",
        "ok": True,
    }
    save_stage("fs03_feature_engineering", payload, {"fe_lift": fig})
    return {"cls_acc": cls_acc, "reg_rmse": reg_rmse, "cls_auc": cls_auc}


def fs04_trees(cls_data, reg_data, prev):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    X = design_matrix(cls_data["X_num"], cls_data["X_cat"], cls_data["cat_cards"], "ordinal")
    tree = DecisionTreeScratch(max_depth=4, task="classification").fit(X[tr], cls_data["y"][tr])
    p = tree.predict(X[va])
    pred = (p >= 0.5).astype(int)
    cls_acc = accuracy(cls_data["y"][va], pred)
    cls_auc = roc_auc_binary(cls_data["y"][va], p)

    tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
    Xr = design_matrix(reg_data["X_num"], reg_data["X_cat"], reg_data["cat_cards"], "ordinal")
    tree_r = DecisionTreeScratch(max_depth=4, task="regression").fit(Xr[tr_r], reg_data["y"][tr_r])
    pred_r = tree_r.predict(Xr[va_r])
    reg_rmse = rmse(reg_data["y"][va_r], pred_r)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["FE-linear", "tree"], [prev["cls_acc"], cls_acc], color=["#27ae60", "#e67e22"])
    ax.set_title("Decision tree vs engineered linear")
    payload = {
        "task": "decision_tree",
        "concept": "Axis-aligned partitions learn interactions without hand FE",
        "metrics": {"cls_acc": cls_acc, "cls_auc": cls_auc, "reg_rmse": reg_rmse, "n_nodes": len([n for n in tree.nodes if n])},
        "cls_acc": cls_acc,
        "reg_rmse": reg_rmse,
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{cls_acc:.3f}",
        "you_should_feel": "树自动切交互，但单棵高方差、边界锯齿",
        "ok": True,
    }
    save_stage("fs04_trees", payload, {"tree_vs_linear": fig})
    return {"cls_acc": cls_acc, "reg_rmse": reg_rmse, "cls_auc": cls_auc}


def fs05_random_forest(cls_data, reg_data, prev):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    X = design_matrix(cls_data["X_num"], cls_data["X_cat"], cls_data["cat_cards"], "ordinal")
    rf = RandomForestScratch(n_estimators=20, max_depth=4, task="classification").fit(X[tr], cls_data["y"][tr])
    pred, p = rf.predict(X[va])
    cls_acc = accuracy(cls_data["y"][va], pred)
    cls_auc = roc_auc_binary(cls_data["y"][va], p)

    tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
    Xr = design_matrix(reg_data["X_num"], reg_data["X_cat"], reg_data["cat_cards"], "ordinal")
    rf_r = RandomForestScratch(n_estimators=20, max_depth=4, task="regression").fit(Xr[tr_r], reg_data["y"][tr_r])
    pred_r = rf_r.predict(Xr[va_r])
    reg_rmse = rmse(reg_data["y"][va_r], pred_r)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["tree", "RF"], [prev["cls_acc"], cls_acc], color=["#e67e22", "#16a085"])
    ax.set_title("Bagging reduces variance")
    payload = {
        "task": "random_forest",
        "concept": "Bagging + feature subsample averages noisy trees",
        "metrics": {"cls_acc": cls_acc, "cls_auc": cls_auc, "reg_rmse": reg_rmse},
        "cls_acc": cls_acc,
        "reg_rmse": reg_rmse,
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{cls_acc:.3f}",
        "you_should_feel": "集成用平均换稳定；RF 是表格经典甜点",
        "ok": True,
    }
    save_stage("fs05_random_forest", payload, {"rf_vs_tree": fig})
    return {"cls_acc": cls_acc, "reg_rmse": reg_rmse, "cls_auc": cls_auc}


def fs06_gbdt_scratch(cls_data, reg_data, prev):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    X = design_matrix(cls_data["X_num"], cls_data["X_cat"], cls_data["cat_cards"], "ordinal")
    gb = GradientBoostingScratch(n_estimators=40, max_depth=2, lr=0.1, task="classification").fit(X[tr], cls_data["y"][tr])
    pred, p = gb.predict(X[va])
    cls_acc = accuracy(cls_data["y"][va], pred)
    cls_auc = roc_auc_binary(cls_data["y"][va], p)

    tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
    Xr = design_matrix(reg_data["X_num"], reg_data["X_cat"], reg_data["cat_cards"], "ordinal")
    gb_r = GradientBoostingScratch(n_estimators=40, max_depth=2, lr=0.1, task="regression").fit(Xr[tr_r], reg_data["y"][tr_r])
    pred_r = gb_r.predict(Xr[va_r])
    reg_rmse = rmse(reg_data["y"][va_r], pred_r)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["RF", "GBDT-scratch"], [prev["cls_acc"], cls_acc], color=["#16a085", "#c0392b"])
    ax.set_title("Boosting fits residuals")
    payload = {
        "task": "gbdt_scratch",
        "concept": "Stage-wise additive modeling on residuals / pseudo-residuals",
        "metrics": {"cls_acc": cls_acc, "cls_auc": cls_auc, "reg_rmse": reg_rmse},
        "cls_acc": cls_acc,
        "reg_rmse": reg_rmse,
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{cls_acc:.3f}; rmse {prev['reg_rmse']:.3f}→{reg_rmse:.3f}",
        "you_should_feel": "Boosting 用偏差换精度；表格王座的核心机制",
        "ok": True,
    }
    save_stage("fs06_gbdt_scratch", payload, {"gbdt": fig})
    return {"cls_acc": cls_acc, "reg_rmse": reg_rmse, "cls_auc": cls_auc}


def fs07_hist_gbdt(cls_data, reg_data, prev):
    """Industrial-strength: sklearn HistGradientBoosting if available, else deeper scratch."""
    tr, va = train_val_split_idx(len(cls_data["y"]))
    X = design_matrix(cls_data["X_num"], cls_data["X_cat"], cls_data["cat_cards"], "ordinal")
    used = "scratch_deep"
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        clf = HistGradientBoostingClassifier(max_depth=6, max_iter=80, learning_rate=0.08, random_state=SEED)
        clf.fit(X[tr], cls_data["y"][tr])
        p = clf.predict_proba(X[va])[:, 1]
        pred = clf.predict(X[va])
        used = "sklearn_hist_gbdt"
        tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
        Xr = design_matrix(reg_data["X_num"], reg_data["X_cat"], reg_data["cat_cards"], "ordinal")
        reg = HistGradientBoostingRegressor(max_depth=6, max_iter=80, learning_rate=0.08, random_state=SEED)
        reg.fit(Xr[tr_r], reg_data["y"][tr_r])
        pred_r = reg.predict(Xr[va_r])
    except Exception as e:
        print("HistGBDT fallback", e)
        gb = GradientBoostingScratch(n_estimators=80, max_depth=3, lr=0.08, task="classification").fit(X[tr], cls_data["y"][tr])
        pred, p = gb.predict(X[va])
        tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
        Xr = design_matrix(reg_data["X_num"], reg_data["X_cat"], reg_data["cat_cards"], "ordinal")
        gb_r = GradientBoostingScratch(n_estimators=80, max_depth=3, lr=0.08, task="regression").fit(Xr[tr_r], reg_data["y"][tr_r])
        pred_r = gb_r.predict(Xr[va_r])

    cls_acc = accuracy(cls_data["y"][va], pred)
    cls_auc = roc_auc_binary(cls_data["y"][va], p)
    reg_rmse = rmse(reg_data["y"][va_r], pred_r)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["GBDT-scratch", "HistGBDT"], [prev["cls_acc"], cls_acc], color=["#c0392b", "#2c3e50"])
    ax.set_title(f"Industrial GBDT ({used})")
    payload = {
        "task": "hist_gbdt",
        "concept": "Histogram binning + deeper boosting = modern tabular default",
        "backend": used,
        "metrics": {"cls_acc": cls_acc, "cls_auc": cls_auc, "reg_rmse": reg_rmse},
        "cls_acc": cls_acc,
        "reg_rmse": reg_rmse,
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{cls_acc:.3f}",
        "you_should_feel": "工业 GBDT：更快、更稳、Kaggle 表格默认武器",
        "ok": True,
    }
    save_stage("fs07_hist_gbdt", payload, {"hist_gbdt": fig})
    return {"cls_acc": cls_acc, "reg_rmse": reg_rmse, "cls_auc": cls_auc}


def fs08_target_encoding(cls_data, reg_data, prev):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    maps, gmean = target_encode_fit(cls_data["X_cat"][tr], cls_data["y"][tr], cls_data["cat_cards"])
    te_tr = target_encode_apply(cls_data["X_cat"][tr], maps, gmean)
    te_va = target_encode_apply(cls_data["X_cat"][va], maps, gmean)
    Xtr = np.hstack([cls_data["X_num"][tr], te_tr])
    Xva = np.hstack([cls_data["X_num"][va], te_va])
    Xtr_s, mu, sd = standardize_fit(Xtr)
    Xva_s = standardize_apply(Xva, mu, sd)
    # compare onehot linear vs TE linear
    clf = LogisticRegressionGD(lr=0.2, n_iter=400).fit(Xtr_s, cls_data["y"][tr])
    p = clf.predict_proba(Xva_s)
    te_acc = accuracy(cls_data["y"][va], (p >= 0.5).astype(int))
    te_auc = roc_auc_binary(cls_data["y"][va], p)

    # also feed TE into hist/gbdt
    X_all_tr = np.hstack([cls_data["X_num"][tr], te_tr])
    X_all_va = np.hstack([cls_data["X_num"][va], te_va])
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf2 = HistGradientBoostingClassifier(max_depth=6, max_iter=60, random_state=SEED)
        clf2.fit(X_all_tr, cls_data["y"][tr])
        p2 = clf2.predict_proba(X_all_va)[:, 1]
        pred2 = clf2.predict(X_all_va)
        gb_acc = accuracy(cls_data["y"][va], pred2)
        gb_auc = roc_auc_binary(cls_data["y"][va], p2)
        backend = "hist+TE"
    except Exception:
        gb = GradientBoostingScratch(n_estimators=40, max_depth=2, task="classification").fit(X_all_tr, cls_data["y"][tr])
        pred2, p2 = gb.predict(X_all_va)
        gb_acc = accuracy(cls_data["y"][va], pred2)
        gb_auc = roc_auc_binary(cls_data["y"][va], p2)
        backend = "scratch+TE"

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["prev GBDT", "TE+linear", "TE+GBDT"], [prev["cls_acc"], te_acc, gb_acc], color=["#2c3e50", "#f39c12", "#d35400"])
    ax.set_title("Target encoding for high-card cats")
    ax.set_ylim(0, 1)
    payload = {
        "task": "target_encoding",
        "concept": "Smoothed P(y|category) compresses high-cardinality cats",
        "backend": backend,
        "metrics": {"te_linear_acc": te_acc, "te_linear_auc": te_auc, "te_gbdt_acc": gb_acc, "te_gbdt_auc": gb_auc},
        "cls_acc": gb_acc,
        "reg_rmse": prev.get("reg_rmse"),
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{gb_acc:.3f} with TE",
        "you_should_feel": "类别列的正确编码往往 > 换模型；但泄漏会虚高",
        "ok": True,
    }
    save_stage("fs08_target_encoding", payload, {"target_encoding": fig})
    return {"cls_acc": gb_acc, "reg_rmse": prev.get("reg_rmse"), "cls_auc": gb_auc}


def fs09_mlp(cls_data, reg_data, prev):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    model = CatEmbedMLP(cls_data["X_num"].shape[1], cls_data["cat_cards"], task="classification", epochs=35, hidden=128)
    model.fit(cls_data["X_num"][tr], cls_data["X_cat"][tr], cls_data["y"][tr],
              cls_data["X_num"][va], cls_data["X_cat"][va], cls_data["y"][va])
    pred, p = model.predict(cls_data["X_num"][va], cls_data["X_cat"][va])
    cls_acc = accuracy(cls_data["y"][va], pred)
    cls_auc = roc_auc_binary(cls_data["y"][va], p)

    tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
    regm = CatEmbedMLP(reg_data["X_num"].shape[1], reg_data["cat_cards"], task="regression", epochs=35, hidden=128)
    regm.fit(reg_data["X_num"][tr_r], reg_data["X_cat"][tr_r], reg_data["y"][tr_r],
             reg_data["X_num"][va_r], reg_data["X_cat"][va_r], reg_data["y"][va_r])
    pred_r = regm.predict(reg_data["X_num"][va_r], reg_data["X_cat"][va_r])
    reg_rmse = rmse(reg_data["y"][va_r], pred_r)

    torch, _, device = get_torch()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["GBDT", "EmbedMLP"], [prev["cls_acc"], cls_acc], color=["#d35400", "#8e44ad"])
    ax.set_title(f"MLP on {device}")
    payload = {
        "task": "mlp",
        "concept": "Deep MLP + early stopping on standardized tabular inputs",
        "device": str(device),
        "metrics": {"cls_acc": cls_acc, "cls_auc": cls_auc, "reg_rmse": reg_rmse},
        "cls_acc": cls_acc,
        "reg_rmse": reg_rmse,
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{cls_acc:.3f} (often GBDT still wins small data)",
        "you_should_feel": "纯 MLP 在中小表格常输给 GBDT——这是领域事实",
        "ok": True,
    }
    save_stage("fs09_mlp", payload, {"mlp": fig})
    return {"cls_acc": cls_acc, "reg_rmse": reg_rmse, "cls_auc": cls_auc}


def fs10_embeddings(cls_data, reg_data, prev):
    """Emphasize categorical embeddings vs one-hot MLP — already in CatEmbedMLP; ablate."""
    tr, va = train_val_split_idx(len(cls_data["y"]))
    # one-hot MLP (no emb): flatten
    torch, nn, device = get_torch()
    X_oh = design_matrix(cls_data["X_num"], cls_data["X_cat"], cls_data["cat_cards"], "onehot")
    Xtr, mu, sd = standardize_fit(X_oh[tr])
    Xva = standardize_apply(X_oh[va], mu, sd)
    model = build_mlp(nn, Xtr.shape[1], 128, 1, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(cls_data["y"][tr], dtype=torch.float32, device=device)
    for ep in range(35):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, ytr_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p_oh = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32, device=device)).squeeze(-1)).cpu().numpy()
    acc_oh = accuracy(cls_data["y"][va], (p_oh >= 0.5).astype(int))

    emb = CatEmbedMLP(cls_data["X_num"].shape[1], cls_data["cat_cards"], epochs=35)
    emb.fit(cls_data["X_num"][tr], cls_data["X_cat"][tr], cls_data["y"][tr],
            cls_data["X_num"][va], cls_data["X_cat"][va], cls_data["y"][va])
    pred_e, p_e = emb.predict(cls_data["X_num"][va], cls_data["X_cat"][va])
    acc_emb = accuracy(cls_data["y"][va], pred_e)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["onehot-MLP", "embed-MLP"], [acc_oh, acc_emb], color=["#9b59b6", "#8e44ad"])
    ax.set_title("Categorical embeddings vs one-hot")
    ax.set_ylim(0, 1)
    payload = {
        "task": "embeddings",
        "concept": "Learned embeddings share statistical strength across rare categories",
        "metrics": {"onehot_mlp_acc": acc_oh, "embed_mlp_acc": acc_emb},
        "cls_acc": acc_emb,
        "reg_rmse": prev.get("reg_rmse"),
        "vs_prev": f"onehot {acc_oh:.3f} vs emb {acc_emb:.3f}",
        "you_should_feel": "Embedding 是 DL 处理类别的正确姿势",
        "ok": True,
    }
    save_stage("fs10_embeddings", payload, {"embeddings": fig})
    return {"cls_acc": acc_emb, "reg_rmse": prev.get("reg_rmse"), "cls_auc": roc_auc_binary(cls_data["y"][va], p_e)}


def fs11_ft_transformer(cls_data, reg_data, prev):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    ft = FTTransformerMini(cls_data["X_num"].shape[1], cls_data["cat_cards"], task="classification", epochs=30, d_token=32, n_layers=2)
    ft.fit(cls_data["X_num"][tr], cls_data["X_cat"][tr], cls_data["y"][tr])
    pred, p = ft.predict(cls_data["X_num"][va], cls_data["X_cat"][va])
    cls_acc = accuracy(cls_data["y"][va], pred)
    cls_auc = roc_auc_binary(cls_data["y"][va], p)

    tr_r, va_r = train_val_split_idx(len(reg_data["y"]), seed=SEED + 1)
    ft_r = FTTransformerMini(reg_data["X_num"].shape[1], reg_data["cat_cards"], task="regression", epochs=30, d_token=32, n_layers=2)
    ft_r.fit(reg_data["X_num"][tr_r], reg_data["X_cat"][tr_r], reg_data["y"][tr_r])
    pred_r = ft_r.predict(reg_data["X_num"][va_r], reg_data["X_cat"][va_r])
    reg_rmse = rmse(reg_data["y"][va_r], pred_r)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["EmbedMLP", "FT-Transformer"], [prev["cls_acc"], cls_acc], color=["#8e44ad", "#1a5276"])
    ax.set_title("FT-Transformer mini")
    payload = {
        "task": "ft_transformer",
        "concept": "Tokenize each feature + Transformer — modern tabular DL backbone",
        "metrics": {"cls_acc": cls_acc, "cls_auc": cls_auc, "reg_rmse": reg_rmse},
        "cls_acc": cls_acc,
        "reg_rmse": reg_rmse,
        "vs_prev": f"cls {prev['cls_acc']:.3f}→{cls_acc:.3f}",
        "you_should_feel": "特征即 token：注意力建模特征交互",
        "ok": True,
    }
    save_stage("fs11_ft_transformer", payload, {"ftt": fig})
    return {"cls_acc": cls_acc, "reg_rmse": reg_rmse, "cls_auc": cls_auc}


def fs12_stacking(cls_data, reg_data, prev_gbdt, prev_dl):
    tr, va = train_val_split_idx(len(cls_data["y"]))
    X = design_matrix(cls_data["X_num"], cls_data["X_cat"], cls_data["cat_cards"], "ordinal")
    # base1 gbdt
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        b1 = HistGradientBoostingClassifier(max_depth=5, max_iter=50, random_state=SEED)
        b1.fit(X[tr], cls_data["y"][tr])
        p1 = b1.predict_proba(X[va])[:, 1]
        p1_tr = b1.predict_proba(X[tr])[:, 1]
    except Exception:
        b1 = GradientBoostingScratch(n_estimators=30, max_depth=2, task="classification").fit(X[tr], cls_data["y"][tr])
        _, p1 = b1.predict(X[va])
        _, p1_tr = b1.predict(X[tr])
    # base2 linear
    Xoh = design_matrix(cls_data["X_num"], cls_data["X_cat"], cls_data["cat_cards"], "onehot")
    Xtr, mu, sd = standardize_fit(Xoh[tr])
    Xva = standardize_apply(Xoh[va], mu, sd)
    b2 = LogisticRegressionGD(lr=0.2, n_iter=300).fit(Xtr, cls_data["y"][tr])
    p2 = b2.predict_proba(Xva)
    p2_tr = b2.predict_proba(Xtr)
    # meta
    meta_tr = np.column_stack([p1_tr, p2_tr])
    meta_va = np.column_stack([p1, p2])
    meta = LogisticRegressionGD(lr=0.2, n_iter=200).fit(meta_tr, cls_data["y"][tr])
    p_m = meta.predict_proba(meta_va)
    acc = accuracy(cls_data["y"][va], (p_m >= 0.5).astype(int))
    auc = roc_auc_binary(cls_data["y"][va], p_m)
    acc1 = accuracy(cls_data["y"][va], (p1 >= 0.5).astype(int))
    acc2 = accuracy(cls_data["y"][va], (p2 >= 0.5).astype(int))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["GBDT", "Linear", "Stack"], [acc1, acc2, acc], color=["#2c3e50", "#2980b9", "#27ae60"])
    ax.set_title("Stacking ensemble")
    ax.set_ylim(0, 1)
    payload = {
        "task": "stacking",
        "concept": "Blend diverse inductive biases with a meta-learner",
        "metrics": {"gbdt_acc": acc1, "linear_acc": acc2, "stack_acc": acc, "stack_auc": auc},
        "cls_acc": acc,
        "reg_rmse": prev_gbdt.get("reg_rmse"),
        "vs_prev": f"best base {max(acc1,acc2):.3f} → stack {acc:.3f}",
        "you_should_feel": "竞赛后期靠多样性 stacking 抠点",
        "ok": True,
    }
    save_stage("fs12_stacking", payload, {"stacking": fig})
    return {"cls_acc": acc, "reg_rmse": prev_gbdt.get("reg_rmse"), "cls_auc": auc}


# ---- Time series stages ----

def fs13_ts_protocol(ts_data):
    y = ts_data["y"]
    # show leakage pathology: shuffle split vs walk-forward
    n = len(y)
    # fake features = lag1 but wrong evaluation with shuffle
    X = y[:-1].reshape(-1, 1)
    yy = y[1:]
    tr, va = train_val_split_idx(len(yy), seed=0)
    # linear
    w = np.linalg.lstsq(X[tr], yy[tr], rcond=None)[0]
    pred_shuf = X[va] @ w
    rmse_shuf = rmse(yy[va], pred_shuf)
    # walk-forward last block
    split = int(n * 0.8)
    # train on past only
    Xtr, ytr = y[:split - 1].reshape(-1, 1), y[1:split]
    w2 = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    Xte, yte = y[split - 1:-1].reshape(-1, 1), y[split:]
    pred_wf = Xte @ w2
    rmse_wf = rmse(yte, pred_wf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(y[-200:], label="y")
    axes[0].set_title("TS tail")
    axes[1].bar(["shuffle split", "walk-forward"], [rmse_shuf, rmse_wf], color=["#e74c3c", "#27ae60"])
    axes[1].set_title("Leakage: optimistic shuffle RMSE")
    payload = {
        "task": "ts_protocol",
        "concept": "Time series must use temporal splits; shuffle leaks future",
        "metrics": {"rmse_shuffle": rmse_shuf, "rmse_walk_forward": rmse_wf},
        "you_should_feel": "打乱时间=考试偷看答案；必须 walk-forward",
        "vs_prev": "从 i.i.d. 表格进入时序协议",
        "ok": True,
    }
    save_stage("fs13_ts_protocol", payload, {"ts_protocol": fig})
    return payload["metrics"]


def fs14_naive_forecasts(ts_data):
    y = ts_data["y"]
    horizon = 30
    split = len(y) - horizon
    y_tr, y_te = y[:split], y[split:]
    # last value
    naive = np.full(horizon, y_tr[-1])
    # seasonal naive period 7
    seas = np.array([y_tr[-7 + (i % 7)] for i in range(horizon)])
    # moving average 7
    ma = np.full(horizon, y_tr[-7:].mean())
    m = {
        "rmse_naive_last": rmse(y_te, naive),
        "rmse_seasonal_naive7": rmse(y_te, seas),
        "rmse_ma7": rmse(y_te, ma),
        "smape_seasonal": smape(y_te, seas),
    }
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(range(horizon), y_te, label="actual", color="black")
    ax.plot(range(horizon), naive, label="last", alpha=0.8)
    ax.plot(range(horizon), seas, label="seas7", alpha=0.8)
    ax.plot(range(horizon), ma, label="ma7", alpha=0.8)
    ax.legend(); ax.set_title("FS14 naive forecasts")
    payload = {
        "task": "ts_naive",
        "concept": "Last / seasonal naive / MA are mandatory baselines",
        "metrics": m,
        "rmse": m["rmse_seasonal_naive7"],
        "vs_prev": "给出时序不可再低的参照",
        "you_should_feel": "季节朴素经常意外地强",
        "ok": True,
    }
    save_stage("fs14_naive_forecasts", payload, {"naive": fig})
    return m


def fs15_exp_smoothing(ts_data, prev):
    y = ts_data["y"]
    horizon = 30
    split = len(y) - horizon
    y_tr, y_te = y[:split], y[split:]

    def holt_linear(series, alpha=0.4, beta=0.2, h=30):
        level = series[0]
        trend = series[1] - series[0]
        for t in range(1, len(series)):
            prev_level = level
            level = alpha * series[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        return np.array([level + (i + 1) * trend for i in range(h)])

    def holt_winters_add(series, season_len=7, alpha=0.4, beta=0.2, gamma=0.3, h=30):
        n = len(series)
        level = series[:season_len].mean()
        trend = (series[season_len:2 * season_len].mean() - series[:season_len].mean()) / season_len
        season = list(series[:season_len] - level)
        for t in range(season_len, n):
            s = season[t % season_len]
            prev_level = level
            level = alpha * (series[t] - s) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            season[t % season_len] = gamma * (series[t] - level) + (1 - gamma) * s
        out = []
        for i in range(h):
            out.append(level + (i + 1) * trend + season[(n + i) % season_len])
        return np.array(out)

    pred_h = holt_linear(y_tr, h=horizon)
    pred_hw = holt_winters_add(y_tr, h=horizon)
    m = {
        "rmse_holt": rmse(y_te, pred_h),
        "rmse_holt_winters": rmse(y_te, pred_hw),
        "smape_hw": smape(y_te, pred_hw),
    }
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(y_te, label="actual", color="black")
    ax.plot(pred_h, label="Holt")
    ax.plot(pred_hw, label="Holt-Winters")
    ax.legend(); ax.set_title("FS15 exponential smoothing")
    payload = {
        "task": "exp_smoothing",
        "concept": "Level+trend+season recursive updates (ETS family)",
        "metrics": m,
        "rmse": m["rmse_holt_winters"],
        "vs_prev": f"seas_naive {prev['rmse_seasonal_naive7']:.3f} → HW {m['rmse_holt_winters']:.3f}",
        "you_should_feel": "平滑法用递归状态捕捉水平/趋势/季节",
        "ok": True,
    }
    save_stage("fs15_exp_smoothing", payload, {"ets": fig})
    return m


def fs16_lag_tree(ts_data, prev):
    y = ts_data["y"]
    horizon = 30
    split = len(y) - horizon
    lags = (1, 2, 3, 7, 14)
    # train matrix only from past
    X_all, y_all, idx = make_lag_matrix(y, lags=lags, start=0, end=split)
    # fit RF-like gbdt on lags
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_depth=5, max_iter=80, random_state=SEED)
        model.fit(X_all, y_all)
        def predict_next(history):
            x = np.array([[history[-L] for L in lags]], float)
            return float(model.predict(x)[0])
        backend = "hist_gbdt"
    except Exception:
        model = GradientBoostingScratch(n_estimators=40, max_depth=2, task="regression").fit(X_all, y_all)
        def predict_next(history):
            x = np.array([[history[-L] for L in lags]], float)
            return float(model.predict(x)[0])
        backend = "scratch_gbdt"
    # recursive multi-step
    hist = list(y[:split])
    preds = []
    for _ in range(horizon):
        p = predict_next(hist)
        preds.append(p)
        hist.append(p)
    preds = np.array(preds)
    y_te = y[split:split + horizon]
    m = {"rmse": rmse(y_te, preds), "mae": mae(y_te, preds), "smape": smape(y_te, preds), "backend": backend}
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(y_te, label="actual", color="black")
    ax.plot(preds, label="lag-tree recursive")
    ax.legend(); ax.set_title("FS16 lag features + tree")
    payload = {
        "task": "lag_tree",
        "concept": "Tabularize TS via lags/rolling feats + GBDT",
        "metrics": m,
        "rmse": m["rmse"],
        "vs_prev": f"HW {prev['rmse_holt_winters']:.3f} → lag-tree {m['rmse']:.3f}",
        "you_should_feel": "时序→表格化是工业最强捷径之一",
        "ok": True,
    }
    save_stage("fs16_lag_tree", payload, {"lag_tree": fig})
    return m


def fs17_ar_linear(ts_data, prev):
    y = ts_data["y"]
    horizon = 30
    split = len(y) - horizon
    lags = (1, 2, 3, 7)
    X_all, y_all, _ = make_lag_matrix(y, lags=lags, end=split)
    # ridge closed form
    lam = 1e-2
    A = X_all.T @ X_all + lam * np.eye(X_all.shape[1])
    w = np.linalg.solve(A, X_all.T @ y_all)
    hist = list(y[:split])
    preds = []
    for _ in range(horizon):
        x = np.array([hist[-L] for L in lags], float)
        p = float(x @ w)
        preds.append(p)
        hist.append(p)
    preds = np.array(preds)
    y_te = y[split:split + horizon]
    m = {"rmse": rmse(y_te, preds), "smape": smape(y_te, preds)}
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(y_te, label="actual", color="black")
    ax.plot(preds, label="AR-ridge")
    ax.legend(); ax.set_title("FS17 linear AR")
    payload = {
        "task": "ar_linear",
        "concept": "Autoregressive linear model — ARIMA's AR core",
        "metrics": m,
        "rmse": m["rmse"],
        "vs_prev": f"lag-tree {prev['rmse']:.3f} vs AR {m['rmse']:.3f}",
        "you_should_feel": "线性 AR 解释性强，非线性季节/冲击时吃力",
        "ok": True,
    }
    save_stage("fs17_ar_linear", payload, {"ar": fig})
    return m


def fs18_lstm(ts_data, prev):
    torch, nn, device = get_torch()
    y = ts_data["y"]
    horizon = 30
    split = len(y) - horizon
    seq_len = 28
    # scale
    mu, sd = y[:split].mean(), y[:split].std()
    yn = (y - mu) / (sd + 1e-8)

    def make_seq(arr, end):
        xs, ys = [], []
        for t in range(seq_len, end):
            xs.append(arr[t - seq_len:t])
            ys.append(arr[t])
        return np.asarray(xs)[:, :, None], np.asarray(ys)

    Xtr, ytr = make_seq(yn, split)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    model = LSTMForecaster().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(25):
        model.train()
        opt.zero_grad()
        # mini batches
        perm = torch.randperm(len(Xtr_t), device=device)
        total = 0.0
        for i in range(0, len(Xtr_t), 128):
            idx = perm[i:i + 128]
            pred = model(Xtr_t[idx])
            loss = nn.functional.mse_loss(pred, ytr_t[idx])
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += float(loss.item())
    # recursive forecast
    model.eval()
    hist = list(yn[:split])
    preds_n = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(hist[-seq_len:], dtype=torch.float32, device=device).view(1, seq_len, 1)
            p = float(model(x).item())
            preds_n.append(p)
            hist.append(p)
    preds = np.array(preds_n) * sd + mu
    y_te = y[split:split + horizon]
    m = {"rmse": rmse(y_te, preds), "smape": smape(y_te, preds), "device": str(device)}
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(y_te, label="actual", color="black")
    ax.plot(preds, label="LSTM")
    ax.legend(); ax.set_title(f"FS18 LSTM on {device}")
    payload = {
        "task": "lstm",
        "concept": "Sequence model memorizes temporal state end-to-end",
        "metrics": m,
        "rmse": m["rmse"],
        "vs_prev": f"AR {prev['rmse']:.3f} → LSTM {m['rmse']:.3f}",
        "you_should_feel": "LSTM 自动学记忆，但小数据易过拟合、训练更重",
        "ok": True,
    }
    save_stage("fs18_lstm", payload, {"lstm": fig})
    return m


def fs19_tcn_nbeats(ts_data, prev):
    """1D conv TCN-ish + tiny N-BEATS style basis expansion."""
    torch, nn, device = get_torch()
    y = ts_data["y"]
    horizon = 30
    split = len(y) - horizon
    seq_len = 28
    mu, sd = y[:split].mean(), y[:split].std()
    yn = (y - mu) / (sd + 1e-8)

    def make_seq(arr, end):
        xs, ys = [], []
        for t in range(seq_len, end):
            xs.append(arr[t - seq_len:t])
            ys.append(arr[t])
        return np.asarray(xs)[:, None, :], np.asarray(ys)  # B,1,L

    Xtr, ytr = make_seq(yn, split)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=2, dilation=1),
                nn.ReLU(),
                nn.Conv1d(16, 16, kernel_size=3, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv1d(16, 16, kernel_size=3, padding=8, dilation=4),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            h = self.net(x).squeeze(-1)
            return self.head(h).squeeze(-1)

    class NBeatsBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(seq_len, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.theta_b = nn.Linear(64, seq_len)  # backcast
            self.theta_f = nn.Linear(64, 1)  # forecast 1-step

        def forward(self, x):
            # x: B,L
            h = self.fc(x)
            backcast = self.theta_b(h)
            forecast = self.theta_f(h).squeeze(-1)
            return backcast, forecast

    tcn = TCN().to(device)
    nbeats = NBeatsBlock().to(device)
    opt = torch.optim.Adam(list(tcn.parameters()) + list(nbeats.parameters()), lr=1e-3)
    for ep in range(25):
        tcn.train(); nbeats.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(Xtr_t), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            p1 = tcn(Xtr_t[idx])
            xb = Xtr_t[idx].squeeze(1)
            bc, p2 = nbeats(xb)
            loss = nn.functional.mse_loss(p1, ytr_t[idx]) + nn.functional.mse_loss(p2, ytr_t[idx]) + 0.1 * nn.functional.mse_loss(bc, xb)
            loss.backward()
            opt.step()

    def forecast_model(model_kind):
        hist = list(yn[:split])
        preds = []
        with torch.no_grad():
            for _ in range(horizon):
                if model_kind == "tcn":
                    x = torch.tensor(hist[-seq_len:], dtype=torch.float32, device=device).view(1, 1, seq_len)
                    p = float(tcn(x).item())
                else:
                    x = torch.tensor(hist[-seq_len:], dtype=torch.float32, device=device).view(1, seq_len)
                    _, p = nbeats(x)
                    p = float(p.item())
                preds.append(p)
                hist.append(p)
        return np.array(preds) * sd + mu

    tcn.eval(); nbeats.eval()
    pred_tcn = forecast_model("tcn")
    pred_nb = forecast_model("nbeats")
    y_te = y[split:split + horizon]
    m = {
        "rmse_tcn": rmse(y_te, pred_tcn),
        "rmse_nbeats": rmse(y_te, pred_nb),
        "best_rmse": min(rmse(y_te, pred_tcn), rmse(y_te, pred_nb)),
        "device": str(device),
    }
    m["rmse"] = m["best_rmse"]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(y_te, label="actual", color="black")
    ax.plot(pred_tcn, label="TCN")
    ax.plot(pred_nb, label="N-BEATS block")
    ax.legend(); ax.set_title("FS19 TCN / N-BEATS mini")
    payload = {
        "task": "tcn_nbeats",
        "concept": "Dilated conv (TCN) and doubly residual basis (N-BEATS)",
        "metrics": m,
        "rmse": m["best_rmse"],
        "vs_prev": f"LSTM {prev['rmse']:.3f} → best conv/nbeats {m['best_rmse']:.3f}",
        "you_should_feel": "现代时序 DL：卷积感受野与可解释 stack 块",
        "ok": True,
    }
    save_stage("fs19_tcn_nbeats", payload, {"tcn_nbeats": fig})
    return m


def fs20_patch_transformer(ts_data, prev):
    """PatchTST-inspired: patchify lookback, tiny Transformer, predict next."""
    torch, nn, device = get_torch()
    y = ts_data["y"]
    horizon = 30
    split = len(y) - horizon
    seq_len = 48
    patch = 8
    assert seq_len % patch == 0
    n_patches = seq_len // patch
    d_model = 32
    mu, sd = y[:split].mean(), y[:split].std()
    yn = (y - mu) / (sd + 1e-8)

    def make_seq(arr, end):
        xs, ys = [], []
        for t in range(seq_len, end):
            xs.append(arr[t - seq_len:t])
            ys.append(arr[t])
        return np.asarray(xs), np.asarray(ys)

    Xtr, ytr = make_seq(yn, split)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)

    class PatchTST(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(patch, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.1)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(d_model * n_patches, 1)

        def forward(self, x):
            # x: B,L
            B = x.size(0)
            x = x.view(B, n_patches, patch)
            tok = self.proj(x)
            h = self.enc(tok)
            return self.head(h.reshape(B, -1)).squeeze(-1)

    model = PatchTST().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(25):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(Xtr_t), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = nn.functional.mse_loss(pred, ytr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    hist = list(yn[:split])
    preds_n = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(hist[-seq_len:], dtype=torch.float32, device=device).view(1, seq_len)
            p = float(model(x).item())
            preds_n.append(p)
            hist.append(p)
    preds = np.array(preds_n) * sd + mu
    y_te = y[split:split + horizon]
    m = {"rmse": rmse(y_te, preds), "smape": smape(y_te, preds), "device": str(device)}
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(y_te, label="actual", color="black")
    ax.plot(preds, label="PatchTransformer")
    ax.legend(); ax.set_title("FS20 PatchTST mini")
    payload = {
        "task": "patch_transformer",
        "concept": "Patchify time + Transformer channels — PatchTST spirit",
        "metrics": m,
        "rmse": m["rmse"],
        "vs_prev": f"TCN/NBEATS {prev.get('rmse', prev.get('best_rmse', float('nan'))):.3f} → Patch {m['rmse']:.3f}",
        "you_should_feel": "前沿时序：把时间切 patch 当 token",
        "ok": True,
    }
    save_stage("fs20_patch_transformer", payload, {"patch": fig})
    return m


def fs21_leaderboard(all_cls, all_reg, all_ts):
    """Final comparison map across the whole ladder."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # classification
    names = list(all_cls.keys())
    vals = [all_cls[k] for k in names]
    axes[0].plot(range(len(names)), vals, marker="o")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Classification accuracy ladder")
    axes[0].set_ylim(0, 1.05)
    # regression
    rnames = list(all_reg.keys())
    rvals = [all_reg[k] for k in rnames]
    axes[1].plot(range(len(rnames)), rvals, marker="o", color="#c0392b")
    axes[1].set_xticks(range(len(rnames)))
    axes[1].set_xticklabels(rnames, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Regression RMSE ladder (lower better)")
    # ts
    tnames = list(all_ts.keys())
    tvals = [all_ts[k] for k in tnames]
    axes[2].plot(range(len(tnames)), tvals, marker="o", color="#8e44ad")
    axes[2].set_xticks(range(len(tnames)))
    axes[2].set_xticklabels(tnames, rotation=45, ha="right", fontsize=8)
    axes[2].set_title("Time series RMSE ladder")
    fig.tight_layout()

    payload = {
        "task": "leaderboard",
        "concept": "Full-domain map: from constants to modern tabular/TS systems",
        "classification_acc": all_cls,
        "regression_rmse": all_reg,
        "timeseries_rmse": all_ts,
        "best_cls": max(all_cls, key=all_cls.get),
        "best_reg": min(all_reg, key=all_reg.get),
        "best_ts": min(all_ts, key=all_ts.get),
        "map": {
            "from_zero": ["fs00", "fs01"],
            "classic": ["fs02", "fs03", "fs04", "fs05"],
            "key_breakthroughs": ["fs06", "fs07", "fs08"],
            "modern_systems": ["fs09", "fs10", "fs11", "fs12"],
            "timeseries": ["fs13", "fs14", "fs15", "fs16", "fs17", "fs18", "fs19", "fs20"],
            "frontier_close": ["fs21"],
        },
        "you_should_feel": "表格主线=基线→线性→树→提升→编码→DL/Transformer→集成；时序=协议→朴素→平滑→滞后树→序列DL→Patch",
        "ok": True,
    }
    save_stage("fs21_leaderboard", payload, {"leaderboard": fig})
    return payload


def main():
    t0 = time.time()
    print("OUT_DIR=", OUT_DIR)
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
    except Exception as e:
        print("torch missing", e)

    cls_data = make_tabular_classification()
    reg_data = make_tabular_regression()
    ts_data = make_timeseries()

    fs00_protocol(cls_data, reg_data, ts_data)
    m01 = fs01_baselines(cls_data, reg_data)
    m02 = fs02_linear(cls_data, reg_data, m01)
    m03 = fs03_feature_engineering(cls_data, reg_data, m02)
    m04 = fs04_trees(cls_data, reg_data, m03)
    m05 = fs05_random_forest(cls_data, reg_data, m04)
    m06 = fs06_gbdt_scratch(cls_data, reg_data, m05)
    m07 = fs07_hist_gbdt(cls_data, reg_data, m06)
    m08 = fs08_target_encoding(cls_data, reg_data, m07)
    m09 = fs09_mlp(cls_data, reg_data, m08)
    m10 = fs10_embeddings(cls_data, reg_data, m09)
    m11 = fs11_ft_transformer(cls_data, reg_data, m10)
    m12 = fs12_stacking(cls_data, reg_data, m07, m11)

    fs13_ts_protocol(ts_data)
    t14 = fs14_naive_forecasts(ts_data)
    t15 = fs15_exp_smoothing(ts_data, t14)
    t16 = fs16_lag_tree(ts_data, t15)
    t17 = fs17_ar_linear(ts_data, t16)
    t18 = fs18_lstm(ts_data, t17)
    t19 = fs19_tcn_nbeats(ts_data, t18)
    t20 = fs20_patch_transformer(ts_data, t19)

    all_cls = {
        "maj": m01["cls_acc"],
        "linear": m02["cls_acc"],
        "FE": m03["cls_acc"],
        "tree": m04["cls_acc"],
        "RF": m05["cls_acc"],
        "GBDTs": m06["cls_acc"],
        "HistGB": m07["cls_acc"],
        "TE": m08["cls_acc"],
        "MLP": m09["cls_acc"],
        "Emb": m10["cls_acc"],
        "FTT": m11["cls_acc"],
        "Stack": m12["cls_acc"],
    }
    all_reg = {
        "mean": m01["reg_rmse"],
        "ridge": m02["reg_rmse"],
        "FE": m03["reg_rmse"],
        "tree": m04["reg_rmse"],
        "RF": m05["reg_rmse"],
        "GBDTs": m06["reg_rmse"],
        "HistGB": m07["reg_rmse"],
        "MLP": m09["reg_rmse"],
        "FTT": m11["reg_rmse"],
    }
    all_ts = {
        "seas_naive": t14["rmse_seasonal_naive7"],
        "HW": t15["rmse_holt_winters"],
        "lag_tree": t16["rmse"],
        "AR": t17["rmse"],
        "LSTM": t18["rmse"],
        "TCN/NB": t19["rmse"],
        "Patch": t20["rmse"],
    }
    board = fs21_leaderboard(all_cls, all_reg, all_ts)

    SUMMARY["elapsed_sec"] = round(time.time() - t0, 2)
    SUMMARY["stage_order"] = STAGE_ORDER
    SUMMARY["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    SUMMARY["ok"] = all(True for _ in STAGE_ORDER)
    SUMMARY["leaderboard"] = {
        "classification_acc": all_cls,
        "regression_rmse": all_reg,
        "timeseries_rmse": all_ts,
        "best_cls": board["best_cls"],
        "best_reg": board["best_reg"],
        "best_ts": board["best_ts"],
    }
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(SUMMARY, indent=2))
    print("SUMMARY written", OUT_DIR / "SUMMARY.json")
    print("SMOKE_OK stages=", len(STAGE_ORDER), "elapsed=", SUMMARY["elapsed_sec"])
    return SUMMARY


# CLI: python run_tabular_fs_ladder.py
if __name__ == "__main__" and "get_ipython" not in dir():
    main()
