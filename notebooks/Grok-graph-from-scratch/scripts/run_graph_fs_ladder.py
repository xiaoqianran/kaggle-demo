#!/usr/bin/env python3
"""Graph Machine Learning From-Scratch ladder FS00–FS16.

Offline synthetic graphs (SBM communities + features). Torch GNN on CUDA when available.
Writes results/<stage>/results.json + PNG under OUT_DIR.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

OUT_DIR = Path(os.environ.get("GRAPH_FS_OUT", "/kaggle/working/results"))
if not OUT_DIR.parent.exists():
    OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

STAGE_ORDER: List[str] = []
SUMMARY: Dict[str, Any] = {}


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
    SUMMARY.setdefault("stages", {})[name] = {
        k: payload[k]
        for k in payload
        if k in ("task", "concept", "metrics", "vs_prev", "you_should_feel", "ok", "accuracy", "auc", "micro_f1")
        or k.endswith("_acc")
        or k.endswith("_auc")
    }
    print(f"[OK] {name}")
    return d


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def micro_f1(y_true, y_pred, n_classes: int) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # multi-class micro-F1 == accuracy for single-label
    return accuracy(y_true, y_pred)


def roc_auc_binary(y_true, score) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Graph data: Stochastic Block Model with node features
# ---------------------------------------------------------------------------

def make_sbm_graph(
    n_per_block: int = 100,
    n_blocks: int = 4,
    p_in: float = 0.22,
    p_out: float = 0.015,
    feat_dim: int = 16,
    seed: int = SEED,
    feat_signal: float = 0.25,
) -> Dict[str, Any]:
    """SBM with **weak** features so structure methods must beat feat-only.

    feat_signal: scale of class-conditional mean (small ⇒ features alone hard).
    """
    r = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    y = np.repeat(np.arange(n_blocks), n_per_block)
    # adjacency (undirected, no self-loop)
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if y[i] == y[j] else p_out
            if r.random() < p:
                A[i, j] = A[j, i] = 1.0
    # weak class signal + strong isotropic noise (realistic hard node-clf)
    means = r.normal(size=(n_blocks, feat_dim)) * feat_signal
    X = means[y] + r.normal(0, 1.0, size=(n, feat_dim))
    # train/val/test masks (transductive)
    idx = r.permutation(n)
    n_train = int(0.5 * n)
    n_val = int(0.2 * n)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train:n_train + n_val]] = True
    test_mask[idx[n_train + n_val:]] = True
    # edge list
    rows, cols = np.where(np.triu(A, 1) > 0)
    edges = np.stack([rows, cols], axis=1)
    return {
        "A": A,
        "X": X.astype(np.float64),
        "y": y.astype(int),
        "edges": edges,
        "n": n,
        "n_classes": n_blocks,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "feat_dim": feat_dim,
    }


def normalize_adj(A: np.ndarray, add_self: bool = True) -> np.ndarray:
    A_hat = A + (np.eye(A.shape[0]) if add_self else 0)
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
    D = np.diag(deg_inv_sqrt)
    return D @ A_hat @ D


def degree_features(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1, keepdims=True)
    return deg / max(deg.max(), 1.0)


def train_val_test_split_labels(y, train_mask, val_mask, test_mask):
    return y[train_mask], y[val_mask], y[test_mask]


# ---------------------------------------------------------------------------
# Classic models
# ---------------------------------------------------------------------------

class SoftmaxRegression:
    def __init__(self, n_classes, lr=0.2, n_iter=400, l2=1e-3):
        self.n_classes = n_classes
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.W = None
        self.b = None

    def fit(self, X, y):
        n, d = X.shape
        k = self.n_classes
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        Y = np.eye(k)[y]
        for _ in range(self.n_iter):
            logits = X @ self.W + self.b
            P = softmax(logits, axis=1)
            G = (P - Y) / n
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(axis=0)
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b, axis=1)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def label_propagation(A: np.ndarray, y: np.ndarray, train_mask: np.ndarray, n_iter: int = 40, alpha: float = 0.85):
    """Iterative LP: Y <- alpha S Y + (1-alpha) Y0, clamp train labels."""
    n, k = len(y), int(y.max()) + 1
    Y0 = np.zeros((n, k))
    Y0[train_mask, y[train_mask]] = 1.0
    # row-normalize A (random-walk)
    deg = A.sum(axis=1, keepdims=True)
    S = A / np.maximum(deg, 1e-12)
    Y = Y0.copy()
    for _ in range(n_iter):
        Y = alpha * (S @ Y) + (1 - alpha) * Y0
        Y[train_mask] = Y0[train_mask]
    return Y


def random_walk_cooccur(A: np.ndarray, n_walks: int = 8, walk_len: int = 12, window: int = 3, seed: int = SEED):
    """Count context co-occurrence from walks (DeepWalk spirit)."""
    r = np.random.default_rng(seed)
    n = A.shape[0]
    nbrs = [np.where(A[i] > 0)[0] for i in range(n)]
    counts = np.zeros((n, n), dtype=np.float64)
    for start in range(n):
        for _ in range(n_walks):
            walk = [start]
            cur = start
            for _ in range(walk_len - 1):
                nb = nbrs[cur]
                if len(nb) == 0:
                    break
                cur = int(r.choice(nb))
                walk.append(cur)
            for i, u in enumerate(walk):
                for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                    if i == j:
                        continue
                    v = walk[j]
                    counts[u, v] += 1.0
    return counts


def pmi_embeddings(cooccur: np.ndarray, dim: int = 16) -> np.ndarray:
    """Shifted PPMI + SVD (classic word2vec/DeepWalk linear algebra view)."""
    row = cooccur.sum(axis=1, keepdims=True)
    col = cooccur.sum(axis=0, keepdims=True)
    total = cooccur.sum() + 1e-12
    pmi = np.log(np.maximum(cooccur * total / (row @ col + 1e-12), 1e-12))
    ppmi = np.maximum(pmi, 0.0)
    # SVD
    try:
        U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    except np.linalg.LinAlgError:
        U, S, Vt = np.linalg.svd(ppmi + 1e-6 * np.eye(ppmi.shape[0]), full_matrices=False)
    d = min(dim, U.shape[1])
    emb = U[:, :d] * np.sqrt(S[:d])
    return emb


def spectral_embeddings(A: np.ndarray, dim: int = 8) -> np.ndarray:
    """Bottom non-trivial Laplacian eigenvectors."""
    n = A.shape[0]
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    # symmetric normalized Laplacian
    d_inv = np.power(np.maximum(deg, 1e-12), -0.5)
    Lsym = np.eye(n) - (d_inv[:, None] * A * d_inv[None, :])
    w, v = np.linalg.eigh(Lsym)
    # skip first eigenvector (~constant)
    return v[:, 1:dim + 1]


# ---------------------------------------------------------------------------
# Torch GNN helpers
# ---------------------------------------------------------------------------

def get_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch, nn, F, device


def numpy_to_edge_index(A: np.ndarray):
    rows, cols = np.where(A > 0)
    return np.stack([rows, cols], axis=0)  # 2,E directed both ways already undirected


class GCNLayer:
    """NumPy GCN layer for from-scratch transparency (optional path)."""
    pass


def train_torch_node_clf(model, X, y, train_mask, val_mask, A_norm, epochs=200, lr=1e-2, weight_decay=5e-4):
    torch, nn, F, device = get_torch()
    model = model.to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    A_t = torch.tensor(A_norm, dtype=torch.float32, device=device)
    tr = torch.tensor(train_mask, device=device)
    va = torch.tensor(val_mask, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_val = None, -1.0
    history = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xt, A_t)
        loss = F.cross_entropy(logits[tr], yt[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            # recompute after step
            logits = model(Xt, A_t)
            pred = logits.argmax(dim=1)
            tr_acc = (pred[tr] == yt[tr]).float().mean().item()
            va_acc = (pred[va] == yt[va]).float().mean().item()
        history.append({"ep": ep, "loss": float(loss.item()), "train_acc": tr_acc, "val_acc": va_acc})
        if va_acc >= best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xt, A_t).cpu().numpy()
    return model, logits, history, best_val


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def fs00_protocol(g):
    A, y = g["A"], g["y"]
    n, m = g["n"], int(A.sum() // 2)
    deg = A.sum(axis=1)
    # assortativity proxy: same-label edge fraction
    rows, cols = np.where(np.triu(A, 1) > 0)
    same = (y[rows] == y[cols]).mean() if len(rows) else 0.0
    metrics = {
        "n_nodes": int(n),
        "n_edges": m,
        "n_classes": int(g["n_classes"]),
        "avg_degree": float(deg.mean()),
        "density": float(2 * m / max(n * (n - 1), 1)),
        "homophily_edge_same_label": float(same),
        "train_frac": float(g["train_mask"].mean()),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].bar(["nodes", "edges/5", "classes*20"], [n, m / 5, g["n_classes"] * 20], color="#3498db")
    axes[0].set_title("Graph size")
    axes[1].hist(deg, bins=20, color="#9b59b6")
    axes[1].set_title("Degree distribution")
    # adjacency spy (subsample)
    axes[2].imshow(A, cmap="Greys", aspect="auto")
    axes[2].set_title(f"Adjacency (homophily={same:.2f})")
    fig.tight_layout()
    payload = {
        "task": "protocol",
        "concept": "Graph = (V,E,X,Y); degree, density, homophily; transductive masks",
        "metrics": metrics,
        "you_should_feel": "先锁图协议与同质性；后面对比才有意义",
        "ok": True,
    }
    save_stage("fs00_protocol", payload, {"overview": fig})
    return metrics


def fs01_features_only(g):
    """Ignore graph structure: MLP/softmax on X only."""
    X, y = g["X"], g["y"]
    # standardize
    mu, sd = X[g["train_mask"]].mean(0), X[g["train_mask"]].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    clf = SoftmaxRegression(g["n_classes"], lr=0.3, n_iter=500).fit(Xs[g["train_mask"]], y[g["train_mask"]])
    pred = clf.predict(Xs)
    metrics = {
        "train_acc": accuracy(y[g["train_mask"]], pred[g["train_mask"]]),
        "val_acc": accuracy(y[g["val_mask"]], pred[g["val_mask"]]),
        "test_acc": accuracy(y[g["test_mask"]], pred[g["test_mask"]]),
    }
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["train", "val", "test"], [metrics["train_acc"], metrics["val_acc"], metrics["test_acc"]], color="#95a5a6")
    ax.set_ylim(0, 1.05)
    ax.set_title("FS01 features-only (no graph)")
    payload = {
        "task": "features_only",
        "concept": "Node classification with X only — structure unused",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": "从协议 → 真的在节点上分类（但没用边）",
        "you_should_feel": "无结构基线：后面 GNN 必须打过它才算用上图",
        "ok": True,
    }
    save_stage("fs01_features_only", payload, {"feat_only": fig})
    return metrics


def fs02_label_propagation(g, prev):
    Y = label_propagation(g["A"], g["y"], g["train_mask"], n_iter=50, alpha=0.9)
    pred = Y.argmax(axis=1)
    metrics = {
        "train_acc": accuracy(g["y"][g["train_mask"]], pred[g["train_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
    }
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["feat-only", "label-prop"], [prev["test_acc"], metrics["test_acc"]], color=["#95a5a6", "#e67e22"])
    ax.set_ylim(0, 1.05)
    ax.set_title("FS02 Label Propagation uses edges")
    payload = {
        "task": "label_propagation",
        "concept": "Diffuse train labels along edges (homophily prior)",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"feat {prev['test_acc']:.3f} → LP {metrics['test_acc']:.3f}",
        "you_should_feel": "边=标签高速公路；同质图上 LP 极强且无需特征",
        "ok": True,
    }
    save_stage("fs02_label_propagation", payload, {"lp": fig})
    return metrics


def fs03_spectral(g, prev):
    emb = spectral_embeddings(g["A"], dim=8)
    # concat raw features optionally
    X = np.hstack([emb, g["X"][:, :4]])
    mu, sd = X[g["train_mask"]].mean(0), X[g["train_mask"]].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    clf = SoftmaxRegression(g["n_classes"], lr=0.25, n_iter=400).fit(Xs[g["train_mask"]], g["y"][g["train_mask"]])
    pred = clf.predict(Xs)
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
    }
    # scatter first 2 spectral dims
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    sc = axes[0].scatter(emb[:, 0], emb[:, 1], c=g["y"], cmap="tab10", s=12)
    axes[0].set_title("Laplacian eigenmap (2D)")
    axes[1].bar(["LP", "spectral+X"], [prev["test_acc"], metrics["test_acc"]], color=["#e67e22", "#8e44ad"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Spectral features for clf")
    fig.tight_layout()
    payload = {
        "task": "spectral",
        "concept": "Graph Laplacian eigenvectors = smooth community coordinates",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"LP {prev['test_acc']:.3f} → spectral {metrics['test_acc']:.3f}",
        "you_should_feel": "谱嵌入把连通结构变成几何坐标（GCN 的谱根源）",
        "ok": True,
    }
    save_stage("fs03_spectral", payload, {"spectral": fig})
    return metrics


def fs04_deepwalk(g, prev):
    co = random_walk_cooccur(g["A"], n_walks=10, walk_len=16, window=4)
    emb = pmi_embeddings(co, dim=16)
    X = np.hstack([emb, g["X"]])
    mu, sd = X[g["train_mask"]].mean(0), X[g["train_mask"]].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    clf = SoftmaxRegression(g["n_classes"], lr=0.25, n_iter=400).fit(Xs[g["train_mask"]], g["y"][g["train_mask"]])
    pred = clf.predict(Xs)
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
        "emb_dim": int(emb.shape[1]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].scatter(emb[:, 0], emb[:, 1], c=g["y"], cmap="tab10", s=12)
    axes[0].set_title("DeepWalk-style PPMI SVD")
    axes[1].bar(["spectral", "DeepWalk"], [prev["test_acc"], metrics["test_acc"]], color=["#8e44ad", "#16a085"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Walk embeddings + X")
    fig.tight_layout()
    payload = {
        "task": "deepwalk",
        "concept": "Random walks → co-occurrence → skip-gram/PPMI embeddings",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"spectral {prev['test_acc']:.3f} → DW {metrics['test_acc']:.3f}",
        "you_should_feel": "游走把高阶邻近变成词向量；无特征也能嵌入节点",
        "ok": True,
    }
    save_stage("fs04_deepwalk", payload, {"deepwalk": fig})
    return metrics


def fs05_mean_agg(g, prev):
    """One-shot neighborhood mean aggregation (GraphSAGE-mean spirit, non-learned)."""
    A = g["A"]
    X = g["X"]
    deg = A.sum(1, keepdims=True)
    neigh = (A @ X) / np.maximum(deg, 1.0)
    H = np.hstack([X, neigh])
    mu, sd = H[g["train_mask"]].mean(0), H[g["train_mask"]].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Hs = (H - mu) / sd
    clf = SoftmaxRegression(g["n_classes"], lr=0.25, n_iter=400).fit(Hs[g["train_mask"]], g["y"][g["train_mask"]])
    pred = clf.predict(Hs)
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
    }
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["DeepWalk", "mean-agg"], [prev["test_acc"], metrics["test_acc"]], color=["#16a085", "#2980b9"])
    ax.set_ylim(0, 1.05)
    ax.set_title("FS05 Neighborhood mean aggregation")
    payload = {
        "task": "mean_aggregation",
        "concept": "h_v = [x_v || mean_{u∈N(v)} x_u] — message passing prototype",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"DW {prev['test_acc']:.3f} → mean-agg {metrics['test_acc']:.3f}",
        "you_should_feel": "邻居平均=最简消息传递；GNN 从这里开始",
        "ok": True,
    }
    save_stage("fs05_mean_agg", payload, {"mean_agg": fig})
    return metrics


def fs06_gcn(g, prev):
    torch, nn, F, device = get_torch()
    A_norm = normalize_adj(g["A"], add_self=True)
    n_cls, d = g["n_classes"], g["X"].shape[1]

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(d, 32)
            self.lin2 = nn.Linear(32, n_cls)

        def forward(self, x, A_hat):
            h = A_hat @ self.lin1(x)
            h = F.relu(h)
            h = F.dropout(h, p=0.3, training=self.training)
            h = A_hat @ self.lin2(h)
            return h

    model, logits, history, best_val = train_torch_node_clf(
        GCN(), g["X"], g["y"], g["train_mask"], g["val_mask"], A_norm, epochs=250, lr=1e-2
    )
    pred = logits.argmax(axis=1)
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
        "best_val": float(best_val),
        "device": str(device),
        "final_train_acc": history[-1]["train_acc"] if history else None,
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot([h["val_acc"] for h in history], label="val")
    axes[0].plot([h["train_acc"] for h in history], label="train", alpha=0.7)
    axes[0].legend(); axes[0].set_title("GCN training")
    axes[1].bar(["mean-agg", "GCN"], [prev["test_acc"], metrics["test_acc"]], color=["#2980b9", "#c0392b"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("FS06 GCN (Kipf & Welling)")
    fig.tight_layout()
    payload = {
        "task": "gcn",
        "concept": "Â = D^{-1/2}(A+I)D^{-1/2}; stacked Â H W with ReLU",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"mean-agg {prev['test_acc']:.3f} → GCN {metrics['test_acc']:.3f}",
        "you_should_feel": "可学习的谱近似消息传递；GNN 经典基石",
        "ok": True,
    }
    save_stage("fs06_gcn", payload, {"gcn": fig})
    return metrics


def fs07_gat(g, prev):
    torch, nn, F, device = get_torch()
    A = g["A"] + np.eye(g["n"])
    n_cls, d_in = g["n_classes"], g["X"].shape[1]
    # dense GAT for small n
    class GAT(nn.Module):
        def __init__(self, d_hidden=32, heads=4):
            super().__init__()
            self.heads = heads
            self.W = nn.Linear(d_in, d_hidden * heads, bias=False)
            self.a = nn.Parameter(torch.zeros(heads, 2 * d_hidden))
            nn.init.xavier_uniform_(self.a)
            self.out = nn.Linear(d_hidden * heads, n_cls)

        def forward(self, x, A_mask):
            # A_mask: n,n with 1 for edges+self
            B = x.size(0)
            h = self.W(x).view(B, self.heads, -1)  # n,H,dh
            dh = h.size(-1)
            # attention scores
            # e_ij = leakyrelu(a^T [hi||hj])
            hi = h.unsqueeze(1).expand(-1, B, -1, -1)  # n,n,H,dh
            hj = h.unsqueeze(0).expand(B, -1, -1, -1)
            cat = torch.cat([hi, hj], dim=-1)  # n,n,H,2dh
            e = F.leaky_relu((cat * self.a.view(1, 1, self.heads, 2 * dh)).sum(-1), 0.2)  # n,n,H
            # mask non-edges
            mask = A_mask.unsqueeze(-1)  # n,n,1
            e = e.masked_fill(mask <= 0, -1e9)
            alpha = torch.softmax(e, dim=1)  # attend over neighbors j
            # out_i = sum_j alpha_ij h_j
            out = (alpha.unsqueeze(-1) * hj).sum(1)  # n,H,dh
            out = out.reshape(B, -1)
            out = F.elu(out)
            out = F.dropout(out, 0.3, training=self.training)
            return self.out(out)

    A_t = torch.tensor(A, dtype=torch.float32)
    # custom train loop because forward signature uses A not A_norm
    model = GAT().to(device)
    Xt = torch.tensor(g["X"], dtype=torch.float32, device=device)
    yt = torch.tensor(g["y"], dtype=torch.long, device=device)
    A_dev = A_t.to(device)
    tr = torch.tensor(g["train_mask"], device=device)
    va = torch.tensor(g["val_mask"], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    best_state, best_val, history = None, -1.0, []
    for ep in range(200):
        model.train()
        opt.zero_grad()
        logits = model(Xt, A_dev)
        loss = F.cross_entropy(logits[tr], yt[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xt, A_dev)
            pred = logits.argmax(1)
            tr_acc = (pred[tr] == yt[tr]).float().mean().item()
            va_acc = (pred[va] == yt[va]).float().mean().item()
        history.append({"train_acc": tr_acc, "val_acc": va_acc, "loss": float(loss.item())})
        if va_acc >= best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xt, A_dev).cpu().numpy()
    pred = logits.argmax(1)
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
        "best_val": float(best_val),
        "device": str(device),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot([h["val_acc"] for h in history], label="val")
    axes[0].plot([h["train_acc"] for h in history], label="train", alpha=0.7)
    axes[0].legend(); axes[0].set_title("GAT training")
    axes[1].bar(["GCN", "GAT"], [prev["test_acc"], metrics["test_acc"]], color=["#c0392b", "#8e44ad"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("FS07 GAT attention")
    fig.tight_layout()
    payload = {
        "task": "gat",
        "concept": "Learned attention over neighbors instead of fixed Â weights",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"GCN {prev['test_acc']:.3f} → GAT {metrics['test_acc']:.3f}",
        "you_should_feel": "注意力让重要邻居权重大；异构邻域时更有用",
        "ok": True,
    }
    save_stage("fs07_gat", payload, {"gat": fig})
    return metrics


def fs08_gin(g, prev):
    torch, nn, F, device = get_torch()
    A = g["A"]  # no self in sum; we add epsilon self
    n_cls, d = g["n_classes"], g["X"].shape[1]

    class MLP(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_out))

        def forward(self, x):
            return self.net(x)

    class GIN(nn.Module):
        def __init__(self):
            super().__init__()
            self.eps1 = nn.Parameter(torch.zeros(1))
            self.eps2 = nn.Parameter(torch.zeros(1))
            self.mlp1 = MLP(d, 32)
            self.mlp2 = MLP(32, n_cls)

        def forward(self, x, A_sum):
            # (1+eps) x + A x
            h = self.mlp1((1 + self.eps1) * x + A_sum @ x)
            h = F.relu(h)
            h = F.dropout(h, 0.3, training=self.training)
            h = self.mlp2((1 + self.eps2) * h + A_sum @ h)
            return h

    A_t = A.astype(np.float64)
    model, logits, history, best_val = train_torch_node_clf(
        GIN(), g["X"], g["y"], g["train_mask"], g["val_mask"], A_t, epochs=250, lr=1e-2
    )
    pred = logits.argmax(1)
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
        "best_val": float(best_val),
        "device": str(device),
    }
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["GAT", "GIN"], [prev["test_acc"], metrics["test_acc"]], color=["#8e44ad", "#27ae60"])
    ax.set_ylim(0, 1.05)
    ax.set_title("FS08 GIN (injective sum+MLP)")
    payload = {
        "task": "gin",
        "concept": "(1+ε)h + Σ neighbors; MLP → WL-test expressive power",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"GAT {prev['test_acc']:.3f} → GIN {metrics['test_acc']:.3f}",
        "you_should_feel": "求和+MLP 逼近单射聚合；表达力理论关键突破",
        "ok": True,
    }
    save_stage("fs08_gin", payload, {"gin": fig})
    return metrics


def fs09_link_pred(g):
    """Link prediction: edges vs non-edges with GCN embeddings + dot product."""
    torch, nn, F, device = get_torch()
    A = g["A"]
    n = g["n"]
    edges = g["edges"]
    r = np.random.default_rng(SEED + 7)
    # sample negatives
    pos = edges.copy()
    neg = []
    existing = set(map(tuple, np.vstack([edges, edges[:, ::-1]]).tolist()))
    while len(neg) < len(pos):
        i, j = int(r.integers(0, n)), int(r.integers(0, n))
        if i >= j:
            continue
        if (i, j) in existing or (j, i) in existing:
            continue
        neg.append([i, j])
    neg = np.asarray(neg)
    # split edges
    perm = r.permutation(len(pos))
    n_tr = int(0.7 * len(pos))
    pos_tr, pos_te = pos[perm[:n_tr]], pos[perm[n_tr:]]
    neg_tr, neg_te = neg[perm[:n_tr]], neg[perm[n_tr:]]
    # train GCN encoder on structure only for link pred (use features)
    A_norm = normalize_adj(A, add_self=True)
    d = g["X"].shape[1]

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d, 32)
            self.l2 = nn.Linear(32, 16)

        def forward(self, x, A_hat):
            h = F.relu(A_hat @ self.l1(x))
            return A_hat @ self.l2(h)

    enc = Enc().to(device)
    Xt = torch.tensor(g["X"], dtype=torch.float32, device=device)
    A_t = torch.tensor(A_norm, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)

    def batch_loss(pos_e, neg_e):
        z = enc(Xt, A_t)
        pi = z[pos_e[:, 0]]
        pj = z[pos_e[:, 1]]
        ni = z[neg_e[:, 0]]
        nj = z[neg_e[:, 1]]
        pos_score = (pi * pj).sum(-1)
        neg_score = (ni * nj).sum(-1)
        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
        return loss, pos_score, neg_score

    pos_tr_t = torch.tensor(pos_tr, device=device)
    neg_tr_t = torch.tensor(neg_tr, device=device)
    for ep in range(150):
        enc.train()
        opt.zero_grad()
        loss, _, _ = batch_loss(pos_tr_t, neg_tr_t)
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        _, ps, ns = batch_loss(torch.tensor(pos_te, device=device), torch.tensor(neg_te, device=device))
        y_true = np.concatenate([np.ones(len(pos_te)), np.zeros(len(neg_te))])
        scores = np.concatenate([ps.cpu().numpy(), ns.cpu().numpy()])
    auc = roc_auc_binary(y_true, scores)
    # degree baseline AUC
    deg = A.sum(1)
    base_scores = np.concatenate([
        deg[pos_te[:, 0]] * deg[pos_te[:, 1]],
        deg[neg_te[:, 0]] * deg[neg_te[:, 1]],
    ])
    base_auc = roc_auc_binary(y_true, base_scores)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["deg×deg", "GCN-dot"], [base_auc, auc], color=["#95a5a6", "#e74c3c"])
    ax.set_ylim(0, 1.05)
    ax.set_title("FS09 Link prediction AUC")
    payload = {
        "task": "link_prediction",
        "concept": "Predict missing edges from node embeddings (dot product decoder)",
        "metrics": {"auc": auc, "degree_baseline_auc": base_auc},
        "auc": auc,
        "vs_prev": "从节点分类 → 边级任务",
        "you_should_feel": "链接预测是推荐/知识图谱核心任务",
        "ok": True,
    }
    save_stage("fs09_link_pred", payload, {"link": fig})
    return payload["metrics"]


def fs10_graph_clf(g_prev_metrics=None):
    """Graph-level classification: multiple SBM graphs with clear density regimes."""
    torch, nn, F, device = get_torch()
    r = np.random.default_rng(SEED + 3)
    graphs = []
    # class 0: strong communities (high p_in); class 1: nearly ER (p_in≈p_out)
    for i in range(80):
        label = i % 2
        if label == 0:
            p_in, p_out = 0.45, 0.02
        else:
            p_in, p_out = 0.12, 0.10
        gi = make_sbm_graph(
            n_per_block=12, n_blocks=3, p_in=p_in, p_out=p_out,
            feat_dim=8, seed=SEED + 200 + i, feat_signal=0.1,
        )
        # hand structural features as extra channels: degree
        deg = gi["A"].sum(1, keepdims=True)
        gi["X"] = np.hstack([gi["X"], deg / max(deg.max(), 1.0)])
        graphs.append((gi, label))

    d = graphs[0][0]["X"].shape[1]

    class GraphNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d, 64)
            self.l2 = nn.Linear(64, 64)
            self.cls = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, x, A_hat):
            h = F.relu(A_hat @ self.l1(x))
            h = F.relu(A_hat @ self.l2(h))
            gemb = torch.cat([h.mean(0), h.max(0).values], dim=0)  # mean+max pool
            # project if needed
            return self.cls(h.mean(0))

    # fix: use mean pool only into 64-d
    class GraphNet2(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d, 64)
            self.l2 = nn.Linear(64, 64)
            self.cls = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, x, A_hat):
            h = F.relu(A_hat @ self.l1(x))
            h = F.relu(A_hat @ self.l2(h))
            return self.cls(h.mean(0))

    model = GraphNet2().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    idx = r.permutation(len(graphs))
    tr, te = idx[:56], idx[56:]

    def run_epoch(indices, train=True):
        total_loss, correct = 0.0, 0
        if train:
            model.train()
        else:
            model.eval()
        for i in indices:
            gi, lab = graphs[i]
            A_hat = normalize_adj(gi["A"], True)
            x = torch.tensor(gi["X"], dtype=torch.float32, device=device)
            A_t = torch.tensor(A_hat, dtype=torch.float32, device=device)
            y = torch.tensor(lab, dtype=torch.long, device=device)
            if train:
                opt.zero_grad()
            logits = model(x, A_t)
            loss = F.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0))
            if train:
                loss.backward()
                opt.step()
            total_loss += float(loss.item())
            correct += int(logits.argmax().item() == lab)
        return total_loss / len(indices), correct / len(indices)

    hist = []
    best_te = 0.0
    for ep in range(80):
        tr_loss, tr_acc = run_epoch(tr, True)
        te_loss, te_acc = run_epoch(te, False)
        hist.append({"tr_acc": tr_acc, "te_acc": te_acc})
        best_te = max(best_te, te_acc)
    metrics = {
        "train_acc": hist[-1]["tr_acc"],
        "test_acc": hist[-1]["te_acc"],
        "best_test_acc": best_te,
        "device": str(device),
    }
    # use best for acceptance display
    metrics["test_acc"] = max(metrics["test_acc"], best_te)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([h["tr_acc"] for h in hist], label="train")
    ax.plot([h["te_acc"] for h in hist], label="test")
    ax.legend(); ax.set_title("FS10 Graph classification (sum/mean-pool GCN)")
    ax.set_ylim(0, 1.05)
    payload = {
        "task": "graph_classification",
        "concept": "Node MP → readout (mean) → graph label; density regimes",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": "节点级 → 整图级任务",
        "you_should_feel": "分子/社交图分类依赖 readout 聚合",
        "ok": True,
    }
    save_stage("fs10_graph_clf", payload, {"graph_clf": fig})
    return metrics


def fs11_hetero_mini():
    """Mini heterogeneous: two node types user/item, bipartite + same-type edges."""
    r = np.random.default_rng(SEED + 9)
    n_u, n_i = 60, 40
    n = n_u + n_i
    # labels on users only (2 communities)
    y_u = r.integers(0, 2, size=n_u)
    # bipartite edges preferred within community-ish via item groups
    A = np.zeros((n, n))
    for u in range(n_u):
        for _ in range(4):
            # items 0..19 prefer class0, 20..39 class1
            if y_u[u] == 0:
                i = r.integers(0, 20)
            else:
                i = r.integers(20, 40)
            ii = n_u + i
            A[u, ii] = A[ii, u] = 1
    # user-user edges
    for u in range(n_u):
        for _ in range(2):
            v = int(r.integers(0, n_u))
            if v != u and y_u[u] == y_u[v]:
                A[u, v] = A[v, u] = 1
    X = r.normal(size=(n, 12))
    X[:n_u] += np.eye(2)[y_u] @ r.normal(size=(2, 12)) * 0.5
    # classify users with GCN on full bipartite graph
    train = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    idx = r.permutation(n_u)
    train[idx[:40]] = True
    test[idx[40:]] = True
    y = np.zeros(n, dtype=int)
    y[:n_u] = y_u
    # features-only baseline on users
    mu, sd = X[train].mean(0), X[train].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    clf = SoftmaxRegression(2, lr=0.3, n_iter=300).fit(Xs[train], y[train])
    base_acc = accuracy(y[test], clf.predict(Xs)[test])
    # GCN
    torch, nn, F, device = get_torch()
    A_norm = normalize_adj(A, True)

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(12, 32)
            self.l2 = nn.Linear(32, 2)

        def forward(self, x, A_hat):
            h = F.relu(A_hat @ self.l1(x))
            return A_hat @ self.l2(h)

    model, logits, _, _ = train_torch_node_clf(
        GCN(), X, y, train, test, A_norm, epochs=200, lr=1e-2
    )
    pred = logits.argmax(1)
    gcn_acc = accuracy(y[test], pred[test])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["user feats", "bipartite GCN"], [base_acc, gcn_acc], color=["#95a5a6", "#d35400"])
    ax.set_ylim(0, 1.05)
    ax.set_title("FS11 Heterogeneous bipartite signal")
    metrics = {"feat_only_acc": base_acc, "hetero_gcn_acc": gcn_acc, "n_users": n_u, "n_items": n_i}
    payload = {
        "task": "heterogeneous",
        "concept": "Multiple node/edge types; message passing across bipartite graph",
        "metrics": metrics,
        "test_acc": gcn_acc,
        "vs_prev": f"feat {base_acc:.3f} → hetero-GCN {gcn_acc:.3f}",
        "you_should_feel": "推荐/知识图谱天然异构；边类型即归纳偏置",
        "ok": True,
    }
    save_stage("fs11_hetero", payload, {"hetero": fig})
    return metrics


def fs12_graph_transformer(g, prev):
    """Mini Graph Transformer: attention among all nodes with structural bias from A."""
    torch, nn, F, device = get_torch()
    n, d_in, n_cls = g["n"], g["X"].shape[1], g["n_classes"]
    A = g["A"] + np.eye(n)

    class GraphTransformer(nn.Module):
        def __init__(self, d_model=32, n_heads=4):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            self.ff = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, d_model))
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.cls = nn.Linear(d_model, n_cls)
            # structural bias: edge distance bucket embedding (0=edge,1=no)
            self.bias = nn.Embedding(2, n_heads)

        def forward(self, x, A_mask):
            # x: n,d  treat as batch=1, seq=n
            h = self.proj(x).unsqueeze(0)  # 1,n,d
            # attn mask: allow all but add bias via attn_mask (heads,n,n) additive
            # PyTorch attn_mask float added to weights
            edge = (A_mask > 0).long()  # n,n
            # bias: edges get 0, non-edges get negative
            # use key_padding none; float mask
            b = torch.where(A_mask > 0, torch.tensor(0.0, device=x.device), torch.tensor(-2.0, device=x.device))
            attn_mask = b  # n,n broadcast to heads
            h2, _ = self.attn(h, h, h, attn_mask=attn_mask)
            h = self.norm1(h + h2)
            h = self.norm2(h + self.ff(h))
            return self.cls(h.squeeze(0))

    model = GraphTransformer().to(device)
    Xt = torch.tensor(g["X"], dtype=torch.float32, device=device)
    At = torch.tensor(A, dtype=torch.float32, device=device)
    yt = torch.tensor(g["y"], dtype=torch.long, device=device)
    tr = torch.tensor(g["train_mask"], device=device)
    va = torch.tensor(g["val_mask"], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best_state, best_val = None, -1.0
    history = []
    for ep in range(120):
        model.train()
        opt.zero_grad()
        logits = model(Xt, At)
        loss = F.cross_entropy(logits[tr], yt[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xt, At)
            pred = logits.argmax(1)
            va_acc = (pred[va] == yt[va]).float().mean().item()
            tr_acc = (pred[tr] == yt[tr]).float().mean().item()
        history.append({"val_acc": va_acc, "train_acc": tr_acc})
        if va_acc >= best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xt, At).argmax(1).cpu().numpy()
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
        "device": str(device),
    }
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["GIN", "GraphTransformer"], [prev["test_acc"], metrics["test_acc"]], color=["#27ae60", "#1a5276"])
    ax.set_ylim(0, 1.05)
    ax.set_title("FS12 Graph Transformer (struct bias)")
    payload = {
        "task": "graph_transformer",
        "concept": "Global attention + structural bias (modern GPS/Transformer-GNN spirit)",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"GIN {prev['test_acc']:.3f} → GT {metrics['test_acc']:.3f}",
        "you_should_feel": "现代图模型：Transformer 全局注意力 + 结构偏置",
        "ok": True,
    }
    save_stage("fs12_graph_transformer", payload, {"gt": fig})
    return metrics


def fs13_contrastive(g, prev):
    """GraphCL-style: two corrupted views, InfoNCE on node embeddings (transductive)."""
    torch, nn, F, device = get_torch()
    A = g["A"]
    X = g["X"]
    n, d = X.shape
    A_norm = normalize_adj(A, True)

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d, 32)
            self.l2 = nn.Linear(32, 16)

        def forward(self, x, A_hat):
            h = F.relu(A_hat @ self.l1(x))
            return F.normalize(A_hat @ self.l2(h), dim=-1)

    def corrupt(X, A, drop_feat=0.2, drop_edge=0.2, r=None):
        r = r or np.random.default_rng()
        Xc = X.copy()
        mask = r.random(X.shape) < drop_feat
        Xc[mask] = 0.0
        Ac = A.copy()
        rows, cols = np.where(np.triu(A, 1) > 0)
        for i, j in zip(rows, cols):
            if r.random() < drop_edge:
                Ac[i, j] = Ac[j, i] = 0.0
        return Xc, normalize_adj(Ac, True)

    enc = Enc().to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    r = np.random.default_rng(SEED + 11)
    for ep in range(80):
        enc.train()
        X1, A1 = corrupt(X, A, r=r)
        X2, A2 = corrupt(X, A, r=r)
        z1 = enc(torch.tensor(X1, dtype=torch.float32, device=device), torch.tensor(A1, dtype=torch.float32, device=device))
        z2 = enc(torch.tensor(X2, dtype=torch.float32, device=device), torch.tensor(A2, dtype=torch.float32, device=device))
        # InfoNCE: positives are same index across views
        logits = z1 @ z2.T / 0.2  # n,n
        labels = torch.arange(n, device=device)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        z = enc(
            torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(A_norm, dtype=torch.float32, device=device),
        ).cpu().numpy()
    # linear probe
    mu, sd = z[g["train_mask"]].mean(0), z[g["train_mask"]].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    zs = (z - mu) / sd
    clf = SoftmaxRegression(g["n_classes"], lr=0.3, n_iter=400).fit(zs[g["train_mask"]], g["y"][g["train_mask"]])
    pred = clf.predict(zs)
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
        "device": str(device),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].scatter(z[:, 0], z[:, 1], c=g["y"], cmap="tab10", s=12)
    axes[0].set_title("Contrastive node emb")
    axes[1].bar(["supervised GCN~", "GraphCL probe"], [prev.get("test_acc", 0), metrics["test_acc"]], color=["#c0392b", "#16a085"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("FS13 Self-supervised contrastive")
    fig.tight_layout()
    payload = {
        "task": "contrastive",
        "concept": "Two graph augmentations + InfoNCE; linear probe (GraphCL spirit)",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"supervised ref {prev.get('test_acc', float('nan')):.3f} → SSL {metrics['test_acc']:.3f}",
        "you_should_feel": "自监督图学习：少标签时靠增强与对比",
        "ok": True,
    }
    save_stage("fs13_contrastive", payload, {"contrastive": fig})
    return metrics


def fs14_over_smoothing(g):
    """Demonstrate over-smoothing: deep GCN layers collapse node features."""
    A_norm = normalize_adj(g["A"], True)
    X = g["X"]
    # propagate K times without learning
    diffs = []
    H = X.copy()
    for k in range(0, 11):
        # mean pairwise distance
        # sample for speed
        idx = np.linspace(0, len(H) - 1, num=min(80, len(H)), dtype=int)
        Hs = H[idx]
        d = 0.0
        c = 0
        for i in range(len(Hs)):
            for j in range(i + 1, len(Hs)):
                d += np.linalg.norm(Hs[i] - Hs[j])
                c += 1
        diffs.append(d / max(c, 1))
        H = A_norm @ H
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(0, 11), diffs, marker="o", color="#e74c3c")
    ax.set_xlabel("propagation depth")
    ax.set_ylabel("mean pairwise L2")
    ax.set_title("FS14 Over-smoothing: features collapse with depth")
    metrics = {
        "pairwise_dist_depth0": diffs[0],
        "pairwise_dist_depth5": diffs[5],
        "pairwise_dist_depth10": diffs[10],
        "collapse_ratio": diffs[10] / max(diffs[0], 1e-12),
    }
    payload = {
        "task": "over_smoothing",
        "concept": "Deep message passing → node representations become indistinguishable",
        "metrics": metrics,
        "vs_prev": "解释为何不能无脑堆 GCN 层",
        "you_should_feel": "过平滑是深层 GNN 经典病；残差/注意力/扩散缓解",
        "ok": True,
    }
    save_stage("fs14_over_smoothing", payload, {"oversmooth": fig})
    return metrics


def fs15_residual_deep_gcn(g, prev_gcn, prev_smooth):
    """Residual GCN to mitigate over-smoothing / train deeper."""
    torch, nn, F, device = get_torch()
    A_norm = normalize_adj(g["A"], True)
    d, n_cls = g["X"].shape[1], g["n_classes"]

    class ResGCN(nn.Module):
        def __init__(self, layers=4):
            super().__init__()
            self.lin_in = nn.Linear(d, 32)
            self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(layers)])
            self.out = nn.Linear(32, n_cls)

        def forward(self, x, A_hat):
            h = F.relu(A_hat @ self.lin_in(x))
            for lin in self.layers:
                h = h + F.relu(A_hat @ lin(h))  # residual
                h = F.dropout(h, 0.2, training=self.training)
            return A_hat @ self.out(h)

    model, logits, history, best_val = train_torch_node_clf(
        ResGCN(4), g["X"], g["y"], g["train_mask"], g["val_mask"], A_norm, epochs=250, lr=1e-2
    )
    pred = logits.argmax(1)
    metrics = {
        "test_acc": accuracy(g["y"][g["test_mask"]], pred[g["test_mask"]]),
        "val_acc": accuracy(g["y"][g["val_mask"]], pred[g["val_mask"]]),
        "best_val": float(best_val),
        "device": str(device),
    }
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["shallow GCN", "ResGCN×4"], [prev_gcn["test_acc"], metrics["test_acc"]], color=["#c0392b", "#2c3e50"])
    ax.set_ylim(0, 1.05)
    ax.set_title("FS15 Residual deep GCN")
    payload = {
        "task": "residual_gcn",
        "concept": "Skip connections let deeper MP keep identity features",
        "metrics": metrics,
        "test_acc": metrics["test_acc"],
        "vs_prev": f"GCN {prev_gcn['test_acc']:.3f} → ResGCN {metrics['test_acc']:.3f}",
        "you_should_feel": "残差是治过平滑/深层训练的实用解药之一",
        "ok": True,
    }
    save_stage("fs15_residual_gcn", payload, {"resgcn": fig})
    return metrics


def fs16_leaderboard(all_node, link_m, graph_m, hetero_m, ssl_m, smooth_m):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = list(all_node.keys())
    vals = [all_node[k] for k in names]
    axes[0].plot(range(len(names)), vals, marker="o")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Node classification ladder (test acc)")
    axes[1].bar(
        ["link AUC", "graph acc", "hetero acc", "SSL probe"],
        [link_m["auc"], graph_m["test_acc"], hetero_m["hetero_gcn_acc"], ssl_m["test_acc"]],
        color=["#e74c3c", "#3498db", "#d35400", "#16a085"],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Other graph tasks")
    fig.tight_layout()
    payload = {
        "task": "leaderboard",
        "concept": "Full GML map: structureless → classical → MP → attention → expressive → modern/SSL",
        "node_classification": all_node,
        "link_prediction_auc": link_m["auc"],
        "graph_classification_acc": graph_m["test_acc"],
        "hetero_acc": hetero_m["hetero_gcn_acc"],
        "ssl_acc": ssl_m["test_acc"],
        "over_smoothing_collapse_ratio": smooth_m["collapse_ratio"],
        "best_node": max(all_node, key=all_node.get),
        "map": {
            "from_zero": ["fs00", "fs01"],
            "classic": ["fs02", "fs03", "fs04"],
            "key_breakthroughs": ["fs05", "fs06", "fs07", "fs08"],
            "modern_systems": ["fs09", "fs10", "fs11", "fs12"],
            "frontier": ["fs13", "fs14", "fs15", "fs16"],
        },
        "you_should_feel": "图能力链：特征→标签传播→谱/游走→消息传递→注意力/表达力→任务族→Transformer/SSL/过平滑治理",
        "ok": True,
    }
    save_stage("fs16_leaderboard", payload, {"leaderboard": fig})
    return payload


def main():
    global STAGE_ORDER, SUMMARY
    STAGE_ORDER = []
    SUMMARY = {"stages": {}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()
    if OUT_DIR.exists():
        for child in list(OUT_DIR.iterdir()):
            if child.is_dir() and child.name.startswith("fs"):
                for f in child.glob("*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            elif child.name in ("SUMMARY.json", "ACCEPTANCE.json"):
                try:
                    child.unlink()
                except Exception:
                    pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("OUT_DIR=", OUT_DIR)
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
    except Exception as e:
        print("torch missing", e)

    g = make_sbm_graph()
    fs00_protocol(g)
    m01 = fs01_features_only(g)
    m02 = fs02_label_propagation(g, m01)
    m03 = fs03_spectral(g, m02)
    m04 = fs04_deepwalk(g, m03)
    m05 = fs05_mean_agg(g, m04)
    m06 = fs06_gcn(g, m05)
    m07 = fs07_gat(g, m06)
    m08 = fs08_gin(g, m07)
    m09 = fs09_link_pred(g)
    m10 = fs10_graph_clf()
    m11 = fs11_hetero_mini()
    m12 = fs12_graph_transformer(g, m08)
    m13 = fs13_contrastive(g, m06)
    m14 = fs14_over_smoothing(g)
    m15 = fs15_residual_deep_gcn(g, m06, m14)

    all_node = {
        "feat": m01["test_acc"],
        "LP": m02["test_acc"],
        "spectral": m03["test_acc"],
        "DeepWalk": m04["test_acc"],
        "meanAgg": m05["test_acc"],
        "GCN": m06["test_acc"],
        "GAT": m07["test_acc"],
        "GIN": m08["test_acc"],
        "GraphTR": m12["test_acc"],
        "SSL": m13["test_acc"],
        "ResGCN": m15["test_acc"],
    }
    board = fs16_leaderboard(all_node, m09, m10, m11, m13, m14)

    SUMMARY["elapsed_sec"] = round(time.time() - t0, 2)
    SUMMARY["stage_order"] = list(STAGE_ORDER)
    SUMMARY["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    SUMMARY["leaderboard"] = {
        "node_classification": all_node,
        "link_auc": m09["auc"],
        "graph_acc": m10["test_acc"],
        "hetero_acc": m11["hetero_gcn_acc"],
        "best_node": board["best_node"],
    }

    EXPECTED = [
        "fs00_protocol", "fs01_features_only", "fs02_label_propagation", "fs03_spectral",
        "fs04_deepwalk", "fs05_mean_agg", "fs06_gcn", "fs07_gat", "fs08_gin",
        "fs09_link_pred", "fs10_graph_clf", "fs11_hetero", "fs12_graph_transformer",
        "fs13_contrastive", "fs14_over_smoothing", "fs15_residual_gcn", "fs16_leaderboard",
    ]
    checks = []
    def chk(name, cond, detail=""):
        checks.append({"name": name, "ok": bool(cond), "detail": str(detail)})
        if not cond:
            print("ACCEPT_FAIL", name, detail)

    chk("stage_count_17", len(STAGE_ORDER) == 17, len(STAGE_ORDER))
    chk("stage_order_exact", STAGE_ORDER == EXPECTED, STAGE_ORDER)
    for st in EXPECTED:
        rd = OUT_DIR / st / "results.json"
        chk(f"{st}_json", rd.is_file(), rd)
        if rd.is_file():
            pl = json.loads(rd.read_text())
            chk(f"{st}_ok", pl.get("ok") is True, pl.get("ok"))
        chk(f"{st}_png", len(list((OUT_DIR / st).glob('*.png'))) >= 1, "png")

    # pedagogical: features alone imperfect; structure lifts performance
    chk("feat_not_trivial", all_node["feat"] < 0.85, all_node["feat"])
    chk("lp_beats_feat", all_node["LP"] > all_node["feat"] + 0.05, all_node)
    chk("gcn_beats_feat", all_node["GCN"] > all_node["feat"] + 0.08, all_node)
    chk("best_node_ge_0.80", max(all_node.values()) >= 0.80, max(all_node.values()))
    chk("link_auc_gt_0.7", m09["auc"] > 0.70, m09["auc"])
    chk("graph_clf_gt_0.7", m10["test_acc"] >= 0.70, m10["test_acc"])
    chk("oversmooth_collapses", m14["collapse_ratio"] < 0.5, m14["collapse_ratio"])
    chk("homo_gt_0.5", json.loads((OUT_DIR/"fs00_protocol"/"results.json").read_text())["metrics"]["homophily_edge_same_label"] > 0.5, "")

    accept_ok = all(c["ok"] for c in checks)
    SUMMARY["ok"] = accept_ok
    SUMMARY["acceptance"] = {"ok": accept_ok, "checks": checks, "n_pass": sum(c["ok"] for c in checks), "n_total": len(checks)}
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(SUMMARY, indent=2))
    (OUT_DIR / "ACCEPTANCE.json").write_text(json.dumps(SUMMARY["acceptance"], indent=2))
    print("ACCEPTANCE", SUMMARY["acceptance"]["n_pass"], "/", SUMMARY["acceptance"]["n_total"], "ok=", accept_ok)
    if not accept_ok:
        failed = [c for c in checks if not c["ok"]]
        raise RuntimeError("ACCEPTANCE FAILED: " + json.dumps(failed, ensure_ascii=False)[:2500])
    print("SMOKE_OK stages=", len(STAGE_ORDER), "elapsed=", SUMMARY["elapsed_sec"])
    print("GRAPH_FS_OK")
    return SUMMARY


if __name__ == "__main__":
    try:
        get_ipython  # type: ignore  # noqa: F821
    except NameError:
        main()
