"""k-NN / nearest-centroid evaluation of frozen features (no training of any kind).

Vendored private copy of the protocol for the video benchmark.  `scripts/knn_eval.py` is owned by
the image-accuracy agent and did not exist when this landed; `scripts/bench_accuracy_video.py`
imports that module when it is present and exposes the same three entry points, and falls back to
this file otherwise, so the two benchmarks share one protocol at merge time.  Keep the two in sync.

Protocol (identical for every backend and dtype, so the only variable is the feature vector):
  * features are L2-normalized, similarity = cosine = dot product of the normalized vectors;
  * k-NN: the k = 20 most similar gallery clips vote for their class with weight exp(sim / 0.07)
    (the DINO weighting, Caron et al. 2021, `T = 0.07`); the class with the largest total weight
    wins, ties broken by the smaller class id;
  * nearest-class-centroid: the gallery mean per class (of the normalized vectors), re-normalized;
    the query is assigned the most similar centroid.  No hyper-parameters at all.
Both are pure look-ups over frozen features — nothing is fitted.
"""
from __future__ import annotations

import numpy as np

K_DEFAULT = 20
T_DEFAULT = 0.07


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def knn_predict(gallery: np.ndarray, gallery_labels: np.ndarray, query: np.ndarray,
                n_classes: int, k: int = K_DEFAULT, temperature: float = T_DEFAULT) -> np.ndarray:
    """DINO-style weighted k-NN top-1 prediction for every query row."""
    g = l2_normalize(gallery)
    q = l2_normalize(query)
    sim = q @ g.T                                              # [n_query, n_gallery] cosine
    k = min(k, g.shape[0])
    nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]           # k best, unordered
    nn_sim = np.take_along_axis(sim, nn, axis=1)
    w = np.exp(nn_sim / temperature)
    votes = np.zeros((q.shape[0], n_classes), dtype=np.float64)
    lab = gallery_labels[nn]                                   # [n_query, k]
    for j in range(k):
        np.add.at(votes, (np.arange(q.shape[0]), lab[:, j]), w[:, j])
    return votes.argmax(1)


def centroid_predict(gallery: np.ndarray, gallery_labels: np.ndarray, query: np.ndarray,
                     n_classes: int) -> np.ndarray:
    """Nearest class centroid (parameter-free) top-1 prediction for every query row."""
    g = l2_normalize(gallery)
    q = l2_normalize(query)
    cent = np.stack([g[gallery_labels == c].mean(0) if np.any(gallery_labels == c)
                     else np.zeros(g.shape[1]) for c in range(n_classes)])
    return (q @ l2_normalize(cent).T).argmax(1)


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(labels)).mean())


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row cosine between two feature matrices of the same shape."""
    return float((l2_normalize(a) * l2_normalize(b)).sum(-1).mean())
