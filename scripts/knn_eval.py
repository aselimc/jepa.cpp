#!/usr/bin/env python3
"""k-NN / nearest-centroid evaluation of frozen features (no training of any kind).

The protocol is fixed here once and used unchanged for every backend (PyTorch, jepa.cpp f32/f16/
q8_0/q4_k), so a number only ever moves because the *features* moved:

  1. L2-normalise every feature vector.
  2. cosine similarity between each query and every gallery vector (a plain dot product after 1.).
  3. k = 20 nearest gallery vectors, DINO-style weighted vote: each neighbour contributes
     exp(sim / 0.07) to its class; the arg-max class is the prediction  ->  "kNN top-1".
  4. nearest class centroid: mean of the L2-normalised gallery vectors of a class, re-normalised;
     the arg-max cosine class is the prediction  ->  "centroid top-1" (parameter-free second number).

Two cross-backend numbers, both against the PyTorch run of the same model and feature:

  * agreement   = fraction of query items whose kNN prediction equals PyTorch's (not "both right" --
                  a shared mistake counts as agreement),
  * feat cosine = mean over query items of cos(feature_backend, feature_pytorch) *before*
                  L2-normalisation matters (cosine is scale free), plus the worst item.

Everything is deterministic: no sampling, no ties broken at random (numpy argmax takes the lowest
class index on an exact tie, which never happened in the reported runs).

CLI (features are float32 .npy of shape [n, D], labels are int .npy or a JSON list of ints):

    scripts/knn_eval.py --gallery g.npy --gallery-labels gl.json \
                        --query   q.npy --query-labels   ql.json \
                        [--ref-query r.npy] [--ref-preds p.json] [--k 20] [--temp 0.07] [--json out.json]

Importable API: `evaluate(gallery, gallery_labels, query, query_labels, ...) -> dict`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_K = 20
DEFAULT_TEMP = 0.07


def l2norm(x: np.ndarray) -> np.ndarray:
    """Rows to unit L2 norm (float32 in, float32 out); a zero row stays zero."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def knn_predict(gallery: np.ndarray, gallery_labels: np.ndarray, query: np.ndarray,
                n_classes: int, k: int = DEFAULT_K, temp: float = DEFAULT_TEMP,
                chunk: int = 1024) -> np.ndarray:
    """DINO-style weighted k-NN vote.  Returns the predicted class of every query row."""
    g = l2norm(gallery)
    q = l2norm(query)
    gl = np.asarray(gallery_labels, dtype=np.int64)
    k = min(k, g.shape[0])
    out = np.empty(q.shape[0], dtype=np.int64)
    for s in range(0, q.shape[0], chunk):
        sims = q[s:s + chunk] @ g.T                                  # [b, n_gallery]
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]           # k nearest (unordered)
        top = np.take_along_axis(sims, idx, axis=1).astype(np.float64)
        w = np.exp(top / temp)                                       # DINO weight
        lab = gl[idx]                                                # [b, k]
        votes = np.zeros((top.shape[0], n_classes), dtype=np.float64)
        np.add.at(votes, (np.arange(top.shape[0])[:, None], lab), w)
        out[s:s + chunk] = votes.argmax(1)
    return out


def centroid_predict(gallery: np.ndarray, gallery_labels: np.ndarray, query: np.ndarray,
                     n_classes: int) -> np.ndarray:
    """Nearest class centroid of the L2-normalised gallery features (centroids re-normalised)."""
    g = l2norm(gallery)
    q = l2norm(query)
    gl = np.asarray(gallery_labels, dtype=np.int64)
    cent = np.zeros((n_classes, g.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = gl == c
        if m.any():
            cent[c] = g[m].mean(0)
    cent = l2norm(cent)
    return (q @ cent.T).argmax(1).astype(np.int64)


def nn_margin(gallery: np.ndarray, gallery_labels, query: np.ndarray, n_classes: int,
              chunk: int = 1024) -> np.ndarray:
    """How decided a query item is, independent of any backend: the cosine to its best gallery
    neighbour minus the cosine to the best neighbour of any *other* class.  Near zero = the two
    classes are tied at the top and a tiny feature perturbation can flip the vote."""
    g = l2norm(gallery)
    q = l2norm(query)
    gl = np.asarray(gallery_labels, dtype=np.int64)
    out = np.empty(q.shape[0], dtype=np.float64)
    for s in range(0, q.shape[0], chunk):
        sims = q[s:s + chunk] @ g.T
        per_class = np.full((sims.shape[0], n_classes), -2.0, dtype=np.float64)
        for c in range(n_classes):
            m = gl == c
            if m.any():
                per_class[:, c] = sims[:, m].max(1)
        srt = np.sort(per_class, axis=1)
        out[s:s + chunk] = srt[:, -1] - srt[:, -2]
    return out


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two equally-shaped feature matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-12)


def evaluate(gallery: np.ndarray, gallery_labels, query: np.ndarray, query_labels,
             n_classes: int, k: int = DEFAULT_K, temp: float = DEFAULT_TEMP,
             ref_query: np.ndarray | None = None, ref_preds=None,
             margin: np.ndarray | None = None) -> dict:
    """Full protocol for one (backend, dtype, feature) run.  Returns a JSON-ready dict."""
    ql = np.asarray(query_labels, dtype=np.int64)
    preds = knn_predict(gallery, gallery_labels, query, n_classes, k, temp)
    cpreds = centroid_predict(gallery, gallery_labels, query, n_classes)
    res = {
        "n_gallery": int(np.asarray(gallery).shape[0]),
        "n_query": int(np.asarray(query).shape[0]),
        "dim": int(np.asarray(query).shape[1]),
        "k": k, "temp": temp,
        "knn_top1": float((preds == ql).mean()),
        "centroid_top1": float((cpreds == ql).mean()),
        "knn_preds": preds.tolist(),
    }
    if ref_query is not None:
        cos = pairwise_cosine(query, ref_query)
        res["feat_cos_mean"] = float(cos.mean())
        res["feat_cos_min"] = float(cos.min())
    if ref_preds is not None:
        rp = np.asarray(ref_preds, dtype=np.int64)
        flip = preds != rp
        res["agreement"] = float((preds == rp).mean())
        res["n_flipped"] = int(flip.sum())
        res["flip_ref_right"] = int((rp[flip] == ql[flip]).sum())
        res["flip_this_right"] = int((preds[flip] == ql[flip]).sum())
        res["flip_both_wrong"] = res["n_flipped"] - res["flip_ref_right"] - res["flip_this_right"]
        if margin is not None and flip.any():
            res["margin_flipped_median"] = float(np.median(np.asarray(margin)[flip]))
    if margin is not None:
        res["margin_all_median"] = float(np.median(np.asarray(margin)))
    return res


def _load_labels(p: str) -> np.ndarray:
    path = Path(p)
    if path.suffix == ".json":
        v = json.loads(path.read_text())
        if isinstance(v, dict):
            v = v["labels"]
        return np.asarray(v, dtype=np.int64)
    return np.load(path).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--gallery-labels", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--query-labels", required=True)
    ap.add_argument("--ref-query", help="query features of the reference backend (mean feature cosine)")
    ap.add_argument("--ref-preds", help="JSON list of the reference backend's kNN predictions (agreement)")
    ap.add_argument("--n-classes", type=int, default=0, help="0 = max(label)+1")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--json", help="write the result dict here")
    a = ap.parse_args()

    g, q = np.load(a.gallery), np.load(a.query)
    gl, ql = _load_labels(a.gallery_labels), _load_labels(a.query_labels)
    n_classes = a.n_classes or int(max(gl.max(), ql.max()) + 1)
    ref_q = np.load(a.ref_query) if a.ref_query else None
    ref_p = json.loads(Path(a.ref_preds).read_text()) if a.ref_preds else None
    if isinstance(ref_p, dict):
        ref_p = ref_p["knn_preds"]
    res = evaluate(g, gl, q, ql, n_classes, a.k, a.temp, ref_q, ref_p)
    show = {k: v for k, v in res.items() if k != "knn_preds"}
    print(json.dumps(show, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
