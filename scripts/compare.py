#!/usr/bin/env python3
"""Compare two .npy tensors, or two reference directories written by scripts/dump_reference.py.

    scripts/compare.py A.npy B.npy [--topk 5]
    scripts/compare.py tests/fixtures/ref/ijepa-vith14-1k  build/out/ijepa   [--tensors last_hidden_state,pooled_mean]

A is the candidate (e.g. jepa.cpp output), B is the reference.  For directories, samples are matched by name through
manifest.json (files `<sample>.<tensor>.npy`); a directory without a manifest is matched by file name.  Reported per
tensor: cosine similarity per token (mean and worst over rows of the last axis), max abs error, relative errors
(max|a-b| / max|b| and ||a-b|| / ||b||), and for logits the top-1 / top-k agreement.  Exit status is non-zero when
any compared tensor violates a threshold (--min-cos, --max-abs, --max-rel, --min-top1, --min-topk).

Importable: compare_arrays(a, b, topk) -> dict of metrics; compare_dirs(A, B, ...) -> list of rows.
Only numpy is needed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

DEFAULT_MIN_COS = 0.9999
DEFAULT_MAX_REL = 1e-3
SKIP_DEFAULT = ("frames_u8",)  # uint8 inputs only make sense with an exact check


def _rows(x: np.ndarray) -> np.ndarray:
    """View as [N, D] with D = last axis (a 1-D tensor is one row)."""
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, x.shape[-1]) if x.ndim >= 1 else x.reshape(1, 1)


def compare_arrays(a, b, topk: int = 5, is_logits: bool | None = None) -> dict:
    a = np.asarray(a)
    b = np.asarray(b)
    m: dict = {"shape_a": list(a.shape), "shape_b": list(b.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}
    if a.shape != b.shape:
        if a.size == b.size:
            m["note"] = f"reshaped {list(a.shape)} -> {list(b.shape)}"
            a = a.reshape(b.shape)
        else:
            m["error"] = "shape mismatch"
            return m
    if a.dtype.kind in "iu" and b.dtype.kind in "iu":  # index tensors (top5_idx): exact / set agreement
        m["exact"] = bool(np.array_equal(a, b))
        m["set_overlap"] = float(len(set(a.ravel().tolist()) & set(b.ravel().tolist())) / max(1, b.size))
        return m
    af, bf = a.astype(np.float64), b.astype(np.float64)
    if not (np.isfinite(af).all() and np.isfinite(bf).all()):
        m["error"] = "non-finite values"
        m["nan_a"], m["nan_b"] = int((~np.isfinite(af)).sum()), int((~np.isfinite(bf)).sum())
        return m
    diff = af - bf
    ra, rb = _rows(af), _rows(bf)
    na, nb = np.linalg.norm(ra, axis=1), np.linalg.norm(rb, axis=1)
    denom = na * nb
    cos = np.where(denom > 0, (ra * rb).sum(1) / np.where(denom > 0, denom, 1.0), np.where((na == 0) & (nb == 0), 1.0, 0.0))
    m["n_rows"] = int(ra.shape[0])
    m["cos_mean"] = float(cos.mean())
    m["cos_min"] = float(cos.min())
    m["cos_min_row"] = int(cos.argmin())
    m["max_abs"] = float(np.abs(diff).max())
    m["max_abs_ref"] = float(np.abs(bf).max())
    m["rel_max_abs"] = float(m["max_abs"] / m["max_abs_ref"]) if m["max_abs_ref"] > 0 else (0.0 if m["max_abs"] == 0 else math.inf)
    nrm_b = float(np.linalg.norm(bf))
    m["rel_fro"] = float(np.linalg.norm(diff) / nrm_b) if nrm_b > 0 else (0.0 if m["max_abs"] == 0 else math.inf)
    m["mean_abs"] = float(np.abs(diff).mean())
    if is_logits:  # top-k metrics only when the caller says the tensor holds class scores
        k = min(topk, b.size)
        ta, tb = np.argsort(-af.ravel(), kind="stable")[:k], np.argsort(-bf.ravel(), kind="stable")[:k]
        m["top1_match"] = bool(ta[0] == tb[0])
        m["topk"] = k
        m["topk_overlap"] = float(len(set(ta.tolist()) & set(tb.tolist())) / k)
        m["top1_a"], m["top1_b"] = int(ta[0]), int(tb[0])
    return m


def _load_manifest(d: Path) -> dict | None:
    p = d / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def _index_dir(d: Path) -> dict[str, dict[str, Path]]:
    """{sample: {tensor: path}} from a manifest, or from `<sample>.<tensor>.npy` file names."""
    man = _load_manifest(d)
    idx: dict[str, dict[str, Path]] = {}
    if man:
        for s in man["samples"]:
            idx[s["name"]] = {t: d / v["file"] for t, v in s["tensors"].items()}
        return idx
    for p in sorted(d.glob("*.npy")):
        parts = p.name[:-4].rsplit(".", 1)
        sample, tensor = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        idx.setdefault(sample, {})[tensor] = p
    return idx


def compare_dirs(A: Path, B: Path, tensors: list[str] | None = None, skip=SKIP_DEFAULT, topk: int = 5,
                 logits_names=("logits",)) -> list[dict]:
    ia, ib = _index_dir(A), _index_dir(B)
    rows = []
    for sample, tb in ib.items():  # iterate over the reference
        ta = ia.get(sample)
        if ta is None:
            rows.append({"sample": sample, "tensor": "*", "error": f"sample missing in {A}"})
            continue
        for tname, pb in tb.items():
            if tensors and tname not in tensors:
                continue
            if tname in skip:
                continue
            pa = ta.get(tname)
            if pa is None:
                rows.append({"sample": sample, "tensor": tname, "error": f"tensor missing in {A}"})
                continue
            m = compare_arrays(np.load(pa), np.load(pb), topk=topk, is_logits=any(n in tname for n in logits_names))
            rows.append({"sample": sample, "tensor": tname, **m})
    return rows


def check_thresholds(row: dict, min_cos: float, max_abs: float | None, max_rel: float | None, min_top1: float, min_topk: float) -> list[str]:
    fails = []
    if "error" in row:
        return [row["error"]]
    if "exact" in row:
        if row["set_overlap"] < min_topk:
            fails.append(f"set_overlap {row['set_overlap']:.2f} < {min_topk}")
        return fails
    if row["cos_min"] < min_cos:
        fails.append(f"cos_min {row['cos_min']:.6f} < {min_cos}")
    if max_abs is not None and row["max_abs"] > max_abs:
        fails.append(f"max_abs {row['max_abs']:.3e} > {max_abs}")
    if max_rel is not None and row["rel_max_abs"] > max_rel:
        fails.append(f"rel_max_abs {row['rel_max_abs']:.3e} > {max_rel}")
    if "top1_match" in row:
        if (1.0 if row["top1_match"] else 0.0) < min_top1:
            fails.append(f"top1 mismatch ({row['top1_a']} vs ref {row['top1_b']})")
        if row["topk_overlap"] < min_topk:
            fails.append(f"top{row['topk']} overlap {row['topk_overlap']:.2f} < {min_topk}")
    return fails


def fmt_row(row: dict, fails: list[str]) -> str:
    tag = "FAIL" if fails else "ok  "
    head = f"{tag} {row.get('sample', ''):28s} {row.get('tensor', ''):26s}"
    if "error" in row:
        return f"{head} ERROR: {row['error']}"
    if "exact" in row:
        return f"{head} exact={row['exact']} set_overlap={row['set_overlap']:.2f}"
    s = (f"cos mean={row['cos_mean']:.6f} min={row['cos_min']:.6f}  max_abs={row['max_abs']:.3e}  "
         f"rel_max={row['rel_max_abs']:.2e} rel_fro={row['rel_fro']:.2e}")
    if "top1_match" in row:
        s += f"  top1={'match' if row['top1_match'] else 'MISMATCH'} top{row['topk']}={row['topk_overlap']:.2f}"
    if "note" in row:
        s += f"  ({row['note']})"
    return f"{head} {s}" + (f"   <- {'; '.join(fails)}" if fails else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("A", type=Path, help="candidate: .npy file or directory")
    ap.add_argument("B", type=Path, help="reference: .npy file or directory (manifest.json optional)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--tensors", type=str, default=None, help="comma list of tensor names to compare (default: all)")
    ap.add_argument("--skip", type=str, default=",".join(SKIP_DEFAULT), help="comma list of tensor names to skip")
    ap.add_argument("--logits", type=str, default="logits", help="comma list of name substrings treated as logits (top-k metrics)")
    ap.add_argument("--min-cos", type=float, default=DEFAULT_MIN_COS, help="fail if the worst per-token cosine is below this")
    ap.add_argument("--max-abs", type=float, default=None, help="fail if max |a-b| exceeds this (absolute; off by default)")
    ap.add_argument("--max-rel", type=float, default=DEFAULT_MAX_REL, help="fail if max|a-b| / max|b| exceeds this (use -1 to disable)")
    ap.add_argument("--min-top1", type=float, default=1.0, help="required top-1 agreement for logits (0 or 1 per tensor)")
    ap.add_argument("--min-topk", type=float, default=0.0, help="required top-k overlap fraction for logits / index tensors")
    ap.add_argument("--json", type=Path, default=None, help="also write all rows as JSON")
    ap.add_argument("--quiet", action="store_true", help="only print failures and the summary")
    a = ap.parse_args(argv)
    max_rel = None if (a.max_rel is not None and a.max_rel < 0) else a.max_rel
    logits_names = tuple(x for x in a.logits.split(",") if x)
    if a.A.is_dir() and a.B.is_dir():
        rows = compare_dirs(a.A, a.B, tensors=a.tensors.split(",") if a.tensors else None, skip=tuple(a.skip.split(",")),
                            topk=a.topk, logits_names=logits_names)
    elif a.A.is_file() and a.B.is_file():
        is_logits = any(n in a.A.name or n in a.B.name for n in logits_names)
        rows = [{"sample": a.A.name, "tensor": a.B.name, **compare_arrays(np.load(a.A), np.load(a.B), topk=a.topk, is_logits=is_logits)}]
    else:
        sys.exit("A and B must both be .npy files or both be directories")
    n_fail = 0
    for r in rows:
        fails = check_thresholds(r, a.min_cos, a.max_abs, max_rel, a.min_top1, a.min_topk)
        r["fails"] = fails
        n_fail += bool(fails)
        if fails or not a.quiet:
            print(fmt_row(r, fails))
    fl = [r for r in rows if "cos_min" in r]
    if fl:
        print(f"summary: {len(rows)} tensors, {n_fail} failing; worst cos_min={min(r['cos_min'] for r in fl):.6f}, "
              f"worst rel_max={max(r['rel_max_abs'] for r in fl):.2e}, worst max_abs={max(r['max_abs'] for r in fl):.3e}")
    else:
        print(f"summary: {len(rows)} tensors, {n_fail} failing")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
