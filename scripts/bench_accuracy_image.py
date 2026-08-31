#!/usr/bin/env python3
"""Image k-NN accuracy on Imagenette: PyTorch vs jepa.cpp, per model and per GGUF dtype.

Inference only.  Nothing is trained: the encoders are frozen, the "classifier" is a k-NN /
nearest-centroid vote over gallery features (scripts/knn_eval.py).

    scripts/bench_accuracy_image.py --stage all            # splits -> torch -> cpp -> eval
    scripts/bench_accuracy_image.py --stage cpp   --models ijepa-vith14-1k --dtypes q4_k
    scripts/bench_accuracy_image.py --stage eval                        # re-score cached features
    scripts/bench_accuracy_image.py --stage all --limit 60 --dtypes f16 --out tmp/smoke.json

`--limit N` is the smoke-test mode: it caps every split at N images, caches its features under
their own `<split>-limitN` file names and *requires* an explicit `--out`, so a one-minute smoke run
can neither be scored as a full run nor overwrite tests/results/accuracy-image.json.  Cache hits are
additionally checked for row count, so a stale matrix of the wrong size is a miss, never a silently
wrong number.

Stages
  splits  deterministic gallery/query lists   -> tests/results/accuracy-image-splits.json (compact,
          per-class indices into the sorted file list) and tmp/splits/<name>.json (resolved paths)
  torch   PyTorch float32 features (the exact HF / reference preprocessing of each model, see
          tests/fixtures/README.md)          -> tmp/feat/<model>/torch.f32.<split>.<feat>.npy
  cpp     build/jepa-embed straight from the JPEG (its own decoder + preprocessor), one GGUF dtype
          per pass                            -> tmp/feat/<model>/cpp.<dtype>.<split>.<feat>.npy
  eval    k-NN + centroid + agreement + feature cosine -> tests/results/accuracy-image.json

Feature per model (its own convention):
  ijepa-vith14-1k  mean of the 256 patch tokens   (IJepaModel last_hidden_state.mean(1) / --pool mean)
  lejepa-vits16    CLS token, and the mean of the 196 patch tokens as a second feature
  lewm-pusht       emb = enc.proj(CLS), the world-model state              (--pool lewm)  [sanity row]

Throughput is end-to-end wall time over a whole split (JPEG decode + preprocess + encode): PyTorch
runs batch 32 with the model loaded once and the load excluded, jepa-embed encodes ONE image per
call inside a process that is handed a chunk of 512 images and reloads the GGUF once per chunk (that
load IS in the number).  The asymmetry is deliberate -- it is what each backend's normal path costs
-- and is reported as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
DEFAULT_ROOT = HERE.parents[1]
SHARED = Path("/home/overseer2/workdir/jepa.cpp")          # git-ignored models/ + data/ live here

# --------------------------------------------------------------------------------------------------
# model table
# --------------------------------------------------------------------------------------------------
MODELS = {
    "ijepa-vith14-1k": {
        "gguf": "ijepa_vith14_1k-{dtype}.gguf",
        "dtypes": ["f16", "q8_0", "q4_k"],
        "feats": ["mean"],              # pooled feature names
        "pool": "mean",                 # jepa-embed --pool for the single-feature models
        "galleries": ["train2000"],
        "query": "val",
        "chunk": 512,
        "torch_dim": 1280,
    },
    "lejepa-vits16": {
        "gguf": "lejepa-vits16-pretrain-in1k-{dtype}.gguf",
        "dtypes": ["f32", "f16", "q8_0", "q4_k"],
        "feats": ["cls", "mean"],       # both come out of a single --pool none pass
        "pool": "none",
        "galleries": ["train2000", "train_full"],
        "query": "val",
        "chunk": 512,
        "torch_dim": 384,
    },
    "lewm-pusht": {
        "gguf": "lewm-pusht-{dtype}.gguf",
        "dtypes": ["f32", "f16", "q8_0", "q4_k"],
        "feats": ["emb"],
        "pool": "lewm",
        "galleries": ["train2000"],
        "query": "val1000",
        "chunk": 512,
        "torch_dim": 192,
    },
}

WNID_NAMES = {
    "n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
    "n03888257": "parachute",
}


# --------------------------------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------------------------------
def class_files(data: Path, split: str) -> dict[str, list[Path]]:
    """{wnid: sorted list of its JPEGs} for 'train' or 'val'."""
    out = {}
    for d in sorted((data / split).iterdir()):
        if d.is_dir():
            out[d.name] = sorted(d.glob("*.JPEG"))
    return out


def build_splits(data: Path, seed: int, per_class: int, val_per_class: int) -> dict:
    """Deterministic gallery/query lists.  Selections are stored as indices into the sorted
    per-class file list, so the JSON stays small and reproduces exactly on any checkout."""
    train, val = class_files(data, "train"), class_files(data, "val")
    wnids = sorted(train)
    idx_of = {w: i for i, w in enumerate(wnids)}

    rng = np.random.default_rng(seed)
    sel = {}
    for w in wnids:
        n = len(train[w])
        take = min(per_class, n)
        sel[w] = sorted(int(i) for i in rng.choice(n, size=take, replace=False))
    rng_v = np.random.default_rng(seed + 1)
    selv = {}
    for w in wnids:
        n = len(val[w])
        take = min(val_per_class, n)
        selv[w] = sorted(int(i) for i in rng_v.choice(n, size=take, replace=False))

    def resolve(files, picks=None):
        paths, labels = [], []
        for w in wnids:
            ids = range(len(files[w])) if picks is None else picks[w]
            for i in ids:
                paths.append(str(files[w][i]))
                labels.append(idx_of[w])
        return paths, labels

    splits = {}
    for name, (files, picks) in {
        "train2000": (train, sel),
        "train_full": (train, None),
        "val": (val, None),
        "val1000": (val, selv),
    }.items():
        p, l = resolve(files, picks)
        splits[name] = {"paths": p, "labels": l}

    compact = {
        "dataset": str(data), "seed": seed, "classes": wnids,
        "class_names": [WNID_NAMES.get(w, w) for w in wnids],
        "recipe": {
            "train2000": f"numpy default_rng({seed}).choice(n_class_files, {per_class}, replace=False), "
                         "sorted, per class in sorted-wnid order; indices into the sorted *.JPEG list",
            "train_full": "every train image, per class in sorted-wnid order",
            "val": "every val image, per class in sorted-wnid order",
            "val1000": f"numpy default_rng({seed + 1}).choice(n_class_files, {val_per_class}, replace=False), sorted",
        },
        "sizes": {k: len(v["paths"]) for k, v in splits.items()},
        "indices": {"train2000": sel, "val1000": selv},
        "sha256": {k: hashlib.sha256("\n".join(Path(p).name for p in v["paths"]).encode()).hexdigest()[:16]
                   for k, v in splits.items()},
    }
    return {"splits": splits, "compact": compact}


# --------------------------------------------------------------------------------------------------
# PyTorch feature extraction
# --------------------------------------------------------------------------------------------------
def torch_setup(root: Path, threads: int):
    sys.path.insert(0, str(HERE.parent))
    import dump_reference as dr

    dr.setup_env(SHARED, threads)
    import torch

    torch.set_num_threads(threads)
    return dr, torch


def load_torch_model(name: str, dr, torch, models_dir: Path):
    """(encode_fn, load_seconds).  encode_fn(list_of_PIL) -> {feat_name: np.ndarray [B, D]}."""
    t0 = time.time()
    if name == "ijepa-vith14-1k":
        from transformers import AutoImageProcessor, IJepaModel
        d = models_dir / "facebook/ijepa_vith14_1k"
        proc = AutoImageProcessor.from_pretrained(d)
        model = IJepaModel.from_pretrained(d, dtype=torch.float32).eval()

        def enc(ims):
            x = proc(images=ims, return_tensors="pt")["pixel_values"].float()
            with torch.inference_mode():
                lhs = model(pixel_values=x).last_hidden_state
            return {"mean": lhs.mean(1).numpy()}

    elif name == "lejepa-vits16":
        d = models_dir / "OK-AI/lejepa-vits16-pretrain-in1k"
        sys.path.insert(0, str(d))
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(d)
        model = AutoModel.from_pretrained(d, trust_remote_code=True, dtype=torch.float32).eval()

        def enc(ims):
            x = proc(images=ims, return_tensors="pt")["pixel_values"].float()
            with torch.inference_mode():
                out = model(x)
            return {"cls": out["latent"].numpy(), "mean": out["patch_latent"].mean(1).numpy()}

    elif name == "lewm-pusht":
        d = models_dir / "quentinll/lewm-pusht"
        cfg = json.loads((d / "config.json").read_text())
        model = dr.build_lewm(cfg)
        sd = torch.load(d / "weights.pt", map_location="cpu", weights_only=False)
        if "encoder.layers.0.attention.q_proj.weight" in model.state_dict():
            sd = dr._remap_vit_keys(sd)
        model.load_state_dict(sd, strict=True)
        model.eval()
        size = cfg["encoder"]["image_size"]

        def enc(ims):
            x = torch.cat([dr.image_to_tensor_imagenet(im, size) for im in ims], 0)
            with torch.inference_mode():
                _, _, emb = model.encode(x)
            return {"emb": emb.numpy()}
    else:
        raise SystemExit(f"unknown model {name}")
    return enc, time.time() - t0


def loadavg() -> list[float]:
    """1 / 5 / 15-minute run-queue length (`uptime`), recorded around every timed pass so the JSON
    says what else the box was doing while the img/s number was taken."""
    return [round(v, 2) for v in os.getloadavg()]


class Occupancy:
    """How much of the box belonged to somebody else while a timed stage ran.

    `uptime`'s load average cannot answer that: back-to-back 32-thread stages keep it near 32 no
    matter how idle the machine otherwise is.  So every timed pass also brackets itself with
    /proc/stat (CPU-seconds the *whole machine* spent out of idle) and os.times() (CPU-seconds this
    process and its reaped children spent).  The difference is foreign work.  On an idle box it is a
    fraction of a core; a second 32-thread agent shows up as tens of cores.
    """

    def __init__(self) -> None:
        self.t0 = self._sample()

    @staticmethod
    def _sample() -> tuple[float, float, float]:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        hz = os.sysconf("SC_CLK_TCK")
        busy = (sum(v) - v[3] - v[4]) / hz              # everything but idle + iowait
        t = os.times()
        return time.time(), busy, t.user + t.system + t.children_user + t.children_system

    def close(self) -> dict:
        w1, b1, o1 = self._sample()
        w0, b0, o0 = self.t0
        wall = max(w1 - w0, 1e-9)
        machine, own = b1 - b0, o1 - o0
        foreign = max(machine - own, 0.0)
        return {"wall_s": round(wall, 2),
                "machine_cpu_s": round(machine, 1),   # CPU-seconds the whole box spent non-idle
                "own_cpu_s": round(own, 1),           # ... of which this process tree
                "foreign_cpu_s": round(foreign, 1),   # ... left for everything else
                "foreign_cores": round(foreign / wall, 2)}


def run_torch(name: str, paths: list[str], out_prefix: Path, models_dir: Path, threads: int,
              batch: int = 32) -> dict:
    dr, torch = torch_setup(DEFAULT_ROOT, threads)
    from PIL import Image

    enc, load_s = load_torch_model(name, dr, torch, models_dir)
    feats: dict[str, list] = {}
    la0, occ = loadavg(), Occupancy()
    t0 = time.time()
    for s in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        for k, v in enc(ims).items():
            feats.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        if (s // batch) % 20 == 0:
            done = min(s + batch, len(paths))
            el = time.time() - t0
            print(f"  torch {name}: {done}/{len(paths)}  {done / max(el, 1e-9):.2f} img/s", flush=True)
    wall = time.time() - t0
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for k, v in feats.items():
        np.save(f"{out_prefix}.{k}.npy", np.concatenate(v, 0))
    return {"n": len(paths), "wall_s": wall, "img_per_s": len(paths) / wall, "load_s": load_s,
            "batch": batch, "loadavg_start": la0, "loadavg_end": loadavg(), "occupancy": occ.close()}


# --------------------------------------------------------------------------------------------------
# jepa.cpp feature extraction
# --------------------------------------------------------------------------------------------------
def run_cpp(name: str, dtype: str, paths: list[str], out_prefix: Path, root: Path, gguf_dir: Path,
            threads: int, chunk: int, log: Path) -> dict:
    cfg = MODELS[name]
    gguf = gguf_dir / cfg["gguf"].format(dtype=dtype)
    if not gguf.exists():
        raise SystemExit(f"missing {gguf}")
    embed = root / "build" / "jepa-embed"
    tmpd = root / "tmp" / "chunks"
    tmpd.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list] = {}
    wall = 0.0
    la0, occ = loadavg(), Occupancy()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as lf:
        for s in range(0, len(paths), chunk):
            part = paths[s:s + chunk]
            cpath = tmpd / f"{name}.{dtype}.{s}.npy"
            cmd = [str(embed), "-m", str(gguf)]
            for p in part:
                cmd += ["-i", p]
            cmd += ["--pool", cfg["pool"], "-o", str(cpath), "-t", str(threads), "--print-n", "0"]
            t0 = time.time()
            r = subprocess.run(cmd, stdout=lf, stderr=lf)
            wall += time.time() - t0
            if r.returncode != 0:
                raise SystemExit(f"jepa-embed failed ({r.returncode}); see {log}")
            a = np.load(cpath)
            cpath.unlink()
            if cfg["pool"] == "none":
                a = a.reshape(len(part), -1, a.shape[-1])       # [n, 1 + n_patches, D], CLS first
                acc.setdefault("cls", []).append(a[:, 0].copy())
                acc.setdefault("mean", []).append(a[:, 1:].mean(1))
            else:
                acc.setdefault(cfg["feats"][0], []).append(a.reshape(len(part), -1))
            done = min(s + chunk, len(paths))
            print(f"  cpp {name} {dtype}: {done}/{len(paths)}  {done / wall:.2f} img/s", flush=True)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for k, v in acc.items():
        np.save(f"{out_prefix}.{k}.npy", np.concatenate(v, 0).astype(np.float32))
    # the wall time above is the sum of the chunk subprocesses, i.e. it *includes* one model load
    # per chunk; that is the number a user would see running the tool, so it is the one reported.
    return {"n": len(paths), "wall_s": wall, "img_per_s": len(paths) / wall,
            "chunk": chunk, "n_chunks": (len(paths) + chunk - 1) // chunk, "gguf": gguf.name,
            "loadavg_start": la0, "loadavg_end": loadavg(), "occupancy": occ.close()}


# --------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------
def feat_prefix(root: Path, model: str, backend: str, dtype: str, split: str, limit: int = 0) -> Path:
    """Where one (backend, dtype, split) pass caches its features.

    A `--limit N` smoke run gets its own `<split>-limitN` names: its 60-row matrices must never be
    picked up by a later full run (or vice versa), and a mismatch that only shows up as a wrong
    accuracy number is worse than a cache miss."""
    tag = f"{split}-limit{limit}" if limit else split
    return root / "tmp" / "feat" / model / f"{backend}.{dtype}.{tag}"


def feat_path(root: Path, model: str, backend: str, dtype: str, split: str, feat: str,
              limit: int = 0) -> Path:
    return Path(f"{feat_prefix(root, model, backend, dtype, split, limit)}.{feat}.npy")


def cache_hit(prefix: Path, feats: list[str], n_expected: int) -> bool:
    """True only if every feature file of this pass exists *and* holds exactly `n_expected` rows."""
    for f in feats:
        p = Path(f"{prefix}.{f}.npy")
        if not p.exists():
            return False
        rows = int(np.load(p, mmap_mode="r").shape[0])
        if rows != n_expected:
            print(f"note: {p} has {rows} rows, expected {n_expected} — ignoring the cache")
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--data", type=Path, default=SHARED / "data/imagenette/imagenette2-160")
    ap.add_argument("--models-dir", type=Path, default=SHARED / "models")
    ap.add_argument("--gguf-dir", type=Path, default=SHARED / "models/gguf")
    ap.add_argument("--stage", default="all", choices=["splits", "torch", "cpp", "eval", "all"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--dtypes", default="", help="restrict the cpp dtypes (comma separated)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--val-per-class", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap every split at N images")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    root: Path = a.root
    models = [m for m in a.models.split(",") if m]
    if a.limit and a.out is None:
        sys.exit("--limit is a smoke test over a truncated split: its numbers are not the benchmark, "
                 "so it will not write the default tests/results/accuracy-image.json.  Pass an "
                 "explicit --out (e.g. --out tmp/smoke.json).")
    out_json = a.out or root / "tests/results/accuracy-image.json"
    split_dir = root / "tmp" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    timing_path = root / "tmp" / ("timing.json" if not a.limit else f"timing-limit{a.limit}.json")
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    # ---- splits
    built = build_splits(a.data, a.seed, a.per_class, a.val_per_class)
    splits, compact = built["splits"], built["compact"]
    if a.limit:
        for k, v in splits.items():
            step = max(1, len(v["paths"]) // a.limit)
            v["paths"], v["labels"] = v["paths"][::step][:a.limit], v["labels"][::step][:a.limit]
        # keep train2000 a subset of train_full (the eval indexes one into the other)
        tf = splits["train_full"]
        splits["train2000"] = {"paths": tf["paths"][::2], "labels": tf["labels"][::2]}
        compact["limit"] = a.limit
        compact["sizes"] = {k: len(v["paths"]) for k, v in splits.items()}
        compact["note"] = (f"SMOKE RUN: every split truncated to at most {a.limit} images; the "
                           "recipe and indices above describe the full splits, not these.")
    if a.stage in ("splits", "all"):
        (split_dir / "resolved.json").write_text(json.dumps(splits))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        (out_json.parent / "accuracy-image-splits.json").write_text(json.dumps(compact, indent=1))
        print(json.dumps({k: len(v["paths"]) for k, v in splits.items()}, indent=2))

    def split_of(model: str, which: str) -> str:
        return MODELS[model]["query"] if which == "query" else which

    # ---- torch
    if a.stage in ("torch", "all"):
        for m in models:
            cfg = MODELS[m]
            need = sorted({*cfg["galleries"], cfg["query"]})
            # train2000 is a subset of train_full: encode the superset once and index into it
            if "train_full" in need and "train2000" in need:
                need.remove("train2000")
            for sp in need:
                pre = feat_prefix(root, m, "torch", "f32", sp, a.limit)
                if cache_hit(pre, cfg["feats"], len(splits[sp]["paths"])):
                    print(f"skip torch {m} {sp} (cached, {len(splits[sp]['paths'])} rows)")
                    continue
                t = run_torch(m, splits[sp]["paths"], pre, a.models_dir, a.threads)
                timing[f"torch|{m}|f32|{sp}"] = t
                timing_path.write_text(json.dumps(timing, indent=1))
                print(f"torch {m} {sp}: {t['wall_s']:.1f} s, {t['img_per_s']:.2f} img/s", flush=True)

    # ---- cpp
    if a.stage in ("cpp", "all"):
        for m in models:
            cfg = MODELS[m]
            dts = [d for d in cfg["dtypes"] if not a.dtypes or d in a.dtypes.split(",")]
            need = sorted({*cfg["galleries"], cfg["query"]})
            if "train_full" in need and "train2000" in need:
                need.remove("train2000")
            for dt in dts:
                for sp in need:
                    pre = feat_prefix(root, m, "cpp", dt, sp, a.limit)
                    if cache_hit(pre, cfg["feats"], len(splits[sp]["paths"])):
                        print(f"skip cpp {m} {dt} {sp} (cached, {len(splits[sp]['paths'])} rows)")
                        continue
                    t = run_cpp(m, dt, splits[sp]["paths"], pre, root, a.gguf_dir, a.threads,
                                cfg["chunk"], root / "tmp" / "logs" / f"{m}.{dt}.{sp}.log")
                    timing[f"cpp|{m}|{dt}|{sp}"] = t
                    timing_path.write_text(json.dumps(timing, indent=1))
                    print(f"cpp {m} {dt} {sp}: {t['wall_s']:.1f} s, {t['img_per_s']:.2f} img/s", flush=True)

    # ---- eval
    if a.stage in ("eval", "all"):
        sys.path.insert(0, str(HERE.parent))
        from knn_eval import evaluate, nn_margin

        # index of the train2000 rows inside train_full (so one encode pass serves both galleries)
        full_idx = {p: i for i, p in enumerate(splits["train_full"]["paths"])}
        sub_rows = np.array([full_idx[p] for p in splits["train2000"]["paths"]], dtype=np.int64)

        def load(model, backend, dtype, split, feat):
            cfg = MODELS[model]
            if split == "train2000" and "train_full" in cfg["galleries"]:
                f = feat_path(root, model, backend, dtype, "train_full", feat, a.limit)
                return np.load(f)[sub_rows]
            return np.load(feat_path(root, model, backend, dtype, split, feat, a.limit))

        rows = []
        for m in models:
            cfg = MODELS[m]
            q_split = cfg["query"]
            ql = np.asarray(splits[q_split]["labels"])
            for feat in cfg["feats"]:
                ref_q = np.load(feat_path(root, m, "torch", "f32", q_split, feat, a.limit))
                for gal in cfg["galleries"]:
                    gl = np.asarray(splits[gal]["labels"])
                    ref_g = load(m, "torch", "f32", gal, feat)
                    # how decided each query item is under the reference backend (see knn_eval)
                    margin = nn_margin(ref_g, gl, ref_q, 10)
                    ref = evaluate(ref_g, gl, ref_q, ql, 10, a.k, a.temp, margin=margin)
                    ref_preds = ref["knn_preds"]
                    rows.append(dict(model=m, backend="pytorch", dtype="f32", feature=feat,
                                     gallery=gal, n_gallery=ref["n_gallery"], n_query=ref["n_query"],
                                     knn_top1=ref["knn_top1"], centroid_top1=ref["centroid_top1"],
                                     agreement=1.0, feat_cos_mean=1.0, feat_cos_min=1.0,
                                     margin_all_median=ref["margin_all_median"],
                                     img_per_s=timing.get(f"torch|{m}|f32|{q_split}", {}).get("img_per_s")))
                    for dt in cfg["dtypes"]:
                        qp = feat_path(root, m, "cpp", dt, q_split, feat, a.limit)
                        if not qp.exists():
                            print(f"note: no jepa.cpp features for {m} {dt} {feat} -- row skipped")
                            continue
                        q = np.load(qp)
                        r = evaluate(load(m, "cpp", dt, gal, feat), gl, q, ql, 10, a.k, a.temp,
                                     ref_query=ref_q, ref_preds=ref_preds, margin=margin)
                        rows.append(dict(model=m, backend="jepa.cpp", dtype=dt, feature=feat,
                                         gallery=gal, n_gallery=r["n_gallery"], n_query=r["n_query"],
                                         knn_top1=r["knn_top1"], centroid_top1=r["centroid_top1"],
                                         agreement=r["agreement"], feat_cos_mean=r["feat_cos_mean"],
                                         feat_cos_min=r["feat_cos_min"],
                                         n_flipped=r["n_flipped"], flip_pytorch_right=r["flip_ref_right"],
                                         flip_jepacpp_right=r["flip_this_right"],
                                         flip_both_wrong=r["flip_both_wrong"],
                                         margin_flipped_median=r.get("margin_flipped_median"),
                                         margin_all_median=r["margin_all_median"],
                                         img_per_s=timing.get(f"cpp|{m}|{dt}|{q_split}", {}).get("img_per_s")))
        result = {
            "task": "image k-NN accuracy on Imagenette (frozen features, no training)",
            "protocol": {"k": a.k, "temp": a.temp, "metric": "cosine on L2-normalised features",
                         "vote": "DINO weighted vote, weight = exp(sim/temp)",
                         "centroid": "nearest L2-normalised class-mean of the L2-normalised gallery features",
                         "agreement": "fraction of query items whose kNN prediction equals PyTorch's",
                         "feat_cos": "mean / worst per-item cosine between the jepa.cpp and PyTorch query features",
                         "margin": "cosine to the best gallery neighbour minus the cosine to the best neighbour of "
                                   "any other class, computed on the PyTorch features; near 0 = the top two classes "
                                   "are tied and a tiny feature perturbation flips the vote",
                         "flip_*": "of the items whose prediction differs from PyTorch's, how many PyTorch got right, "
                                   "how many jepa.cpp got right, and how many both got wrong"},
            "dataset": compact, "threads": a.threads, "limit": a.limit,
            "timing": timing, "rows": rows,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=1))
        print(f"wrote {out_json}  ({len(rows)} rows)")
        hdr = f"{'model':16} {'backend':9} {'dtype':5} {'feat':5} {'gallery':11} {'kNN':>7} {'cent':>7} {'agree':>7} {'cos':>9} {'img/s':>7}"
        print(hdr)
        for r in rows:
            print(f"{r['model']:16} {r['backend']:9} {r['dtype']:5} {r['feature']:5} "
                  f"{r['gallery']}({r['n_gallery']}) {100 * r['knn_top1']:7.2f} {100 * r['centroid_top1']:7.2f} "
                  f"{100 * r['agreement']:7.2f} {r['feat_cos_mean']:9.6f} "
                  f"{(r['img_per_s'] or 0):7.2f}")


if __name__ == "__main__":
    main()
