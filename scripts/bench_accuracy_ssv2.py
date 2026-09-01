#!/usr/bin/env python3
"""Something-Something-v2 validation accuracy: PyTorch vs jepa.cpp, per dtype and backend.

The real task accuracy of `facebook/vjepa2-vitl-fpc16-256-ssv2` (V-JEPA 2 ViT-L/16 encoder +
3-block attentive pooler + 174-way linear head) on the full 24 777-clip SSv2 validation split.
It is the companion of the UCF-101 harness (`scripts/bench_accuracy_video.py`), which runs the
same head on UCF clips and can therefore only score *agreement* with PyTorch; the shared plumbing
(load average, foreign-occupancy accounting, environment stamp, model/paths) is imported from it.

Stages (run in this order; each writes into <work>/ and can be re-run on its own):

    scripts/bench_accuracy_ssv2.py frames  --jobs 48
    scripts/bench_accuracy_ssv2.py lists
    scripts/bench_accuracy_ssv2.py torch   --device cuda:1
    scripts/bench_accuracy_ssv2.py torch   --device cpu --scope sub100
    scripts/bench_accuracy_ssv2.py cpp     --dtype f16  --device cuda:1
    scripts/bench_accuracy_ssv2.py cpp     --dtype f16  --device cpu --scope sub10
    scripts/bench_accuracy_ssv2.py report  --out-json tests/results/accuracy-ssv2.json

`frames` decodes every validation clip once into a THWC uint8 `.npy` under
`data/ssv2/frames-val/` so that every backend sees byte-identical pixels; `lists` freezes the clip
order and the CPU subsets in `clips.json`, which every logits `.npy` is indexed by.  The frame
cache is ≈106 GB (measured for the validation split) and is meant to be deleted once the sweep is done — `report` copies the clip
order, the sampled frame indices and a few tensor digests into the committed JSON so the run can
be reproduced without it.

Scopes
    full    all 24 777 validation clips                    (PyTorch reference, jepa.cpp on CUDA)
    sub10   every 10th clip of `full`  — 2 478 clips       (jepa.cpp on the CPU)
    sub100  every 100th clip of `full` —   248 clips       (PyTorch on the CPU, the f32 anchor)
sub100 is a subset of sub10 is a subset of full, so a CPU run over sub10 also answers sub100 and
every scope can be compared against the CUDA and PyTorch rows clip for clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import bench_accuracy_video as _v                                    # noqa: E402
from video_frames import decode_video, sample_frames                # noqa: E402

SHARED = _v.SHARED                     # git-ignored checkpoints / datasets / venvs live there
shared = _v.shared
loadavg = _v.loadavg
Occupancy = _v.Occupancy

MODEL = _v.SSV2_MODEL                  # vjepa2-vitl-fpc16-256-ssv2
HF_DIR = _v.SSV2_HF_DIR
LABEL = "V-JEPA 2 ViT-L/16 (fpc16-256, SSv2 head)"
DATA = "data/ssv2"
FRAMES = 16
CPU_THREADS = _v.THREADS               # 32, the number every CPU benchmark in this repo uses
SCOPES = {"full": 1, "sub10": 10, "sub100": 100}

# The published figure this benchmark is measured against.  V-JEPA 2 (arXiv:2506.09985) Table 4.
PUBLISHED = {
    "top1": 73.7, "model": "V-JEPA 2 ViT-L", "source": "arXiv:2506.09985 Table 4",
    "views": "16 frames x 2 temporal crops x 3 spatial crops, logits averaged across the 6 clips",
    "probe": "4-block attentive probe (3 self-attention blocks + a cross-attention block)",
}


# --------------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------------
def hf_label_list() -> list[str]:
    """`id2label` of the HF checkpoint, in class-index order — the order the head's logits are in."""
    cfg = json.loads((shared(HF_DIR) / "config.json").read_text())
    i2l = cfg["id2label"]
    return [i2l[str(i)] for i in range(len(i2l))]


def gguf_label_list(gguf: Path) -> list[str] | None:
    """`jepa.head.labels` of a GGUF, so the C++ side's class order can be checked against HF's."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    r = GGUFReader(str(gguf))
    f = r.fields.get("jepa.head.labels")
    if f is None:
        return None
    return [str(bytes(f.parts[i]), "utf-8") for i in f.data]


def label_mapping() -> dict:
    """Map every validation entry to a class index, and prove the mapping three ways.

    The dataset ships `labels/labels.json` (class name -> id) and, per clip,
    `labels/validation.json` with a `template` field.  The templates are written with the
    placeholder in brackets (`Bending [something] until it breaks`); `labels.json` writes the same
    string without them.  The HF checkpoint's `id2label` uses the *bracketed* spelling, so a
    validation `template` is a verbatim key into `id2label` — no normalisation at all — and
    stripping the brackets lands on the same id in `labels.json`.  Both are asserted here, because
    an off-by-one class order is the failure mode that looks like a plausible accuracy.
    """
    hf = hf_label_list()
    hf_idx = {t: i for i, t in enumerate(hf)}
    ds = {k: int(v) for k, v in json.loads((shared(DATA) / "labels/labels.json").read_text()).items()}
    strip = lambda s: s.replace("[", "").replace("]", "")            # noqa: E731
    if len(hf) != len(ds):
        sys.exit(f"class count differs: HF {len(hf)} vs labels.json {len(ds)}")
    bad = [t for t in hf if ds.get(strip(t)) != hf_idx[t]]
    if bad:
        sys.exit(f"{len(bad)} classes disagree between HF id2label and labels.json, e.g. {bad[:3]}")
    return {"n_classes": len(hf), "index_source": "config.json id2label of " + HF_DIR,
            "clip_to_class": "validation.json `template` is a verbatim id2label value",
            "dataset_agreement": "labels.json id == id2label index for all "
                                 f"{len(hf)} classes after removing the '[' ']' placeholder brackets",
            "labels": hf}


def validation_entries() -> list[dict]:
    return json.loads((shared(DATA) / "labels/validation.json").read_text())


# --------------------------------------------------------------------------------------------
# stage: frames
# --------------------------------------------------------------------------------------------
def _npy_path(out: Path, cid: str) -> Path:
    return out / cid[-2:] / (cid + ".npy")


def _decode_one(job):
    cid, src, dst, n_frames, prev = job
    dst = Path(dst)
    if dst.exists():
        # A warm cache still has to answer with the *whole* record — `frame_indices`,
        # `n_frames_total` and `fps` are what the committed manifest is built from, and a run that
        # dropped them would write an index that silently degrades the artifact on the next
        # `report`.  They are carried over from the previous index rather than re-decoded.
        a = np.load(dst, mmap_mode="r")
        rec = {k: v for k, v in (prev or {}).items()
               if k in ("n_frames_total", "frame_indices", "fps")}
        return {"id": cid, "npy": str(dst), "cached": True,
                "frame_size_hw": [int(a.shape[1]), int(a.shape[2])], **rec}
    try:
        frames, fps = decode_video(Path(src))
        fr, idx = sample_frames(frames, n_frames)
    except Exception as e:                     # a clip that will not decode is reported, not fatal
        return {"id": cid, "error": f"{type(e).__name__}: {e}"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".npy.tmp")          # atomic: a killed run never leaves a half file
    with open(tmp, "wb") as f:
        np.save(f, fr)
    tmp.replace(dst)
    return {"id": cid, "npy": str(dst), "cached": False, "n_frames_total": int(len(frames)),
            "frame_indices": idx, "fps": fps,
            "frame_size_hw": [int(fr.shape[1]), int(fr.shape[2])]}


def stage_frames(a) -> None:
    out = Path(a.out) if a.out else shared(DATA) / "frames-val"
    vids = shared(DATA) / "videos"
    ents = validation_entries()
    idx_p = out / "index.json"
    prev_idx = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    prev = {r["id"]: r for r in prev_idx.get("clips", [])}
    jobs = [(e["id"], str(vids / (e["id"] + ".webm")), str(_npy_path(out, e["id"])), a.frames,
             prev.get(e["id"])) for e in ents]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} validation clips, {a.frames} frames each -> {out} ({a.jobs} jobs)", flush=True)
    t0, recs = time.time(), []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(_decode_one, jobs, chunksize=8), 1):
            recs.append(r)
            if i % 1000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time()-t0:.1f}s", flush=True)
    wall = time.time() - t0
    bad = [r for r in recs if "error" in r]
    n_new = sum(1 for r in recs if not r.get("cached") and "error" not in r)
    # `wall_s` is the time the cache cost to build, and a second pass over a warm cache costs
    # nothing — recording that 0.0 would quietly erase a measured figure from the artifact, so a
    # pass that decodes nothing keeps the one it found.
    idx = idx_p
    idx.write_text(json.dumps({
        "dataset": DATA, "split": "validation", "frames": a.frames,
        "sampling": "idx = round(linspace(0, T_total-1, n)) over all PyAV-decoded rgb24 frames",
        "layout": "THWC uint8 .npy per clip", "n_clips": len(recs), "n_failed": len(bad),
        "n_decoded_this_pass": n_new,
        "wall_s": round(wall, 1) if n_new else float(prev_idx.get("wall_s", 0.0)),
        "clips": recs}, indent=1))
    print(f"decoded {sum(0 if r.get('cached') or 'error' in r else 1 for r in recs)}/{len(recs)} "
          f"clips in {wall:.1f}s, {len(bad)} failed -> {idx}")
    for r in bad[:20]:
        print(f"  FAILED {r['id']}: {r['error']}")


# --------------------------------------------------------------------------------------------
# stage: lists
# --------------------------------------------------------------------------------------------
def stage_lists(a) -> None:
    out = Path(a.frames_dir) if a.frames_dir else shared(DATA) / "frames-val"
    index = json.loads((out / "index.json").read_text())
    have = {r["id"]: r for r in index["clips"] if "npy" in r}
    lm = label_mapping()
    hf_idx = {t: i for i, t in enumerate(lm["labels"])}

    rows, skipped = [], []
    for e in validation_entries():                 # validation.json order, which is fixed upstream
        if e["template"] not in hf_idx:
            sys.exit(f"clip {e['id']}: template {e['template']!r} is not one of the 174 classes")
        r = have.get(e["id"])
        if r is None:
            skipped.append({"id": e["id"], "reason": "no frame cache (decode failed or not run)"})
            continue
        rows.append({"i": len(rows), "id": e["id"], "label": hf_idx[e["template"]],
                     "template": e["template"], "npy": r["npy"]})
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    scopes = {name: list(range(0, len(rows), step)) for name, step in SCOPES.items()}
    (work / "clips.json").write_text(json.dumps(
        {"dataset": DATA, "split": "validation", "frames": index["frames"],
         "sampling": index["sampling"], "n_entries": len(validation_entries()),
         "skipped": skipped, "label_mapping": {k: v for k, v in lm.items() if k != "labels"},
         "labels": lm["labels"],
         "order": "the order of labels/validation.json, decode failures removed — the row order of "
                  "every logits .npy this benchmark writes",
         "scopes": {n: {"step": SCOPES[n], "n": len(ix), "idx": ix} for n, ix in scopes.items()},
         "clips": rows}, indent=1))
    for name, ix in scopes.items():
        (work / f"{name}.txt").write_text("".join(rows[i]["npy"] + "\n" for i in ix))
    print(f"{len(rows)} clips ({len(skipped)} skipped), {len(lm['labels'])} classes -> "
          f"{work/'clips.json'}")
    for n, ix in scopes.items():
        print(f"  {n}: {len(ix)} clips -> {work/(n+'.txt')}")


def load_clips(work: Path) -> dict:
    return json.loads((work / "clips.json").read_text())


# --------------------------------------------------------------------------------------------
# stage: torch  (the fp32 reference)
# --------------------------------------------------------------------------------------------
def stage_torch(a) -> None:
    import torch
    # TF32 would silently drop the reference to ~10 mantissa bits on an Ada GPU; the reference is
    # fp32 and says so in the JSON.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(a.threads)   # OMP_NUM_THREADS has to be exported before torch is
                                       # imported to have any effect, so the recipes do that
    from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

    work = Path(a.work)
    cj = load_clips(work)
    idx = cj["scopes"][a.scope]["idx"]
    dev = a.device
    d = shared(HF_DIR)
    t0 = time.time()
    proc = AutoVideoProcessor.from_pretrained(d)
    model = VJEPA2ForVideoClassification.from_pretrained(d, dtype=torch.float32).eval().to(dev)
    load_s = time.time() - t0
    hf_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if hf_labels != cj["labels"]:
        sys.exit("id2label of the loaded model differs from the one `lists` mapped the clips with")
    la0 = loadavg()
    print(f"{MODEL}: loaded in {load_s:.1f}s on {dev}, scope {a.scope} = {len(idx)} clips, "
          f"batch {a.batch}, tf32={torch.backends.cuda.matmul.allow_tf32}, loadavg {la0}", flush=True)

    # The preprocessing (short side 292 bilinear -> centre crop 256 -> /255 -> mean/std) runs on
    # the CPU exactly as the processor ships it; a small thread pool keeps it off the GPU's
    # critical path without changing a single pixel.
    from concurrent.futures import ThreadPoolExecutor
    paths = [cj["clips"][i]["npy"] for i in idx]

    def prep(batch_paths):
        fr = [np.load(p) for p in batch_paths]
        return proc(videos=fr, return_tensors="pt")["pixel_values_videos"].float()

    batches = [paths[i:i + a.batch] for i in range(0, len(paths), a.batch)]
    occ = Occupancy()
    logits, t0, n = [], time.time(), 0
    # A *bounded* prefetch window.  `Executor.map` submits every batch at once, which on the full
    # split queues 6 000 preprocessed clips — 300 GB of float32 — and the run is OOM-killed.
    with ThreadPoolExecutor(max_workers=4) as pool:
        window, nxt = [], 0
        while nxt < min(a.prefetch, len(batches)):
            window.append(pool.submit(prep, batches[nxt])); nxt += 1
        while window:
            x = window.pop(0).result()
            if nxt < len(batches):
                window.append(pool.submit(prep, batches[nxt])); nxt += 1
            with torch.no_grad():
                logits.append(model(pixel_values_videos=x.to(dev)).logits.float().cpu().numpy())
            n += len(x)
            del x
            if n % 500 < a.batch or n == len(paths):
                print(f"  {n}/{len(paths)}  {time.time()-t0:.1f}s "
                      f"({n/max(time.time()-t0,1e-9):.2f} clips/s)", flush=True)
    wall = time.time() - t0
    L = np.concatenate(logits).astype(np.float32)
    tag = f"torch-{_devtag(dev)}-{a.scope}"
    np.save(work / f"{tag}-logits.npy", L)
    (work / f"{tag}.json").write_text(json.dumps(
        {"model": MODEL, "backend": "pytorch", "device": dev, "dtype": "f32", "tf32": False,
         "scope": a.scope, "n_clips": int(len(L)), "n_classes": int(L.shape[1]),
         "batch": a.batch, "threads": a.threads, "torch": torch.__version__,
         "model_load_s": round(load_s, 2), "wall_s": round(wall, 1),
         "clips_per_s": round(len(L) / wall, 3),
         "loadavg_start": la0, "loadavg_end": loadavg(), "occupancy": occ.close()}, indent=1))
    print(f"{tag} logits {L.shape} in {wall:.1f}s ({len(L)/wall:.2f} clips/s)")


def _devtag(dev: str) -> str:
    return dev.replace(":", "")


# --------------------------------------------------------------------------------------------
# stage: cpp
# --------------------------------------------------------------------------------------------
def stage_cpp(a) -> None:
    work = Path(a.work)
    cj = load_clips(work)
    gpu = a.device.startswith("cuda")
    build = ROOT / ("build-cuda" if gpu else "build")
    gguf = shared("models/gguf") / f"{MODEL}-{a.dtype}.gguf"
    if not gguf.exists():
        sys.exit(f"missing {gguf}")
    lst = work / f"{a.scope}.txt"
    if not lst.exists():
        sys.exit(f"missing {lst} — run the `lists` stage first")
    tag = f"cpp-{_devtag(a.device)}-{a.dtype}-{a.scope}"
    cmd = [str(build / "jepa-embed"), "-m", str(gguf), "--batch", "1",
           "--frames-list", str(lst), "-o", str(work / f"{tag}-feats.npy"),
           "--logits", str(work / f"{tag}-logits.npy"),
           "--json", str(work / f"{tag}.json"), "--progress", "500"]
    cmd += ["--gpu", a.device.split(":")[1] if ":" in a.device else "0"] if gpu else \
           ["-t", str(a.threads)]
    print(" ".join(cmd), flush=True)
    t0, la0, occ = time.time(), loadavg(), Occupancy()
    subprocess.run(cmd, check=True)
    p = work / f"{tag}.json"
    d = json.loads(p.read_text())
    # `dtype` is the file's suffix rather than jepa-embed's `file_type`.  The two agree on every
    # file this benchmark runs (q4_k included — `GGML_FTYPE_MOSTLY_Q4_K` exists and jepa-info
    # prints it), but the suffix is what `--dtype` asked for and what the docs label the row with,
    # so the label cannot drift from the request if a future writer stamps the enum differently.
    d.update({"scope": a.scope, "device": a.device, "backend": "cuda" if gpu else "cpu",
              "dtype": a.dtype, "gguf": str(gguf.relative_to(SHARED)), "loadavg_start": la0,
              "loadavg_end": loadavg(), "occupancy": occ.close()})
    p.write_text(json.dumps(d, indent=1))
    print(f"{tag} in {time.time()-t0:.1f}s (loadavg {la0} -> {loadavg()})")


# --------------------------------------------------------------------------------------------
# stage: report
# --------------------------------------------------------------------------------------------
def _topk_acc(L: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argpartition(-L, k - 1, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def _cos(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    b = B / np.linalg.norm(B, axis=1, keepdims=True)
    c = (a * b).sum(1)
    return float(c.mean()), float(c.min())


class _Raw:
    """A sequence that is written as one compact JSON line instead of one element per line.

    Deliberately *not* a `list` subclass: `json` serialises those itself and never consults the
    encoder's `default`, which is the hook this uses.
    """

    def __init__(self, seq):
        self.seq = list(seq)


def _dumps(obj) -> str:
    """`json.dumps(indent=1)`, except that _Raw lists stay on a single line.

    The per-clip arrays here are 24 777 elements long; indenting them would turn a 1 MB artifact
    into a 12 MB one for no gain in readability.
    """
    box: list[str] = []

    class E(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, _Raw):
                box.append(json.dumps(o.seq, separators=(",", ":")))
                return f"@@{len(box) - 1}@@"
            return super().default(o)

    s = json.dumps(obj, indent=1, cls=E)
    for i, raw in enumerate(box):
        s = s.replace(f'"@@{i}@@"', raw)
    return s


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(cj: dict, index: dict, n: int) -> dict:
    """Enough to rebuild the frame cache and prove it was rebuilt identically.

    `frames-val/` is ≈106 GB (measured) and is deleted after the sweep, so the committed artifact carries the
    sampled frame indices and a sha256 of the first `n` clip tensors instead: re-running
    `frames` and hashing those files reproduces the digests exactly.
    """
    by_id = {r["id"]: r for r in index["clips"]}
    out = []
    for r in cj["clips"][:n]:
        rec = by_id.get(r["id"], {})
        p = Path(r["npy"])
        out.append({"id": r["id"], "class": r["label"],
                    "n_frames_total": rec.get("n_frames_total"),
                    "frame_indices": _Raw(rec.get("frame_indices") or []),
                    "frame_size_hw": rec.get("frame_size_hw"),
                    "sha256": _sha(p) if p.exists() else None})
    return {"note": "sha256 of the THWC uint8 .npy of the first "
                    f"{n} validation clips, in clips.json order", "clips": out}


def _commit(prev: dict, restamp: bool) -> str:
    """The revision this artifact's numbers were first written at.

    Stamping the checkout's HEAD on every `report` would make the file change every time the
    documents are regenerated, which is exactly the round-trip the frame-cache fallback exists to
    protect; the sweep's own revision is also the honest answer, since a later `report` re-derives
    the same metrics from the same logits.  `--restamp` takes the current HEAD instead.
    """
    if prev.get("commit") and not restamp:
        return prev["commit"]
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _relpath(p: Path) -> str:
    """Repo-relative when it can be, absolute when it cannot.

    `--frames-dir` may point anywhere; `Path.relative_to` raises for a path outside the checkout,
    and doing that here would kill `report` after a multi-hour sweep.
    """
    try:
        return str(p.resolve().relative_to(SHARED))
    except ValueError:
        return str(p)


def _frame_facts(index: dict, cj: dict, n: int, prev: dict) -> tuple[dict, dict]:
    """The frame-cache half of the artifact: from the live index, or from the previous artifact.

    The cache is a hundred gigabytes and change, and the sweep deletes it, so a later `report` —
    regenerating the docs, say — would otherwise overwrite a measured decode count, wall time and
    manifest with zeros and nulls, and `docs/accuracy-video.md` would render "0 clips decoded".
    Whatever this run can still see wins; the rest is carried forward, the way
    scripts/bench_accuracy_video.py carries the UCF `decode_s` forward.  With no cache present the
    two together make `report` round-trip byte for byte from the committed artifacts alone.
    """
    live = [r for r in index.get("clips", []) if "npy" in r]
    if live:
        return ({"n_clips_decoded": len(live),
                 "n_decode_failures": int(index.get("n_failed", 0)),
                 "decode_failures": [r for r in index["clips"] if "error" in r][:50],
                 "decode_s": float(index.get("wall_s", 0.0))},
                _manifest(cj, index, n))
    pd = prev.get("dataset") or {}
    man = (prev.get("frame_cache") or {}).get("manifest")
    if not man:
        return ({"n_clips_decoded": 0, "n_decode_failures": 0, "decode_failures": [],
                 "decode_s": 0.0}, _manifest(cj, index, n))
    man = {**man, "clips": [{**c, "frame_indices": _Raw(c.get("frame_indices") or [])}
                            for c in man["clips"]]}
    return ({"n_clips_decoded": int(pd.get("n_clips_decoded", 0)),
             "n_decode_failures": int(pd.get("n_decode_failures", 0)),
             "decode_failures": pd.get("decode_failures") or [],
             "decode_s": float(pd.get("decode_s", 0.0))}, man)


def _run_rows(work: Path, cj: dict, y_all: np.ndarray, ref_all: np.ndarray | None) -> list[dict]:
    rows = []
    for p in sorted(work.glob("cpp-*-logits.npy")) + sorted(work.glob("torch-*-logits.npy")):
        tag = p.name[: -len("-logits.npy")]
        meta_p = work / (tag + ".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        scope = meta.get("scope") or tag.split("-")[-1]
        idx = np.array(cj["scopes"][scope]["idx"])
        L = np.load(p).astype(np.float64)
        if len(L) != len(idx):
            print(f"  !! {tag}: {len(L)} logits for a {len(idx)}-clip scope — skipped")
            continue
        y = y_all[idx]
        row = {
            "tag": tag,
            "backend": meta.get("backend", "pytorch" if tag.startswith("torch") else "?"),
            "device": meta.get("device"), "dtype": meta.get("dtype") or meta.get("file_type"),
            "scope": scope, "n_clips": int(len(L)),
            "top1": _topk_acc(L, y, 1), "top5": _topk_acc(L, y, 5),
            "pred_top1": _Raw([int(v) for v in L.argmax(1)]),
            "stats": {k: meta[k] for k in
                      ("model_load_s", "wall_s", "clips_per_s", "weights_mib", "preprocess_s",
                       "encode_s", "head_s", "threads", "batch", "gguf", "tf32", "torch",
                       "loadavg_start", "loadavg_end", "occupancy") if k in meta},
        }
        if ref_all is not None and not tag.startswith("torch-"):
            R = ref_all[idx]
            cm, cn = _cos(L, R)
            row["vs_pytorch"] = {"top1_agreement": float(np.mean(L.argmax(1) == R.argmax(1))),
                                 "logit_cos_mean": cm, "logit_cos_min": cn,
                                 "logit_max_abs_diff": float(np.abs(L - R).max())}
        row["source"] = "measured"
        rows.append(row)
    return rows


def _derived_rows(work: Path, cj: dict, y_all: np.ndarray, ref_all: np.ndarray | None,
                  rows: list[dict], scopes: list[str]) -> list[dict]:
    """Score a full-split run again on a subset, so a CPU row has a GPU twin on the same clips.

    Re-running a 24 777-clip CUDA sweep on 2 478 of those clips would produce the same logits, so
    the subset rows are sliced out of the full run instead and marked `derived`.
    """
    out = []
    full = {r["tag"]: r for r in rows if r["scope"] == "full"}
    for tag, r in full.items():
        L_all = np.load(work / (tag + "-logits.npy")).astype(np.float64)
        for scope in scopes:
            idx = np.array(cj["scopes"][scope]["idx"])
            L, y = L_all[idx], y_all[idx]
            d = {"tag": f"{tag}@{scope}", "backend": r["backend"], "device": r["device"],
                 "dtype": r["dtype"], "scope": scope, "n_clips": int(len(L)),
                 "top1": _topk_acc(L, y, 1), "top5": _topk_acc(L, y, 5),
                 "pred_top1": _Raw([int(v) for v in L.argmax(1)]),
                 "source": f"derived — the {tag} logits scored on the {scope} clips",
                 "stats": {}}
            if ref_all is not None and r["backend"] != "pytorch":
                R = ref_all[idx]
                cm, cn = _cos(L, R)
                d["vs_pytorch"] = {"top1_agreement": float(np.mean(L.argmax(1) == R.argmax(1))),
                                   "logit_cos_mean": cm, "logit_cos_min": cn,
                                   "logit_max_abs_diff": float(np.abs(L - R).max())}
            out.append(d)
    return out


def stage_report(a) -> None:
    work = Path(a.work)
    cj = load_clips(work)
    y_all = np.array([r["label"] for r in cj["clips"]])
    frames_dir = Path(a.frames_dir) if a.frames_dir else shared(DATA) / "frames-val"
    index_p = frames_dir / "index.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else {"clips": [], "n_failed": 0}
    out_p = Path(a.out_json)
    prev = json.loads(out_p.read_text()) if out_p.exists() else {}
    frames, manifest = _frame_facts(index, cj, a.manifest_clips, prev)

    ref_p = work / "torch-cuda1-full-logits.npy"
    if not ref_p.exists():
        cand = sorted(work.glob("torch-*-full-logits.npy"))
        ref_p = cand[0] if cand else None
    ref_all = np.load(ref_p).astype(np.float64) if ref_p else None
    if ref_all is not None and len(ref_all) != len(y_all):
        sys.exit(f"{ref_p} has {len(ref_all)} rows for {len(y_all)} clips")

    rows = _run_rows(work, cj, y_all, ref_all)
    want = sorted({r["scope"] for r in rows if r["scope"] != "full"})
    rows += _derived_rows(work, cj, y_all, ref_all, rows, want)

    # the PyTorch CPU anchor, compared clip for clip against the jepa.cpp CPU f32 row
    anchors = {}
    cpu_ref = work / "torch-cpu-sub100-logits.npy"
    if cpu_ref.exists():
        R = np.load(cpu_ref).astype(np.float64)
        ix100 = np.array(cj["scopes"]["sub100"]["idx"])
        pos = {v: i for i, v in enumerate(cj["scopes"]["sub10"]["idx"])}

        def anchor(L):
            cm, cn = _cos(L, R)
            return {"n_clips": int(len(R)),
                    "top1_agreement": float(np.mean(L.argmax(1) == R.argmax(1))),
                    "logit_cos_mean": cm, "logit_cos_min": cn,
                    "logit_max_abs_diff": float(np.abs(L - R).max())}

        for p in sorted(work.glob("cpp-cpu-*-sub10-logits.npy")):
            anchors[p.name[: -len("-logits.npy")]] = anchor(
                np.load(p).astype(np.float64)[[pos[i] for i in ix100]])
        # The control that gives the row above a scale: PyTorch's *own* fp32 GPU logits against its
        # fp32 CPU logits on the same clips.  Whatever jepa.cpp's f32 CPU row costs is only
        # meaningful next to what a backend change costs PyTorch itself.
        if ref_all is not None:
            anchors["pytorch-cuda (control)"] = anchor(ref_all[ix100])

    # the C++ side reads its class names out of the GGUF, so the two label orders are compared
    # here for every file the sweep actually ran
    gguf_check = {}
    for dt in sorted({(r["stats"].get("gguf") or "").rsplit("-", 1)[-1][:-5]
                      for r in rows if r["stats"].get("gguf")}):
        g = shared("models/gguf") / f"{MODEL}-{dt}.gguf"
        labels = gguf_label_list(g)
        if labels is None:                      # no `gguf` module here — recorded, not asserted
            gguf_check[dt] = "not checked — the `gguf` module is not installed here"
        elif labels != cj["labels"]:
            sys.exit(f"{g.name}: jepa.head.labels is not the checkpoint's id2label order "
                     f"({len(labels)} labels) — every logit column in this sweep would be "
                     "mislabelled")
        else:
            gguf_check[dt] = "identical to id2label order"

    # CPU against CUDA at the same dtype, on the clips both ran: the one comparison that isolates
    # the backend, with the engine and the file held fixed.
    cpu_vs_cuda = {}
    for p in sorted(work.glob("cpp-cpu-*-logits.npy")):
        tag = p.name[: -len("-logits.npy")]
        meta = json.loads((work / (tag + ".json")).read_text())
        gpu = sorted(work.glob(f"cpp-cuda*-{meta['dtype']}-full-logits.npy"))
        if not gpu:
            continue
        idx = np.array(cj["scopes"][meta["scope"]]["idx"])
        A = np.load(p).astype(np.float64)
        B = np.load(gpu[0]).astype(np.float64)[idx]
        cm, cn = _cos(A, B)
        cpu_vs_cuda[tag] = {"cuda_run": gpu[0].name[: -len("-logits.npy")],
                            "dtype": meta["dtype"], "n_clips": int(len(A)),
                            "top1_agreement": float(np.mean(A.argmax(1) == B.argmax(1))),
                            "logit_cos_mean": cm, "logit_cos_min": cn,
                            "logit_max_abs_diff": float(np.abs(A - B).max())}

    payload = {
        "benchmark": "Something-Something-v2 validation accuracy (single view) — "
                     "PyTorch vs jepa.cpp",
        "date": time.strftime("%Y-%m-%d"),
        "commit": _commit(prev, a.restamp),
        "model": {"name": MODEL, "label": LABEL, "hf": "facebook/" + MODEL,
                  "head": "attentive pooler (3 blocks + cross-attention) + 174-way linear",
                  "gguf": f"models/gguf/{MODEL}-<dtype>.gguf"},
        # like `commit`: a re-report from another interpreter must not restamp the sweep's
        # environment (torch build, GPU); `--restamp` takes the current one instead
        "env": prev["env"] if (prev.get("env") and not a.restamp) else _v._env(),
        "protocol": {
            "views": "single view, no test-time augmentation: 1 temporal clip x 1 spatial crop",
            "frames": f"{cj['frames']} frames, {cj['sampling']}",
            "preprocess": "shortest edge -> 292 (bilinear), centre crop 256, /255, "
                          "mean (0.485,0.456,0.406) / std (0.229,0.224,0.225) — the checkpoint's "
                          "own video_preprocessor_config.json, applied by each backend to the same "
                          "THWC uint8 frames",
            "reference": "transformers VJEPA2ForVideoClassification, torch.no_grad, float32, "
                         "TF32 disabled on both matmul and cuDNN, 4 clips per forward (the "
                         "processor is batch-invariant, the model forward moves a logit by ~2e-04 "
                         "and no argmax with it)",
            "metric": "top-1 / top-5 over the 174 SSv2 classes; agreement = fraction of clips where "
                      "the jepa.cpp argmax equals the PyTorch argmax; logit cos = per-clip cosine "
                      "between the two 174-vectors",
            "scopes": {n: f"every {s}th clip of the validation order" if s > 1 else
                          "all validation clips" for n, s in SCOPES.items()},
        },
        "published": PUBLISHED,
        "dataset": {
            "root": DATA, "split": "validation", "n_entries": cj["n_entries"],
            "n_classes": len(cj["labels"]),
            "n_clips_decoded": frames["n_clips_decoded"],
            "n_decode_failures": frames["n_decode_failures"],
            "decode_failures": frames["decode_failures"],
            "n_skipped": len(cj["skipped"]), "skipped": cj["skipped"][:50],
            "decode_s": frames["decode_s"],
            "label_mapping": {**cj["label_mapping"], "gguf_head_labels": gguf_check},
            "clip_ids": _Raw([r["id"] for r in cj["clips"]]),
            "labels_true": _Raw([int(v) for v in y_all]),
            "class_names": cj["labels"],
        },
        # `full` is range(n) by construction, so only the subsets carry an explicit index list
        "scopes": {n: {"step": SCOPES[n], "n": cj["scopes"][n]["n"],
                       **({} if n == "full" else {"idx": _Raw(cj["scopes"][n]["idx"])})}
                   for n in SCOPES},
        "runs": rows,
        "cpu_f32_anchor": anchors,
        "cpu_vs_cuda": cpu_vs_cuda,
        "frame_cache": {"path": _relpath(frames_dir),
                        "note": "git-ignored, 106 GB measured for the validation split, deleted after the sweep; "
                                "`frames` rebuilds it byte for byte",
                        "manifest": manifest},
    }
    out = Path(a.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dumps(payload))
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")
    for r in rows:
        v = r.get("vs_pytorch") or {}
        tail = (f" agree={100*v['top1_agreement']:.2f} cos={v['logit_cos_mean']:.6f}"
                f"/{v['logit_cos_min']:.6f}" if v else "")
        print(f"  {r['tag']:32s} n={r['n_clips']:6d} top1={100*r['top1']:6.2f} "
              f"top5={100*r['top5']:6.2f}{tail}")


# --------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(ROOT / "tmp" / "accuracy-ssv2"),
                    help="scratch directory for logits / clip lists (default tmp/accuracy-ssv2)")
    sub = ap.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("frames", help="decode the validation split to THWC uint8 .npy")
    s.add_argument("--out", default="", help="frame cache (default data/ssv2/frames-val)")
    s.add_argument("--frames", type=int, default=FRAMES)
    s.add_argument("--jobs", type=int, default=48)
    s.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    s.set_defaults(fn=stage_frames)

    s = sub.add_parser("lists", help="freeze the clip order and the CPU subsets")
    s.add_argument("--frames-dir", default="", help="frame cache to index (default data/ssv2/frames-val)")
    s.set_defaults(fn=stage_lists)

    s = sub.add_parser("torch", help="PyTorch fp32 reference logits")
    s.add_argument("--device", default="cuda:1", help="cpu | cuda:N — pick a free card with nvidia-smi")
    s.add_argument("--scope", default="full", choices=list(SCOPES),
                   help="which clips to run (see the scope table above)")
    s.add_argument("--batch", type=int, default=4, help="clips per forward")
    s.add_argument("--prefetch", type=int, default=8,
                   help="preprocessed batches held in flight (bounded: each is ~12 MB per clip)")
    s.add_argument("--threads", type=int, default=CPU_THREADS,
                   help="torch CPU threads: the whole forward with --device cpu, the "
                        "preprocessing only with --device cuda:N")
    s.set_defaults(fn=stage_torch)

    s = sub.add_parser("cpp", help="jepa.cpp logits through jepa-embed --frames-list --logits")
    s.add_argument("--dtype", required=True, help="f32 | f16 | q8_0 | q4_k | ...")
    s.add_argument("--device", default="cuda:1", help="cpu | cuda:N")
    s.add_argument("--scope", default="full", choices=list(SCOPES),
                   help="which clips to run (see the scope table above)")
    s.add_argument("--threads", type=int, default=CPU_THREADS,
                   help=f"CPU threads (default {CPU_THREADS}); unused with --device cuda:N")
    s.set_defaults(fn=stage_cpp)

    s = sub.add_parser("report", help="metrics -> tests/results/accuracy-ssv2.json")
    s.add_argument("--frames-dir", default="",
                   help="frame cache the manifest digests (default data/ssv2/frames-val)")
    s.add_argument("--manifest-clips", type=int, default=16,
                   help="how many clip tensors to digest into the committed manifest")
    s.add_argument("--restamp", action="store_true",
                   help="record the current HEAD as `commit` instead of keeping the revision the "
                        "artifact was first written at")
    s.add_argument("--out-json", default=str(ROOT / "tests" / "results" / "accuracy-ssv2.json"),
                   help="the committed artifact; `bench_accuracy_video.py report` renders its "
                        "tables into docs/accuracy-video.md")
    s.set_defaults(fn=stage_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
