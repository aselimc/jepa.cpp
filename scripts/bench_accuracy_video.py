#!/usr/bin/env python3
"""Video k-NN accuracy + backend agreement: UCF101 subset, PyTorch vs jepa.cpp.

Inference only — the encoders are frozen and nothing is fitted; k-NN and nearest-class-centroid are
look-ups over the frozen features (see scripts/_knn_video.py for the protocol).

Stages (run in this order; each writes into <work>/ and can be re-run on its own):

    scripts/bench_accuracy_video.py lists   --index tmp/frames/index.json
    scripts/bench_accuracy_video.py torch   --model vjepa2-vitl-fpc64-256
    scripts/bench_accuracy_video.py torch   --model vjepa2_1-vitb-384
    scripts/bench_accuracy_video.py cpp     --model vjepa2-vitl-fpc64-256 --dtype f16
    scripts/bench_accuracy_video.py ssv2-torch
    scripts/bench_accuracy_video.py ssv2-cpp --dtype f16
    scripts/bench_accuracy_video.py report  --out-json tests/results/accuracy-video.json
                                            --out-md docs/accuracy-video.md

`lists` fixes the gallery (train) and query (val, test) clip order once, in a JSON that every later
stage indexes into, so all backends see exactly the same clips in exactly the same order.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# The protocol lives in scripts/knn_eval.py (owned by the image-accuracy agent) once that lands;
# until then scripts/_knn_video.py carries an identical private copy.
try:
    import knn_eval as _knn                                          # type: ignore
    if not all(hasattr(_knn, f) for f in ("knn_predict", "centroid_predict", "accuracy", "mean_cosine")):
        raise ImportError("knn_eval lacks the expected entry points")
    KNN_IMPL = "scripts/knn_eval.py"
except ImportError:
    import _knn_video as _knn                                        # type: ignore
    KNN_IMPL = "scripts/_knn_video.py"

MODELS = {
    "vjepa2-vitl-fpc64-256": {
        "kind": "hf", "hf_dir": "models/facebook/vjepa2-vitl-fpc64-256", "crop": 256,
        "gguf": "vjepa2-vitl-fpc64-256-{dtype}.gguf", "label": "V-JEPA 2 ViT-L/16 (fpc64-256)"},
    "vjepa2_1-vitb-384": {
        "kind": "meta", "ckpt": "models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt", "crop": 384,
        "hub": "vjepa2_1_vit_base_384", "gguf": "vjepa2_1-vitb-384-{dtype}.gguf",
        "label": "V-JEPA 2.1 ViT-B/16 @384"},
}
SSV2_MODEL = "vjepa2-vitl-fpc16-256-ssv2"
SSV2_HF_DIR = "models/facebook/vjepa2-vitl-fpc16-256-ssv2"
THREADS = 32
SHARED = Path("/home/overseer2/workdir/jepa.cpp")     # git-ignored checkpoints / datasets / venv


def shared(p: str) -> Path:
    return SHARED / p


# --------------------------------------------------------------------------------------------
# stage: lists
# --------------------------------------------------------------------------------------------
def stage_lists(a) -> None:
    idx = json.loads(Path(a.index).read_text())
    clips = idx["clips"]
    classes = idx["classes"]
    order = {"train": 0, "val": 1, "test": 2}
    clips = sorted(clips, key=lambda r: (order.get(r["split"], 9), r["class"], r["stem"]))
    rows = [{"i": i, "split": r["split"], "class": r["class"], "label": r["label"],
             "stem": r["stem"], "npy": r["npy"], "clip": r["clip"]} for i, r in enumerate(clips)]
    splits = {sp: [r["i"] for r in rows if r["split"] == sp] for sp in ("train", "val", "test")}
    out = {
        "dataset": idx["dataset"], "frames": idx["frames"], "classes": classes,
        "gallery": {"split": "train", "n": len(splits["train"]), "idx": splits["train"]},
        "queries": {
            "test": {"n": len(splits["test"]), "idx": splits["test"]},
            "val": {"n": len(splits["val"]), "idx": splits["val"]},
            "val+test": {"n": len(splits["val"]) + len(splits["test"]),
                         "idx": sorted(splits["val"] + splits["test"])}},
        "order": "sorted by (split: train, val, test) then class then clip stem — the row order of "
                 "every feature .npy this benchmark writes",
        "clips": rows,
    }
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "clips.json").write_text(json.dumps(out, indent=1))
    (work / "clips.txt").write_text("".join(r["npy"] + "\n" for r in rows))
    print(f"{len(rows)} clips ({len(splits['train'])} gallery / {len(splits['val'])} val / "
          f"{len(splits['test'])} test), {len(classes)} classes -> {work/'clips.json'}")


def load_clips(work: Path) -> dict:
    return json.loads((work / "clips.json").read_text())


# --------------------------------------------------------------------------------------------
# stage: torch  (PyTorch pooled-mean features)
# --------------------------------------------------------------------------------------------
def _torch_setup():
    import torch
    torch.set_num_threads(THREADS)
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    return torch


def _load_hf_encoder(name):
    """VJEPA2Model + its video processor; features = last_hidden_state.mean(1)."""
    import torch
    from transformers import AutoVideoProcessor, VJEPA2Model
    d = shared(MODELS[name]["hf_dir"])
    proc = AutoVideoProcessor.from_pretrained(d)
    model = VJEPA2Model.from_pretrained(d, dtype=torch.float32).eval()

    def fwd(frames):
        x = proc(videos=frames, return_tensors="pt")["pixel_values_videos"].float()
        with torch.inference_mode():
            return model(pixel_values_videos=x).last_hidden_state[0].mean(0).numpy()
    return fwd


def _load_meta_encoder(name):
    """V-JEPA 2.1 through the Meta code path (torch.hub local clone), exactly as dump_reference.py."""
    import torch
    from transformers import VJEPA2VideoProcessor
    src = shared("tmp/vjepa2-src")
    crop = MODELS[name]["crop"]
    encoder, _pred = torch.hub.load(str(src), MODELS[name]["hub"], source="local", pretrained=False)
    sd = torch.load(shared(MODELS[name]["ckpt"]), map_location="cpu", weights_only=False)
    clean = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd["ema_encoder"].items()}
    encoder.load_state_dict(clean, strict=True)
    encoder = encoder.float().eval()
    del sd
    proc = VJEPA2VideoProcessor(crop_size=crop)

    def fwd(frames):
        x = proc(videos=frames, return_tensors="pt")["pixel_values_videos"].float()
        x = x.permute(0, 2, 1, 3, 4).contiguous()          # NTCHW -> NCTHW for the Meta encoder
        with torch.inference_mode():
            return encoder(x)[0].mean(0).numpy()
    return fwd


def stage_torch(a) -> None:
    _torch_setup()
    work = Path(a.work)
    cj = load_clips(work)
    name = a.model
    t0 = time.time()
    fwd = _load_hf_encoder(name) if MODELS[name]["kind"] == "hf" else _load_meta_encoder(name)
    load_s = time.time() - t0
    print(f"{name}: loaded in {load_s:.1f}s, {len(cj['clips'])} clips, {THREADS} threads", flush=True)

    feats, t0 = [], time.time()
    for i, r in enumerate(cj["clips"], 1):
        feats.append(fwd(np.load(r["npy"])))
        if i % 25 == 0 or i == len(cj["clips"]):
            print(f"  {i}/{len(cj['clips'])}  {time.time()-t0:.1f}s", flush=True)
    wall = time.time() - t0
    F = np.stack(feats).astype(np.float32)
    out = work / f"{name}-torch.npy"
    np.save(out, F)
    (work / f"{name}-torch.json").write_text(json.dumps(
        {"model": name, "backend": "pytorch", "dtype": "f32", "n_clips": len(F), "dim": int(F.shape[1]),
         "threads": THREADS, "model_load_s": round(load_s, 2), "wall_s": round(wall, 1),
         "clips_per_s": round(len(F) / wall, 3)}, indent=1))
    print(f"{out} {F.shape} in {wall:.1f}s ({len(F)/wall:.2f} clips/s)")


# --------------------------------------------------------------------------------------------
# stage: cpp  (jepa.cpp pooled-mean features through the batch driver)
# --------------------------------------------------------------------------------------------
def stage_cpp(a) -> None:
    work = Path(a.work)
    load_clips(work)
    name, dt = a.model, a.dtype
    gguf = shared("models/gguf") / MODELS[name]["gguf"].format(dtype=dt)
    if not gguf.exists():
        sys.exit(f"missing {gguf}")
    out = work / f"{name}-{dt}.npy"
    cmd = [str(ROOT / "build" / "jepa-embed-clips"), "-m", str(gguf), "-l", str(work / "clips.txt"),
           "-o", str(out), "--pool", "mean", "-t", str(THREADS), "--json", str(work / f"{name}-{dt}.json")]
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"{out} in {time.time()-t0:.1f}s")


# --------------------------------------------------------------------------------------------
# stage: ssv2-torch / ssv2-cpp  (SSv2 classification head, query clips only)
# --------------------------------------------------------------------------------------------
def stage_ssv2_torch(a) -> None:
    torch = _torch_setup()
    from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification
    work = Path(a.work)
    cj = load_clips(work)
    qidx = cj["queries"]["val+test"]["idx"]
    d = shared(SSV2_HF_DIR)
    t0 = time.time()
    proc = AutoVideoProcessor.from_pretrained(d)
    model = VJEPA2ForVideoClassification.from_pretrained(d, dtype=torch.float32).eval()
    load_s = time.time() - t0
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"ssv2 head: loaded in {load_s:.1f}s, {len(qidx)} query clips, {model.config.num_labels} classes", flush=True)

    logits, t0 = [], time.time()
    for n, i in enumerate(qidx, 1):
        fr = np.load(cj["clips"][i]["npy"])
        x = proc(videos=fr, return_tensors="pt")["pixel_values_videos"].float()
        with torch.inference_mode():
            logits.append(model(pixel_values_videos=x).logits[0].numpy())
        if n % 25 == 0 or n == len(qidx):
            print(f"  {n}/{len(qidx)}  {time.time()-t0:.1f}s", flush=True)
    wall = time.time() - t0
    L = np.stack(logits).astype(np.float32)
    np.save(work / "ssv2-torch-logits.npy", L)
    (work / "ssv2-torch.json").write_text(json.dumps(
        {"model": SSV2_MODEL, "backend": "pytorch", "dtype": "f32", "n_clips": len(L),
         "n_classes": int(L.shape[1]), "query_idx": qidx, "labels": labels, "threads": THREADS,
         "model_load_s": round(load_s, 2), "wall_s": round(wall, 1),
         "clips_per_s": round(len(L) / wall, 3)}, indent=1))
    print(f"ssv2 torch logits {L.shape} in {wall:.1f}s ({len(L)/wall:.2f} clips/s)")


def stage_ssv2_cpp(a) -> None:
    work = Path(a.work)
    cj = load_clips(work)
    qidx = cj["queries"]["val+test"]["idx"]
    lst = work / "query.txt"
    lst.write_text("".join(cj["clips"][i]["npy"] + "\n" for i in qidx))
    gguf = shared("models/gguf") / f"{SSV2_MODEL}-{a.dtype}.gguf"
    if not gguf.exists():
        sys.exit(f"missing {gguf}")
    cmd = [str(ROOT / "build" / "jepa-embed-clips"), "-m", str(gguf), "-l", str(lst),
           "-o", str(work / f"ssv2-{a.dtype}-feats.npy"), "--logits", str(work / f"ssv2-{a.dtype}-logits.npy"),
           "-t", str(THREADS), "--json", str(work / f"ssv2-{a.dtype}.json")]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------------------------
# stage: report
# --------------------------------------------------------------------------------------------
def _eval_one(F, cj, n_classes, k):
    """k-NN + centroid predictions for every query split of one feature matrix."""
    gid = np.array(cj["gallery"]["idx"])
    glab = np.array([cj["clips"][i]["label"] for i in gid])
    res = {}
    for sp, q in cj["queries"].items():
        qid = np.array(q["idx"])
        qlab = np.array([cj["clips"][i]["label"] for i in qid])
        knn = _knn.knn_predict(F[gid], glab, F[qid], n_classes, k=k)
        cen = _knn.centroid_predict(F[gid], glab, F[qid], n_classes)
        res[sp] = {"n": int(len(qid)), "knn_top1": _knn.accuracy(knn, qlab),
                   "centroid_top1": _knn.accuracy(cen, qlab),
                   "knn_pred": knn.tolist(), "centroid_pred": cen.tolist(), "labels": qlab.tolist()}
    return res


def stage_report(a) -> None:
    work = Path(a.work)
    cj = load_clips(work)
    n_classes = len(cj["classes"])
    k = a.k

    results = {}
    for name in MODELS:
        per_backend = {}
        for dt in ["torch"] + a.dtypes.split(","):
            f = work / f"{name}-{dt}.npy"
            if not f.exists():
                continue
            F = np.load(f).astype(np.float64)
            stats = json.loads((work / f"{name}-{dt}.json").read_text()) if (work / f"{name}-{dt}.json").exists() else {}
            per_backend[dt] = {"eval": _eval_one(F, cj, n_classes, k), "stats": stats, "feats": F}
        if per_backend:
            results[name] = per_backend

    # agreement + feature cosine of every jepa.cpp dtype against the PyTorch reference
    for name, per in results.items():
        ref = per.get("torch")
        for dt, r in per.items():
            r["agreement"] = {}
            if ref is None:
                continue
            for sp, q in cj["queries"].items():
                qid = np.array(q["idx"])
                same_knn = float(np.mean(np.array(r["eval"][sp]["knn_pred"]) == np.array(ref["eval"][sp]["knn_pred"])))
                same_cen = float(np.mean(np.array(r["eval"][sp]["centroid_pred"]) == np.array(ref["eval"][sp]["centroid_pred"])))
                r["agreement"][sp] = {"knn": same_knn, "centroid": same_cen,
                                      "feat_cos": _knn.mean_cosine(r["feats"][qid], ref["feats"][qid])}
            r["agreement"]["all_clips"] = {"feat_cos": _knn.mean_cosine(r["feats"], ref["feats"])}

    # ---- SSv2 head agreement (backend fidelity on real video, NOT a task accuracy)
    ssv2 = {}
    ref_p = work / "ssv2-torch-logits.npy"
    if ref_p.exists():
        ref_l = np.load(ref_p).astype(np.float64)
        meta = json.loads((work / "ssv2-torch.json").read_text())
        ref_top1 = ref_l.argmax(1)
        ref_top5 = np.argsort(-ref_l, axis=1)[:, :5]
        ssv2 = {"n_clips": int(len(ref_l)), "n_classes": int(ref_l.shape[1]),
                "pytorch": {"stats": {kk: meta[kk] for kk in ("model_load_s", "wall_s", "clips_per_s", "threads")}},
                "backends": {}}
        for dt in a.dtypes.split(","):
            p = work / f"ssv2-{dt}-logits.npy"
            if not p.exists():
                continue
            L = np.load(p).astype(np.float64)
            t1 = L.argmax(1)
            t5 = np.argsort(-L, axis=1)[:, :5]
            ov = np.mean([len(set(t5[i]) & set(ref_top5[i])) / 5.0 for i in range(len(L))])
            in5 = float(np.mean([ref_top1[i] in set(t5[i]) for i in range(len(L))]))
            stats = json.loads((work / f"ssv2-{dt}.json").read_text()) if (work / f"ssv2-{dt}.json").exists() else {}
            ssv2["backends"][dt] = {
                "top1_agreement": float(np.mean(t1 == ref_top1)),
                "top5_overlap": float(ov),
                "pytorch_top1_in_cpp_top5": in5,
                "logit_max_abs_diff": float(np.abs(L - ref_l).max()),
                "logit_cos": _knn.mean_cosine(L, ref_l),
                "stats": stats}

    payload = {
        "benchmark": "video k-NN accuracy + backend agreement (UCF101 subset)",
        "date": time.strftime("%Y-%m-%d"),
        "dataset": {"root": cj["dataset"], "classes": cj["classes"], "frames_per_clip": cj["frames"],
                    "gallery": {"split": "train", "n": cj["gallery"]["n"]},
                    "queries": {sp: cj["queries"][sp]["n"] for sp in cj["queries"]},
                    "frame_sampling": "idx = round(linspace(0, T_total-1, 16)) over all PyAV rgb24 frames"},
        "protocol": {"impl": KNN_IMPL, "feature": "mean over encoder tokens (pooled_mean), L2-normalized",
                     "knn": {"k": k, "similarity": "cosine", "weight": "exp(sim / 0.07)"},
                     "centroid": "nearest L2-normalized class mean of the gallery",
                     "training": "none — frozen features, look-up only"},
        "threads": THREADS,
        "models": {}, "ssv2_head": ssv2,
    }
    for name, per in results.items():
        payload["models"][name] = {"label": MODELS[name]["label"], "backends": {}}
        for dt, r in per.items():
            payload["models"][name]["backends"][dt] = {
                "backend": "pytorch" if dt == "torch" else "jepa.cpp", "dtype": "f32" if dt == "torch" else dt,
                "dim": int(r["feats"].shape[1]),
                "splits": {sp: {"n": v["n"], "knn_top1": v["knn_top1"], "centroid_top1": v["centroid_top1"]}
                           for sp, v in r["eval"].items()},
                "agreement": r["agreement"],
                "stats": {kk: r["stats"].get(kk) for kk in
                          ("model_load_s", "wall_s", "clips_per_s", "weights_mib", "encode_s")
                          if kk in r["stats"]},
            }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(payload, indent=1))
    print(f"wrote {a.out_json}")
    if a.out_md:
        Path(a.out_md).write_text(render_md(payload, cj))
        print(f"wrote {a.out_md}")


def _pct(x):
    return "—" if x is None else f"{100*x:.1f}"


def render_md(p: dict, cj: dict) -> str:
    L = []
    A = L.append
    A("# Accuracy — video k-NN on the UCF101 subset (PyTorch vs jepa.cpp)\n")
    A(f"Frozen-feature evaluation, {p['date']}. **Inference only** — nothing is trained: the encoders "
      "are frozen and both metrics are look-ups over their pooled clip features.\n")
    A(f"- **Dataset** `{p['dataset']['root']}` — {len(p['dataset']['classes'])} classes, "
      f"gallery = train ({p['dataset']['gallery']['n']} clips), queries = "
      + " / ".join(f"{sp} ({n})" for sp, n in p['dataset']['queries'].items()) + ".")
    A(f"- **Clip** {p['dataset']['frames_per_clip']} frames, `{p['dataset']['frame_sampling']}`, decoded once "
      "with PyAV to a THWC uint8 `.npy` that *both* backends read, so the two see identical pixels.")
    A(f"- **Feature** {p['protocol']['feature']}.")
    A(f"- **k-NN** k = {p['protocol']['knn']['k']}, cosine similarity, DINO-style weighted vote "
      f"(`{p['protocol']['knn']['weight']}`). **Centroid** = {p['protocol']['centroid']} (no hyper-parameters).")
    A(f"- **Agreement** = fraction of query clips where the jepa.cpp k-NN prediction equals the PyTorch one; "
      "**feat cos** = mean per-clip cosine between the two backends' feature vectors.")
    A(f"- {p['threads']} threads everywhere; protocol implemented in `{p['protocol']['impl']}`.\n")

    for name, m in p["models"].items():
        A(f"## {m['label']} — `{name}`\n")
        A("| backend | dtype | query split | k-NN top-1 % | centroid top-1 % | k-NN agreement % | feat cos | clips/s |")
        A("|---|---|---|--:|--:|--:|--:|--:|")
        for dt, b in m["backends"].items():
            for sp in ("test", "val", "val+test"):
                if sp not in b["splits"]:
                    continue
                s = b["splits"][sp]
                ag = b["agreement"].get(sp, {})
                agr = "ref" if dt == "torch" else _pct(ag.get("knn"))
                cos = "ref" if dt == "torch" else (f"{ag['feat_cos']:.6f}" if "feat_cos" in ag else "—")
                cps = b["stats"].get("clips_per_s")
                A(f"| {b['backend']} | {b['dtype']} | {sp} ({s['n']}) | {_pct(s['knn_top1'])} | "
                  f"{_pct(s['centroid_top1'])} | {agr} | {cos} | {'' if cps is None else f'{cps:.2f}'} |")
        A("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(ROOT / "tmp" / "accuracy-video"))
    sub = ap.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("lists"); s.add_argument("--index", default=str(ROOT / "tmp" / "frames" / "index.json")); s.set_defaults(fn=stage_lists)
    s = sub.add_parser("torch"); s.add_argument("--model", required=True, choices=list(MODELS)); s.set_defaults(fn=stage_torch)
    s = sub.add_parser("cpp"); s.add_argument("--model", required=True, choices=list(MODELS))
    s.add_argument("--dtype", required=True); s.set_defaults(fn=stage_cpp)
    s = sub.add_parser("ssv2-torch"); s.set_defaults(fn=stage_ssv2_torch)
    s = sub.add_parser("ssv2-cpp"); s.add_argument("--dtype", required=True); s.set_defaults(fn=stage_ssv2_cpp)
    s = sub.add_parser("report")
    s.add_argument("--k", type=int, default=20)
    s.add_argument("--dtypes", default="f32,f16,q8_0")
    s.add_argument("--out-json", default=str(ROOT / "tests" / "results" / "accuracy-video.json"))
    s.add_argument("--out-md", default=str(ROOT / "docs" / "accuracy-video.md"))
    s.set_defaults(fn=stage_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
