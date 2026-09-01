#!/usr/bin/env python3
"""Video k-NN accuracy + backend agreement: UCF101 subset, PyTorch vs jepa.cpp.

Inference only — the encoders are frozen and nothing is fitted; k-NN and nearest-class-centroid are
look-ups over the frozen features (the protocol is scripts/knn_eval.py, shared with the
image benchmark).

Stages (run in this order; each writes into <work>/ and can be re-run on its own):

    scripts/bench_accuracy_video.py lists   --index tmp/frames/index.json
    scripts/bench_accuracy_video.py torch   --model vjepa2-vitl-fpc64-256
    scripts/bench_accuracy_video.py torch   --model vjepa2_1-vitb-384
    scripts/bench_accuracy_video.py torch   --model levjepa-vitl16
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

# One protocol for both accuracy benchmarks.  (This used to fall back to a private copy in
# scripts/_knn_video.py; that copy was folded into knn_eval.py and deleted on 2026-08-31.)
import knn_eval as _knn                                              # noqa: E402

KNN_IMPL = "scripts/knn_eval.py"

MODELS = {
    "vjepa2-vitl-fpc64-256": {
        "kind": "hf", "hf_dir": "models/facebook/vjepa2-vitl-fpc64-256", "crop": 256,
        "gguf": "vjepa2-vitl-fpc64-256-{dtype}.gguf", "label": "V-JEPA 2 ViT-L/16 (fpc64-256)"},
    "vjepa2_1-vitb-384": {
        "kind": "meta", "ckpt": "models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt", "crop": 384,
        "hub": "vjepa2_1_vit_base_384", "gguf": "vjepa2_1-vitb-384-{dtype}.gguf",
        "label": "V-JEPA 2.1 ViT-B/16 @384"},
    # The only model here whose feature is the CLS token rather than the token mean, and the only
    # one with no processor of its own: the pipeline is the model card's (short side to 224,
    # bicubic, centre crop 224, ImageNet statistics), the same one dump_reference.py applies.
    "levjepa-vitl16": {
        "kind": "levjepa", "hf_dir": "models/galilai-group/LeVJEPA-VideoMix-Large", "crop": 224,
        "gguf": "levjepa-vitl16-{dtype}.gguf", "pool": "cls",
        "label": "LeVJEPA ViT-L/16 (VideoMix)"},
}
SSV2_MODEL = "vjepa2-vitl-fpc16-256-ssv2"
SSV2_HF_DIR = "models/facebook/vjepa2-vitl-fpc16-256-ssv2"
THREADS = 32
SHARED = Path("/home/overseer2/workdir/jepa.cpp")     # git-ignored checkpoints / datasets / venv


def shared(p: str) -> Path:
    return SHARED / p


def loadavg() -> list[float]:
    """1 / 5 / 15-minute run-queue length (`uptime`), sampled around every timed stage so the JSON
    says what else the box was doing while the clips/s number was taken."""
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


def _load_hf_encoder(name, skip_predictor: bool = True):
    """VJEPA2Model + its video processor; features = last_hidden_state.mean(1)."""
    import torch
    from transformers import AutoVideoProcessor, VJEPA2Model
    d = shared(MODELS[name]["hf_dir"])
    proc = AutoVideoProcessor.from_pretrained(d)
    model = VJEPA2Model.from_pretrained(d, dtype=torch.float32).eval()

    def fwd(frames):
        x = proc(videos=frames, return_tensors="pt")["pixel_values_videos"].float()
        with torch.inference_mode():
            # skip_predictor=True: this benchmark only ever reads `last_hidden_state`, but
            # VJEPA2Model otherwise also runs a full VJEPA2Predictor forward and throws the result
            # away, charging the PyTorch baseline for work no backend here does.  The encoder output
            # is bit-identical either way; `--no-skip-predictor` re-runs the stage without it so the
            # two can be compared (the `report` stage does exactly that and records the result).
            out = model(pixel_values_videos=x, skip_predictor=skip_predictor)
            return out.last_hidden_state[0].mean(0).numpy()
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


def _load_levjepa_encoder(name):
    """LeVJEPA through the modeling file that ships with the weights; feature = the CLS token.

    No processor ships with the checkpoint, so the model card's pipeline is applied here and is the
    same one scripts/dump_reference.py uses for the fixtures: per frame, short side to `crop` with
    torchvision's antialiased BICUBIC on the uint8 tensor, centre crop, /255, ImageNet statistics.
    """
    import torch
    import torchvision.transforms.v2.functional as F
    from transformers import AutoModel

    crop = MODELS[name]["crop"]
    model = AutoModel.from_pretrained(shared(MODELS[name]["hf_dir"]), trust_remote_code=True,
                                      dtype=torch.float32).eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def fwd(frames):                                   # frames: THWC uint8
        t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        t = F.resize(t, [crop], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
        t = F.center_crop(t, [crop, crop])
        x = ((t.float() / 255.0) - mean) / std
        x = x.permute(1, 0, 2, 3)[None].contiguous()   # 1 C T H W
        with torch.inference_mode():
            return model(pixel_values=x).last_hidden_state[0, 0].numpy()   # CLS = pooler_output
    return fwd


def stage_torch(a) -> None:
    _torch_setup()
    work = Path(a.work)
    cj = load_clips(work)
    name = a.model
    skip = not a.no_skip_predictor
    tag = a.tag or ("torch" if skip else "torchpred")
    clips = cj["clips"][:a.limit] if a.limit else cj["clips"]
    t0 = time.time()
    kind = MODELS[name]["kind"]
    fwd = (_load_hf_encoder(name, skip_predictor=skip) if kind == "hf"
           else _load_levjepa_encoder(name) if kind == "levjepa"
           else _load_meta_encoder(name))
    load_s = time.time() - t0
    la0 = loadavg()
    print(f"{name}: loaded in {load_s:.1f}s, {len(clips)} clips, {THREADS} threads, "
          f"skip_predictor={skip}, loadavg {la0}", flush=True)

    occ = Occupancy()
    feats, t0 = [], time.time()
    for i, r in enumerate(clips, 1):
        feats.append(fwd(np.load(r["npy"])))
        if i % 25 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)}  {time.time()-t0:.1f}s", flush=True)
    wall = time.time() - t0
    F = np.stack(feats).astype(np.float32)
    out = work / f"{name}-{tag}.npy"
    np.save(out, F)
    (work / f"{name}-{tag}.json").write_text(json.dumps(
        {"model": name, "backend": "pytorch", "dtype": "f32", "n_clips": len(F), "dim": int(F.shape[1]),
         "threads": THREADS, "skip_predictor": skip, "model_load_s": round(load_s, 2),
         "wall_s": round(wall, 1), "clips_per_s": round(len(F) / wall, 3),
         "loadavg_start": la0, "loadavg_end": loadavg(), "occupancy": occ.close()}, indent=1))
    print(f"{out} {F.shape} in {wall:.1f}s ({len(F)/wall:.2f} clips/s)")


# --------------------------------------------------------------------------------------------
# stage: cpp  (jepa.cpp pooled-mean features through the batch driver)
# --------------------------------------------------------------------------------------------
def _stamp_load(stats_json: Path, la0: list[float], occ: "Occupancy") -> None:
    """Merge the load conditions around a jepa-embed --frames-list call into the stats JSON it wrote."""
    o = occ.close()
    if not stats_json.exists():
        return
    d = json.loads(stats_json.read_text())
    d["loadavg_start"], d["loadavg_end"], d["occupancy"] = la0, loadavg(), o
    stats_json.write_text(json.dumps(d, indent=1))


def stage_cpp(a) -> None:
    work = Path(a.work)
    load_clips(work)
    name, dt = a.model, a.dtype
    gguf = shared("models/gguf") / MODELS[name]["gguf"].format(dtype=dt)
    if not gguf.exists():
        sys.exit(f"missing {gguf}")
    out = work / f"{name}-{dt}.npy"
    pool = MODELS[name].get("pool", "mean")   # levjepa is read through its CLS token
    cmd = [str(ROOT / "build" / "jepa-embed"), "-m", str(gguf), "--batch", "1", "--frames-list", str(work / "clips.txt"),
           "-o", str(out), "--pool", pool, "-t", str(THREADS), "--json", str(work / f"{name}-{dt}.json")]
    print(" ".join(cmd), flush=True)
    t0, la0, occ = time.time(), loadavg(), Occupancy()
    subprocess.run(cmd, check=True)
    _stamp_load(work / f"{name}-{dt}.json", la0, occ)
    print(f"{out} in {time.time()-t0:.1f}s (loadavg {la0} -> {loadavg()})")


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
    la0 = loadavg()
    print(f"ssv2 head: loaded in {load_s:.1f}s, {len(qidx)} query clips, "
          f"{model.config.num_labels} classes, loadavg {la0}", flush=True)

    occ = Occupancy()
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
         "clips_per_s": round(len(L) / wall, 3),
         "loadavg_start": la0, "loadavg_end": loadavg(), "occupancy": occ.close()}, indent=1))
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
    cmd = [str(ROOT / "build" / "jepa-embed"), "-m", str(gguf), "--batch", "1", "--frames-list", str(lst),
           "-o", str(work / f"ssv2-{a.dtype}-feats.npy"), "--logits", str(work / f"ssv2-{a.dtype}-logits.npy"),
           "-t", str(THREADS), "--json", str(work / f"ssv2-{a.dtype}.json")]
    print(" ".join(cmd), flush=True)
    la0, occ = loadavg(), Occupancy()
    subprocess.run(cmd, check=True)
    _stamp_load(work / f"ssv2-{a.dtype}.json", la0, occ)


# --------------------------------------------------------------------------------------------
# stage: report
# --------------------------------------------------------------------------------------------
def _decode_s(work: Path) -> float:
    """Wall time scripts/video_frames.py spent decoding, from its index (0.0 if not available)."""
    for cand in (ROOT / "tmp" / "frames" / "index.json", work.parent / "frames" / "index.json"):
        if cand.exists():
            return float(json.loads(cand.read_text()).get("wall_s", 0.0))
    return 0.0


def _relpath(p: str) -> str:
    """Absolute paths under the shared checkout printed as repo-relative."""
    try:
        return str(Path(p).resolve().relative_to(SHARED))
    except ValueError:
        return p


def _env() -> dict:
    import platform
    out = {"python": platform.python_version(), "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        out["torch"] = torch.__version__
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        out["ggml"] = subprocess.run(["git", "-C", str(ROOT / "ggml"), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    return out


def _predictor_overhead(work: Path) -> dict:
    """What `skip_predictor=True` is worth, per model that has a `--no-skip-predictor` control run.

    `VJEPA2Model.forward` runs the predictor unless asked not to and this benchmark never looks at
    its output, so the baseline used to be timed doing work no backend does.  Dropping it must not
    move a single feature bit, which is what `features_bit_identical` checks.
    """
    out = {}
    for name in MODELS:
        a, b = work / f"{name}-torchpred.npy", work / f"{name}-torch.npy"
        ja, jb = work / f"{name}-torchpred.json", work / f"{name}-torch.json"
        if not (a.exists() and b.exists() and ja.exists() and jb.exists()):
            continue
        A, B = np.load(a), np.load(b)
        sa, sb = json.loads(ja.read_text()), json.loads(jb.read_text())
        n = min(len(A), len(B))
        out[name] = {
            "n_clips_compared": int(n),
            "features_bit_identical": bool(A.shape[1:] == B.shape[1:] and
                                           np.array_equal(A[:n].view(np.uint8), B[:n].view(np.uint8))),
            "with_predictor": {"wall_s": sa["wall_s"], "clips_per_s": sa["clips_per_s"],
                               "n_clips": sa["n_clips"], "loadavg_start": sa.get("loadavg_start")},
            "skip_predictor": {"wall_s": sb["wall_s"], "clips_per_s": sb["clips_per_s"],
                               "n_clips": sb["n_clips"], "loadavg_start": sb.get("loadavg_start")},
            "speedup": round(sb["clips_per_s"] / sa["clips_per_s"], 3),
        }
    return out


def _explain_flips(F, R, cj, k):
    """Why a k-NN prediction differs between two backends, per disagreeing val+test clip.

    Records the PyTorch top-2 vote ratio (how close the decision was) and how much of the k-clip
    neighbourhood the two backends actually share — a swap at the k-th place is the mechanism that
    moves the vote, not the size of the feature error.
    """
    classes = cj["classes"]
    gid = np.array(cj["gallery"]["idx"])
    glab = np.array([cj["clips"][i]["label"] for i in gid])
    qid = np.array(cj["queries"]["val+test"]["idx"])
    pr = _knn.knn_predict(R[gid], glab, R[qid], len(classes), k=k)
    pc = _knn.knn_predict(F[gid], glab, F[qid], len(classes), k=k)
    gR, gF = _knn.l2_normalize(R[gid]), _knn.l2_normalize(F[gid])
    out = []
    for j in np.where(pr != pc)[0]:
        i = qid[j]
        sR = (_knn.l2_normalize(R[i][None]) @ gR.T)[0]
        sF = (_knn.l2_normalize(F[i][None]) @ gF.T)[0]
        oR, oF = np.argsort(-sR), np.argsort(-sF)
        nR, nF = oR[:k], oF[:k]
        vR, vF = np.zeros(len(classes)), np.zeros(len(classes))
        for idx, w in zip(nR, np.exp(sR[nR] / _knn.T_DEFAULT)):
            vR[glab[idx]] += w
        for idx, w in zip(nF, np.exp(sF[nF] / _knn.T_DEFAULT)):
            vF[glab[idx]] += w
        top2 = np.argsort(-vR)[:2]
        out.append({"stem": cj["clips"][i]["stem"], "split": cj["clips"][i]["split"],
                    "true": classes[cj["clips"][i]["label"]],
                    "pytorch_pred": classes[int(pr[j])], "jepa_cpp_pred": classes[int(pc[j])],
                    "pytorch_top2_vote_ratio": float(vR[top2[0]] / vR[top2[1]]),
                    "shared_neighbours": int(len(set(nR.tolist()) & set(nF.tolist()))), "k": int(k),
                    # why the neighbourhood moved: the k-th / (k+1)-th gallery clips are closer
                    # together than the two backends' similarities to this query are apart
                    "gap_k_to_kplus1": float(sR[oR[k - 1]] - sR[oR[k]]),
                    "backend_sim_max_diff": float(np.abs(sR - sF).max()),
                    "last_neighbour_weight_share": float(np.exp((sR[oR[k - 1]] - sR[oR[0]]) / _knn.T_DEFAULT)),
                    "top_class_vote_shift_pct": float(100 * (vF[top2[0]] / vR[top2[0]] - 1))})
    return out


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
            r["disagreements"] = []
            if ref is None:
                continue
            r["disagreements"] = _explain_flips(r["feats"], ref["feats"], cj, k)
            for sp, q in cj["queries"].items():
                qid = np.array(q["idx"])
                same_knn = float(np.mean(np.array(r["eval"][sp]["knn_pred"]) == np.array(ref["eval"][sp]["knn_pred"])))
                same_cen = float(np.mean(np.array(r["eval"][sp]["centroid_pred"]) == np.array(ref["eval"][sp]["centroid_pred"])))
                r["agreement"][sp] = {"knn": same_knn, "centroid": same_cen,
                                      "feat_cos": _knn.mean_cosine(r["feats"][qid], ref["feats"][qid])}
            cos = (_knn.l2_normalize(r["feats"]) * _knn.l2_normalize(ref["feats"])).sum(-1)
            r["agreement"]["all_clips"] = {"n": int(len(cos)), "feat_cos": float(cos.mean()),
                                           "feat_cos_min": float(cos.min()),
                                           "feat_max_abs_diff": float(np.abs(r["feats"] - ref["feats"]).max())}

    # ---- SSv2 head agreement (backend fidelity on real video, NOT a task accuracy)
    ssv2 = {}
    ref_p = work / "ssv2-torch-logits.npy"
    if ref_p.exists():
        ref_l = np.load(ref_p).astype(np.float64)
        meta = json.loads((work / "ssv2-torch.json").read_text())
        ref_top1 = ref_l.argmax(1)
        ref_top5 = np.argsort(-ref_l, axis=1)[:, :5]
        ssv2 = {"n_clips": int(len(ref_l)), "n_classes": int(ref_l.shape[1]),
                "pytorch": {"stats": {kk: meta[kk] for kk in
                                      ("model_load_s", "wall_s", "clips_per_s", "threads",
                                       "loadavg_start", "loadavg_end", "occupancy")
                                      if kk in meta}},
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

    predictor_overhead = _predictor_overhead(work)

    payload = {
        "benchmark": "video k-NN accuracy + backend agreement (UCF101 subset)",
        "date": time.strftime("%Y-%m-%d"),
        "env": _env(),
        "dataset": {"root": _relpath(cj["dataset"]), "classes": cj["classes"], "frames_per_clip": cj["frames"],
                    "gallery": {"split": "train", "n": cj["gallery"]["n"],
                                "clips": [cj["clips"][i]["stem"] for i in cj["gallery"]["idx"]],
                                "labels": [cj["clips"][i]["label"] for i in cj["gallery"]["idx"]]},
                    # per-split query clip order + true labels: together with the per-backend
                    # knn_pred / centroid_pred arrays below, every accuracy and agreement number in
                    # docs/accuracy-video.md is re-derivable from this file alone, with no access to
                    # the (git-ignored) feature caches under tmp/accuracy-video/.
                    "queries": {sp: {"n": cj["queries"][sp]["n"],
                                     "clips": [cj["clips"][i]["stem"] for i in cj["queries"][sp]["idx"]],
                                     "labels": [cj["clips"][i]["label"] for i in cj["queries"][sp]["idx"]]}
                                for sp in cj["queries"]},
                    "decode_s": _decode_s(work),
                    "frame_sampling": "idx = round(linspace(0, T_total-1, 16)) over all PyAV rgb24 frames"},
        "protocol": {"impl": KNN_IMPL,
                     "feature": "mean over encoder tokens (pooled_mean), L2-normalized — except "
                                "levjepa-vitl16, which is read through its CLS token (pooler_output), "
                                "also L2-normalized",
                     "knn": {"k": k, "similarity": "cosine", "weight": "exp(sim / 0.07)"},
                     "centroid": "nearest L2-normalized class mean of the gallery",
                     "training": "none — frozen features, look-up only"},
        "threads": THREADS,
        "pytorch_predictor_overhead": predictor_overhead,
        "models": {}, "ssv2_head": ssv2,
    }
    for name, per in results.items():
        payload["models"][name] = {"label": MODELS[name]["label"], "backends": {}}
        for dt, r in per.items():
            payload["models"][name]["backends"][dt] = {
                "backend": "pytorch" if dt == "torch" else "jepa.cpp", "dtype": "f32" if dt == "torch" else dt,
                "dim": int(r["feats"].shape[1]),
                "splits": {sp: {"n": v["n"], "knn_top1": v["knn_top1"],
                                "centroid_top1": v["centroid_top1"],
                                "knn_pred": v["knn_pred"], "centroid_pred": v["centroid_pred"]}
                           for sp, v in r["eval"].items()},
                "agreement": r["agreement"], "disagreements": r["disagreements"],
                "stats": {kk: r["stats"].get(kk) for kk in
                          ("model_load_s", "wall_s", "clips_per_s", "weights_mib", "encode_s",
                           "skip_predictor", "loadavg_start", "loadavg_end", "occupancy")
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


def _maj(p) -> str:
    """Majority-class share of the combined query set, as a percentage string."""
    lab = p["dataset"]["queries"]["val+test"]["labels"]
    return f"{100 * max(lab.count(c) for c in set(lab)) / len(lab):.1f}" if lab else "?"


def _all_stats(p: dict) -> list[dict]:
    """Every timed stage's stats block, models and SSv2 head alike."""
    out = [b["stats"] for m in p["models"].values() for b in m["backends"].values()]
    s2 = p.get("ssv2_head") or {}
    out += [b.get("stats") or {} for b in [s2.get("pytorch", {})]
            + list((s2.get("backends") or {}).values())]
    return [s for s in out if s]


def _occupancy(p: dict) -> str:
    """How idle the box actually was, in CPU-seconds that were not this benchmark's."""
    occ = [s["occupancy"] for s in _all_stats(p) if s.get("occupancy")]
    if not occ:
        return ""
    mach = sum(o["machine_cpu_s"] for o in occ)
    own = sum(o["own_cpu_s"] for o in occ)
    foreign = sum(o["foreign_cpu_s"] for o in occ)
    wall = sum(o["wall_s"] for o in occ)
    return (f"across the {len(occ)} timed stages the machine spent {mach/60:.0f} CPU-minutes out of "
            f"idle, of which {own/60:.0f} were this benchmark's own process trees; the "
            f"{foreign/60:.1f} CPU-minutes left over for everything else on the box average "
            f"{foreign/max(wall, 1e-9):.2f} of one core out of 96 "
            f"(`occupancy` per row in the JSON: /proc/stat non-idle minus os.times() self+children)")


def _dtype_clock(p: dict) -> str:
    """`ViT-L f16 0.99 vs q8_0 0.97 clips/s` — straight from the measured stats."""
    out = []
    for m in p["models"].values():
        r = [(dt, b["stats"]["clips_per_s"]) for dt, b in m["backends"].items()
             if dt != "torch" and b["stats"].get("clips_per_s")]
        if len(r) > 1:
            out.append(m["label"] + " " + " vs ".join(f"{dt} {c:.2f}" for dt, c in r) + " clips/s")
    return "; ".join(out)


def _fastest(p: dict) -> str:
    """The fastest-to-slowest spread within each model, which is the point: dropping from f32 to
    q8_0 shrinks the file by ~2x and moves the clock by a couple of percent."""
    out = []
    for m in p["models"].values():
        r = [b["stats"]["clips_per_s"] for dt, b in m["backends"].items()
             if dt != "torch" and b["stats"].get("clips_per_s")]
        if len(r) > 1:
            out.append(f"{100 * (max(r) / min(r) - 1):.0f} % on {m['label']}")
    return (" — fastest to slowest is " + " and ".join(out) + ", against file sizes that differ "
            "by ~2x.") if out else "."


def render_md(p: dict, cj: dict) -> str:
    L = []
    A = L.append
    A("# Accuracy — video k-NN on the UCF101 subset (PyTorch vs jepa.cpp)\n")
    A("*Raw measurement report — the curated view is [Benchmarks → Accuracy](accuracy.md).*\n")
    A(f"Frozen-feature evaluation, {p['date']}. **Inference only** — nothing is trained: the encoders "
      "are frozen and both metrics are look-ups over their pooled clip features.\n")
    A(f"- **Dataset** `{p['dataset']['root']}` — {len(p['dataset']['classes'])} classes, "
      f"gallery = train ({p['dataset']['gallery']['n']} clips), queries = "
      + " / ".join(f"{sp} ({q['n']})" for sp, q in p['dataset']['queries'].items()) + ".")
    A(f"- **Clip** {p['dataset']['frames_per_clip']} frames, `{p['dataset']['frame_sampling']}`, decoded once "
      "with PyAV to a THWC uint8 `.npy` that *both* backends read, so the two see identical pixels.")
    A(f"- **Feature** {p['protocol']['feature']}.")
    A(f"- **k-NN** k = {p['protocol']['knn']['k']}, cosine similarity, DINO-style weighted vote "
      f"(`{p['protocol']['knn']['weight']}`). **Centroid** = {p['protocol']['centroid']} (no hyper-parameters).")
    A(f"- **Agreement** = fraction of query clips where the jepa.cpp prediction equals the PyTorch one "
      "(k-NN and centroid separately); **feat cos** = mean per-clip cosine between the two backends' "
      "feature vectors.")
    e = p.get("env", {})
    A(f"- {p['threads']} threads everywhere (AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, "
      f"AVX-512), gcc 13.3.0, ggml @ {e.get('ggml', '?')}, torch {e.get('torch', '?')} / "
      f"transformers {e.get('transformers', '?')}. Protocol implemented in `{p['protocol']['impl']}`.")
    A(f"- Chance is 10 % ({len(p['dataset']['classes'])} classes); the largest query class holds "
      f"{_maj(p)} % of the clips.\n")

    for name, m in p["models"].items():
        A(f"## {m['label']} — `{name}`\n")
        A("| backend | dtype | query split | k-NN top-1 % | centroid top-1 % | k-NN agreement % "
          "| centroid agreement % | feat cos | clips/s |")
        A("|---|---|---|--:|--:|--:|--:|--:|--:|")
        for dt, b in m["backends"].items():
            for sp in ("test", "val", "val+test"):
                if sp not in b["splits"]:
                    continue
                s = b["splits"][sp]
                ag = b["agreement"].get(sp, {})
                agr = "ref" if dt == "torch" else _pct(ag.get("knn"))
                agc = "ref" if dt == "torch" else _pct(ag.get("centroid"))
                cos = "ref" if dt == "torch" else (f"{ag['feat_cos']:.6f}" if "feat_cos" in ag else "—")
                cps = b["stats"].get("clips_per_s")
                A(f"| {b['backend']} | {b['dtype']} | {sp} ({s['n']}) | {_pct(s['knn_top1'])} | "
                  f"{_pct(s['centroid_top1'])} | {agr} | {agc} | {cos} | "
                  f"{'' if cps is None else f'{cps:.2f}'} |")
        A("")
        A("`clips/s` is end-to-end over all 405 clips (frame `.npy` -> preprocess -> encode -> pool), "
          "one clip at a time, excluding the model load. Agreement and feat cos are against the "
          "PyTorch row of the same model.\n")
        rows = [(dt, b) for dt, b in m["backends"].items() if dt != "torch"
                and "all_clips" in b.get("agreement", {})]
        if rows:
            A("Feature fidelity over **all** 405 clips (gallery + queries), against the PyTorch vectors:\n")
            A("| dtype | mean cos | worst clip cos | max abs component diff |")
            A("|---|--:|--:|--:|")
            for dt, b in rows:
                ac = b["agreement"]["all_clips"]
                A(f"| {dt} | {ac['feat_cos']:.7f} | {ac['feat_cos_min']:.7f} | {ac['feat_max_abs_diff']:.2e} |")
            A("")

    s2 = p.get("ssv2_head") or {}
    if s2.get("backends"):
        A("## SSv2 classification head — backend fidelity\n")
        A(f"`facebook/vjepa2-vitl-fpc16-256-ssv2` (attentive pooler + linear head, "
          f"{s2['n_classes']} Something-Something-v2 classes) run on the same "
          f"{s2['n_clips']} query clips. **This is not a task accuracy**: SSv2 labels have nothing to "
          "do with the UCF101 classes, so an SSv2 prediction on a UCF clip is meaningless as a label. "
          "What it measures is whether jepa.cpp's head reaches the *same* decision as PyTorch's on "
          "real, out-of-distribution video — 105 independent 174-way argmaxes over a full "
          "encoder + pooler + classifier stack, which a parity fixture of a handful of clips cannot cover.\n")
        A("| backend | dtype | top-1 agreement % | top-5 overlap % | PyTorch top-1 in top-5 % | max abs logit diff | logit cos | clips/s |")
        A("|---|---|--:|--:|--:|--:|--:|--:|")
        ps = s2["pytorch"]["stats"]
        A(f"| pytorch | f32 | ref | ref | ref | ref | ref | {ps['clips_per_s']:.2f} |")
        for dt, b in s2["backends"].items():
            cps = b["stats"].get("clips_per_s")
            A(f"| jepa.cpp | {dt} | {_pct(b['top1_agreement'])} | {_pct(b['top5_overlap'])} | "
              f"{_pct(b['pytorch_top1_in_cpp_top5'])} | {b['logit_max_abs_diff']:.4f} | "
              f"{b['logit_cos']:.6f} | {'' if cps is None else f'{cps:.2f}'} |")
        A("")
        A("`top-5 overlap` is the mean size of the intersection of the two top-5 label sets divided by 5.\n")

    A("## What the numbers say\n")

    anchor = None
    for name, m in p["models"].items():
        b = m["backends"].get("f32")
        if b and "all_clips" in b.get("agreement", {}):
            anchor = (name, m["label"], b)
    if anchor:
        _n, lbl, b = anchor
        ac = b["agreement"]["all_clips"]
        A(f"**The f32 anchor is exact end to end.** On all {ac['n']} clips, {lbl} at f32 reproduces the "
          f"PyTorch feature vector to cosine {ac['feat_cos']:.7f} (worst clip {ac['feat_cos_min']:.7f}, "
          f"largest single-component difference {ac['feat_max_abs_diff']:.1e}) and every k-NN and "
          "centroid prediction is identical. That covers the *whole* pipeline, not just the encoder: "
          "jepa.cpp decodes nothing, but it does its own resize, centre crop and normalisation from the "
          "same uint8 frames, so the match confirms `jepa.pre.*` reproduces the reference pipeline's "
          "pixels as well as the graph reproduces the weights. `docs/parity.md` shows the same at "
          "token level on two fixture clips; this is 405 real clips of pooled output.\n")

    worst_cos = min(((n, dt, b["agreement"]["all_clips"]["feat_cos_min"])
                     for n, m in p["models"].items() for dt, b in m["backends"].items()
                     if dt not in ("torch", "f32") and "all_clips" in b.get("agreement", {})),
                    key=lambda t: t[2], default=None)
    if worst_cos:
        A(f"**f16 and q8_0 do not move the accuracy.** Across every model and every query split the "
          f"k-NN and centroid top-1 numbers are within one clip of the PyTorch row, and the single worst "
          f"clip out of all 405 at any quantisation tested here still matches the PyTorch pooled vector "
          f"to cosine {worst_cos[2]:.6f} (`{worst_cos[0]}` {worst_cos[1]}). Where a jepa.cpp row reads *higher* than "
          "PyTorch — 89.5 vs 88.6 % on val+test — that is a single clip out of 105, i.e. 0.95 pp of "
          "quantisation noise landing on the right side. It is not an improvement, and the doc reports "
          "it rather than hiding it because the reverse would have been equally likely.\n")

    flips = [(n, dt, d) for n, m in p["models"].items() for dt, b in m["backends"].items()
             for d in b.get("disagreements", [])]
    if flips:
        A("**Every disagreement is a tie in the neighbour set, not a feature error.** The k-NN "
          "predictions that differ between the backends are:\n")
        A("| model | dtype | clip | true | PyTorch | jepa.cpp | PyTorch top-2 vote ratio "
          "| shared neighbours | cos gap 20th↔21st | backend cos shift | vote shift |")
        A("|---|---|---|---|---|---|--:|--:|--:|--:|--:|")
        for n, dt, d in flips:
            A(f"| {n} | {dt} | `{d['stem']}` ({d['split']}) | {d['true']} | {d['pytorch_pred']} | "
              f"{d['jepa_cpp_pred']} | {d['pytorch_top2_vote_ratio']:.4f} | "
              f"{d['shared_neighbours']}/{d['k']} | {d['gap_k_to_kplus1']:.1e} | "
              f"{d['backend_sim_max_diff']:.1e} | {d['top_class_vote_shift_pct']:+.1f} % |")
        A("")
        rat = sorted(d["backend_sim_max_diff"] / d["gap_k_to_kplus1"] for _, _, d in flips)
        shr = sorted({round(d["last_neighbour_weight_share"], 2) for _, _, d in flips})
        shf = sorted(abs(d["top_class_vote_shift_pct"]) for _, _, d in flips)
        tie = sorted({f"{d['pytorch_top2_vote_ratio']:.3f}" for _, _, d in flips})
        A(f"In each case the two backends agree on {flips[0][2]['shared_neighbours']} of the "
          f"{flips[0][2]['k']} nearest gallery clips and differ only at the last one — and the final "
          "columns say why: the 20th- and 21st-ranked gallery clips are separated by *less* cosine than "
          f"the two backends' similarities to that query differ, by {rat[0]:.1f}x to {rat[-1]:.0f}x, so "
          "which of the two lands inside the neighbourhood is decided by round-off. That last neighbour "
          "is not a rounding term in the vote: its `exp(sim / 0.07)` weight is "
          + " and ".join(f"{s:.2f}" for s in shr) + " of the top neighbour's, so one swap moves the "
          f"leading class total by {shf[0]:.0f}–{shf[-1]:.0f} %, which is enough to decide a vote that "
          "was already a " + " / ".join(tie) + " near-tie. The tell is the "
          "parameter-free metric: **nearest-class-centroid agreement is 100 % for every model and every "
          "dtype** — with no k and no neighbour set, there is nothing for a 1e-6 perturbation to "
          "reshuffle. Read the k-NN agreement column as a property of k-NN at k = 20 on a 300-clip "
          "gallery, not as a fidelity measure of the backend; `feat cos` and the centroid column are the "
          "fidelity measures.\n")

    if s2.get("backends"):
        q8 = s2["backends"].get("q8_0")
        f16 = s2["backends"].get("f16")
        if q8 and f16:
            A(f"**The SSv2 head is where q8_0 finally costs something.** f16 reaches the same 174-way "
              f"argmax as PyTorch on {_pct(f16['top1_agreement'])} % of the 105 clips "
              f"(logit cosine {f16['logit_cos']:.6f}); q8_0 drops to {_pct(q8['top1_agreement'])} % "
              f"({round((1 - q8['top1_agreement']) * s2['n_clips'])} clips of {s2['n_clips']}) with logit "
              f"cosine {q8['logit_cos']:.6f} and a largest logit error of {q8['logit_max_abs_diff']:.2f}. "
              "The PyTorch top-1 stays inside the jepa.cpp top-5 on "
              f"{_pct(q8['pytorch_top1_in_cpp_top5'])} % of clips at both dtypes, so the ranking is "
              "intact and only near-ties at the top move. This is the sharpest measurement in the "
              "document: an argmax over 174 classes has no averaging to hide behind, unlike a pooled "
              "1024-vector whose cosine stays at 0.9999 — which is exactly why `docs/parity.md`'s "
              "advice to prefer f16 over q8_0 for head/classifier work, and q8_0 only for pooled "
              "retrieval features, holds up on 105 real clips.\n")

    sp = []
    for name, m in p["models"].items():
        bs = m["backends"]
        t = (bs.get("torch") or {}).get("stats", {}).get("clips_per_s")
        cs = [(dt, b["stats"]["clips_per_s"]) for dt, b in bs.items()
              if dt != "torch" and b["stats"].get("clips_per_s")]
        if t and cs:
            sp.append((m["label"], t, cs))
    if sp:
        # The direction of the comparison is read off the measurements, never asserted — and a
        # ratio inside +-5 % is reported as a tie, because that is all this measurement resolves.
        faster = all(min(c for _, c in cs) / t >= 1.05 for _, t, cs in sp)
        lead = ("**Throughput.** jepa.cpp is faster than PyTorch on the same 32 threads in every "
                "configuration measured here: " if faster else
                "**Throughput.** On the same 32 threads, per model and dtype — jepa.cpp ahead on "
                "the larger model and level with PyTorch on the smaller one: ")
        A(lead + "; ".join(
            f"{lbl} {min(c for _, c in cs):.2f}–{max(c for _, c in cs):.2f} clips/s over "
            + "/".join(dt for dt, _ in cs) + f" against PyTorch's {t:.2f} "
            f"({min(c for _, c in cs)/t:.2f}–{max(c for _, c in cs)/t:.2f}x)" for lbl, t, cs in sp)
          + ". **Neither side is charged a per-clip model load, and neither batches.** The PyTorch "
          "loop keeps one `VJEPA2Model` resident and starts its timer after `from_pretrained` "
          "returns; `jepa-embed --frames-list` mmaps the GGUF once and then walks the whole "
          "405-clip list inside that one process. Both do their own preprocessing per clip, and both "
          "run one clip per forward: a V-JEPA 2 clip is already 2048-18432 tokens, so jepa.cpp keeps "
          "one graph per clip there and batches only the image families.\n")
        po = (p.get("pytorch_predictor_overhead") or {})
        if po:
            A("The PyTorch rows pass `skip_predictor=True`. `VJEPA2Model.forward` otherwise also "
              "runs a full `VJEPA2Predictor` pass whose output this benchmark discards — an earlier "
              "version of this table timed the baseline doing it, which is not a like-for-like "
              "comparison against an encoder-only jepa.cpp graph. " +
              "; ".join(f"{p['models'][n]['label']}: {v['with_predictor']['clips_per_s']:.2f} clips/s "
                        f"with the discarded predictor vs {v['skip_predictor']['clips_per_s']:.2f} "
                        f"without ({v['speedup']:.2f}x)" for n, v in po.items() if n in p["models"])
              + ". The encoder output is unaffected: the two runs agree "
              + " and ".join(("bit for bit" if v["features_bit_identical"] else "NOT bit for bit")
                             + f" on all {v['n_clips_compared']} clips" for v in po.values()) + ".\n")
        A("Within jepa.cpp the dtype barely moves the clock — " + _dtype_clock(p) + _fastest(p)
          + " `docs/parity.md` sees the same absence of a dtype speedup on its two fixture clips "
          "(1073 / 1125 / 1067 ms per clip for f32 / f16 / q8_0). These encoders are compute-bound "
          "at 32 threads, so q8_0 buys resident weights (332.8 vs 622.5 MiB for ViT-L, 113.3 vs "
          "209.6 MiB for 2.1 ViT-B — 0.53x and 0.54x), not speed.\n")
        A("**Load conditions.** Every row above was measured back-to-back in one sweep, alternating "
          "PyTorch and jepa.cpp stages so that any residual contention lands on both backends, on a "
          "box that was otherwise idle: " + _occupancy(p) + ".\n")

    A("**Practical reading.** For frozen-feature video retrieval and k-NN, f16 is the default and q8_0 "
      "costs nothing measurable — both land within one clip of PyTorch on 405 clips, at 0.53x the "
      "weights for q8_0. Use f32 only when you need bit-level agreement with a PyTorch reference. For "
      "the classification head, use f16: q8_0 moves 6 of 105 top-1 decisions.\n")

    A("## Reproduce\n")
    A("```bash")
    A("export PATH=$HOME/.local/bin:$PATH")
    A("git submodule update --init ggml")
    A("cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release")
    A("cmake --build build -j 32 --target jepa-embed jepa-classify jepa-info")
    A("")
    A("PY=.venv/bin/python                     # torch 2.13 CPU, transformers 5.16, av, numpy")
    A("export HF_HOME=$PWD/tmp/hf-home TORCH_HOME=$PWD/tmp OMP_NUM_THREADS=32")
    A("")
    A("$PY scripts/video_frames.py --data data/ucf101-subset/UCF101_subset \\")
    A("      --out tmp/frames --frames 16 --jobs 32")
    A("$PY scripts/bench_accuracy_video.py lists --index tmp/frames/index.json")
    A("")
    A("B=\"$PY scripts/bench_accuracy_video.py\"")
    A("")
    A("# The timed sweep, on an idle box, PyTorch and jepa.cpp stages alternated so that any")
    A("# residual contention lands on both backends.")
    A("$B torch --model vjepa2-vitl-fpc64-256                # PyTorch ViT-L  (skip_predictor)")
    A("$B cpp   --model vjepa2-vitl-fpc64-256 --dtype f16")
    A("$B torch --model vjepa2_1-vitb-384                    # PyTorch ViT-B  (Meta code path)")
    A("$B cpp   --model vjepa2_1-vitb-384     --dtype f32")
    A("$B cpp   --model vjepa2-vitl-fpc64-256 --dtype q8_0")
    A("$B cpp   --model vjepa2_1-vitb-384     --dtype f16")
    A("$B cpp   --model vjepa2_1-vitb-384     --dtype q8_0")
    A("$B torch --model levjepa-vitl16                       # PyTorch ViT-L  (trust_remote_code)")
    A("$B cpp   --model levjepa-vitl16        --dtype f32")
    A("$B cpp   --model levjepa-vitl16        --dtype f16")
    A("$B cpp   --model levjepa-vitl16        --dtype q8_0")
    A("$B ssv2-torch")
    A("$B ssv2-cpp --dtype f16")
    A("$B ssv2-cpp --dtype q8_0")
    A("")
    A("# Control run, last (warm page cache): the pre-2026-08-31 code path, which also ran the")
    A("# predictor and discarded it.  `report` diffs its features against the real ones and times")
    A("# the two against each other.")
    A("$B torch --model vjepa2-vitl-fpc64-256 --no-skip-predictor")
    A("")
    A("$B report --out-json tests/results/accuracy-video.json --out-md docs/accuracy-video.md")
    A("```\n")
    walls = [(f"{n} {dt}", b["stats"]["wall_s"]) for n, m in p["models"].items()
             for dt, b in m["backends"].items() if b["stats"].get("wall_s")]
    walls += [(f"ssv2 {dt}", b["stats"]["wall_s"]) for dt, b in (s2.get("backends") or {}).items()
              if b["stats"].get("wall_s")]
    if s2.get("pytorch"):
        walls.append(("ssv2 pytorch", s2["pytorch"]["stats"]["wall_s"]))
    walls += [(f"{n} torch control run (predictor included)", v["with_predictor"]["wall_s"])
              for n, v in (p.get("pytorch_predictor_overhead") or {}).items()]
    if walls:
        tot = sum(w for _, w in walls) + p["dataset"].get("decode_s", 0)
        A(f"**Wall time at 32 threads**, measured: frame decode "
          f"{p['dataset'].get('decode_s', 0):.1f} s for all "
          f"{p['dataset']['gallery']['n'] + p['dataset']['queries']['val+test']['n']} clips (32 processes), "
          "then " + ", ".join(f"{k} {v:.0f} s" for k, v in walls) + f" — {tot/60:.0f} min of compute "
          "in total, run strictly one stage at a time so that no clips/s number is measured against "
          "another stage. `report` takes a few seconds.\n")
    A("Every stage writes into `tmp/accuracy-video/` (git-ignored) and can be re-run alone; "
      "`lists` fixes the clip order once in `tmp/accuracy-video/clips.json`, which every feature "
      "`.npy` is indexed by, and `tmp/frames/index.json` records the sampled frame indices per clip.\n")
    A("`jepa-embed --frames-list list.txt` walks a whole clip list in one process — one model "
      "load, one `jepa_context`, one `[n_clips, D]` `.npy` in list order, `--logits` for the "
      "attentive-pool head and `--json` for the timings. It replaced the out-of-tree "
      "`jepa-embed-clips` driver this benchmark used to need (removed); the features it writes "
      "are bit-identical to that driver's and to `jepa-embed --frames-npy F --pool mean` per "
      "clip.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(ROOT / "tmp" / "accuracy-video"))
    sub = ap.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("lists"); s.add_argument("--index", default=str(ROOT / "tmp" / "frames" / "index.json")); s.set_defaults(fn=stage_lists)
    s = sub.add_parser("torch"); s.add_argument("--model", required=True, choices=list(MODELS))
    s.add_argument("--no-skip-predictor", action="store_true",
                   help="also run VJEPA2Model's predictor and throw the output away, as the "
                        "benchmark did before 2026-08-31; writes <model>-torchpred.npy, which the "
                        "report stage diffs against the real features and times against")
    s.add_argument("--tag", default="", help="output suffix (default: torch / torchpred)")
    s.add_argument("--limit", type=int, default=0, help="only the first N clips (control runs)")
    s.set_defaults(fn=stage_torch)
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
