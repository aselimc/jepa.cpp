#!/usr/bin/env python3
"""Accuracy check of a (quantized) jepa.cpp GGUF against the PyTorch golden outputs, without the C++ graph.

The GGUF is read with the python `gguf` package, every quantized tensor is dequantized
(`gguf.quants.dequantize`: Q8_0 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q4_K / Q5_K / Q6_K / ...), and the numpy forward of
`scripts/jepa_convert/selftest.py` (ijepa / hfvit / lewm) or `scripts/jepa_convert/vjepa2_numpy_ref.py`
(vjepa2 / vjepa2_1) is run on the *stored* `input` tensors of a reference set written by `scripts/dump_reference.py`.
The outputs are compared with `scripts/compare.py` metrics (per-token cosine mean / min, max-abs, rel_max, rel_fro,
top-1 / top-5 for logits).  The numbers therefore isolate the weight-quantization error: the same math in f32,
the same preprocessed pixels, only the weights differ.

  gguf_dequant_selftest.py --gguf models/gguf/lejepa-vits16-pretrain-in1k-q8_0.gguf --ref tests/fixtures/ref/lejepa-vits16
  gguf_dequant_selftest.py --gguf models/gguf/ijepa_vith14_1k-q4_k.gguf --ref tests/fixtures/ref/ijepa-vith14-1k --samples 2
  gguf_dequant_selftest.py --gguf models/gguf/vjepa2_1-vitb-384-q8_0.gguf --ref tests/fixtures/ref/vjepa2_1-vitb-384 --samples coco_000000000139

Exit status 1 if any compared tensor has a worst per-token cosine below --min-cos (default 0.999, the Q8_0 threshold
of docs/architecture.md) or, when --max-rel is given, a rel_max above it.  --json writes all rows plus the per-tensor
summary for docs/quantization.md.  Threads: --threads sets OMP/OpenBLAS threads before numpy is imported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _threads_from_argv(default: int = 32) -> int:
    n = default
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    return n


_N_THREADS = _threads_from_argv()

import numpy as np  # noqa: E402  (after the thread env vars)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "jepa_convert"))

import compare  # noqa: E402
import selftest  # noqa: E402  (scripts/jepa_convert/selftest.py: Model + numpy graphs of the image families)

IMAGE_FAMILIES = ("ijepa", "hfvit", "lewm")
VIDEO_FAMILIES = ("vjepa2", "vjepa2_1")


# ----------------------------------------------------------------------------- forward passes per family
def run_image_family(m: selftest.Model, sample: dict, ref_dir: Path) -> dict[str, np.ndarray]:
    """ijepa / hfvit / lewm: one image (or the lewm 3-frame sequence) -> every reference tensor we can compute."""
    hp = m.hp
    fam = hp["jepa.family"]
    tens = sample["tensors"]
    x = np.load(ref_dir / tens["input"]["file"]).astype(np.float32)  # [B, 3, H, W]
    if x.ndim != 4:
        raise SystemExit(f"{sample['name']}: expected a [B,3,H,W] input, got {x.shape}")
    encs = [selftest.encoder_forward(m, img) for img in x]  # [N_tok, D] each
    out: dict[str, np.ndarray] = {}
    n_reg = int(hp.get("jepa.enc.n_registers", 0) or 0)
    n_prefix = (1 if hp["jepa.enc.cls_token"] else 0) + n_reg
    if fam == "ijepa":
        enc = encs[0]
        out["last_hidden_state"] = enc
        out["pooled_mean"] = enc[n_prefix:].mean(0)
    elif fam == "hfvit":
        enc = encs[0]
        out["last_hidden_state"] = enc
        out["cls"] = enc[0]
        out["pooled_mean"] = enc[n_prefix:].mean(0)
    elif fam == "lewm":
        proj_act = selftest.act_fn(hp["jepa.enc.proj_act"])
        aact = selftest.act_fn(hp["jepa.pred.action_act"])
        cls = np.stack([e[0] for e in encs])  # [B, D]
        emb = selftest.mlp2(m, "enc.proj", cls, proj_act)  # [B, D]
        if "emb_seq" in tens:  # the T-frame causal rollout sample
            actions = np.load(ref_dir / tens["action_seq"]["file"]).astype(np.float32)  # [T, A]
            out["emb_seq"] = emb
            out["act_emb_seq"] = selftest.mlp2(m, "pred.action_embed", actions, aact)
            out["pred_seq"] = selftest.lewm_predictor_forward(m, emb, actions)
        else:
            action = np.load(ref_dir / tens["action"]["file"]).astype(np.float32)  # [A]
            out["last_hidden_state"] = encs[0]
            out["cls"] = cls[0]
            out["emb"] = emb[0]
            out["act_emb"] = selftest.mlp2(m, "pred.action_embed", action[None], aact)[0]
            out["pred_next"] = selftest.lewm_predictor_forward(m, emb[:1], action[None])[-1]  # T = 1, end to end
    return out


def run_video_family(kv: dict, W: dict, sample: dict, ref_dir: Path, vref) -> dict[str, np.ndarray]:
    """vjepa2 / vjepa2_1: encoder [+ attentive-pool head] [+ full-context predictor] on the stored clip / image."""
    tens = sample["tensors"]
    x = np.load(ref_dir / tens["input"]["file"]).astype(np.float32)
    layout = str(tens["input"].get("layout") or "").split(" ")[0]  # "NCTHW", "NTCHW", "NCTHW (T=1, image path)"
    if not layout:  # dump_reference.py always writes it; infer for hand-made sets
        layout = "NCTHW" if x.shape[1] == 3 else "NTCHW"
    if layout not in ("NCTHW", "NTCHW"):
        raise SystemExit(f"{sample['name']}: unsupported input layout {layout!r}")
    x = x[0] if layout == "NCTHW" else x[0].transpose(1, 0, 2, 3)  # -> [C, T, H, W]
    mode = "image" if (kv["jepa.family"] == "vjepa2_1" and x.shape[1] == 1) else "video"
    enc, (Tt, gh, gw) = vref.encoder_forward(kv, W, x, mode=mode)
    out: dict[str, np.ndarray] = {"last_hidden_state": enc, "pooled_mean": enc.mean(0)}
    if kv.get("jepa.head.kind") == "attentive_pool" and "logits" in tens:
        logits = vref.head_forward(kv, W, enc)
        out["logits"] = logits
        out["top5_idx"] = np.argsort(-logits, kind="stable")[:5].astype(np.int64)
    if "predictor_last_hidden_state" in tens and "pred.norm.weight" in W:
        N = enc.shape[0]
        gp = kv["jepa.enc.img_size"] // kv["jepa.enc.patch_size"]  # HF decodes mask ids on the config grid
        pred, _ = vref.predictor_forward(kv, W, enc, np.arange(N), np.arange(N), (gp, gp), mask_index=1, mode=mode)
        out["predictor_last_hidden_state"] = pred
    return out


# ----------------------------------------------------------------------------- main
def select_samples(samples: list[dict], spec: str | None) -> list[dict]:
    if not spec:
        return samples
    if spec.isdigit():
        return samples[: int(spec)]
    names = set(spec.split(","))
    chosen = [s for s in samples if s["name"] in names]
    missing = names - {s["name"] for s in chosen}
    if missing:
        raise SystemExit(f"unknown samples {sorted(missing)}; available: {[s['name'] for s in samples]}")
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", required=True, type=Path, help="tests/fixtures/ref/<model> (manifest.json + .npy)")
    ap.add_argument("--samples", default=None, help="N (first N samples) or a comma list of sample names (default: all)")
    ap.add_argument("--tensors", default=None, help="comma list of reference tensors to compare (default: all computable)")
    ap.add_argument("--threads", type=int, default=_N_THREADS, help="numpy / OpenBLAS threads (default 32)")
    ap.add_argument("--min-cos", type=float, default=0.999, help="fail if the worst per-token cosine is below this")
    ap.add_argument("--max-rel", type=float, default=None, help="fail if rel_max (max|a-b| / max|b|) exceeds this")
    ap.add_argument("--json", type=Path, default=None, help="write rows + summary as JSON")
    ap.add_argument("--out", type=Path, default=None,
                    help="also save the computed tensors as <sample>.<tensor>.npy (compare.py / analysis)")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    args = ap.parse_args(argv)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    man = json.loads((args.ref / "manifest.json").read_text())
    samples = select_samples(man["samples"], args.samples)
    want = set(args.tensors.split(",")) if args.tensors else None

    t0 = time.time()
    m = selftest.Model(args.gguf)
    fam = m.hp["jepa.family"]
    types = m.tensor_types()
    print(f"{args.gguf}: family={fam} name={m.hp.get('general.name')} file_type={m.hp.get('general.file_type')} "
          f"tensor types={types}")
    print(f"reference: {man['model']} ({man.get('framework', {}).get('torch', '?')}), {len(samples)} sample(s), "
          f"{args.threads} threads")
    vref = kv = W = None
    if fam in VIDEO_FAMILIES:
        import vjepa2_numpy_ref as vref  # noqa: F811
        kv, W = vref.load_gguf(args.gguf)  # dequantizes every tensor once
    elif fam not in IMAGE_FAMILIES:
        raise SystemExit(f"family {fam} not supported")
    print(f"weights loaded / dequantized in {time.time() - t0:.1f}s")

    rows: list[dict] = []
    n_fail = 0
    for s in samples:
        t1 = time.time()
        if fam in IMAGE_FAMILIES:
            outs = run_image_family(m, s, args.ref)
        else:
            outs = run_video_family(kv, W, s, args.ref, vref)
        dt = time.time() - t1
        if args.out:
            for tname, arr in outs.items():
                np.save(args.out / f"{s['name']}.{tname}.npy", np.ascontiguousarray(arr))
        for tname, ref_info in s["tensors"].items():
            if tname in ("input", "frames_u8") or tname not in outs or (want and tname not in want):
                continue
            ref = np.load(args.ref / ref_info["file"])
            r = {"sample": s["name"], "tensor": tname,
                 **compare.compare_arrays(outs[tname], ref, topk=5, is_logits=("logits" in tname))}
            fails = compare.check_thresholds(r, args.min_cos, None, args.max_rel, 0.0, 0.0)
            r["fails"] = fails
            n_fail += bool(fails)
            rows.append(r)
            if fails or not args.quiet:
                print(compare.fmt_row(r, fails))
        if not args.quiet:
            print(f"  ({s['name']}: forward {dt:.1f}s)")

    # per-tensor summary over samples (the docs/quantization.md table)
    summary: dict[str, dict] = {}
    for r in rows:
        if "cos_min" not in r and "exact" not in r:
            continue
        g = summary.setdefault(r["tensor"], {"tensor": r["tensor"], "n": 0, "cos_mean": 0.0, "cos_min": 1.0,
                                             "rel_max": 0.0, "rel_fro": 0.0, "max_abs": 0.0, "top1": 0, "top5": 0.0,
                                             "set_overlap": 0.0})
        g["n"] += 1
        if "exact" in r:  # top5_idx
            g["set_overlap"] += r["set_overlap"]
            continue
        g["cos_mean"] += r["cos_mean"]
        g["cos_min"] = min(g["cos_min"], r["cos_min"])
        g["rel_max"] = max(g["rel_max"], r["rel_max_abs"])
        g["rel_fro"] = max(g["rel_fro"], r["rel_fro"])
        g["max_abs"] = max(g["max_abs"], r["max_abs"])
        if "top1_match" in r:
            g["top1"] += int(r["top1_match"])
            g["top5"] += r["topk_overlap"]
    print("summary (per tensor, over samples): cos_mean = mean, cos_min = worst token, rel_* = worst sample")
    print(f"  {'tensor':28s} {'n':>2s} {'cos_mean':>10s} {'cos_min':>10s} {'rel_max':>9s} {'rel_fro':>9s} {'max_abs':>9s}  top1/top5")
    for g in summary.values():
        n = g["n"]
        if g["tensor"] == "top5_idx":
            g["set_overlap"] /= n
            print(f"  {g['tensor']:28s} {n:2d} {'':>10s} {'':>10s} {'':>9s} {'':>9s} {'':>9s}  set_overlap={g['set_overlap']:.2f}")
            continue
        g["cos_mean"] /= n
        extra = f"  {g['top1']}/{n}, {g['top5'] / n:.2f}" if g["tensor"] == "logits" else ""
        print(f"  {g['tensor']:28s} {n:2d} {g['cos_mean']:10.6f} {g['cos_min']:10.6f} {g['rel_max']:9.2e} "
              f"{g['rel_fro']:9.2e} {g['max_abs']:9.2e}{extra}")
    fl = [r for r in rows if "cos_min" in r]
    worst_cos = min((r["cos_min"] for r in fl), default=float("nan"))
    worst_rel = max((r["rel_max_abs"] for r in fl), default=float("nan"))
    print(f"result: {len(rows)} tensors compared, {n_fail} failing (min-cos {args.min_cos}); worst cos_min={worst_cos:.6f}, "
          f"worst rel_max={worst_rel:.2e}; {time.time() - t0:.1f}s total")
    if args.json:
        args.json.write_text(json.dumps({"gguf": str(args.gguf), "ref": str(args.ref), "family": fam,
                                         "tensor_types": types, "file_type": m.hp.get("general.file_type"),
                                         "min_cos": args.min_cos, "rows": rows, "summary": list(summary.values())},
                                        indent=1))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
