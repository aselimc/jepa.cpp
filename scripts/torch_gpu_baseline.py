#!/usr/bin/env python3
"""PyTorch-on-the-same-card baseline for the four shapes docs/performance.md compares against.

    python scripts/torch_gpu_baseline.py --device 0 -o tmp/bench-gpu/torch-gpu.json

Protocol, as the document states it: batch 1, `torch.inference_mode`, TF32 off on both matmul and
cuDNN, `attn_implementation="sdpa"`, 3 warmup then 7 timed forwards with `cuda.synchronize()` around
each, at fp32 and fp16.  The input of every row is the stored preprocessed tensor of a reference
fixture, so PyTorch and jepa.cpp see the same pixels; `VJEPA2Model` runs with `skip_predictor=True`
so its forward is the encoder alone, like ours.

The JSON is folded into tests/results/benchmarks-gpu.json by
`scripts/gen_benchmarks_md.py --torch-gpu`, which is how scripts/bench_gpu.sh calls it.  It needs a
CUDA-enabled torch: the box's is in tmp/venv-cuda (torch 2.13.0+cu130).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# label -> (checkpoint under models/, loader, fixture input .npy, the kwarg the tensor goes in)
MODELS = {
    "ijepa_vith14_1k": {
        "dir": "facebook/ijepa_vith14_1k", "loader": "ijepa",
        "input": "ijepa-vith14-1k/coco_000000000139.input.npy", "kwarg": "pixel_values",
        "shape": "224x224", "frames": 1, "tokens": 256,
    },
    "vjepa2-vitl-fpc64-256@16": {
        "dir": "facebook/vjepa2-vitl-fpc64-256", "loader": "vjepa2",
        "input": "vjepa2-vitl-fpc64-256/archery_f16.input.npy", "kwarg": "pixel_values_videos",
        "shape": "16f 256x256", "frames": 16, "tokens": 2048,
    },
    "vjepa2-vitl-fpc64-256@64": {
        "dir": "facebook/vjepa2-vitl-fpc64-256", "loader": "vjepa2",
        "input": "vjepa2-vitl-fpc64-256/archery_f64.input.npy", "kwarg": "pixel_values_videos",
        "shape": "64f 256x256", "frames": 64, "tokens": 8192,
    },
    "levjepa-vitl16@16": {
        "dir": "galilai-group/LeVJEPA-VideoMix-Large", "loader": "auto",
        "input": "levjepa-vitl16/archery_f16.input.npy", "kwarg": "pixel_values",
        "shape": "16f 224x224", "frames": 16, "tokens": 3137,
    },
}


def load_model(spec, path, dtype, attn: str):
    import torch
    if spec["loader"] == "ijepa":
        from transformers import IJepaModel
        return IJepaModel.from_pretrained(path, dtype=dtype, attn_implementation=attn).eval()
    if spec["loader"] == "vjepa2":
        from transformers import VJEPA2Model
        return VJEPA2Model.from_pretrained(path, dtype=dtype, attn_implementation=attn).eval()
    from transformers import AutoModel
    # LeVJEPA ships its own modelling code, which takes no attn_implementation
    return AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--ref-dir", default=str(ROOT / "tests" / "fixtures" / "ref"))
    ap.add_argument("--only", default="", help="substring filter over the row keys")
    ap.add_argument("-o", "--out", default=str(ROOT / "tmp" / "bench-gpu" / "torch-gpu.json"))
    a = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import torch
    import transformers

    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 1
    # TF32 would silently make the fp32 rows a different measurement from the one the page describes
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{a.device}")

    blob = {
        "task": "PyTorch encoder forward on one GPU, the baseline docs/performance.md compares "
                "jepa.cpp CUDA against",
        "protocol": {
            "timing": f"{a.warmup} warmup + {a.repeat} timed forwards, cuda.synchronize() around each",
            "batch": 1,
            "tf32": False,
            "attn_implementation": "sdpa where the model accepts it (LeVJEPA ships its own code)",
            "input": "the stored preprocessed tensor of a reference fixture — the same pixels "
                     "jepa.cpp is timed on",
            "vjepa2": "skip_predictor=True, so the forward is the encoder alone",
            "ms": "mean/min/median/std over the timed forwards of the full model call",
        },
        "box": {
            "device": torch.cuda.get_device_name(a.device),
            "device_index": a.device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": ".".join(str(v) for v in sys.version_info[:3]),
            "cuda_runtime": torch.version.cuda,
            "driver": None,
            "date_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "loadavg_start": open("/proc/loadavg").read().split()[0],
        },
        "rows": [],
    }
    try:
        import subprocess
        blob["box"]["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(a.device)],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except Exception:
        pass

    for key, spec in MODELS.items():
        if a.only and a.only not in key:
            continue
        x_np = np.load(pathlib.Path(a.ref_dir) / spec["input"])
        path = pathlib.Path(a.models_dir) / spec["dir"]
        for prec, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            model = load_model(spec, path, dtype, "sdpa").to(dev)
            x = torch.from_numpy(x_np).to(device=dev, dtype=dtype)
            kw = {spec["kwarg"]: x}
            if spec["loader"] == "vjepa2":
                kw["skip_predictor"] = True

            def once():
                with torch.inference_mode():
                    model(**kw)

            for _ in range(a.warmup):
                once()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ms = []
            for _ in range(a.repeat):
                t = time.perf_counter()
                once()
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t) * 1000.0)
            peak = torch.cuda.max_memory_allocated() / 2**30
            row = {
                "model": key.split("@")[0], "key": key, "precision": prec,
                "shape": spec["shape"], "frames": spec["frames"], "tokens": spec["tokens"],
                "warmup": a.warmup, "repeat": a.repeat,
                "ms_mean": round(statistics.fmean(ms), 3), "ms_min": round(min(ms), 3),
                "ms_median": round(statistics.median(ms), 3),
                "ms_std": round(statistics.pstdev(ms), 3) if len(ms) > 1 else 0.0,
                "peak_gib": round(peak, 3),
                "input": spec["input"],
            }
            blob["rows"].append(row)
            print(f"{key:32} {prec}  mean {row['ms_mean']:8.2f}  min {row['ms_min']:8.2f}  "
                  f"median {row['ms_median']:8.2f}  sd {row['ms_std']:6.3f}  peak {peak:.2f} GiB")
            del model, x, kw
            torch.cuda.empty_cache()

    blob["box"]["loadavg_end"] = open("/proc/loadavg").read().split()[0]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1) + "\n")
    print(f"wrote {out} ({len(blob['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
