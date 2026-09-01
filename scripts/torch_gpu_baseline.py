#!/usr/bin/env python3
"""PyTorch-on-the-same-card baseline for the shapes docs/performance.md compares against.

    python scripts/torch_gpu_baseline.py --device 0 -o tmp/bench-gpu/torch-gpu.json
    python scripts/torch_gpu_baseline.py --device 1 --batch 1,8,32 --compile --only ijepa

Protocol, as the document states it: `torch.inference_mode`, TF32 off on both matmul and
cuDNN, `attn_implementation="sdpa"`, 3 warmup then 7 timed forwards with `cuda.synchronize()` around
each, at fp32 and fp16.  The input of every row is the stored preprocessed tensor of a reference
fixture, so PyTorch and jepa.cpp see the same pixels; `VJEPA2Model` runs with `skip_predictor=True`
so its forward is the encoder alone, like ours.

`--batch` takes a comma-separated list and repeats the fixture tensor along the batch axis, which is
the axis `jepa-bench --batch` sweeps: the same pixels B times, so the two engines are timed on the
same work.  `--compile` times each configuration a second time through `torch.compile`, after its
own warmups, so compilation is excluded from the milliseconds; a model `torch.compile` cannot
handle is recorded as a skipped row rather than aborting the sweep.

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
    "lejepa-vits16-pretrain-in1k": {
        # ViTv2.forward(xs) takes its tensor positionally, as scripts/dump_reference.py calls it.
        "dir": "OK-AI/lejepa-vits16-pretrain-in1k", "loader": "auto",
        "input": "lejepa-vits16/coco_000000000139.input.npy", "kwarg": None,
        "shape": "224x224", "frames": 1, "tokens": 197,
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
    # LeJEPA's modelling_vitv2.py imports `configuration_vitv2` and `hf_src.*` by plain name out of
    # the checkout, so its directory has to be importable before the dynamic module is compiled —
    # the same thing scripts/dump_reference.py does for it.
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
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
    ap.add_argument("--batch", default="1",
                    help="comma-separated item counts; the fixture tensor is repeated along the "
                         "batch axis (default 1)")
    ap.add_argument("--compile", action="store_true",
                    help="also time each configuration through torch.compile")
    ap.add_argument("--compile-mode", default="default",
                    help="torch.compile(mode=...) (default, reduce-overhead, max-autotune)")
    ap.add_argument("--max-batch-tokens", type=int, default=0,
                    help="skip a (model, batch) whose batch x tokens exceeds this, recording it as "
                         "a skipped row; 0 (the default) measures every combination asked for")
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
    # torch.cuda.synchronize() and the memory counters act on the CURRENT device when called with no
    # argument. Without this line a --device 1 run would synchronise device 0 — i.e. not wait for the
    # forward at all — and report device 0's peak memory, which is zero.
    torch.cuda.set_device(dev)

    blob = {
        "task": "PyTorch encoder forward on one GPU, the baseline docs/performance.md compares "
                "jepa.cpp CUDA against",
        "protocol": {
            "timing": f"{a.warmup} warmup + {a.repeat} timed forwards, cuda.synchronize() around each",
            "batch": "per row; the fixture tensor repeated along the batch axis, the axis "
                     "jepa-bench --batch sweeps",
            "tf32": False,
            "attn_implementation": "sdpa where the model accepts it (LeVJEPA ships its own code)",
            "input": "the stored preprocessed tensor of a reference fixture — the same pixels "
                     "jepa.cpp is timed on",
            "vjepa2": "skip_predictor=True, so the forward is the encoder alone",
            "ms": "mean/min/median/std over the timed forwards of the full model call",
            "runtime": "eager, or compile for a torch.compile'd module warmed up before timing "
                       f"(mode={a.compile_mode!r}); the compile itself is not in the milliseconds",
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

    batches = [int(b) for b in a.batch.split(",") if b.strip()]
    runtimes = ["eager"] + (["compile"] if a.compile else [])

    def time_call(fn, warmup: int, repeat: int) -> tuple[list[float], float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ms = []
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1000.0)
        return ms, torch.cuda.max_memory_allocated() / 2**30

    for key, spec in MODELS.items():
        if a.only and a.only not in key:
            continue
        x_np = np.load(pathlib.Path(a.ref_dir) / spec["input"])
        path = pathlib.Path(a.models_dir) / spec["dir"]
        for prec, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            model = load_model(spec, path, dtype, "sdpa").to(dev)
            for runtime in runtimes:
                # One compiled module per (model, precision), reused across batch sizes; a new batch
                # is a new shape and torch.compile recompiles for it, which the warmups absorb.
                m = model
                if runtime == "compile":
                    try:
                        m = torch.compile(model, mode=a.compile_mode)
                    except Exception as e:
                        print(f"{key} {prec}: torch.compile unavailable ({e}) — skipped",
                              file=sys.stderr)
                        blob["rows"].append({"model": key.split("@")[0], "key": key,
                                             "precision": prec, "runtime": runtime,
                                             "skipped": f"torch.compile failed: {e}"})
                        continue
                for nb in batches:
                    if a.max_batch_tokens and spec["tokens"] * nb > a.max_batch_tokens:
                        blob["rows"].append({
                            "model": key.split("@")[0], "key": key, "precision": prec,
                            "runtime": runtime, "batch": nb, "shape": spec["shape"],
                            "skipped": f"{spec['tokens'] * nb} rows is over the "
                                       f"--max-batch-tokens {a.max_batch_tokens} ceiling"})
                        continue
                    # The fixture tensor is batch 1; repeat it along the batch axis.
                    x = torch.from_numpy(x_np).to(device=dev, dtype=dtype)
                    if nb > 1:
                        x = x.repeat(nb, *([1] * (x.dim() - 1))).contiguous()
                    args, kw = (), {}
                    if spec["kwarg"]:
                        kw = {spec["kwarg"]: x}
                    else:
                        args = (x,)
                    if spec["loader"] == "vjepa2":
                        kw["skip_predictor"] = True

                    def once(_m=m, _a=args, _kw=kw):
                        with torch.inference_mode():
                            _m(*_a, **_kw)

                    try:
                        ms, peak = time_call(once, a.warmup, a.repeat)
                    except Exception as e:
                        print(f"{key} {prec} {runtime} batch {nb}: {e} — skipped", file=sys.stderr)
                        blob["rows"].append({"model": key.split("@")[0], "key": key,
                                             "precision": prec, "runtime": runtime, "batch": nb,
                                             "skipped": str(e)})
                        del x, args, kw
                        torch.cuda.empty_cache()
                        continue
                    mean = statistics.fmean(ms)
                    row = {
                        "model": key.split("@")[0], "key": key, "precision": prec,
                        "runtime": runtime, "batch": nb,
                        "shape": spec["shape"], "frames": spec["frames"],
                        "tokens": spec["tokens"] * nb,
                        "tokens_per_item": spec["tokens"],
                        "warmup": a.warmup, "repeat": a.repeat,
                        "ms_mean": round(mean, 3), "ms_min": round(min(ms), 3),
                        "ms_median": round(statistics.median(ms), 3),
                        "ms_std": round(statistics.pstdev(ms), 3) if len(ms) > 1 else 0.0,
                        "ms_per_item_mean": round(mean / nb, 4),
                        "items_per_s": round(1000.0 * nb / mean, 2),
                        "peak_gib": round(peak, 3),
                        "input": spec["input"],
                    }
                    blob["rows"].append(row)
                    print(f"{key:32} {prec} {runtime:8} b={nb:<3} mean {row['ms_mean']:9.2f}  "
                          f"min {row['ms_min']:9.2f}  sd {row['ms_std']:6.3f}  "
                          f"{row['ms_per_item_mean']:8.3f} ms/item  peak {peak:.2f} GiB")
                    del x, args, kw
                    torch.cuda.empty_cache()
                if runtime == "compile":
                    del m
            del model
            torch.cuda.empty_cache()

    blob["box"]["loadavg_end"] = open("/proc/loadavg").read().split()[0]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1) + "\n")
    print(f"wrote {out} ({len(blob['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
