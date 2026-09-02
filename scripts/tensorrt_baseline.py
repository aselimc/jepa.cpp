#!/usr/bin/env python3
"""TensorRT fp16 baseline — the honest ceiling for an encoder on this card.

TensorRT is **not jepa.cpp**. It is NVIDIA's own inference compiler for NVIDIA hardware: it fuses
the graph, picks kernels by autotuning against the card in front of it and writes a single engine.
Nothing portable can be expected to match it, which is exactly why it is worth measuring — it says
how much of the card a given encoder can be made to use at all.

Two phases, because the export needs PyTorch and the engine needs TensorRT and the box keeps them in
different environments:

    # 1. ONNX export (needs torch; tmp/venv-cuda on this box)
    tmp/venv-cuda/bin/python scripts/tensorrt_baseline.py export --out-dir tmp/trt

    # 2. build the fp16 engine and time it (needs tensorrt; tmp/venv-trt)
    tmp/venv-trt/bin/python scripts/tensorrt_baseline.py run --device 1 --out-dir tmp/trt \\
        -o tests/results/tensorrt.json

Protocol, matched to scripts/torch_gpu_baseline.py so the numbers sit in one table: the stored
preprocessed tensor of a reference fixture as input, 3 warmup then 7 timed executions with a
`cuda.synchronize()` equivalent (an explicit stream synchronise) around each, batch 1.

Precision comes from the ONNX, not from a builder flag. A TensorRT 11 network is *strongly typed*
by default and `BuilderFlag.FP16` no longer exists, so the fp16 engine needs an fp16 graph. The
route that works is export-fp32-then-convert: `torch.onnx.export` of a half module emits an
inconsistent graph (a float LayerNorm input against a half scale, which the strongly-typed parser
rejects) and refuses V-JEPA 2 outright, while
`onnxconverter_common.float16.convert_float_to_float16` rewrites a clean fp32 graph and inserts the
casts the mixture needs. Hence three phases:

    tmp/venv-cuda/bin/python scripts/tensorrt_baseline.py export  --out-dir tmp/trt
    tmp/venv-trt/bin/python  scripts/tensorrt_baseline.py convert --out-dir tmp/trt
    tmp/venv-trt/bin/python  scripts/tensorrt_baseline.py run --device 1 --out-dir tmp/trt

The timing excludes the host-to-device copy of the input and the device-to-host copy of the output,
matching what `jepa-bench` reports for jepa.cpp (`ggml_backend_graph_compute` alone).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

MODELS = {
    "ijepa_vith14_1k": {
        "dir": "facebook/ijepa_vith14_1k", "loader": "ijepa",
        "input": "ijepa-vith14-1k/coco_000000000139.input.npy",
        "shape": "224x224", "frames": 1, "tokens": 256,
    },
    "vjepa2-vitl-fpc64-256@16": {
        "dir": "facebook/vjepa2-vitl-fpc64-256", "loader": "vjepa2",
        "input": "vjepa2-vitl-fpc64-256/archery_f16.input.npy",
        "shape": "16f 256x256", "frames": 16, "tokens": 2048,
    },
}


def initializer_dtypes(onnx_path) -> dict:
    """How many weights the export actually stored at each precision.

    A `--dtype fp16` export is mixed, not uniform: PyTorch keeps LayerNorm scales and the biases in
    float32 while the projection matrices go to half, and a strongly-typed TensorRT engine honours
    exactly that. Counting them is how the artifact says which engine was built."""
    import collections
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)
    c = collections.Counter(onnx.TensorProto.DataType.Name(i.data_type) for i in m.graph.initializer)
    return dict(sorted(c.items()))


# ---------------------------------------------------------------------------------------------
# phase 1: ONNX export
# ---------------------------------------------------------------------------------------------
def do_export(a) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import torch

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for key, spec in MODELS.items():
        if a.only and a.only not in key:
            continue
        # One directory per model: torch.onnx.export(external_data=True) writes one file per
        # initializer, named after the parameter, and two ViTs would overwrite each other's.
        case_dir = out_dir / f"{key.replace('@', '-')}-{a.dtype}"
        case_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = case_dir / "model.onnx"
        x_np = np.load(pathlib.Path(a.ref_dir) / spec["input"])
        path = pathlib.Path(a.models_dir) / spec["dir"]
        try:
            dt = torch.float16 if a.dtype == "fp16" else torch.float32
            if spec["loader"] == "ijepa":
                from transformers import IJepaModel
                base = IJepaModel.from_pretrained(path, dtype=dt,
                                                  attn_implementation="sdpa").eval()

                class Wrap(torch.nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m

                    def forward(self, x):
                        return self.m(pixel_values=x).last_hidden_state
            else:
                from transformers import VJEPA2Model
                base = VJEPA2Model.from_pretrained(path, dtype=dt,
                                                   attn_implementation="sdpa").eval()

                class Wrap(torch.nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m

                    def forward(self, x):
                        return self.m(pixel_values_videos=x, skip_predictor=True).last_hidden_state

            model = Wrap(base).eval().to(a.export_device)
            x = torch.from_numpy(x_np).to(dtype=dt, device=a.export_device)
            with torch.inference_mode():
                torch.onnx.export(model, (x,), str(onnx_path), input_names=["input"],
                                  output_names=["last_hidden_state"], opset_version=17,
                                  dynamo=False, external_data=True)
            size = sum(f.stat().st_size for f in case_dir.rglob("*") if f.is_file())
            report[key] = {"onnx": str(onnx_path.relative_to(out_dir)), "bytes": size,
                           "dtype": a.dtype, "traced_on": a.export_device,
                           "initializer_dtypes": initializer_dtypes(onnx_path),
                           "input_shape": list(x_np.shape), "tokens": spec["tokens"]}
            print(f"{key}: exported {onnx_path.relative_to(out_dir)} ({size/2**20:.0f} MiB)")
        except Exception as e:  # noqa: BLE001 — an export failure is a result, not a crash
            report[key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{key}: ONNX export FAILED — {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            for n in ("model", "base", "x"):
                if n in dir():
                    del n
    manifest = out_dir / f"export-{a.dtype}.json"
    manifest.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {manifest}")
    return 0


def ref_cos(a, spec, out_np):
    """Mean per-token cosine of the engine's output against the PyTorch dump of the same fixture.

    An engine that is fast because it computed something else is not a baseline, and the fixture
    already carries the reference `last_hidden_state` every other measurement here is judged on."""
    import numpy as np
    ref_path = (pathlib.Path(a.ref_dir) / spec["input"]).with_name(
        pathlib.Path(spec["input"]).name.replace(".input.npy", ".last_hidden_state.npy"))
    if not ref_path.exists():
        return None
    ref = np.load(ref_path).astype(np.float64)
    got = np.asarray(out_np, dtype=np.float64).reshape(-1, ref.shape[-1])[:ref.shape[0]]
    num = (got * ref).sum(-1)
    den = np.linalg.norm(got, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-30
    return round(float((num / den).mean()), 7)


# ---------------------------------------------------------------------------------------------
# phase 1b: fp32 graph -> fp16 graph
# ---------------------------------------------------------------------------------------------
def do_convert(a) -> int:
    """Rewrite each fp32 export as fp16, the form a strongly-typed TensorRT 11 engine needs."""
    import onnx
    from onnxconverter_common import float16

    out_dir = pathlib.Path(a.out_dir)
    src_manifest = out_dir / "export-fp32.json"
    if not src_manifest.exists():
        print(f"no {src_manifest}: run the export phase first", file=sys.stderr)
        return 1
    src = json.loads(src_manifest.read_text())
    report = {}
    for key, info in src.items():
        if a.only and a.only not in key:
            continue
        if "error" in info:
            report[key] = info
            continue
        src_path = out_dir / info["onnx"]
        dst_dir = out_dir / f"{key.replace('@', '-')}-fp16"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / "model.onnx"
        try:
            m = onnx.load(str(src_path))
            # keep_io_types leaves the graph's own input and output float32 and inserts the casts
            # at the boundary. Converting them too leaves the first Conv with a float input against
            # a half kernel, which a strongly-typed TensorRT network refuses.
            block = [o for o in a.fp16_block_ops.split(",") if o]
            m16 = float16.convert_float_to_float16(
                m, keep_io_types=True, disable_shape_infer=True,
                op_block_list=(float16.DEFAULT_OP_BLOCK_LIST + block) if block else None)
            onnx.save(m16, str(dst_path), save_as_external_data=True, all_tensors_to_one_file=True,
                      location="weights.bin", size_threshold=1024)
            size = sum(f.stat().st_size for f in dst_dir.rglob("*") if f.is_file())
            report[key] = {"onnx": str(dst_path.relative_to(out_dir)), "bytes": size,
                           "dtype": "fp16", "converted_from": info["onnx"],
                           "initializer_dtypes": initializer_dtypes(dst_path),
                           "input_shape": info["input_shape"], "tokens": info["tokens"]}
            print(f"{key}: converted to fp16 ({size/2**20:.0f} MiB)")
        except Exception as e:  # noqa: BLE001
            report[key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{key}: fp16 conversion FAILED — {type(e).__name__}: {e}", file=sys.stderr)
    manifest = out_dir / "export-fp16.json"
    manifest.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {manifest}")
    return 0


# ---------------------------------------------------------------------------------------------
# phase 2: engine build + timing
# ---------------------------------------------------------------------------------------------
def do_run(a) -> int:
    import numpy as np
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    def chk(res):
        err, *rest = res if isinstance(res, tuple) else (res,)
        if int(err) != 0:
            raise RuntimeError(f"CUDA error {err}")
        return rest[0] if len(rest) == 1 else tuple(rest)

    out_dir = pathlib.Path(a.out_dir)
    manifest = out_dir / f"export-{a.dtype}.json"
    export = json.loads(manifest.read_text()) if manifest.exists() else {}
    chk(cudart.cudaSetDevice(a.device))
    logger = trt.Logger(trt.Logger.WARNING)

    blob = {
        "task": "TensorRT fp16 encoder baseline on one GPU — NOT jepa.cpp, the vendor's own "
                "compiler for this card and the ceiling the portable engines are read against",
        "generated_by": "scripts/tensorrt_baseline.py",
        "protocol": {
            "engine": "trt.Builder with FP16 enabled, built from an opset-17 ONNX export of the "
                      "same HuggingFace module scripts/torch_gpu_baseline.py times",
            "timing": f"{a.warmup} warmup + {a.repeat} timed executions, an explicit stream "
                      "synchronise around each; the H2D input copy and the D2H output copy are "
                      "outside the timed region, matching what jepa-bench reports",
            "input": "the stored preprocessed tensor of a reference fixture — the same pixels "
                     "jepa.cpp and PyTorch are timed on",
            "ms": "mean/min/median/std over the timed executions",
        },
        "box": {"device_index": a.device, "tensorrt": trt.__version__,
                "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        "rows": [],
    }
    try:
        import subprocess
        blob["box"]["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader",
             "-i", str(a.device)], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass

    for key, spec in MODELS.items():
        if a.only and a.only not in key:
            continue
        info = export.get(key, {})
        if "error" in info:
            blob["rows"].append({"model": key.split("@")[0], "key": key, "shape": spec["shape"],
                                 "skipped": f"ONNX export failed: {info['error']}"})
            print(f"{key}: skipped, {info['error']}", file=sys.stderr)
            continue
        onnx_path = out_dir / f"{key.replace('@', '-')}-{a.dtype}" / "model.onnx"
        if not onnx_path.exists():
            blob["rows"].append({"model": key.split("@")[0], "key": key, "shape": spec["shape"],
                                 "skipped": f"no {onnx_path}; run the export phase first"})
            continue
        plan_path = onnx_path.with_suffix(".plan")
        t0 = time.time()
        try:
            if plan_path.exists() and a.keep:
                plan = plan_path.read_bytes()
            else:
                builder = trt.Builder(logger)
                network = builder.create_network()
                parser = trt.OnnxParser(network, logger)
                if not parser.parse_from_file(str(onnx_path)):
                    msgs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                    raise RuntimeError(f"ONNX parse failed: {msgs}")
                config = builder.create_builder_config()
                # A TensorRT 11 network is strongly typed: the precision is the ONNX's, and the
                # BuilderFlag.FP16 of earlier releases no longer exists.
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_gib << 30)
                plan = builder.build_serialized_network(network, config)
                if plan is None:
                    raise RuntimeError("build_serialized_network returned None")
                plan_path.write_bytes(bytes(plan))
            build_s = time.time() - t0

            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(plan)
            ctx = engine.create_execution_context()
            in_name = engine.get_tensor_name(0)
            # The engine's own input dtype, not an assumption: an fp16 graph converted with
            # keep_io_types still takes float32 at the boundary.
            x_np = np.load(pathlib.Path(a.ref_dir) / spec["input"]).astype(
                trt.nptype(engine.get_tensor_dtype(in_name)))
            out_name = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT][0]
            ctx.set_input_shape(in_name, x_np.shape)
            out_shape = tuple(ctx.get_tensor_shape(out_name))
            out_np = np.empty(out_shape, dtype=trt.nptype(engine.get_tensor_dtype(out_name)))

            d_in = chk(cudart.cudaMalloc(x_np.nbytes))
            d_out = chk(cudart.cudaMalloc(out_np.nbytes))
            stream = chk(cudart.cudaStreamCreate())
            chk(cudart.cudaMemcpy(d_in, x_np.ctypes.data, x_np.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            ctx.set_tensor_address(in_name, int(d_in))
            ctx.set_tensor_address(out_name, int(d_out))

            def once():
                ctx.execute_async_v3(stream_handle=int(stream))

            for _ in range(a.warmup):
                once()
            chk(cudart.cudaStreamSynchronize(stream))
            ms = []
            for _ in range(a.repeat):
                t = time.perf_counter()
                once()
                chk(cudart.cudaStreamSynchronize(stream))
                ms.append((time.perf_counter() - t) * 1000.0)
            chk(cudart.cudaMemcpy(out_np.ctypes.data, d_out, out_np.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            chk(cudart.cudaFree(d_in))
            chk(cudart.cudaFree(d_out))
            chk(cudart.cudaStreamDestroy(stream))

            row = {
                "model": key.split("@")[0], "key": key, "precision": a.dtype,
                "shape": spec["shape"], "frames": spec["frames"], "tokens": spec["tokens"],
                "warmup": a.warmup, "repeat": a.repeat,
                "ms_mean": round(statistics.fmean(ms), 3), "ms_min": round(min(ms), 3),
                "ms_median": round(statistics.median(ms), 3),
                "ms_std": round(statistics.pstdev(ms), 3) if len(ms) > 1 else 0.0,
                "engine_build_s": round(build_s, 1),
                "engine_bytes": len(bytes(plan)),
                "output_shape": list(out_shape),
                "io_dtypes": [str(engine.get_tensor_dtype(in_name)),
                              str(engine.get_tensor_dtype(out_name))],
                "output_cos_vs_reference": ref_cos(a, spec, out_np),
            }
            blob["rows"].append(row)
            print(f"{key:32} {row['precision']:5} mean {row['ms_mean']:8.3f}  "
                  f"min {row['ms_min']:8.3f}  sd {row['ms_std']:6.3f}  "
                  f"(engine built in {build_s:.0f} s)")
            del engine, ctx
        except Exception as e:  # noqa: BLE001 — a build failure is a result, not a crash
            blob["rows"].append({"model": key.split("@")[0], "key": key, "shape": spec["shape"],
                                 "skipped": f"{type(e).__name__}: {e}"})
            print(f"{key}: TensorRT FAILED — {type(e).__name__}: {e}", file=sys.stderr)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # fp16 and fp32 are two invocations and one table: rows are merged on (key, precision) so the
    # second run adds its half instead of erasing the first.
    if out.exists() and a.merge:
        try:
            base = json.loads(out.read_text())
            rows = {(r.get("key"), r.get("precision")): r for r in base.get("rows", [])}
            for r in blob["rows"]:
                rows[(r.get("key"), r.get("precision"))] = r
            blob["rows"] = sorted(rows.values(), key=lambda r: (r.get("key", ""),
                                                                r.get("precision", "")))
        except (OSError, ValueError) as e:
            print(f"warning: cannot merge into {out}: {e}", file=sys.stderr)
    out.write_text(json.dumps(blob, indent=1) + "\n")
    print(f"wrote {out} ({len(blob['rows'])} rows)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=["export", "convert", "run"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--ref-dir", default=str(ROOT / "tests" / "fixtures" / "ref"))
    ap.add_argument("--out-dir", default=str(ROOT / "tmp" / "trt"))
    ap.add_argument("--only", default="")
    ap.add_argument("--export-device", default="cpu",
                    help="device the module is traced on during export; half-precision tracing of "
                         "some modules only works on cuda:N")
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"],
                    help="for export: the precision of the traced module (fp32 is the one that "
                         "works for both models). For convert/run: which graph to act on.")
    ap.add_argument("--keep", action="store_true", help="reuse an engine already on disk")
    ap.add_argument("--workspace-gib", type=int, default=8)
    ap.add_argument("--merge", action="store_true",
                    help="merge these rows into an existing -o instead of replacing it, so the "
                         "fp16 and fp32 passes end up in one table")
    ap.add_argument("--fp16-block-ops", default="",
                    help="comma-separated ONNX ops the fp16 conversion must leave in float32, on "
                         "top of onnxconverter_common's own list (e.g. Conv for a 5-D patch "
                         "embedding the converter leaves with a float input and a half kernel)")
    ap.add_argument("-o", "--out", default=str(ROOT / "tests" / "results" / "tensorrt.json"))
    a = ap.parse_args()
    return {"export": do_export, "convert": do_convert, "run": do_run}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
