#!/usr/bin/env python3
"""Where the GPU time of an encode goes, per CUDA kernel, and what that says about RoPE and about
attention tiling.

    scripts/profile_gpu.py --device 1 -o tests/results/gpu-profile.json
    scripts/profile_gpu.py --device 1 --only levjepa --keep      # one case, reusing its .nsys-rep

Each case is one `tools/jepa-bench` process under `nsys profile --trace=cuda`. The report groups the
CUDA kernels by (name, grid) — the grid is what separates two graph ops that share a kernel, e.g.
RoPE's broadcast `mul` against LayerNorm's — and attributes each group to a graph op by the launch
count, which is fixed by the model's geometry: a 24-layer encoder timed over 7 passes runs its
per-layer `roll` exactly 24 x 2 x 7 = 336 times. Every attribution in `ops` names the arithmetic it
comes from, so the split can be re-derived from the raw counts alone.

Device-to-device memcpys are counted as GPU time too: ggml serves a contiguous `GGML_OP_CONT` with
`cudaMemcpyAsync`, so the copy in front of RoPE's `roll` is a memcpy rather than a kernel and would
otherwise vanish from the total.

The `attention_tiling` section is the second measurement: the same encoder at token counts that
straddle CUDA flash attention's `FATTN_KQ_STRIDE` (256), which answers whether an off-stride count
pays for the tile it does not fill.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# label -> jepa-bench arguments, and the geometry the attribution arithmetic needs
CASES = {
    "ijepa-vith14-224": {
        "gguf": "ijepa_vith14_1k-f16.gguf", "args": [],
        "note": "image encoder, 256 tokens, no RoPE — the control",
        "layers": 32, "tokens": 256, "rope": False,
    },
    "vjepa2-vitl-16f": {
        "gguf": "vjepa2-vitl-fpc64-256-f16.gguf", "args": ["--frames", "16"],
        "note": "video encoder, 2048 tokens, 3-D RoPE, no mask",
        "layers": 24, "tokens": 2048, "rope": True,
    },
    "vjepa2-vitl-64f": {
        "gguf": "vjepa2-vitl-fpc64-256-f16.gguf", "args": ["--frames", "64"],
        "note": "the same encoder at 8192 tokens",
        "layers": 24, "tokens": 8192, "rope": True,
    },
    "levjepa-vitl16-16f": {
        "gguf": "levjepa-vitl16-f16.gguf", "args": ["--frames", "16"],
        "note": "video encoder, 3137 tokens, 3-D RoPE and an explicit block-causal mask",
        "layers": 24, "tokens": 3137, "rope": True,
    },
}

# The attention-tiling sweep: V-JEPA 2 ViT-L at 16 frames, patch 16, tubelet 2, so the token count
# is 8 * (H/16) * (W/16). 1792 / 2048 / 2304 are multiples of 256 and 1920 / 2176 are not.
TILING = {"256x224": 1792, "256x240": 1920, "256x256": 2048, "256x272": 2176, "256x288": 2304}
FATTN_KQ_STRIDE = 256


def run_nsys(out_base: pathlib.Path, cmd: list[str], keep: bool) -> pathlib.Path:
    rep = out_base.with_suffix(".nsys-rep")
    if keep and rep.exists():
        return rep
    out_base.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["nsys", "profile", "-o", str(out_base), "--force-overwrite", "true",
                    "--trace=cuda", "--sample=none", "--cpuctxsw=none"] + cmd,
                   check=True, capture_output=True)
    return rep


def sqlite_of(rep: pathlib.Path) -> sqlite3.Connection:
    sq = rep.with_suffix(".sqlite")
    if not sq.exists():
        subprocess.run(["nsys", "export", "--type", "sqlite", "-o", str(sq),
                        "--force-overwrite", "true", str(rep)], check=True, capture_output=True)
    return sqlite3.connect(sq)


def kernel_groups(c: sqlite3.Connection) -> tuple[list[dict], float, float]:
    rows = list(c.execute("""select s.value, k.gridX, k.gridY, k.gridZ, sum(k.end-k.start), count(*)
                             from CUPTI_ACTIVITY_KIND_KERNEL k
                             join StringIds s on s.id = k.demangledName
                             group by s.value, k.gridX, k.gridY, k.gridZ
                             order by sum(k.end-k.start) desc"""))
    kern_ns = float(sum(r[4] for r in rows))
    d2d_ns = float(c.execute("select coalesce(sum(end-start), 0) from CUPTI_ACTIVITY_KIND_MEMCPY "
                             "where copyKind = 8").fetchone()[0])
    d2d_cnt = int(c.execute("select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY "
                            "where copyKind = 8").fetchone()[0])
    groups = [{"kernel": r[0], "grid": [r[1], r[2], r[3]], "ns": r[4], "launches": r[5]}
              for r in rows]
    if d2d_cnt:
        groups.append({"kernel": "[CUDA memcpy Device-to-Device] (ggml serves a contiguous "
                                 "GGML_OP_CONT with cudaMemcpyAsync)",
                       "grid": None, "ns": d2d_ns, "launches": d2d_cnt})
    total = kern_ns + d2d_ns
    for g in groups:
        g["pct"] = round(100.0 * g["ns"] / total, 2)
        g["ms_total"] = round(g["ns"] / 1e6, 3)
        del g["ns"]
    return groups, kern_ns, d2d_ns


def rope_share(groups: list[dict], layers: int, runs: int, total_ns: float) -> dict:
    """RoPE's kernels, identified by launch count against the model's geometry.

    Per layer the chain runs once for q and once for k, so every RoPE group is a multiple of
    2 * layers * runs; the `mul` is doubled again (cos and sin) and the `add` shares its launch
    configuration with the two residual adds of the same block, which is the one group that has to
    be halved rather than taken whole."""
    per_qk = 2 * layers * runs
    parts, note = [], []
    for g in groups:
        nm, n = g["kernel"], g["launches"]
        ms = g["ms_total"]
        if nm.startswith("roll_f32_cuda"):
            parts.append(("roll", ms, n, "all of it: nothing else in the graph rolls"))
        elif "memcpy Device-to-Device" in nm and n == per_qk:
            parts.append(("cont", ms, n, f"{n} copies = 2 x {layers} layers x {runs} passes"))
        elif "op_mul" in nm and n == 2 * per_qk:
            parts.append(("mul (cos and sin)", ms, n,
                          f"{n} launches = 2 tables x 2 x {layers} x {runs}; LayerNorm's mul has a "
                          "different grid (it broadcasts over rows, not over the head axis)"))
        elif "op_add" in nm and n == 2 * per_qk:
            parts.append(("add", ms / 2, n // 2,
                          f"half of {n}: RoPE's add and the block's two residual adds are both "
                          f"same-shape adds over the same element count, {per_qk} launches each"))
    tot = sum(p[1] for p in parts)
    return {
        "ms_per_pass": {p[0]: round(p[1] / runs, 3) for p in parts},
        "launches": {p[0]: p[2] for p in parts},
        "attribution": {p[0]: p[3] for p in parts},
        "total_ms_per_pass": round(tot / runs, 3),
        "pct_of_gpu_time": round(100.0 * tot * 1e6 / total_ns, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--bench", default=str(ROOT / "build-cuda" / "jepa-bench"))
    ap.add_argument("--gguf-dir", default=str(ROOT / "models" / "gguf"))
    ap.add_argument("--out-dir", default=str(ROOT / "tmp" / "gpu-profile"))
    ap.add_argument("--only", default="", help="substring filter over the case names")
    ap.add_argument("--keep", action="store_true", help="reuse an existing .nsys-rep")
    ap.add_argument("--no-tiling", action="store_true", help="skip the attention-tiling sweep")
    ap.add_argument("-o", "--out", default=str(ROOT / "tests" / "results" / "gpu-profile.json"))
    a = ap.parse_args()

    if not shutil.which("nsys"):
        print("nsys not on PATH (it ships with the CUDA toolkit)", file=sys.stderr)
        return 1
    out_dir = pathlib.Path(a.out_dir)
    runs = a.warmup + a.repeat
    common = ["--gpu", str(a.device), "--warmup", str(a.warmup), "--repeat", str(a.repeat)]

    blob = {
        "task": "where the GPU time of a jepa.cpp encode goes, per CUDA kernel — the profile behind "
                "the fused-RoPE and attention-tiling decisions in docs/performance.md",
        "generated_by": "scripts/profile_gpu.py",
        "protocol": {
            "profiler": "nsys profile --trace=cuda --sample=none --cpuctxsw=none around one "
                        f"tools/jepa-bench process ({a.warmup} warmup + {a.repeat} measured passes, "
                        "so every launch count is a multiple of the passes)",
            "gpu_time": "CUDA kernel time plus device-to-device memcpy time; host-to-device and "
                        "device-to-host transfers are the input upload and the output copy and are "
                        "outside the timed graph",
            "grouping": "by (kernel, grid) — one graph op can share a kernel with another and the "
                        "grid is what tells them apart",
            "precision": "each model's own default accumulation precision (src/jepa.cpp, "
                         "jepa_gpu_prec_f32_default), i.e. the shipping configuration",
            "caveat": "profiling adds launch overhead, so these milliseconds are a breakdown of one "
                      "another and not a substitute for tests/results/benchmarks-gpu.json's timings",
        },
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "device_index": a.device,
        "cases": [],
    }

    for name, spec in CASES.items():
        if a.only and a.only not in name:
            continue
        gguf = pathlib.Path(a.gguf_dir) / spec["gguf"]
        if not gguf.exists():
            print(f"missing {gguf} — skipping {name}", file=sys.stderr)
            continue
        print(f"--- profiling {name}", file=sys.stderr)
        rep = run_nsys(out_dir / name, [a.bench, "-m", str(gguf)] + spec["args"] + common, a.keep)
        c = sqlite_of(rep)
        groups, kern_ns, d2d_ns = kernel_groups(c)
        total_ns = kern_ns + d2d_ns
        flash_ns = float(c.execute("""select coalesce(sum(k.end-k.start), 0)
                                      from CUPTI_ACTIVITY_KIND_KERNEL k
                                      join StringIds s on s.id = k.demangledName
                                      where s.value like 'void flash_attn_ext_f16%'""").fetchone()[0])
        case = {
            "case": name, "note": spec["note"], "gguf": spec["gguf"],
            "args": spec["args"], "layers": spec["layers"], "tokens": spec["tokens"],
            "passes": runs,
            "gpu_ms_per_pass": round(total_ns / 1e6 / runs, 3),
            "kernel_ms_per_pass": round(kern_ns / 1e6 / runs, 3),
            "d2d_memcpy_ms_per_pass": round(d2d_ns / 1e6 / runs, 3),
            "flash_attention": {
                "ms_per_pass": round(flash_ns / 1e6 / runs, 3),
                "pct_of_gpu_time": round(100.0 * flash_ns / total_ns, 2),
                "ns_per_score_element_per_layer":
                    round(flash_ns / runs / spec["layers"] / (spec["tokens"] ** 2), 5),
            },
            "groups": groups,
        }
        if spec["rope"]:
            case["rope_plus_cont"] = rope_share(groups, spec["layers"], runs, total_ns)
        blob["cases"].append(case)

    if not a.no_tiling:
        gguf = pathlib.Path(a.gguf_dir) / "vjepa2-vitl-fpc64-256-f16.gguf"
        tiling = {
            "question": "does a token count that is not a multiple of CUDA flash attention's "
                        "FATTN_KQ_STRIDE (256) pay for the tile it does not fill?",
            "model": "vjepa2-vitl-fpc64-256-f16", "frames": 16, "layers": 24,
            "method": "the same encoder at crops whose token count 8*(H/16)*(W/16) straddles the "
                      "stride; the flash-attention kernel is read on its own out of the profile and "
                      "normalised by N^2 (the true score matrix) and by N*ceil(N/256)*256 (the "
                      "matrix a fully-padded kernel would compute)",
            "rows": [],
        }
        for size, n in TILING.items():
            if not gguf.exists():
                break
            print(f"--- profiling tiling {size} (N={n})", file=sys.stderr)
            rep = run_nsys(out_dir / f"tiling-{size}",
                           [a.bench, "-m", str(gguf), "--frames", "16", "--size", size] + common,
                           a.keep)
            c = sqlite_of(rep)
            flash_ns = float(c.execute("""select coalesce(sum(k.end-k.start), 0)
                                          from CUPTI_ACTIVITY_KIND_KERNEL k
                                          join StringIds s on s.id = k.demangledName
                                          where s.value like 'void flash_attn_ext_f16%'
                                       """).fetchone()[0])
            tiles = math.ceil(n / FATTN_KQ_STRIDE)
            padded = tiles * FATTN_KQ_STRIDE
            tiling["rows"].append({
                "size": size, "tokens": n, "aligned": n % FATTN_KQ_STRIDE == 0,
                "tiles": tiles, "padded_tokens": padded,
                "padding_pct": round(100.0 * (padded - n) / n, 1),
                "flash_ms_per_pass": round(flash_ns / 1e6 / runs, 3),
                "ns_per_score_element_per_layer": round(flash_ns / runs / 24 / (n * n), 5),
                "ns_per_padded_element_per_layer": round(flash_ns / runs / 24 / (n * padded), 5),
            })
        blob["attention_tiling"] = tiling

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1) + "\n")
    print(f"wrote {out} ({len(blob['cases'])} cases, {os.path.getsize(out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
