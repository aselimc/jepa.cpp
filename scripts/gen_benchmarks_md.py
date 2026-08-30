#!/usr/bin/env python3
"""Merge the JSONs written by tools/jepa-bench into docs/benchmarks.md.

    scripts/gen_benchmarks_md.py --bench-dir tmp/bench --ref-dir tests/fixtures/ref -o docs/benchmarks.md

Reads every ``*.json`` in ``--bench-dir`` (``meta.json`` carries the box/toolchain description written
by ``scripts/bench_all.sh``) and the PyTorch golden-dump manifests in ``--ref-dir`` for the baseline
column.  Called automatically at the end of ``scripts/bench_all.sh``; run it by hand to rebuild the
document from JSONs collected in several sessions (e.g. a 32-thread sweep plus a 96-thread one).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# GGUF label (basename minus the -<ftype> suffix) -> reference dump and what its forward_s covers.
BASELINES = {
    "ijepa_vith14_1k":             {"ref": "ijepa-vith14-1k",             "scope": "encoder"},
    "lejepa-vits16-pretrain-in1k": {"ref": "lejepa-vits16",               "scope": "encoder"},
    "lewm-pusht":                  {"ref": "lewm-pusht",                  "scope": "encoder+projector+1-frame predictor", "fn": "lewm"},
    "vjepa2_1-vitb-384":           {"ref": "vjepa2_1-vitb-384",           "scope": "encoder"},
    "vjepa2-vitl-fpc16-256-ssv2":  {"ref": "vjepa2-vitl-fpc16-256-ssv2",  "scope": "encoder+attentive pooler+classifier", "fn": "ssv2"},
    "vjepa2-vitl-fpc64-256":       {"ref": "vjepa2-vitl-fpc64-256",       "scope": "encoder+predictor", "fn": "fpc64"},
}
# A speedup is only printed where the reference forward is the same work as our encoder.
COMPARABLE = {"encoder", "encoder+projector+1-frame predictor"}

# docs/parity.md, "ms/item"/"ms/clip" of the same encoder graphs on the real fixture inputs (steady
# state, i.e. the second sample of a run for the video models). Used only for the cross-check table.
#   (label, ftype, frames, threads) -> ms
PARITY_MS = {
    ("lejepa-vits16-pretrain-in1k", "f32",  1, 32):   14.0,
    ("lejepa-vits16-pretrain-in1k", "f16",  1, 32):   13.4,
    ("lejepa-vits16-pretrain-in1k", "q8_0", 1, 32):   12.6,
    ("lewm-pusht",                  "f32",  1, 32):   10.3,
    ("lewm-pusht",                  "f16",  1, 32):   10.7,
    ("lewm-pusht",                  "q8_0", 1, 32):    9.7,
    ("ijepa_vith14_1k",             "f32",  1, 32):  185.0,
    ("ijepa_vith14_1k",             "f16",  1, 32):  156.0,
    ("ijepa_vith14_1k",             "q8_0", 1, 32):  138.0,
    ("ijepa_vith14_1k",             "f16",  1, 96):  122.0,
    ("vjepa2-vitl-fpc64-256",       "f16", 16, 32):  827.0,
    ("vjepa2-vitl-fpc64-256",       "f16", 64, 32): 6386.0,
    ("vjepa2-vitl-fpc64-256",       "f16", 64, 96): 4076.0,
    ("vjepa2-vitl-fpc64-256",       "q8_0", 16, 32): 852.0,
    ("vjepa2-vitl-fpc64-256",       "q8_0", 64, 32): 7203.0,
    ("vjepa2-vitl-fpc16-256-ssv2",  "f32", 16, 32):  924.0,
    ("vjepa2-vitl-fpc16-256-ssv2",  "f16", 16, 32):  814.0,
    ("vjepa2-vitl-fpc16-256-ssv2",  "q8_0", 16, 32): 771.0,
    ("vjepa2_1-vitb-384",           "f32",  1, 32):   70.0,
    ("vjepa2_1-vitb-384",           "f32", 16, 32):  909.0,
    ("vjepa2_1-vitb-384",           "f16",  1, 32):   63.0,
    ("vjepa2_1-vitb-384",           "f16", 16, 32):  875.0,
    ("vjepa2_1-vitb-384",           "q8_0", 1, 32):   59.0,
    ("vjepa2_1-vitb-384",           "q8_0", 16, 32): 878.0,
}

FTYPE_ORDER = {"f32": 0, "f16": 1, "q8_0": 2, "q6_k": 3, "q5_k": 4, "q5_1": 5, "q5_0": 6,
               "q4_k": 7, "q4_1": 8, "q4_0": 9}
MODE_ORDER = {"encoder": 0, "head": 1, "predictor": 2, "lewm-step": 3, "lewm-rollout": 4}


def mib(n_bytes: float) -> float:
    return n_bytes / (1024.0 * 1024.0)


def ms_str(v: float) -> str:
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def load_runs(bench_dir: Path) -> tuple[list[dict], dict]:
    meta, runs = {}, []
    for p in sorted(bench_dir.glob("*.json")):
        blob = json.loads(p.read_text())
        if p.name == "meta.json":
            meta = blob
            continue
        for r in blob.get("runs", []):
            r["_file"] = p.name
            runs.append(r)
    return runs, meta


def load_baselines(ref_dir: Path) -> dict[str, dict]:
    """label -> {"scope": str, "fn": str|None, "by_frames": {n_frames: (mean_forward_ms, n_samples)}}"""
    out = {}
    for label, spec in BASELINES.items():
        man = ref_dir / spec["ref"] / "manifest.json"
        if not man.exists():
            continue
        blob = json.loads(man.read_text())
        samples = blob.get("samples", [])
        fw = blob.get("framework", {})
        by_frames: dict[int, tuple[float, int]] = {}
        for s in samples:
            n = int(s.get("frames", 1))
            fwd = s.get("timing_s", {}).get("forward_s")
            if fwd is None:
                continue
            tot, cnt = by_frames.get(n, (0.0, 0))
            by_frames[n] = (tot + float(fwd), cnt + 1)
        out[label] = {"scope": spec["scope"], "fn": spec.get("fn"), "framework": fw,
                      "by_frames": {n: (1000.0 * t / c, c) for n, (t, c) in by_frames.items()}}
    return out


def baseline_for(bl: dict, label: str, frames: int):
    """-> (ms, n_samples, scope, footnote) or None"""
    b = bl.get(label)
    if not b:
        return None
    hit = b["by_frames"].get(frames)
    if not hit:
        return None
    return hit[0], hit[1], b["scope"], b["fn"]


def shape_label(r: dict) -> str:
    if r["mode"] in ("lewm-step", "lewm-rollout"):
        return r["shape"]
    if r["frames"] > 1:
        return f"{r['frames']}f {r['height']}x{r['width']}"
    return f"{r['height']}x{r['width']}"


def sort_key(r: dict):
    return (r["model"], MODE_ORDER.get(r["mode"], 9), r["frames"], FTYPE_ORDER.get(r["ftype"], 99), r["threads"])


def table(rows: list[list[str]], header: list[str], align: str) -> str:
    sep = "|" + "|".join({"l": "---", "r": "---:", "c": ":---:"}[a] for a in align) + "|"
    out = ["| " + " | ".join(header) + " |", sep]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="tmp/bench")
    ap.add_argument("--ref-dir", default="tests/fixtures/ref")
    ap.add_argument("-o", "--out", default="docs/benchmarks.md")
    a = ap.parse_args()

    runs, meta = load_runs(Path(a.bench_dir))
    if not runs:
        print(f"no bench JSONs in {a.bench_dir}")
        return 1
    bl = load_baselines(Path(a.ref_dir))
    runs.sort(key=sort_key)
    threads_seen = sorted({r["threads"] for r in runs})
    used_fn: set[str] = set()

    L: list[str] = []
    A = L.append
    A("# jepa.cpp — measured benchmarks")
    A("")
    A("Every number here comes from `tools/jepa-bench` on the box described below, on **synthetic but "
      "deterministic** input (a seeded uint8 stream put through the model's own `jepa.pre.*` "
      "normalisation), so the tables can be reproduced without the fixture media or a Python "
      "environment. They are cross-checked against `docs/parity.md`, which times the same graphs on "
      "the real reference clips (see the cross-check table below).")
    A("")

    # ---- how to reproduce -----------------------------------------------------------------
    A("## How to reproduce")
    A("")
    A("```bash")
    A("git submodule update --init ggml")
    A("cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 32")
    A("")
    A("# the whole matrix at 32 threads (every f32/f16/q8_0 GGUF in models/gguf, every mode it supports)")
    A("scripts/bench_all.sh 32")
    A("")
    A("# the big configurations at 96 threads, appended to the same tmp/bench directory")
    A("scripts/bench_all.sh 96 --keep --only 'ijepa.*-f16|vjepa2-vitl.*-f16|vjepa2_1.*-f16'")
    A("")
    A("# a single configuration by hand")
    A("build/jepa-bench -m models/gguf/vjepa2-vitl-fpc64-256-f16.gguf --frames 64 --threads 32,96 --md")
    A("```")
    A("")
    A("`bench_all.sh` writes one JSON per (file, mode, shape) into `tmp/bench/` plus a `meta.json`, then "
      "rebuilds this document with `scripts/gen_benchmarks_md.py --bench-dir tmp/bench "
      "--ref-dir tests/fixtures/ref -o docs/benchmarks.md`. Add `--include-quants` to sweep the "
      "q4/q5/q6 files as well (skipped by default).")
    A("")

    # ---- box ------------------------------------------------------------------------------
    A("## Box and build")
    A("")
    box = [
        ["CPU", f"{meta.get('cpu', '?')} — {meta.get('cores', '?')} hardware threads, AVX-512"],
        ["RAM", f"{meta.get('mem_gb', '?')} GB"],
        ["Kernel", meta.get("kernel", "?")],
        ["Compiler", f"{meta.get('compiler', '?')}, `-O3 -march=native` (`JEPA_NATIVE=ON`)"],
        ["ggml", f"`{meta.get('ggml_commit', '?')}`, **`GGML_LLAMAFILE={meta.get('ggml_llamafile', '?')}`**"],
        ["Attention", "`ggml_flash_attn_ext`; K/V dtype auto (F32 for f32 files, F16 otherwise)"],
        ["Thread counts", ", ".join(str(t) for t in threads_seen)],
    ]
    A(table([[k, v] for k, v in box], ["setting", "value"], "ll"))
    A("")

    sessions = meta.get("sessions", [])
    if sessions:
        A("Measurement sessions (one `bench_all.sh` invocation each):")
        A("")
        A(table([[s.get("threads", "?"),
                  f"{s.get('warmup', '?')} + {s.get('repeat', '?')}",
                  s.get("start_utc", "?"), s.get("end_utc", "?"),
                  f"{s.get('loadavg_start', '?')} → {s.get('loadavg_end', '?')}",
                  s.get("note", "") or "—"] for s in sessions],
                ["threads", "warmup + measured", "start", "end", "1-min load avg", "note"], "rlllll"))
        A("")
        A(f"The box is shared with other agents, so the 1-minute load average is recorded per session "
          f"(out of {meta.get('cores', '?')} hardware threads; a session's own run contributes its "
          "thread count). Where a session ran against a busy box, `ms min` — the least contended of "
          "the measured runs — is the better estimate of the uncontended cost, and the tables print "
          "it next to the mean.")
        A("")

    A("**What the milliseconds are.** `ms` is the wall time of `ggml_backend_graph_compute` for the "
      "named graph (`jepa_context_last_compute_ms`) — model load, graph build/allocation and the "
      "host-side patchify are excluded; they add well under 1 ms even at 8192 tokens, and the JSONs "
      "keep the full API-call time as `wall_ms_mean` if you want it. `tokens/s` is "
      "`tokens / ms_mean`. `peak RSS` is the process `VmHWM` after the run, i.e. weights + the "
      "largest graph allocation, not a per-graph figure. For `lewm-rollout` the reported ms is **per "
      "rollout step** (the K graphs of one `jepa_lewm_rollout` call divided by K).")
    A("")

    # ---- encoder table --------------------------------------------------------------------
    enc = [r for r in runs if r["mode"] == "encoder"]
    if enc:
        A("## Encoder")
        A("")
        rows = []
        for r in enc:
            b = baseline_for(bl, r["model"], r["frames"])
            if b:
                ms, n, scope, fn = b
                if fn:
                    used_fn.add(fn)
                pt = ms_str(ms) + (f"<sup>{fn}</sup>" if fn else "")
                sp = f"{ms / r['ms_mean']:.2f}x" if scope in COMPARABLE and r["ms_mean"] > 0 else f"n/a<sup>{fn}</sup>"
            else:
                pt, sp = "–", "–"
            rows.append([r["model"], r["ftype"], shape_label(r), f"{r['tokens']:,}".replace(",", " "),
                         r["threads"], f"{r['ms_mean']:.1f}", f"{r['ms_min']:.1f}",
                         f"{1000.0 * r['tokens'] / r['ms_mean']:.0f}" if r["ms_mean"] else "–",
                         pt, sp])
        A(table(rows, ["model", "ftype", "shape", "tokens", "threads", "ms mean", "ms min",
                       "tokens/s", "PyTorch ms", "speedup"], "lllrrrrrrr"))
        A("")
        fw = next((b["framework"] for b in bl.values() if b.get("framework")), {})
        env = ", ".join(x for x in [
            f"torch {fw['torch']}" if fw.get("torch") else "",
            f"transformers {fw['transformers']}" if fw.get("transformers") else "",
            f"{fw['threads']} threads" if fw.get("threads") else "",
        ] if x)
        A("PyTorch baseline = the mean `timing_s.forward_s` over the reference samples with the same "
          "frame count in `tests/fixtures/ref/<model>/manifest.json` — the same box, CPU float32"
          + (f", {env}" if env else "") + ". It is the model forward alone (no decode, no "
          "preprocessing). The speedup column is filled only where that forward is the same work as "
          "our encoder; see the footnotes for the three models where it is not.")
        A("")

    # ---- tokens/s summary -----------------------------------------------------------------
    if enc:
        A("### Encoder throughput summary (tokens/s)")
        A("")
        cols = sorted({(r["ftype"], r["threads"]) for r in enc},
                      key=lambda c: (c[1], FTYPE_ORDER.get(c[0], 99)))
        keys, seen = [], set()
        for r in enc:
            k = (r["model"], r["frames"], r["height"], r["width"])
            if k not in seen:
                seen.add(k)
                keys.append(k)
        cell = {}
        for r in enc:
            cell[((r["model"], r["frames"], r["height"], r["width"]), (r["ftype"], r["threads"]))] = (
                1000.0 * r["tokens"] / r["ms_mean"] if r["ms_mean"] else 0.0)
        tok = {(r["model"], r["frames"], r["height"], r["width"]): r["tokens"] for r in enc}
        rows = []
        for k in keys:
            model, frames, h, w = k
            shp = f"{frames}f {h}x{w}" if frames > 1 else f"{h}x{w}"
            row = [model, shp, f"{tok[k]:,}".replace(",", " ")]
            for c in cols:
                v = cell.get((k, c))
                row.append(f"{v:.0f}" if v else "–")
            rows.append(row)
        A(table(rows, ["model", "shape", "tokens"] + [f"{f} t={t}" for f, t in cols],
                "llr" + "r" * len(cols)))
        A("")

    # ---- dtype effect ----------------------------------------------------------------------
    if enc:
        lo = threads_seen[0]
        cfgs: dict[tuple, dict[str, dict]] = {}
        for r in enc:
            if r["threads"] != lo:
                continue
            cfgs.setdefault((r["model"], r["frames"], r["height"], r["width"]), {})[r["ftype"]] = r
        rows = []
        for k in cfgs:
            d = cfgs[k]
            if "f16" not in d:
                continue
            any_r = next(iter(d.values()))
            f32, f16, q8 = d.get("f32"), d["f16"], d.get("q8_0")
            rows.append([
                any_r["model"], shape_label(any_r), f"{any_r['tokens']:,}".replace(",", " "),
                ms_str(f32["ms_mean"]) if f32 else "–",
                ms_str(f16["ms_mean"]),
                ms_str(q8["ms_mean"]) if q8 else "–",
                f"{f32['ms_mean'] / f16['ms_mean']:.2f}x" if f32 else "–",
                f"{f16['ms_mean'] / q8['ms_mean']:.2f}x" if q8 else "–",
            ])
        if rows:
            A(f"### Effect of the weight dtype (encoder, t={lo})")
            A("")
            A(table(rows, ["model", "shape", "tokens", "f32 ms", "f16 ms", "q8_0 ms",
                           "f32 → f16", "f16 → q8_0"], "llrrrrrr"))
            A("")
            A("Quantisation buys memory, not reliably time. The matmuls do get faster "
              "(`docs/ggml-notes.md` §5 measures 2.2 → 3.3 TFLOP/s going F32 → F16, and ~3.6-4.1 for "
              "Q8_0 on this box), but q8_0 pays for quantising the *activations* on every matmul, "
              "and on the long clips the flash-attention time — F32 work whatever the weights are — "
              "dominates the layer (`docs/ggml-notes.md` §3: 158 ms per ViT-L layer at 8192 tokens "
              "against ~62 ms of matmul). The small image models are launch- and LayerNorm-bound and "
              "barely move at all. Pick the dtype on the accuracy tables in `docs/parity.md` and "
              "`docs/quantization.md` and on the memory table below, not on these milliseconds.")
            A("")

    # ---- thread scaling --------------------------------------------------------------------
    if enc and len(threads_seen) > 1:
        lo, hi = threads_seen[0], threads_seen[-1]
        by_cfg: dict[tuple, dict[int, dict]] = {}
        for r in runs:
            by_cfg.setdefault((r["model"], r["ftype"], r["mode"], r["frames"], r["height"]), {})[r["threads"]] = r
        rows = []
        for k in sorted(by_cfg, key=lambda k: (k[0], MODE_ORDER.get(k[2], 9), k[3])):
            got = by_cfg[k]
            if lo not in got or hi not in got:
                continue
            a_, b_ = got[lo], got[hi]
            rows.append([a_["model"], a_["ftype"], a_["mode"], shape_label(a_),
                         f"{a_['tokens']:,}".replace(",", " "),
                         ms_str(a_["ms_mean"]), ms_str(b_["ms_mean"]),
                         f"{a_['ms_mean'] / b_['ms_mean']:.2f}x" if b_["ms_mean"] else "–"])
        if rows:
            A(f"### Thread scaling ({lo} → {hi} threads)")
            A("")
            A(table(rows, ["model", "ftype", "mode", "shape", "tokens", f"ms t={lo}", f"ms t={hi}",
                           "speedup"], "llllrrrr"))
            A("")
            A(f"Tripling the threads never triples the throughput: the {lo}-thread runs already "
              "saturate a good part of the memory bandwidth, and both the LayerNorm/GELU passes and "
              "the graph launch overhead scale poorly. The gain is largest where a single matmul or "
              "flash-attention tile is big enough to keep 96 workers busy.")
            A("")

    # ---- cross-check against docs/parity.md -----------------------------------------------
    xrows = []
    for r in enc:
        p = PARITY_MS.get((r["model"], r["ftype"], r["frames"], r["threads"]))
        if p is None:
            continue
        d = 100.0 * (r["ms_min"] - p) / p
        xrows.append([r["model"], r["ftype"], shape_label(r), r["threads"],
                      ms_str(r["ms_min"]), ms_str(p), f"{d:+.1f} %"])
    if xrows:
        A("### Cross-check against `docs/parity.md`")
        A("")
        A("`docs/parity.md` times the *same* encoder graphs on the real preprocessed fixture inputs, "
          "reporting the second sample of a run (after the weights are paged in) — effectively a "
          "best-of figure, so `ms min` is what it should be compared against. The synthetic input "
          "used here has the same shape and scale, so the two agree to within run-to-run noise; "
          "`docs/parity.md` itself puts that at ±10-15 % on this shared box.")
        A("")
        A(table(xrows, ["model", "ftype", "shape", "threads", "bench ms min", "parity.md ms", "delta"],
                "lllrrrr"))
        A("")

    # ---- memory ---------------------------------------------------------------------------
    if enc:
        A("## Memory")
        A("")
        rows = [[r["model"], r["ftype"], shape_label(r), f"{mib(r['weight_bytes']):.0f}",
                 f"{mib(r['peak_rss_bytes']):.0f}", f"{r['load_ms']:.0f}"]
                for r in enc if r["threads"] == min(threads_seen)]
        A(table(rows, ["model", "ftype", "shape", "weights MiB", "peak RSS MiB", "model load ms"],
                "lllrrr"))
        A("")
        A("`weights MiB` is `jepa_model_n_bytes()` (the tensor bytes resident after the load, i.e. the "
          "GGUF payload); `peak RSS` additionally covers the graph allocation, the host-side patch "
          "buffer and the output rows, so it grows with the token count — the same weights are "
          "listed once per shape so that growth is visible. The loader `fread`s every tensor into "
          "its own buffer (no `mmap`), so `model load ms` tracks the file size and the page-cache "
          "state; these numbers are all warm-cache. Compare the weight column with the GGUF file "
          "sizes in `docs/quantization.md`.")
        A("")

    # ---- other modes ----------------------------------------------------------------------
    for mode, title, note in (
        ("head", "Attentive-pool head",
         "The classifier head (3 self-attention blocks over the tokens + one cross-attention query + "
         "MLP + linear) on top of the encoder output of the same clip, so the end-to-end "
         "classification cost is encoder + head."),
        ("predictor", "Masked predictor",
         "Worst case for the predictor: context = target = **every** token, i.e. a sequence of "
         "2 x tokens through the 12-layer 384-d predictor."),
    ):
        sel = [r for r in runs if r["mode"] == mode]
        if not sel:
            continue
        A(f"## {title}")
        A("")
        rows = [[r["model"], r["ftype"], shape_label(r), f"{r['tokens']:,}".replace(",", " "),
                 r["threads"], f"{r['ms_mean']:.1f}", f"{r['ms_min']:.1f}", f"{r['encoder_ms']:.1f}",
                 f"{mib(r['peak_rss_bytes']):.0f}"] for r in sel]
        A(table(rows, ["model", "ftype", "shape", "tokens", "threads", "ms mean", "ms min",
                       "encoder ms", "peak RSS MiB"], "lllrrrrrr"))
        A("")
        A(note + " `encoder ms` is the pass that produced this row's input — a **single** graph "
                 "(the second of two, so the weights are warm), not an average of `repeat` runs like "
                 "the `ms` columns, so read the Encoder table for the encoder cost proper.")
        A("")
        if mode == "head":
            # Encoder + head IS like-for-like with a classification manifest's forward_s, unlike the
            # encoder row alone — spell the comparison out, since the footnote promises it.
            crows = []
            for r in sel:
                b = bl.get(r["model"])
                if not b or b["scope"] != "encoder+attentive pooler+classifier":
                    continue
                e = next((q for q in enc if q["model"] == r["model"] and q["ftype"] == r["ftype"]
                          and q["frames"] == r["frames"] and q["threads"] == r["threads"]), None)
                hit = b["by_frames"].get(r["frames"])
                if not e or not hit:
                    continue
                tot = e["ms_mean"] + r["ms_mean"]
                crows.append([r["model"], r["ftype"], shape_label(r), r["threads"],
                              ms_str(e["ms_mean"]), ms_str(r["ms_mean"]), ms_str(tot),
                              ms_str(hit[0]), f"{hit[0] / tot:.2f}x"])
            if crows:
                A("End-to-end classification against the reference — this *is* like-for-like, because "
                  "the manifest's forward is `VJEPA2ForVideoClassification` (encoder + attentive "
                  "pooler + classifier, predictor skipped):")
                A("")
                A(table(crows, ["model", "ftype", "shape", "threads", "encoder ms", "head ms",
                                "total ms", "PyTorch ms", "speedup"], "lllrrrrrr"))
                A("")

    lewm = [r for r in runs if r["mode"] in ("lewm-step", "lewm-rollout")]
    if lewm:
        A("## LeWM world model")
        A("")
        rows = [[r["model"], r["ftype"], r["mode"], r["shape"], r["threads"],
                 f"{r['ms_mean']:.3f}", f"{r['ms_min']:.3f}",
                 f"{1000.0 / r['ms_mean']:.0f}" if r["ms_mean"] else "–"] for r in lewm]
        A(table(rows, ["model", "ftype", "mode", "shape", "threads", "ms mean", "ms min", "steps/s"],
                "llllrrrr"))
        A("")
        A("`lewm-step` is one `jepa_lewm_predict` over the predictor's full 3-frame window; "
          "`lewm-rollout` is `jepa_lewm_rollout` and its ms is **per step** (the growing window means "
          "the first steps are cheaper than the last). Neither includes the encoder or the projector "
          "— see the encoder table for `lewm-pusht` for the cost of turning an image into a "
          "world-model state.")
        A("")

    # ---- footnotes ------------------------------------------------------------------------
    fns = []
    if "fpc64" in used_fn:
        fns.append("<sup>fpc64</sup> the `vjepa2-vitl-fpc64-256` manifest times one `VJEPA2Model` "
                   "forward, which always runs the **predictor** as well (its "
                   "`predictor_last_hidden_state` comes from the same call), so it is an upper bound "
                   "on the encoder and no speedup is claimed against it.")
    if "ssv2" in used_fn:
        fns.append("<sup>ssv2</sup> the SSv2 manifest times `VJEPA2ForVideoClassification`, i.e. "
                   "encoder + attentive pooler + classifier with the predictor skipped. It is "
                   "therefore not comparable with the encoder row alone; the end-to-end table under "
                   "*Attentive-pool head* adds our encoder and head and makes the comparison there.")
    if "lewm" in used_fn:
        fns.append("<sup>lewm</sup> the LeWM manifest times encode + projector + one 1-frame "
                   "predictor call; the two extra graphs are ~1 ms of it (see the world-model "
                   "table), so the speedup is a slight over-estimate.")
    if fns:
        A("## Footnotes")
        A("")
        for f in fns:
            A(f)
            A("")

    A("---")
    A("")
    root = Path(__file__).resolve().parent.parent
    try:
        bench_shown = Path(a.bench_dir).resolve().relative_to(root)
    except ValueError:
        bench_shown = Path(a.bench_dir)
    A(f"Generated by `scripts/gen_benchmarks_md.py` from {len(runs)} runs in `{bench_shown}`. "
      "Cross-check against `docs/parity.md` (same graphs, real fixture inputs) and "
      "`docs/quantization.md` (accuracy per dtype).")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(runs)} runs, {os.path.getsize(out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
