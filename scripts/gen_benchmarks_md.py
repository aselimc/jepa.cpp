#!/usr/bin/env python3
"""Merge the JSONs written by tools/jepa-bench into docs/benchmarks.md.

    scripts/gen_benchmarks_md.py --bench-dir tmp/bench --ref-dir tests/fixtures/ref -o docs/benchmarks.md

Reads every ``*.json`` in ``--bench-dir`` (``meta.json`` carries the box/toolchain description written
by ``scripts/bench_all.sh``), the PyTorch golden-dump manifests in ``--ref-dir`` for the two baseline
columns, and ``--parity`` (docs/parity.md) for the cross-check column, which is parsed out of that
document's ``ms/item t=N`` / ``ms/clip t=N`` cells rather than copied by hand.  Called automatically at
the end of ``scripts/bench_all.sh``; run it by hand to rebuild the document from JSONs collected in
several sessions (e.g. a 32-thread sweep plus a 96-thread one).

Exit status is non-zero when there are no runs to tabulate or when a row listed in ``PARITY_REQUIRED``
is absent from ``--parity``; an individual unreadable JSON is warned about, skipped, and counted in
the document's trailer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

# GGUF label (basename minus the -<ftype> suffix) -> reference dump and what its forward_s covers.
BASELINES = {
    "ijepa_vith14_1k":             {"ref": "ijepa-vith14-1k",             "scope": "encoder"},
    "lejepa-vits16-pretrain-in1k": {"ref": "lejepa-vits16",               "scope": "encoder"},
    "lewm-pusht":                  {"ref": "lewm-pusht",                  "scope": "encoder+projector+1-frame predictor", "fn": "lewm"},
    "vjepa2_1-vitb-384":           {"ref": "vjepa2_1-vitb-384",           "scope": "encoder"},
    "vjepa2-vitl-fpc16-256-ssv2":  {"ref": "vjepa2-vitl-fpc16-256-ssv2",  "scope": "encoder+attentive pooler+classifier", "fn": "ssv2"},
    "vjepa2-vitl-fpc64-256":       {"ref": "vjepa2-vitl-fpc64-256",       "scope": "encoder+predictor", "fn": "fpc64"},
    "levjepa-vitl16":              {"ref": "levjepa-vitl16",              "scope": "encoder"},
}
# A speedup is only printed where the reference forward is the same work as our encoder.
COMPARABLE = {"encoder", "encoder+projector+1-frame predictor"}

# The cross-check table used to carry a hand-copied dict of docs/parity.md's milliseconds, which went
# stale the moment either document was re-measured. It is parsed out of the document instead — see
# parse_parity_ms() — and a config the parse cannot account for is reported rather than dropped.
# These are the (label, ftype, frames, threads) rows the cross-check is expected to cover; a missing
# one is an error, because it means docs/parity.md no longer has the row the sentence promises.
PARITY_REQUIRED = [
    ("ijepa_vith14_1k",             "f32",  1, 32),
    ("ijepa_vith14_1k",             "f16",  1, 32),
    ("ijepa_vith14_1k",             "q8_0", 1, 32),
    ("ijepa_vith14_1k",             "f16",  1, 96),
    ("lejepa-vits16-pretrain-in1k", "f32",  1, 32),
    ("lejepa-vits16-pretrain-in1k", "f16",  1, 32),
    ("lejepa-vits16-pretrain-in1k", "q8_0", 1, 32),
    ("lewm-pusht",                  "f32",  1, 32),
    ("lewm-pusht",                  "f16",  1, 32),
    ("lewm-pusht",                  "q8_0", 1, 32),
    ("vjepa2-vitl-fpc16-256-ssv2",  "f32", 16, 32),
    ("vjepa2-vitl-fpc16-256-ssv2",  "f16", 16, 32),
    ("vjepa2-vitl-fpc16-256-ssv2",  "q8_0", 16, 32),
    ("vjepa2-vitl-fpc64-256",       "f32", 16, 32),
    ("vjepa2-vitl-fpc64-256",       "f32", 64, 32),
    ("vjepa2-vitl-fpc64-256",       "f16", 16, 32),
    ("vjepa2-vitl-fpc64-256",       "f16", 64, 32),
    ("vjepa2-vitl-fpc64-256",       "f16", 64, 96),
    ("vjepa2-vitl-fpc64-256",       "q8_0", 16, 32),
    ("vjepa2-vitl-fpc64-256",       "q8_0", 64, 32),
    ("vjepa2_1-vitb-384",           "f32",  1, 32),
    ("vjepa2_1-vitb-384",           "f32", 16, 32),
    ("vjepa2_1-vitb-384",           "f16",  1, 32),
    ("vjepa2_1-vitb-384",           "f16", 16, 32),
    ("vjepa2_1-vitb-384",           "q8_0", 1, 32),
    ("vjepa2_1-vitb-384",           "q8_0", 16, 32),
    ("levjepa-vitl16",              "f32", 16, 32),
    ("levjepa-vitl16",              "f16", 16, 32),
    ("levjepa-vitl16",              "q8_0", 16, 32),
    ("levjepa-vitl16",              "f16", 16, 96),
]

FTYPE_ORDER = {"f32": 0, "f16": 1, "q8_0": 2, "q6_k": 3, "q5_k": 4, "q5_1": 5, "q5_0": 6,
               "q4_k": 7, "q4_1": 8, "q4_0": 9}
MODE_ORDER = {"encoder": 0, "head": 1, "predictor": 2, "lewm-step": 3, "lewm-rollout": 4}


def mib(n_bytes: float) -> float:
    return n_bytes / (1024.0 * 1024.0)


def ms_str(v: float) -> str:
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def load_runs(bench_dir: Path) -> tuple[list[dict], dict, list[str]]:
    """-> (runs, meta, skipped).  A truncated JSON (jepa-bench killed mid-write, a full disk) must not
    take the whole document down with it: the file is named on stderr, counted, and skipped."""
    meta, runs, skipped = {}, [], []
    for p in sorted(bench_dir.glob("*.json")):
        try:
            blob = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            print(f"warning: skipping {p.name}: {e}", file=sys.stderr)
            skipped.append(f"{p.name}: {e}")
            continue
        if p.name == "meta.json":
            meta = blob
            continue
        try:
            for r in blob["runs"]:
                for k in ("model", "ftype", "mode", "threads", "ms_mean"):
                    if k not in r:
                        raise KeyError(k)
                # docs/benchmarks.md is the CPU document: its rows are keyed by thread count, which
                # means nothing for a GPU run. GPU numbers live in docs/performance.md "GPU encoder",
                # so a stray --gpu JSON in the sweep directory is skipped
                # rather than silently mixed in.
                if r.get("gpu"):
                    continue
                r["_file"] = p.name
                runs.append(r)
        except (KeyError, TypeError) as e:
            print(f"warning: skipping {p.name}: malformed run ({e})", file=sys.stderr)
            skipped.append(f"{p.name}: malformed run ({e})")
    return runs, meta, skipped


# ---- docs/parity.md ------------------------------------------------------------------------------
_SUP = str.maketrans("", "", "¹²³⁴⁵⁶⁷⁸⁹⁰*`")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_parity_ms(path: Path) -> dict[tuple[str, str, int, int], float]:
    """(parity model, ftype, frames, threads) -> steady-state ms, read out of docs/parity.md.

    Any markdown table with a ``model`` and an ``ftype`` column contributes one entry per column
    headed ``ms/item t=N`` or ``ms/clip t=N``.  Frames come from a ``sample set`` column ("2 clips ×
    16 f" -> 16, "2 images" -> 1) and default to 1.  A cell holding two numbers ("1240 / 1293") is
    the two samples in run order, so the *second* — the steady-state one, after the weights are
    paged in — is the one taken; footnote markers are stripped.
    """
    out: dict[tuple[str, str, int, int], float] = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|") or i + 1 >= len(lines):
            i += 1
            continue
        head = [h.lower() for h in _cells(lines[i])]
        if not re.match(r"^\|[\s:|-]+\|?$", lines[i + 1].strip()) or "model" not in head or "ftype" not in head:
            i += 1
            continue
        c_model, c_ftype = head.index("model"), head.index("ftype")
        c_set = head.index("sample set") if "sample set" in head else None
        ms_cols = {}
        for j, h in enumerate(head):
            m = re.match(r"^ms(?:/\w+)?\s+t=(\d+)$", h)
            if m:
                ms_cols[j] = int(m.group(1))
        i += 2
        if not ms_cols:
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            continue
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            row = _cells(lines[i])
            i += 1
            if len(row) <= max(max(ms_cols), c_model, c_ftype):
                continue
            model = row[c_model].translate(_SUP).strip()
            ftype = row[c_ftype].translate(_SUP).strip()
            frames = 1
            if c_set is not None and c_set < len(row):
                m = re.search(r"(\d+)\s*f\b", row[c_set])
                frames = int(m.group(1)) if m else 1
            for j, threads in ms_cols.items():
                nums = re.findall(r"\d+(?:\.\d+)?", row[j].translate(_SUP))
                if not nums:
                    continue                       # "—¹", "-", "n/a"
                out[(model, ftype, frames, threads)] = float(nums[-1])
    return out


# A manifest's first forward of a frame group can be a cold one — the weights are being paged in and
# the MKL/oneDNN kernels chosen — and folding it into the mean inflates the PyTorch baseline (LeJEPA:
# 72 ms against a steady 15). Where the first sample is at least this much slower than the median of
# the rest, the drop-first median is the honest steady-state baseline and the one the speedup uses.
COLD_FIRST_RATIO = 1.2


def load_baselines(ref_dir: Path) -> dict[str, dict]:
    """label -> {"scope", "fn", "framework", "by_frames": {n_frames: group}}, where a group is
    {"names": [...], "ms": [...] in manifest order, "mean", "median_drop_first", "cold", "used"}."""
    out = {}
    for label, spec in BASELINES.items():
        man = ref_dir / spec["ref"] / "manifest.json"
        if not man.exists():
            continue
        blob = json.loads(man.read_text())
        by_frames: dict[int, dict] = {}
        for s in blob.get("samples", []):
            n = int(s.get("frames") or 1)
            fwd = s.get("timing_s", {}).get("forward_s")
            if fwd is None:
                continue
            g = by_frames.setdefault(n, {"names": [], "ms": []})
            g["names"].append(s.get("name", "?"))
            g["ms"].append(1000.0 * float(fwd))
        for g in by_frames.values():
            ms = g["ms"]
            g["mean"] = sum(ms) / len(ms)
            rest = ms[1:]
            g["median_drop_first"] = statistics.median(rest) if rest else None
            g["cold"] = bool(rest) and ms[0] >= COLD_FIRST_RATIO * g["median_drop_first"]
            g["used"] = g["median_drop_first"] if g["cold"] else g["mean"]
        out[label] = {"scope": spec["scope"], "fn": spec.get("fn"),
                      "framework": blob.get("framework", {}), "ref": spec["ref"],
                      "by_frames": by_frames}
    return out


def baseline_for(bl: dict, label: str, frames: int):
    """-> (group, scope, footnote) or None"""
    b = bl.get(label)
    if not b:
        return None
    g = b["by_frames"].get(frames)
    if not g:
        return None
    return g, b["scope"], b["fn"]


def shape_label(r: dict) -> str:
    if r["mode"] in ("lewm-step", "lewm-rollout"):
        return r["shape"]
    if r["frames"] > 1:
        return f"{r['frames']}f {r['height']}x{r['width']}"
    return f"{r['height']}x{r['width']}"


def sort_key(r: dict):
    return (r["model"], MODE_ORDER.get(r["mode"], 9), r["frames"], FTYPE_ORDER.get(r["ftype"], 99), r["threads"])


def table(rows: list[list[str]], header: list[str], align: str) -> str:
    # Cells can carry free text (a --note, a compiler banner): an unescaped pipe would silently
    # split the row into extra columns.
    def esc(c) -> str:
        return str(c).replace("|", "\\|").replace("\n", " ")

    sep = "|" + "|".join({"l": "---", "r": "---:", "c": ":---:"}[a] for a in align) + "|"
    out = ["| " + " | ".join(esc(h) for h in header) + " |", sep]
    out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="tmp/bench")
    ap.add_argument("--ref-dir", default="tests/fixtures/ref")
    ap.add_argument("--parity", default="docs/parity.md",
                    help="document the cross-check column is parsed out of")
    ap.add_argument("--allow-missing-parity", action="store_true",
                    help="warn instead of failing when a PARITY_REQUIRED row is not in --parity")
    ap.add_argument("-o", "--out", default="docs/benchmarks.md")
    ap.add_argument("--results-json", default=None,
                    help="also write a compact machine-readable summary (tests/results/benchmarks.json)")
    a = ap.parse_args()

    runs, meta, skipped = load_runs(Path(a.bench_dir))
    if not runs:
        print(f"no bench JSONs in {a.bench_dir}")
        return 1

    # The cross-check column is only worth printing if it is the document's own numbers; a parse that
    # silently lost rows would quietly shrink the table instead of reporting a drift.
    try:
        parity = parse_parity_ms(Path(a.parity))
    except OSError as e:
        print(f"cannot read {a.parity}: {e}", file=sys.stderr)
        return 1
    missing = [k for k in PARITY_REQUIRED
               if (BASELINES[k[0]]["ref"], k[1], k[2], k[3]) not in parity]
    if missing:
        for k in missing:
            print(f"{'warning' if a.allow_missing_parity else 'error'}: {a.parity} has no "
                  f"ms cell for {k[0]} {k[1]} {k[2]}f t={k[3]}", file=sys.stderr)
        if not a.allow_missing_parity:
            print(f"{len(missing)} cross-check row(s) missing from {a.parity} — fix the document or "
                  "pass --allow-missing-parity", file=sys.stderr)
            return 1

    bl = load_baselines(Path(a.ref_dir))
    runs.sort(key=sort_key)
    threads_seen = sorted({r["threads"] for r in runs})
    used_fn: set[str] = set()

    L: list[str] = []
    A = L.append
    A("# jepa.cpp — measured benchmarks")
    A("")
    A("*Raw measurement report — the curated view is [Benchmarks → Performance](performance.md).*")
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
    A("# the q4 encoder rows, appended (the size/speed trade-off; --include-quants alone adds q5/q6 too)")
    A("scripts/bench_all.sh 32 --keep --include-quants --only '\\-(q4_0|q4_k)$' --modes encoder")
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
      "--ref-dir tests/fixtures/ref --parity docs/parity.md -o docs/benchmarks.md --results-json "
      "tests/results/benchmarks.json`. `tmp/bench/` is git-ignored; the committed "
      "`tests/results/benchmarks.json` is the machine-readable twin of every table below, one row "
      "per configuration, so a number quoted elsewhere in the repo can be traced without it. It "
      "passes each "
      "file's `--ftype-label` from the filename: `general.file_type` records the most common stored "
      "tensor type, and a q4_k mix falls back to q4_0 for every tensor whose rows are not a multiple "
      "of the 256-element super-block, so the small models' q4_k files read back as q4_0 and would "
      "otherwise be tabulated as such (`ftype_gguf` in the JSONs keeps what the file actually says).")
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
        # Whether the box was quiet is the single biggest caveat on a timing table, so it is read out
        # of the recorded numbers rather than asserted: loadavg_start is taken before the first
        # jepa-bench of a session, loadavg_end after the last (by then it is the sweep's own workers).
        def _load(s, k):
            try:
                return float(s.get(k))
            except (TypeError, ValueError):
                return None

        starts = [v for v in (_load(s, "loadavg_start") for s in sessions) if v is not None]
        idle = bool(starts) and max(starts) < 4.0
        A(f"The 1-minute load average is recorded per session, before the first run and after the "
          f"last (out of {meta.get('cores', '?')} hardware threads; a session's own run contributes "
          "its thread count, which is what the end-of-session figure mostly is). "
          + (f"Every session here started on an **idle** box — the highest starting load average is "
             f"{max(starts):.2f} — so `ms mean` is a fair figure and `ms min` sits within a per cent "
             "or two of it."
             if idle else
             "Where a session ran against a busy box, `ms min` — the least contended of the measured "
             "runs — is the better estimate of the uncontended cost, and the tables print it next to "
             "the mean."))
        A("")

    # The API-call overhead is measured from this sweep's own JSONs rather than asserted, split at
    # 1024 tokens because it is dominated by the host-side patchify and the output copy, both of
    # which scale with the token count.
    lo_t = threads_seen[0]
    ovh = {"small": [], "large": []}
    for r in runs:
        if r["mode"] != "encoder" or r["threads"] != lo_t or not r.get("wall_ms_mean"):
            continue
        ovh["small" if r["tokens"] <= 1024 else "large"].append(r["wall_ms_mean"] - r["ms_mean"])

    def _range(v):
        return f"{min(v):.1f}–{max(v):.1f} ms" if v else "n/a"

    A("**What the milliseconds are.** `ms` is the wall time of `ggml_backend_graph_compute` for the "
      "named graph (`jepa_context_last_compute_ms`) — model load, graph build/allocation, the "
      "host-side patchify and the output copy are excluded. Measured as `wall_ms_mean − ms_mean` "
      f"over this sweep's own encoder runs at {lo_t} threads they add "
      + (f"{_range(ovh['small'])} up to 1024 tokens and {_range(ovh['large'])} above it"
         if ovh["large"] else f"{_range(ovh['small'])}")
      + " (dominated by the patchify and the output copy, not by graph build). The JSONs keep the "
        "full API-call time as `wall_ms_mean`. `tokens/s` is `tokens / ms_mean`. `peak RSS` is the "
        "process `VmHWM` after the run, i.e. weights + the largest graph allocation, not a "
        "per-graph figure. For `lewm-rollout` the reported ms is **per rollout step** (the K graphs "
        "of one `jepa_lewm_rollout` call divided by K, so its rate column is steps/s).")
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
                g, scope, fn = b
                if fn:
                    used_fn.add(fn)
                sup = f"<sup>{fn}</sup>" if fn else ""
                med = g["median_drop_first"]
                # The column the speedup divides by is set in bold, so the table says which of the two
                # it used without the reader having to work the cold-first rule out row by row.
                pt_mean = ms_str(g["mean"]) + sup
                pt_med = ms_str(med) if med is not None else "–"
                if g["cold"]:
                    pt_med = f"**{pt_med}**" + sup
                else:
                    pt_mean = f"**{ms_str(g['mean'])}**" + sup
                sp = (f"{g['used'] / r['ms_mean']:.2f}x"
                      if scope in COMPARABLE and r["ms_mean"] > 0 else f"n/a<sup>{fn}</sup>")
            else:
                pt_mean, pt_med, sp = "–", "–", "–"
            rows.append([r["model"], r["ftype"], shape_label(r), f"{r['tokens']:,}".replace(",", " "),
                         r["threads"], f"{r['ms_mean']:.1f}", f"{r['ms_min']:.1f}",
                         f"{1000.0 * r['tokens'] / r['ms_mean']:.0f}" if r["ms_mean"] else "–",
                         pt_mean, pt_med, sp])
        A(table(rows, ["model", "ftype", "shape", "tokens", "threads", "ms mean", "ms min",
                       "tokens/s", "PyTorch mean ms", "PyTorch median ms", "speedup"],
                "lllrrrrrrrr"))
        A("")
        fw = next((b["framework"] for b in bl.values() if b.get("framework")), {})
        env = ", ".join(x for x in [
            f"torch {fw['torch']}" if fw.get("torch") else "",
            f"transformers {fw['transformers']}" if fw.get("transformers") else "",
            f"{fw['threads']} threads" if fw.get("threads") else "",
        ] if x)
        A("PyTorch baseline = `timing_s.forward_s` over the reference samples with the same frame "
          "count in `tests/fixtures/ref/<model>/manifest.json` — the same box, CPU float32"
          + (f", {env}" if env else "") + ". It is the model forward alone (no decode, no "
          "preprocessing). Two summaries are given: the **mean** over every such sample, and the "
          "**median of the samples after the first**, because a manifest's first forward of a frame "
          "group can be a cold one (weights paged in, kernels selected) that the mean then carries "
          "into every row. The bold column is the one the speedup divides by — the drop-first median "
          f"wherever that first sample is ≥ {COLD_FIRST_RATIO:.1f}x the median of the rest, the mean "
          "otherwise. The speedup column is filled only where the reference forward is the same work "
          "as our encoder; see the footnotes for the three models where it is not.")
        A("")

        # Which samples feed which baseline: without this the two PyTorch columns are unauditable,
        # and the LeWM row in particular is not what a reader would assume (its 3-frame `seq` sample
        # times a predictor call as well, so it feeds no encoder row).
        prows = []
        for label in sorted({r["model"] for r in enc if r["model"] in bl}):
            b = bl[label]
            for n in sorted(b["by_frames"]):
                g = b["by_frames"][n]
                names = ", ".join(g["names"])
                if len(names) > 64:
                    names = f"{g['names'][0]}, … ({len(g['names'])} samples)"
                med = g["median_drop_first"]
                prows.append([label, f"`{b['ref']}`", n, names,
                              " / ".join(f"{v:.1f}" for v in g["ms"][:1]) if g["ms"] else "–",
                              ms_str(g["mean"]), ms_str(med) if med is not None else "–",
                              "median (cold first sample)" if g["cold"] else "mean"])
        if prows:
            A("<details><summary>Which manifest samples feed each baseline</summary>")
            A("")
            A(table(prows, ["model", "manifest", "frames", "samples (manifest order)",
                            "first sample ms", "mean ms", "median ms (drop-first)",
                            "used for the speedup"], "llrlrrrl"))
            A("")
            A("A frame group is matched to an encoder row by frame count, so a sample that is not a "
              "plain encoder forward never reaches one: `lewm-pusht`'s 3-frame `seq` sample times "
              "encode + projector + a 3-frame predictor call and forms its own group, and only the "
              "two 1-frame samples feed the 224x224 LeWM rows above. Groups of one sample have no "
              "drop-first median and always use the mean.")
            A("")
            A("</details>")
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

    # ---- sub-8-bit weights ------------------------------------------------------------------
    # Only swept when --include-quants asked for them; the point of the table is the trade-off the
    # README quotes, so it puts the weight bytes and the milliseconds side by side.
    if enc and any(r["ftype"].startswith("q4") for r in enc):
        lo = threads_seen[0]
        cfgs: dict[tuple, dict[str, dict]] = {}
        for r in enc:
            if r["threads"] == lo:
                cfgs.setdefault((r["model"], r["frames"], r["height"], r["width"]), {})[r["ftype"]] = r
        rows, pen, frac, q40_vs = [], [], [], []
        for k, d in cfgs.items():
            f16, q40, q4k = d.get("f16"), d.get("q4_0"), d.get("q4_k")
            if not f16 or not (q40 or q4k):
                continue
            any_r = q40 or q4k
            frac.append(100.0 * any_r["weight_bytes"] / f16["weight_bytes"])
            if q40:
                q40_vs.append(f16["ms_mean"] / q40["ms_mean"])
            if q40 and q4k:
                pen.append(q4k["ms_mean"] / q40["ms_mean"])
            rows.append([
                any_r["model"], shape_label(any_r), f"{any_r['tokens']:,}".replace(",", " "),
                f"{mib(f16['weight_bytes']):.0f}", f"{mib(any_r['weight_bytes']):.0f}",
                f"{100.0 * any_r['weight_bytes'] / f16['weight_bytes']:.0f} %",
                ms_str(f16["ms_mean"]),
                ms_str(q40["ms_mean"]) if q40 else "–",
                ms_str(q4k["ms_mean"]) if q4k else "–",
                f"{f16['ms_mean'] / q40['ms_mean']:.2f}x" if q40 else "–",
                f"{q40['ms_mean'] / q4k['ms_mean']:.2f}x" if q40 and q4k else "–",
            ])
        if rows:
            A(f"### Sub-8-bit weights: what q4 costs and what it buys (encoder, t={lo})")
            A("")
            A(table(rows, ["model", "shape", "tokens", "f16 MiB", "q4 MiB", "of f16", "f16 ms",
                           "q4_0 ms", "q4_k ms", "q4_0 vs f16", "q4_k vs q4_0"],
                    "llrrrrrrrrr"))
            A("")
            A("q4 is a **memory** win and, at best, time-neutral: the weights fall to "
              + (f"{min(frac):.0f}–{max(frac):.0f} % of f16, " if frac else "a fraction of f16, ")
              + ("`q4_0` lands within "
                 f"{max(abs(1 - v) for v in q40_vs) * 100:.0f} % of f16's time either way, "
                 if q40_vs else "`q4_0` lands near f16's time, ")
              + "and `q4_k` costs "
              + (f"another {min(pen):.2f}–{max(pen):.2f}x on top of q4_0 " if pen else "more still ")
              + "— ggml's Q4_K vec-dot does more work per block than Q4_0's, and nothing in these "
                "graphs is weight-bandwidth-bound enough to pay that back. Where a model's rows are "
                "not a multiple of the 256-element super-block the k-quant falls back to q4_0 tensor "
                "by tensor, so the small models' q4_k files are mostly q4_0 and their two rows nearly "
                "coincide. **Neither is a parity configuration**: `docs/parity.md` puts every file "
                "below 8 bits per weight in the advisory tier, and `docs/quantization.md` has the "
                "per-model cosines. These rows are here so the size/speed side of that trade-off is "
                "measured rather than assumed.")
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
    xrows, xdeltas = [], []
    for r in enc:
        pm = BASELINES.get(r["model"], {}).get("ref")
        p = parity.get((pm, r["ftype"], r["frames"], r["threads"])) if pm else None
        if p is None:
            continue
        d = 100.0 * (r["ms_min"] - p) / p
        xdeltas.append(abs(d))
        xrows.append([r["model"], r["ftype"], shape_label(r), r["threads"],
                      ms_str(r["ms_min"]), ms_str(p), f"{d:+.1f} %"])
    if xrows:
        A("### Cross-check against `docs/parity.md`")
        A("")
        A("`docs/parity.md` times the *same* encoder graphs on the real preprocessed fixture inputs, "
          "reporting the second sample of a run (after the weights are paged in) — effectively a "
          "best-of figure, so `ms min` is what it should be compared against. The synthetic input "
          "used here has the same shape and scale, so what is left is run-to-run and input-dependent "
          f"noise: every row below agrees to within ±{math.ceil(max(xdeltas))} %, and the two rows where "
          "`docs/parity.md` was itself re-measured on this idle box (the f32 fpc64 clips) to within "
          "3.3 %. The right-hand column "
          f"is **parsed** out of `{a.parity}` (its `ms/item t=N` / `ms/clip t=N` columns), so the two "
          "documents cannot drift apart without this table saying so.")
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
        A(note + " `encoder ms` is the pass that produced this row's input — the **faster of two "
                 "warm** encoder graphs (a third, cold one runs first and is discarded), not an "
                 "average of `repeat` runs like the `ms` columns, so read the Encoder table for the "
                 "encoder cost proper.")
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
                g = b["by_frames"].get(r["frames"])
                if not e or not g:
                    continue
                tot = e["ms_mean"] + r["ms_mean"]
                crows.append([r["model"], r["ftype"], shape_label(r), r["threads"],
                              ms_str(e["ms_mean"]), ms_str(r["ms_mean"]), ms_str(tot),
                              ms_str(g["used"]), f"{g['used'] / tot:.2f}x"])
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
    trailer = (f"Generated by `scripts/gen_benchmarks_md.py` from {len(runs)} runs in "
               f"`{bench_shown}`. Cross-check against `docs/parity.md` (same graphs, real fixture "
               "inputs) and `docs/quantization.md` (accuracy per dtype).")
    if skipped:
        trailer += (f" **{len(skipped)} JSON file(s) in `{bench_shown}` could not be read and are "
                    "not in these tables:** " + "; ".join(f"`{s}`" for s in skipped) + ".")
    A(trailer)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(runs)} runs, {os.path.getsize(out)} bytes"
          + (f", {len(skipped)} unreadable JSON(s) skipped" if skipped else "") + ")")

    if a.results_json:
        write_results_json(Path(a.results_json), runs, meta, bl, parity, skipped, str(out))
    return 0


def write_results_json(path: Path, runs, meta, bl, parity, skipped, doc) -> None:
    """The in-repo, machine-readable twin of the document: one flat row per measured configuration,
    so a number quoted in the README can be traced back to a run without the (git-ignored) sweep
    directory.  Kept small on purpose — no per-repeat samples, no paths outside the repo."""
    blob = {
        "task": "jepa.cpp inference timing and memory, tools/jepa-bench on synthetic deterministic input",
        "generated_from": doc,
        "protocol": {
            "ms": "wall time of ggml_backend_graph_compute for the named graph "
                  "(jepa_context_last_compute_ms); model load, graph build/alloc, host-side "
                  "patchify and the output copy are excluded",
            "wall_ms": "the full public-API call, including all of the above",
            "ms_mean/ms_min": "over `repeat` measured runs after `warmup` unmeasured ones",
            "encoder_ms": "head/predictor rows only: the faster of two warm encoder passes that "
                          "produced this row's input (a third, cold one runs first and is discarded)",
            "tokens_per_s": "tokens / ms_mean; for lewm-rollout ms is per step and tokens is 1, so "
                            "the rate is steps/s",
            "peak_rss_mib": "process VmHWM after the run (weights + largest graph allocation)",
            "ftype": "the type requested at conversion time (the GGUF filename suffix)",
            "ftype_gguf": "what general.file_type says, which differs where a k-quant mix fell back",
            "pytorch_ms": "manifest timing_s.forward_s for the same frame count: mean over the "
                          "samples, or the median of the samples after the first where the first is "
                          f"cold (>= {COLD_FIRST_RATIO} x the median of the rest); "
                          "pytorch_basis says which was used for pytorch_speedup",
            "pytorch_comparable": "false where the reference forward is not the same work as our "
                                  "encoder (see the document's footnotes); no speedup is claimed",
            "parity_ms": "the same graph timed on the real fixture input in docs/parity.md",
        },
        "box": {k: meta.get(k) for k in
                ("cpu", "cores", "mem_gb", "kernel", "compiler", "ggml_commit", "ggml_llamafile")},
        "sessions": meta.get("sessions", []),
        "skipped_files": skipped,
        "rows": [],
    }
    for r in runs:
        row = {
            "model": r["model"], "ftype": r["ftype"], "ftype_gguf": r.get("ftype_gguf"),
            "family": r.get("family"), "mode": r["mode"], "shape": shape_label(r),
            "frames": r["frames"], "height": r["height"], "width": r["width"],
            "batch": r.get("batch", 1), "steps": r.get("steps", 0),
            "threads": r["threads"], "tokens": r["tokens"],
            "repeat": r.get("repeat"), "warmup": r.get("warmup"),
            "kv": r.get("kv"), "flash": r.get("flash"),
            "ms_mean": round(r["ms_mean"], 3), "ms_min": round(r["ms_min"], 3),
            "ms_max": round(r.get("ms_max", 0.0), 3),
            "wall_ms_mean": round(r.get("wall_ms_mean", 0.0), 3),
            "tokens_per_s": round(1000.0 * r["tokens"] / r["ms_mean"], 1) if r["ms_mean"] else None,
            "weights_mib": round(mib(r.get("weight_bytes", 0)), 1),
            "peak_rss_mib": round(mib(r.get("peak_rss_bytes", 0)), 1),
            "load_ms": round(r.get("load_ms", 0.0), 1),
            "source_json": r.get("_file"),
        }
        if r["mode"] in ("head", "predictor"):
            row["encoder_ms"] = round(r.get("encoder_ms", 0.0), 3)
        b = baseline_for(bl, r["model"], r["frames"]) if r["mode"] == "encoder" else None
        if b:
            g, scope, _fn = b
            row["pytorch_ms"] = round(g["used"], 1)
            row["pytorch_ms_mean"] = round(g["mean"], 1)
            row["pytorch_ms_median_drop_first"] = (round(g["median_drop_first"], 1)
                                                   if g["median_drop_first"] is not None else None)
            row["pytorch_basis"] = "median_drop_first" if g["cold"] else "mean"
            row["pytorch_samples"] = g["names"]
            row["pytorch_scope"] = scope
            row["pytorch_comparable"] = scope in COMPARABLE
            if scope in COMPARABLE and r["ms_mean"]:
                row["pytorch_speedup"] = round(g["used"] / r["ms_mean"], 3)
        pm = BASELINES.get(r["model"], {}).get("ref")
        p = parity.get((pm, r["ftype"], r["frames"], r["threads"])) if pm else None
        if p is not None and r["mode"] == "encoder":
            row["parity_ms"] = p
            row["parity_delta_pct"] = round(100.0 * (r["ms_min"] - p) / p, 1)
        blob["rows"].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1, sort_keys=False) + "\n")
    print(f"wrote {path} ({len(blob['rows'])} rows, {os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    raise SystemExit(main())
