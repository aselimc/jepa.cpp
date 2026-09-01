#!/usr/bin/env python3
"""Generate docs/assets/results.svg — the three-panel results overview README.md and docs/index.md show.

Every value on the figure is read from a committed artifact; nothing is typed in here.

    tests/results/benchmarks.json      CPU encoder latency per model / shape / dtype / thread count,
                                       resident weights, and the PyTorch CPU baseline of each row
    tests/results/accuracy-image.json  Imagenette k-NN top-1, PyTorch against jepa.cpp per dtype
    tests/results/accuracy-video.json  UCF-101 k-NN top-1, the same comparison on clips
    tests/results/accuracy-ssv2.json   SSv2 validation top-1 of the classifier checkpoint, PyTorch
                                       against jepa.cpp per dtype (absent unless the licence-gated
                                       dataset was on the machine)
    tests/results/benchmarks-gpu.json  CUDA encoder latency per model / shape / dtype and the
                                       PyTorch-on-the-same-card baseline, from scripts/bench_gpu.sh
    docs/performance.md                the fallback for those GPU series, so the figure still builds
                                       from a checkout without the artifact: the tables are parsed
                                       out of the document the way scripts/gen_benchmarks_md.py
                                       parses docs/parity.md.  A GPU table is recognised by its
                                       "CPU f16 t=32" (or "jepa.cpp CPU t=32") column, and that column
                                       is also the join key: it repeats the 32-thread f16 millisecond
                                       of a row benchmarks.json already holds.

The three panels are latency per item across the four backends, k-NN top-1 against PyTorch, and the
weights/latency trade-off per dtype on both backends.  Each is also written on its own — the
documentation puts a panel beside the tables it draws, while README.md leads with the four headline
numbers instead (docs/assets/hero.svg, scripts/gen_hero_figure.py).

    python scripts/gen_results_figure.py                  # writes docs/assets/results.svg
    python scripts/gen_results_figure.py --split          # and results-{latency,accuracy,
                                                          #             quantization}.svg
    python scripts/gen_results_figure.py --png tmp/x.png  # also write a PNG to look at
    python scripts/gen_results_figure.py --check          # exit 1 if any of the four is stale

matplotlib is the only dependency, and the figure is the only thing in the repository that needs it,
so it stays out of docs/requirements.txt:

    uv pip install --python .venv/bin/python matplotlib

The committed SVG was written with matplotlib 3.11.1.  Output is byte-identical across runs — no
timestamp, a fixed hash salt for the generated element ids, glyphs as paths — but another matplotlib
may re-emit those paths and make --check report drift, which is why the CI job that runs --check
installs that exact version (.github/workflows/ci.yml, `checks`).

A missing or unreadable artifact drops the series it feeds, with a warning on stderr, rather than
failing the run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH_JSON = ROOT / "tests" / "results" / "benchmarks.json"
BENCH_GPU_JSON = ROOT / "tests" / "results" / "benchmarks-gpu.json"
ACC_IMAGE_JSON = ROOT / "tests" / "results" / "accuracy-image.json"
ACC_VIDEO_JSON = ROOT / "tests" / "results" / "accuracy-video.json"
ACC_SSV2_JSON = ROOT / "tests" / "results" / "accuracy-ssv2.json"
PERF_MD = ROOT / "docs" / "performance.md"
OUT_SVG = ROOT / "docs" / "assets" / "results.svg"

# Okabe-Ito, one colour per implementation + backend, held across all three panels.
C_TORCH_CPU = "#D55E00"   # vermillion
C_TORCH_GPU = "#CC79A7"   # reddish purple
C_CPP_CPU = "#0072B2"     # blue
C_CPP_GPU = "#009E73"     # bluish green
INK = "#1a1a1a"
MUTED = "#5c5c5c"
GRID = "#d8d8d8"
PAGE = "#fbfbfb"          # opaque, so the SVG reads the same in GitHub's light and dark themes

# Dtypes in the order they appear wherever a dtype axis does; anything the artifacts add later is
# appended in sorted order (see dtype_order()).
DTYPES = ("f32", "f16", "q8_0", "q4_0", "q4_k")

# Display names only.  Which models appear is decided by the artifacts; this table spells the ones we
# know about the way the documentation does, and a model that is not in it keeps its raw id, so a
# family added to the JSONs later still shows up.
PRETTY = {
    "ijepavith14": "I-JEPA ViT-H/14",
    "lejepavits16": "LeJEPA ViT-S/16",
    "levjepavitl16": "LeVJEPA ViT-L/16",
    "lewm": "LeWM ViT-Ti/14",
    "vjepa2vitlfpc16256ssv2": "V-JEPA 2 ViT-L SSv2",
    "vjepa2vitlfpc64": "V-JEPA 2 ViT-L fpc64",
    "vjepa21vitb": "V-JEPA 2.1 ViT-B/384",
}

# An Imagenette row whose PyTorch baseline is this low is an off-task sanity row rather than an
# accuracy claim — docs/accuracy-image.md says exactly that of LeWM, at 27.0 % — and on the same
# delta strip as the two real k-NN benchmarks it would read as a fidelity result.
MIN_BASELINE_TOP1 = 0.50


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:  # missing, unreadable or malformed — the caller drops that series
        warn(f"cannot read {path.relative_to(ROOT)}: {e}")
        return None


# ---- naming ---------------------------------------------------------------------------------------

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def pretty(model_id: str) -> str:
    """Display name for a model id from any of the artifacts (their spellings differ)."""
    s = slug(model_id)
    best = ""
    for key in sorted(PRETTY):
        if s.startswith(key) and len(key) > len(best):
            best = key
    return PRETTY[best] if best else model_id


def shape_label(row: dict) -> str:
    """'16 f 256², 2 048 tok' — the shape of one benchmarks.json encoder row."""
    frames = int(row.get("frames") or 1)
    h, w = int(row.get("height") or 0), int(row.get("width") or 0)
    px = f"{h}²" if h == w else f"{h}×{w}"
    head = px if frames <= 1 else f"{frames} f {px}"
    return f"{head}, {int(row['tokens']):,} tok".replace(",", " ")


def dtype_order(found) -> list[str]:
    return [d for d in DTYPES if d in found] + sorted(d for d in found if d not in DTYPES)


def fmt_ms(v: float) -> str:
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


# ---- docs/performance.md --------------------------------------------------------------------------

_STRIP = str.maketrans("", "", "*`ᵃᵇᶜᵈ")   # bold, code ticks, footnote letters

# The column both GPU tables repeat from the CPU encoder table, and the key their rows join on.
JOIN_RE = re.compile(r"^(jepa\.cpp\s+)?cpu(\s+f16)?(\s+t=32)?$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _num(cell: str):
    """(value, text as printed) of a markdown cell, or None for '–', 'n/a', an empty cell."""
    txt = cell.translate(_STRIP).strip()
    m = re.search(r"\d+(?:\.\d+)?", txt)
    return (float(m.group(0)), m.group(0)) if m else None


def md_tables(path: pathlib.Path):
    """Yield (headers, rows) for every markdown table in a document."""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|") or i + 1 >= len(lines):
            i += 1
            continue
        if not re.match(r"^\|[\s:|-]+\|?$", lines[i + 1].strip()):
            i += 1
            continue
        headers = [h.translate(_STRIP).strip().lower() for h in _cells(lines[i])]
        rows = []
        i += 2
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            rows.append(_cells(lines[i]))
            i += 1
        yield headers, rows


def parse_gpu_tables(path: pathlib.Path, cpu_f16: dict) -> dict:
    """key -> {'cuda': {dtype: (ms, text)}, 'torch_gpu': {precision: (ms, text)}}.

    ``cpu_f16`` maps a benchmarks.json encoder key to its 32-thread f16 millisecond, and a markdown
    row is attached to the key whose millisecond its own CPU column repeats.  A table without such a
    column, or without a column this function recognises as a value, is skipped — that covers every
    CPU-only table on the page and the predictor/head table, whose rows are not encoder rows.
    """
    out: dict = {}
    if not cpu_f16:
        return out
    for headers, rows in md_tables(path):
        join_col = next((j for j, h in enumerate(headers) if JOIN_RE.match(h)), None)
        values = {}
        for j, h in enumerate(headers):
            if h in DTYPES:
                values[j] = ("cuda", h)
            elif h == "jepa.cpp cuda":
                values[j] = ("cuda", "f16")          # that table's default-precision column
            elif h.startswith("torch fp"):
                values[j] = ("torch_gpu", h.split()[-1])
        if join_col is None or not values:
            continue
        for row in rows:
            if len(row) <= max(max(values), join_col):
                continue
            anchor = _num(row[join_col])
            if anchor is None:
                continue
            key, rel = min(((k, abs(v - anchor[0]) / v) for k, v in sorted(cpu_f16.items())),
                           key=lambda kv: kv[1])
            if rel > 0.02:
                warn(f"{path.name}: no benchmarks.json row within 2 % of the CPU column "
                     f"{anchor[1]} ms — row {row[0]!r} dropped")
                continue
            slot = out.setdefault(key, {"cuda": {}, "torch_gpu": {}})
            for j, (kind, name) in sorted(values.items()):
                got = _num(row[j])
                if got is None:
                    continue
                have = slot[kind].get(name)
                if have and abs(have[0] - got[0]) / got[0] > 0.02:
                    warn(f"{path.name}: {key} {kind} {name} appears as {have[1]} and {got[1]}")
                slot[kind][name] = got
    return out


# ---- tests/results/benchmarks-gpu.json ------------------------------------------------------------

def fmt_torch_ms(v: float) -> str:
    """PyTorch's GPU rows are small enough that a tenth would be a per cent of the number, so
    docs/performance.md quotes them to the hundredth below 100 ms.  Same rule here."""
    return f"{v:.2f}" if v < 100 else f"{v:.1f}"


def gpu_from_artifact(bench_gpu: dict) -> dict:
    """The same key → {'cuda', 'torch_gpu'} structure parse_gpu_tables() builds, straight out of
    the GPU sweep's artifact — no join and no rounding-trip through the document.

    Only the default-precision, single-item encoder rows become dtype series. A row measured with
    the accumulation precision the family does *not* default to is the same file on a different
    numeric path and is a column on the page rather than a dtype; a `--batch B` row is B items in
    one graph and belongs to the batching table. `gpu_prec_explicit` says which is which, and for
    rows written before that field existed the two questions had the same answer, because every
    default was then F32.

    The artifact spans two cards, and the choice is made per (model, shape) rather than per dtype:
    the card is the one the PyTorch baseline beside this figure ran on, and a shape that card never
    measured falls back to the other one whole. Per-dtype fallback would put two cards inside one
    group, where the point of the group is that its bars differ only in the dtype.
    """
    want_dev = (bench_gpu or {}).get("pytorch_gpu", {}).get("box", {}).get("device_index")
    want_dev = f"CUDA{want_dev}" if want_dev is not None else None
    by_key: dict = {}
    for row in (bench_gpu or {}).get("rows", []):
        if row.get("mode") != "encoder" or (row.get("batch") or 1) != 1:
            continue
        if row.get("gpu_prec_explicit", row.get("gpu_prec", "f32") != "f32"):
            continue
        key = (row["model"], int(row["tokens"]))
        by_key.setdefault(key, {}).setdefault(row["device"], {})[row["ftype"]] = float(row["ms_mean"])
    out: dict = {}
    for key, per_dev in by_key.items():
        dev = want_dev if want_dev in per_dev else next(iter(per_dev))
        out[key] = {"cuda": {ft: (ms, fmt_ms(ms)) for ft, ms in per_dev[dev].items()},
                    "torch_gpu": {}}
    for row in (bench_gpu or {}).get("pytorch_gpu", {}).get("rows", []):
        key = (row["model"], int(row["tokens"]))
        if key not in out:
            continue                       # a baseline for a shape we did not time ourselves
        ms = float(row["ms_mean"])
        out[key]["torch_gpu"][row["precision"]] = (ms, fmt_torch_ms(ms))
    return out


def gpu_series(perf: str, cpu_f16: dict) -> dict:
    """Every GPU series both figures draw.  The committed artifact is preferred; docs/performance.md
    is the fallback, so a checkout that has the document but not the JSON still draws the figure."""
    if BENCH_GPU_JSON.exists():
        got = gpu_from_artifact(load_json(BENCH_GPU_JSON))
        if got:
            return got
        warn(f"{BENCH_GPU_JSON.relative_to(ROOT)} has no usable GPU encoder rows — "
             "falling back to the tables in docs/performance.md")
    return parse_gpu_tables(PERF_MD, cpu_f16) if perf else {}


# ---- benchmarks.json ------------------------------------------------------------------------------

def encoder_groups(bench: dict) -> dict:
    """(model, frames, tokens) -> one plotted row: the CPU sweep, the weights, the PyTorch CPU
    baseline and whether that baseline is like-for-like work."""
    groups: dict = {}
    for row in bench.get("rows", []):
        if row.get("mode") != "encoder" or int(row.get("batch") or 1) != 1:
            continue
        key = (row["model"], int(row.get("frames") or 1), int(row["tokens"]))
        g = groups.setdefault(key, {
            "model": row["model"], "label": pretty(row["model"]), "shape": shape_label(row),
            "tokens": int(row["tokens"]), "cpu": {}, "weights": {},
            "torch_cpu": None, "torch_speedup": None, "torch_comparable": False,
        })
        g["cpu"].setdefault(row["ftype"], {})[int(row["threads"])] = float(row["ms_mean"])
        if row.get("weights_mib"):
            g["weights"][row["ftype"]] = float(row["weights_mib"])
        if row.get("pytorch_ms"):
            g["torch_cpu"] = float(row["pytorch_ms"])
            g["torch_comparable"] = bool(row.get("pytorch_comparable"))
            if row.get("pytorch_speedup") and row["ftype"] == "f16" and int(row["threads"]) == 32:
                g["torch_speedup"] = float(row["pytorch_speedup"])
    return groups


# ---- panel 1: latency -----------------------------------------------------------------------------

def panel_speed(ax, groups: dict, gpu: dict) -> None:
    rows = [g for g in groups.values() if 32 in g["cpu"].get("f16", {})]
    if not rows:
        ax.text(0.5, 0.5, "no encoder rows in tests/results/benchmarks.json", ha="center",
                va="center", transform=ax.transAxes, color=MUTED)
        ax.set_axis_off()
        return
    rows.sort(key=lambda g: (g["cpu"]["f16"][32], g["label"]))

    bars = [("torch_cpu", C_TORCH_CPU), ("cpp_cpu", C_CPP_CPU),
            ("torch_gpu", C_TORCH_GPU), ("cpp_gpu", C_CPP_GPU)]
    h, xmin, texts = 0.20, 0.45, []
    for i, g in enumerate(rows):
        gk = (g["model"], g["tokens"])
        cuda = gpu.get(gk, {}).get("cuda", {})
        torch_gpu = gpu.get(gk, {}).get("torch_gpu", {})
        cpu32, cpu96 = g["cpu"]["f16"][32], g["cpu"]["f16"].get(96)
        gpu_dtype = "f16" if "f16" in cuda else (dtype_order(cuda)[0] if cuda else None)
        tg_prec = "fp16" if "fp16" in torch_gpu else (sorted(torch_gpu)[0] if torch_gpu else None)
        vals = {
            "torch_cpu": (g["torch_cpu"], fmt_ms(g["torch_cpu"]) if g["torch_cpu"] else None, ""),
            "cpp_cpu": (cpu32, fmt_ms(cpu32),
                        f"  {g['torch_speedup']:.1f}× vs torch CPU" if g["torch_speedup"] else ""),
            "torch_gpu": ((*torch_gpu[tg_prec], f"  {tg_prec}") if tg_prec else (None, None, "")),
            "cpp_gpu": ((*cuda[gpu_dtype], f"  {gpu_dtype}, {cpu32 / cuda[gpu_dtype][0]:.1f}× vs our CPU")
                        if gpu_dtype else (None, None, "")),
        }
        for slot, (name, colour) in enumerate(bars):
            v, txt, note = vals[name]
            if not v:
                continue
            y = i + (slot - 1.5) * h
            hatch = "///" if (name == "torch_cpu" and not g["torch_comparable"]) else None
            ax.barh(y, v - xmin, left=xmin, height=h * 0.82, color=colour, zorder=3,
                    hatch=hatch, edgecolor="white" if hatch else "none", linewidth=0.5)
            texts.append((v * 1.10, y, f"{txt}{note}", colour))
            if name == "cpp_cpu" and cpu96:
                ax.plot([cpu96, cpu96], [y - h * 0.41, y + h * 0.41], color="white", lw=1.3,
                        zorder=4, solid_capstyle="butt")

    for x, y, s, c in texts:
        # the knockout keeps a label readable where it runs past its own bar over a longer neighbour
        ax.text(x, y, s, va="center", ha="left", fontsize=6.5, color=c, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", facecolor=PAGE, edgecolor="none", alpha=0.72))

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{g['label']}\n{g['shape']}" for g in rows], fontsize=7.6, linespacing=1.4)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
    ax.set_xscale("log")
    ax.set_xlim(xmin, 42000)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1 000", "10 000"], fontsize=7.5)
    ax.set_xlabel("milliseconds per image or clip through the encoder graph — log scale",
                  fontsize=8, color=MUTED, labelpad=4)
    ax.set_title("Latency per item: PyTorch against jepa.cpp, on the CPU and on one GPU",
                 fontsize=10.5, color=INK, loc="left", pad=8)
    style_axes(ax, xgrid=True)


# ---- panel 2: accuracy ----------------------------------------------------------------------------

def image_knn_series(acc: dict):
    """One entry per Imagenette model: its largest gallery, strongest feature, every dtype."""
    by_model: dict = {}
    for r in acc.get("rows", []):
        by_model.setdefault(r["model"], []).append(r)
    out = []
    for model in sorted(by_model):
        groups: dict = {}
        for r in by_model[model]:
            groups.setdefault((r["feature"], r["gallery"]), []).append(r)

        def rank(item):
            rs = item[1]
            base = next((r["knn_top1"] for r in rs if r["backend"] == "pytorch"), 0.0)
            return (rs[0].get("n_gallery", 0), base)

        (feature, _gallery), rs = max(sorted(groups.items()), key=rank)
        base = next((r["knn_top1"] for r in rs if r["backend"] == "pytorch"), None)
        if base is None:
            warn(f"accuracy-image.json: {model} has no PyTorch row — dropped from the accuracy panel")
            continue
        if base < MIN_BASELINE_TOP1:
            warn(f"accuracy-image.json: {model} k-NN baseline is {base * 100:.1f} % — an off-task "
                 "sanity row, dropped from the accuracy panel")
            continue
        vals = {r["dtype"]: r["knn_top1"] for r in rs if r["backend"] != "pytorch"}
        out.append((model, feature, vals, base, rs[0].get("n_query", 0)))
    return out


def video_knn_series(acc: dict):
    """One entry per UCF-101 model, on the split with the most clips."""
    out = []
    for model in sorted(acc.get("models", {})):
        backends = acc["models"][model].get("backends", {})
        torch = next((b for b in backends.values() if b.get("backend") == "pytorch"), None)
        if not torch or not torch.get("splits"):
            warn(f"accuracy-video.json: {model} has no PyTorch split — dropped")
            continue
        split = max(sorted(torch["splits"]), key=lambda s: torch["splits"][s]["n"])
        vals = {b["dtype"]: b["splits"][split]["knn_top1"] for b in backends.values()
                if b.get("backend") != "pytorch" and split in b.get("splits", {})}
        out.append((model, split, vals, torch["splits"][split]["knn_top1"],
                    torch["splits"][split]["n"]))
    return out


def ssv2_series(acc: dict):
    """The SSv2 classifier's real top-1 on the validation split, per jepa.cpp dtype.

    The only block on this panel whose task has a trained head: the other two are look-ups over
    frozen features, this one is the checkpoint's own attentive pooler and 174-way classifier.
    """
    runs = acc.get("runs", [])
    ref = next((r for r in runs if r["backend"] == "pytorch" and r["scope"] == "full"), None)
    if ref is None:
        warn("accuracy-ssv2.json: no full-split PyTorch row — dropped from the accuracy panel")
        return []
    vals = {r["dtype"]: r["top1"] for r in runs
            if r["backend"] != "pytorch" and r["scope"] == "full" and r.get("dtype")}
    if not vals:
        return []
    return [(acc["model"]["name"], "174-way head", vals, ref["top1"], ref["n_clips"])]


def panel_accuracy(ax, acc_img, acc_vid, acc_ssv2) -> None:
    blocks = []
    if acc_img:
        series = image_knn_series(acc_img)
        if series:
            n = series[0][4]
            blocks.append((f"Imagenette · {n:,} val images · k = 20 cosine vote".replace(",", " "),
                           n, "images", series))
    if acc_vid:
        series = video_knn_series(acc_vid)
        if series:
            n = series[0][4]
            blocks.append((f"UCF-101 · {n} clips · k = 20 cosine vote", n, "clips", series))
    if acc_ssv2:
        series = ssv2_series(acc_ssv2)
        if series:
            n = series[0][4]
            blocks.append((f"SSv2 · {n:,} val clips · the checkpoint's own head".replace(",", " "),
                           n, "clips", series))
    if not blocks:
        ax.text(0.5, 0.5, "no k-NN artifacts", ha="center", va="center", transform=ax.transAxes,
                color=MUTED)
        ax.set_axis_off()
        return

    rows, heads, seps = [], [], []
    y = 0.0
    for title, n_query, noun, series in blocks:
        worst, block = 0.0, []
        for model, _feature, vals, base, _n in series:
            for dt in dtype_order(vals):
                d = (vals[dt] - base) * 100.0
                block.append((y, f"{pretty(model)}   {dt}", d, vals[dt] * 100.0))
                worst = d if abs(d) > abs(worst) else worst
                y += 1
        items = round(abs(worst) * n_query / 100.0)
        note = f"largest deviation {abs(worst):.2f} pp"
        if items and abs(abs(worst) * n_query / 100.0 - items) < 0.05:
            note += f" = {items} {noun[:-1] if items == 1 else noun}"
        heads.append((block[0][0] - 1.0, f"{title}   —   {note}"))
        rows += block
        seps.append(y - 0.5)
        y += 1.7
    seps.pop()

    ax.axvline(0, color=C_TORCH_CPU, lw=1.2, zorder=2)
    for yy, _lbl, d, a in rows:
        ax.plot([0, d], [yy, yy], color=C_CPP_CPU, lw=1.1, zorder=3, alpha=0.8)
        ax.plot([d], [yy], marker="o", ms=5.5, color=C_CPP_CPU, zorder=4)
        ax.text(d + (0.04 if d >= 0 else -0.04), yy, f"{a:.2f} %", fontsize=7, color=INK,
                va="center", ha="left" if d >= 0 else "right", zorder=5)
    for s in seps:
        ax.axhline(s + 0.35, color=GRID, lw=0.8, ls=(0, (4, 3)), zorder=1)

    lim = max(1.1, max(abs(r[2]) for r in rows) * 1.45)
    ax.set_xlim(-lim, lim)
    # the extra room at the bottom is the "PyTorch f32" caption's; without it the caption sits on
    # top of the last row's value label
    ax.set_ylim(rows[-1][0] + 1.9, heads[0][0] - 0.7)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=7.4)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
    for yy, title in heads:
        ax.text(-lim * 0.98, yy, title, fontsize=7.8, color=MUTED, va="center", ha="left",
                bbox=dict(boxstyle="square,pad=0.15", facecolor=PAGE, edgecolor="none", alpha=0.8))
    ax.text(lim * 0.03, rows[-1][0] + 1.75, "PyTorch f32", fontsize=7.4, color=C_TORCH_CPU,
            va="bottom", ha="left")
    ax.set_xlabel("jepa.cpp top-1 minus PyTorch's, percentage points", fontsize=8, color=MUTED,
                  labelpad=4)
    ax.set_title("Task accuracy: the same pixels through both engines, nothing refitted",
                 fontsize=10.5, color=INK, loc="left", pad=8)
    ax.tick_params(axis="x", labelsize=7.5)
    style_axes(ax, xgrid=True)


# ---- panel 3: quantization ------------------------------------------------------------------------

def panel_quant(axes, groups: dict, gpu: dict) -> None:
    """One small axes per model measured over the full dtype sweep on both backends.

    Latency is drawn against the same backend's own f16 file, because that ratio is what the dtype
    choice costs or buys — the absolute milliseconds are three panels' worth of range apart and the
    shape of both curves would flatten into two straight lines on a shared log axis.  Each axes
    names its two f16 anchors in milliseconds, so the ratios convert back.
    """
    eligible = []
    for g in sorted(groups.values(), key=lambda g: (g["tokens"], g["model"])):
        cuda = {d: v[0] for d, v in gpu.get((g["model"], g["tokens"]), {}).get("cuda", {}).items()}
        cpu = {d: t[32] for d, t in g["cpu"].items() if 32 in t and d in g["weights"]}
        if len(cuda) >= 3 and len(cpu) >= 4 and "f16" in cuda and "f16" in cpu:
            eligible.append((g, cpu, cuda))
    if not eligible:
        axes[0].text(0.5, 0.5, "no model measured over the full dtype sweep on both backends",
                     ha="center", va="center", transform=axes[0].transAxes, color=MUTED, fontsize=8)
        for ax in axes:
            ax.set_axis_off()
        return
    for g, _cpu, _cuda in eligible[len(axes):]:
        warn(f"quantization panel has room for {len(axes)} models; {g['label']} {g['shape']} omitted")

    for i, (ax, (g, cpu, cuda)) in enumerate(zip(axes, eligible)):
        # the x axis is the file's weights, so the dtypes at one size share a tick; only a size that
        # holds more than one of them (q4_0 and q4_k do) needs its points named in the plot as well
        at_size: dict = {}
        for d in sorted(set(cpu) | set(cuda)):
            if d in g["weights"]:
                at_size.setdefault(g["weights"][d] / 1024.0, set()).add(d)
        sizes = sorted(at_size)
        ratios = [1.0]
        ax.axhline(1.0, color=GRID, lw=1.0, zorder=1)
        for series, colour, name, above in ((cpu, C_CPP_CPU, "CPU, 32 threads", True),
                                            (cuda, C_CPP_GPU, "CUDA", False)):
            base = series["f16"]
            pts = sorted((g["weights"][d] / 1024.0, series[d] / base, d)
                         for d in series if d in g["weights"])
            ratios += [p[1] for p in pts]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, lw=1.3, marker="o",
                    ms=4.5, zorder=4)
            top = {x: max(p[1] for p in pts if p[0] == x) for x, _y, _d in pts}
            for x, yv, d in pts:
                if len(at_size[x]) < 2:
                    continue
                if sum(1 for p in pts if p[0] == x) < 2:    # one point at an ambiguous size: beside it
                    ax.annotate(d, (x, yv), textcoords="offset points", xytext=(6, 4), fontsize=6.9,
                                color=colour, ha="left", va="bottom", zorder=5)
                    continue
                up = top[x] == yv                          # two of them: one above, one below
                ax.annotate(d, (x, yv), textcoords="offset points",
                            xytext=(0, 5) if up else (-3, -8), fontsize=6.9, color=colour,
                            ha="center" if up else "right", va="bottom" if up else "top", zorder=5)
            ax.annotate(f"{name}, f16 = {fmt_ms(base)} ms", (pts[-1][0], pts[-1][1]),
                        textcoords="offset points", xytext=(6, 0), fontsize=7.4, color=colour,
                        ha="left", va="center", zorder=5)

        lines = []
        if g["torch_cpu"] and g["torch_comparable"]:
            lines.append((f"PyTorch CPU, {fmt_ms(g['torch_cpu'])} ms",
                          g["torch_cpu"] / cpu["f16"], C_TORCH_CPU))
        torch_gpu = gpu.get((g["model"], g["tokens"]), {}).get("torch_gpu", {})
        if "fp16" in torch_gpu:
            lines.append((f"PyTorch CUDA fp16, {torch_gpu['fp16'][1]} ms",
                          torch_gpu["fp16"][0] / cuda["f16"], C_TORCH_GPU))
        for label, yv, colour in lines:
            ratios.append(yv)
            ax.axhline(yv, color=colour, lw=1.0, ls=(0, (5, 3)), zorder=2)
            ax.text(0.995, yv, label, transform=ax.get_yaxis_transform(), fontsize=7.0, color=colour,
                    ha="right", va="bottom", zorder=5)

        lo, hi = min(ratios) - 0.22, max(ratios) + 0.22
        ax.set_xscale("log")
        ax.set_xlim(min(sizes) * 0.80, max(sizes) * 2.55)
        ax.set_ylim(lo, hi)
        ax.set_xticks(sizes)
        ax.set_xticklabels([" · ".join(dtype_order(at_size[s])) + f"\n{s:.2f} GB" for s in sizes],
                           fontsize=7.4, linespacing=1.4)
        ticks = [t for t in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75) if lo <= t <= hi]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}×" for t in ticks], fontsize=7.4)
        ax.minorticks_off()
        ax.text(0.0, 1.03, f"{g['label']} · {g['shape']}", transform=ax.transAxes, fontsize=8.2,
                color=INK, va="bottom", ha="left")
        ax.set_ylabel("latency vs own f16", fontsize=7.6, color=MUTED, labelpad=3)
        style_axes(ax, ygrid=True)
        if i == 0:
            ax.set_title("Quantization: memory on both backends, speed only on CUDA", fontsize=10.5,
                         color=INK, loc="left", pad=22)

    for ax in axes[len(eligible):]:
        ax.set_axis_off()
    axes[min(len(eligible), len(axes)) - 1].set_xlabel(
        "resident weights per dtype (jepa_model_n_bytes) — log scale", fontsize=8, color=MUTED,
        labelpad=4)


# ---- figure ---------------------------------------------------------------------------------------

def style_axes(ax, xgrid=False, ygrid=False) -> None:
    ax.set_facecolor(PAGE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=3, width=0.8)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)


def environment(bench, perf: str) -> str:
    """The box, from the artifacts: CPU and date out of benchmarks.json, GPU out of performance.md."""
    cpu = (bench or {}).get("box", {}).get("cpu", "")
    cpu = re.sub(r"^AMD Ryzen ", "", cpu).replace(" 96-Cores", ", 96 cores")
    date = ((bench or {}).get("sessions") or [{}])[0].get("start_utc", "").split(" ")[0]
    gpu = ""
    if perf:
        for _headers, table in md_tables(PERF_MD):
            for row in table:
                if row and row[0].strip().lower() == "gpu" and len(row) > 1:
                    gpu = re.sub(r"^\d+\s*×\s*", "", row[1].split(",")[0]).replace("NVIDIA ", "")
    # both cards in the box are the same model; the timing rows run on device 0 and the SSv2
    # accuracy sweep on device 1, so the strip names the card rather than an index
    bits = [b for b in (cpu, gpu, "idle box", date) if b]
    return "  ·  ".join(bits)


def load_artifacts() -> dict:
    """Everything both the combined figure and the single-panel ones read, loaded once."""
    bench = load_json(BENCH_JSON)
    perf = PERF_MD.read_text() if PERF_MD.exists() else ""
    if not perf:
        warn(f"cannot read {PERF_MD.relative_to(ROOT)}: the card is dropped from the box strip, "
             "and with it the GPU series unless tests/results/benchmarks-gpu.json is present")

    groups = encoder_groups(bench) if bench else {}
    cpu_f16 = {(g["model"], g["tokens"]): g["cpu"]["f16"][32]
               for g in groups.values() if 32 in g["cpu"].get("f16", {})}
    return {
        "bench": bench,
        "acc_img": load_json(ACC_IMAGE_JSON),
        "acc_vid": load_json(ACC_VIDEO_JSON),
        "acc_ssv2": load_json(ACC_SSV2_JSON) if ACC_SSV2_JSON.exists() else None,
        "perf": perf,
        "groups": groups,
        "gpu": gpu_series(perf, cpu_f16),
    }


def speed_legend(fig, bbox, ncol: int):
    """The key panel 1 needs: who each bar is, and what the two marks on it mean."""
    handles = [
        Patch(facecolor=C_TORCH_CPU),
        Patch(facecolor=C_CPP_CPU),
        (Patch(facecolor=C_CPP_CPU),
         Line2D([], [], color="white", marker="|", ms=7, mew=1.4, ls="none")),
        Patch(facecolor=C_TORCH_GPU),
        Patch(facecolor=C_CPP_GPU),
        Patch(facecolor=C_TORCH_CPU, hatch="///", edgecolor="white"),
    ]
    labels = ["PyTorch CPU, f32, 32 threads", "jepa.cpp CPU, f16, 32 threads",
              "the same run at 96 threads", "PyTorch CUDA, fp16", "jepa.cpp CUDA, f16",
              "the PyTorch forward is not the same work — an upper bound"]
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=bbox, ncol=ncol,
                     frameon=False, fontsize=8.2, handlelength=1.7, handleheight=0.95,
                     columnspacing=1.6, handletextpad=0.6,
                     handler_map={tuple: HandlerTuple(ndivide=None)})
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


def build_figure(data: dict | None = None):
    data = data if data is not None else load_artifacts()
    bench, perf, groups, gpu = data["bench"], data["perf"], data["groups"], data["gpu"]
    acc_img, acc_vid, acc_ssv2 = data["acc_img"], data["acc_vid"], data["acc_ssv2"]

    fig = plt.figure(figsize=(16, 9.6), dpi=100)
    fig.patch.set_facecolor(PAGE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.42, 1.0], width_ratios=[1.06, 1.0],
                          left=0.112, right=0.988, top=0.862, bottom=0.068, hspace=0.34, wspace=0.30)
    ax_speed = fig.add_subplot(gs[0, :])
    ax_acc = fig.add_subplot(gs[1, 0])
    quant_gs = gs[1, 1].subgridspec(2, 1, hspace=0.62)
    ax_quant = [fig.add_subplot(quant_gs[0]), fig.add_subplot(quant_gs[1])]

    panel_speed(ax_speed, groups, gpu)
    panel_accuracy(ax_acc, acc_img, acc_vid, acc_ssv2)
    panel_quant(ax_quant, groups, gpu)

    fig.text(0.008, 0.975, "jepa.cpp — the same features as PyTorch, faster on a CPU and much "
                           "faster on CUDA", fontsize=14.5, color=INK, va="top", ha="left")
    fig.text(0.008, 0.943, environment(bench, perf), fontsize=8.5, color=MUTED, va="top", ha="left")

    speed_legend(fig, (0.006, 0.922), 6)

    fig.text(0.008, 0.012, "generated by scripts/gen_results_figure.py from tests/results/*.json "
                           "and docs/performance.md", fontsize=7.2, color=MUTED, va="bottom")
    return fig


# ---- one panel per file ---------------------------------------------------------------------------

# docs/performance.md and docs/accuracy.md carry a panel each beside the tables it draws, so each one
# is also written on its own.  Same data, same panel code, same look; only the frame around it —
# figure size, legend, provenance line — is per file.
PANELS = ("latency", "accuracy", "quantization")


def panel_paths(out: pathlib.Path) -> dict:
    return {name: out.with_name(f"{out.stem}-{name}{out.suffix}") for name in PANELS}


def build_panel_figure(name: str, data: dict):
    """One panel of the combined figure as a figure of its own."""
    bench, perf, groups, gpu = data["bench"], data["perf"], data["groups"], data["gpu"]
    if name == "latency":
        fig = plt.figure(figsize=(12.0, 6.0), dpi=100)
        gs = fig.add_gridspec(1, 1, left=0.148, right=0.985, top=0.815, bottom=0.105)
        panel_speed(fig.add_subplot(gs[0]), groups, gpu)
        speed_legend(fig, (0.008, 0.995), 3)
    elif name == "accuracy":
        fig = plt.figure(figsize=(8.6, 5.4), dpi=100)
        gs = fig.add_gridspec(1, 1, left=0.295, right=0.985, top=0.905, bottom=0.115)
        panel_accuracy(fig.add_subplot(gs[0]), data["acc_img"], data["acc_vid"],
                       data["acc_ssv2"])
    elif name == "quantization":
        fig = plt.figure(figsize=(8.6, 6.2), dpi=100)
        gs = fig.add_gridspec(2, 1, left=0.125, right=0.985, top=0.885, bottom=0.15, hspace=0.62)
        panel_quant([fig.add_subplot(gs[0]), fig.add_subplot(gs[1])], groups, gpu)
    else:
        raise ValueError(f"unknown panel {name!r}")
    fig.patch.set_facecolor(PAGE)
    fig.text(0.008, 0.012, f"{environment(bench, perf)}  ·  generated by "
             "scripts/gen_results_figure.py", fontsize=7.2, color=MUTED, va="bottom")
    return fig


def save(fig, path: pathlib.Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = ({"Date": None, "Creator": "scripts/gen_results_figure.py"} if fmt == "svg"
                else {"Software": "scripts/gen_results_figure.py"})
    fig.savefig(path, format=fmt, facecolor=PAGE, metadata=metadata)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default=str(OUT_SVG),
                    help="SVG to write (default docs/assets/results.svg)")
    ap.add_argument("--png", help="also write a PNG of the combined figure here, to look at")
    ap.add_argument("--split", action="store_true",
                    help="also write each panel on its own, as results-latency.svg, "
                         "results-accuracy.svg and results-quantization.svg next to --out")
    ap.add_argument("--check", action="store_true",
                    help="regenerate all four files into a temporary directory and exit 1 if any "
                         "of them differs from what is committed")
    a = ap.parse_args()

    # Deterministic output: no timestamp, a fixed salt for the generated element ids, and glyphs as
    # paths, so the file carries no font dependency.
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "svg.hashsalt": "jepa.cpp-results",
        "svg.fonttype": "path",
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.unicode_minus": False,
        "path.simplify": True,
    })

    # All four are built on every run, in this order, whatever is then written: --check and a plain
    # run must hand savefig the same figures in the same sequence for the bytes to compare equal.
    data = load_artifacts()
    out = pathlib.Path(a.out)
    paths = panel_paths(out)
    figures = [(out, build_figure(data))] + \
              [(paths[name], build_panel_figure(name, data)) for name in PANELS]
    if a.png:
        png = pathlib.Path(a.png)
        save(figures[0][1], png, "png")
        print(f"wrote {png}")

    if a.check:
        stale = []
        with tempfile.TemporaryDirectory() as td:
            for path, fig in figures:
                tmp = pathlib.Path(td) / path.name
                save(fig, tmp, "svg")
                new, old = tmp.read_bytes(), (path.read_bytes() if path.exists() else b"")
                if new != old:
                    stale.append(f"{path} ({len(old)} bytes on disk, {len(new)} regenerated)")
                else:
                    print(f"{path} is up to date ({len(new)} bytes)")
        if stale:
            print("stale: " + ", ".join(stale) + " — run scripts/gen_results_figure.py --split",
                  file=sys.stderr)
            return 1
        return 0

    for path, fig in figures if a.split else figures[:1]:
        save(fig, path, "svg")
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
