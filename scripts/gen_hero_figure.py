#!/usr/bin/env python3
"""Generate docs/assets/hero.svg — the one-screen figure README.md leads with.

Four KPI cards and one chart, sized to be read at the 880 px of a GitHub README column without
enlarging.  The detail — every model, every dtype, every backend — is the three-panel figure
`scripts/gen_results_figure.py` draws; this one carries the four claims that decide whether the
detail is worth reading.  Every value comes from an artifact through the loaders of that script,
which this one imports; the only literals here are layout and wording.

    tests/results/benchmarks.json      CPU encoder latency per model / shape / dtype / thread count,
                                       resident weights, and the PyTorch CPU baseline of each row
    tests/results/accuracy-image.json  Imagenette k-NN top-1, PyTorch against jepa.cpp per dtype
    tests/results/accuracy-ssv2.json   SSv2 validation top-1 of the classifier checkpoint against
                                       PyTorch, when the licence-gated dataset was on the machine —
                                       absent, the figure drops that card and keeps the other three
    tests/results/benchmarks-gpu.json  CUDA encoder latency per model / shape / dtype, from
                                       scripts/bench_gpu.sh; docs/performance.md is the fallback
                                       when it is absent (gen_results_figure.gpu_series())
    docs/performance.md                the card named in the source line, and that GPU fallback

    python scripts/gen_hero_figure.py                                # writes docs/assets/hero.svg
    python scripts/gen_hero_figure.py --png tmp/h.png --width-px 880 # rasterise at README width
    python scripts/gen_hero_figure.py --check                        # exit 1 if the SVG is stale

matplotlib is the only dependency and stays out of docs/requirements.txt, as it does for the results
figure; CI installs it pinned for the --check gate:

    uv pip install --python .venv/bin/python matplotlib

The committed SVG was written with matplotlib 3.11.1.  Output is byte-identical across runs — no
timestamp, a fixed hash salt for the generated element ids, glyphs as paths — but another matplotlib
may re-emit those paths and make --check report drift.

A missing or unreadable artifact drops what it feeds, with a warning on stderr, rather than failing
the run.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_results_figure as R  # noqa: E402  the artifact loaders and the palette live there

ROOT = R.ROOT
ACC_SSV2_JSON = ROOT / "tests" / "results" / "accuracy-ssv2.json"
OUT_SVG = ROOT / "docs" / "assets" / "hero.svg"

# One colour per claim, held between the cards and the chart: dark = fidelity, blue = the CPU
# engine, green = the CUDA engine, vermillion = PyTorch, orange = what a file costs.  Okabe-Ito.
C_MEMORY = "#E69F00"
CARD_BG = "#ffffff"
CARD_EDGE = "#e6e6e6"

# Layout, in pixels of the 1200 × 600 canvas; at the README's 880 px every one of them shrinks by
# 880/1200 and a point of font size lands at 1.02 px, so 13 pt is the floor for readable text.
W, H = 1200.0, 600.0
MARGIN = 26.0
TITLE_Y, SUBTITLE_Y = H - 22.0, H - 62.0
CARD_TOP, CARD_H, CARD_GAP = H - 92.0, 216.0, 16.0
CARD_PAD = 18.0
RULE_W, RULE_H = 46.0, 5.0
CHART_TITLE_Y = 264.0
BAR_X0, BAR_MAX_W, BAR_H = 326.0, 712.0, 34.0
BAR_Y = (200.0, 142.0, 84.0)
SOURCE_Y = 30.0

FS_TITLE, FS_SUBTITLE = 24.0, 13.5
FS_EYEBROW, FS_BIG, FS_LINE = 13.0, 42.0, 13.0
FS_CHART_TITLE, FS_BAR_LABEL, FS_BAR_VALUE = 15.0, 13.5, 16.0
FS_SOURCE = 13.0


def warn(msg: str) -> None:
    R.warn(msg)


# ---- the numbers ----------------------------------------------------------------------------------

def n_models(groups: dict) -> int:
    return len({g["model"] for g in groups.values()})


def cpu_speedups(groups: dict) -> dict:
    """model key -> jepa.cpp CPU f16 speed-up over PyTorch, on the rows that run the same work."""
    return {k: g["torch_speedup"] for k, g in sorted(groups.items())
            if g.get("torch_speedup") and g.get("torch_comparable")}


def gpu_speedups(groups: dict, gpu: dict) -> dict:
    """model key -> CUDA f16 speed-up over the same file on 32 CPU threads."""
    out = {}
    for key, g in sorted(groups.items()):
        cuda = gpu.get((g["model"], g["tokens"]), {}).get("cuda", {})
        cpu32 = g["cpu"].get("f16", {}).get(32)
        if cpu32 and "f16" in cuda:
            out[key] = cpu32 / cuda["f16"][0]
    return out


def q8_weight_ratio(groups: dict):
    """The smallest q8_0/f16 ratio of resident weights over the models that carry both."""
    ratios = [g["weights"]["q8_0"] / g["weights"]["f16"] for g in sorted(groups.values(),
              key=lambda g: g["model"]) if "q8_0" in g["weights"] and "f16" in g["weights"]]
    return min(ratios) if ratios else None


def imagenette_bound(acc: dict, dtypes=("f32", "f16", "q8_0")):
    """Largest |jepa.cpp − PyTorch| Imagenette k-NN gap in percentage points, over the dtypes the
    documentation recommends shipping.  Every feature and every gallery of the artifact counts, not
    only the one the results figure plots, so the bound is the one accuracy.md quotes."""
    by_run: dict = {}
    for row in acc.get("rows", []):
        by_run.setdefault((row["model"], row["feature"], row["gallery"]), []).append(row)
    worst = None
    for _key, rows in sorted(by_run.items()):
        base = next((r["knn_top1"] for r in rows if r["backend"] == "pytorch"), None)
        if base is None or base < R.MIN_BASELINE_TOP1:   # no baseline, or an off-task sanity row
            continue
        for r in rows:
            if r["backend"] == "pytorch" or r["dtype"] not in dtypes:
                continue
            d = abs(r["knn_top1"] - base) * 100.0
            worst = d if worst is None or d > worst else worst
    return worst


def ssv2_headline(acc: dict):
    """(dtype, top-1 %, PyTorch top-1 %, clips) of the full validation split.

    f16 is the tier the documentation ships for head work, so it is the one the card names; any
    other measured dtype stands in for it if that run is not in the artifact.
    """
    runs = acc.get("runs", [])
    ref = next((r for r in runs
                if r.get("backend") == "pytorch" and r.get("scope") == "full"), None)
    ours = [r for r in runs if r.get("backend") != "pytorch" and r.get("scope") == "full"
            and r.get("source") == "measured" and r.get("dtype")]
    if ref is None or not ours:
        warn("accuracy-ssv2.json: no full-split PyTorch row or no jepa.cpp run — card dropped")
        return None
    order = R.dtype_order({r["dtype"] for r in ours})
    pick = min(ours, key=lambda r: (r["dtype"] != "f16", order.index(r["dtype"])))
    return pick["dtype"], pick["top1"] * 100.0, ref["top1"] * 100.0, int(ref["n_clips"])


def showcase(groups: dict, gpu: dict):
    """The workload the chart draws: the largest CPU win over PyTorch among the rows that also have
    a CUDA number, i.e. the one place where all three engines ran the same graph on the same shape."""
    best = None
    for _key, g in sorted(groups.items()):
        cuda = gpu.get((g["model"], g["tokens"]), {}).get("cuda", {})
        cpu32 = g["cpu"].get("f16", {}).get(32)
        if not (cpu32 and g.get("torch_cpu") and g.get("torch_comparable") and "f16" in cuda):
            continue
        if best is None or g["torch_cpu"] / cpu32 > best["torch_cpu"] / best["cpu32"]:
            best = {"g": g, "cpu32": cpu32, "torch_cpu": g["torch_cpu"], "cuda": cuda["f16"][0]}
    if best is None:
        warn("no encoder row has a PyTorch CPU baseline and a CUDA f16 time — chart dropped")
    return best


def source_line(bench, perf: str) -> str:
    """'Measured on a 96-core Threadripper …' — the box out of the artifacts, then the provenance."""
    box = (bench or {}).get("box", {})
    # the vendor's sub-brands ("AMD Ryzen", "PRO", "Generation") are dropped so that the line fits
    # the width once, the way README.md names the same two parts
    cpu = re.sub(r"^AMD Ryzen ", "", box.get("cpu", "")).replace("Threadripper PRO", "Threadripper")
    cores = re.search(r"(\d+)-Cores", cpu)
    cpu = re.sub(r"\s*\d+-Cores$", "", cpu)
    if cores:
        cpu = f"{cores.group(1)}-core {cpu}"
    gpu = ""
    if perf:
        for _headers, table in R.md_tables(R.PERF_MD):
            for row in table:
                if row and row[0].strip().lower() == "gpu" and len(row) > 1:
                    gpu = re.sub(r"^\d+\s*×\s*", "", row[1].split(",")[0]).replace("NVIDIA ", "")
    gpu = gpu.replace(" Generation", "")
    box_bits = " + ".join(b for b in (cpu, gpu) if b)
    return (f"Measured on a {box_bits}" if box_bits else "Measured") + \
        " · numbers from tests/results/*.json and docs/performance.md"


# ---- the cards ------------------------------------------------------------------------------------

def fmt_x(v: float) -> str:
    return f"{v:.0f}×" if v >= 10 else f"{v:.1f}×"


def fmt_n(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def cards(groups: dict, gpu: dict, acc_img, acc_ssv2) -> list:
    """The KPI cards, in reading order; a claim whose artifact is missing is left out."""
    out = []

    ssv2 = ssv2_headline(acc_ssv2) if acc_ssv2 else None
    if ssv2:
        dtype, top1, torch_top1, clips = ssv2
        out.append((R.INK, "SAME ANSWERS", f"{top1:.1f} %",
                    [f"SSv2 val top-1 at {dtype}",
                     f"over all {fmt_n(clips)} clips.",
                     f"PyTorch f32: {torch_top1:.1f} %"]))

    cpu = cpu_speedups(groups)
    if cpu:
        lo, hi = min(cpu.values()), max(cpu.values())
        out.append((R.C_CPP_CPU, "FASTER ON A CPU", fmt_x(hi),
                    ["than PyTorch at 32 CPU",
                     f"threads — {fmt_x(lo)[:-1]}–{fmt_x(hi)}",
                     "across the models"]))

    cuda = gpu_speedups(groups, gpu)
    if cuda:
        lo, hi = min(cuda.values()), max(cuda.values())
        out.append((R.C_CPP_GPU, "FASTER ON ONE GPU", fmt_x(hi),
                    ["than the same file on",
                     "32 CPU threads —",
                     f"{fmt_x(lo)[:-1]}–{fmt_x(hi)} across them"]))

    ratio = q8_weight_ratio(groups)
    bound = imagenette_bound(acc_img) if acc_img else None
    if ratio and bound:
        out.append((C_MEMORY, "SMALLER", f"{ratio:.2f}×",
                    ["the f16 weights at q8_0,",
                     "and Imagenette k-NN",
                     f"within {bound:.2f} pp of PyTorch"]))
    elif ratio:
        out.append((C_MEMORY, "SMALLER", f"{ratio:.2f}×",
                    ["the f16 weights at q8_0", "", ""]))
    return out


def draw_card(ax, fig, x: float, w: float, accent: str, eyebrow: str, big: str, lines) -> None:
    top = CARD_TOP
    ax.add_patch(FancyBboxPatch((x, top - CARD_H), w, CARD_H,
                                boxstyle="round,pad=0,rounding_size=10", linewidth=1.0,
                                facecolor=CARD_BG, edgecolor=CARD_EDGE, zorder=2))
    ax.add_patch(Rectangle((x + CARD_PAD, top - 26.0), RULE_W, RULE_H, facecolor=accent,
                           edgecolor="none", zorder=3))
    text = w - 2 * CARD_PAD
    fit(fig, eyebrow, FS_EYEBROW, text, "bold", "card eyebrow")
    ax.text(x + CARD_PAD, top - 44.0, eyebrow, fontsize=FS_EYEBROW, fontweight="bold", color=accent,
            va="top", ha="left", zorder=4)
    fit(fig, big, FS_BIG, text, "bold", "card number")
    ax.text(x + CARD_PAD, top - 74.0, big, fontsize=FS_BIG, fontweight="bold", color=R.INK,
            va="top", ha="left", zorder=4)
    for i, line in enumerate(lines):
        fit(fig, line, FS_LINE, text, "normal", "card line")
        ax.text(x + CARD_PAD, top - 146.0 - i * 25.0, line, fontsize=FS_LINE, color=R.MUTED,
                va="top", ha="left", zorder=4)


# ---- the chart ------------------------------------------------------------------------------------

def draw_chart(ax, fig, best: dict) -> None:
    g = best["g"]
    # 'shape' is '16 f 256²  2 048 tok' — the chart names the pixels, not the token count
    shape = re.sub(r"[\s,]+[\d\s]+tok$", "", g["shape"])
    noun = "clip" if " f " in shape else "image"
    title = f"One {shape} {noun} through {g['label']} — milliseconds for one encoder pass"
    fit(fig, title, FS_CHART_TITLE, W - 2 * MARGIN, "normal", "chart title")
    ax.text(MARGIN, CHART_TITLE_Y, title, fontsize=FS_CHART_TITLE, color=R.INK, va="top", ha="left",
            zorder=4)

    bars = [("PyTorch f32, 32 threads", best["torch_cpu"], R.C_TORCH_CPU),
            ("jepa.cpp f16, 32 threads", best["cpu32"], R.C_CPP_CPU),
            ("jepa.cpp f16, one GPU", best["cuda"], R.C_CPP_GPU)]
    scale = BAR_MAX_W / max(v for _l, v, _c in bars)
    ax.plot([BAR_X0, BAR_X0], [BAR_Y[-1] - BAR_H, BAR_Y[0] + BAR_H], color=R.GRID, lw=1.0, zorder=2)
    for (label, value, colour), y in zip(bars, BAR_Y):
        ax.add_patch(FancyBboxPatch((BAR_X0, y - BAR_H / 2), value * scale, BAR_H,
                                    boxstyle="round,pad=0,rounding_size=3", linewidth=0,
                                    facecolor=colour, zorder=3))
        fit(fig, label, FS_BAR_LABEL, BAR_X0 - MARGIN - 14.0, "normal", "bar label")
        ax.text(BAR_X0 - 14.0, y, label, fontsize=FS_BAR_LABEL, color=R.INK, va="center", ha="right",
                zorder=4)
        ax.text(BAR_X0 + value * scale + 12.0, y, f"{R.fmt_ms(value)} ms", fontsize=FS_BAR_VALUE,
                fontweight="bold", color=colour, va="center", ha="left", zorder=4)


# ---- the figure -----------------------------------------------------------------------------------

def fit(fig, text: str, fontsize: float, room: float, weight: str, what: str) -> None:
    """Warn when a string is wider than the box drawn for it — the layout is hand-placed, so a
    reworded line that no longer fits has to say so rather than run over its neighbour."""
    if not text:
        return
    renderer = fig.canvas.get_renderer()
    props = FontProperties(family="DejaVu Sans", size=fontsize, weight=weight)
    width = renderer.get_text_width_height_descent(text, props, ismath=False)[0]
    if width > room:
        warn(f"{what} {text!r} is {width:.0f} px wide, {room:.0f} px of room")


def build_figure():
    bench = R.load_json(R.BENCH_JSON)
    acc_img = R.load_json(R.ACC_IMAGE_JSON)
    acc_ssv2 = R.load_json(ACC_SSV2_JSON) if ACC_SSV2_JSON.exists() else None
    if acc_ssv2 is None and not ACC_SSV2_JSON.exists():
        warn(f"{ACC_SSV2_JSON.relative_to(ROOT)} is absent — the accuracy card is dropped")
    perf = R.PERF_MD.read_text() if R.PERF_MD.exists() else ""
    if not perf:
        warn(f"cannot read {R.PERF_MD.relative_to(ROOT)}: the card is dropped from the source line, "
             "and with it the GPU numbers unless tests/results/benchmarks-gpu.json is present")

    groups = R.encoder_groups(bench) if bench else {}
    cpu_f16 = {(g["model"], g["tokens"]): g["cpu"]["f16"][32]
               for g in groups.values() if 32 in g["cpu"].get("f16", {})}
    gpu = R.gpu_series(perf, cpu_f16)

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_facecolor(R.PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_facecolor(R.PAGE)
    ax.set_axis_off()

    title = "jepa.cpp — run Meta's JEPA vision models without PyTorch"
    n = n_models(groups)
    subtitle = (f"{n} model bundles, one GGUF file each · CPU and optional CUDA · one C header · "
                "no Python at run time") if n else "one GGUF file per model · CPU and optional CUDA"
    fit(fig, title, FS_TITLE, W - 2 * MARGIN, "bold", "title")
    fit(fig, subtitle, FS_SUBTITLE, W - 2 * MARGIN, "normal", "subtitle")
    ax.text(MARGIN, TITLE_Y, title, fontsize=FS_TITLE, fontweight="bold", color=R.INK, va="top",
            ha="left")
    ax.text(MARGIN, SUBTITLE_Y, subtitle, fontsize=FS_SUBTITLE, color=R.MUTED, va="top", ha="left")

    kpis = cards(groups, gpu, acc_img, acc_ssv2)
    if kpis:
        room = W - 2 * MARGIN - CARD_GAP * (len(kpis) - 1)
        width = room / len(kpis)
        for i, (accent, eyebrow, big, lines) in enumerate(kpis):
            draw_card(ax, fig, MARGIN + i * (width + CARD_GAP), width, accent, eyebrow, big, lines)

    best = showcase(groups, gpu)
    if best:
        draw_chart(ax, fig, best)

    src = source_line(bench, perf)
    fit(fig, src, FS_SOURCE, W - 2 * MARGIN, "normal", "source line")
    ax.text(MARGIN, SOURCE_Y, src, fontsize=FS_SOURCE, color=R.MUTED, va="bottom", ha="left")
    return fig


def save(fig, path: pathlib.Path, fmt: str, dpi=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = ({"Date": None, "Creator": "scripts/gen_hero_figure.py"} if fmt == "svg"
                else {"Software": "scripts/gen_hero_figure.py"})
    fig.savefig(path, format=fmt, facecolor=R.PAGE, metadata=metadata, dpi=dpi or fig.dpi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default=str(OUT_SVG),
                    help="SVG to write (default docs/assets/hero.svg)")
    ap.add_argument("--png", help="also write a PNG here, to look at; never committed")
    ap.add_argument("--width-px", type=float, default=W,
                    help="width of that PNG in pixels (880 is the GitHub README column)")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temporary file and exit 1 if --out differs")
    a = ap.parse_args()

    # Deterministic output: no timestamp, a fixed salt for the generated element ids, and glyphs as
    # paths, so the file carries no font dependency.
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "svg.hashsalt": "jepa.cpp-hero",
        "svg.fonttype": "path",
        "font.family": "DejaVu Sans",
        "text.color": R.INK,
        "axes.unicode_minus": False,
        "path.simplify": True,
    })

    fig = build_figure()
    out = pathlib.Path(a.out)
    if a.png:
        png = pathlib.Path(a.png)
        save(fig, png, "png", dpi=a.width_px / (W / 100.0))
        print(f"wrote {png} at {a.width_px:.0f} px wide")

    if a.check:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td) / "hero.svg"
            save(fig, tmp, "svg")
            new = tmp.read_bytes()
        old = out.read_bytes() if out.exists() else b""
        if new != old:
            print(f"{out} is stale ({len(old)} bytes on disk, {len(new)} regenerated) — "
                  "run scripts/gen_hero_figure.py", file=sys.stderr)
            return 1
        print(f"{out} is up to date ({len(new)} bytes)")
        return 0

    save(fig, out, "svg")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
