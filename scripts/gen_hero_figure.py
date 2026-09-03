#!/usr/bin/env python3
"""Generate docs/assets/hero.svg — the one figure README.md leads with.

Three panels, one claim each, every value read from a committed artifact:

    latency on a CPU    jepa.cpp f16 against PyTorch f32, both at 32 threads, per image or clip
                        (tests/results/benchmarks.json; only the rows whose PyTorch baseline
                        measures the same work, `pytorch_comparable`)
    latency on a GPU    jepa.cpp CUDA f16 against PyTorch fp16 and fp32 on one card
                        (tests/results/benchmarks-gpu.json, rows + pytorch_gpu)
    accuracy            jepa.cpp f16 against PyTorch f32 on real tasks: Imagenette k-NN,
                        UCF-101 k-NN and the full Something-Something-v2 validation split
                        (tests/results/accuracy-{image,video,ssv2}.json)

    python scripts/gen_hero_figure.py                                # writes docs/assets/hero.svg
    python scripts/gen_hero_figure.py --png tmp/h.png --width-px 880 # rasterise at README width
    python scripts/gen_hero_figure.py --check                        # exit 1 if the SVG is stale

matplotlib is the only dependency; CI installs it pinned for the --check gate (see ci.yml).  Output
is byte-identical across runs — no timestamp, a fixed hash salt for the element ids, glyphs as
paths — but another matplotlib may re-emit those paths and make --check report drift.

A missing artifact drops the panel it feeds, with a warning on stderr, rather than failing the run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_results_figure as R  # noqa: E402  palette and the warn() helper live there

ROOT = R.ROOT
RESULTS = ROOT / "tests" / "results"
OUT_SVG = ROOT / "docs" / "assets" / "hero.svg"

W, H = 1200, 800            # canvas in px; GitHub shows it at 880 px, so everything is drawn large
CPU_THREADS = 32

# The workloads the CPU panel walks, left to right.  Display names only: the numbers come from the
# artifact, and a workload whose rows are missing is dropped from the axis.
CPU_WORK = [
    ("lewm-pusht",                  1,  "LeWM\nimage"),
    ("lejepa-vits16-pretrain-in1k", 1,  "LeJEPA S\nimage"),
    ("vjepa2_1-vitb-384",           1,  "V-JEPA 2.1 B\nimage"),
    ("ijepa_vith14_1k",             1,  "I-JEPA H\nimage"),
    ("vjepa2_1-vitb-384",           16, "V-JEPA 2.1 B\n16 frames"),
    ("levjepa-vitl16",              16, "LeVJEPA L\n16 frames"),
]
GPU_WORK = [
    ("ijepa_vith14_1k",      1,  "I-JEPA H\nimage"),
    ("vjepa2-vitl-fpc64-256", 16, "V-JEPA 2 L\n16 frames"),
    ("levjepa-vitl16",       16, "LeVJEPA L\n16 frames"),
    ("vjepa2-vitl-fpc64-256", 64, "V-JEPA 2 L\n64 frames"),
]


def load(name: str):
    path = RESULTS / name
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        R.warn(f"{path.relative_to(ROOT)}: {e} — its panel is dropped")
        return None


# --- CPU ------------------------------------------------------------------------------------------
def cpu_series(bench: dict) -> list[tuple[str, float, float]]:
    """(label, jepa.cpp f16 ms, PyTorch f32 ms) per workload, both at CPU_THREADS threads."""
    out = []
    for model, frames, label in CPU_WORK:
        rows = [r for r in bench["rows"]
                if r["model"] == model and r["frames"] == frames and r["ftype"] == "f16"
                and r["threads"] == CPU_THREADS and r["batch"] == 1 and r["mode"] == "encoder"]
        if not rows:
            R.warn(f"benchmarks.json: no f16 {CPU_THREADS}-thread row for {model} {frames}f")
            continue
        r = min(rows, key=lambda x: x["ms_mean"])
        if not r.get("pytorch_comparable") or not r.get("pytorch_ms"):
            R.warn(f"benchmarks.json: {model} {frames}f has no comparable PyTorch baseline")
            continue
        out.append((label, r["ms_mean"], r["pytorch_ms"]))
    return out


# --- GPU ------------------------------------------------------------------------------------------
def gpu_series(bench_gpu: dict) -> list[tuple[str, float, float | None, float | None]]:
    """(label, jepa.cpp CUDA f16 ms, PyTorch fp16 ms, PyTorch fp32 ms) per workload, batch 1."""
    torch = {}
    for r in bench_gpu.get("pytorch_gpu", {}).get("rows", []):
        torch[(r["model"], r["frames"], r["precision"])] = r["ms_mean"]
    out = []
    for model, frames, label in GPU_WORK:
        rows = [r for r in bench_gpu["rows"]
                if r["model"] == model and r["frames"] == frames and r["ftype"] == "f16"
                and r["batch"] == 1 and r["mode"] == "encoder" and r.get("gpu")]
        if not rows:
            R.warn(f"benchmarks-gpu.json: no f16 row for {model} {frames}f")
            continue
        # The row at the family's shipped accumulation precision — f16 for hfvit (LeJEPA) and
        # levjepa, f32 for the rest (jepa_gpu_prec_f32_default in src/jepa.cpp) — so the panel
        # shows what a caller gets with no flag; the best session wins where several were measured.
        prec = "f16" if rows[0]["family"] in ("hfvit", "levjepa") else "f32"
        at_default = [r for r in rows if r.get("gpu_prec") == prec] or rows
        r = min(at_default, key=lambda x: x["ms_mean"])
        out.append((label, r["ms_mean"], torch.get((model, frames, "fp16")),
                    torch.get((model, frames, "fp32"))))
    return out


# --- accuracy -------------------------------------------------------------------------------------
def accuracy_series(acc_img, acc_vid, acc_ssv2) -> list[tuple[str, str, float, float]]:
    """(model, task, PyTorch %, jepa.cpp f16 %) — one row per real-task check that has both sides."""
    out = []
    if acc_img:
        best = {}   # (model) -> best PyTorch row by feature, and its jepa.cpp f16 twin
        for r in acc_img["rows"]:
            key = (r["model"], r["feature"], r["gallery"])
            best.setdefault(key, {})[(r["backend"], r["dtype"])] = r["knn_top1"]
        per_model = {}   # the protocol with the strongest PyTorch baseline speaks for the model
        for (model, feature, gallery), by in sorted(best.items()):
            pt, cpp = by.get(("pytorch", "f32")), by.get(("jepa.cpp", "f16"))
            if pt is None or cpp is None or pt < R.MIN_BASELINE_TOP1:
                continue
            if model not in per_model or pt > per_model[model][0]:
                per_model[model] = (pt, cpp)
        for model, (pt, cpp) in per_model.items():
            out.append((R.pretty(model), "Imagenette k-NN", 100 * pt, 100 * cpp))
    if acc_vid:
        for model, m in acc_vid["models"].items():
            b = m["backends"]
            if "torch" not in b or "f16" not in b:
                continue
            split = "val+test"
            out.append((R.pretty(model), "UCF-101 k-NN",
                        100 * b["torch"]["splits"][split]["knn_top1"],
                        100 * b["f16"]["splits"][split]["knn_top1"]))
    if acc_ssv2:
        runs = {r["tag"]: r for r in acc_ssv2["runs"]}
        pt = next((r for t, r in runs.items() if t.startswith("torch") and r["scope"] == "full"), None)
        cpp = next((r for t, r in runs.items()
                    if t.startswith("cpp") and r["dtype"] == "f16" and r["scope"] == "full"), None)
        if pt and cpp:
            out.append(("V-JEPA 2 ViT-L SSv2", f"SSv2 top-1",
                        100 * pt["top1"], 100 * cpp["top1"]))
    return out


# --- drawing --------------------------------------------------------------------------------------
def fmt_ms(v, _pos=None) -> str:
    if v >= 1000:
        return f"{v / 1000:g} s"
    return f"{v:g} ms"


def style(ax, title: str) -> None:
    ax.set_facecolor(R.PAGE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(R.GRID)
    ax.tick_params(colors=R.MUTED, labelsize=12, length=0)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=R.INK, pad=14)


def latency_panel(ax, title: str, labels, series) -> None:
    """series: list of (name, colour, linestyle, values-or-None)"""
    x = list(range(len(labels)))
    ax.set_yscale("log")
    for name, colour, ls, vals in series:
        pts = [(i, v) for i, v in zip(x, vals) if v is not None]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=colour, lw=2.6, marker="o",
                ms=8, mec=R.PAGE, mew=1.5, label=name, zorder=3, solid_capstyle="round")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, color=R.INK)
    ax.set_xlim(-0.45, len(labels) - 0.55)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_ms))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color=R.GRID, lw=0.8, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=12.5, handlelength=2.2, borderaxespad=0.2,
              ncol=2, columnspacing=1.4, handletextpad=0.5)
    style(ax, title)


def cpu_panel(ax, rows) -> None:
    labels = [r[0] for r in rows]
    cpp = [r[1] for r in rows]
    pt = [r[2] for r in rows]
    latency_panel(ax, f"On a CPU ({CPU_THREADS} threads), lower is better", labels,
                  [("PyTorch f32", R.C_TORCH_CPU, "--", pt),
                   ("jepa.cpp f16", R.C_CPP_CPU, "-", cpp)])
    lo = min(min(cpp), min(pt)); hi = max(max(cpp), max(pt))
    ax.set_ylim(lo / 3.2, hi * 3.6)
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000]))
    for i, (c, p) in enumerate(zip(cpp, pt)):
        ax.annotate(f"{p / c:.1f}×", (i, c), xytext=(0, -20), textcoords="offset points",
                    ha="center", fontsize=12, color=R.C_CPP_CPU, fontweight="bold")


def gpu_panel(ax, rows) -> None:
    labels = [r[0] for r in rows]
    cpp = [r[1] for r in rows]
    p16 = [r[2] for r in rows]
    p32 = [r[3] for r in rows]
    latency_panel(ax, "On one GPU (RTX 4500 Ada), batch 1", labels,
                  [("PyTorch fp32", R.C_TORCH_CPU, "--", p32),
                   ("PyTorch fp16", R.C_TORCH_GPU, ":", p16),
                   ("jepa.cpp CUDA f16", R.C_CPP_GPU, "-", cpp)])
    vals = [v for v in cpp + p16 + p32 if v is not None]
    ax.set_ylim(min(vals) / 1.8, max(vals) * 7.0)
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000]))


def accuracy_panel(ax, rows) -> None:
    n = len(rows)
    ys = list(range(n))[::-1]
    ax.set_facecolor(R.PAGE)
    for y, (model, task, pt, cpp) in zip(ys, rows):
        ax.plot([pt, cpp], [y, y], color=R.GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.plot([pt], [y], "o", color=R.C_TORCH_CPU, ms=11, mec=R.PAGE, mew=1.5, zorder=3)
        ax.plot([cpp], [y], "o", color=R.C_CPP_CPU, ms=11, mec=R.PAGE, mew=1.5, zorder=4)
        ax.text(1.01, y, f"{pt:.1f} %  →  {cpp:.1f} %", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=12.5, color=R.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{m}  ·  {t}" for m, t, _, _ in rows], fontsize=12, color=R.INK)
    for lab in ax.get_yticklabels():
        lab.set_ha("right")
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100 %"])
    ax.grid(axis="x", color=R.GRID, lw=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    style(ax, "Accuracy on real tasks, PyTorch f32 → jepa.cpp f16")



def build_figure():
    bench, bench_gpu = load("benchmarks.json"), load("benchmarks-gpu.json")
    acc_img, acc_vid, acc_ssv2 = load("accuracy-image.json"), load("accuracy-video.json"), load("accuracy-ssv2.json")
    cpu = cpu_series(bench) if bench else []
    gpu = gpu_series(bench_gpu) if bench_gpu else []
    acc = accuracy_series(acc_img, acc_vid, acc_ssv2)
    if not (cpu or gpu or acc):
        sys.exit("no artifact could be read; nothing to draw")

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(R.PAGE)
    top = [k for k, v in (("cpu", cpu), ("gpu", gpu)) if v]
    rows = (1 if top else 0) + (1 if acc else 0)
    gs = fig.add_gridspec(rows, 1, left=0.06, right=0.97, top=0.93, bottom=0.12,
                          hspace=0.55, height_ratios=([1.0, 0.95] if rows == 2 else [1.0]))
    if top:
        sub = gs[0].subgridspec(1, len(top), wspace=0.28, width_ratios=[1.35, 1.0][:len(top)])
        for i, kind in enumerate(top):
            ax = fig.add_subplot(sub[0, i])
            (cpu_panel if kind == "cpu" else gpu_panel)(ax, cpu if kind == "cpu" else gpu)
    if acc:
        sub = gs[rows - 1].subgridspec(1, 3, width_ratios=[0.42, 1.0, 0.30], wspace=0.0)
        ax = fig.add_subplot(sub[0, 1])
        accuracy_panel(ax, acc)
    fig.text(0.06, 0.02,
             "Every number is read from tests/results/*.json: one image or one clip per call on a Threadripper 7995WX "
             "and one RTX 4500 Ada;\nfrozen features + nearest neighbours for the k-NN rows, the shipped classifier "
             "head for SSv2 (24,777 validation clips).", fontsize=11, color=R.MUTED, va="bottom", linespacing=1.5)
    return fig


def save(fig, path: pathlib.Path, fmt: str, dpi=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = ({"Date": None, "Creator": "scripts/gen_hero_figure.py"} if fmt == "svg"
                else {"Software": "scripts/gen_hero_figure.py"})
    fig.savefig(path, format=fmt, facecolor=R.PAGE, metadata=metadata, dpi=dpi or fig.dpi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default=str(OUT_SVG), help="SVG to write (default docs/assets/hero.svg)")
    ap.add_argument("--png", help="also write a PNG here, to look at; never committed")
    ap.add_argument("--width-px", type=float, default=W,
                    help="width of that PNG in pixels (880 is the GitHub README column)")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temporary file and exit 1 if --out differs")
    a = ap.parse_args()

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
            print(f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} is stale: "
                  f"run scripts/gen_hero_figure.py", file=sys.stderr)
            return 1
        print(f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} is up to date")
        return 0

    save(fig, out, "svg")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
