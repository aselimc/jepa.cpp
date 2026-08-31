#!/usr/bin/env python3
"""Render the markdown tables of docs/accuracy-image.md from tests/results/accuracy-image.json.

    scripts/render_accuracy_md.py [tests/results/accuracy-image.json]

Prints one accuracy table per model (rows: backend/dtype x feature x gallery) plus the throughput
table, so no number in the doc is ever retyped by hand.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = {
    "ijepa-vith14-1k": "I-JEPA ViT-H/14 (`ijepa_vith14_1k`) — feature = mean of the 256 patch tokens",
    "lejepa-vits16": "LeJEPA ViT-S/16 (`lejepa-vits16-pretrain-in1k`)",
    "lewm-pusht": "LeWorldModel encoder (`lewm-pusht`) — feature = `emb` = enc.proj(CLS), sanity row",
}
GAL = {"train2000": "2 000 (200/class)", "train_full": "9 469 (full train)"}


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}"


def main():
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/results/accuracy-image.json"
    d = json.loads(p.read_text())
    rows = d["rows"]
    for m in dict.fromkeys(r["model"] for r in rows):
        print(f"### {TITLE.get(m, m)}\n")
        print("| backend | dtype | feature | gallery | kNN top-1 % | centroid top-1 % | "
              "agreement % | mean feat cos | worst feat cos | img/s |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        rs = [r for r in rows if r["model"] == m]
        rs.sort(key=lambda r: (r["feature"], r["gallery"], r["backend"] != "pytorch"))
        for r in rs:
            torchrow = r["backend"] == "pytorch"
            cos = "1" if torchrow else f"{r['feat_cos_mean']:.6f}"
            wcos = "1" if torchrow else f"{r['feat_cos_min']:.6f}"
            agr = "—" if torchrow else pct(r["agreement"])
            ips = "—" if r.get("img_per_s") is None else f"{r['img_per_s']:.1f}"
            print(f"| {r['backend']} | {r['dtype']} | {r['feature']} | {GAL.get(r['gallery'], r['gallery'])} | "
                  f"{pct(r['knn_top1'])} | {pct(r['centroid_top1'])} | {agr} | {cos} | {wcos} | {ips} |")
        print()

    print("### Where the predictions differ (largest gallery of each model)\n")
    print("| model | feature | dtype | items flipped vs PyTorch | PyTorch right | jepa.cpp right | "
          "both wrong | median NN margin of the flipped items | median NN margin, all items |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in dict.fromkeys(r["model"] for r in rows):
        big = max(r["n_gallery"] for r in rows if r["model"] == m)
        for r in rows:
            if r["model"] != m or r["backend"] == "pytorch" or r["n_gallery"] != big:
                continue
            mf = r.get("margin_flipped_median")
            print(f"| {m} | {r['feature']} | {r['dtype']} | {r['n_flipped']} / {r['n_query']} | "
                  f"{r['flip_pytorch_right']} | {r['flip_jepacpp_right']} | {r['flip_both_wrong']} | "
                  f"{'—' if mf is None else f'{mf:+.4f}'} | {r['margin_all_median']:+.4f} |")
    print()

    print("### Throughput (32 threads, end-to-end: JPEG decode + preprocess + encode)\n")
    print("| backend | model | dtype | split | images | wall s | img/s |")
    print("|---|---|---|---|---|---|---|")
    for k, v in d["timing"].items():
        be, m, dt, sp = k.split("|")
        be = "PyTorch, batch 32" if be == "torch" else "jepa-embed, 1 img/call"
        print(f"| {be} | {m} | {dt} | {sp} | {v['n']} | {v['wall_s']:.1f} | {v['img_per_s']:.2f} |")


if __name__ == "__main__":
    main()
