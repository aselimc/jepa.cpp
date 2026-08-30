#!/usr/bin/env bash
# Small parity fixtures: 8 COCO val images + a handful of short Kinetics clips. Lands in tests/fixtures/media.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/tests/fixtures/media"; mkdir -p "$D"
get() { [ -f "$2" ] || { echo "  get $1"; curl -sSL --retry 5 -o "$2" "$1"; }; }
for id in 000000039769 000000219578 000000000139 000000000285 000000000632 000000000724 000000000776 000000000785; do
  get "http://images.cocodataset.org/val2017/$id.jpg" "$D/coco_$id.jpg"
done
KM="https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val"
get "$KM/archery/-Qz25rXdMjE_000014_000024.mp4" "$D/archery.mp4"
get "$KM/bowling/-WH-lxmGJVY_000005_000015.mp4" "$D/bowling.mp4"
# discover a few more classes from the dataset listing
curl -sL "https://huggingface.co/api/datasets/nateraw/kinetics-mini" | python3 - "$D" <<'PY'
import sys, json, subprocess, os, collections
d = json.load(sys.stdin); out = sys.argv[1]
by = collections.OrderedDict()
for s in d.get("siblings", []):
    n = s["rfilename"]
    if n.startswith("val/") and n.endswith(".mp4"):
        cls = n.split("/")[1]
        by.setdefault(cls, n)
picked = [v for k, v in by.items() if k not in ("archery", "bowling")][:6]
for n in picked:
    cls = n.split("/")[1]; dst = os.path.join(out, f"{cls}.mp4")
    if not os.path.exists(dst):
        print("  get", n)
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", dst,
                        "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/" + n], check=True)
PY
ls -la "$D"
