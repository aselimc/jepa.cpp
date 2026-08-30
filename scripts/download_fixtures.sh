#!/usr/bin/env bash
# Small parity fixtures: 8 COCO val2017 images + 6 short Kinetics-400 clips (nateraw/kinetics-mini).
# Lands in tests/fixtures/media (override with JEPA_FIXTURES_DIR=...). Images are tracked in git, clips are not.
#
# The clip list is hard-coded on purpose: the golden references in tests/fixtures/ref are tied to these exact
# files, so discovery via the HF listing API (which returns HTML, not JSON, for datasets without a loading
# script) would only add non-determinism. To browse the dataset yourself use the tree endpoint, e.g.
#   curl -sL https://huggingface.co/api/datasets/nateraw/kinetics-mini/tree/main/val/marching
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="${JEPA_FIXTURES_DIR:-$ROOT/tests/fixtures/media}"; mkdir -p "$D"
get() { # get <url> <dest>
  if [ -s "$2" ]; then return; fi
  echo "  get $1"
  curl -sSL --retry 5 --retry-delay 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}
echo "== COCO val2017 images"
for id in 000000039769 000000219578 000000000139 000000000285 000000000632 000000000724 000000000776 000000000785; do
  get "http://images.cocodataset.org/val2017/$id.jpg" "$D/coco_$id.jpg"
done
echo "== Kinetics-mini clips (10 s each, 15-30 fps, <=480p)"
KM="https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main"
get "$KM/val/archery/-Qz25rXdMjE_000014_000024.mp4"     "$D/archery.mp4"      # 300 frames, 480x360, 29.97 fps
get "$KM/val/bowling/-WH-lxmGJVY_000005_000015.mp4"     "$D/bowling.mp4"      # 150 frames, 480x270, 15 fps
get "$KM/val/flying_kite/0QH8uFjXiW4_000003_000013.mp4" "$D/flying_kite.mp4"
get "$KM/val/high_jump/0oL36GHlSXw_000022_000032.mp4"   "$D/high_jump.mp4"
get "$KM/val/marching/1m-Kdky1y84_000022_000032.mp4"    "$D/marching.mp4"
get "$KM/train/high_jump/-B228_dxIVc_000001_000011.mp4" "$D/high_jump2.mp4"   # val split only has 5 classes; 6th clip from train
ls -la "$D"
