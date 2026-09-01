#!/usr/bin/env bash
# Parity fixtures: the PyTorch golden dumps (ref/, from the jepacpp/jepa.cpp-fixtures dataset) and the
# media they were computed on (8 COCO val2017 images + 6 short Kinetics-400 clips from nateraw/kinetics-mini).
#
# Usage: scripts/download_fixtures.sh [all|media|ref] [REFNAME...]
#   all     media + ref (default; ~665 MB)
#   media   the 8 jpgs + 6 clips only (~5 MB)  -> tests/fixtures/media  (override with JEPA_FIXTURES_DIR)
#   ref     the golden dumps only (660 MB)     -> tests/fixtures/ref
#   REFNAME limits ref to those directories, e.g. `ref lejepa-vits16 lewm-pusht`
#
# `test-parity` runs two passes per file: the stored reference input (needs ref/) and jepa.cpp's own
# preprocessing of the source media (needs media/). The dumps are regenerable instead of downloadable
# with scripts/dump_reference.py --model all (needs the torch venv and the source checkpoints).
#
# The clip list is hard-coded on purpose: the golden references in tests/fixtures/ref are tied to these exact
# files, so discovery via the HF listing API (which returns HTML, not JSON, for datasets without a loading
# script) would only add non-determinism. To browse the dataset yourself use the tree endpoint, e.g.
#   curl -sL https://huggingface.co/api/datasets/nateraw/kinetics-mini/tree/main/val/marching
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="${JEPA_FIXTURES_DIR:-$ROOT/tests/fixtures/media}"
R="$ROOT/tests/fixtures/ref"
DS="jepacpp/jepa.cpp-fixtures"
HFBIN="${HF_CLI:-$ROOT/.venv/bin/hf}"; [ -x "$HFBIN" ] || HFBIN="$(command -v hf || true)"

what="${1:-all}"; [ $# -gt 0 ] && shift || true
case "$what" in -h|--help) sed -n '2,13p' "$0"; exit 0;; esac

get() { # get <url> <dest>
  if [ -s "$2" ]; then return; fi
  echo "  get $1"
  curl -sSL --retry 5 --retry-delay 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

if [ "$what" = all ] || [ "$what" = media ]; then
  mkdir -p "$D"
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
fi

if [ "$what" = all ] || [ "$what" = ref ]; then
  echo "== golden reference dumps from https://huggingface.co/datasets/$DS"
  mkdir -p "$R"
  # --include keeps the download inside ref/, so the tracked tests/fixtures/README.md is never overwritten
  inc=(); for n in "$@"; do inc+=(--include "ref/$n/*"); done
  [ ${#inc[@]} -eq 0 ] && inc=(--include "ref/*")
  if [ -n "$HFBIN" ]; then
    "$HFBIN" download "$DS" --repo-type dataset "${inc[@]}" --local-dir "$ROOT/tests/fixtures" >/dev/null
  else
    echo "  the 'hf' CLI is not installed (pip install huggingface_hub); fetching with curl" >&2
    # the tree API paginates, so follow rel="next" rather than reading one page and hoping
    for f in $(python3 - "$DS" "$@" <<'EOF'
import json, sys, urllib.request
ds, names = sys.argv[1], sys.argv[2:]
url, out = f"https://huggingface.co/api/datasets/{ds}/tree/main/ref?recursive=true", []
while url:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "jepa.cpp"}), timeout=60) as r:
        out += [e["path"] for e in json.load(r) if e.get("type") == "file"]
        link = r.headers.get("Link", "")
    url = next((p.split(";")[0].strip().strip("<>") for p in link.split(",") if 'rel="next"' in p), None)
print("\n".join(p for p in out if not names or p.split("/")[1] in names))
EOF
    ); do
      mkdir -p "$ROOT/tests/fixtures/$(dirname "$f")"
      get "https://huggingface.co/datasets/$DS/resolve/main/$f" "$ROOT/tests/fixtures/$f"
    done
  fi
  du -sh "$R"/* 2>/dev/null || true
fi
echo "done. (re-run cmake once tests/fixtures/ref is populated: the parity tests register at configure time)"
