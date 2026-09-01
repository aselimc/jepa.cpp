#!/usr/bin/env bash
# Small labeled datasets for the inference-only accuracy benchmarks (k-NN on frozen features, no training).
# Everything lands in ./data (git-ignored, ~400 MB). Re-runnable; skips what is already extracted.
#   data/imagenette/imagenette2-160/{train,val}/<wnid>/*.JPEG   10 classes, 9469 / 3925 images (fast.ai, 160px short side)
#   data/ucf101-subset/UCF101_subset/{train,val,test}/<class>/*.avi   10 classes, 300 / 30 / 75 clips (HF sayakpaul/ucf101-subset)
# Something-Something-v2 (data/ssv2, ~19 GB) is licence-gated and stays a manual step — see the note at the end.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/data"; mkdir -p "$D/imagenette" "$D/ucf101-subset"
export HF_HOME="$ROOT/tmp/hf-home"

if [ ! -d "$D/imagenette/imagenette2-160/train" ]; then
  echo "== Imagenette-160 (99 MB, fast.ai)"
  curl -sSL --retry 3 -o "$D/imagenette/imagenette2-160.tgz" https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
  tar -xzf "$D/imagenette/imagenette2-160.tgz" -C "$D/imagenette"
fi
echo "imagenette: $(find "$D/imagenette/imagenette2-160/train" -name '*.JPEG' | wc -l) train / $(find "$D/imagenette/imagenette2-160/val" -name '*.JPEG' | wc -l) val images"

if [ ! -d "$D/ucf101-subset/UCF101_subset/train" ]; then
  echo "== UCF101 subset (171 MB, HF sayakpaul/ucf101-subset)"
  "$ROOT/.venv/bin/python" - "$D/ucf101-subset" <<'PY'
import sys; from huggingface_hub import snapshot_download
snapshot_download(repo_id="sayakpaul/ucf101-subset", repo_type="dataset", local_dir=sys.argv[1], allow_patterns=["UCF101_subset.tar.gz", "README.md"])
PY
  tar -xf "$D/ucf101-subset/UCF101_subset.tar.gz" -C "$D/ucf101-subset"   # plain POSIX tar despite the .gz name
fi
for s in train val test; do echo "ucf101-subset $s: $(ls "$D/ucf101-subset/UCF101_subset/$s" | wc -l) classes, $(find "$D/ucf101-subset/UCF101_subset/$s" -name '*.avi' | wc -l) clips"; done

# --- Something-Something-v2: manual, and deliberately not automated here --------------------------
# The SSv2 validation accuracy benchmark (scripts/bench_accuracy_ssv2.py, and the SSv2 section of
# docs/accuracy.md) needs the full dataset, which is not ours to redistribute: Qualcomm hosts it
# under its own dataset licence terms and nothing below fetches, mirrors or embeds a byte of it.
# Fetch it yourself from
#
#     https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads
#
# The page is a client-rendered app, so the links live in its page-model JSON rather than the static
# HTML.  It listed three files when this benchmark was measured (2026-09-01) and needed no login:
# two ~10 GB parts of one split tar.gz stream and a labels zip.  The download endpoints reject a
# bare curl (HTTP 403) and serve the file once a normal browser User-Agent and a Referer of the
# downloads page are sent.  Reassemble and lay the result out as the benchmark expects:
#
#     cat 20bn-something-something-v2-00 20bn-something-something-v2-01 | tar -xzf -   # ~19 GB
#     # -> data/ssv2/videos/<id>.webm            220 847 clips, VP9, 240 px short side
#     # -> data/ssv2/labels/labels.json          174 class names -> class ids
#     # -> data/ssv2/labels/validation.json      24 777 entries (id, template, placeholders, label)
#
if [ -d "$D/ssv2/videos" ]; then
  echo "ssv2: $(find "$D/ssv2/videos" -name '*.webm' | wc -l) clips, $(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$D/ssv2/labels/validation.json" 2>/dev/null || echo '?') validation entries"
else
  echo "ssv2: not present (optional, ~19 GB, manual download — see the note in $0)"
fi
