#!/usr/bin/env bash
# Small labeled datasets for the inference-only accuracy benchmarks (k-NN on frozen features, no training).
# Everything lands in ./data (git-ignored, ~400 MB). Re-runnable; skips what is already extracted.
#   data/imagenette/imagenette2-160/{train,val}/<wnid>/*.JPEG   10 classes, 9469 / 3925 images (fast.ai, 160px short side)
#   data/ucf101-subset/UCF101_subset/{train,val,test}/<class>/*.avi   10 classes, 300 / 30 / 75 clips (HF sayakpaul/ucf101-subset)
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
