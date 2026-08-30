#!/usr/bin/env bash
# Download the reference checkpoints jepa.cpp converts from. Everything lands in ./models (git-ignored).
# Resumable (curl -C -). Usage: scripts/download_models.sh [all|small|ijepa|vjepa2|vjepa2-ssv2|vjepa21|lewm|lejepa]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$ROOT/models"
HF="https://huggingface.co"
FB="https://dl.fbaipublicfiles.com"
get() { # get <url> <dest>
  mkdir -p "$(dirname "$2")"
  if [ -f "$2.done" ]; then echo "  skip $(basename "$2")"; return; fi
  echo "  get  $1"
  curl -sSL --retry 5 --retry-delay 3 -C - -o "$2" "$1"
  touch "$2.done"
}
hf() { # hf <repo> <file...>
  local repo="$1"; shift
  for f in "$@"; do get "$HF/$repo/resolve/main/$f" "$M/$repo/$f"; done
}
want="${1:-all}"
has() { [ "$want" = all ] || [ "$want" = "$1" ] || { [ "$want" = small ] && [ "$2" = small ]; }; }

has lejepa small && { echo "== LeJEPA ViT-S/16 (OK-AI, 107 MB)"
  hf OK-AI/lejepa-vits16-pretrain-in1k config.json preprocessor_config.json model.safetensors configuration_vitv2.py modelling_vitv2.py README.md; }
has lewm small && { echo "== LeWorldModel Push-T (72 MB)"
  hf quentinll/lewm-pusht config.json weights.pt README.md; }
has vjepa21 small && { echo "== V-JEPA 2.1 ViT-B/16 384 (1.66 GB, torch.hub pickle)"
  get "$FB/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt" "$M/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt"; }
has vjepa2 big && { echo "== V-JEPA 2 ViT-L/16 256 (1.3 GB)"
  hf facebook/vjepa2-vitl-fpc64-256 config.json video_preprocessor_config.json model.safetensors README.md; }
has vjepa2-ssv2 big && { echo "== V-JEPA 2 ViT-L SSv2 classifier (1.5 GB)"
  hf facebook/vjepa2-vitl-fpc16-256-ssv2 config.json video_preprocessor_config.json model.safetensors; }
has ijepa big && { echo "== I-JEPA ViT-H/14 IN1k (2.5 GB)"
  hf facebook/ijepa_vith14_1k config.json preprocessor_config.json model.safetensors; }
echo "done."
