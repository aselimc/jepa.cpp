#!/usr/bin/env bash
# Download jepa.cpp models.
#
#   default    the converted GGUFs from https://huggingface.co/jepacpp -> ./models/gguf
#   --convert  the *source* checkpoints jepa.cpp converts from        -> ./models
#              (git-ignored either way; --convert is what scripts/convert.py reads)
#
# Usage: scripts/download_models.sh [--convert] [--ftype TYPE[,TYPE...]] [all|small|NAME...]
#   NAME    ijepa | lejepa | lewm | vjepa2 | vjepa2-ssv2 | vjepa21 | levjepa | vjepa2-vitg | vjepa2-ac
#   small   lejepa, lewm, vjepa21 ("all" adds the other four)
#   TYPE    f16 (default) | f32 | q8_0 | q4_0 | q4_k | all
# Resumable (hf download's cache, or curl -C -).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$ROOT/models"
G="$M/gguf"
HF="https://huggingface.co"
FB="https://dl.fbaipublicfiles.com"
ORG="jepacpp"
HFBIN="${HF_CLI:-$ROOT/.venv/bin/hf}"; [ -x "$HFBIN" ] || HFBIN="$(command -v hf || true)"

mode=gguf
ftypes=f16
while [ $# -gt 0 ]; do
  case "$1" in
    --convert)  mode=source; shift;;
    --ftype)    ftypes="$2"; shift 2;;
    --ftype=*)  ftypes="${1#--ftype=}"; shift;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0;;
    *)          break;;
  esac
done
[ "$ftypes" = all ] && ftypes="f32,f16,q8_0,q4_0,q4_k"
want="${1:-all}"
has() { [ "$want" = all ] || [ "$want" = "$1" ] || { [ "$want" = small ] && [ "$2" = small ]; }; }

get() { # get <url> <dest>
  mkdir -p "$(dirname "$2")"
  if [ -f "$2.done" ]; then echo "  skip $(basename "$2")"; return; fi
  echo "  get  $1"
  curl -sSL --retry 5 --retry-delay 3 -C - -o "$2" "$1"
  touch "$2.done"
}
hfget() { # hfget <repo> <file...>
  local repo="$1"; shift
  for f in "$@"; do get "$HF/$repo/resolve/main/$f" "$M/$repo/$f"; done
}
gguf() { # gguf <basename>  -- one repo, every requested ftype
  local base="$1" repo="$ORG/$1-GGUF" f
  mkdir -p "$G"
  for t in ${ftypes//,/ }; do
    f="$base-$t.gguf"
    if [ -s "$G/$f" ]; then echo "  skip $f"; continue; fi
    echo "  get  $repo/$f"
    if [ -n "$HFBIN" ]; then
      "$HFBIN" download "$repo" "$f" --local-dir "$G" >/dev/null
    else
      curl -sSL --retry 5 --retry-delay 3 -C - -o "$G/$f.part" "$HF/$repo/resolve/main/$f" && mv "$G/$f.part" "$G/$f"
    fi
  done
}

if [ "$mode" = gguf ]; then
  echo "== GGUFs from $HF/$ORG (ftype: $ftypes) -> models/gguf"
  has lejepa      small && { echo "== LeJEPA ViT-S/16";               gguf lejepa-vits16-pretrain-in1k; }
  has lewm        small && { echo "== LeWorldModel Push-T";           gguf lewm-pusht; }
  has vjepa21     small && { echo "== V-JEPA 2.1 ViT-B/16 384";       gguf vjepa2_1-vitb-384; }
  has vjepa2      big   && { echo "== V-JEPA 2 ViT-L/16 256";         gguf vjepa2-vitl-fpc64-256; }
  has vjepa2-ssv2 big   && { echo "== V-JEPA 2 ViT-L SSv2 classifier"; gguf vjepa2-vitl-fpc16-256-ssv2; }
  has ijepa       big   && { echo "== I-JEPA ViT-H/14 IN1k";          gguf ijepa_vith14_1k; }
  has levjepa     big   && { echo "== LeVJEPA ViT-L/16 VideoMix";     gguf levjepa-vitl16; }
  has vjepa2-vitg big   && { echo "== V-JEPA 2 ViT-g/16 256";         gguf vjepa2-vitg-fpc64-256; }
  has vjepa2-ac   big   && { echo "== V-JEPA 2-AC ViT-g world model"; gguf vjepa2-ac-vitg; }
  echo "done. (re-run cmake once models/gguf is populated: the parity tests register at configure time)"
  exit 0
fi

echo "== source checkpoints -> models/ (convert with scripts/convert.py, see docs/getting-started.md)"
has lejepa small && { echo "== LeJEPA ViT-S/16 (OK-AI, 107 MB)"
  hfget OK-AI/lejepa-vits16-pretrain-in1k config.json preprocessor_config.json model.safetensors configuration_vitv2.py modelling_vitv2.py README.md
  # modelling_vitv2.py imports hf_src/ (not listed in the model card); fetch the package so trust_remote_code works offline
  for f in $(curl -sL "https://huggingface.co/api/models/OK-AI/lejepa-vits16-pretrain-in1k" | python3 -c "import sys,json; print(\" \".join(s[\"rfilename\"] for s in json.load(sys.stdin)[\"siblings\"] if s[\"rfilename\"].startswith(\"hf_src/\")))"); do hfget OK-AI/lejepa-vits16-pretrain-in1k "$f"; done; }
has lewm small && { echo "== LeWorldModel Push-T (72 MB)"
  hfget quentinll/lewm-pusht config.json weights.pt README.md; }
has vjepa21 small && { echo "== V-JEPA 2.1 ViT-B/16 384 (1.66 GB, torch.hub pickle)"
  get "$FB/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt" "$M/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt"; }
has vjepa2 big && { echo "== V-JEPA 2 ViT-L/16 256 (1.3 GB)"
  hfget facebook/vjepa2-vitl-fpc64-256 config.json video_preprocessor_config.json model.safetensors README.md; }
has vjepa2-ssv2 big && { echo "== V-JEPA 2 ViT-L SSv2 classifier (1.5 GB)"
  hfget facebook/vjepa2-vitl-fpc16-256-ssv2 config.json video_preprocessor_config.json model.safetensors; }
has ijepa big && { echo "== I-JEPA ViT-H/14 IN1k (2.5 GB)"
  hfget facebook/ijepa_vith14_1k config.json preprocessor_config.json model.safetensors; }
has levjepa big && { echo "== LeVJEPA ViT-L/16 VideoMix (1.2 GB; modeling_*.py is the PyTorch reference, trust_remote_code)"
  hfget galilai-group/LeVJEPA-VideoMix-Large config.json modeling_levjepa.py configuration_levjepa.py model.safetensors README.md; }
has vjepa2-vitg big && { echo "== V-JEPA 2 ViT-g/16 256 (4.1 GB)"
  hfget facebook/vjepa2-vitg-fpc64-256 config.json video_preprocessor_config.json model.safetensors README.md; }
has vjepa2-ac big && { echo "== V-JEPA 2-AC ViT-g (11.8 GB torch.hub checkpoint: encoder + AC predictor + optimizer state)"
  get "$FB/vjepa2/vjepa2-ac-vitg.pt" "$M/vjepa2_ac/vjepa2-ac-vitg.pt"; }
echo "done."
