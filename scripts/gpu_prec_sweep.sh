#!/usr/bin/env bash
# gpu_prec_sweep.sh — run tests/test-parity on a CUDA device at BOTH mul_mat accumulation
# precisions and write the verdict table docs/performance.md publishes, tests/results/gpu-prec.json.
#
#   scripts/gpu_prec_sweep.sh [device] [options]
#
#   device         CUDA device index, default 0
#
#   --grid FILE    configuration list, default the table built into this script
#   --only REGEX   only configurations whose "<gguf label>" matches this egrep pattern
#   --out DIR      where the per-run test-parity JSONs go (default tmp/gpu-prec)
#   --keep         do not delete existing JSONs in --out before the sweep
#   --results-json FILE   the committed artifact (default tests/results/gpu-prec.json)
#   --no-json      run the checks but write no artifact
#   -n/--dry-run   print the test-parity command lines and exit
#
# Why it exists. `GGML_PREC_F32` on every `mul_mat` is what a GPU context marks its GEMMs with by
# default, per family (src/jepa.cpp, jepa_gpu_prec_f32_default). Choosing that default is two
# questions — does the family still clear its GPU parity tier with f16 accumulation, and is f16
# accumulation actually faster at its shapes — and this script answers the first for every fixture
# sample of every dtype. The second is scripts/bench_gpu.sh, whose grid pairs each family's default
# row with the setting it does not default to.
#
# The tiers are test-parity's own (POLICY in tests/test-parity.cpp, mirrored in docs/parity.md) and
# are not touched here: this script only chooses which side of them each family lands on.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build-cuda/test-parity"
GGUF_DIR="$ROOT/models/gguf"
REF_DIR="$ROOT/tests/fixtures/ref"
OUT="$ROOT/tmp/gpu-prec"
RESULTS_JSON="$ROOT/tests/results/gpu-prec.json"
GRID=""
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for cand in "$ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
        if [ -x "$cand" ]; then PYTHON="$cand"; break; fi
    done
fi
[ -n "$PYTHON" ] || { echo "no python3 found (set PYTHON=...)" >&2; exit 1; }

DEVICE=0
ONLY=""
KEEP=0
WRITE_JSON=1
DRY=0

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then DEVICE="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --grid)  GRID="$2"; shift ;;
        --only)  ONLY="$2"; shift ;;
        --out)   OUT="$2"; shift ;;
        --results-json) RESULTS_JSON="$2"; shift ;;
        --keep)  KEEP=1 ;;
        --no-json) WRITE_JSON=0 ;;
        -n|--dry-run) DRY=1 ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument $1" >&2; exit 1 ;;
    esac
    shift
done

[ -x "$BIN" ] || { echo "missing $BIN — configure with -DJEPA_CUDA=ON and build build-cuda" >&2; exit 1; }

# <gguf label> | <reference directory under tests/fixtures/ref> | [--samples list]
#
# Every family at every dtype that has fixtures. The big ViT-g files are checked on their two
# 16-frame clips rather than the 64-frame ones as well: the extra samples cost minutes of file I/O
# and the verdict is the same, and docs/parity.md's own ViT-g table is where the 64-frame rows live.
read -r -d '' DEFAULT_GRID <<'GRID_EOF'
ijepa_vith14_1k-f32                 | ijepa-vith14-1k            |
ijepa_vith14_1k-f16                 | ijepa-vith14-1k            |
ijepa_vith14_1k-q8_0                | ijepa-vith14-1k            |
ijepa_vith14_1k-q4_k                | ijepa-vith14-1k            |
lejepa-vits16-pretrain-in1k-f32     | lejepa-vits16              |
lejepa-vits16-pretrain-in1k-f16     | lejepa-vits16              |
lejepa-vits16-pretrain-in1k-q8_0    | lejepa-vits16              |
lejepa-vits16-pretrain-in1k-q4_k    | lejepa-vits16              |
lewm-pusht-f32                      | lewm-pusht                 |
lewm-pusht-f16                      | lewm-pusht                 |
lewm-pusht-q8_0                     | lewm-pusht                 |
lewm-pusht-q4_k                     | lewm-pusht                 |
vjepa2-vitl-fpc64-256-f32           | vjepa2-vitl-fpc64-256      |
vjepa2-vitl-fpc64-256-f16           | vjepa2-vitl-fpc64-256      |
vjepa2-vitl-fpc64-256-q8_0          | vjepa2-vitl-fpc64-256      |
vjepa2-vitl-fpc64-256-q4_k          | vjepa2-vitl-fpc64-256      |
vjepa2-vitl-fpc16-256-ssv2-f16      | vjepa2-vitl-fpc16-256-ssv2 |
vjepa2_1-vitb-384-f32               | vjepa2_1-vitb-384          |
vjepa2_1-vitb-384-f16               | vjepa2_1-vitb-384          |
vjepa2_1-vitb-384-q8_0              | vjepa2_1-vitb-384          |
vjepa2_1-vitb-384-q4_k              | vjepa2_1-vitb-384          |
levjepa-vitl16-f32                  | levjepa-vitl16             |
levjepa-vitl16-f16                  | levjepa-vitl16             |
levjepa-vitl16-q8_0                 | levjepa-vitl16             |
levjepa-vitl16-q4_k                 | levjepa-vitl16             |
vjepa2-vitg-fpc64-256-f16           | vjepa2-vitg-fpc64-256      | --samples archery_f16,bowling_f16
vjepa2-vitg-fpc64-256-q8_0          | vjepa2-vitg-fpc64-256      | --samples archery_f16,bowling_f16
vjepa2-ac-vitg-f16                  | vjepa2-ac-vitg             | --samples frame0,frame1,goalframe
vjepa2-ac-vitg-q8_0                 | vjepa2-ac-vitg             | --samples frame0,frame1,goalframe
GRID_EOF

mkdir -p "$OUT"
[ "$KEEP" = 0 ] && [ "$DRY" = 0 ] && find "$OUT" -maxdepth 1 -name '*.json' -delete

FAILED_TO_RUN=0
while IFS='|' read -r label ref extra; do
    label="${label// /}"; ref="${ref// /}"
    [ -n "$label" ] || continue
    case "$label" in \#*) continue ;; esac
    [ -z "$ONLY" ] || echo "$label" | grep -Eq "$ONLY" || continue
    gguf="$GGUF_DIR/${label}.gguf"
    if [ ! -f "$gguf" ]; then echo "!!! MISSING $gguf" >&2; continue; fi
    # shellcheck disable=SC2206
    extra_args=($extra)
    for prec in f32 f16; do
        json="$OUT/${label}__${prec}.json"
        cmd=("$BIN" "$gguf" "$REF_DIR/$ref" --gpu "$DEVICE" --quiet --json "$json" "${extra_args[@]}")
        if [ "$DRY" = 1 ]; then printf 'JEPA_GPU_PREC=%s ' "$prec"; printf '%q ' "${cmd[@]}"; echo; continue; fi
        echo "--- $label  prec=$prec  (CUDA$DEVICE)" >&2
        JEPA_GPU_PREC="$prec" "${cmd[@]}" > "$OUT/${label}__${prec}.log" 2>&1
        rc=$?
        # Exit status 1 is a tier failure, which is a RESULT here, not an error; anything else means
        # the check did not run and the artifact must not pretend it did.
        if [ "$rc" != 0 ] && [ "$rc" != 1 ]; then
            echo "!!! test-parity exited $rc for $label prec=$prec" >&2
            FAILED_TO_RUN=$((FAILED_TO_RUN + 1))
        fi
        echo "    $(grep -h RESULT "$OUT/${label}__${prec}.log" | tail -1)" >&2
    done
done <<< "${GRID:+$(cat "$GRID")}${GRID:-$DEFAULT_GRID}"

[ "$DRY" = 1 ] && exit 0
[ "$WRITE_JSON" = 1 ] || exit $(( FAILED_TO_RUN > 0 ))

"$PYTHON" - "$OUT" "$RESULTS_JSON" "$DEVICE" "$ROOT" <<'PY'
"""Fold the per-run test-parity JSONs into the committed verdict artifact."""
import datetime, json, os, pathlib, subprocess, sys

out_dir, results_json, device, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
DERIVED = ("pooled_mean", "pooled", "cls", "emb", "emb_seq", "logits")


def worst(blob: dict) -> dict:
    """The worst sample of a run, over the stored-input pass of every sample: the minimum cosine and
    the maximum relative error, which is what the tiers are read against."""
    tm = {"cos_mean": 1.0, "cos_med": 1.0, "cos_min": 1.0, "rel_max": 0.0}
    der = {"cos_mean": 1.0, "rel_max": 0.0}
    tokens = 0
    for s in blob.get("samples", []):
        p = s.get("stored_input") or s.get("own_preprocess")
        if not p:
            continue
        tokens = max(tokens, p.get("n_tokens", 0))
        t = p.get("last_hidden_state")
        if t:
            for k in ("cos_mean", "cos_med", "cos_min"):
                tm[k] = min(tm[k], t[k])
            tm["rel_max"] = max(tm["rel_max"], t["rel_max"])
        for k in DERIVED:
            d = p.get(k)
            if isinstance(d, dict):
                der["cos_mean"] = min(der["cos_mean"], d["cos_mean"])
                der["rel_max"] = max(der["rel_max"], d["rel_max"])
    return {"tokens": tokens, "token_map": tm, "derived": der}


rows, by_file = [], {}
for p in sorted(out_dir.glob("*.json")):
    label, prec = p.stem.rsplit("__", 1)
    blob = json.loads(p.read_text())
    by_file.setdefault(label, {})[prec] = (blob, worst(blob))

for label, d in sorted(by_file.items()):
    any_blob = next(iter(d.values()))[0]
    row = {
        "file": label, "family": any_blob.get("family"), "ftype": any_blob.get("file_type"),
        "ref": pathlib.Path(any_blob.get("ref", "")).name,
        "samples": len(any_blob.get("samples", [])),
        "device": any_blob.get("device"),
        "thresholds": any_blob.get("thresholds"),
    }
    for prec in ("f32", "f16"):
        if prec not in d:
            continue
        blob, w = d[prec]
        row["tokens"] = w["tokens"]
        row[f"prec_{prec}"] = {"pass": blob.get("pass"), **w}
    if "prec_f32" in row and "prec_f16" in row:
        row["verdict"] = ("f16 accumulation passes the same tier"
                          if row["prec_f16"]["pass"] and row["prec_f32"]["pass"] else
                          "f16 accumulation fails a tier the default passes"
                          if row["prec_f32"]["pass"] else
                          "neither precision passes (a known non-pass, see docs/parity.md)")
    rows.append(row)

box = {}
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,power.limit,driver_version",
                          "--format=csv,noheader", "-i", device],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    name, mem, cc, power, driver = [c.strip() for c in gpu.split(",")]
    box = {"device": name, "device_index": int(device), "device_memory": mem, "compute_cap": cc,
           "power_limit": power, "driver": driver}
except Exception:
    box = {"device_index": int(device)}
try:
    box["git_commit"] = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"],
                                       capture_output=True, text=True, check=True).stdout.strip()
    box["ggml_commit"] = subprocess.run(["git", "-C", str(root / "ggml"), "rev-parse", "--short=8", "HEAD"],
                                        capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    pass
box["kernel"] = os.uname().release

blob = {
    "task": "GPU parity of every jepa.cpp family at both mul_mat accumulation precisions — the "
            "measurement behind the per-family GGML_PREC_F32 default in src/jepa.cpp "
            "(jepa_gpu_prec_f32_default)",
    "generated_by": "scripts/gpu_prec_sweep.sh",
    "protocol": {
        "runner": "tests/test-parity, one process per (file, precision), all fixture samples unless "
                  "the grid names a subset",
        "precision": "$JEPA_GPU_PREC=f32 forces GGML_PREC_F32 on every mul_mat, =f16 hands cuBLAS "
                     "its own compute type; both override the family default under test",
        "thresholds": "test-parity's own POLICY[gpu][family class][tier], unchanged — this sweep "
                      "chooses which side of them a family lands on and never moves them",
        "metrics": "worst sample of the run over the stored-input pass: minimum per-token cosine "
                   "mean/median/min and maximum rel_max, and the same for the worst derived tensor",
        "affected": "only f16 weights change: a quantized file takes mmq and never reaches cuBLAS, "
                    "and ggml's f32 CUDA path is TF32 regardless — the small movements in the "
                    "quantized rows are the f16/f32 tensors a quantized file still carries",
        "timing": "not measured here; the speed half of the decision is "
                  "tests/results/benchmarks-gpu.json (scripts/bench_gpu.grid pairs each family's "
                  "default row with the setting it does not default to)",
    },
    "box": box,
    "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "rows": rows,
}
results_json.parent.mkdir(parents=True, exist_ok=True)
results_json.write_text(json.dumps(blob, indent=1) + "\n")
print(f"wrote {results_json} ({len(rows)} files, {os.path.getsize(results_json)} bytes)")
PY

exit $(( FAILED_TO_RUN > 0 ))
