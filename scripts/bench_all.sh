#!/usr/bin/env bash
# bench_all.sh — run tools/jepa-bench over every GGUF in models/gguf and regenerate docs/benchmarks.md.
#
#   scripts/bench_all.sh [threads-list] [options]
#
#   threads-list   comma-separated thread counts, default "32" (e.g. "32,96")
#
#   --include-quants     also sweep the q4_*/q5_*/q6_k files (default: f32, f16, q8_0 only)
#   --only REGEX         only GGUF basenames matching this egrep pattern
#   --modes LIST         comma list out of encoder,head,predictor,lewm-step,lewm-rollout
#                        (default: every mode the file supports)
#   --frames LIST        override the encoder frame counts (default: per model, see below)
#   --repeat R           measured runs per config (default 3)
#   --warmup W           warmup runs per config (default 1)
#   --steps K            lewm-rollout steps (default 20)
#   --out DIR            where the per-run JSONs go (default tmp/bench)
#   --keep               do not delete existing JSONs in --out before the sweep
#   --note TEXT          free-text note stored in the run metadata (e.g. concurrent load)
#   --no-doc             do not regenerate docs/benchmarks.md at the end
#   --doc FILE           output document (default docs/benchmarks.md)
#   -n / --dry-run       print the jepa-bench command lines and exit
#
# Encoder frame counts, when --frames is not given:
#   image families                    1
#   image+video (V-JEPA 2.1)          1 and min(jepa.enc.n_frames, 16)
#   video, jepa.enc.n_frames > 16     16 and jepa.enc.n_frames   (the fpc64 ViT-L: 2048 and 8192 tokens)
#   video otherwise                   min(jepa.enc.n_frames, 16)
#
# Every jepa-bench process writes one JSON holding one run per thread count; the generator
# (scripts/gen_benchmarks_md.py) merges every JSON in --out into the document.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/build/jepa-bench"
INFO="$ROOT/build/jepa-info"
GGUF_DIR="$ROOT/models/gguf"
OUT="$ROOT/tmp/bench"
DOC="$ROOT/docs/benchmarks.md"
PYTHON="${PYTHON:-/home/overseer2/workdir/jepa.cpp/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

THREADS="32"
INCLUDE_QUANTS=0
ONLY=""
MODES=""
FRAMES_OVERRIDE=""
REPEAT=3
WARMUP=1
STEPS=20
KEEP=0
NOTE=""
GEN_DOC=1
DRY=0

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then THREADS="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --include-quants) INCLUDE_QUANTS=1 ;;
        --only)      ONLY="$2"; shift ;;
        --modes)     MODES="$2"; shift ;;
        --frames)    FRAMES_OVERRIDE="${2//,/ }"; shift ;;
        --repeat)    REPEAT="$2"; shift ;;
        --warmup)    WARMUP="$2"; shift ;;
        --steps)     STEPS="$2"; shift ;;
        --out)       OUT="$2"; shift ;;
        --doc)       DOC="$2"; shift ;;
        --keep)      KEEP=1 ;;
        --note)      NOTE="$2"; shift ;;
        --no-doc)    GEN_DOC=0 ;;
        -n|--dry-run) DRY=1 ;;
        -h|--help)   sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument $1" >&2; exit 1 ;;
    esac
    shift
done

[ -x "$BENCH" ] || { echo "missing $BENCH — build first (cmake --build build -j 32)" >&2; exit 1; }
[ -x "$INFO"  ] || { echo "missing $INFO — build first" >&2; exit 1; }
[ -d "$GGUF_DIR" ] || { echo "missing $GGUF_DIR" >&2; exit 1; }

mkdir -p "$OUT"
if [ "$KEEP" = 0 ] && [ "$DRY" = 0 ]; then rm -f "$OUT"/*.json; fi

want_mode() { [ -z "$MODES" ] || [[ ",$MODES," == *",$1,"* ]]; }

# ---------------------------------------------------------------------------------------------
# metadata: box, toolchain, ggml commit, load at start/end
# ---------------------------------------------------------------------------------------------
write_meta() {  # $1 = "start" | "end"
    local cache="$ROOT/build/CMakeCache.txt"
    local llamafile ggml_commit cxx cpu cores mem
    llamafile=$(grep -m1 '^GGML_LLAMAFILE:BOOL=' "$cache" 2>/dev/null | cut -d= -f2)
    ggml_commit=$(git -C "$ROOT/ggml" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
    cxx=$(c++ --version 2>/dev/null | head -1)
    cpu=$(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
    cores=$(nproc)
    mem=$(grep -m1 MemTotal /proc/meminfo | awk '{printf "%.0f", $2/1048576}')
    "$PYTHON" - "$OUT/meta.json" "$1" <<PY
import json, os, sys, datetime, subprocess
path, phase = sys.argv[1], sys.argv[2]
meta = json.load(open(path)) if os.path.exists(path) else {}
if phase == "start":
    meta.update({
        "date_utc":     datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "cpu":          """$cpu""",
        "cores":        int("""$cores"""),
        "mem_gb":       int("""$mem"""),
        "kernel":       os.uname().release,
        "compiler":     """$cxx""",
        "ggml_commit":  """$ggml_commit""",
        "ggml_llamafile": """$llamafile""" or "unknown",
        "threads":      """$THREADS""",
        "repeat":       $REPEAT,
        "warmup":       $WARMUP,
        "note":         """$NOTE""",
        "loadavg_start": open("/proc/loadavg").read().split()[:3],
    })
else:
    meta["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    meta["date_end_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
json.dump(meta, open(path, "w"), indent=2)
PY
}

# ---------------------------------------------------------------------------------------------
# one jepa-bench invocation
# ---------------------------------------------------------------------------------------------
run_bench() {  # $1 gguf  $2 label  $3 mode  $4 tag  $5... extra args
    local gguf="$1" label="$2" mode="$3" tag="$4"; shift 4
    local stem json
    stem="$(basename "$gguf" .gguf)"
    json="$OUT/${stem}__${mode}__${tag}__t${THREADS//,/_}.json"
    local cmd=("$BENCH" -m "$gguf" --mode "$mode" --label "$label" --threads "$THREADS"
               --repeat "$REPEAT" --warmup "$WARMUP" --steps "$STEPS" --json "$json" "$@")
    if [ "$DRY" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
    echo "--- $stem  $mode  $tag  (threads $THREADS)" >&2
    if ! "${cmd[@]}"; then
        echo "!!! FAILED: $stem $mode $tag" >&2
        FAILURES=$((FAILURES + 1))
    fi
}

FAILURES=0
[ "$DRY" = 1 ] || write_meta start

shopt -s nullglob
for gguf in "$GGUF_DIR"/*.gguf; do
    base="$(basename "$gguf" .gguf)"
    ftype="${base##*-}"
    label="${base%-*}"
    case "$ftype" in
        f32|f16|q8_0) ;;
        q4_0|q4_1|q4_k|q5_0|q5_1|q5_k|q6_k) [ "$INCLUDE_QUANTS" = 1 ] || continue ;;
        *) echo "skipping $base (unrecognised ftype suffix '$ftype')" >&2; continue ;;
    esac
    if [ -n "$ONLY" ] && ! echo "$base" | grep -Eq "$ONLY"; then continue; fi

    info="$("$INFO" "$gguf" 2>/dev/null | head -40)"
    modality=$(echo "$info" | sed -n 's/^family: .*(modality \([a-z+]*\).*/\1/p')
    n_frames=$(echo "$info" | sed -n 's/.* frames=\([0-9]*\) chans.*/\1/p' | head -1)
    tubelet=$(echo "$info"  | sed -n 's/.* tubelet=\([0-9]*\) .*/\1/p' | head -1)
    pred_kind=$(echo "$info" | sed -n 's/^predictor: *kind=\([a-z]*\).*/\1/p')
    has_head=$(echo "$info" | grep -c '^head: *kind=')
    [ -n "$n_frames" ] || n_frames=1
    [ -n "$tubelet"  ] || tubelet=1

    # encoder frame counts
    if [ -n "$FRAMES_OVERRIDE" ]; then
        frame_list="$FRAMES_OVERRIDE"
    elif [ "$tubelet" = 1 ] && [ "$modality" = image ]; then
        frame_list="1"
    else
        base_T=$(( n_frames < 16 ? n_frames : 16 ))
        frame_list="$base_T"
        [ "$n_frames" -gt 16 ] && frame_list="$frame_list $n_frames"
        [ "$modality" = "image+video" ] && frame_list="1 $frame_list"
    fi

    if want_mode encoder; then
        for T in $frame_list; do
            run_bench "$gguf" "$label" encoder "T$T" --frames "$T"
        done
    fi
    if [ "$has_head" -gt 0 ] && want_mode head; then
        T=${frame_list##* }
        [ "$modality" = "image+video" ] && T=$(( n_frames < 16 ? n_frames : 16 ))
        run_bench "$gguf" "$label" head "T$T" --frames "$T"
    fi
    if [ "$pred_kind" = "masked" ] && want_mode predictor; then
        T=$(( n_frames < 16 ? n_frames : 16 ))
        run_bench "$gguf" "$label" predictor "T$T" --frames "$T"
    fi
    if [ "$pred_kind" = "lewm" ]; then
        want_mode lewm-step    && run_bench "$gguf" "$label" lewm-step    "F$(echo "$info" | sed -n 's/.*n_frames=\([0-9]*\).*/\1/p' | tail -1)"
        want_mode lewm-rollout && run_bench "$gguf" "$label" lewm-rollout "K$STEPS"
    fi
done
shopt -u nullglob

[ "$DRY" = 1 ] && exit 0
write_meta end

echo >&2
if [ "$FAILURES" -gt 0 ]; then echo "$FAILURES config(s) failed" >&2; fi
echo "JSONs in $OUT" >&2

if [ "$GEN_DOC" = 1 ]; then
    "$PYTHON" "$ROOT/scripts/gen_benchmarks_md.py" --bench-dir "$OUT" --ref-dir "$ROOT/tests/fixtures/ref" -o "$DOC"
fi
exit $(( FAILURES > 0 ? 1 : 0 ))
