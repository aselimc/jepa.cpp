#!/usr/bin/env bash
# bench_gpu.sh — run tools/jepa-bench on a CUDA device over the configurations docs/performance.md
# tabulates, and write their machine-readable twin, tests/results/benchmarks-gpu.json.
#
#   scripts/bench_gpu.sh [device] [options]
#
#   device         CUDA device index, default 0 (docs/performance.md times device 0 alone)
#
#   --grid FILE    configuration list, default scripts/bench_gpu.grid (see that file's header)
#   --only REGEX   only rows whose "<label> <mode> <tag> <ftype>" matches this egrep pattern
#   --repeat R     measured runs per config (default 5)
#   --warmup W     unmeasured runs before them (default 2 — ggml's CUDA backend captures a graph on
#                  the third call of a topology, so fewer would time the uncaptured path)
#   --out DIR      where the per-run JSONs go (default tmp/bench-gpu)
#   --keep         do not delete existing JSONs in --out before the sweep
#   --note TEXT    free-text note stored in the session metadata
#   --torch FILE   PyTorch-on-the-same-card baseline JSON to fold into the artifact
#                  (default tmp/bench-gpu/torch-gpu.json, from scripts/torch_gpu_baseline.py)
#   --torch-batched FILE  a second scripts/torch_gpu_baseline.py output, its --batch/--compile
#                  sweep, stored beside the batch-1 baseline rather than replacing it
#                  (default tmp/bench-gpu/torch-gpu-batched.json)
#   --merge        overlay this sweep's rows on the committed artifact instead of replacing it,
#                  which is what a partial sweep (--only, or a grid of a few lines) wants: a row is
#                  replaced when its (model, ftype, mode, shape, device, gpu_prec, steps) matches
#                  and every other row is carried through. Without it the artifact holds exactly
#                  what this sweep measured.
#   --results-json FILE   the committed artifact (default tests/results/benchmarks-gpu.json)
#   --doc FILE     document to regenerate (default docs/benchmarks.md)
#   --no-doc       do not regenerate it
#   -n/--dry-run   print the jepa-bench command lines and exit
#
# The sweep is the GPU half of scripts/bench_all.sh and deliberately mirrors it: one jepa-bench
# process per configuration, one JSON each, plus a meta.json holding the box, the toolchain and the
# load the session saw. What differs is the key — a GPU row is keyed by device and precision rather
# than by thread count — and the protocol: best of 5 after 2 warmups at the accumulation precision
# the model's family defaults to, unless the grid's fifth field names one.
#
# The aggregate is written by scripts/gen_benchmarks_md.py --gpu-dir, which also renders the GPU
# tables of docs/benchmarks.md; with the raw JSONs gone the same generator rebuilds those tables
# from the committed artifact alone.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/build-cuda/jepa-bench"
GGUF_DIR="$ROOT/models/gguf"
OUT="$ROOT/tmp/bench-gpu"
GRID="$ROOT/scripts/bench_gpu.grid"
DOC="$ROOT/docs/benchmarks.md"
RESULTS_JSON="$ROOT/tests/results/benchmarks-gpu.json"
TORCH_JSON=""
TORCH_BATCHED_JSON=""
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for cand in "$ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
        if [ -x "$cand" ]; then PYTHON="$cand"; break; fi
    done
fi
[ -n "$PYTHON" ] || { echo "no python3 found (set PYTHON=...)" >&2; exit 1; }

DEVICE=0
ONLY=""
REPEAT=5
WARMUP=2
KEEP=0
NOTE=""
GEN_DOC=1
DRY=0
MERGE=0

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then DEVICE="$1"; shift; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --grid)    GRID="$2"; shift ;;
        --only)    ONLY="$2"; shift ;;
        --repeat)  REPEAT="$2"; shift ;;
        --warmup)  WARMUP="$2"; shift ;;
        --out)     OUT="$2"; shift ;;
        --note)    NOTE="$2"; shift ;;
        --torch)   TORCH_JSON="$2"; shift ;;
        --torch-batched) TORCH_BATCHED_JSON="$2"; shift ;;
        --results-json) RESULTS_JSON="$2"; shift ;;
        --doc)     DOC="$2"; shift ;;
        --keep)    KEEP=1 ;;
        --merge)   MERGE=1 ;;
        --no-doc)  GEN_DOC=0 ;;
        -n|--dry-run) DRY=1 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument $1" >&2; exit 1 ;;
    esac
    shift
done

[ -x "$BENCH" ] || { echo "missing $BENCH — configure with -DJEPA_CUDA=ON and build build-cuda" >&2; exit 1; }
[ -f "$GRID" ]  || { echo "missing grid file $GRID" >&2; exit 1; }
[ -d "$GGUF_DIR" ] || { echo "missing $GGUF_DIR" >&2; exit 1; }
[ -n "$TORCH_JSON" ] || TORCH_JSON="$OUT/torch-gpu.json"
[ -n "$TORCH_BATCHED_JSON" ] || TORCH_BATCHED_JSON="$OUT/torch-gpu-batched.json"

mkdir -p "$OUT"
# The PyTorch baseline is written by a separate, much slower script and is not part of a re-sweep,
# so a plain (non---keep) run must not delete it along with the previous sweep's JSONs.
if [ "$KEEP" = 0 ] && [ "$DRY" = 0 ]; then
    find "$OUT" -maxdepth 1 -name '*.json' ! -name "$(basename "$TORCH_JSON")" \
                                       ! -name "$(basename "$TORCH_BATCHED_JSON")" -delete
fi

# ---------------------------------------------------------------------------------------------
# metadata: card, driver, toolchain, ggml/git commit, and how busy the box was
# ---------------------------------------------------------------------------------------------
# Values reach the Python through argv, never through the heredoc (which is quoted): a device name
# or a --note carrying a quote would otherwise be spliced into the source. A failure aborts the
# sweep — GPU milliseconds without the card and the driver behind them are not a measurement.
write_meta() {  # $1 = "start" | "end"
    local cache="$ROOT/build-cuda/CMakeCache.txt"
    local llamafile ggml_commit git_commit cxx nvcc gpu driver cuda_driver_api own_cpu
    llamafile=$(grep -m1 '^GGML_LLAMAFILE:BOOL=' "$cache" 2>/dev/null | cut -d= -f2)
    ggml_commit=$(git -C "$ROOT/ggml" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
    git_commit=$(git -C "$ROOT" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
    cxx=$(c++ --version 2>/dev/null | head -1)
    nvcc=$(nvcc --version 2>/dev/null | sed -n 's/.*, V\([0-9.]*\).*/\1/p' | head -1)
    gpu=$(nvidia-smi --query-gpu=name,memory.total,compute_cap,power.limit \
                     --format=csv,noheader -i "$DEVICE" 2>/dev/null | head -1)
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$DEVICE" 2>/dev/null | head -1)
    cuda_driver_api=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
    # The sweep's own CPU time has to come from *this* shell — it is the process that reaps every
    # jepa-bench — so `times` (its second line is the children's user and system time) is read here
    # and handed over; a fresh Python interpreter has no children and would report zero, turning
    # the sweep's own work into "foreign" occupancy. The redirection keeps the builtin in this
    # shell: inside a command substitution it would run in a fork, whose child times start at zero.
    times > "$OUT/.times" 2>/dev/null
    own_cpu=$(tail -1 "$OUT/.times" 2>/dev/null)
    "$PYTHON" - "$OUT/meta.json" "$1" "$DEVICE" "$gpu" "$driver" "$cuda_driver_api" "$nvcc" \
                 "$cxx" "$ggml_commit" "$git_commit" "${llamafile:-unknown}" \
                 "$REPEAT" "$WARMUP" "$NOTE" "$own_cpu" <<'PY'
import json, os, re, sys, datetime, time
(path, phase, device, gpu, driver, cuda_driver_api, nvcc, cxx, ggml_commit, git_commit,
 llamafile, repeat, warmup, note, own_cpu) = sys.argv[1:16]

def shell_children_s(s: str) -> float:
    """Seconds out of bash's `times` children line, "1m2.345s 0m6.789s"."""
    total = 0.0
    for m in re.finditer(r"(\d+)m([\d.]+)s", s):
        total += int(m.group(1)) * 60 + float(m.group(2))
    return total

def occupancy_sample():
    """(wall, machine busy CPU-seconds, the sweep's own CPU-seconds).  The difference between the
    two CPU figures over a session is work that belonged to somebody else — the load average cannot
    say that, because the sweep's own runs move it too."""
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    hz = os.sysconf("SC_CLK_TCK")
    return (time.time(), (sum(v) - v[3] - v[4]) / hz, shell_children_s(own_cpu))

name, mem, cc, power = ([c.strip() for c in gpu.split(",")] + ["", "", "", ""])[:4]
meta = json.load(open(path)) if os.path.exists(path) else {}
meta.update({
    "device_index":    int(device),
    "device":          name,
    "device_memory":   mem,
    "compute_cap":     cc,
    "power_limit":     power,
    "driver":          driver,
    "cuda_driver_api": cuda_driver_api,
    "nvcc":            nvcc,
    "kernel":          os.uname().release,
    "compiler":        cxx,
    "ggml_commit":     ggml_commit,
    "git_commit":      git_commit,
    "ggml_llamafile":  llamafile,
})
sessions = meta.setdefault("sessions", [])
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
load = open("/proc/loadavg").read().split()[0]
wall, busy, own = occupancy_sample()
if phase == "start":
    sessions.append({"device": f"CUDA{device}", "repeat": int(repeat), "warmup": int(warmup),
                     "note": note, "start_utc": now, "loadavg_start": load,
                     "_t0": [wall, busy, own]})
elif sessions:
    s = sessions[-1]
    s["end_utc"], s["loadavg_end"] = now, load
    t0 = s.pop("_t0", None)
    if t0:
        elapsed = max(wall - t0[0], 1e-9)
        machine, mine = busy - t0[1], own - t0[2]
        foreign = max(machine - mine, 0.0)
        s.update({"wall_s": round(elapsed, 1),
                  "machine_cpu_s": round(machine, 1),
                  "own_cpu_s": round(mine, 1),
                  "foreign_cpu_s": round(foreign, 1),
                  "foreign_cores": round(foreign / elapsed, 2)})
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(meta, f, indent=2)
os.replace(tmp, path)
PY
    local rc=$?
    if [ "$rc" != 0 ]; then
        echo "meta.json ($1) failed with status $rc — aborting the sweep" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------------------------
# one jepa-bench invocation
# ---------------------------------------------------------------------------------------------
run_bench() {  # $1 label  $2 ftype  $3 mode  $4 tag  $5 prec  $6... extra args
    local label="$1" ftype="$2" mode="$3" tag="$4" prec="$5"; shift 5
    local gguf="$GGUF_DIR/${label}-${ftype}.gguf"
    if [ ! -f "$gguf" ]; then
        echo "!!! MISSING: $gguf" >&2
        FAILURES=$((FAILURES + 1)); FAILED_CONFIGS+=("$label-$ftype $mode $tag (no such GGUF)")
        return
    fi
    local suffix="" json
    [ "$prec" = "default" ] || suffix="__prec$prec"
    json="$OUT/${label}-${ftype}__${mode}__${tag}__gpu${DEVICE}${suffix}.json"
    local cmd=("$BENCH" -m "$gguf" --mode "$mode" --label "$label" --ftype-label "$ftype"
               --gpu "$DEVICE" --repeat "$REPEAT" --warmup "$WARMUP" --json "$json")
    [ "$prec" = "default" ] || cmd+=(--gpu-prec "$prec")
    cmd+=("$@")
    if [ "$DRY" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
    echo "--- $label-$ftype  $mode  $tag  prec=$prec  (CUDA$DEVICE)" >&2
    if ! "${cmd[@]}"; then
        echo "!!! FAILED: $label-$ftype $mode $tag" >&2
        FAILURES=$((FAILURES + 1)); FAILED_CONFIGS+=("$label-$ftype $mode $tag")
    fi
}

FAILURES=0
FAILED_CONFIGS=()
[ "$DRY" = 1 ] || write_meta start

while IFS='|' read -r label mode arg dtypes prec batch; do
    case "${label// /}" in ""|\#*) continue ;; esac
    label="${label// /}"; mode="${mode// /}"; arg="${arg// /}"; dtypes="${dtypes// /}"
    prec="${prec// /}"; [ -n "$prec" ] || prec="default"
    batch="${batch// /}"; [ -n "$batch" ] || batch=1
    case "$mode" in
        encoder)                tag="T$arg"; extra=(--frames "$arg")
                                if [ "$batch" -gt 1 ]; then tag="${tag}B${batch}"; extra+=(--batch "$batch"); fi ;;
        head|predictor)         tag="T$arg"; extra=(--frames "$arg") ;;
        lewm-step)              tag="F$arg"; extra=() ;;
        lewm-rollout)           tag="K$arg"; extra=(--steps "$arg") ;;
        ac)                     tag="K$arg"; extra=(--batch "$arg") ;;
        # For the two rollout modes <arg> is K or K:H (the horizon; default 2, the planner's).
        ac-rollout)             _k="${arg%%:*}"; _h="${arg#*:}"; [ "$_h" = "$arg" ] && _h=2
                                tag="K${_k}H${_h}"; extra=(--batch "$_k" --steps "$_h") ;;
        ac-plan)                _k="${arg%%:*}"; _h="${arg#*:}"; [ "$_h" = "$arg" ] && _h=2
                                tag="K${_k}H${_h}"; extra=(--batch "$_k" --steps "$_h" --cem-steps 1) ;;
        *) echo "grid: unknown mode '$mode'" >&2; exit 1 ;;
    esac
    for ftype in ${dtypes//,/ }; do
        [ -z "$ONLY" ] || echo "$label $mode $tag $ftype" | grep -Eq "$ONLY" || continue
        run_bench "$label" "$ftype" "$mode" "$tag" "$prec" "${extra[@]}"
    done
done < "$GRID"

[ "$DRY" = 1 ] && exit 0
write_meta end
rm -f "$OUT/.times"

echo >&2
if [ "$FAILURES" -gt 0 ]; then
    echo "$FAILURES config(s) failed — the tables will be missing their rows:" >&2
    printf '  - %s\n' "${FAILED_CONFIGS[@]}" >&2
fi
echo "JSONs in $OUT" >&2

GEN_RC=0
# The CPU half of the document is not re-measured by a GPU sweep, so it is seeded from its own
# committed artifact and tmp/bench (usually empty here) is overlaid on top. Without that seed the
# generator would find no CPU runs at all and refuse to write anything.
gen=("$PYTHON" "$ROOT/scripts/gen_benchmarks_md.py" --bench-dir "$ROOT/tmp/bench"
     --merge-json "$ROOT/tests/results/benchmarks.json"
     --ref-dir "$ROOT/tests/fixtures/ref" --parity "$ROOT/docs/parity.md"
     --gpu-dir "$OUT" --results-json-gpu "$RESULTS_JSON" -o "$DOC")
[ -f "$TORCH_JSON" ] && gen+=(--torch-gpu "$TORCH_JSON")
[ -f "$TORCH_BATCHED_JSON" ] && gen+=(--torch-gpu-batched "$TORCH_BATCHED_JSON")
[ "$MERGE" = 1 ] && gen+=(--merge-json-gpu "$RESULTS_JSON")
[ "$GEN_DOC" = 1 ] || gen+=(--no-doc)
"${gen[@]}" || GEN_RC=$?
[ "$GEN_RC" = 0 ] || echo "gen_benchmarks_md.py failed with status $GEN_RC" >&2

[ "$FAILURES" = 0 ] && [ "$GEN_RC" = 0 ] || exit 1
exit 0
