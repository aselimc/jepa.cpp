#!/usr/bin/env python3
"""Load test for ``jepa-server``: throughput and latency by concurrency and by ``--max-batch``.

    scripts/bench_server.py --out tests/results/server-bench.json --note "idle box"
    scripts/bench_server.py --skip-gpu --skip-planner --requests 40   # a quick CPU-only pass

Standard library plus :mod:`jepa_cpp.client`, which is itself standard library — no third-party HTTP
stack is involved, so what this measures is what a caller running the shipped client would get.

What is measured, and what that costs
-------------------------------------
For every (backend, ``--max-batch``, concurrency) cell the script starts a fresh server, warms it,
then has ``concurrency`` threads each send ``requests / concurrency`` single-image
``/v1/embeddings`` requests back to back. Reported per cell: requests per second over the measured
window, and the p50/p95/p99 of the per-request wall time as the *client* saw it — queueing,
connection setup and JSON on both sides included.

Two costs are inside those numbers on purpose, because they are inside a real caller's numbers too:

* :mod:`urllib` opens a new connection per request — it has no keep-alive — so every request pays a
  loopback TCP handshake;
* the load generator is Python threads under one GIL. The ``health`` control cell measures exactly
  that: the same concurrency against ``GET /health``, which does no work at all. Where a cell's
  throughput approaches its control, the harness is the limit and not the server, and the artifact
  records both so the reader can tell which.

Images are base64-encoded once and reused, so the per-request client work is a ``json.dumps`` of an
already-built string and a socket round trip.

The planner section is separate: it starts one V-JEPA 2-AC server per worker count and samples
``nvidia-smi`` while that many ``/rollout`` requests run concurrently, which is what turns "VRAM per
additional concurrent planner" into a number.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

REPO = pathlib.Path(__file__).resolve().parents[1]

try:
    from jepa_cpp.client import Client, ServerError
except ImportError:  # a source checkout with the package not installed
    sys.path.insert(0, str(REPO / "python" / "src"))
    try:
        from jepa_cpp.client import Client, ServerError
    except Exception as e:  # pragma: no cover - the message is the point
        raise SystemExit(
            f"cannot import jepa_cpp.client ({e}).\n"
            "Install the bindings — `pip install ./python` from a recursive clone — or point\n"
            "$JEPA_CPP_LIB at a built libjepa.so."
        ) from None


# --- the box ---------------------------------------------------------------------------------
def sh(*cmd: str, default: str = "") -> str:
    """One command's first line of stdout, or ``default`` if it is not there or fails."""
    if not shutil.which(cmd[0]):
        return default
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return default
    return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else default


def cpu_model() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def tctl_celsius() -> float | None:
    """The CPU package temperature from k10temp, or None where that sensor is not there."""
    for hwmon in sorted(pathlib.Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (hwmon / "name").read_text().strip() != "k10temp":
                continue
            for label in sorted(hwmon.glob("temp*_label")):
                if label.read_text().strip() == "Tctl":
                    raw = (label.parent / label.name.replace("_label", "_input")).read_text()
                    return int(raw) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def gpu_memory_mib(device: int) -> int | None:
    """``nvidia-smi``'s used memory for one device, in MiB."""
    if not shutil.which("nvidia-smi"):
        return None
    out = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def gpu_box(device: int) -> dict:
    fields = "name,memory.total,compute_cap,power.limit,driver_version"
    line = sh("nvidia-smi", f"--id={device}", f"--query-gpu={fields}", "--format=csv,noheader")
    parts = [p.strip() for p in line.split(",")] if line else []
    keys = ["device", "device_memory", "compute_cap", "power_limit", "driver"]
    return dict(zip(keys, parts)) if len(parts) == len(keys) else {}


def box(threads: int, device: int) -> dict:
    cache = REPO / "build" / "CMakeCache.txt"
    llamafile = "unknown"
    if cache.is_file():
        m = re.search(r"^GGML_LLAMAFILE:BOOL=(\w+)$", cache.read_text(), re.M)
        if m:
            llamafile = m.group(1)
    return {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "kernel": platform.release(),
        "compiler": sh("c++", "--version", default="unknown"),
        "ggml_commit": sh("git", "-C", str(REPO / "ggml"), "rev-parse", "--short=8", "HEAD",
                          default="unknown"),
        "git_commit": sh("git", "-C", str(REPO), "rev-parse", "--short=8", "HEAD", default="unknown"),
        "ggml_llamafile": llamafile,
        "threads": threads,
        "python": platform.python_version(),
        **gpu_box(device),
    }


def cool_down(ceiling: float = 88.0, resume: float = 85.0, limit_s: float = 600.0) -> float | None:
    """Wait out a hot package before starting a measured pass. Returns the temperature seen."""
    t = tctl_celsius()
    if t is None or t < ceiling:
        return t
    deadline = time.monotonic() + limit_s
    print(f"  Tctl {t:.0f} C — waiting for {resume:.0f} C", file=sys.stderr)
    while time.monotonic() < deadline:
        time.sleep(5)
        t = tctl_celsius()
        if t is None or t <= resume:
            break
    return t


# --- the server under test --------------------------------------------------------------------
class Server:
    """One ``jepa-server`` child on a port it picked, with a client pointed at it."""

    def __init__(self, exe: pathlib.Path, model: pathlib.Path, args: list[str], verbose: bool):
        self.cmd = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", "0", *args]
        self.verbose = verbose
        self.proc: subprocess.Popen | None = None
        self.client: Client | None = None
        self.port = 0

    def __enter__(self) -> Client:
        if self.verbose:
            print("  $ " + " ".join(self.cmd), file=sys.stderr)
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"jepa-server exited with {self.proc.returncode}")
                continue
            m = re.search(r"http://127\.0\.0\.1:(\d+)", line)
            if m:
                self.port = int(m.group(1))
                self.client = Client(f"http://127.0.0.1:{self.port}", timeout=1800.0)
                return self.client
        raise RuntimeError("jepa-server never printed its port")

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self.proc.kill()
                self.proc.wait(timeout=30)
            self.proc.stdout.close()
        return False


def free_port_exists() -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return True
    finally:
        s.close()


# --- the load generator ------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float:
    """The q-th percentile by nearest rank, which needs no interpolation story."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[k]


def drive(call, concurrency: int, requests: int) -> dict:
    """``requests`` calls of ``call()`` spread over ``concurrency`` threads, back to back."""
    per = max(1, requests // concurrency)
    total = per * concurrency
    latencies: list[list[float]] = [[] for _ in range(concurrency)]
    errors: list[str] = []
    start_barrier = threading.Barrier(concurrency + 1)

    def worker(slot: int):
        start_barrier.wait()
        for _ in range(per):
            t0 = time.perf_counter()
            try:
                call()
            except Exception as e:  # noqa: BLE001 - a failed request is a measurement, not a crash
                errors.append(f"{type(e).__name__}: {e}")
                continue
            latencies[slot].append(time.perf_counter() - t0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    start_barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    flat = [x for slot in latencies for x in slot]
    return {
        "concurrency": concurrency,
        "requests": total,
        "ok": len(flat),
        "errors": len(errors),
        "error_sample": errors[:3],
        "wall_s": round(wall, 4),
        "req_per_s": round(len(flat) / wall, 2) if wall > 0 else 0.0,
        "p50_ms": round(1000 * percentile(flat, 50), 3) if flat else None,
        "p95_ms": round(1000 * percentile(flat, 95), 3) if flat else None,
        "p99_ms": round(1000 * percentile(flat, 99), 3) if flat else None,
        "mean_ms": round(1000 * statistics.fmean(flat), 3) if flat else None,
        "min_ms": round(1000 * min(flat), 3) if flat else None,
    }


def batch_counters(cli) -> tuple[int, int]:
    """(items folded into graphs, graphs) out of /metrics — how much batching actually happened."""
    total = graphs = 0
    for line in cli.metrics().splitlines():
        if line.startswith("jepa_batch_size_sum "):
            total = int(float(line.split()[1]))
        elif line.startswith("jepa_batch_size_count "):
            graphs = int(float(line.split()[1]))
    return total, graphs


def embed_sweep(exe, model, server_args, payload, concurrencies, requests, warmup, verbose, label):
    """One server per cell, warmed, then driven at each concurrency."""
    rows = []
    for c in concurrencies:
        cool_down()
        with Server(exe, model, server_args, verbose) as cli:
            for _ in range(warmup):
                cli.embed(payload)
            items0, graphs0 = batch_counters(cli)
            row = drive(lambda: cli.embed(payload), c, requests)
            items1, graphs1 = batch_counters(cli)
            control = drive(cli.health, c, max(concurrencies) * 4)
            row["health_req_per_s"] = control["req_per_s"]
            row["health_p50_ms"] = control["p50_ms"]
            # The measured window's own graphs, warmup excluded: this is the batching that happened,
            # not the batching that was allowed.
            row["graphs"] = graphs1 - graphs0
            row["mean_batch"] = (round((items1 - items0) / (graphs1 - graphs0), 2)
                                 if graphs1 > graphs0 else None)
            row.update(label)
            row["tctl_c"] = tctl_celsius()
            rows.append(row)
            print(
                f"  {label['backend']:<6} batch<={label['max_batch']:<2} c={c:<3} "
                f"{row['req_per_s']:>9.2f} req/s   p50 {row['p50_ms']:>8.2f} ms  "
                f"p99 {row['p99_ms']:>9.2f} ms   mean batch {row['mean_batch']}   "
                f"(control {control['req_per_s']:.0f} req/s)",
                file=sys.stderr,
            )
    return rows


# --- the planner section ------------------------------------------------------------------------
def planner_vram(exe, model, device, workers_list, k, horizon, image_b64, goal_b64, verbose,
                 per_worker):
    """Peak device memory while ``w`` /rollout requests of K candidates run at once."""
    rows = []
    baseline_idle = gpu_memory_mib(device)
    for w in workers_list:
        cool_down()
        args = ["--gpu", str(device), "--workers", str(w), "--max-items", "8"]
        with Server(exe, model, args, verbose) as cli:
            health = cli.health()
            actions = [[[0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * horizon] * k
            loaded = gpu_memory_mib(device)
            cli.rollout({"b64": image_b64}, actions, goal={"b64": goal_b64})  # warm the graphs
            warm = gpu_memory_mib(device)

            peak = [warm or 0]
            stop = threading.Event()

            def sample():
                while not stop.is_set():
                    m = gpu_memory_mib(device)
                    if m is not None:
                        peak[0] = max(peak[0], m)
                    time.sleep(0.05)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            row = drive(
                lambda: cli.rollout({"b64": image_b64}, actions, goal={"b64": goal_b64}),
                w,
                w * per_worker,
            )
            stop.set()
            sampler.join(timeout=5)

            row.update(
                {
                    "workers": w,
                    "candidates": k,
                    "horizon": horizon,
                    "device": device,
                    "backend": health["backend"],
                    "weights_mib": round(health["weights_mib"], 1),
                    "idle_mib": baseline_idle,
                    "loaded_mib": loaded,
                    "warm_one_planner_mib": warm,
                    "peak_mib": peak[0],
                }
            )
            rows.append(row)
            print(
                f"  planners={w:<2} peak {peak[0]:>6} MiB   "
                f"{row['req_per_s']:.2f} rollout/s  p50 {row['p50_ms']:.0f} ms",
                file=sys.stderr,
            )
    return rows


# --- main -----------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "tests/results/server-bench.json"))
    ap.add_argument("--server", default=str(REPO / "build/jepa-server"))
    ap.add_argument("--server-cuda", default=str(REPO / "build-cuda/jepa-server"))
    ap.add_argument("--cpu-model", default=str(REPO / "models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf"))
    ap.add_argument("--gpu-model", default=str(REPO / "models/gguf/ijepa_vith14_1k-f16.gguf"))
    ap.add_argument("--ac-model", default=str(REPO / "models/gguf/vjepa2-ac-vitg-f16.gguf"))
    ap.add_argument("--image", default=str(REPO / "tests/fixtures/media/coco_000000000139.jpg"))
    ap.add_argument("--goal", default=str(REPO / "tests/fixtures/media/coco_000000000285.jpg"))
    ap.add_argument("--device", type=int, default=1, help="GPU device index (default 1)")
    ap.add_argument("--threads", type=int, default=32, help="server --threads on the CPU (default 32)")
    ap.add_argument("--workers", type=int, default=1, help="server --workers (default 1)")
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--max-batch", default="1,8,32")
    ap.add_argument("--requests", type=int, default=128)
    ap.add_argument("--gpu-requests", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--planner-workers", default="1,2,4")
    ap.add_argument("--planner-candidates", type=int, default=16)
    ap.add_argument("--planner-horizon", type=int, default=2)
    ap.add_argument("--planner-requests", type=int, default=8,
                    help="rollout requests per concurrent planner (default 8)")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-planner", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    concurrencies = [int(x) for x in args.concurrency.split(",") if x]
    batches = [int(x) for x in args.max_batch.split(",") if x]
    planner_workers = [int(x) for x in args.planner_workers.split(",") if x]

    exe_cpu = pathlib.Path(args.server)
    exe_gpu = pathlib.Path(args.server_cuda)
    image = pathlib.Path(args.image)
    if not image.is_file():
        raise SystemExit(f"{image} is not there — run scripts/download_fixtures.sh")
    payload = {"b64": base64.b64encode(image.read_bytes()).decode("ascii")}
    goal_b64 = base64.b64encode(pathlib.Path(args.goal).read_bytes()).decode("ascii")
    free_port_exists()

    started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    results = {
        "task": "jepa-server load test: /v1/embeddings throughput and latency by concurrency and "
                "--max-batch, and device memory per concurrent /rollout planner",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_by": "scripts/bench_server.py",
        "jepa_version": sh(str(REPO / "build/jepa-info"), "--version", default="unknown"),
        "protocol": {
            "req_per_s": "completed requests divided by the wall time of the measured window, which "
                         "starts when every client thread is at the barrier",
            "p50_ms/p95_ms/p99_ms": "percentiles by nearest rank of the per-request wall time the "
                                    "CLIENT saw: queueing, the loopback connection urllib opens per "
                                    "request, and JSON on both sides are all inside them",
            "health_req_per_s": "the same concurrency against GET /health, which does no inference. "
                                "It bounds what this Python-threaded harness can drive at all; a "
                                "cell close to its control is measuring the harness",
            "warmup": "requests served and discarded before the window, so graph allocation and the "
                      "first-call costs are not in the measurement",
            "payload": "one 640x480 COCO JPEG, base64-encoded once and reused, one item per request",
            "mean_batch": "items per encoder graph over the measured window, read off /metrics before "
                          "and after it — the batching that HAPPENED, not the batching --max-batch "
                          "allowed",
            "graphs": "encoder graphs the window ran",
            "peak_mib": "the largest nvidia-smi memory.used sampled at 20 Hz while the rollouts ran "
                        "— whole-device, so it includes anything else on the card",
            "batching": "grouping changes scheduling, not arithmetic: on the CPU a batched encode is "
                        "bit-identical to the same items encoded one at a time (tests/test-batch, "
                        "ctest `server`), and on CUDA the two agree to ~1e-7 cosine but not "
                        "bit-for-bit because GEMM tiling varies with the batch shape",
        },
        "box": box(args.threads, args.device),
        "session": {
            "start_utc": started,
            "note": args.note,
            "loadavg_start": pathlib.Path("/proc/loadavg").read_text().split()[0],
            "concurrency": concurrencies,
            "max_batch": batches,
            "requests": args.requests,
            "warmup": args.warmup,
        },
        "commands": {
            "cpu": f"scripts/bench_server.py --skip-gpu --skip-planner --threads {args.threads} "
                   f"--workers {args.workers} --requests {args.requests}",
            "gpu": f"scripts/bench_server.py --skip-cpu --skip-planner --device {args.device} "
                   f"--gpu-requests {args.gpu_requests}",
            "planner": f"scripts/bench_server.py --skip-cpu --skip-gpu --device {args.device} "
                       f"--planner-candidates {args.planner_candidates}",
        },
        "cpu": [],
        "gpu": [],
        "planner": [],
        "skipped": [],
    }

    if not args.skip_cpu:
        model = pathlib.Path(args.cpu_model)
        if not exe_cpu.is_file() or not model.is_file():
            results["skipped"].append(f"cpu: {exe_cpu} or {model} is not there")
        else:
            print(f"CPU: {model.name}, {args.threads} threads, {args.workers} worker(s)", file=sys.stderr)
            for b in batches:
                server_args = ["--threads", str(args.threads), "--workers", str(args.workers),
                               "--max-batch", str(b), "--max-wait-ms", "5"]
                try:
                    results["cpu"] += embed_sweep(
                        exe_cpu, model, server_args, payload, concurrencies, args.requests,
                        args.warmup, args.verbose,
                        {"backend": "CPU", "model": model.stem, "max_batch": b,
                         "threads": args.threads, "workers": args.workers},
                    )
                except (RuntimeError, ServerError, OSError) as e:
                    # A cell that will not run is a hole in the table, not a lost sweep.
                    results["skipped"].append(f"cpu max_batch={b}: {e}")
                    print(f"  skipped cpu max_batch={b}: {e}", file=sys.stderr)

    if not args.skip_gpu:
        model = pathlib.Path(args.gpu_model)
        if not exe_gpu.is_file() or not model.is_file():
            results["skipped"].append(f"gpu: {exe_gpu} or {model} is not there")
        else:
            print(f"GPU {args.device}: {model.name}", file=sys.stderr)
            for b in batches:
                server_args = ["--gpu", str(args.device), "--workers", str(args.workers),
                               "--max-batch", str(b), "--max-wait-ms", "5"]
                try:
                    results["gpu"] += embed_sweep(
                        exe_gpu, model, server_args, payload, concurrencies, args.gpu_requests,
                        args.warmup, args.verbose,
                        {"backend": f"CUDA{args.device}", "model": model.stem, "max_batch": b,
                         "threads": 0, "workers": args.workers},
                    )
                except (RuntimeError, ServerError, OSError) as e:
                    results["skipped"].append(f"gpu max_batch={b}: {e}")
                    print(f"  skipped gpu max_batch={b}: {e}", file=sys.stderr)

    if not args.skip_planner:
        model = pathlib.Path(args.ac_model)
        if not exe_gpu.is_file() or not model.is_file():
            results["skipped"].append(f"planner: {exe_gpu} or {model} is not there")
        else:
            print(f"planner VRAM on GPU {args.device}: {model.name}, K={args.planner_candidates}",
                  file=sys.stderr)
            try:
                results["planner"] = planner_vram(
                    exe_gpu, model, args.device, planner_workers, args.planner_candidates,
                    args.planner_horizon, payload["b64"], goal_b64, args.verbose,
                    args.planner_requests,
                )
            except (RuntimeError, ServerError, OSError) as e:
                results["skipped"].append(f"planner: {e}")
                print(f"  skipped planner: {e}", file=sys.stderr)

    results["session"]["end_utc"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    results["session"]["loadavg_end"] = pathlib.Path("/proc/loadavg").read_text().split()[0]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    for note in results["skipped"]:
        print(f"  skipped — {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
