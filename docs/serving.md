# Serving

`jepa-server` puts the engine behind HTTP. It is one process holding one model, and it is built out
of the same C API everything else in this repository calls — so a vector it returns is the vector
`jepa-embed` writes for the same input, and the `server` suite in `ctest` compares the two bit
pattern by bit pattern on every run.

```bash
build/jepa-server -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf --workers 4 -t 8
```

```
jepa-server 0.3.0: lejepa-vits16-pretrain-in1k-f16 (hfvit, f16, CPU) on http://127.0.0.1:8080
  workers 4 x 8 threads | max-batch 8 | max-wait 5 ms | local files refused
```

## The shape of the process

One `jepa_model` is loaded once and shared. Each worker owns one `jepa_context`, because a context
owns the graph arena and two threads must never be inside one — that is the
[thread contract](architecture.md#robustness) the header states and `tests/test-threads.cpp`
checks. `--workers` scales calls, `--threads` scales one call, and their product is what lands on
the machine: past the core count, doing both oversubscribes it. The default divides, giving each
worker `cores / workers` threads unless `--threads` says otherwise.

A request is cut into one task per input item and queued. A dispatcher takes the head task and every
compatible task behind it — same frame count, same preprocessed crop — into one `jepa_encode` call
of that many batch slices, up to `--max-batch` (default 8, capped at the engine's 32-slice graph
limit). A group that is not yet full waits up to `--max-wait-ms` (default 5) for company, but only
while there is nothing else to run: work that cannot join a group is never held up by it.

**Grouping changes the scheduling and not the arithmetic.** On the CPU a batched encode is
bit-identical to the same items encoded one at a time; on a CUDA device the two agree to ~1e-7
cosine but not bit-for-bit, because GEMM tiling varies with the batch shape. The
[`batch`](architecture.md#batching) suite gates that inside the engine and the `server` suite gates
it through the socket: a four-item request returns the rows four one-item requests return. Video
families run one clip per graph in the library, so for those the group stays at 1 unless
`--max-batch` is given explicitly — the call `jepa-embed` already makes, for the same reason.

Decoding and preprocessing run on the HTTP thread that received the request. Those entry points keep
no state and are re-entrant, so that work parallelises on its own and leaves the workers to the
graph. Frames become a clip through `tools/jepa-frames.h`, the same function `jepa-embed` calls,
which is what makes the two agree by construction rather than by inspection.

One process serves one device — a model and its contexts never straddle two backends — so two cards
are two processes, each with `--gpu N` and its own port.

## Endpoints

### `POST /v1/embeddings`

OpenAI-shaped. `input` is one item or an array of them; an item is a `{"b64": ...}` image, a
`{"path": ...}` or bare-string server-local path (only when the server was started with
`--allow-local-files`), or `{"frames": [...]}` for the frames of one clip. `pool` is `mean`, `cls`,
`lewm` or `none`, defaulting to what `jepa-embed` picks: the CLS token when the model has one, the
mean over patch tokens otherwise. `encoding_format` is `float` (default, an array of numbers) or
`base64` (the little-endian float32 bytes — smaller, and free of any question about decimal
round-trips; both carry the same bits).

```bash
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' -d "{
  \"model\": \"lejepa-vits16-pretrain-in1k-f16\",
  \"input\": [{\"b64\": \"$(base64 -w0 cat.jpg)\"}],
  \"pool\": \"cls\"
}"
```

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "dim": 384, "embedding": [0.0421, -0.318, "..."]}],
 "model": "lejepa-vits16-pretrain-in1k-f16",
 "usage": {"prompt_tokens": 197, "total_tokens": 197}}
```

`dim` is not part of the OpenAI shape and does no harm there; it is what lets a `base64` reader
reshape a `pool: "none"` vector, whose bytes are rows × dim.

### `POST /classify`

For a model carrying an attentive-pool head. `top_k` defaults to 5.

```json
{"input": {"frames": [{"b64": "..."}, {"b64": "..."}]}, "top_k": 3}
```

```json
{"object": "list",
 "data": [{"object": "classification", "index": 0, "predictions": [
     {"index": 6, "label": "Covering [something] with [something]", "probability": 0.1416, "logit": 5.127},
     {"index": 7, "label": "Digging [something] out of [something]", "probability": 0.1252, "logit": 5.005}]}],
 "model": "vjepa2-vitl-fpc16-256-ssv2-f16",
 "usage": {"prompt_tokens": 2048, "total_tokens": 2048}}
```

A model with no head answers 400 and says so, rather than returning something shaped like an answer.

### `POST /rollout`

The world models: V-JEPA 2-AC and LeWM. `context` is the observed frame(s), the last of which is
where planning starts; `actions` is `[K][H][action_dim]` candidate sequences (or `[H][action_dim]`
for one candidate); `goal` is the frame the energies are scored against. Without a goal the same
number is reported against the last observed frame, which reads as how far the rollout has drifted.
`state` is the seed pose for V-JEPA 2-AC, and `return_latents` adds the predicted latents as
base64 with their shape.

```json
{"context": [{"b64": "..."}], "goal": {"b64": "..."},
 "actions": [[[0.02, 0, 0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0, 0, 0]],
             [[-0.02, 0, 0, 0, 0, 0, 0], [-0.02, 0, 0, 0, 0, 0, 0]]]}
```

```json
{"object": "rollout", "model": "vjepa2-ac-vitg-f16",
 "energies": [[0.673374, 0.680428], [0.673164, 0.680289]],
 "best": {"index": 1, "energy": 0.680289, "actions": [-0.02, 0, 0, 0, 0, 0, 0, -0.02, 0, 0, 0, 0, 0, 0]},
 "n_candidates": 2, "horizon": 2,
 "usage": {"prompt_tokens": 512, "total_tokens": 512}}
```

`energies[c][h]` is Meta's L1 planning energy of candidate `c` after step `h`, the quantity a CEM
planner minimises; lower is closer to the goal. Adding a `plan` object runs that planner instead —
`samples`, `topk`, `cem_steps`, `horizon`, `maxnorm`, `gripper_clamp`, `seed`, mirroring
[`jepa_ac_plan`](api.md) — and returns the chosen action sequence with its energy per iteration:

```json
{"plan": {"actions": [-0.00814, 0.04248, -0.04298, 0, 0, 0, 0, "..."],
          "action_dim": 7, "horizon": 2,
          "energy_per_iteration": [0.678107, 0.678813]}}
```

A rollout runs whole on one worker and is never grouped with anything, so `--workers` is the number
of planners that can be in flight at once, and the memory table below is what each one costs.

### `GET /health`, `GET /v1/models`, `GET /metrics`

`/health` reports the loaded model, the backend and device it runs on, the pool settings and the
current queue depth. `/v1/models` is the OpenAI listing of the one model. `/metrics` is Prometheus
text: `jepa_requests_total` by endpoint and status, `jepa_items_total`,
`jepa_request_duration_seconds` as a histogram, `jepa_batch_size` as an exact histogram of the graph
sizes the dispatcher formed, and `jepa_queue_depth` / `jepa_requests_in_flight` / `jepa_workers` as
gauges.

## Flags

| flag | default | effect |
|---|---|---|
| `--host H` | `127.0.0.1` | bind address. Anything else is a deliberate exposure, and the server says so on startup |
| `--port N` | `8080` | `0` picks a free port and prints it |
| `--workers N` | `1` | worker threads, one `jepa_context` each |
| `-t, --threads N` | cores / workers | ggml threads inside one graph |
| `--max-batch N` | `8` | items folded into one encoder graph, capped at 32 |
| `--max-wait-ms N` | `5` | how long a partial group waits; `0` never waits |
| `--gpu [N]` | – | run on GPU device N (needs `-DJEPA_CUDA=ON`) |
| `--model-name S` | the file's base name | the id `/v1/models` reports |
| `--allow-local-files` | off | accept server-local paths, not only `{"b64": ...}` |
| `--max-body-mb N` | `32` | request body limit |
| `--max-items N` | `64` | items per request |
| `--max-frames N` | `128` | frames per item |

## Refusals

Every malformed request is a JSON error object, and none of them takes the process down:

```json
{"error": {"message": "\"b64\": not base64: byte 0x21", "type": "invalid_request_error", "code": null}}
```

`message` carries the engine's own `jepa_error_text()` where the failure came from a call, so a
refused encode explains itself. `type` follows the OpenAI vocabulary, so a client written against
that API can branch on it. A body that is not JSON, a JSON array where an object belongs, a
truncated base64 string, bytes that are not an image, an unknown model name, a local path without
`--allow-local-files` and a `"url"` input (the server fetches nothing, ever) are all 4xx, and the
`server` suite checks that `/health` still answers after each of them.

## Security

`jepa-server` binds `127.0.0.1` and has neither TLS nor authentication. It is a component to put
behind something that has both, not an edge. Binding it elsewhere takes an explicit `--host`, which
prints a warning naming what that means. Local paths are refused unless `--allow-local-files` says
otherwise, and even then the server only ever *reads* what a request names.

## The Python client

`jepa_cpp.client` is `urllib` and `json` and nothing else, so talking to a server adds no dependency
to the wheel.

```python
from jepa_cpp.client import Client, clip

c = Client("http://127.0.0.1:8080")
c.health()["backend"]                      # 'CPU'
c.embed("cat.jpg").shape                   # (1, 384)
c.embed(["a.jpg", "b.jpg"]).shape          # (2, 384) — one request, one graph
c.classify(clip(["f0.jpg", "f1.jpg"]))     # [[{'index': 6, 'label': ..., 'probability': ...}, ...]]
c.rollout("now.png", actions, goal="goal.png")["energies"]
```

A path or `bytes` is read here and sent base64, so a local caller needs no `--allow-local-files`. A
list is always a list of items; `clip()` marks the frames of one clip, which is the server's own
rule. A refused request raises `ServerError` — a `JepaError` — carrying the server's message and its
HTTP status.

## Docker

`docker/Dockerfile.cpu` and `docker/Dockerfile.cuda` are multi-stage builds that end in an image
running `jepa-server` and nothing else, with the model mounted rather than baked in. Both are built
with `JEPA_NATIVE=OFF`, because an image is a distributable artefact and the engine's arithmetic
follows the ISA: the CPU image's vector is bit-identical to a `JEPA_NATIVE=OFF` host build and
cosine 0.999999958 from the `-march=native` one. Build and run instructions, and what is inside each
image: [`docker/README.md`](https://github.com/aselimc/jepa.cpp/blob/main/docker/README.md).

## Measured

<!-- BEGIN generated by scripts/render_serving_md.py from tests/results/server-bench.json -->

### Environment

| | |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 7995WX 96-Cores, 192 threads |
| Build | c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, ggml `36da5713`, `GGML_LLAMAFILE=ON`, kernel 6.17.0-1032-oem |
| GPU | NVIDIA RTX 4500 Ada Generation, 24570 MiB, compute 8.9, 210.00 W board limit, driver 580.173.02 |
| Client | Python 3.12.13, `jepa_cpp.client` over `urllib` |
| Commit | `f426f1c6`, jepa.cpp 0.3.0 |
| Date | 2026-09-02 01:59 UTC, idle box, one measured pass at a time |

### CPU — LeJEPA ViT-S/16 f16, 32 threads, 1 worker

| `--max-batch` | clients | req/s | p50 ms | p95 ms | p99 ms | mean batch | graphs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 44.3 | 23 | 23 | 24 | 1.0 | 512 |
| 1 | 8 | 74.8 | 106 | 114 | 138 | 1.0 | 512 |
| 1 | 32 | 75.3 | 424 | 430 | 432 | 1.0 | 512 |
| 8 | 1 | 35.8 | 28 | 29 | 29 | 1.0 | 512 |
| 8 | 8 | 92.4 | 86 | 90 | 122 | 8.0 | 64 |
| 8 | 32 | 106.0 | 297 | 306 | 346 | 8.0 | 64 |
| 32 | 1 | 35.9 | 28 | 29 | 30 | 1.0 | 512 |
| 32 | 8 | 90.8 | 86 | 95 | 126 | 4.3 | 118 |
| 32 | 32 | **106.8** | 290 | 362 | 442 | 16.0 | 32 |

512 requests per cell, 3 warmup, one 640×480 COCO JPEG each. The `GET /health` control ran at 2890–5232 req/s over the same cells, against a best measured 106.8 req/s of real work — a factor of 27 of headroom, so these cells are measuring the server.

### GPU — I-JEPA ViT-H/14 f16, device 1, 1 worker

| `--max-batch` | clients | req/s | p50 ms | p95 ms | p99 ms | mean batch | graphs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 43.9 | 23 | 24 | 25 | 1.0 | 384 |
| 1 | 8 | 61.2 | 130 | 132 | 132 | 1.0 | 384 |
| 1 | 32 | 61.0 | 523 | 525 | 525 | 1.0 | 384 |
| 8 | 1 | 35.7 | 28 | 29 | 30 | 1.0 | 384 |
| 8 | 8 | 78.0 | 102 | 106 | 107 | 8.0 | 48 |
| 8 | 32 | **87.2** | 364 | 369 | 390 | 8.0 | 48 |
| 32 | 1 | 35.2 | 28 | 29 | 34 | 1.0 | 384 |
| 32 | 8 | 73.9 | 108 | 111 | 115 | 4.9 | 78 |
| 32 | 32 | 80.2 | 397 | 406 | 413 | 16.0 | 24 |

384 requests per cell, 3 warmup, one 640×480 COCO JPEG each. The `GET /health` control ran at 2963–5224 req/s over the same cells, against a best measured 87.2 req/s of real work — a factor of 34 of headroom, so these cells are measuring the server.

### Device memory per concurrent planner

| concurrent planners | peak device MiB | added per planner | rollout/s | p50 ms | p99 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 3105 | – | 1.98 | 506 | 511 |
| 2 | 3451 | 346 | 2.20 | 908 | 945 |
| 4 | 4145 | 347 | 2.16 | 1816 | 2234 |

V-JEPA 2-AC ViT-g f16 on device 1, 16 candidates × 2 steps per request, 2519 MiB of weights. The card holds 15 MiB idle, 2737 MiB with the model resident and nothing running, and 3105 MiB once one planner has built its graphs.

<!-- END generated -->

*Source: `tests/results/server-bench.json`, written by `scripts/bench_server.py`, one server started
per cell on an idle box. Latencies are what the client saw, so queueing, the loopback connection
`urllib` opens per request (it has no keep-alive) and JSON on both sides are inside them. The
`mean batch` row is read off `/metrics` before and after each measured window: it is the batching
that happened, not the batching `--max-batch` allowed.*

### What the numbers say

**Batching pays under load and costs at idle.** At 32 concurrent clients it is worth 1.42× on the
CPU (75.3 → 106.8 req/s) and 1.43× on the GPU (61.0 → 87.2), and it takes p50 down with it — 424 →
290 ms and 523 → 364 ms — because a queue that drains in groups drains faster. At one client it is a
pure loss: the mean batch is 1.0, because there is nothing to group with, and every request pays
`--max-wait-ms` waiting for company that never arrives — which is exactly the 5 ms the p50 column
moves by, on both backends. `--max-wait-ms 0` is the right setting for a latency-sensitive,
low-concurrency caller; the default is the right one for a saturated queue.

**`--max-batch 32` buys nothing over 8 here.** It forms the larger groups the `mean batch` column
shows — 16 instead of 8 at 32 clients — and returns the same throughput on the CPU (106.8 against
106.0) and less on the GPU (80.2 against 87.2), with a worse p99 on both. Batching's return flattens
once a graph is wide enough to keep the cores or the SMs busy, and past that a bigger group only
adds queueing to the requests at the front of it. Eight is the default for that reason.

**More concurrent planners buy memory and latency, not throughput.** A second planner takes rollout
throughput from 1.98 to 2.20 per second and p50 from 506 to 908 ms; a fourth gives 2.16 — no more
than one — and 1816 ms, exactly double again. One ViT-g planner already saturates the card, so what
`--workers` above 1 is for on a planning server is keeping a second request from waiting behind a
first. Each one costs a flat ~347 MiB of device memory on top of the 2519 MiB of shared weights.

## Reproduce

```bash
cmake --build build && cmake --build build-cuda       # jepa-server on both backends
ctest --test-dir build -R server --output-on-failure  # the bit-identity gate, ~1 s

# the whole load test: 9 CPU cells, 9 GPU cells, 3 planner cells
scripts/bench_server.py --out tests/results/server-bench.json \
    --threads 32 --workers 1 --requests 512 --gpu-requests 384 --planner-requests 8 \
    --device 1 --note "idle box, one measured pass at a time"
scripts/render_serving_md.py --write docs/serving.md   # the tables above
scripts/render_serving_md.py --check                   # exit 1 if they are stale
```

The load test needs `jepa_cpp.client` importable — `pip install ./python` from a recursive clone, or
`$JEPA_CPP_LIB` pointed at a built `libjepa.so`.
