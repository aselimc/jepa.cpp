# Changelog

Notable changes to jepa.cpp, newest first. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the version numbers follow
[semantic versioning](https://semver.org/spec/v2.0.0.html) as narrowed by
[docs/releases.md](docs/releases.md), which also states what the C API and the GGUF schema promise
across releases.

## [Unreleased]

### Added

- **V-JEPA 2-AC, the action-conditioned world model** (`jepa.pred.kind = "ac"`). One GGUF bundle
  carries the frozen ViT-g/16 encoder of Meta's `vjepa2-ac-vitg.pt` and its 24-layer, 1024-dim
  predictor, which takes a 7-d end-effector action and a 7-d pose per frame and predicts the next
  frame's encoder latents. Attention is block-causal over whole frames, so one call with *T* frames
  also answers every shorter prefix. New entry points, all append-only:
  `jepa_ac_predict` / `jepa_ac_predict_all`, `jepa_ac_rollout` (K candidate action sequences over H
  steps, batched on the graph's batch axis — one graph per step, not per candidate),
  `jepa_ac_next_state` (Meta's `compute_new_pose`), `jepa_ac_energy` (the L1 planning energy a CEM
  planner minimises), `jepa_ac_normalize` and the `jepa_ac_*` accessors. `jepa-worldmodel --ac`
  encodes a frame, rolls candidates out and prints their per-step energy against a `--goal` frame;
  `jepa-bench` gains `--mode ac` and `--mode ac-rollout`.
  Against Meta's own modules on their Franka demo trajectory, f32 on the CPU: every prediction and
  every rollout step at cosine **1.0000000**, the energy to 7.5e-07, the same candidate ranking, and
  **K batched == K sequential, bit-identical**. On CUDA at f32/f16 the worst rollout row is 0.997.
  `docs/parity.md` "V-JEPA 2-AC" has the full dtype × backend matrix — including the finding that a
  q4_k rollout on a GPU **misranks the candidates**, so plan with f16.
- **V-JEPA 2 ViT-g/16** (`facebook/vjepa2-vitg-fpc64-256`), the first 1.03 B-parameter, 22-head,
  ffn-48/11 encoder here. The converter and the graph were already metadata-driven and needed no
  change; what is new is the golden dumps, the parity fixtures and the honest record of where a
  40-layer model sits against bars fitted on ViT-L (`docs/parity.md` "V-JEPA 2 ViT-g/16"). The numpy
  spec run on the GGUF's own weights matches HF at cosine **1.0000000**, rel 2.8e-05.
- `scripts/jepa_convert/vjepa2_ac.py`, `scripts/jepa_convert/selftest.py::ac_predictor_forward` (the
  executable spec), `scripts/torch_ac_baseline.py` (the PyTorch-on-the-same-card timing baseline) and
  `dump_reference.py --model vjepa2-ac-vitg | vjepa2-vitg-fpc64-256`.

### Changed

- `jepa_rope3d_apply` accepts a batched input (`ne[3] > 1`); the cos/sin tables broadcast over it.
  Nothing changes for the existing single-sequence callers.
- Eight new loader checks for the AC keys and nine new `jepa_ac_*` argument guards in `test-errors`
  (118 cases, up from 101), forged from a complete tiny AC model.

## [0.1.1] — 2026-09-01

### Added

- **Hardening of the untrusted-input paths** — a GGUF is a download and an image is bytes off a
  network, and both are now range-checked before anything is allocated. The loader validates every
  `jepa.*` integer and float against a documented bound, refuses an unknown activation or attention
  mode, checks that each tensor's bytes are actually inside the file, that its dtype is one the
  engine can compute with, and that every weight, bias and vector of every block has the shape and
  the f32-ness the graph will assume — so a file that promises a predictor or a head it does not
  carry is refused at load rather than aborting on first use. Call-time guards cover non-positive
  and overflowing shapes, ids off the predictor's grid, and the `$JEPA_MAX_GRAPH_MIB` ceiling, which
  now also applies to the masked predictor and the LeWM rollout. The image pipeline caps the
  intermediate of the shortest-edge resize at 64 megapixels. Nothing is clamped silently: every
  refusal names the key or tensor, on `stderr` and in `jepa_error_text()`.
  [docs/architecture.md "Robustness"](docs/architecture.md#robustness).
- **Thread-safety contract**, stated in `include/jepa.h` and checked by the new `threads` ctest
  suite: a `jepa_model` is immutable after load and shareable across threads, a `jepa_context`
  belongs to one thread, `jepa_error_text()` is thread-local, preprocessing is re-entrant, and
  concurrent encodes through per-thread contexts are bit-identical to the same work run serially.
- **`errors` and `threads` ctest suites** (`tests/test-errors.cpp`, `tests/test-threads.cpp`), and a
  **GGUF loader fuzz target** (`tests/fuzz/fuzz-gguf-load.cpp`, `-DJEPA_FUZZ=ON`, off by default)
  with a corpus generator (`scripts/make_fuzz_corpus.py`). The first two need no weights, so the
  ASAN+UBSAN CI job runs them; the fuzz target is build-only in CI.

### Fixed

- Fourteen input classes that crashed, hung or over-allocated the loader or a tool now return an
  error: a zero head count (SIGFPE), an integer-typed matmul weight (a null kernel pointer),
  block vectors of the wrong length, an f16 table in the f32 graph, tensor bytes beyond the end of
  the file, an unknown activation, a promised predictor or head with no tensors, an odd head width
  on a 3-D RoPE family, mask or frame counts past their tables, RoPE interpolation without a
  reference grid, signed overflow in the rollout and batch size arithmetic, a degenerate image
  aspect ratio, and a 32-bit seek that made GGUFs over 2 GB unloadable on Windows. Every case has a
  regression test in `tests/test-errors.cpp`.
- A data race on the engine's one-shot warning flags (now atomic).

## [0.1.0] — 2026-09-01

First release. A single C/C++ engine on [ggml](https://github.com/ggml-org/ggml) that runs seven
JEPA checkpoints from six families on a CPU, from one GGUF file per model, with no Python at
inference time.

### Added

- **Encoder engine** — one shared pre-LayerNorm ViT graph parameterised entirely from GGUF metadata:
  2-D patch and 3-D tubelet tokenizers, sincos position tables (bicubic/trilinear-interpolated off
  the training grid) and 3-D RoPE including V-JEPA 2's tiled frequency layout, optional CLS token
  and registers, fused or split QKV, layer scale, flash attention (`ggml_flash_attn_ext`) with a
  naive fallback, and full or block-causal attention masks. A new checkpoint of a known family needs
  no C++ change.
- **Model families** — I-JEPA (ViT-H/14), LeJEPA (ViT-S/16), LeWorldModel (Push-T), V-JEPA 2
  (ViT-L/16 fpc64 and the ViT-L SSv2 classifier), V-JEPA 2.1 (ViT-B/16 @384, image *and* video),
  and LeVJEPA (ViT-L/16, block-causal, CLS-pooled).
- **Heads and predictors** — the V-JEPA 2 / 2.1 masked predictor (`jepa_predict`, `jepa_predict_ex`,
  `jepa_predict_mod`), the LeWorldModel action-conditioned world model (`jepa_lewm_project`,
  `jepa_lewm_predict`, `jepa_lewm_rollout`) and the attentive-pool classifier with its 174
  Something-Something-v2 labels carried inside the file (`jepa_head`, `jepa_head_ex`).
- **Preprocessing** — a bit-exact port of torchvision's antialiased uint8 resize plus centre crop and
  per-model normalisation, so the tensor entering the network is byte-identical to the reference
  pipeline's.
- **C API** — one header, `include/jepa.h`: opaque handles, plain structs, no C++ types.
  `jepa_version()` reports the release version. The generated reference is
  [docs/api.md](docs/api.md).
- **Tools** — `jepa-embed` (images, clips and batches to feature vectors, `--batch`, `--frames-npy`,
  `--frames-list`), `jepa-classify`, `jepa-worldmodel`, `jepa-quantize`, `jepa-bench` and
  `jepa-info`.
- **Quantization** — `f16`, `q8_0`, `q6_k`, `q5_k`, `q5_1`, `q5_0`, `q4_1`, `q4_k`, `q4_0`, written
  in two passes so peak memory is one tensor rather than the whole model, with per-type accuracy
  measured on real datasets rather than assumed.
- **Batched image encoding** — up to 32 image items through one ggml graph, bit-identical to the
  per-item path on the CPU, worth 1.7–2.1× of encoder time.
- **GPU backend (optional)** — `-DJEPA_CUDA=ON` adds `--gpu [N]` to the tools and `$JEPA_DEVICE`;
  every graph is validated against the backend's own `supports_op` before it runs, and
  `GGML_PREC_F32` accumulation is the default on a device.
- **Memory guards** — `$JEPA_MAX_GRAPH_MIB` refuses a clip whose attention mask would not fit,
  instead of allocating it.
- **Converter** — `scripts/convert.py` writes schema v1 GGUF for all six families from Hugging Face
  and Meta checkpoints, carrying dimensions, positional scheme, preprocessing recipe, class labels
  and the checkpoint's licence into the file.
- **Tests** — ten `ctest` suites: five PyTorch-golden-dump replays (`parity-*`, `predictor-*`),
  `batch` (batched vs per-item bit-exactness), `ops` (3-D RoPE vectors and the block-causal mask),
  `attn` (flash vs naive against a double-precision reference) and `backend` (GPU graph validation
  and CPU/GPU agreement). The suites that need weights register themselves only when the files are
  present, so a bare checkout still runs `ops` and `attn`.
- **Measured results** — every number in the documentation traces to a committed artifact under
  `tests/results/`: encoder latency and memory (CPU and CUDA), Imagenette and UCF-101 k-NN accuracy,
  the full 24 777-clip Something-Something-v2 validation split (72.39 % top-1, the same as PyTorch),
  and per-dtype quantization accuracy.
- **Documentation** — the [MkDocs site](https://aselimc.github.io/jepa.cpp/) with getting started,
  architecture, the GGUF schema, performance, accuracy, the generated C API page and the raw
  measurement reports.
- **CI** — `.github/workflows/ci.yml` builds and tests on Ubuntu 22.04 and 24.04, builds and tests
  on macOS arm64 (with a Metal build), builds on Windows/MSVC, runs an ASAN+UBSAN build, and gates
  the generated documentation artifacts. `.github/workflows/release.yml` publishes the Linux
  x86-64 CPU archive from a `v*` tag.


- **Python bindings** — `pip install jepa-cpp` (import `jepa_cpp`), a thin wrapper over
  `include/jepa.h` with numpy on both ends: `Model.encode` / `pool` / `classify` / `predict` /
  `lewm_rollout`, the file's metadata as properties, and `jepa_cpp._api` as the unwrapped header.
  Preprocessing and every floating-point operation stay in C, so on a CPU the bindings reproduce
  `jepa-embed` bit for bit — `python/tests/test_parity.py` gates that alongside the golden dumps.
  The wheel bundles one self-contained shared library; `.github/workflows/wheels.yml` builds
  manylinux x86-64 and macOS arm64.
- **`jepa_error_reset()` / `jepa_error_text()`** — the text the library logs for a failed call,
  readable by a caller with no stderr to read. Append-only; nothing else changes.
- **Native video ingest** — `jepa-embed --video clip.mp4` (repeatable, plus `--video-list`) and
  `jepa-classify --video clip.mp4` decode a container by running `ffmpeg` and sample `--frames`
  frames (default: the model's own `jepa.enc.n_frames`) uniformly over the whole clip, so a clip no
  longer has to be turned into a `.npy` by `scripts/video_frames.py` first. The sampler and the
  decode match that script's, and the frames are byte-identical to the ones PyAV hands it — checked
  by the new `video` ctest suite. `ffmpeg` is a run-time dependency of those two tools only: nothing
  links against it and the build never looks for it. Both tools also gained `--dump-frames`, which
  writes the sampled frames as a THWC uint8 `.npy`.

### Known limitations

- Batching is images only; V-JEPA 2 / 2.1 and LeVJEPA run one clip per graph.
- V-JEPA 2 ViT-L scatters individual tokens under f16 rounding — a property of that checkpoint's
  activation range, reproduced in numpy — so dense per-token work on it wants f32.
- There is no f32 parity tier on a GPU: ggml's CUDA "F32" matmul is TF32 and its flash attention
  accumulates in F16. Use f16 or q8_0 there, and the CPU when f32 exactness matters.
- Not converted yet: V-JEPA 1, V-JEPA 2-AC, the larger V-JEPA 2 / 2.1 and I-JEPA sizes, audio and
  vision-language variants.

[Unreleased]: https://github.com/aselimc/jepa.cpp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aselimc/jepa.cpp/releases/tag/v0.1.0
