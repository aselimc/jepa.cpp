# Changelog

Notable changes to jepa.cpp, newest first. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the version numbers follow
[semantic versioning](https://semver.org/spec/v2.0.0.html) as narrowed by
[docs/releases.md](docs/releases.md), which also states what the C API and the GGUF schema promise
across releases.

## [Unreleased]

### Added

- **Python bindings** — `pip install jepa-cpp` (import `jepa_cpp`), a thin wrapper over
  `include/jepa.h` with numpy on both ends: `Model.encode` / `pool` / `classify` / `predict` /
  `lewm_rollout`, the file's metadata as properties, and `jepa_cpp._api` as the unwrapped header.
  Preprocessing and every floating-point operation stay in C, so on a CPU the bindings reproduce
  `jepa-embed` bit for bit — `python/tests/test_parity.py` gates that alongside the golden dumps.
  The wheel bundles one self-contained shared library; `.github/workflows/wheels.yml` builds
  manylinux x86-64 and macOS arm64.
- **`jepa_error_reset()` / `jepa_error_text()`** — the text the library logs for a failed call,
  readable by a caller with no stderr to read. Append-only; nothing else changes.

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
