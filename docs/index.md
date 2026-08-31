# jepa.cpp documentation

jepa.cpp runs Meta's JEPA vision models — I-JEPA, V-JEPA 2, V-JEPA 2.1 — plus LeJEPA-style ViTs and
LeWorldModel on a plain CPU, in C/C++ on [ggml](https://github.com/ggml-org/ggml), from single-file
GGUF bundles. The [README](https://github.com/aselimc/jepa.cpp#readme) carries the pitch, the
supported-model table and the headline numbers; this site is the depth behind it.

Start here:

- **[Results](results.md)** — every measured table: speed at 32/96 threads, memory, f32 exactness,
  the Imagenette and UCF-101 k-NN accuracy studies, the SSv2 head-fidelity numbers, and the
  dtype-recommendation table.
- **[C API](api.md)** — the complete `include/jepa.h` reference, generated from the header itself.

## Design and formats

**[architecture.md](architecture.md)** — the shortest complete description of the engine: the repo
layout, the single ViT graph all six families share, the per-family deltas (tokenizer, position scheme,
extras), the full V-JEPA 2 / 2.1 3-D RoPE specification down to the tiled-vs-interleaved frequency
table, and the parity protocol. Read this before touching `src/`.

**[gguf-schema.md](gguf-schema.md)** — the file format, version 1. Every `general.*` and `jepa.*`
metadata key with its type and meaning, the canonical tensor names for encoder / predictor / head, the
token order, the quantization rules, and the family-specific notes. The converter, the loader and the
graph builder all implement exactly this document; change it here first.

**[converter.md](converter.md)** — the Python side: one converter module per family, the `JepaWriter`
helpers (sincos tables, qkv fusing, BatchNorm folding, the f16 dtype rule), and `selftest.py`, a numpy
forward pass driven straight from a GGUF that serves as the executable spec the C++ graph must match.

**[vjepa-notes.md](vjepa-notes.md)** — the V-JEPA 2 / 2.1 tensor maps and the 3-D RoPE derivation
(tiled vs interleaved cos/sin layouts, `interpolate_rope`) cited throughout the schema, architecture
and parity pages.

**[fixtures.md](fixtures.md)** — the parity fixtures: which media files, how `dump_reference.py`
produces the PyTorch golden `.npy` dumps, and the per-model tensor list each
`ref/<model>/manifest.json` carries.

## Correctness

**[parity.md](parity.md)** — the correctness page. Per-token cosine, `rel_max`, pooled/logit agreement
and per-item timing for every shipped GGUF × dtype, images and video, plus the predictors. Contains the
`POLICY` threshold table `test-parity` judges with (family class × file-type tier, including the
length-aware `REL(N)` bound), the analysis of why V-JEPA 2 ViT-L f16 tokens scatter while its pooled
outputs do not, the flash-attention K/V dtype policy, and the preprocessing-parity table showing
resize + crop + normalisation are bit-exact against torchvision.

**[quantization.md](quantization.md)** — `tools/jepa-quantize`: which tensors are re-typed, the CLI,
the K-quant per-tensor fallback, GGUF file sizes for all ten types, and the measured accuracy of each
type against the PyTorch references (weight-only: the quantized GGUF is dequantized in Python and pushed
through the numpy graph). Ends with a use-case → dtype recommendation table.

## Performance

**[benchmarks.md](benchmarks.md)** — every timing in the repo, from `tools/jepa-bench` on synthetic but
deterministic input, so it reproduces without fixtures or Python. Encoder, attentive-pool head, masked
predictor and LeWM rollout, at 32 and 96 threads, each dtype, with the PyTorch CPU baseline and its
provenance, a memory table, a thread-scaling table, a sub-8-bit size/speed table, and a cross-check
against parity.md's independent timings. Machine-readable twin:
[`tests/results/benchmarks.json`](https://github.com/aselimc/jepa.cpp/blob/main/tests/results/benchmarks.json).

**[ggml-notes.md](ggml-notes.md)** — the ggml-level findings the video graph needed: what
`ggml_flash_attn_ext` wants (shapes, dtypes, which kernel runs), the block-causal mask recipe, measured
attention accuracy and wall time, memory behaviour, this box's matmul throughput per dtype, and the
gotchas collected on the way. Read it before adding an op.

## Accuracy on real data

**[accuracy-image.md](accuracy-image.md)** — frozen-feature k-NN on Imagenette (10 classes), PyTorch vs
jepa.cpp per dtype, for I-JEPA ViT-H/14, LeJEPA ViT-S/16 (CLS and patch-mean) and LeWM. Nothing is
trained: the encoders are frozen and the classifier is a cosine vote over gallery features. Also carries
the flip analysis, the JPEG-decoder floor, an end-to-end throughput table, and a graph-compute-only
table. Machine-readable twin:
[`tests/results/accuracy-image.json`](https://github.com/aselimc/jepa.cpp/blob/main/tests/results/accuracy-image.json).

**[accuracy-video.md](accuracy-video.md)** — the same protocol on a UCF-101 subset for V-JEPA 2 ViT-L/16
and V-JEPA 2.1 ViT-B/384, plus an SSv2-head fidelity section scoring 105 independent 174-way argmaxes
against PyTorch's. Machine-readable twin:
[`tests/results/accuracy-video.json`](https://github.com/aselimc/jepa.cpp/blob/main/tests/results/accuracy-video.json).

The accuracy benchmarks need the datasets from `scripts/download_datasets.sh`
(Imagenette-160 and the UCF101 subset, ~400 MB into `data/`).

## Building this site

```bash
pip install -r docs/requirements.txt
mkdocs serve          # live preview at http://127.0.0.1:8000
```

The site deploys to GitHub Pages automatically on every push to `main`
(`.github/workflows/docs.yml`; the `api.md` page is generated from `include/jepa.h` by
`scripts/gen_api_md.py`).
