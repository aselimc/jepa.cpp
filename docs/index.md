# jepa.cpp documentation index

One paragraph per document, in roughly the order a new reader wants them. Start at
[`../README.md`](../README.md) for the pitch, the supported-model table and the headline benchmark and
accuracy numbers; everything here is the depth behind one of its sections.

## Design and formats

**[`architecture.md`](architecture.md)** — the shortest complete description of the engine: the repo
layout (what every file under `src/`, `tools/`, `tests/` and `scripts/` is for), the single ViT graph all
six families share, the per-family deltas (tokenizer, position scheme, extras), the full V-JEPA 2 / 2.1
3-D RoPE specification down to the tiled-vs-interleaved frequency table, and the four-step parity protocol
every phase of the project has to end with. Read this before touching `src/`.

**[`gguf-schema.md`](gguf-schema.md)** — the file format, version 1. Every `general.*` and `jepa.*`
metadata key with its type and meaning, the canonical tensor names for encoder / predictor / head, the
token order, the quantization rules (which tensors may change type and which never do), and the
family-specific notes for `ijepa`, `hfvit` and `lewm`. The converter, the loader and the graph builder all
implement exactly this document; change it here first.

**[`../scripts/jepa_convert/README.md`](../scripts/jepa_convert/README.md)** — the Python side: one
converter module per family, the `JepaWriter` helpers (sincos tables, qkv fusing, BatchNorm folding, the
f16 dtype rule), and `selftest.py`, a numpy forward pass driven straight from a GGUF that serves as the
executable spec the C++ graph builder must match.

**[`../scripts/jepa_convert/VJEPA_NOTES.md`](../scripts/jepa_convert/VJEPA_NOTES.md)** — the V-JEPA 2 / 2.1 tensor maps and the 3-D RoPE derivation (tiled vs interleaved cos/sin layouts, `interpolate_rope`) cited by `docs/gguf-schema.md`, `docs/architecture.md` and `docs/parity.md`.

**[`../tests/fixtures/README.md`](../tests/fixtures/README.md)** — the parity fixtures: which media files
(8 COCO images, 6 Kinetics-mini clips), how `dump_reference.py` produces the PyTorch golden `.npy` dumps,
and the per-model tensor list each `ref/<model>/manifest.json` carries.

## Correctness

**[`parity.md`](parity.md)** — the correctness page. Per-token cosine, `rel_max`, pooled/logit agreement
and per-item timing for every shipped GGUF × dtype, images and video, plus the predictors (V-JEPA 2 masked,
V-JEPA 2.1 image-vs-video modality, LeWM world model). Contains the `POLICY` threshold table `test-parity`
judges with (family class × file-type tier, including the length-aware `REL(N)` bound), the analysis of why
V-JEPA 2 ViT-L f16 tokens scatter while its pooled outputs do not, the flash-attention K/V dtype policy,
and the preprocessing-parity table showing resize + crop + normalisation are bit-exact against torchvision.

**[`quantization.md`](quantization.md)** — `tools/jepa-quantize`: the rule for which tensors are re-typed,
the CLI, the K-quant per-tensor fallback, GGUF file sizes for all ten types, and the measured accuracy of
each type against the PyTorch references. These numbers isolate the *weight* error — the quantized GGUF is
dequantized in Python and pushed through the numpy graph, so no C++ kernel is involved. Ends with a
use-case → dtype recommendation table.

## Performance

**[`benchmarks.md`](benchmarks.md)** — every timing in the repo, from `tools/jepa-bench` on synthetic but
deterministic input, so it reproduces without fixtures or Python. Encoder, attentive-pool head, masked
predictor and LeWM rollout, at 32 and 96 threads, each dtype, with the PyTorch CPU baseline and its
provenance, a memory table (weights, peak RSS, load time), a thread-scaling table, a sub-8-bit
size/speed table, and a cross-check against `parity.md`'s independent timings of the same graphs.
Machine-readable twin: `../tests/results/benchmarks.json`.

**[`ggml-notes.md`](ggml-notes.md)** — the ggml-level findings the video graph needed: what
`ggml_flash_attn_ext` wants (shapes, dtypes, which kernel runs), the block-causal mask recipe, measured
attention accuracy and wall time, memory behaviour, this box's matmul throughput per dtype, and a list of
gotchas collected on the way. Read it before adding an op.

## Accuracy on real data

**[`accuracy-image.md`](accuracy-image.md)** — frozen-feature k-NN on Imagenette (10 classes), PyTorch vs
jepa.cpp per dtype, for I-JEPA ViT-H/14, LeJEPA ViT-S/16 (CLS and patch-mean) and LeWM. Nothing is trained:
the encoders are frozen and the classifier is a cosine vote over gallery features. Also carries the flip
analysis (which items change and how tied they were), the JPEG-decoder floor that limits any
start-from-a-JPEG comparison, an end-to-end throughput table, and a graph-compute-only table that separates
the dtype effect from decode and preprocessing. Machine-readable twin:
`../tests/results/accuracy-image.json`.

**[`accuracy-video.md`](accuracy-video.md)** — the same protocol on a UCF-101 subset for V-JEPA 2 ViT-L/16
and V-JEPA 2.1 ViT-B/384: k-NN and nearest-centroid top-1, per-clip agreement with PyTorch, feature
fidelity over all 405 clips, and an SSv2-head fidelity section that scores 105 independent 174-way argmaxes
against PyTorch's. Includes the per-clip disagreement forensics and the measured load conditions of every
timed pass. Machine-readable twin: `../tests/results/accuracy-video.json`.

The accuracy benchmarks need the datasets from `scripts/download_datasets.sh` (Imagenette-160 and the UCF101 subset, ~400 MB into `data/`).
