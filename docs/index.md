# jepa.cpp

jepa.cpp runs Meta's JEPA vision models — I-JEPA, V-JEPA 2, V-JEPA 2.1 — plus LeJEPA-style ViTs and
LeWorldModel on a plain CPU, in C/C++ on [ggml](https://github.com/ggml-org/ggml), with an optional
CUDA backend. Each checkpoint is converted once into a single [GGUF](gguf-schema.md) file that carries
the weights *and* everything needed to run them — dimensions, positional-encoding scheme,
preprocessing recipe, class labels — so at run time the requirement is one binary and one file. There
is no Python in the inference path and no per-model C++ code: a new checkpoint of a known family is a
converter run, because the loader builds the graph from the file's metadata.

Six model bundles ship today: I-JEPA ViT-H/14, LeJEPA ViT-S/16, LeWorldModel Push-T, V-JEPA 2 ViT-L/16
(encoder + masked predictor), V-JEPA 2 ViT-L SSv2 (174-class video classifier) and V-JEPA 2.1 ViT-B/16
at 384 px (image *and* video, with a predictor for both). They expose image and video embedding, video
classification, latent-space prediction and action-conditioned world-model rollout, through four
command-line tools and one C header.

On 32 CPU threads the engine runs **1.1–1.8× faster than PyTorch** on the models where the comparison
is like-for-like, and 2.1–2.2× on 96 threads, at half to a quarter of the memory; a CUDA build is
9–21× faster again on one RTX 4500 Ada. Fidelity is measured rather than assumed: f32 files reproduce
their PyTorch reference to cosine 1.000000 per token, preprocessing is bit-exact against torchvision,
and quantized files are scored on real datasets — Imagenette k-NN within **0.13 pp** of PyTorch at
f16 and q8_0 (0.16 pp on the parameter-free centroid metric), UCF-101 k-NN within one clip of it.

![Three panels: encoder latency per item for PyTorch and jepa.cpp on the CPU and on one GPU, k-NN
top-1 against PyTorch on Imagenette and UCF-101, and what each dtype costs in weights and in
time](assets/results.svg)

*Every number on it is read from `tests/results/{benchmarks,accuracy-image,accuracy-video}.json` and
the GPU tables of [performance.md](performance.md), measured on a 96-core Threadripper 7995WX and one
RTX 4500 Ada; `scripts/gen_results_figure.py` redraws it.*

## Where to go

| page | what is on it |
|---|---|
| [Getting started](getting-started.md) | build (CPU and CUDA), one-time Python environment, download and convert every supported checkpoint, one worked example per tool, running the test suite |
| [Architecture](architecture.md) | the shared ViT graph, the family matrix, the 3-D RoPE specification, preprocessing, attention and precision, batching, the GPU backend, the runtime switches, and the parity methodology |
| [GGUF format](gguf-schema.md) | the file format, version 1: every metadata key, the canonical tensor names, the token order, the quantization rules |
| [Performance](performance.md) | the speed and memory scores: CPU and CUDA encoders against PyTorch, end-to-end classification, predictors, batching, thread scaling, quantization |
| [Accuracy](accuracy.md) | the fidelity and task scores: f32 exactness, f16/q8_0 and GPU cosines, Imagenette and UCF-101 k-NN, SSv2 head agreement, the dtype recommendation per backend |
| [C API](api.md) | the complete `include/jepa.h` reference, generated from the header |
| Internals | the [converter](converter.md), the [V-JEPA tensor and RoPE notes](vjepa-notes.md), the [ggml-level notes](ggml-notes.md) behind the video graph, and the [fixtures](fixtures.md) the parity tests replay |
| Appendix | the raw measurement reports the curated pages draw from — [parity](parity.md), [benchmarks](benchmarks.md), [quantization](quantization.md), [image accuracy](accuracy-image.md), [video accuracy](accuracy-video.md) |

Machine-readable twins of the measured tables live in
[`tests/results/*.json`](https://github.com/aselimc/jepa.cpp/tree/main/tests/results): `benchmarks.json`,
`accuracy-image.json`, `accuracy-video.json`, `batching.json`.
