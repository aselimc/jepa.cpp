# jepa.cpp

jepa.cpp runs Meta's JEPA vision models — I-JEPA, V-JEPA 2, V-JEPA 2.1 — plus LeJEPA-style ViTs,
LeVJEPA and LeWorldModel on a plain CPU, in C/C++ on [ggml](https://github.com/ggml-org/ggml), with an optional
CUDA backend. Each checkpoint is converted once into a single [GGUF](gguf-schema.md) file that carries
the weights *and* everything needed to run them — dimensions, positional-encoding scheme,
preprocessing recipe, class labels — so at run time the requirement is one binary and one file. There
is no Python in the inference path and no per-model C++ code: a new checkpoint of a known family is a
converter run, because the loader builds the graph from the file's metadata. Seven model bundles of
six families ship today — image and video embedding, video classification, latent-space prediction
and action-conditioned world-model rollout — through four command-line tools and one C header. The
converted files are published on Hugging Face under [**jepacpp**](https://huggingface.co/jepacpp), so
converting anything yourself is optional: `scripts/download_models.sh` fetches them.

![jepa.cpp in four numbers: the same SSv2 validation top-1 as PyTorch, faster on a CPU,
much faster on one GPU, and half the weights at q8_0, over a bar chart of one image
through V-JEPA 2.1 ViT-B on PyTorch, on jepa.cpp's CPU engine and on its CUDA engine](assets/hero.svg)

*Every number on it is read from `tests/results/*.json` and the GPU tables of
[performance.md](performance.md), measured on a 96-core Threadripper 7995WX and two RTX 4500 Ada;
`scripts/gen_hero_figure.py` redraws it.*

## Where to go

| page | what is on it |
|---|---|
| [Getting started](getting-started.md) | build (CPU and CUDA), download the published GGUFs or convert the checkpoints yourself, the per-model licence table, one worked example per tool, running the test suite |
| [Architecture](architecture.md) | the shared ViT graph, the family matrix, the 3-D RoPE specification, preprocessing, attention and precision, batching, the GPU backend, the runtime switches, and the parity methodology |
| [GGUF format](gguf-schema.md) | the file format, version 1: every metadata key, the canonical tensor names, the token order, the quantization rules |
| [Performance](performance.md) | the speed and memory scores: CPU and CUDA encoders against PyTorch, end-to-end classification, predictors, batching, thread scaling, quantization |
| [Accuracy](accuracy.md) | the fidelity and task scores: f32 exactness, f16/q8_0 and GPU cosines, Imagenette and UCF-101 k-NN, the SSv2 validation top-1, the dtype recommendation per backend |
| [C API](api.md) | the complete `include/jepa.h` reference, generated from the header |
| Internals | the [converter](converter.md), the [V-JEPA tensor and RoPE notes](vjepa-notes.md), the [ggml-level notes](ggml-notes.md) behind the video graph, and the [fixtures](fixtures.md) the parity tests replay |
| Appendix | the raw measurement reports the curated pages draw from — [parity](parity.md), [benchmarks](benchmarks.md), [quantization](quantization.md), [image accuracy](accuracy-image.md), [video accuracy](accuracy-video.md) |

## The full results figure

![Three panels: encoder latency per item for PyTorch and jepa.cpp on the CPU and on one GPU, top-1
against PyTorch on Imagenette, UCF-101 and SSv2, and what each dtype costs in weights and in
time](assets/results.svg)

*The same measurements at model resolution: latency per item across the four backends, every accuracy
row against its PyTorch baseline, and what each dtype costs in weights and in time. The three panels also
stand alone, beside the tables they draw — latency and quantization on
[performance.md](performance.md), accuracy on [accuracy.md](accuracy.md);
`scripts/gen_results_figure.py --split` redraws all four.*

Machine-readable twins of the measured tables live in
[`tests/results/*.json`](https://github.com/aselimc/jepa.cpp/tree/main/tests/results): `benchmarks.json`,
`accuracy-image.json`, `accuracy-video.json`, `accuracy-ssv2.json`, `batching.json`.
