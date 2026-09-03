# jepa.cpp

[![ci](https://github.com/aselimc/jepa.cpp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aselimc/jepa.cpp/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/aselimc/jepa.cpp?label=release&color=2f5fa3)](https://github.com/aselimc/jepa.cpp/releases/latest)
[![models](https://img.shields.io/badge/🤗%20models-jepacpp-ffcc4d)](https://huggingface.co/jepacpp)
[![docs](https://img.shields.io/badge/docs-aselimc.github.io%2Fjepa.cpp-2f5fa3)](https://aselimc.github.io/jepa.cpp/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**JEPA vision models in plain C/C++.** One binary, one GGUF file, any CPU, optional CUDA. No Python
in the loop.

jepa.cpp runs the JEPA family of self-supervised vision encoders and world models — I-JEPA,
V-JEPA 2 / 2.1 / 2-AC, LeJEPA, LeVJEPA and LeWorldModel — on [ggml](https://github.com/ggml-org/ggml),
the engine behind llama.cpp. Every model reproduces its PyTorch reference to cosine 1.000 in f32,
runs **1.1–1.8× faster than PyTorch on a CPU**, and comes pre-converted on
[Hugging Face](https://huggingface.co/jepacpp).

- **Embed** images and video clips into feature vectors — search, clustering, k-NN, your own heads
- **Classify** video with V-JEPA 2's Something-Something-v2 head (174 actions)
- **Predict** in latent space with the V-JEPA 2 / 2.1 masked predictors
- **Roll out and plan** with world models — LeWorldModel, and V-JEPA 2-AC with its CEM planner
- **Serve** it over HTTP, call it from C, or from Python with numpy in and out

![Three panels: latency per image or clip on a 32-thread CPU, jepa.cpp f16 against PyTorch f32;
latency on one RTX 4500 Ada, jepa.cpp CUDA against PyTorch fp16 and fp32; and accuracy on
Imagenette, UCF-101 and the full SSv2 validation split, PyTorch against jepa.cpp](docs/assets/hero.svg)

*Every number is read from a committed artifact under `tests/results/`. The full tables, the
quantization study and the fine print are on the [docs site](https://aselimc.github.io/jepa.cpp/performance/).*

## Model zoo

Nine models, six families, all on Hugging Face under [**jepacpp**](https://huggingface.co/jepacpp).
Each repo ships `f32`, `f16`, `q8_0`, `q4_0` and `q4_k` files with their sha256 and measured parity on
the card. **f16 is the default; q8_0 halves it again and is safe for pooled features, retrieval and
rollouts.**

| model | from | params | does | f16 / q8_0 | GGUF | licence |
|---|---|---|---|---|---|---|
| LeJEPA ViT-S/16 | [OK-AI](https://huggingface.co/OK-AI/lejepa-vits16-pretrain-in1k) | 22 M | image features, fast | 42 / 23 MB | [↓](https://huggingface.co/jepacpp/lejepa-vits16-pretrain-in1k-GGUF) | Apache-2.0 |
| LeWorldModel Push-T | [le-wm](https://huggingface.co/quentinll/lewm-pusht) | 18 M | image features + action-conditioned world model | 38 / 23 MB | [↓](https://huggingface.co/jepacpp/lewm-pusht-GGUF) | MIT |
| V-JEPA 2.1 ViT-B/16 @384 | [Meta FAIR](https://github.com/facebookresearch/vjepa2) | 110 M | image *and* video features + predictor | 210 / 113 MB | [↓](https://huggingface.co/jepacpp/vjepa2_1-vitb-384-GGUF) | MIT |
| LeVJEPA ViT-L/16 | [galilai](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large) | 303 M | video features, block-causal attention | 579 / 310 MB | [↓](https://huggingface.co/jepacpp/levjepa-vitl16-GGUF) | CC-BY-NC-4.0 |
| V-JEPA 2 ViT-L/16 | [Meta FAIR](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | 326 M | video features (16–64 frames) + masked predictor | 623 / 333 MB | [↓](https://huggingface.co/jepacpp/vjepa2-vitl-fpc64-256-GGUF) | MIT |
| V-JEPA 2 ViT-L SSv2 | [Meta FAIR](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) | 375 M | video action classification, 174 classes | 717 / 383 MB | [↓](https://huggingface.co/jepacpp/vjepa2-vitl-fpc16-256-ssv2-GGUF) | MIT |
| I-JEPA ViT-H/14 | [Meta FAIR](https://huggingface.co/facebook/ijepa_vith14_1k) | 631 M | image features | 1.2 / 0.64 GB | [↓](https://huggingface.co/jepacpp/ijepa_vith14_1k-GGUF) | CC-BY-NC-4.0 |
| V-JEPA 2 ViT-g/16 | [Meta FAIR](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256) | 1.03 B | video features (16–64 frames) + masked predictor | 2.0 / 1.1 GB | [↓](https://huggingface.co/jepacpp/vjepa2-vitg-fpc64-256-GGUF) | Apache-2.0 |
| V-JEPA 2-AC ViT-g | [Meta FAIR](https://github.com/facebookresearch/vjepa2) | 1.32 B | action-conditioned world model + CEM planner | 2.5 / 1.3 GB | [↓](https://huggingface.co/jepacpp/vjepa2-ac-vitg-GGUF) | MIT |

A new checkpoint of a known family needs no C++ change: the converter writes the metadata, the
loader builds the graph from it.

## Quickstart

Prebuilt Linux x86-64 binaries are on the [releases page](https://github.com/aselimc/jepa.cpp/releases).
From source (CMake ≥ 3.16, Ninja, a C++17 compiler):

```bash
git clone --recursive https://github.com/aselimc/jepa.cpp && cd jepa.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
scripts/download_models.sh small      # LeJEPA, LeWorldModel, V-JEPA 2.1 at f16 (290 MB); "all" for the rest
scripts/download_fixtures.sh          # sample media (+ the PyTorch golden dumps for ctest)
```

```bash
# image -> feature vector
build/jepa-embed -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf -i tests/fixtures/media/coco_000000000139.jpg --pool cls

# video -> feature vector (16 frames sampled like the PyTorch pipeline; needs ffmpeg)
build/jepa-embed -m models/gguf/vjepa2_1-vitb-384-f16.gguf --video tests/fixtures/media/archery.mp4 --frames 16 --pool mean

# video -> "what is happening?"  (SSv2, from download_models.sh all)
build/jepa-classify -m models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf --video tests/fixtures/media/archery.mp4 -k 5

# image -> latent world model -> 8 action steps
build/jepa-worldmodel -m models/gguf/lewm-pusht-f16.gguf --image tests/fixtures/media/coco_000000000139.jpg --random-actions 8

# HTTP server: POST /v1/embeddings (OpenAI-shaped), /classify, /rollout, /metrics
build/jepa-server -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf --workers 4
```

- **GPU:** `cmake -S . -B build-cuda -DJEPA_CUDA=ON`, then `--gpu` on any tool.
- **C:** one header, [`include/jepa.h`](include/jepa.h) — load, encode, pool or predict.
- **Python:** `pip install ./python` wraps that header behind numpy, bit-identical to the CLI on a CPU
  ([python/README.md](python/README.md)).
- **Smaller files:** `build/jepa-quantize in-f16.gguf out-q8_0.gguf q8_0`.

## Learn more

| | |
|---|---|
| [Getting started](https://aselimc.github.io/jepa.cpp/getting-started/) | build, download or convert, one worked example per tool, licences |
| [Performance](https://aselimc.github.io/jepa.cpp/performance/) · [Accuracy](https://aselimc.github.io/jepa.cpp/accuracy/) | every measured number: CPU, GPU, batching, quantization, real-task accuracy |
| [Architecture](https://aselimc.github.io/jepa.cpp/architecture/) | the shared ViT graph, 3-D RoPE, preprocessing, the family matrix, GPU notes |
| [C API](https://aselimc.github.io/jepa.cpp/api/) · [Serving](https://aselimc.github.io/jepa.cpp/serving/) · [GGUF schema](https://aselimc.github.io/jepa.cpp/gguf-schema/) | the header, the HTTP server, what a file carries |
| [Parity](https://aselimc.github.io/jepa.cpp/parity/) | how correctness is tested: golden dumps, per-token thresholds, 19 ctest suites |

## Credits and licence

Built on [ggml](https://github.com/ggml-org/ggml). Models by **Meta FAIR** (I-JEPA, V-JEPA 2 / 2.1 / 2-AC),
**Balestriero & LeCun / OK-AI** (LeJEPA), **galilai** (LeVJEPA) and **le-wm** (LeWorldModel).

Code is **MIT**. Converted GGUFs keep their checkpoint's licence, carried inside the file; note that
I-JEPA and LeVJEPA are **CC-BY-NC-4.0, non-commercial**.
