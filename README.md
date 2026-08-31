# jepa.cpp

**Run Meta's JEPA vision models on a plain CPU, from a single file, with no Python.**

JEPA (Joint-Embedding Predictive Architecture) is the family of self-supervised vision models behind
Meta's I-JEPA, V-JEPA 2 and V-JEPA 2.1, and the research line around LeJEPA and LeWorldModel. They are
excellent general-purpose feature extractors for images and video — but the official code needs PyTorch,
a GPU-sized dependency stack, and a different repo per model.

jepa.cpp fixes that. It is a small C/C++ engine built on [ggml](https://github.com/ggml-org/ggml)
(the library behind llama.cpp and whisper.cpp). Each model is converted once into a single
[GGUF](https://aselimc.github.io/jepa.cpp/gguf-schema/) file that carries the weights *and* everything
needed to run them — dimensions, positional-encoding scheme, preprocessing recipe, class labels — so at
run time you need nothing but one binary and one file.

What you can do with it today:

- **Embed images and video** — `jepa-embed` turns a JPEG or a clip into a feature vector for search,
  clustering, k-NN classification, or as input to your own model.
- **Classify video** — `jepa-classify` runs V-JEPA 2's Something-Something-v2 head (174 actions) on a clip.
- **Predict in latent space** — the V-JEPA 2 / 2.1 masked predictors ("what would the features of the
  hidden patches be?") through `jepa_predict` in the C API.
- **Roll out a world model** — `jepa-worldmodel` takes an image and a sequence of actions and steps
  LeWorldModel's latent state forward at ~1 ms per step.
- **Shrink models** — `jepa-quantize` produces q8_0 (half of f16, accuracy-safe) down to q4_k files.

Correctness is the point of this project, not an afterthought: every f32 conversion reproduces its
PyTorch reference to cosine 1.000000 (mean; every single token ≥ 0.99999), preprocessing is bit-exact
against torchvision, and quantized files are measured — on real datasets — rather than assumed.
The full evidence lives in **[docs/accuracy.md](docs/accuracy.md)** and
**[docs/performance.md](docs/performance.md)**, and on the
**[documentation site](https://aselimc.github.io/jepa.cpp/)**.

## Supported models

| model | size | what you can do | licence |
|---|---|---|---|
| [I-JEPA ViT-H/14](https://huggingface.co/facebook/ijepa_vith14_1k) | 631 M | image features | CC-BY-NC-4.0 (non-commercial) |
| [LeJEPA ViT-S/16](https://huggingface.co/OK-AI/lejepa-vits16-pretrain-in1k) (community) | 22 M | image features, fast | Apache-2.0 |
| [LeWorldModel Push-T](https://huggingface.co/quentinll/lewm-pusht) | 18 M | image features + action-conditioned world-model rollout | MIT |
| [V-JEPA 2 ViT-L/16](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | 326 M | video features (16–64 frames) + masked predictor | MIT |
| [V-JEPA 2 ViT-L SSv2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) | 375 M | video action classification, 174 classes | MIT |
| [V-JEPA 2.1 ViT-B/16 @384](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt) | 110 M | image *and* video features + predictor (both modalities) | MIT |

A new checkpoint of a known family needs **no C++ change** — the converter writes the metadata and the
loader builds the graph from it.

## How fast, and does it stay accurate?

Short version: on a many-core CPU, jepa.cpp is **1.1–2.2× faster than PyTorch** on the models that
matter, uses **half to a quarter of the memory** at f16/q8_0, and its predictions are **statistically
indistinguishable from PyTorch's** on real datasets. We checked the last claim the hard way — k-NN
classification on Imagenette (images) and a UCF-101 subset (video), where the "classifier" is frozen
features plus nearest neighbours, so any backend error would show up directly as lost accuracy. It doesn't:

| model | jepa.cpp f16, 32 threads | PyTorch f32, 32 threads | real-task check (jepa.cpp vs PyTorch) |
|---|---|---|---|
| I-JEPA ViT-H/14 | 147 ms / image | 250 ms | Imagenette k-NN **95.31 %** vs **95.36 %** |
| LeJEPA ViT-S/16 | 12.8 ms / image | 15.4 ms | Imagenette k-NN **94.55 %** vs **94.45 %** |
| LeWorldModel | 9.8 ms + 0.9 ms per rollout step | 16.8 ms | world-model outputs match to cosine 1.0000000 |
| V-JEPA 2 ViT-L (SSv2, 16-frame clip, end-to-end) | 922 ms / clip (631 ms @ 96 threads) | 1051 ms | UCF-101 k-NN **89.5 %** vs **88.6 %**; SSv2 top-5 identical on fixtures |
| V-JEPA 2.1 ViT-B @384 | 60 ms / image · 853 ms / 16-frame clip | 110 ms · 908 ms | UCF-101 k-NN predictions identical at f32; **89.5 %** vs **88.6 %** at f16 |

*(Threadripper 7995WX, ggml `36da5713`, `GGML_LLAMAFILE=ON`, idle box, 2026-08-31. Every number is
copied from a committed, regenerable artifact — sources, more shapes, 96-thread rows, memory tables and
the fine print are in [docs/performance.md](docs/performance.md).)*

An optional CUDA build is **9–21× faster than those 32-thread numbers** on one RTX 4500 Ada — I-JEPA
15.5 ms, the 16-frame V-JEPA 2 ViT-L clip 46.5 ms, the 64-frame one 306 ms — at 38–62 % of PyTorch's
throughput on the same card; see the GPU paragraph below and
[docs/performance.md](docs/performance.md#gpu-encoder).

Which file should you actually ship? The quantization study boils down to:

| you want | use | why |
|---|---|---|
| the default | **f16** | half of f32, outputs ≥ 0.9998 cosine everywhere |
| pooled features, retrieval, k-NN, world-model rollouts | **q8_0** | half of f16 again; within 0.05 pp of PyTorch on Imagenette |
| classification with the SSv2 head | **f16** | 99.0 % argmax agreement with PyTorch vs 94.3 % at q8_0 |
| dense per-token features from V-JEPA 2 ViT-L | **f32** | that checkpoint scatters individual tokens under f16 rounding |
| the smallest possible file | **q4_k** | 0.29× f16 — fine for pooled use, *slower* than q8_0, below our parity bars |

## Quickstart

Build (needs CMake ≥ 3.16, Ninja and a C++17 compiler; ggml is a submodule):

```bash
git clone --recursive https://github.com/aselimc/jepa.cpp && cd jepa.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Python is needed **only once**, to download and convert checkpoints:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install transformers safetensors numpy gguf pillow huggingface_hub av

scripts/download_models.sh small     # LeJEPA, LeWorldModel, V-JEPA 2.1  ("all" adds I-JEPA + V-JEPA 2, ~7 GB)
scripts/download_fixtures.sh         # a few test images and clips

python scripts/convert.py --family hfvit    --src models/OK-AI/lejepa-vits16-pretrain-in1k --ftype f16
python scripts/convert.py --family lewm     --src models/quentinll/lewm-pusht              --ftype f16
python scripts/convert.py --family vjepa2_1 --src models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \
                          --out models/gguf/vjepa2_1-vitb-384-f16.gguf
```

Then everything is C++:

```bash
# image -> feature vector
build/jepa-embed -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
    -i tests/fixtures/media/coco_000000000139.jpg --pool cls -t 32 --time

# 16-frame clip -> pooled feature (frames from a .npy, or --as-video -i f0.jpg -i f1.jpg ...)
build/jepa-embed -m models/gguf/vjepa2_1-vitb-384-f16.gguf \
    --frames-npy tests/fixtures/ref/vjepa2_1-vitb-384/archery_f16.frames_u8.npy --pool mean -t 32

# clip -> "what is happening?" (SSv2, 174 classes; model comes with download_models.sh all)
build/jepa-classify -m models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf \
    --frames-npy tests/fixtures/ref/vjepa2-vitl-fpc16-256-ssv2/archery_f16.frames_u8.npy -k 5 -t 32

# image -> latent world model -> 8 action steps
build/jepa-worldmodel -m models/gguf/lewm-pusht-f16.gguf \
    --image tests/fixtures/media/coco_000000000139.jpg --random-actions 8 -t 32

# make it smaller
build/jepa-quantize models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
                    models/gguf/lejepa-vits16-pretrain-in1k-q8_0.gguf q8_0 -t 32
```

The C API is one header, [`include/jepa.h`](include/jepa.h) — opaque handles, plain structs, no C++
types. Load a model, make a context, `jepa_encode`, then pool or predict. The full reference is on the
[API page](https://aselimc.github.io/jepa.cpp/api/).

## How it works, in one paragraph

All six families run one shared ViT graph: patchify (a host-side rearrangement plus one matmul), add
positions, pre-LayerNorm transformer blocks with flash attention, final norm. Families differ only in
the tokenizer (2-D patches vs video tubelets), the positional scheme (sincos tables vs 3-D RoPE —
including a faithful reproduction of V-JEPA 2's *tiled* RoPE, a quirk Meta keeps for checkpoint
compatibility, without which every token lands at cosine 0.63), and the optional heads (masked
predictor, attentive-pool classifier, world-model predictor). Preprocessing is a bit-exact port of
torchvision's antialiased uint8 resize, so the tensor entering the network is byte-identical to the
reference pipeline's. The deeper story — architecture, GGUF schema, RoPE derivation, flash-attention
policy, parity thresholds — is on the [documentation site](https://aselimc.github.io/jepa.cpp/).

## Testing

```bash
ctest --test-dir build        # 8 suites: parity, predictors, batching, RoPE vectors, attention
```

`test-parity` replays PyTorch golden dumps through the engine and gates per-token cosine, pooled
outputs and classifier top-1/top-5 with per-family thresholds; `test-predictor` does the same for the
three predictors, including a bit-exactness causality check on the world model. Details:
[parity](https://aselimc.github.io/jepa.cpp/parity/) · [accuracy](docs/accuracy.md).

## Optional: running on a GPU

The engine is CPU-first and stays that way — `libggml-cuda.so` is 45 MB for a *single* GPU
architecture — but the graphs are backend-clean, so a CUDA build is one CMake flag:
`cmake -S . -B build-cuda -DJEPA_CUDA=ON`, then `--gpu [N]` on `jepa-embed`, `jepa-classify`,
`jepa-worldmodel` and `jepa-bench` (or `JEPA_DEVICE=cuda:0`). Weights and activations live entirely
on the card; before every graph runs, jepa.cpp walks it against the backend's own `supports_op` and
refuses to compute anything the device cannot really run, because a GPU backend that is handed such
a node returns a *wrong answer with no error*. One caveat worth knowing before you switch: **there
is no f32 parity tier on a GPU** — ggml's CUDA "F32" matmul is really TF32 and its flash attention
always accumulates in F16 — so use f16 or q8_0 there (both hold their bars, and quantized weights
are the *fastest* GPU path), and the CPU when you need f32 exactness. The design:
[docs/architecture.md](docs/architecture.md); every measured number:
[docs/performance.md](docs/performance.md) and [docs/accuracy.md](docs/accuracy.md).

## Limitations

- **Batching is images only** — `jepa_encode` puts up to 32 image items through one ggml graph
  (`jepa-embed --batch`, bit-identical to one-at-a-time), worth 1.7–2.1× of encoder time and taking
  end-to-end throughput at 32 threads from 67 to 94 img/s on LeJEPA ViT-S/16 and 95 to 159 on LeWM
  against PyTorch batch-32's 86–89 and 190–206, and 6.2 to 6.9 (7.6 at q8_0) on I-JEPA ViT-H/14
  against 5.5. So LeJEPA now passes PyTorch, LeWM is still 0.77–0.84× of it, and a V-JEPA 2 clip is
  deliberately not batched — one clip is already 2 048–18 432 tokens and saturates the cores.
- **V-JEPA 2 ViT-L scatters individual tokens at f16** (a property of the checkpoint's activation
  range, reproduced in numpy — not an engine bug); use f32 for dense per-token work on that one model.
- **Not converted yet:** V-JEPA 1, V-JEPA 2-AC (the schema and kernels exist), the larger V-JEPA 2 /
  2.1 and I-JEPA sizes, audio and VL variants.
- **Real SSv2 accuracy is unmeasured** (the dataset is licence-gated); we report agreement with
  PyTorch on independent clips instead.

## Docs, acknowledgements, licence

Documentation site: **<https://aselimc.github.io/jepa.cpp/>** — built from `docs/` with MkDocs
(`pip install -r docs/requirements.txt && mkdocs serve` to browse locally). All detailed tables:
[docs/performance.md](docs/performance.md) and [docs/accuracy.md](docs/accuracy.md), with the raw
measurement reports under `docs/{parity,benchmarks,quantization,accuracy-image,accuracy-video}.md`;
machine-readable twins: `tests/results/*.json`.

Built on [ggml](https://github.com/ggml-org/ggml). Models by **Meta FAIR** (I-JEPA, V-JEPA 2 / 2.1),
**OK-AI** (LeJEPA ViT-S/16) and **quentinll / le-wm** (LeWorldModel Push-T).

Code: **MIT** ([`LICENSE`](LICENSE)). Converted GGUFs keep their checkpoint's licence (carried in
`general.license` inside the file); note **I-JEPA is CC-BY-NC-4.0 — non-commercial only**.
