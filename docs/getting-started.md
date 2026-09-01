# Getting started

From a clean checkout to a feature vector: build the engine, create the one-time Python environment
that converts checkpoints, convert a model, run a tool.

## Build

Requirements: CMake ≥ 3.16, Ninja, a C++17 compiler. ggml is a submodule, so clone recursively.

```bash
git clone --recursive https://github.com/aselimc/jepa.cpp && cd jepa.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

That produces `build/jepa-{info,embed,classify,worldmodel,quantize,bench}` and the static library
`libjepa.a`. The relevant CMake options:

| option | default | effect |
|---|---|---|
| `JEPA_BUILD_TOOLS` | `ON` | build the command-line tools |
| `JEPA_BUILD_TESTS` | `ON` | build the test binaries and register the `ctest` suites |
| `JEPA_NATIVE` | `ON` | `-march=native`; turn off for a portable binary |
| `JEPA_CUDA` | `OFF` | build the ggml CUDA backend and add `--gpu` to the tools |
| `GGML_LLAMAFILE` | forced `ON` | llamafile's accelerated sgemm — 1.3–3.2× faster `mul_mat` on AVX-512 |

### CUDA build

The CUDA backend is optional and off by default: `libggml-cuda.so` is 45 MB for a *single* GPU
architecture, thirty times the CPU backend. Configure it in its own directory so the CPU build stays
intact.

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DJEPA_CUDA=ON
cmake --build build-cuda -j 16
build-cuda/jepa-info --devices          # list what the ggml backend registry can see
```

`JEPA_NATIVE=ON` resolves `CMAKE_CUDA_ARCHITECTURES` to `native`, which is right for a from-source
build and wrong for a distributed binary — pass an explicit
`-DCMAKE_CUDA_ARCHITECTURES=89-real;80-virtual` for one of those. Under CUDA 13 the toolkit no longer
offers the 50/61/70 virtual architectures, so a non-native build starts at `75-virtual`.

Budget for the build: 138 CUDA translation units, about 171 s wall at `-j8` for a single
architecture, 114 MB of build tree, and the 45 MB `libggml-cuda.so` at the end of it.

Every tool then takes `--gpu [N]` (or `$JEPA_DEVICE=cuda:N`); see
[Architecture → GPU backend](architecture.md#gpu-backend) for what changes on a device, and
[Accuracy](accuracy.md#backends-and-precision) for which dtype to use there. `--gpu N` picks one
device per process — there is no split across cards, so two GPUs means two processes.

**Other backends.** The device lookup is the backend-agnostic ggml registry, so a Vulkan, Metal or
ROCm build would appear in `jepa-info --devices` with no engine change. Vulkan is not buildable on
this box as it stands: `-DGGML_VULKAN=ON` needs `glslc` (shaderc) and the Vulkan/SPIRV headers, which
are a prerequisite to install rather than a code change.

## Python, once

Python is needed only to download and convert checkpoints. Nothing in the inference path uses it.

```bash
uv venv .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install transformers safetensors numpy gguf pillow huggingface_hub av
```

`torch` and `transformers` are needed only by the converters that read Hugging Face checkpoints and
by the reference-dump scripts; `gguf` and `numpy` are the minimum for conversion itself.

## Download and convert

`scripts/download_models.sh` fetches the reference checkpoints into `models/` (git-ignored).
`small` takes LeJEPA, LeWorldModel and V-JEPA 2.1 (~1.8 GB); `all` adds I-JEPA, both V-JEPA 2 files and
LeVJEPA (~8 GB); a single name fetches one.

```bash
scripts/download_models.sh small     # or: all | ijepa | vjepa2 | vjepa2-ssv2 | vjepa21 | levjepa | lewm | lejepa
scripts/download_fixtures.sh         # a few test images and clips, for the parity tests
```

Conversion writes `models/gguf/<name>-<ftype>.gguf`. `--ftype f16` stores the attention, FFN,
projection and classifier matrices as F16 and everything else (patch embeddings, norms, biases,
position tables, tokens) as F32; `--ftype f32` stores everything as F32. Quantized files come from
`jepa-quantize`, never from the converter.

| model | download name | convert |
|---|---|---|
| LeJEPA ViT-S/16 | `lejepa` | `python scripts/convert.py --family hfvit --src models/OK-AI/lejepa-vits16-pretrain-in1k --ftype f16` |
| LeWorldModel Push-T | `lewm` | `python scripts/convert.py --family lewm --src models/quentinll/lewm-pusht --ftype f16` |
| V-JEPA 2.1 ViT-B/16 @384 | `vjepa21` | `python scripts/convert.py --family vjepa2_1 --src models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt --out models/gguf/vjepa2_1-vitb-384-f16.gguf` |
| I-JEPA ViT-H/14 | `ijepa` | `python scripts/convert.py --family ijepa --src models/facebook/ijepa_vith14_1k --ftype f16` |
| V-JEPA 2 ViT-L/16 (fpc64) | `vjepa2` | `python scripts/convert.py --family vjepa2 --src models/facebook/vjepa2-vitl-fpc64-256 --ftype f16` |
| V-JEPA 2 ViT-L SSv2 | `vjepa2-ssv2` | `python scripts/convert.py --family vjepa2 --src models/facebook/vjepa2-vitl-fpc16-256-ssv2 --ftype f16` |
| LeVJEPA ViT-L/16 | `levjepa` | `python scripts/convert.py --family levjepa --src models/galilai-group/LeVJEPA-VideoMix-Large --out models/gguf/levjepa-vitl16-f16.gguf --ftype f16` |

I-JEPA and LeVJEPA are CC-BY-NC-4.0 (non-commercial); the other five are MIT or Apache-2.0. The licence
travels inside the GGUF as `general.license`.

## Running the tools

### `jepa-info` — what is in a file

```bash
build/jepa-info models/gguf/vjepa2_1-vitb-384-f16.gguf          # hparams + tensor table
build/jepa-info models/gguf/vjepa2_1-vitb-384-f16.gguf --kv     # every general.* / jepa.* key verbatim
build/jepa-info --devices                                       # GPU devices the registry can see
```

### `jepa-embed` — images and clips to feature vectors

```bash
# one image -> CLS feature, timed
build/jepa-embed -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
    -i tests/fixtures/media/coco_000000000139.jpg --pool cls -t 32 --time

# many images in one graph (bit-identical to one at a time on the CPU, 1.7-2.2x faster on the small models)
build/jepa-embed -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
    -i a.jpg -i b.jpg -i c.jpg --batch 32 -o feats.npy -t 32

# a 16-frame clip from a THWC uint8 .npy -> pooled feature
build/jepa-embed -m models/gguf/vjepa2_1-vitb-384-f16.gguf \
    --frames-npy tests/fixtures/ref/vjepa2_1-vitb-384/archery_f16.frames_u8.npy --pool mean -t 32

# frames given as images, in order
build/jepa-embed -m models/gguf/vjepa2_1-vitb-384-f16.gguf --as-video -i f0.jpg -i f1.jpg -t 32

# LeVJEPA reads a clip through its CLS token; a still image is repeated to the model's 16 frames,
# which is how its model card feeds one, and jepa-embed says so on stderr when it does. The repeat
# is jepa-embed's: jepa_encode() itself runs n_frames frames as given, so a library caller passing
# one frame gets a one-frame clip. Any clip length other than the trained 16 also gets a note.
build/jepa-embed -m models/gguf/levjepa-vitl16-f16.gguf -i cat.jpg --pool cls -t 32

# a whole list of clips, one model load, output [n_clips, D] in list order
build/jepa-embed -m models/gguf/vjepa2-vitl-fpc64-256-f16.gguf \
    --frames-list clips.txt -o feats.npy -t 32
```

`--pool` selects `mean` (mean of the patch tokens), `cls`, `lewm` (the world-model state projection)
or `none` (the full `[N, D]` token map). `scripts/video_frames.py` writes the THWC uint8 `.npy` clips
that `--frames-npy` and `--frames-list` read. Clip frames may have different source sizes: each is
resized and centre-cropped on its own before the planes are concatenated.

### `jepa-classify` — a clip to labels

```bash
build/jepa-classify -m models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf \
    --frames-npy tests/fixtures/ref/vjepa2-vitl-fpc16-256-ssv2/archery_f16.frames_u8.npy -k 5 -t 32 --time
```

```
1.  59.36%  [ 90] Pulling two ends of [something] but nothing happens
2.  14.87%  [162] Trying to bend [something unbendable] so nothing happens
...  preprocess 26 ms | encoder 968 ms | head 96 ms | 2115 tokens/s
```

The 174 Something-Something-v2 label strings come out of the GGUF, so no side file is needed.

### `jepa-worldmodel` — image to latent state to rollout

```bash
build/jepa-worldmodel -m models/gguf/lewm-pusht-f16.gguf \
    --image tests/fixtures/media/coco_000000000139.jpg --random-actions 8 -t 32
```

It encodes the image, projects the CLS token to the world-model state, rolls the predictor out over
the actions and prints the L2 and cosine drift per step. `--actions '0.1,0.2,…;0.3,…'` supplies real
actions; `--ref-check tests/fixtures/ref/lewm-pusht` replays the PyTorch reference and exits non-zero
on a mismatch.

### `jepa-quantize` — make a file smaller

```bash
build/jepa-quantize models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
                    models/gguf/lejepa-vits16-pretrain-in1k-q8_0.gguf q8_0 -t 32
```

Types: `f16`, `q8_0`, `q6_k`, `q5_k`, `q5_1`, `q5_0`, `q4_1`, `q4_k`, `q4_0`. Only the 2-D weight
matrices change type; K-quants need row counts that are multiples of 256 and fall back per tensor
(`q4_k → q4_0`) where they are not. The file is written in two passes, so peak memory is one tensor
rather than the whole model. Sizes and per-type accuracy: [quantization](quantization.md); which one
to ship: [Accuracy → which dtype](accuracy.md#which-dtype-to-ship).

### `jepa-bench` — timing any graph

```bash
build/jepa-bench -m models/gguf/vjepa2-vitl-fpc64-256-f16.gguf --frames 64 --threads 32,96 --md
build/jepa-bench -m models/gguf/ijepa_vith14_1k-f16.gguf --gpu 0 --md
```

Modes are `encoder` (default), `head`, `predictor`, `lewm-step` and `lewm-rollout`. The input is
synthetic but deterministic — a seeded uint8 stream put through the model's own `jepa.pre.*`
normalisation — so the tool runs wherever a GGUF is, with no fixtures and no Python.

### Running on a GPU

With a `-DJEPA_CUDA=ON` build, `jepa-embed`, `jepa-classify`, `jepa-worldmodel` and `jepa-bench` take
`--gpu [N]`:

```bash
build-cuda/jepa-embed -m models/gguf/ijepa_vith14_1k-f16.gguf -i img.jpg --gpu 0
JEPA_DEVICE=cuda:0 build-cuda/jepa-classify -m models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf --frames-npy clip.npy
```

Weights and activations live entirely on the card and `--threads` becomes irrelevant. Use f16 or a
quantized file there: there is no f32 parity tier on a GPU, and quantized weights are the fastest GPU
path.

## The C API

One header, [`include/jepa.h`](https://github.com/aselimc/jepa.cpp/blob/main/include/jepa.h) — opaque
handles, plain structs, no C++ types in the interface. Load a model, make a context, `jepa_encode`,
then pool, predict or classify. The complete reference is on the [C API page](api.md).

```c
jepa_model   * m = jepa_model_load("models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf", false);
jepa_context * c = jepa_context_new(m, jepa_context_default_params());

int h = 0, w = 0;
float * x = jepa_preprocess_image_file(m, "img.jpg", &h, &w);   // NCTHW, N = T = 1
jepa_input  in  = { x, 1, 3, 1, h, w };
jepa_output enc = {0}, cls = {0};
jepa_encode(c, &in, &enc);        // enc.data = [n_tokens, embed_dim]
jepa_pool_cls(m, &enc, &cls);     // cls.data = [embed_dim]
// every .data above is the caller's: jepa_free(cls.data), jepa_free(enc.data), jepa_free(x)
```

### Python bindings

`pip install jepa-cpp` puts the same engine behind numpy — a thin wrapper over `include/jepa.h`,
with the preprocessing, the graph and every floating-point operation still on the C side. The wheel
carries one self-contained shared library, so it needs neither ggml nor a compiler. A source build,
`pip install ./python` from a recursive clone, is the route to a `-march=native` or CUDA-enabled
library, and `$JEPA_CPP_LIB` points an installed package at one built by hand.

```python
import jepa_cpp

with jepa_cpp.Model("models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf", threads=32) as m:
    tokens  = m.encode("img.jpg")                 # [197, 384] float32
    feature = m.encode("img.jpg", pool="cls")     # [384]
    batch   = m.encode(["a.jpg", "b.jpg"], pool="cls")   # [2, 384], one encoder graph

with jepa_cpp.Model("models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf", device="cuda:0") as m:
    print(m.classify(frames_thwc_uint8).top(5))   # [(index, label, probability), ...]
```

`Model` also carries `pool`, `predict`, `classify`, `lewm_project`, `lewm_predict` and
`lewm_rollout`, and the file's metadata as properties; `jepa_cpp._api` is the unwrapped header, one
Python name per C name. On a CPU the bindings reproduce `jepa-embed`'s output bit for bit, which
`python/tests/test_parity.py` gates against the tools and the golden dumps alike. The package's own
reference is [`python/README.md`](https://github.com/aselimc/jepa.cpp/blob/main/python/README.md).

## Tests

```bash
ctest --test-dir build
```

Nine suites: five parity and predictor replays of the PyTorch golden dumps, plus `batch` (batched vs
per-item bit-exactness), `ops` (3-D RoPE against generated vectors), `attn` (flash vs naive attention
against a double-precision reference) and `backend` (GPU graph validation and CPU/GPU agreement,
which exits 0 with a skip line when the build has no GPU, so one `ctest` invocation covers both kinds
of machine). The parity suites register at CMake configure time and need `models/gguf/` and
`tests/fixtures/ref/` populated, so re-run `cmake` once after downloading and converting. The
methodology is in [Architecture → testing and parity](architecture.md#testing-and-parity-methodology).

There is **no build or test CI**: `.github/workflows/` holds only the documentation job, so `ctest`
runs on developer machines. A CPU build-and-test job and a compile-only CUDA job are the obvious
additions whenever CI is set up.

## Documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve          # live preview at http://127.0.0.1:8000
```

The site deploys to GitHub Pages on every push to `main` (`.github/workflows/docs.yml`). `docs/api.md`
is generated from `include/jepa.h` by `scripts/gen_api_md.py`; `docs/benchmarks.md` and the generated
blocks of `docs/accuracy-image.md` / `docs/accuracy-video.md` come from their measurement scripts and
are not edited by hand.
