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

Python is needed only to convert checkpoints yourself. Downloading the already-converted GGUFs needs
nothing but `curl`, and nothing in the inference path uses Python at all.

```bash
uv venv .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install transformers safetensors numpy gguf pillow huggingface_hub av
```

`torch` and `transformers` are needed only by the converters that read Hugging Face checkpoints and
by the reference-dump scripts; `gguf` and `numpy` are the minimum for conversion itself.

## Get the models

The converted files are published on Hugging Face under **[jepacpp](https://huggingface.co/jepacpp)** —
one repository per model, five types each (`f32`, `f16`, `q8_0`, `q4_0`, `q4_k`), named exactly as the
converter and `jepa-quantize` write them. `scripts/download_models.sh` fetches them into `models/gguf/`
(git-ignored) with `hf download`, or plain `curl` if the `hf` CLI is neither on `PATH` nor in `.venv`:

```bash
scripts/download_models.sh small                # LeJEPA + LeWorldModel + V-JEPA 2.1 at f16 (290 MiB)
scripts/download_models.sh all                  # all seven at f16 (3.3 GiB)
scripts/download_models.sh --ftype q8_0 ijepa   # one model, one type
scripts/download_models.sh --ftype all levjepa  # f32, f16, q8_0, q4_0 and q4_k of one model
scripts/download_fixtures.sh                    # golden dumps + media for the parity tests (665 MB)
```

| model | name | repository | f16 | q8_0 | q4_k |
|---|---|---|---:|---:|---:|
| LeJEPA ViT-S/16 | `lejepa` | [`lejepa-vits16-pretrain-in1k-GGUF`](https://huggingface.co/jepacpp/lejepa-vits16-pretrain-in1k-GGUF) | 42.2 MiB | 23.2 | 13.0 |
| LeWorldModel Push-T | `lewm` | [`lewm-pusht-GGUF`](https://huggingface.co/jepacpp/lewm-pusht-GGUF) | 37.7 MiB | 23.1 | 15.3 |
| V-JEPA 2.1 ViT-B/16 @384 | `vjepa21` | [`vjepa2_1-vitb-384-GGUF`](https://huggingface.co/jepacpp/vjepa2_1-vitb-384-GGUF) | 209.7 MiB | 113.3 | 61.9 |
| LeVJEPA ViT-L/16 | `levjepa` | [`levjepa-vitl16-GGUF`](https://huggingface.co/jepacpp/levjepa-vitl16-GGUF) | 578.7 MiB | 310.3 | 166.3 |
| V-JEPA 2 ViT-L/16 (fpc64) | `vjepa2` | [`vjepa2-vitl-fpc64-256-GGUF`](https://huggingface.co/jepacpp/vjepa2-vitl-fpc64-256-GGUF) | 622.5 MiB | 332.8 | 178.3 |
| V-JEPA 2 ViT-L SSv2 | `vjepa2-ssv2` | [`vjepa2-vitl-fpc16-256-ssv2-GGUF`](https://huggingface.co/jepacpp/vjepa2-vitl-fpc16-256-ssv2-GGUF) | 717.1 MiB | 383.2 | 205.1 |
| I-JEPA ViT-H/14 | `ijepa` | [`ijepa_vith14_1k-GGUF`](https://huggingface.co/jepacpp/ijepa_vith14_1k-GGUF) | 1206.2 MiB | 643.7 | 343.7 |

A single file without the script, either way round:

```bash
hf download jepacpp/lewm-pusht-GGUF lewm-pusht-f16.gguf --local-dir models/gguf
curl -L -o models/gguf/lewm-pusht-f16.gguf \
     https://huggingface.co/jepacpp/lewm-pusht-GGUF/resolve/main/lewm-pusht-f16.gguf
```

Every model card lists the sha256 of all five files, so `sha256sum -c` verifies a download. Re-run
`cmake` once `models/gguf/` and `tests/fixtures/ref/` are populated — the parity tests register at
configure time. Which type to use: [Accuracy → which dtype](accuracy.md#which-dtype-to-ship); `f16` is
the default, `q8_0` halves it again for pooled work, `f32` is the bit-exact tier, and the two 4-bit
files sit below the parity bars. The other types `jepa-quantize` can produce (`q4_1`, `q5_0`, `q5_1`,
`q5_k`, `q6_k`, measured in [quantization](quantization.md)) are not published; make them locally.

The golden reference dumps live in the companion dataset
[`jepacpp/jepa.cpp-fixtures`](https://huggingface.co/datasets/jepacpp/jepa.cpp-fixtures), which
`scripts/download_fixtures.sh` pulls into `tests/fixtures/ref/`. The COCO images and Kinetics clips the
dumps were computed on are not redistributed there; the same script fetches those from their own
sources into `tests/fixtures/media/`.

### Licences

Each GGUF carries its source checkpoint's licence verbatim in `general.license` and the origin in
`general.source_url` (`jepa-info --kv` prints both). **I-JEPA and LeVJEPA are non-commercial.**
jepa.cpp's own code is MIT.

| model | licence | source of that statement | redistribution of the converted weights |
|---|---|---|---|
| I-JEPA ViT-H/14 | **CC BY-NC 4.0** | [model card](https://huggingface.co/facebook/ijepa_vith14_1k) + the full CC text as [LICENSE](https://github.com/facebookresearch/ijepa/blob/main/LICENSE) | permitted for **non-commercial** use, with attribution to Meta and a note that the file is modified |
| LeJEPA ViT-S/16 | Apache-2.0 | [model card](https://huggingface.co/OK-AI/lejepa-vits16-pretrain-in1k) metadata (the repo ships no `LICENSE` file) | permitted, with attribution |
| LeWorldModel Push-T | MIT | [model card](https://huggingface.co/quentinll/lewm-pusht) + [le-wm](https://github.com/lucas-maes/le-wm) | permitted, keep the notice |
| V-JEPA 2 ViT-L/16 (fpc64) | MIT | [model card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) + [LICENSE](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE) | permitted, keep the notice |
| V-JEPA 2 ViT-L SSv2 | MIT | [model card](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) + the same LICENSE | permitted, keep the notice |
| V-JEPA 2.1 ViT-B/16 @384 | MIT | the [checkpoint table](https://github.com/facebookresearch/vjepa2#v-jepa-21-pretrained-checkpoints) of `facebookresearch/vjepa2`; the `.pt` on `dl.fbaipublicfiles.com` carries no metadata of its own, so this is the repository-level grant | permitted, keep the notice |
| LeVJEPA ViT-L/16 | **CC BY-NC 4.0** | [model card](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large) metadata (the repo ships no `LICENSE` file); the weights are trained from scratch, so the restriction is the publisher's own | permitted for **non-commercial** use, with attribution to galilai-group and a note that the file is modified |

None of the seven is gated and none carries an acceptable-use policy. The `jepacpp` repositories mirror
these terms one for one; the fixtures dataset is CC BY-NC 4.0, because it stores outputs of the two
non-commercial models.

### Converting instead of downloading

`scripts/download_models.sh --convert` fetches the *source* checkpoints into `models/` (git-ignored)
instead of the GGUFs. `small` takes LeJEPA, LeWorldModel and V-JEPA 2.1 (~1.8 GB); `all` adds I-JEPA,
both V-JEPA 2 files and LeVJEPA (~8 GB); a single name fetches one. This is the path that needs the
Python environment above.

```bash
scripts/download_models.sh --convert small   # or: all | ijepa | vjepa2 | vjepa2-ssv2 | vjepa21 | levjepa | lewm | lejepa
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

The published files come from exactly these commands plus `jepa-quantize`; every model card records the
one that produced it and the jepa.cpp commit it ran at, so a local conversion is reproducible against
the published sha256.

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
jepa_context * c = jepa_context_new(m, NULL);
jepa_encode_image_file(c, "img.jpg");
const float * f = jepa_pooled(c, JEPA_POOL_CLS);
```

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

CI runs on every push to `main` and every pull request (`.github/workflows/ci.yml`): a build and
`ctest` on Ubuntu 22.04, Ubuntu 24.04 and macOS arm64, a Metal build on macOS, an MSVC build on
Windows, and an ASAN+UBSAN build with `detect_leaks=1`. A runner has no converted weights, so those
jobs run the two suites that need none — `ops` and `attn`. A separate `parity` job fetches the
LeJEPA and LeWorldModel GGUFs and their reference dumps from the Hugging Face hub and runs the
`parity-*`, `predictor-lewm`, `batch` and `backend` suites on top; while those files are unpublished
it says so and stops. The same workflow gates the generated documentation — `gen_api_md.py --check`,
both figure scripts, `render_accuracy_md.py --check`, `mkdocs build --strict` and `shellcheck`.

The CUDA build is not in CI, because GitHub-hosted runners have no NVIDIA device: `-DJEPA_CUDA=ON`
and the `backend` suite on a GPU stay a developer-machine check. `docs/benchmarks.md` has no CI gate
either — it is rebuilt from a sweep directory measured on a known-idle box, and
`scripts/gen_benchmarks_md.py --check` is the local gate for it.

## Documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve          # live preview at http://127.0.0.1:8000
```

The site deploys to GitHub Pages on every push to `main` (`.github/workflows/docs.yml`). `docs/api.md`
is generated from `include/jepa.h` by `scripts/gen_api_md.py`; `docs/benchmarks.md` and the generated
blocks of `docs/accuracy-image.md` / `docs/accuracy-video.md` come from their measurement scripts and
are not edited by hand.
