# Getting started

From a clean checkout to a feature vector: build the engine, create the one-time Python environment
that converts checkpoints, convert a model, run a tool.

## Build

Requirements: CMake ≥ 3.16, Ninja, a C++17 compiler. ggml is a submodule, so clone recursively.
Nothing else is needed to build; `ffmpeg` on the `PATH` is an optional **run-time** extra that lets
`jepa-embed` and `jepa-classify` take a video file directly (`--video`, below).

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
| V-JEPA 2 ViT-g/16 (fpc64) | `vjepa2-vitg` | [`vjepa2-vitg-fpc64-256-GGUF`](https://huggingface.co/jepacpp/vjepa2-vitg-fpc64-256-GGUF) | 1974.9 MiB | 1056.7 | 564.8 |
| V-JEPA 2-AC ViT-g (world model) | `vjepa2-ac` | [`vjepa2-ac-vitg-GGUF`](https://huggingface.co/jepacpp/vjepa2-ac-vitg-GGUF) | 2519.0 MiB | 1344.1 | 717.4 |

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

`scripts/hf_publish.py` is what maintains those repositories: `cards` regenerates every model card from
the GGUF metadata and the artifacts behind this site, `upload` pushes a set one repository at a time, and
`check` compares the hub's sha256 and size for every published file — and every card — against the local
ones, exiting non-zero on any drift.

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
| V-JEPA 2 ViT-g/16 (fpc64) | `vjepa2-vitg` | `python scripts/convert.py --family vjepa2 --src models/facebook/vjepa2-vitg-fpc64-256 --ftype f16` |
| V-JEPA 2-AC ViT-g | `vjepa2-ac` | `python scripts/convert.py --family vjepa2_ac --src models/vjepa2_ac/vjepa2-ac-vitg.pt --out models/gguf/vjepa2-ac-vitg-f16.gguf --ftype f16` |

### Planning with V-JEPA 2-AC

The world model answers "what would the scene look like if I did this?", and a planner turns that
into "what should I do?". `jepa-worldmodel --plan` is the loop from the V-JEPA 2-AC paper: it encodes
the current frame and a goal frame, samples action sequences, rolls them all out in one batched graph
per step and keeps the ones whose predicted frame lands closest to the goal.

```bash
scripts/download_models.sh vjepa2-ac                     # 2.5 GiB, the f16 bundle

# ctx.png is what the robot sees now, goal.png is where it should end up (256x256 RGB)
build/jepa-worldmodel --plan -m models/gguf/vjepa2-ac-vitg-f16.gguf \
    --image ctx.png --goal goal.png \
    --state '0.580,-0.002,0.248,-3.067,0.031,-1.913,0.997' \
    --samples 64 --topk 8 --cem-steps 10 --horizon 2 --gpu 0
```

`--state` is the arm's current 7-d pose (xyz, three Euler angles, gripper); it defaults to zeros,
which is fine for exploring but not for a real arm. The output is the best energy per iteration and
the chosen action sequence:

```
CEM: K=64 topk=8 iterations=10 horizon=2 maxnorm=0.050 momentum(mean/std)=0.25/0.95

iteration     best energy
0                0.463301
...
plan (2 steps, 7-d actions):
  step 0:  0.03974  0.03481  0.04139  0.00000  0.00000  0.00000  0.39948
```

The three zeros are the rotation dimensions, which this planner does not sample — that is the
reference's action space, not a limitation of the port. A demo you can run without a robot is in the
V-JEPA 2 repository: `notebooks/franka_example_traj.npz` holds a real Franka trajectory, and
`scripts/dump_reference.py --model vjepa2-ac-vitg` writes its first and last frames into the parity
fixtures. `--noise-npy` replays recorded random draws so a run can be compared against a PyTorch one
byte for byte.

Use **f16**. A rollout compounds, and at q8_0 and below the plan degrades — at q4_k on a GPU the
planner picks a different action ([quantization](quantization.md)).

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

# a video file -> pooled feature: 16 frames sampled uniformly over the whole clip
build/jepa-embed -m models/gguf/vjepa2_1-vitb-384-f16.gguf \
    --video tests/fixtures/media/archery.mp4 --frames 16 --pool mean -t 32

# the same clip from a THWC uint8 .npy that scripts/video_frames.py wrote (identical output)
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
    --video-list clips.txt -o feats.npy -t 32       # or --frames-list for .npy clips
```

`--pool` selects `mean` (mean of the patch tokens), `cls`, `lewm` (the world-model state projection)
or `none` (the full `[N, D]` token map). Clip frames may have different source sizes: each is resized
and centre-cropped on its own before the planes are concatenated.

#### Where the frames come from

There are two routes into a video model, and they produce the same tensor:

* **`--video clip.mp4`** (also `--video-list list.txt`) decodes the container by running `ffmpeg`
  and keeps `--frames` frames sampled uniformly over the whole clip, endpoints included —
  `idx = round(linspace(0, T_total - 1, n))`. `--frames` defaults to the model's own
  `jepa.enc.n_frames`. Any container ffmpeg can read works (mp4, webm, mkv, avi, mov …).
* **`--frames-npy clip.npy`** (also `--frames-list list.txt`) reads a THWC uint8 array that
  `scripts/video_frames.py` wrote — the scripted route the accuracy harnesses use, because they want
  one decode shared by both backends and a cached, indexed frame set.

`ffmpeg` is a **run-time** dependency of `jepa-embed` and `jepa-classify` only: nothing links against
it, the build never looks for it, and a checkout without it loses `--video` and nothing else (the
error names the binary and how to install it). Both routes decode with libswscale and sample with the
same formula, so `--video clip.mp4` and `--frames-npy` on that script's output of the same clip agree
**byte for byte** — measured over the fixture clips and 40 Something-Something-v2 clips, `ctest -R
video`. `--dump-frames out.npy` writes the sampled frames in that same THWC uint8 layout if you want
to diff the two yourself.

### `jepa-classify` — a clip to labels

```bash
build/jepa-classify -m models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf \
    --video tests/fixtures/media/archery.mp4 -k 5 -t 32 --time     # or --frames-npy clip.npy
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

### `jepa-server` — the same engine over HTTP

```bash
build/jepa-server -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
    --threads 8 --workers 4 --max-batch 8            # http://127.0.0.1:8080
```

```
jepa-server 0.3.0: lejepa-vits16-pretrain-in1k-f16 (hfvit, f16, CPU) on http://127.0.0.1:8080
  workers 4 x 8 threads | max-batch 8 | max-wait 5 ms | local files refused
```

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
     -d "{\"input\": {\"b64\": \"$(base64 -w0 tests/fixtures/media/coco_000000000139.jpg)\"}}"
```

`POST /v1/embeddings` is OpenAI-shaped, `POST /classify` returns top-k labels, `POST /rollout` runs a
world-model rollout or the V-JEPA 2-AC planner, and `GET /health`, `GET /metrics` and
`GET /v1/models` describe the process. Requests of the same shape are folded into one encoder graph,
which changes the scheduling and not the arithmetic. The server binds `127.0.0.1` unless `--host`
says otherwise and has neither TLS nor authentication, so it is a component to put behind something.
Endpoints, flags, the Python client and the measured throughput: [Serving](serving.md).

### Running on a GPU

With a `-DJEPA_CUDA=ON` build, `jepa-embed`, `jepa-classify`, `jepa-worldmodel`, `jepa-bench` and
`jepa-server` take `--gpu [N]`:

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

Thirteen suites: five parity and predictor replays of the PyTorch golden dumps, plus `batch` (batched
vs per-item bit-exactness), `ops` (3-D RoPE against generated vectors), `attn` (flash vs naive
attention against a double-precision reference), `video` (`--video` decode and sampling against the
reference dumps' PyAV frames), `backend` (GPU graph validation and CPU/GPU agreement, which exits 0
with a skip line when the build has no GPU, so one `ctest` invocation covers both kinds of machine),
`errors` (malformed GGUFs it forges itself, and the NULL / oversized / budget guards of the inference
entry points) and `threads` (the [thread contract](architecture.md#robustness), N threads sharing one
model). The parity suites register at CMake configure time and need `models/gguf/` and
`tests/fixtures/ref/` populated, so re-run `cmake` once after downloading and converting. `errors`
and `threads` need neither and always register. The methodology is in
[Architecture → testing and parity](architecture.md#testing-and-parity-methodology), and what the
loader validates is in [Architecture → robustness](architecture.md#robustness).

CI runs on every push to `main` and every pull request (`.github/workflows/ci.yml`): a build and
`ctest` on Ubuntu 22.04, Ubuntu 24.04 and macOS arm64, a Metal build on macOS, an MSVC build on
Windows, and an ASAN+UBSAN build with `detect_leaks=1`. A runner has no converted weights, so those
jobs run the four suites that need none — `ops`, `attn`, `errors` and `threads` — and build (never
run) the fuzz target. A separate `parity` job fetches the
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
