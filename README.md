# jepa.cpp

CPU **inference** for the JEPA family in plain C/C++ on top of [ggml](https://github.com/ggml-org/ggml) —
no training code, no Python at run time, no other runtime dependency.
One GGUF file carries a whole model bundle: the ViT **encoder**, and optionally the **masked predictor**
(V-JEPA 2 / 2.1), the **attentive-pool classifier head** (SSv2, 174 classes) or the **LeWorldModel
predictor + rollout**. Shipping today: I-JEPA ViT-H/14, V-JEPA 2 ViT-L/16 (encoder-only and SSv2),
V-JEPA 2.1 ViT-B/16 @384 (image *and* video paths), LeJEPA-style ViTs (`hfvit`) and LeWorldModel Push-T.
Every f32 file reproduces its PyTorch reference **exactly** (cosine 1.000000 on every token of every
fixture); f16 and q8_0 are measured against it, and `tools/jepa-quantize` produces q8_0 / q4_k / q5_k /
q6_k from an f32 or f16 file.

**Status** — `ctest` 7/7 pass (3 parity, 2 predictor, ops, attention). All 18 `test-parity` runs — 9 image
files and 9 video files, each on the stored reference input *and* on our own preprocessing — PASS their
family thresholds, as do all 20 predictor rows; resize + crop + normalisation are bit-exact against
torchvision for every pipeline. Every number on this page is copied from a committed artifact in `docs/`
or `tests/results/`.

## Supported models

| family | checkpoint | arch | params¹ | native input | runs | f32 parity | licence |
|---|---|---|---:|---|---|---|---|
| `ijepa` | [`facebook/ijepa_vith14_1k`](https://huggingface.co/facebook/ijepa_vith14_1k) | ViT-H/14, D=1280, 32L | 631 M | 224², 256 tokens | encoder | cos 1.000000, `rel_max` 7.9e-05 | CC-BY-NC-4.0 |
| `hfvit` | [`OK-AI/lejepa-vits16-pretrain-in1k`](https://huggingface.co/OK-AI/lejepa-vits16-pretrain-in1k) | ViT-S/16, D=384, 12L, CLS | 22 M | 224², 197 tokens | encoder | cos 1.000000, `rel_max` 1.2e-05 | Apache-2.0 |
| `lewm` | [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht) | ViT-Ti/14 D=192 + 6L/192-d predictor | 18 M | 224², 257 tokens | encoder + projector + world-model rollout | cos 1.000000, `rel_max` 1.0e-06 (enc) / 3.5e-07 (pred) | MIT |
| `vjepa2` | [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | ViT-L/16, D=1024, 24L + 12L/384-d predictor | 326 M | 16–64 f @256², 2 048–8 192 tokens | encoder + masked predictor | cos 1.000000, `rel_max` 7.5e-04 (2 048 tok) / 1.2e-03 (8 192) | MIT |
| `vjepa2` | [`facebook/vjepa2-vitl-fpc16-256-ssv2`](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) | same + attentive pooler, 174 classes | 375 M | 16 f @256², 2 048 tokens | encoder + predictor + head | cos 1.000000, logits 1.000000, top-1/top-5 exact | MIT |
| `vjepa2_1` | [`vjepa2_1_vitb_dist_vitG_384.pt`](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt) | ViT-B/16 @384, D=768, 12L + predictor | 110 M | 384² image (576 tok) **and** 16–64 f @384² | encoder + predictor (image & video modality) | cos 1.000000, `rel_max` 8.7e-05 (clip) / 6.2e-05 (image) | MIT |

¹ f32 tensor bytes ÷ 4 (`docs/benchmarks.md` Memory table; `jepa-info` for the two f32-source models).
Parity columns: `docs/parity.md` (worst sample per model, stored reference input, 32 threads).
The licence of the converted GGUF is the licence of its source checkpoint — `general.license` is carried
into the file and printed by `jepa-info`. **I-JEPA is CC-BY-NC-4.0: non-commercial use only.**

## Benchmarks

*All tables in this section: `docs/benchmarks.md`, 2026-08-31, AMD Ryzen Threadripper PRO 7995WX
(96 cores / 192 threads, AVX-512), gcc 13.3.0 `-O3 -march=native`, ggml `36da5713`, `GGML_LLAMAFILE=ON`,
idle box. `ms` = `ggml_backend_graph_compute` wall time for one item; PyTorch = torch 2.13.0+cpu,
transformers 5.16.1, float32, 32 threads, model forward only.*

### Encoder (ms per image / clip)

| model | shape | tokens | f32 t=32 | f16 t=32 | q8_0 t=32 | f16 t=96 | PyTorch t=32 | f16 speedup |
|---|---|---:|---:|---:|---:|---:|---:|---|
| I-JEPA ViT-H/14 | 224² | 256 | 174 | 147 | 129 | 113 | 250 | **1.70×** (2.21× @96) |
| LeJEPA ViT-S/16 | 224² | 197 | 13.1 | 12.8 | 11.3 | – | 15.4 | **1.20×** |
| LeWM ViT-Ti/14 | 224² | 257 | 9.2 | 9.8 | 9.1 | – | 16.8 | **1.72×** |
| V-JEPA 2 ViT-L SSv2 | 16 f 256² | 2 048 | 943 | 823 | 793 | 564 | 1051² | n/a² |
| V-JEPA 2 ViT-L fpc64 | 16 f 256² | 2 048 | 941 | 821 | 794 | 567 | 1293³ | n/a³ |
| V-JEPA 2 ViT-L fpc64 | 64 f 256² | 8 192 | 7020 | 6388 | 6482 | 4027 | 10114³ | n/a³ |
| V-JEPA 2.1 ViT-B/384 | 384² | 576 | 70.0 | 60.3 | 58.5 | 52.7 | 110 | **1.82×** (2.09× @96) |
| V-JEPA 2.1 ViT-B/384 | 16 f 384² | 4 608 | 826 | 853 | 914 | 636 | 908 | **1.06×** |
| V-JEPA 2.1 ViT-B/384 | 64 f 384² | 18 432 | 9050 | 9036 | 9487 | 5040 | – | – |

² the SSv2 reference forward is encoder + pooler + classifier — compared like-for-like in the next table.
³ the fpc64 reference forward always runs the predictor too, so it is an upper bound and no speedup is
claimed against it. The PyTorch column is the mean over the manifest samples, or the median after a cold
first sample (I-JEPA, LeJEPA); `docs/benchmarks.md` names the samples feeding each baseline.

**Quantisation buys memory, not time.** q8_0 is 0.93–1.14× of f16 and q4 is a *loss* on the wide matmuls
(I-JEPA q4_k 198 ms vs f16 147 ms; V-JEPA 2.1 384² q4_k 90.7 vs 60.3) — llamafile's accelerated sgemm
covers F32/F16/Q8_0 and the K-quants fall back to ggml's generic vec-dot.

### End-to-end video classification (V-JEPA 2 ViT-L SSv2, 16 f 256², 2 048 tokens)

| ftype | threads | encoder ms | head ms | total ms | PyTorch ms | speedup |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 32 | 943 | 107 | 1050 | 1051 | 1.00× |
| f16 | 32 | 823 | 99.0 | 922 | 1051 | **1.14×** |
| f16 | 96 | 564 | 67.1 | 631 | 1051 | **1.66×** |
| q8_0 | 32 | 793 | 98.2 | 891 | 1051 | **1.18×** |

### Masked predictor and world model

Predictor worst case: context = target = *every* token, i.e. 2 × tokens through the 12-layer 384-d predictor.

| model | tokens | f32 t=32 | f16 t=32 | q8_0 t=32 | f16 t=96 |
|---|---:|---:|---:|---:|---:|
| V-JEPA 2 ViT-L SSv2 | 2 048 | 333.1 | 340.8 | 336.8 | 208.0 |
| V-JEPA 2 ViT-L fpc64 | 2 048 | 338.3 | 343.8 | 332.7 | 207.4 |
| V-JEPA 2.1 ViT-B/384 | 4 608 | 1296.7 | 1303.6 | 1272.6 | 854.4 |

`docs/parity.md` times the same 2 048-token V-JEPA 2 predictor on the real fixture clips at 452 / 326 ms
(f16) against a PyTorch predictor call of 2524 ms — **~5.6×**, and 40–55 % of the cost of the encoder pass
on the same clip.

The LeWM Push-T world model (D=192, 3-frame causal window) is launch-bound: one full-window
`jepa_lewm_predict` takes 0.911 / 0.918 / 0.739 ms at f32 / f16 / q8_0, and `jepa_lewm_rollout` runs
0.817 / 0.865 / 0.744 ms **per step** — 1225 / 1156 / 1344 steps/s, encoder excluded.

### Memory (weights resident, MiB)

| model | f32 | f16 | q8_0 | q4_k | peak RSS, f16 at its largest shape |
|---|---:|---:|---:|---:|---:|
| I-JEPA ViT-H/14 | 2406 | 1206 | 644 | 344 | 1230 (256 tok) |
| LeJEPA ViT-S/16 | 83 | 42 | 23 | 13 | 52 (197 tok) |
| LeWM Push-T | 69 | 38 | 23 | 15 | 47 (257 tok) |
| V-JEPA 2 ViT-L SSv2 | 1432 | 717 | 383 | 205 | 808 (2 048 tok) |
| V-JEPA 2 ViT-L fpc64 | 1243 | 622 | 333 | 178 | 1034 (8 192 tok) |
| V-JEPA 2.1 ViT-B/384 | 419 | 210 | 113 | 62 | 948 (18 432 tok) |

q8_0 is 0.53× of f16 and q4 is 0.29× (8.5 and 4.5 bits/weight); file sizes per type: `docs/quantization.md`.

## Accuracy

### f32 is exact, everywhere

`tests/test-parity` against the PyTorch golden dumps (`docs/parity.md`, worst sample per model, stored
reference input, 32 threads). Cosine is per token of `last_hidden_state`; `rel_max` = max|a−b| / max|b|.

| model | samples | cos mean | worst token | `rel_max` | derived tensors |
|---|---|---|---|---|---|
| LeJEPA ViT-S/16 | 8 images | 1.000000 | 1.000000 | 1.2e-05 | `cls` 1.000000 |
| LeWM ViT-Ti/14 | 2 images + 3-frame seq | 1.000000 | 1.000000 | 1.0e-06 | `emb`, `emb_seq` 1.000000 |
| I-JEPA ViT-H/14 | 8 images | 1.000000 | 1.000000 | 7.9e-05 | `pooled_mean` 1.000000 |
| V-JEPA 2 ViT-L, 2 048 tok | 2 clips × 16 f | 1.000000 | 0.999999 | 7.5e-04 | `pooled_mean` 1.000000 |
| V-JEPA 2 ViT-L, 8 192 tok | 2 clips × 64 f | 1.000000 | 0.999990 | 1.2e-03 | `pooled_mean` 1.000000 |
| V-JEPA 2 ViT-L SSv2 | 2 clips × 16 f | 1.000000 | 0.999999 | 7.5e-04 | logits 1.000000, top-1/top-5 exact |
| V-JEPA 2.1 ViT-B, 4 608 tok | 2 clips × 16 f | 1.000000 | 1.000000 | 8.7e-05 | `pooled_mean` 1.000000 |
| V-JEPA 2.1 ViT-B, 576 tok | 2 COCO images | 1.000000 | 1.000000 | 6.2e-05 | `pooled_mean` 1.000000 |

Predictors, f32 (`tests/test-predictor`): V-JEPA 2 ViT-L masked predictor 1.0000000 / 1.0000000 with
`rel_max` 3.4e-06–3.9e-06 over 2 048 rows; V-JEPA 2.1 image-modality predictor 1.0000000, `rel_max`
7.6e-05; LeWM `pred_next` / `pred_seq` 1.0000000, `rel_max` 3.5e-07.

### Frozen-feature k-NN — does it survive a real dataset?

**Imagenette**, 10 classes, k = 20 cosine k-NN + nearest centroid over frozen features, no training
(`docs/accuracy-image.md`, 2026-08-31, same box, 32 threads; jepa.cpp decodes the JPEGs itself with
`stb_image` — the real input path, not a fixture replay).

| model | backend | dtype | kNN top-1 % | centroid top-1 % | agreement % | mean feat cos |
|---|---|---|---:|---:|---:|---:|
| I-JEPA ViT-H/14, 3 925 queries, gallery 2 000 | PyTorch | f32 | **95.36** | 92.25 | — | 1 |
| | jepa.cpp | f16 | 95.31 | 92.25 | 99.82 | 0.999944 |
| | jepa.cpp | q8_0 | 95.31 | 92.31 | 99.75 | 0.999774 |
| | jepa.cpp | q4_k | 95.08 | 92.25 | 99.11 | 0.995049 |
| LeJEPA ViT-S/16 (`cls`), gallery 9 469 | PyTorch | f32 | **94.45** | 87.59 | — | 1 |
| | jepa.cpp | f32 | 94.52 | 87.64 | 99.85 | 0.999983 |
| | jepa.cpp | f16 | 94.55 | 87.64 | 99.82 | 0.999983 |
| | jepa.cpp | q8_0 | 94.50 | 87.64 | 99.87 | 0.999843 |
| | jepa.cpp | q4_k | 94.22 | 87.67 | 99.08 | 0.988725 |

Over those two models and both LeJEPA features (ten comparisons) no jepa.cpp row is further than
**0.28 pp** from its PyTorch row, and none further than **0.13 pp** at f32/f16/q8_0. The f32 row is not
100 % agreement because `stb_image` and PIL/libjpeg differ by ±1–2 levels on ~2 % of pixels — that
decoder floor (1−cos = 6.3e-6 … 1.7e-5) is larger than what f16 adds.

**UCF-101 subset**, 10 classes, 300-clip gallery, 105 val+test query clips, 16 frames per clip decoded
once and shared by both backends (`docs/accuracy-video.md`, same box and date).

| model | backend | dtype | kNN top-1 % | centroid top-1 % | kNN agree % | centroid agree % | feat cos |
|---|---|---|---:|---:|---:|---:|---:|
| V-JEPA 2 ViT-L fpc64 | PyTorch | f32 | 88.6 | 95.2 | — | — | — |
| | jepa.cpp | f16 | 89.5 | 95.2 | 99.0 | **100.0** | 0.999996 |
| | jepa.cpp | q8_0 | 89.5 | 95.2 | 99.0 | **100.0** | 0.999886 |
| V-JEPA 2.1 ViT-B/384 | PyTorch | f32 | 88.6 | 86.7 | — | — | — |
| | jepa.cpp | f32 | 88.6 | 86.7 | **100.0** | **100.0** | 1.000000 |
| | jepa.cpp | f16 | 89.5 | 86.7 | 99.0 | **100.0** | 1.000000 |
| | jepa.cpp | q8_0 | 89.5 | 86.7 | 99.0 | **100.0** | 0.999988 |

Every k-NN disagreement is a tie in the neighbour set (19 of 20 neighbours shared; the 20th↔21st gallery
clip is separated by less cosine than the two backends differ by), and the parameter-free centroid metric
agrees **100 % everywhere**. Over all 405 clips the worst single clip at any dtype still matches PyTorch
to cosine 0.999643.

**SSv2 head fidelity** — the same 105 clips through encoder + attentive pooler + 174-way classifier,
scored as *agreement with PyTorch's argmax* (SSv2 labels are meaningless on UCF clips):

| dtype | top-1 agreement % | top-5 overlap % | PyTorch top-1 in our top-5 % | logit cos |
|---|---:|---:|---:|---:|
| f16 | **99.0** | 99.4 | 100.0 | 0.999970 |
| q8_0 | 94.3 | 97.0 | 100.0 | 0.998922 |

### Which dtype to ship

From `docs/quantization.md` (weight-only study: GGUF dequantised in Python, pushed through the numpy
reference graph) confirmed by the two k-NN benchmarks above.

| use | type | evidence |
|---|---|---|
| default deployment, parity | **f16** | pooled/CLS/logit outputs ≥ 0.9998 on every model; 0.5× f32 |
| pooled features, retrieval, k-NN, LeWM rollouts | **q8_0** | pooled cosine ≥ 0.99995 on every model, 0.53× f16; within 0.05 pp of PyTorch on Imagenette and within one clip on UCF-101 |
| classification with the attentive-pool head | **f16** | 99.0 % top-1 agreement on 105 clips vs 94.3 % at q8_0 |
| dense per-token features | **f32** for V-JEPA 2 ViT-L, q8_0 elsewhere | ViT-L worst token 0.51 at f16 and 0.23 at q8_0, while its `pooled_mean` stays ≥ 0.9998 |
| smallest footprint, pooled ≈ 0.99 is enough | **q4_k**, advisory | 0.29× f16, pooled cosine 0.992–0.998; `test-parity` does not gate files below 8 bits/weight |

Pooling hundreds of tokens averages quantisation noise away — q4_k still lands within 0.28 pp on
Imagenette — but `vjepa2-vitl-fpc16-256-ssv2-q4_k` misses even the advisory bar and is reported as a FAIL.

## Quickstart

```bash
git clone --recursive https://github.com/<you>/jepa.cpp && cd jepa.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 32
```

Python is needed **only** to convert checkpoints and to regenerate the golden references
(observed: torch 2.13.0+cpu, transformers 5.16.1, gguf 0.19.0, numpy 2.5.2, av 18.1.0):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install transformers safetensors numpy gguf pillow huggingface_hub av

scripts/download_models.sh small        # LeJEPA ViT-S, LeWM, V-JEPA 2.1 ViT-B  (`all` adds I-JEPA + V-JEPA 2 ViT-L)
scripts/download_fixtures.sh            # 8 COCO images + 6 Kinetics-mini clips -> tests/fixtures/media

.venv/bin/python scripts/convert.py --family hfvit    --src models/OK-AI/lejepa-vits16-pretrain-in1k --ftype f16
.venv/bin/python scripts/convert.py --family lewm     --src models/quentinll/lewm-pusht              --ftype f32
.venv/bin/python scripts/convert.py --family vjepa2_1 --src models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt
# -> models/gguf/<basename>-<ftype>.gguf
```

```bash
# what is in a file
build/jepa-info models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf --no-tensors

# image -> feature vector
build/jepa-embed -m models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
    -i tests/fixtures/media/coco_000000000139.jpg --pool cls -t 32 --time
#   1 frame(s) 640x426 -> 224x224, 197 tokens | preprocess 2.7 ms | encode 22.2 ms (9263 tokens/s)
#   cls [1 x 384] |x|=9.0502 [-0.44056, -0.28211, -2.38613, ...]

# clip -> pooled feature (16 frames, 4608 tokens)
build/jepa-embed -m models/gguf/vjepa2_1-vitb-384-f16.gguf \
    --frames-npy tests/fixtures/ref/vjepa2_1-vitb-384/archery_f16.frames_u8.npy --pool mean -t 32 --time
# ... or straight from images:  --as-video -i f0.jpg -i f1.jpg -i f2.jpg

# clip -> SSv2 top-5
build/jepa-classify -m models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf \
    --frames-npy tests/fixtures/ref/vjepa2-vitl-fpc16-256-ssv2/archery_f16.frames_u8.npy -k 5 -t 32 --time
#   preprocess 29.5 ms | encoder 964.4 ms (32 threads) | head 95.3 ms | 2124 tokens/s
#    1.  59.36%  [ 90] Pulling two ends of [something] but nothing happens
#    2.  14.87%  [162] Trying to bend [something unbendable] so nothing happens

# image -> world-model state -> 8-step action rollout
build/jepa-worldmodel -m models/gguf/lewm-pusht-f16.gguf \
    --image tests/fixtures/media/coco_000000000139.jpg --random-actions 8 -t 32
#   rollout: 8 steps, 8.40 ms total, 1.05 ms/step
build/jepa-worldmodel -m models/gguf/lewm-pusht-f32.gguf --ref-check tests/fixtures/ref/lewm-pusht -t 32
#   PASS (threshold cos >= 0.9999)   — emb / pred_next / emb_seq / pred_seq all 1.0000000

# f32/f16 -> q8_0 (also q4_0 q4_1 q5_0 q5_1 q4_k q5_k q6_k; --dry-run prints the plan)
build/jepa-quantize models/gguf/lejepa-vits16-pretrain-in1k-f32.gguf \
                    models/gguf/lejepa-vits16-pretrain-in1k-q8_0.gguf q8_0 -t 32
#   tensor data: 86662656 bytes (82.6 MiB) -> 24288768 bytes (23.2 MiB), 0.280x

# time one configuration (--threads 32,96 sweeps; --md emits a markdown row; --json writes an artifact)
build/jepa-bench -m models/gguf/vjepa2_1-vitb-384-f16.gguf --frames 16 --threads 32 --md
scripts/bench_all.sh 32                 # the whole matrix -> tmp/bench/ + docs/benchmarks.md
```

C API: [`include/jepa.h`](include/jepa.h) — `jepa_model_load` / `jepa_context_new` /
`jepa_preprocess_*` / `jepa_encode` / `jepa_pool_{mean,cls}` / `jepa_predict{,_ex,_mod}` / `jepa_head` /
`jepa_lewm_{project,predict,rollout}`. Opaque handles, plain structs, no C++ in the header.

## How it works

**One GGUF = one model bundle.** `general.*` + `jepa.enc.*` / `jepa.pred.*` / `jepa.head.*` / `jepa.pre.*`
describe dims, position scheme, activation, class labels and the exact preprocessing; the loader builds the
graph from metadata alone, so a new checkpoint of a known family needs no C++ change. Full key list:
[`docs/gguf-schema.md`](docs/gguf-schema.md).

**One shared ViT graph** for every family:
`tokens = patch_embed(pixels) [+ pos] [+ cls/reg] [+ modality]`, then pre-LN blocks
(`x += attn(ln1(x)); x += ffn(ln2(x))`, GELU-erf, fused qkv, optional layer-scale), then a final LayerNorm.
The patch "conv" is a host-side rearrangement into `[N, C·T·P·P]` plus one `ggml_mul_mat`. Families differ
only in the tokenizer (2-D patch, 2×16×16 tubelet, or 2.1's extra 1×16×16 image embed) and the position
scheme (`sincos2d`, `sincos3d`, learned, `rope3d`).

**3-D RoPE, tiled vs interleaved.** V-JEPA 2 (HF *and* Meta) **tiles** the d/2 per-axis frequencies —
`table[j] = f(pos·ω_{j mod d/2})`, so the two members of a rotation pair see different angles (Meta's own
comment calls it a bug, kept for checkpoint compatibility). V-JEPA 2.1 uses `repeat_interleave` — a true
rotation — plus `interpolate_rope`, which rescales h/w by `(16−1)/(grid−1)`. Using the wrong layout gives
cosine **0.63** (V-JEPA 2) / **0.91** (2.1) on *every* token, which is why `jepa.enc.rope_freq_layout` is
stored per file. ggml's `ggml_rope_multi` uses half-split rotation and cannot be used; `src/rope3d.cpp`
precomputes `[N, head_dim]` cos/sin tables and applies them with stock ops.

**Flash attention with an automatic K/V dtype.** Attention always goes through `ggml_flash_attn_ext`
(naive `mul_mat + soft_max_ext` would need a 15.3 GB score matrix at 18 k tokens against ~0.2 GB).
K/V defaults to **F32 for f32 files, F16 otherwise**: casting K/V to F16 costs ~3 digits of worst-token
cosine on I-JEPA ViT-H, whose activations reach ~2e4, while for f16/quantized files weight rounding
dominates anyway. The attentive pooler's single-query cross-attention always uses F32 K/V. Override with
`--kv-f32` / `--kv-f16`, or `--no-flash`.

**Bit-exact preprocessing.** `jepa_resize_antialias_u8` ports the integer path
`torchvision.transforms.v2.functional.resize(antialias=True)` runs on x86 (separable passes, int16 filter
weights, int32 accumulation, `clamp(acc>>prec, 0, 255)`); normalisation follows the HF fused form
`(px − 255·mean)/(255·std)`. Against the stored reference tensors, resize + crop + normalisation are
**bit-exact for every pipeline**, images and video alike; the only residual is JPEG decoding
(`stb_image` vs PIL/libjpeg, ±1–2 levels on ~2 % of pixels).
Deeper ggml notes — attention shapes, block-causal masks, matmul throughput:
[`docs/ggml-notes.md`](docs/ggml-notes.md).

## Testing and the parity protocol

```bash
ctest --test-dir build                 # 7 tests (they register only when GGUFs + ref dumps are present)
build/test-parity models/gguf/<model>.gguf tests/fixtures/ref/<ref> --threads 32 [--rgb-dir tmp/rgb] [--json out.json]
build/test-predictor --vjepa2 models/gguf/<model>.gguf --ref tests/fixtures/ref/<ref> --samples archery_f16
build/test-predictor --lewm  models/gguf/lewm-pusht-f32.gguf --ref tests/fixtures/ref/lewm-pusht
```

`scripts/dump_reference.py` writes the golden `.npy` dumps; `test-parity` feeds the model the **same
preprocessed tensor** the reference saw, then repeats with its own preprocessor, and reports per-token
cosine (mean / median / worst), `rel_max`, pooled and logit cosines and top-1/top-5 agreement.
Pass/fail bars are **table-driven per model family × file-type tier** (`POLICY` in `tests/test-parity.cpp`,
printed in the header of every run, tabulated in [`docs/parity.md`](docs/parity.md) → *Thresholds*). f32
files keep a hard "every token ≥ 0.9999" bar plus `REL(N) = max(1e-3, 1e-3·√(N/2048))`; f16/q8_0 video
files are gated on the *median* token cosine and on the derived tensors, because the V-JEPA 2 ViT-L f16
token tail is a property of the checkpoint, not of this engine (below). Files under 8 bits/weight are
advisory: results printed, only derived tensors (≥ 0.99) and the top-1 label gated.

## Limitations and roadmap

- **V-JEPA 2 ViT-L scatters individual tokens at f16/q8_0.** Worst token 0.51 (f16) / 0.23 (q8_0) while the
  mean token cosine stays ≥ 0.966, `pooled_mean` ≥ 0.9998 and the SSv2 logits ≥ 0.9985. Not graph noise: ggml rounds the *activations*
  of an f16 `mul_mat` to F16, and re-running the numpy spec with the same rounding reproduces the C++
  numbers to four digits and collapses the same low-norm token cluster. Use f32 for per-token work there.
- **q4 is below the parity bars.** `test-parity` reports it rather than gating it, and the SSv2 q4_k file is
  a FAIL. Fine for pooled retrieval; *slower* than q4_0 and than f16 on the wide matmuls — a memory win only.
- **No batching.** `jepa-embed` encodes one item per call, which is why PyTorch (batch 32) wins end-to-end
  on the two small image models — LeJEPA 67–75 img/s against 86–89, LeWM 85–101 against 190–206 — while
  jepa.cpp leads on I-JEPA (6.2 img/s at f16, 7.0 at q8_0 against 5.5) and on the video models.
- **Not converted yet:** V-JEPA 1 (`vjepa`, 3-D sincos), V-JEPA 2-AC (`ac` predictor kind, block-causal
  mask — the schema and the ggml recipe exist, the converter does not), the larger V-JEPA 2 / 2.1 and
  I-JEPA sizes, and the audio / VL JEPA variants. LeJEPA's own repo ships no weights; the converted file is
  the community `OK-AI/lejepa-vits16-pretrain-in1k`.
- **Real SSv2 accuracy is unmeasured** — the dataset is licence-gated, so `docs/accuracy-video.md` reports
  head *agreement with PyTorch* on 105 UCF clips instead. I-JEPA f32 image k-NN is skipped for budget
  (`docs/accuracy-image.md` → *Not measured*).

## Docs

[`docs/index.md`](docs/index.md) describes each one in a paragraph.
[`architecture.md`](docs/architecture.md) (layout, shared graph, family deltas, RoPE spec) ·
[`gguf-schema.md`](docs/gguf-schema.md) (every key and tensor name) ·
[`parity.md`](docs/parity.md) (cosine per model × dtype, thresholds, preprocessing) ·
[`benchmarks.md`](docs/benchmarks.md) (every timing above, 32 and 96 threads, memory) ·
[`quantization.md`](docs/quantization.md) (`jepa-quantize`, file sizes, per-type accuracy) ·
[`accuracy-image.md`](docs/accuracy-image.md) · [`accuracy-video.md`](docs/accuracy-video.md) (k-NN
benchmarks) · [`ggml-notes.md`](docs/ggml-notes.md) (flash attention, masks, matmul throughput) ·
[`scripts/jepa_convert/README.md`](scripts/jepa_convert/README.md) ·
[`tests/fixtures/README.md`](tests/fixtures/README.md).
Machine-readable twins of the tables: `tests/results/{benchmarks,accuracy-image,accuracy-video}.json`.

## Acknowledgements

[ggml](https://github.com/ggml-org/ggml) for the tensor library and the llamafile sgemm kernels;
**Meta FAIR** for I-JEPA, V-JEPA 2 and V-JEPA 2.1 and for publishing the reference code;
**OK-AI** for the LeJEPA ViT-S/16 release; **quentinll / le-wm** for the LeWorldModel Push-T checkpoint.

## Licence

This code: **MIT** (see [`LICENSE`](LICENSE)). Model weights keep the licence of the checkpoint they were
converted from — V-JEPA 2 / 2.1 and LeWorldModel MIT, LeJEPA Apache-2.0, **I-JEPA CC-BY-NC-4.0
(non-commercial)** — and that licence is carried in `general.license` inside every GGUF.
