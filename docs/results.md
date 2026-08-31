# Results

Every measured table in one place. All numbers were produced on an idle AMD Ryzen Threadripper PRO
7995WX (96 cores / 192 threads, AVX-512), gcc 13.3.0 `-O3 -march=native`, ggml `36da5713`,
`GGML_LLAMAFILE=ON`, 2026-08-31; PyTorch baselines: torch 2.13.0+cpu, transformers 5.16.1, float32,
32 threads. Each section names the source document that carries the methodology and the raw artifacts;
machine-readable twins live in `tests/results/{benchmarks,accuracy-image,accuracy-video}.json`.

## Speed

### Encoder (ms per image / clip) — from [benchmarks.md](benchmarks.md)

`ms` is the `ggml_backend_graph_compute` wall time for one item. The PyTorch column is the mean over the
reference samples, or the median after a cold first sample (I-JEPA, LeJEPA); [benchmarks.md](benchmarks.md)
names the samples feeding each baseline.

| model | shape | tokens | f32 t=32 | f16 t=32 | q8_0 t=32 | f16 t=96 | PyTorch t=32 | f16 speedup |
|---|---|---:|---:|---:|---:|---:|---:|---|
| I-JEPA ViT-H/14 | 224² | 256 | 174 | 147 | 129 | 113 | 250 | **1.70×** (2.21× @96) |
| LeJEPA ViT-S/16 | 224² | 197 | 13.1 | 12.8 | 11.3 | – | 15.4 | **1.20×** |
| LeWM ViT-Ti/14 | 224² | 257 | 9.2 | 9.8 | 9.1 | – | 16.8 | **1.72×** |
| V-JEPA 2 ViT-L SSv2 | 16 f 256² | 2 048 | 943 | 823 | 793 | 564 | 1051¹ | n/a¹ |
| V-JEPA 2 ViT-L fpc64 | 16 f 256² | 2 048 | 941 | 821 | 794 | 567 | 1293² | n/a² |
| V-JEPA 2 ViT-L fpc64 | 64 f 256² | 8 192 | 7020 | 6388 | 6482 | 4027 | 10114² | n/a² |
| V-JEPA 2.1 ViT-B/384 | 384² | 576 | 70.0 | 60.3 | 58.5 | 52.7 | 110 | **1.82×** (2.09× @96) |
| V-JEPA 2.1 ViT-B/384 | 16 f 384² | 4 608 | 826 | 853 | 914 | 636 | 908 | **1.06×** |
| V-JEPA 2.1 ViT-B/384 | 64 f 384² | 18 432 | 9050 | 9036 | 9487 | 5040 | – | – |

¹ the SSv2 reference forward is encoder + pooler + classifier — compared like-for-like in the next table.
² the fpc64 reference forward always runs the predictor too, so it is an upper bound and no speedup is
claimed against it.

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

[parity.md](parity.md) times the same 2 048-token V-JEPA 2 predictor on the real fixture clips at
452 / 326 ms (f16) against a PyTorch predictor call of 2524 ms — **~5.6×**, and 40–55 % of the cost of
the encoder pass on the same clip.

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

q8_0 is 0.53× of f16 and q4 is 0.29× (8.5 and 4.5 bits/weight); file sizes per type:
[quantization.md](quantization.md).

## Correctness

### f32 is exact, everywhere — from [parity.md](parity.md)

`tests/test-parity` against the PyTorch golden dumps (worst sample per model, stored reference input,
32 threads). Cosine is per token of `last_hidden_state`; `rel_max` = max|a−b| / max|b|.

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

## Accuracy on real data

### Imagenette k-NN — from [accuracy-image.md](accuracy-image.md)

10 classes, k = 20 cosine k-NN + nearest centroid over frozen features, no training. jepa.cpp decodes
the JPEGs itself with `stb_image` — the real input path, not a fixture replay.

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

### UCF-101 subset k-NN — from [accuracy-video.md](accuracy-video.md)

10 classes, 300-clip gallery, 105 val+test query clips, 16 frames per clip decoded once and shared by
both backends.

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

### SSv2 head fidelity

The same 105 clips through encoder + attentive pooler + 174-way classifier, scored as *agreement with
PyTorch's argmax* (SSv2 labels are meaningless on UCF clips):

| dtype | top-1 agreement % | top-5 overlap % | PyTorch top-1 in our top-5 % | logit cos |
|---|---:|---:|---:|---:|
| f16 | **99.0** | 99.4 | 100.0 | 0.999970 |
| q8_0 | 94.3 | 97.0 | 100.0 | 0.998922 |

## Which dtype to ship

From [quantization.md](quantization.md) (weight-only study: GGUF dequantised in Python, pushed through
the numpy reference graph), refined by the engine-level measurements in [parity.md](parity.md) (dense
per-token V-JEPA 2 ViT-L → f32) and [accuracy-video.md](accuracy-video.md) (SSv2 head → f16), and
confirmed by the two k-NN benchmarks above.

| use | type | evidence |
|---|---|---|
| default deployment, parity | **f16** | pooled/CLS/logit outputs ≥ 0.9998 on every model; 0.5× f32 |
| pooled features, retrieval, k-NN, LeWM rollouts | **q8_0** | pooled cosine ≥ 0.99995 on every model, 0.53× f16; within 0.05 pp of PyTorch on Imagenette and within one clip on UCF-101 |
| classification with the attentive-pool head | **f16** | 99.0 % top-1 agreement on 105 clips vs 94.3 % at q8_0 |
| dense per-token features | **f32** for V-JEPA 2 ViT-L, q8_0 elsewhere | ViT-L worst token 0.51 at f16 and 0.23 at q8_0, while its `pooled_mean` stays ≥ 0.9998 |
| smallest footprint, pooled ≈ 0.99 is enough | **q4_k**, advisory | 0.29× f16, pooled cosine 0.992–0.998; `test-parity` does not gate files below 8 bits/weight |

Pooling hundreds of tokens averages quantisation noise away — q4_k still lands within 0.28 pp on
Imagenette — but `vjepa2-vitl-fpc16-256-ssv2-q4_k` misses even the advisory bar and is reported as a FAIL.
