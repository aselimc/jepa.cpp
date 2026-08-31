# Accuracy

Fidelity against PyTorch, and task accuracy on real datasets. Every figure on this page is copied from
a committed artifact; each section names its source.

## Environment

| | |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, AVX-512 |
| Build | gcc 13.3.0, `-O3 -march=native`, ggml `36da5713`, `GGML_LLAMAFILE=ON` |
| GPU | one NVIDIA RTX 4500 Ada Generation, compute 8.9, 24 GB; CUDA 13.0.88, driver 580.173.02 |
| Reference | torch 2.13.0+cpu, transformers 5.16.1, float32, 32 threads |
| Date | 2026-08-31 |

Two metrics recur. **Cosine** is per token of `last_hidden_state` unless a column says otherwise;
`cos mean` is the worst per-sample mean, `cos med` the median token, `cos min` the single worst token
of any sample. **`rel_max` = max|a−b| / max|b|** over the compared tensor — the metric that catches a
per-row scale error, which cosine is structurally blind to.

## f32 is exact

`tests/test-parity` against the PyTorch golden dumps, stored reference input, 32 threads, CPU backend,
worst sample per model.

| model | samples | cos mean | worst token | `rel_max` | derived tensors |
|---|---|---|---|---|---|
| LeJEPA ViT-S/16 | 8 images | 1.000000 | 1.000000 | 1.2e-05 | `cls` 1.000000 |
| LeWM ViT-Ti/14 | 2 images + 3-frame sequence | 1.000000 | 1.000000 | 1.0e-06 | `emb`, `emb_seq` 1.000000 |
| I-JEPA ViT-H/14 | 8 images | 1.000000 | 1.000000 | 7.9e-05 | `pooled_mean` 1.000000 |
| V-JEPA 2 ViT-L, 2 048 tok | 2 clips × 16 f | 1.000000 | 0.999999 | 7.5e-04 | `pooled_mean` 1.000000 |
| V-JEPA 2 ViT-L, 8 192 tok | 2 clips × 64 f | 1.000000 | 0.999990 | 1.2e-03 | `pooled_mean` 1.000000 |
| V-JEPA 2 ViT-L SSv2 | 2 clips × 16 f | 1.000000 | 0.999999 | 7.5e-04 | logits 1.000000, top-1/top-5 exact |
| V-JEPA 2.1 ViT-B, 4 608 tok | 2 clips × 16 f | 1.000000 | 1.000000 | 8.7e-05 | `pooled_mean` 1.000000 |
| V-JEPA 2.1 ViT-B, 576 tok | 2 COCO images | 1.000000 | 1.000000 | 6.2e-05 | `pooled_mean` 1.000000 |

Predictors at f32 (`tests/test-predictor`): the V-JEPA 2 ViT-L masked predictor reaches 1.0000000 mean
and worst with `rel_max` 3.4e-06–3.9e-06 over 2 048 rows; the V-JEPA 2.1 image-modality predictor
1.0000000 with `rel_max` 7.6e-05; LeWM `pred_next` and `pred_seq` 1.0000000 with `rel_max` 3.5e-07.

So the tubelet patchify, both RoPE table layouts, `interpolate_rope`, the modality vectors, the image
tokenizer, the attentive pooler and the classifier are bit-faithful. Preprocessing is checked
separately and is bit-exact against torchvision's antialiased uint8 resize.

*Source: [parity.md](parity.md#results-image-models-stored-reference-input-encoder-all-fixture-samples)
and [parity.md](parity.md#results-video-encoders-v-jepa-2-v-jepa-21).*

## f16 and q8_0 on the CPU

| model | ftype | cos mean | cos med | cos min | derived |
|---|---|---|---|---|---|
| LeJEPA ViT-S/16 | f16 | 0.999999 | – | 0.999994 | `cls` ≥ 0.9998 |
| LeJEPA ViT-S/16 | q8_0 | 0.999263 | – | 0.995193 | `cls` ≥ 0.9997 |
| LeWM ViT-Ti/14 | f16 | 1.000000 | – | 1.000000 | `emb` 1.000000 |
| LeWM ViT-Ti/14 | q8_0 | 0.999913 | – | 0.999895 | `emb_seq` 0.999895 |
| I-JEPA ViT-H/14 | f16 | 0.999984 | – | 0.997583 | `pooled_mean` ≥ 0.9998 |
| I-JEPA ViT-H/14 | q8_0 | 0.987843 | – | 0.432576 | `pooled_mean` 0.999748 |
| V-JEPA 2 ViT-L, 2 048 tok | f16 | 0.997144 | 0.999897 | 0.5088 | `pooled_mean` 0.999991 |
| V-JEPA 2 ViT-L, 2 048 tok | q8_0 | 0.966128 | 0.996770 | 0.2305 | `pooled_mean` 0.999876 |
| V-JEPA 2 ViT-L SSv2 | f16 | 0.997144 | 0.999897 | 0.5088 | pooled 0.999897, logits 0.999935, top-1/top-5 exact |
| V-JEPA 2 ViT-L SSv2 | q8_0 | 0.966128 | 0.996770 | 0.2305 | pooled 0.996645, logits 0.998501, top-1/top-5 exact |
| V-JEPA 2.1 ViT-B, 4 608 tok | f16 | 0.999952 | 0.999991 | 0.9697 | `pooled_mean` 1.000000 |
| V-JEPA 2.1 ViT-B, 4 608 tok | q8_0 | 0.999076 | 0.999578 | 0.8302 | `pooled_mean` 0.999986 |

All nine image files and all fifteen video rows pass their family's thresholds, on the stored input and
on jepa.cpp's own preprocessing alike.

**V-JEPA 2 ViT-L scatters individual tokens under f16 rounding.** That is a property of the
checkpoint's activation range, not an engine defect: running the numpy executable specification with
f16 weights and float32 activations reproduces the same effect, and the damage is spread across the
whole network — holding any single matrix at f16 while quantizing the rest leaves the worst token at
0.29–0.49. Everything downstream is unaffected: `pooled_mean` stays ≥ 0.9998, the predictor output
0.9998, the logits 0.9999 with identical top-1 and top-5. **Use f32 for dense per-token work on that
model**; f16 is correct everywhere else, and pooled features are safe at q8_0 on every model.

The same effect in miniature explains the image rows: the tokens that degrade are the ones with the
smallest variance before the final LayerNorm, which divides by that variance and amplifies whatever
error quantization injected. Only 0.4 % of I-JEPA's tokens fall below 0.99 at q8_0.

*Source: [parity.md](parity.md#what-the-f32-rows-prove-and-why-f16-tokens-scatter) and
[quantization.md](quantization.md#reading-the-numbers).*

## Backends and precision

A CUDA build runs the same graphs, and three backend differences are outside jepa.cpp's control: the
CUDA "F32" matmul is really TF32, CUDA flash attention always converts K/V to F16 and accumulates PV
in `half2`, and `ggml_norm` uses a one-pass variance there. **There is no f32 tier on a GPU** — an f32
GGUF is judged with its family's f16 bars, and `test-parity` says so in its header. f16 and quantized
files keep their own bars; with `GGML_PREC_F32` (the GPU default) the f16 matmul term is 2.6e-05, i.e.
better than the CPU's own 3.0e-04.

Encoders on CUDA, worst sample per file, stored-input pass:

| model | ftype | cos mean | cos med | cos min | `rel_max` | derived (worst) |
|---|---|---|---|---|---|---|
| LeJEPA ViT-S/16 | f16 | 0.999994 | 0.999996 | 0.9998 | 5.8e-03 | `pooled_mean` 1.000000 |
| LeJEPA ViT-S/16 | q8_0 | 0.999277 | 0.999426 | 0.9966 | 3.5e-02 | `cls` 0.999808 |
| LeWM ViT-Ti/14 | f16 | 1.000000 | 1.000000 | 1.0000 | 7.1e-04 | `emb` 1.000000 |
| LeWM ViT-Ti/14 | q8_0 | 0.999978 | 0.999980 | 0.9999 | 8.9e-03 | `emb` 0.999923 |
| I-JEPA ViT-H/14 | f16 | 0.999788 | 0.999996 | **0.9613** | 9.1e-02 | `pooled_mean` 0.999994 |
| I-JEPA ViT-H/14 | q8_0 | 0.990211 | 0.999443 | 0.6453 | 3.8e-01 | `pooled_mean` 0.999725 |
| V-JEPA 2 ViT-L fpc64, 8 192 tok | f16 | 0.994793 | 0.999696 | 0.3557 | 4.7e-01 | `pooled_mean` 0.999984 |
| V-JEPA 2 ViT-L fpc64, 8 192 tok | q8_0 | 0.967114 | 0.996680 | 0.1938 | 6.5e-01 | `pooled_mean` 0.999889 |
| V-JEPA 2 ViT-L SSv2 | f16 | 0.997052 | 0.999870 | 0.4352 | 3.6e-01 | pooled 0.999899 |
| V-JEPA 2 ViT-L SSv2 | q8_0 | 0.967114 | 0.997084 | 0.1938 | 6.5e-01 | pooled 0.996336 |
| V-JEPA 2.1 ViT-B/384, 4 608 tok | f16 | 0.999951 | 0.999985 | 0.9769 | 5.1e-02 | `pooled_mean` 0.999999 |
| V-JEPA 2.1 ViT-B/384, 4 608 tok | q8_0 | 0.999073 | 0.999568 | 0.8396 | 1.3e-01 | `pooled_mean` 0.999983 |

**22 of the 24 encoder files pass**, and all 9 predictor rows do. The two that do not —
`lejepa-vits16-q4_k` (derived `cls` 0.9851 against the low-bit tier's 0.99) and
`vjepa2-vitl-fpc16-256-ssv2-q4_k` (logits 0.9781, pooled 0.9807) — fail identically on the CPU: q4_k
is below 8 bits per weight and is not a parity configuration for those two models. Every other q4_k
row passes on the GPU, which matters because q4_k is the fastest GPU path there. The classifier
reproduces the reference top-1 and top-5 exactly at f32, f16 and q8_0.

Two properties of this table are worth stating plainly.

- **I-JEPA ViT-H's worst token is the outlier**: 0.9976 on the CPU at f16 against **0.9613** on a GPU
  (0.9723 with `--no-flash`, i.e. an F32 attention path). It is the F16 PV accumulator plus TF32
  landing on one high-norm row — reference row norm 39.23 against a 32.85 mean. LeJEPA and LeWM stay
  at 0.9998 and 0.99999 on the same backend, and I-JEPA's median stays at 0.999996. The image-family
  worst-token and mean bars are correspondingly looser on the GPU than on the CPU.
- **The V-JEPA 2 ViT-L f32 file on CUDA behaves like its f16 file**: worst token 0.3549 against
  0.3557, where on the CPU the same f32 file is exact. That is the "no f32 tier" statement measured
  end to end.

Predictors on CUDA take the naive attention path at `head_dim` 32 and are *more* accurate than a flash
kernel would be — genuinely F32 end to end, `rel_max` 1.7e-03–2.8e-03 against the PyTorch dump, the
same order as the CPU's 1.3e-03–1.7e-03. LeWM's structural self-consistency checks (causal-prefix
equality, rollout against predict) are bit-identical on the GPU at all three dtypes.

*Source: [parity.md](parity.md#results-encoders-on-cuda0) and
[parity.md](parity.md#results-predictors-on-cuda0), `GGML_PREC_F32` on every `mul_mat`.*

## Imagenette k-NN

10 ImageNet classes, frozen features, k = 20 cosine k-NN plus a nearest-centroid vote. Nothing is
trained — the encoders are used exactly as shipped and the "classifier" is a similarity vote over
gallery features. jepa.cpp decodes the JPEGs itself with `stb_image`, so this is the real input path
rather than a fixture replay.

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

Over those two models and both LeJEPA features — ten comparisons — no jepa.cpp row is further than
**0.28 pp** from its PyTorch row, and none further than **0.13 pp** at f32, f16 or q8_0.

The f32 row is not at 100 % agreement because `stb_image` and PIL/libjpeg differ by ±1–2 levels on
about 2 % of pixels. That decoder floor, `1 − cos` of 6.3e-6 to 1.7e-5, is larger than what f16 adds,
which is why the f32 and f16 rows are indistinguishable from each other.

*Source: [accuracy-image.md](accuracy-image.md#results), 32 threads, PyTorch and jepa.cpp passes
alternated inside one 87-minute sweep on an idle box. Machine-readable twin:
`tests/results/accuracy-image.json`.*

## UCF-101 k-NN

10 classes, gallery 300 clips, 105 val+test query clips, 16 frames per clip decoded once with PyAV and
read by both backends, so they see identical pixels. Same protocol: frozen pooled features, k = 20
cosine k-NN and a nearest-centroid vote.

| model | backend | dtype | kNN top-1 % | centroid top-1 % | kNN agree % | centroid agree % | feat cos |
|---|---|---|---:|---:|---:|---:|---:|
| V-JEPA 2 ViT-L fpc64 | PyTorch | f32 | 88.6 | 95.2 | — | — | — |
| | jepa.cpp | f16 | 89.5 | 95.2 | 99.0 | **100.0** | 0.999996 |
| | jepa.cpp | q8_0 | 89.5 | 95.2 | 99.0 | **100.0** | 0.999886 |
| V-JEPA 2.1 ViT-B/384 | PyTorch | f32 | 88.6 | 86.7 | — | — | — |
| | jepa.cpp | f32 | 88.6 | 86.7 | **100.0** | **100.0** | 1.000000 |
| | jepa.cpp | f16 | 89.5 | 86.7 | 99.0 | **100.0** | 1.000000 |
| | jepa.cpp | q8_0 | 89.5 | 86.7 | 99.0 | **100.0** | 0.999988 |

Where a jepa.cpp row reads *higher* than PyTorch — 89.5 against 88.6 % — that is one clip out of 105.

Every k-NN disagreement is a tie in the neighbour set: the two backends agree on 19 of the 20 nearest
gallery clips and the 20th and 21st are separated by *less* cosine than the two backends' similarities
to the query differ by, so which one lands inside the neighbourhood is decided by round-off. The
parameter-free metric confirms it — nearest-centroid agreement is **100 % for every model and every
dtype**. Over all 405 clips the worst single clip at any dtype still matches PyTorch to cosine
0.999643.

Feature fidelity over all 405 clips: V-JEPA 2.1 ViT-B at f32 reaches mean cosine 1.0000000 with the
largest single-component difference at 7.13e-05; at f16 0.9999999; at q8_0 0.9999874. V-JEPA 2 ViT-L
reaches 0.9999946 at f16 and 0.9998754 at q8_0.

*Source: [accuracy-video.md](accuracy-video.md), 32 threads, PyTorch and jepa.cpp stages alternated in
one sweep on an idle box. Machine-readable twin: `tests/results/accuracy-video.json`.*

## SSv2 head agreement

The same 105 clips through encoder + attentive pooler + 174-way classifier, scored as *agreement with
PyTorch's argmax*. SSv2 labels are meaningless on UCF clips, so this is a backend-fidelity measurement,
not a task accuracy — but it is 105 independent 174-way argmaxes over a full stack, which a handful of
parity fixtures cannot cover.

| dtype | top-1 agreement % | top-5 overlap % | PyTorch top-1 inside jepa.cpp's top-5 % | max abs logit diff | logit cos |
|---|---:|---:|---:|---:|---:|
| f16 | **99.0** | 99.4 | 100.0 | 0.1885 | 0.999970 |
| q8_0 | 94.3 | 97.0 | 100.0 | 1.0421 | 0.998922 |

This is the sharpest measurement in the repository: an argmax over 174 classes has no averaging to
hide behind, unlike a pooled 1024-vector whose cosine stays at 0.9999. q8_0 moves 6 of 105 top-1
decisions while the PyTorch top-1 stays inside jepa.cpp's top-5 on 100 % of clips, so the ranking is
intact and only near-ties at the top move. **Use f16 for head and classifier work.**

*Source: [accuracy-video.md](accuracy-video.md#ssv2-classification-head-backend-fidelity).*

## Which dtype to ship

| use | CPU | CUDA | evidence |
|---|---|---|---|
| default deployment | **f16** | **f16** | pooled, CLS and logit outputs ≥ 0.9998 on every model; 0.5× f32 |
| pooled features, retrieval, k-NN, LeWM rollouts | **q8_0** | **q8_0** | pooled cosine ≥ 0.99995 on every model, 0.53× f16; within 0.05 pp of PyTorch on Imagenette and within one clip on UCF-101 |
| classification with the attentive-pool head | **f16** | **f16** | 99.0 % top-1 agreement on 105 clips against 94.3 % at q8_0 |
| dense per-token features | **f32** for V-JEPA 2 ViT-L, q8_0 elsewhere | not available — use the CPU | that checkpoint's worst token is 0.51 at f16 and 0.23 at q8_0 while its `pooled_mean` stays ≥ 0.9998; a GPU has no f32 tier |
| smallest footprint, pooled ≈ 0.99 is enough | **q4_k**, advisory | **q4_k**, advisory *and the fastest path there* | 0.29× f16, pooled cosine 0.992–0.998; `test-parity` reports but does not gate files below 8 bits per weight |

The accuracy column of this table does not change with the backend; the *speed* column does. On the
CPU q4_k is the slowest type; on CUDA it ties q8_0 and beats f16 (see
[Performance → quantization and speed](performance.md#quantization-and-speed)). Two q4_k files miss
even the advisory bar on both backends: `lejepa-vits16-q4_k` and `vjepa2-vitl-fpc16-256-ssv2-q4_k`.

Pooling hundreds of tokens averages quantization noise away, which is why q4_k still lands within
0.28 pp on Imagenette while individual tokens of the same file are visibly worse.

*Source: [quantization.md](quantization.md#recommendation) for the weight-only study (GGUF
dequantized in Python and pushed through the numpy reference graph), refined by
[parity.md](parity.md) at the engine level and confirmed by both k-NN benchmarks above.*

## Reproduce

```bash
# per-file parity against the golden dumps, CPU then GPU
build/test-parity models/gguf/<model>.gguf tests/fixtures/ref/<ref> --threads 32 [--json out.json]
build-cuda/test-parity models/gguf/<model>.gguf tests/fixtures/ref/<ref> --gpu 0
build/test-predictor --vjepa2 models/gguf/<model>.gguf --ref tests/fixtures/ref/<ref> --threads 32

# the dataset benchmarks (datasets first: scripts/download_datasets.sh, ~400 MB into data/)
.venv/bin/python scripts/bench_accuracy_image.py all --out-json tests/results/accuracy-image.json
.venv/bin/python scripts/bench_accuracy_video.py all --out-json tests/results/accuracy-video.json \
    --out-md docs/accuracy-video.md
scripts/render_accuracy_md.py --write docs/accuracy-image.md

# weight-only quantization accuracy, no C++ graph involved
.venv/bin/python scripts/gguf_dequant_selftest.py --gguf models/gguf/<model>-q8_0.gguf \
    --ref tests/fixtures/ref/<model> --threads 32
```
