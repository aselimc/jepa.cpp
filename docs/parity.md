# Parity — image ViT encoders (I-JEPA, LeJEPA/hfvit, LeWM) and video encoders (V-JEPA 2 / 2.1)

Numbers from `tests/test-parity` against the PyTorch golden dumps of `tests/fixtures/ref/<model>/`
(torch 2.13.0+cpu float32, transformers 5.16.1, 32 threads). Box: AMD Ryzen Threadripper PRO 7995WX
(96 cores / 192 threads, AVX-512), gcc 13.3.0, ggml @ 36da5713, `-O3 -march=native`, `GGML_LLAMAFILE=ON`.

Reproduce (from a checkout with `models/gguf/` and `tests/fixtures/ref/` populated):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 32
.venv/bin/python tests/dump_rgb_u8.py tmp/rgb                    # PIL-decoded pixels (optional, see below)
build/test-parity models/gguf/<model>.gguf tests/fixtures/ref/<ref> --threads 32 [--rgb-dir tmp/rgb] [--json out.json]
ctest --test-dir build                                           # parity-lejepa-vits16, parity-lewm-pusht,
                                                                 # parity-vjepa2_1-vitb-384-images, ops, attn (~11 s)
```

## Results — image models (stored reference input → encoder, all fixture samples)

“cos mean” is the worst per-sample mean of the per-token cosine, “cos min” the worst single token of
any sample, `rel_max` = max|a−b| / max|b| over the `last_hidden_state` (for LeWM the `emb`/`emb_seq`
projector outputs are checked too — 1.000000 everywhere, including the 3-frame causal `seq` sample).

| model | ftype | samples | cos mean | cos min | rel_max | ms/item t=32 | ms/item t=96 | PyTorch t=32 | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| lejepa-vits16 | f32 | 8 | 1.000000 | 1.000000 | 1.2e-05 | 14.0 | —¹ | 22.6 | 99 MiB |
| lejepa-vits16 | f16 | 8 | 0.999999 | 0.999994 | 1.2e-03 | 13.4 | —¹ | 22.6 | 59 MiB |
| lejepa-vits16 | q8_0 | 8 | 0.999263 | 0.995193 | 3.7e-02 | 12.6 | —¹ | 22.6 | 40 MiB |
| lewm-pusht | f32 | 3 | 1.000000 | 1.000000 | 1.0e-06 | 10.3 | —¹ | 18.1 | 88 MiB |
| lewm-pusht | f16 | 3 | 1.000000 | 1.000000 | 5.8e-04 | 10.7 | —¹ | 18.1 | 53 MiB |
| lewm-pusht | q8_0 | 3 | 0.999913 | 0.999895 | 1.5e-02 | 9.7 | —¹ | 18.1 | 39 MiB |
| ijepa-vith14-1k | f32 | 8 | 1.000000 | 1.000000 | 7.9e-05 | 185 | —¹ | 263.0 | 2433 MiB |
| ijepa-vith14-1k | f16 | 8 | 0.999984 | 0.997583 | 2.9e-02 | 156 | 122 | 263.0 | 1233 MiB |
| ijepa-vith14-1k | q8_0 | 8 | 0.987843 | 0.432576 | 5.0e-01 | 138 | —¹ | 263.0 | 671 MiB |

¹ t=96 re-measured only for the mul_mat-bound ijepa f16 (122 ms vs 156 at t=32); the small models are
launch-/LN-bound and within noise of their t=32 numbers. All timings with `GGML_LLAMAFILE=ON`
(1.3–3.2× faster matmuls than stock ggml, see `docs/ggml-notes.md` §5). For the LeWM `seq` sample the
compared tensor is `emb_seq`; q8_0 `cos min` rows reflect the low-variance-token amplification
analysed in `docs/quantization.md` — pooled/CLS/emb stay ≥ 0.9997 for every q8_0 file.

All 18 runs (9 files × {stored input, own preprocessing}) **PASS** the thresholds below; raw
per-sample JSON: `test-parity ... --json`.

## Results — video encoders (V-JEPA 2 / V-JEPA 2.1)

Same protocol; the stored `input` is 5-D (`NTCHW` for the HF V-JEPA 2 dumps, `NCTHW` for V-JEPA 2.1 —
`test-parity` reads the layout from the manifest and transposes) and is fed as **one clip**: the whole
`T/2 × H/16 × W/16` token grid goes through a single graph with 3-D RoPE. `cos med` is the median
per-token cosine (the gate for f16/quantized files, see "Thresholds"), `cos min` the single worst token.
Worst sample per model; both clips (archery, bowling) and, for V-JEPA 2.1, both COCO images are included.

| model | ftype | sample set | tokens | cos mean | cos med | cos min | rel_max | pooled | logits | top-1/top-5 | ms/clip t=32 | tokens/s | PyTorch t=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vjepa2-vitl-fpc64-256 | f16 | 2 clips × 16 f | 2048 | 0.997144 | 0.999897 | 0.5088 | 5.1e-01 | 0.999991 | – | – | 823 / 827 | 2477 | 1290² |
| vjepa2-vitl-fpc64-256 | f16 | 2 clips × 64 f | 8192 | 0.998322 | 0.999934 | 0.5971 | 3.2e-01 | 0.999997 | – | – | 7032 / 6386 | 1283 | 10113² |
| vjepa2-vitl-fpc64-256 | q8_0 | 2 clips × 16 f | 2048 | 0.966128 | 0.996770 | 0.2305 | 6.0e-01 | 0.999876 | – | – | 829 / 852 | 2405 | 1290² |
| vjepa2-vitl-fpc64-256 | q8_0 | 2 clips × 64 f | 8192 | 0.973126 | 0.996880 | 0.2188 | 5.4e-01 | 0.999925 | – | – | 7220 / 7203 | 1137 | 10113² |
| vjepa2-vitl-fpc16-256-ssv2 | **f32** | 2 clips × 16 f | 2048 | 1.000000 | 1.000000 | 0.999999 | **7.5e-04** | 1.000000 | 1.000000 | 2/2 · 5/5 | 1037 / 924 | 2216 | 1051³ |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 2 clips × 16 f | 2048 | 0.997144 | 0.999897 | 0.5088 | 5.1e-01 | 0.999897 | 0.999935 | 2/2 · 5/5 | 988 / 814 | 2515 | 1051³ |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 2 clips × 16 f | 2048 | 0.966128 | 0.996770 | 0.2305 | 6.0e-01 | 0.996645 | 0.998501 | 2/2 · 5/5 | 887 / 771 | 2656 | 1051³ |
| vjepa2_1-vitb-384 | **f32** | 2 clips × 16 f | 4608 | 1.000000 | 1.000000 | 1.000000 | **8.7e-05** | 1.000000 | – | – | 1073 / 909 | 5067 | 908 |
| vjepa2_1-vitb-384 | f32 | 2 images | 576 | 1.000000 | 1.000000 | 1.000000 | 6.2e-05 | 1.000000 | – | – | 71 / 70 | 8288 | 110 |
| vjepa2_1-vitb-384 | f16 | 2 clips × 16 f | 4608 | 0.999952 | 0.999991 | 0.9697 | 6.3e-02 | 1.000000 | – | – | 1125 / 875 | 5268 | 908 |
| vjepa2_1-vitb-384 | f16 | 2 images | 576 | 0.999989 | 0.999996 | 0.9994 | 6.9e-03 | 1.000000 | – | – | 66 / 63 | 9187 | 110 |
| vjepa2_1-vitb-384 | q8_0 | 2 clips × 16 f | 4608 | 0.999076 | 0.999578 | 0.8302 | 1.3e-01 | 0.999986 | – | – | 1067 / 878 | 5249 | 908 |
| vjepa2_1-vitb-384 | q8_0 | 2 images | 576 | 0.999729 | 0.999800 | 0.9954 | 3.5e-02 | 0.999985 | – | – | 63 / 59 | 9779 | 110 |

`pooled` = `pooled_mean` (mean over the tokens) for the two encoder-only models, the **attentive-pooler
output** (`pooled`, the classifier input) for SSv2; `logits`/top-k only exist for SSv2. Two `ms/clip`
numbers are given per row: the two samples in run order — the *first* clip of a process is 10–25 % slower
(weights are paged in on first touch), so the second is the steady-state figure and the one `tokens/s`
uses. The 13 rows come from 8 `test-parity` runs (a run covers all samples of one file); all of them
**PASS**, on the stored input and on our own preprocessing alike.

² the fpc64 manifest's `timing_s.forward_s` is one `VJEPA2Model` forward, which always runs the
**predictor** as well (its `predictor_last_hidden_state` comes from the same call), so it is not a
like-for-like encoder number — treat it as an upper bound.
³ the SSv2 reference forward is encoder + attentive pooler + classifier with the predictor skipped, i.e.
directly comparable to our encoder (814 ms) + head (96 ms, measured with `jepa-classify --time`) = 910 ms.

The 64-frame ViT-L clip is the one case that was also timed at **96 threads** (one run, as budgeted):
4125 / 4076 ms per clip, i.e. 2010 tokens/s and 1.57× the 32-thread throughput (2.5× the manifest's
PyTorch number, which includes the predictor — caveat ²). Peak RSS: 1183 MiB (f16, 8192 tokens),
872 MiB (q8_0), 1572 MiB (ssv2 f32), 454 MiB (2.1 f16), 336 MiB (2.1 q8_0).

### What the f32 rows prove, and why f16 tokens scatter

The f32 files (converted with `scripts/convert.py --family vjepa2{,_1} --ftype f32`) reproduce the
PyTorch reference **exactly**: cosine 1.000000 on every token of every clip, `rel_max` 7.5e-4 (ViT-L,
whose activations reach ±43) and 8.7e-5 (2.1 ViT-B), logits cosine 1.000000, top-1/top-5 identical. So
the tubelet patchify, the tiled vs interleaved RoPE tables, `interpolate_rope`, the modality vectors,
the image tokenizer and the attentive-pool head are all bit-faithful.

At f16 the *mean* stays at 0.9971–0.999999 and every pooled/logit output stays ≥ 0.9998, but individual
tokens drop far below that (worst 0.51 on V-JEPA 2 ViT-L, 0.97 on 2.1 ViT-B). That is **not** graph
noise:

* ggml converts the activations of an f16 `mul_mat` to F16 as well (the vec-dot type of an F16 weight
  matrix; llamafile's AVX-512 kernel only has an F16×F16 path — `sgemm.cpp` `case GGML_TYPE_F16` requires
  `Btype == GGML_TYPE_F16`), so every one of the 24 ViT-L layers rounds its activations to ~3 decimal
  digits.
* Re-running the executable numpy spec with that same rounding reproduces the C++ numbers to four digits
  and picks the *same* worst token:

  | ssv2 bowling_f16, f16 weights | mean cos | worst token | tokens < 0.999 |
  |---|---|---|---|
  | numpy spec, f32 activations (`vjepa2_numpy_ref.py`) | 0.9999968 | 0.999036 @1163 | 0 / 2048 |
  | numpy spec, F16 activations | 0.9971802 | 0.557581 @373 | 420 / 2048 |
  | **jepa.cpp f16** | 0.997144 | 0.508778 @373 | 414 / 2048 |
  | jepa.cpp f16, `--kv-f32` | 0.997139 | 0.558080 @373 | 421 / 2048 |
  | jepa.cpp f16, `--no-flash` | 0.997158 | 0.549800 @373 | 420 / 2048 |
* Flash attention is not involved (F32 K/V and the naive path give the same spread), and
  `docs/quantization.md` independently finds V-JEPA 2 ViT-L to be "by far the most token-sensitive
  model" (the tokens with the smallest pre-final-LN variance are the ones that blow up) — the same
  mechanism that gives I-JEPA q8_0 a worst token of 0.43 in the image table above.

Practical consequence, in one line: **use f16 (or q8_0) for pooled features, retrieval and
classification — they are indistinguishable from f32 there; use f32 if you consume individual V-JEPA 2
ViT-L tokens** (dense/per-token work). V-JEPA 2.1 ViT-B is much more forgiving (worst token 0.97 at f16,
0.83 at q8_0).

## Thresholds (per `general.file_type`)

`test-parity` judges two classes of tensor separately, because the tail above only affects the token map:

| class | f32 | f16 | q8_0 / other |
|---|---|---|---|
| token maps (`last_hidden_state`) | mean & worst token ≥ 0.9999, `rel_max` ≤ 1e-3 | **median ≥ 0.999**, mean ≥ 0.99 | **median ≥ 0.99**, mean ≥ 0.95 |
| derived single-row tensors (`pooled_mean`, `pooled`, `cls`, `emb`, `logits`) | ≥ 0.9999 | ≥ 0.9995 | ≥ 0.995 |
| classifiers | top-1 identical to the reference, ≥ 4 of its top-5 | same | same |
| own-preprocessing pass | as above, with no bar stricter than 0.99 | | |

The lossy bars sit just under the worst fixture value (f16 derived: SSv2 pooler 0.999897; q8_0 derived:
SSv2 q8_0 pooler 0.996645 and logits 0.998501, with top-1/top-5 still exact), and the median is the gate
for the token map because it is insensitive to the f16/q8_0 tail while still collapsing for a real graph
bug: a wrong RoPE layout alone gives cosine ~0.63 (V-JEPA 2) / ~0.91 (2.1) on *every* token
(`docs/architecture.md`, `VJEPA_NOTES.md` §6). Every f32 file keeps the hard 0.9999-on-every-token bar.

Deviation from the protocol’s “f16: cos ≥ 0.9999”: that bar is unattainable for the worst single token
of an f16 file regardless of implementation — running the numpy executable spec
(`scripts/jepa_convert/selftest.py` math: f16 weights, float32 activations) on the stored I-JEPA inputs
already gives worst-token cos 0.99969 (coco_000000219578, token 220; 0.99998 on coco_000000000139), and
ggml’s f16 path (activations rounded to f16 inside f16 `mul_mat`, flash attention) lands between 0.991
and 0.9996 for that token depending on op order (flash+F16 K/V 0.9910, flash+F32 K/V 0.9986, naive
attention 0.9996) while the mean stays ≥ 0.99995. The V-JEPA 2 ViT-L video encoder pushes the same
effect much further (previous section), which is why the token map is read on the median while the
strict bars live on the pooled outputs.

`test-parity` also reports, per sample, the worst token's index and row norm and how many tokens fall
below 0.999 / 0.99 (`--json` keeps `cos_med`, `worst_row`, `n_rows_below_cos_0.999`, …), so a regression
that moves the whole distribution is visible even when the gate passes.

## Flash attention K/V dtype

`ggml_flash_attn_ext` with K/V cast to F16 costs ~3 digits of worst-token cosine on ViT-H/14 f32
(cos min 0.9910 vs 1.000000, rel_max 3.5e-2 vs 8.6e-5) because I-JEPA activations reach ~2e4.
`jepa_context_params.flash_kv` therefore defaults to **auto**: F32 K/V for f32 files, F16 K/V for
f16/quantized files (where weight rounding dominates anyway). Override with `JEPA_KV_F16` /
`JEPA_KV_F32` (`--kv-f16` / `--kv-f32` in the tools); `--no-flash` selects the naive
`mul_mat`+`soft_max_ext` path (~15–30 % slower on ViT-H, used for debugging).

The video encoders use full attention (no mask) and follow the same rule. The **attentive-pool
cross-attention** is the one place that always uses **F32 K/V**: it has a single query row, so ggml takes
the per-row kernel, which with F16 K/V would round q and the PV accumulator to F16
(`docs/ggml-notes.md` §1). It costs nothing at N_q = 1.

## Preprocessing parity

The uint8 antialiased resize (`jepa_resize_antialias_u8`) is a faithful port of the integer path that
`torchvision.transforms.v2.functional.resize(antialias=True)` runs on x86 CPUs (PyTorch
`upsample_avx_bilinear_bicubic_uint8`, a port of Pillow’s `ImagingResample`): separable, horizontal
then vertical pass with an intermediate uint8 image, double-precision filter weights (triangle /
Keys cubic a=−0.5, support scaled by the downscale factor) quantised to int16 with a per-pass
precision, int32 accumulation with a `1<<(prec−1)` rounding offset, `clamp(acc>>prec, 0, 255)`.
Shortest-edge sizes use `int(short*long/short_side)` (truncation — transformers’
`get_resize_output_image_size`), crops use `top=(H−c)/2`. Normalisation follows the HF torchvision
backend’s *fused* form `(px − 255·mean)/(255·std)` in float32 (`fused_norm`, default; the sequential
`(px/255 − mean)/std` form differs by 1 ulp and is used when the manifest has no HF processor).
Video is the same pipeline applied per frame, laid out as NCTHW (`jepa_preprocess_frames_rgb`).

Measured against the stored `input` tensors (worst sample per model):

| model | pipeline | from PIL pixels (`--rgb-dir`) / `frames_u8` | from the JPEG (stb_image decode) |
|---|---|---|---|
| ijepa-vith14-1k | squash 224, bilinear, mean/std 0.5 | **bit-exact** (max abs 0, 100 % equal) | max abs 1.6e-2 (= 2 u8 levels), 98.2 % equal |
| lejepa-vits16 | short 256 bicubic, crop 224, ImageNet | **bit-exact** | max abs 3.5e-2, 97.6 % equal |
| lewm-pusht | squash 224, bilinear, ImageNet | **bit-exact** | max abs 1.8e-2, 98.4 % equal |
| vjepa2-vitl-fpc64-256 | short 292, crop 256, ImageNet — 4 clips (16 f and 64 f) | **bit-exact** | – (no video decoder) |
| vjepa2-vitl-fpc16-256-ssv2 | same — 2 clips × 16 f | **bit-exact** | – |
| vjepa2_1-vitb-384 | short 438, crop 384, ImageNet — 2 clips × 16 f | **bit-exact** | – |
| vjepa2_1-vitb-384 | same — 2 COCO images (1-frame video) | — | max abs 3.5e-2, 98.4 % equal |

i.e. resize + crop + normalisation are bit-exact for every pipeline, images and video alike (video
samples are fed the reference's own sampled frames from `<sample>.frames_u8.npy`, so their
own-preprocessing pass reproduces the stored-input metrics to the digit); every residual difference
comes from JPEG decoding (stb_image vs PIL/libjpeg differ by ±1–2 levels on ~2 % of pixels — there is
no bit-exactness target across JPEG decoders). Effect on the encoder output: harmless for
LeJEPA/LeWM/V-JEPA 2.1 (own-pass worst-token cos ≥ 0.992), but I-JEPA amplifies it (worst token 0.79,
mean still ≥ 0.9984) — feed `frames_u8`/`--rgb-dir` style pixels when exact parity matters.

**Resolved metadata issue (converter):** `lewm-pusht-*.gguf` used to carry `jepa.pre.resize_mode =
shortest_edge` while the reference squashes non-square inputs to 224×224 (identical on LeWM’s native
square PushT renders, emb cosine ~0.93 on COCO). The converter now writes `squash` and the shipped
GGUFs are regenerated; `test-parity` still builds the pipeline from the reference manifest by default
(`--pre model` switches to the GGUF metadata and prints a NOTE if the two ever disagree).

## Tools (video)

```bash
# top-k labels of a clip (frames from the fixture dump, or -i frame.jpg ... in order)
build/jepa-classify -m models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf \
    --frames-npy tests/fixtures/ref/vjepa2-vitl-fpc16-256-ssv2/archery_f16.frames_u8.npy -k 5 -t 32 --time
#   1.  59.36%  [ 90] Pulling two ends of [something] but nothing happens      (reference: 60.23 %)
#   2.  14.87%  [162] Trying to bend [something unbendable] so nothing happens (reference: 14.71 %)
#   ... preprocess 26 ms | encoder 968 ms | head 96 ms | 2115 tokens/s

# clip / image features
build/jepa-embed -m models/gguf/vjepa2_1-vitb-384-f16.gguf --frames-npy <clip>.frames_u8.npy -t 32 --time
build/jepa-embed -m models/gguf/vjepa2-vitl-fpc64-256-f16.gguf --as-video -i f0.jpg -i f1.jpg -i f2.jpg
build/jepa-embed -m models/gguf/vjepa2_1-vitb-384-f16.gguf -i coco.jpg      # 2.1 native image path (576 tokens)
```

A single image given to a tubelet-2 model is repeated to fill the tubelet (what the HF processor does);
V-JEPA 2.1 instead takes its native 1-frame image path (`enc.patch_embed_img` + `img_mod_embed`), which
is what the `coco_*` reference samples use.

## Timing notes

* ms/item is the wall time of `ggml_backend_graph_compute` per image/clip (graph build + alloc excluded;
  they add < 1 ms even at 8192 tokens). PyTorch baseline = mean `timing_s.forward_s` from the manifest
  (32 threads), with the caveats ² and ³ above for the video models.
* I-JEPA ViT-H/14 (with `GGML_LLAMAFILE=ON`): f32 185 ms, f16 156 ms, q8_0 139 ms at 32 threads and
  f16 122 ms at 96 threads vs PyTorch 263 ms (32 t) — 1.7–2.2× faster than the PyTorch CPU baseline;
  f16 halves the weight memory (1.2 GiB peak vs 2.4 GiB), q8_0 uses 671 MiB.
* V-JEPA 2 ViT-L, 64-frame clip (8192 tokens): 6.4 s at 32 threads, 4.1 s at 96, where the raw
  flash-attention cost alone is 24 × 158 ms ≈ 3.8 s (`docs/ggml-notes.md` §3) — attention dominates the
  long clips, matmuls the short ones (16-frame clip: 0.82 s, 2.5 k tokens/s).
* q8_0 is *not* faster than f16 for the big clips (7.2 s vs 6.4 s for 64 frames): at 8192 tokens
  attention dominates and q8_0 pays the on-the-fly activation quantisation. It does cut the weight
  memory (872 MiB vs 1183 MiB peak RSS).
* `ctest` runs the two small f32 image parity checks, the V-JEPA 2.1 **image** parity check (576 tokens,
  0.9 s), the rope3d op test and the quick attention test in ~11 s total; the video clip samples
  (0.8–7 s each) are run by hand with the commands above. Parity tests register only when the GGUFs and
  reference dumps exist.
