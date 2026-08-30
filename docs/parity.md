# Parity — image ViT encoders (I-JEPA, LeJEPA/hfvit, LeWM)

Numbers from `tests/test-parity` against the PyTorch golden dumps of `tests/fixtures/ref/<model>/`
(torch 2.13.0+cpu float32, transformers 5.16.1, 32 threads). Box: AMD Ryzen Threadripper PRO 7995WX
(96 cores / 192 threads, AVX-512), gcc 13.3.0, ggml @ 36da5713, `-O3 -march=native`.

Reproduce (from a checkout with `models/gguf/` and `tests/fixtures/ref/` populated):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 32
.venv/bin/python tests/dump_rgb_u8.py tmp/rgb                    # PIL-decoded pixels (optional, see below)
build/test-parity models/gguf/<model>.gguf tests/fixtures/ref/<ref> --threads 32 [--rgb-dir tmp/rgb] [--json out.json]
ctest --test-dir build                                           # parity-lejepa-vits16, parity-lewm-pusht, ops (~1 s)
```

## Results (stored reference input → encoder, all fixture samples)

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

## Thresholds (per `general.file_type`)

* **f32**: worst-token cosine ≥ 0.9999 and `rel_max` ≤ 1e-3 (as in `docs/architecture.md` / `compare.py`).
* **f16**: mean cosine ≥ 0.9999 and worst-token cosine ≥ 0.99.
* other (q8_0 …): pooled/CLS/emb mean ≥ 0.999 & min ≥ 0.98; `last_hidden_state` mean ≥ 0.98 with no
  worst-token bound (low-variance tokens amplify weight quantisation ~arbitrarily — `docs/quantization.md`).
* own-preprocessing pass: mean cosine ≥ min(0.99, the stored-input bar) — it additionally carries
  JPEG-decoder variance (below).

Deviation from the protocol’s “f16: cos ≥ 0.9999”: that bar is kept for the **mean** cosine but is
unattainable for the worst single token of an f16 I-JEPA file regardless of implementation — running
the numpy executable spec (`scripts/jepa_convert/selftest.py` math: f16 weights, float32 activations)
on the stored inputs already gives worst-token cos 0.99969 (coco_000000219578, token 220; 0.99998 on
coco_000000000139). ggml’s f16 path (activations rounded to f16 inside f16 `mul_mat`, flash attention)
lands between 0.991 and 0.9996 for that token depending on op order (flash+F16 K/V 0.9910,
flash+F32 K/V 0.9986, naive attention 0.9996) while the mean stays ≥ 0.99995. A real graph bug sits
far below this (a wrong RoPE table layout alone gives cosine 0.63/0.91).

## Flash attention K/V dtype

`ggml_flash_attn_ext` with K/V cast to F16 costs ~3 digits of worst-token cosine on ViT-H/14 f32
(cos min 0.9910 vs 1.000000, rel_max 3.5e-2 vs 8.6e-5) because I-JEPA activations reach ~2e4.
`jepa_context_params.flash_kv` therefore defaults to **auto**: F32 K/V for f32 files, F16 K/V for
f16/quantized files (where weight rounding dominates anyway). Override with `JEPA_KV_F16` /
`JEPA_KV_F32` (`--kv-f16` / `--kv-f32` in the tools); `--no-flash` selects the naive
`mul_mat`+`soft_max_ext` path (~15–30 % slower on ViT-H, used for debugging).

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

Measured against the stored `input` tensors (worst sample per model):

| model | pipeline | from PIL pixels (`--rgb-dir`) | from the JPEG (stb_image decode) |
|---|---|---|---|
| ijepa-vith14-1k | squash 224, bilinear, mean/std 0.5 | **bit-exact** (max abs 0, 100 % equal) | max abs 1.6e-2 (= 2 u8 levels), 98.2 % equal |
| lejepa-vits16 | short 256 bicubic, crop 224, ImageNet | **bit-exact** | max abs 3.5e-2, 97.6 % equal |
| lewm-pusht | squash 224, bilinear, ImageNet | **bit-exact** | max abs 1.8e-2, 98.4 % equal |

i.e. resize + crop + normalisation are bit-exact for all three pipelines; every residual difference
comes from JPEG decoding (stb_image vs PIL/libjpeg differ by ±1–2 levels on ~2 % of pixels — there is
no bit-exactness target across JPEG decoders). Effect on the encoder output: harmless for
LeJEPA/LeWM (own-pass worst-token cos ≥ 0.997), but I-JEPA amplifies it (worst token 0.79, mean
still ≥ 0.9984) — feed `frames_u8`/`--rgb-dir` style pixels when exact parity matters.

**Resolved metadata issue (converter):** `lewm-pusht-*.gguf` used to carry `jepa.pre.resize_mode =
shortest_edge` while the reference squashes non-square inputs to 224×224 (identical on LeWM’s native
square PushT renders, emb cosine ~0.93 on COCO). The converter now writes `squash` and the shipped
GGUFs are regenerated; `test-parity` still builds the pipeline from the reference manifest by default
(`--pre model` switches to the GGUF metadata and prints a NOTE if the two ever disagree).

## Timing notes

* ms/item is the wall time of `ggml_backend_graph_compute` per image (graph build + alloc excluded;
  they add <1 ms). PyTorch baseline = mean `timing_s.forward_s` from the manifest (32 threads).
* I-JEPA ViT-H/14 (with `GGML_LLAMAFILE=ON`): f32 185 ms, f16 156 ms, q8_0 139 ms at 32 threads and
  f16 122 ms at 96 threads vs PyTorch 263 ms (32 t) — 1.7–2.2× faster than the PyTorch CPU baseline;
  f16 halves the weight memory (1.2 GiB peak vs 2.4 GiB), q8_0 uses 671 MiB.
* `ctest` runs the two small f32 parity checks + the rope3d op test in ~1.2 s total (8 threads each);
  the parity tests register only when the GGUFs and reference dumps exist.
