# Parity — image ViT encoders (I-JEPA, LeJEPA/hfvit, LeWM) and video encoders (V-JEPA 2 / 2.1)

Numbers from `tests/test-parity` against the PyTorch golden dumps of `tests/fixtures/ref/<model>/`
(torch 2.13.0+cpu float32, transformers 5.16.1, 32 threads). Box: AMD Ryzen Threadripper PRO 7995WX
(96 cores / 192 threads, AVX-512), gcc 13.3.0, ggml @ 36da5713, `-O3 -march=native`,
**`GGML_LLAMAFILE=ON`** (see "Timing notes").

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

| model | ftype | samples | cos mean | cos min | rel_max | ms/item t=32 | PyTorch t=32 | peak RSS |
|---|---|---|---|---|---|---|---|---|
| lejepa-vits16 | f32 | 8 | 1.000000 | 1.000000 | 1.2e-05 | 13.9 | 22.6 | 99 MiB |
| lejepa-vits16 | f16 | 8 | 0.999999 | 0.999994 | 1.2e-03 | 13.4 | 22.6 | 59 MiB |
| lewm-pusht | f32 | 3 | 1.000000 | 1.000000 | 1.0e-06¹ | 10.4 | 18.1 | 88 MiB |
| lewm-pusht | f16 | 3 | 1.000000 | 1.000000 | 4.4e-04 | 10.7 | 18.1 | 53 MiB |
| ijepa-vith14-1k | f32 | 8 | 1.000000 | 1.000000 | 7.9e-05 | 184 | 263 | 2433 MiB |
| ijepa-vith14-1k | f16 | 8 | 0.999984 | 0.997805 | 2.9e-02 | 162 | 263 | 1233 MiB |

¹ for the LeWM `seq` sample the compared tensor is `emb_seq` (rel 1.0e-6 f32); per-image `last_hidden_state`
rel_max is 3.4e-5 (f32).

All 12 runs (6 files × {stored input, own preprocessing}) **PASS**. These numbers were re-measured with
`GGML_LLAMAFILE=ON`; that build is both faster (I-JEPA ViT-H f16 162 ms/image vs 343 ms before) and
slightly *more* accurate on f16 (ViT-H worst token 0.9978 vs 0.9910 with ggml's own f16 kernels), so the
earlier 96-thread column was dropped rather than left stale — the one 96-thread run of this round went to
the biggest video model (below).

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
PyTorch number, which includes the predictor — caveat ²). Peak RSS: 1183 MiB (f16 8192 tokens), 872 MiB (q8_0), 1572 MiB (ssv2 f32), 454 MiB
(2.1 f16), 336 MiB (2.1 q8_0).

### What the f32 rows prove, and why f16 tokens scatter

The f32 files (converted with `scripts/convert.py --family vjepa2{,_1} --ftype f32`) reproduce the
PyTorch reference **exactly**: cosine 1.000000 on every token of every clip, `rel_max` 7.5e-4 (ViT-L,
whose activations reach ±43) and 8.7e-5 (2.1 ViT-B), logits cosine 1.000000, top-1/top-5
identical. So the tubelet patchify, the tiled vs interleaved RoPE tables, `interpolate_rope`, the
modality vectors, the image tokenizer and the attentive-pool head are all bit-faithful.

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
  model" (the tokens with the smallest pre-final-LN variance are the ones that blow up).

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
| own-preprocessing pass | as above, but no bar stricter than 0.99 | | |

The median is the gate for lossy files because it is insensitive to the f16/q8_0 tail while still
collapsing for a real graph bug: a wrong RoPE layout alone gives cosine ~0.63 (V-JEPA 2) / ~0.91 (2.1)
on *every* token (`docs/architecture.md`, `VJEPA_NOTES.md` §6). Every f32 file keeps the hard
0.9999-on-every-token bar, and the strict per-model deviations of the earlier image-only protocol
(I-JEPA f16 worst token, `docs/architecture.md` step 3) are unchanged in spirit: read f16/quantized
per-token parity on the median or the mean, never on the minimum.

`test-parity` also reports, per sample, the worst token's index and row norm and how many tokens fall
below 0.999 / 0.99 (`--json` keeps `cos_med`, `worst_row`, `n_rows_below_cos_0.999`, …), so a regression
that moves the whole distribution is visible even when the gate passes.

## Flash attention K/V dtype

Unchanged for the encoders (full attention, no mask, `JEPA_KV_AUTO`: F32 K/V for f32 files, F16
otherwise; `--kv-f32` / `--kv-f16` / `--no-flash` override). The **attentive-pool cross-attention** is the
one place that always uses **F32 K/V**: it has a single query row, so ggml takes the per-row kernel,
which with F16 K/V would round q and the PV accumulator to F16 (`docs/ggml-notes.md` §1). It costs
nothing at N_q = 1.

## Preprocessing parity (video)

`jepa_preprocess_frames_rgb` runs the image pipeline per frame (shortest-edge resize 292 → centre crop
256 for V-JEPA 2, 438 → 384 for 2.1, `/255`, ImageNet mean/std, fused form) and lays the result out as
NCTHW. Fed the reference's own sampled frames (`<sample>.frames_u8.npy`, THWC uint8), it is
**bit-exact** against the stored `input` tensor for every video sample of all three models:

| model | samples | max abs diff | values exactly equal |
|---|---|---|---|
| vjepa2-vitl-fpc64-256 | 4 clips (16 f and 64 f) | **0** | **100.00 %** |
| vjepa2-vitl-fpc16-256-ssv2 | 2 clips × 16 f | **0** | **100.00 %** |
| vjepa2_1-vitb-384 | 2 clips × 16 f | **0** | **100.00 %** |
| vjepa2_1-vitb-384 | 2 COCO images (from the JPEG) | 3.5e-02 | 98.4 % |

i.e. the whole video path (per-frame antialiased resize, centre crop, normalisation, NCTHW layout) is
exact; the only residual difference is JPEG decoding (stb_image vs PIL) on the image samples, exactly as
documented for the image models above. Consequently the own-preprocessing pass of every video sample
reproduces the stored-input metrics to the digit.

## Tools

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
  they add < 1 ms even for 8192 tokens). PyTorch baseline = `timing_s.forward_s` from the manifest
  (32 threads), with the caveats ² and ³ above.
* **`GGML_LLAMAFILE=ON` is now set in the top-level CMakeLists** (`docs/ggml-notes.md` §5 measured
  1.6× F32 / 3.2× F16 / 2.2× Q8_0 on `mul_mat`; the video encoders are matmul- and attention-bound in
  roughly equal parts). Effect end to end: I-JEPA ViT-H f16 343 → 162 ms/image, and the V-JEPA 2 ViT-L
  64-frame clip lands at 6.4 s where the raw flash-attention cost alone is 24 × 158 ms ≈ 3.8 s
  (`docs/ggml-notes.md` §3) — i.e. attention dominates the long clips, matmuls the short ones.
* q8_0 is *not* faster than f16 for the big clips here (7.2 s vs 6.4 s for 64 frames): at 8192 tokens
  attention dominates and q8_0 pays the on-the-fly activation quantisation. It halves the weight memory
  (872 MiB vs 1183 MiB peak RSS).
* `ctest` runs the two small f32 image parity checks, the V-JEPA 2.1 **image** parity check (576 tokens,
  0.9 s), the rope3d op test and the quick attention test in ~11 s total; the clip samples (0.8–7 s each)
  are run by hand with the commands above. Parity tests register only when the GGUFs and reference dumps
  exist.
