# Parity — image ViT encoders (I-JEPA, LeJEPA/hfvit, LeWM) and video encoders (V-JEPA 2 / 2.1, LeVJEPA)

*Raw measurement report — the curated view is [Benchmarks → Accuracy](accuracy.md).*

Numbers from `tests/test-parity` against the PyTorch golden dumps of `tests/fixtures/ref/<model>/`
(torch 2.13.0+cpu float32, transformers 5.16.1, 32 threads). Box: AMD Ryzen Threadripper PRO 7995WX
(96 cores / 192 threads, AVX-512), gcc 13.3.0, ggml @ 36da5713, `-O3 -march=native`, `GGML_LLAMAFILE=ON`.

Reproduce (from a checkout with `models/gguf/` and `tests/fixtures/ref/` populated):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 32
.venv/bin/python tests/dump_rgb_u8.py tmp/rgb                    # PIL-decoded pixels (optional, see below)
build/test-parity models/gguf/<model>.gguf tests/fixtures/ref/<ref> --threads 32 [--rgb-dir tmp/rgb] [--json out.json]
build/test-predictor --lewm  models/gguf/<lewm>.gguf   --ref tests/fixtures/ref/lewm-pusht --threads 32
build/test-predictor --vjepa2 models/gguf/<vjepa2>.gguf --ref tests/fixtures/ref/<ref> --samples archery_f16 --threads 32
ctest --test-dir build                                           # parity-lejepa-vits16, parity-lewm-pusht, (re-run cmake once GGUFs + refs exist — tests register at configure time)
                                                                 # parity-vjepa2_1-vitb-384-images, parity-levjepa-vitl16-clip,
                                                                 # predictor-lewm, predictor-vjepa2, batch, ops, attn, backend — 10 tests, ~32 s
                                                                 # (`backend` skips with exit 0 unless the build has a GPU; see "Parity on a GPU")
```

`test-parity` prints the threshold row it is judging with (backend × family class × file-type tier,
see "Thresholds"); the numbers below are all from the stored-input pass unless a column says
otherwise. Add `--gpu [N]` to run any of these on a CUDA device — see "Parity on a GPU" below.

## Results — image models (stored reference input → encoder, all fixture samples)

“cos mean” is the worst per-sample mean of the per-token cosine, “cos min” the worst single token of
any sample, `rel_max` = max|a−b| / max|b| over the `last_hidden_state` (for LeWM the `emb`/`emb_seq`
projector outputs are checked too — 1.000000 everywhere, including the 3-frame causal `seq` sample).

| model | ftype | samples | cos mean | cos min | rel_max | ms/item t=32 | ms/item t=96 | PyTorch t=32 | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| lejepa-vits16 | f32 | 8 | 1.000000 | 1.000000 | 1.2e-05 | 14.0 | —¹ | 15.4 | 99 MiB |
| lejepa-vits16 | f16 | 8 | 0.999999 | 0.999994 | 1.2e-03 | 13.4 | —¹ | 15.4 | 59 MiB |
| lejepa-vits16 | q8_0 | 8 | 0.999263 | 0.995193 | 3.7e-02 | 12.6 | —¹ | 15.4 | 40 MiB |
| lewm-pusht | f32 | 3 | 1.000000 | 1.000000 | 1.0e-06 | 10.3 | —¹ | 16.8 | 88 MiB |
| lewm-pusht | f16 | 3 | 1.000000 | 1.000000 | 5.8e-04 | 10.7 | —¹ | 16.8 | 53 MiB |
| lewm-pusht | q8_0 | 3 | 0.999913 | 0.999895 | 1.5e-02 | 9.7 | —¹ | 16.8 | 39 MiB |
| ijepa-vith14-1k | f32 | 8 | 1.000000 | 1.000000 | 7.9e-05 | 185 | —¹ | 249.8 | 2433 MiB |
| ijepa-vith14-1k | f16 | 8 | 0.999984 | 0.997583 | 2.9e-02 | 156 | 122 | 249.8 | 1233 MiB |
| ijepa-vith14-1k | q8_0 | 8 | 0.987843 | 0.432576 | 5.0e-01 | 138 | —¹ | 249.8 | 671 MiB |

¹ t=96 re-measured only for the mul_mat-bound ijepa f16 (122 ms vs 156 at t=32); the small models are
launch-/LN-bound and within noise of their t=32 numbers. All timings with `GGML_LLAMAFILE=ON`
(1.3–3.2× faster matmuls than stock ggml, see `docs/ggml-notes.md` §5). For the LeWM `seq` sample the
compared tensor is `emb_seq`; q8_0 `cos min` rows reflect the low-variance-token amplification
analysed in `docs/quantization.md` — pooled/CLS/emb stay ≥ 0.9997 for every q8_0 file.

**PyTorch t=32** is the manifest's `timing_s.forward_s` summarised exactly as `docs/benchmarks.md`
does it (`scripts/gen_benchmarks_md.py`): the mean over the samples with that frame count, except
where the manifest's first sample of the group is cold (≥ 1.2× the median of the rest), in which case
it is the **median of the samples after the first** — the steady-state figure. LeJEPA's first forward
is 72.2 ms against a steady 15.4 (median of the other seven; the all-sample mean would read 22.6) and
I-JEPA's is 331.8 against 249.8 (mean 263.0); LeWM keeps the mean of its two **1-frame** samples,
16.8, because its 3-frame `seq` sample times encode + projector + a predictor call and is not an
encoder forward at all. The video table below is unaffected: no video group has a cold first sample.

All 9 image files **PASS** on both passes (18 file×pass combinations) against the *image-family* thresholds below
— the strict ones: every token ≥ 0.9999 at f32, token-map mean ≥ 0.9999 with worst ≥ 0.99 at f16, and
mean ≥ 0.98 with pooled/CLS/emb ≥ 0.999 at q8_0. Raw per-sample JSON: `test-parity ... --json`.

## Results — video encoders (V-JEPA 2 / V-JEPA 2.1 / LeVJEPA)

Same protocol; the stored `input` is 5-D (`NTCHW` for the HF V-JEPA 2 dumps, `NCTHW` for V-JEPA 2.1 —
`test-parity` reads the layout from the manifest and transposes) and is fed as **one clip**: the whole
`T/2 × H/16 × W/16` token grid goes through a single graph with 3-D RoPE. `cos med` is the median
per-token cosine (the gate for f16/quantized files, see "Thresholds"), `cos min` the single worst token.
Worst sample per model; both clips (archery, bowling) and, for V-JEPA 2.1, both COCO images are included.

| model | ftype | sample set | tokens | cos mean | cos med | cos min | rel_max | pooled | logits | top-1/top-5 | ms/clip t=32 | tokens/s | PyTorch t=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vjepa2-vitl-fpc64-256 | **f32** | 2 clips × 16 f | 2048 | 1.000000 | 1.000000 | 0.999999 | **7.5e-04** | 1.000000 | – | – | 956 / 970⁴ | 2112 | 1293² |
| vjepa2-vitl-fpc64-256 | **f32** | 2 clips × 64 f | 8192 | 1.000000 | 1.000000 | 0.999990 | **1.2e-03** | 1.000000 | – | – | 7393 / 6845⁴ | 1197 | 10114² |
| vjepa2-vitl-fpc64-256 | f16 | 2 clips × 16 f | 2048 | 0.997144 | 0.999897 | 0.5088 | 5.1e-01 | 0.999991 | – | – | 823 / 827 | 2477 | 1293² |
| vjepa2-vitl-fpc64-256 | f16 | 2 clips × 64 f | 8192 | 0.998322 | 0.999934 | 0.5971 | 3.2e-01 | 0.999997 | – | – | 7032 / 6386 | 1283 | 10114² |
| vjepa2-vitl-fpc64-256 | q8_0 | 2 clips × 16 f | 2048 | 0.966128 | 0.996770 | 0.2305 | 6.0e-01 | 0.999876 | – | – | 829 / 852 | 2405 | 1293² |
| vjepa2-vitl-fpc64-256 | q8_0 | 2 clips × 64 f | 8192 | 0.973126 | 0.996880 | 0.2188 | 5.4e-01 | 0.999925 | – | – | 7220 / 7203 | 1137 | 10114² |
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
numbers are given per row: the two samples in run order, and the second is the steady-state figure and
the one `tokens/s` uses — the *first* clip of a process is 8–25 % slower because the weights are paged
in on first touch. That penalty lands on whichever shape a process reaches first: in the fpc64 run the
sample order is archery 64 f, archery 16 f, bowling 64 f, bowling 16 f, so on an idle box only the
64-frame row still shows it (7393 → 6845, 8 %) while the 16-frame row's two samples are within 1.5 %
of each other. The 15 rows come from 9 `test-parity` runs (a run covers all samples of one file);
all of them **PASS** the video-family thresholds, on the stored input and on our own preprocessing
alike.

² the fpc64 manifest's `timing_s.forward_s` is one `VJEPA2Model` forward, which always runs the
**predictor** as well (its `predictor_last_hidden_state` comes from the same call), so it is not a
like-for-like encoder number — treat it as an upper bound.
³ the SSv2 reference forward is encoder + attentive pooler + classifier with the predictor skipped, i.e.
directly comparable to our encoder (814 ms) + head (96 ms, measured with `jepa-classify --time`) = 910 ms.
⁴ the two f32 rows were **re-measured on an idle box** (two identical `test-parity` runs, the lower
per sample; the two runs agree to within 2.1 %). Their earlier values — 1240 / 1293 ms and 9288 /
9004 ms — came from a sweep sharing the box with a second agent and carried ~20 % of contention; with
that gone the f32/f16 ratio on the 16-frame clip is 970 / 827 = 1.17×, which is what
`docs/benchmarks.md` measures on synthetic input of the same shape (1.15×). The f16 and q8_0 rows of
this table are still from that earlier session and are therefore the pessimistic ones now. Cosines
and `rel_max` are load-independent and unchanged to the last digit.

The 64-frame ViT-L clip is the one case that was also timed at **96 threads** (one run, as budgeted):

| model | ftype | sample set | tokens | ms/clip t=96 | tokens/s |
|---|---|---|---|---|---|
| vjepa2-vitl-fpc64-256 | f16 | 2 clips × 64 f | 8192 | 4125 / 4076 | 2010 |

i.e. 1.57× the 32-thread throughput (2.5× the manifest's PyTorch number, which includes the
predictor — caveat ²). Peak RSS: 1183 MiB (f16, 8192 tokens), 872 MiB (q8_0), 1572 MiB (ssv2 f32),
454 MiB (2.1 f16), 336 MiB (2.1 q8_0).

### What the f32 rows prove, and why f16 tokens scatter

The f32 files (converted with `scripts/convert.py --family vjepa2{,_1} --ftype f32`) reproduce the
PyTorch reference **exactly**: mean cosine 1.000000 and every token ≥ 0.99999 on every clip, `rel_max` 7.5e-4 (ViT-L,
whose activations reach ±43) and 8.7e-5 (2.1 ViT-B), logits cosine 1.000000, top-1/top-5 identical. So
the tubelet patchify, the tiled vs interleaved RoPE tables, `interpolate_rope`, the modality vectors,
the image tokenizer and the attentive-pool head are all bit-faithful. That now includes the
encoder-only ViT-L file at both clip lengths (`vjepa2-vitl-fpc64-256-f32`, rows 1–2 above): at 8192
tokens the worst token is 0.999990 and `rel_max` is 1.22e-3 — larger than 1e-3 not because anything is
wrong but because max|a−b| grows with the length of the reductions in the graph, which is why the f32
`rel_max` bar is length-aware (`REL(N)` under "Thresholds"; 2e-3 at 8192 tokens, 1e-3 at ≤ 2048).

**Quantisation below q8_0 is not a parity configuration for the video encoders.**
`vjepa2_1-vitb-384-q4_k` on the 576-token COCO image measures token-map mean 0.9896 / median 0.9903 /
worst 0.9496 and pooled 0.9979 — usable for retrieval, far outside the f16/q8_0 envelope for per-token
work — and `vjepa2-vitl-fpc16-256-ssv2-q4_k` misses even the advisory bar (token map mean 0.883–0.928,
attentive-pooler output 0.977–0.981, logits 0.979–0.986 with top-1 still exact and 4 of 5 top-5), so it
is reported as a **FAIL**. `test-parity` puts every file below 8 bits per weight in the advisory tier
(results printed, only the derived tensors ≥ 0.99 and top-1 gated) and prints a note saying so.
Per-model quantisation numbers and the recommendation: `docs/quantization.md`.

At f16 the *mean* stays at 0.9971–0.999999 and every pooled/logit output stays ≥ 0.9998, but individual
tokens drop far below that (worst 0.51 on V-JEPA 2 ViT-L, 0.97 on 2.1 ViT-B). That is **not** graph
noise:

* ggml converts the activations of an f16 `mul_mat` to F16 as well (the vec-dot type of an F16 weight
  matrix; llamafile's AVX-512 kernel only has an F16×F16 path — `sgemm.cpp` `case GGML_TYPE_F16` requires
  `Btype == GGML_TYPE_F16`), so every one of the 24 ViT-L layers rounds its activations to ~3 decimal
  digits.
* Re-running the executable numpy spec with that same rounding reproduces the C++ numbers to four digits
  and collapses the *same degenerate low-norm token cluster* — not always the same single index: the
  worst token of the C++ f16 run is 357, of the numpy F16-activation run and of both C++ variants
  below it is 373. Both are low-norm rows of the same cluster (‖ref row‖ 91.1 and 88.3 against a
  sample mean of 95.5) and the whole low tail matches in size:

  | ssv2 bowling_f16, f16 weights | mean cos | worst token | tokens < 0.999 |
  |---|---|---|---|
  | numpy spec, f32 activations (`vjepa2_numpy_ref.py`) | 0.9999968 | 0.999036 @1163 | 0 / 2048 |
  | numpy spec, F16 activations | 0.9971802 | 0.557581 @373 | 420 / 2048 |
  | **jepa.cpp f16** | 0.997144 | 0.508778 **@357** | 414 / 2048 |
  | jepa.cpp f16, `--kv-f32` | 0.997139 | 0.558080 @373 | 421 / 2048 |
  | jepa.cpp f16, `--no-flash` | 0.997158 | 0.549800 @373 | 420 / 2048 |

  (indices and row norms straight from `test-parity`'s per-sample note, which prints the worst
  token, its reference row norm and the size of the low tail for exactly this comparison)
* Flash attention is not involved (F32 K/V and the naive path give the same spread), and
  `docs/quantization.md` independently finds V-JEPA 2 ViT-L to be "by far the most token-sensitive
  model" (the tokens with the smallest pre-final-LN variance are the ones that blow up) — the same
  mechanism that gives I-JEPA q8_0 a worst token of 0.43 in the image table above.

Practical consequence, in one line: **use f16 (or q8_0) for pooled features, retrieval and
classification — they are indistinguishable from f32 there; use f32 if you consume individual V-JEPA 2
ViT-L tokens** (dense/per-token work). V-JEPA 2.1 ViT-B is much more forgiving (worst token 0.97 at f16,
0.83 at q8_0).

### LeVJEPA ViT-L/16 — CLS, block-causal attention, tubelet 1

`levjepa-vitl16` is the fourth video encoder and the only one with a CLS token. A 16-frame 224²
clip is `1 + 16·14·14 = 3137` rows: the CLS first, then the patch tokens T-major. The CLS carries no
position at all (no table row, and the RoPE cos/sin tables get an identity row for it), and attention
runs under the block-causal mask — bidirectional inside a temporal slot, causal across slots, CLS row
open and CLS column closed — which is 53.1 % of the `3137 × 3137` grid. The feature this model is used
through is `cls` = `pooler_output`; `pooled` (the mean over the 3136 patch tokens) is reported as well.

Both fixture clips (archery, bowling) and both still images run in one `test-parity` process per file;
the images take the model card's path of repeating the frame 16 times, so they exercise the same graph
at the same 3137 rows.

| model | ftype | sample set | tokens | cos mean | cos med | cos min | rel_max | cls | pooled | ms/clip t=32 | tokens/s | PyTorch t=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| levjepa-vitl16 | **f32** | 2 clips × 16 f | 3137 | 1.000000 | 1.000000 | 1.000000 | **8.3e-06** | 1.000000 | 1.000000 | 1907 / 1661 | 1889 | 1774 |
| levjepa-vitl16 | **f32** | 2 still images | 3137 | 1.000000 | 1.000000 | 1.000000 | **7.7e-06** | 1.000000 | 1.000000 | 1685 / 1684 | 1862 | 1730 |
| levjepa-vitl16 | f16 | 2 clips × 16 f | 3137 | 0.999998 | 0.999999 | 0.999820 | 3.5e-03 | 1.000000 | 1.000000 | 1824 / 1542 | 2034 | 1774 |
| levjepa-vitl16 | f16 | 2 still images | 3137 | 1.000000 | 1.000000 | 0.999976 | 1.1e-03 | 1.000000 | 1.000000 | 1520 / 1540 | 2037 | 1730 |
| levjepa-vitl16 | q8_0 | 2 clips × 16 f | 3137 | 0.999789 | 0.999937 | 0.991409 | 2.5e-02 | 0.999996 | 0.999997 | 1733 / 1609 | 1950 | 1774 |
| levjepa-vitl16 | q8_0 | 2 still images | 3137 | 0.999809 | 0.999868 | 0.996157 | 1.9e-02 | 0.999989 | 0.999982 | 1675 / 1627 | 1928 | 1730 |
| levjepa-vitl16 | q4_0 | 2 clips × 16 f | 3137 | 0.997971 | 0.998157 | 0.983584 | 3.7e-02 | 0.999612 | 0.999716 | 1775 / 1610 | 1949 | 1774 |
| levjepa-vitl16 | q4_0 | 2 still images | 3137 | 0.998385 | 0.998464 | 0.993149 | 3.3e-02 | 0.999601 | 0.999548 | 1646 / 1633 | 1922 | 1730 |
| levjepa-vitl16 | q4_k | 2 clips × 16 f | 3137 | 0.997277 | 0.998313 | 0.957464 | 5.9e-02 | 0.999622 | 0.999749 | 2145 / 1935 | 1621 | 1774 |
| levjepa-vitl16 | q4_k | 2 still images | 3137 | 0.997031 | 0.997502 | 0.962741 | 4.7e-02 | 0.999457 | 0.999219 | 1928 / 1918 | 1635 | 1730 |

Worst sample of the group per row; `ms/clip` is the two samples of the group in run order and
`tokens/s` uses the second (the first clip of a process pays for paging the weights in — 1907 → 1661 ms
at f32). All five files **PASS**, on the stored input and on our own preprocessing alike, and the two
passes agree to the last digit reported: the fixtures store the sampled frames, our preprocessor
reproduces the reference pixels bit for bit (input max abs 0.0, 100 % of values equal), so this family
has no decoder floor at all. `PyTorch t=32` is the mean `timing_s.forward_s` of the group in the
manifest; re-timed as a loop (1 warmup + 5 forwards, same box, same clip) the reference is 1904 ms
median / 1883 ms minimum, so f16 at 32 threads is 1.2× PyTorch. The f16 file was also timed at **96
threads**, one run:

| model | ftype | sample set | tokens | ms/clip t=96 | tokens/s |
|---|---|---|---|---|---|
| levjepa-vitl16 | f16 | 2 clips × 16 f | 3137 | 1478 / 881 | 3559 |

i.e. 1.75× the 32-thread throughput and 2.1× PyTorch on the same box. Peak RSS: 1371 MiB (f32),
816 MiB (f16), 527 MiB (q8_0), 390 MiB (q4_0), 395 MiB (q4_k).

**This family has no low-cosine token tail.** Where V-JEPA 2 ViT-L f16 drops individual tokens to 0.51,
LeVJEPA f16 keeps *every one* of its 3137 tokens above 0.9998 on all four samples (`tokens < 0.999`:
0 of 3137), and q8_0 puts at most 100 tokens below 0.999 with none below 0.99. The mechanism from the
V-JEPA 2 discussion above — F16 activations amplified by low-variance tokens — is present but has
nothing to bite on here: the reference row norms of this encoder sit in a narrow band (worst-token
norm 29.9 against a sample mean of 29.5 on bowling q8_0), i.e. there is no degenerate low-norm cluster.
The practical reading is the opposite of V-JEPA 2 ViT-L's: f16 is a per-token-grade configuration for
this model, and q8_0 is a pooled-feature-grade one.

Below 8 bits the token map does open up — q4_0 puts ~2900 of 3137 tokens under 0.999 and q4_k reaches a
worst token of 0.957 — while the CLS feature holds at 0.9995 or better. Both stay in the advisory tier
(`test-parity` prints the note), and the recommendation is the project's usual one: q8_0 is the lowest
parity-grade quantisation.

Isolating the weights from the graph, `scripts/gguf_dequant_selftest.py` runs the same numpy forward on
the dequantized GGUF at f32 activations (archery_f16, worst token / CLS): q8_0 0.999961 / 0.999999,
q4_0 0.991279 / 0.999795, q4_k 0.994180 / 0.999862. The gap to the C++ q8_0 row (0.999428) is the F16
activation rounding inside ggml's quantized `mul_mat`, exactly as for the other video families.

## Results — predictors (masked V-JEPA 2 / 2.1 predictor, LeWM world model)

`tests/test-predictor` feeds the predictors the **reference encoder tokens** (never our own encoder
output, so an encoder difference can neither mask nor cause a predictor one) and compares against the
PyTorch dump `<sample>.predictor_last_hidden_state.npy` — the HF default pass, context = target = every
token, `mask_index` 1 — or, where PyTorch dumped no predictor output, against the executable numpy
spec `scripts/jepa_convert/vjepa2_numpy_ref.py::predictor_forward`. Same thresholds as the image
families in "Thresholds" below (f32 mean & worst ≥ 0.9999 and `rel_max` ≤ REL(rows); f16 mean ≥ 0.9999,
worst ≥ 0.99; q8_0 mean ≥ 0.999, worst ≥ 0.98).

```bash
build/test-predictor --vjepa2 models/gguf/vjepa2-vitl-fpc64-256-{f32,f16,q8_0}.gguf \
    --ref tests/fixtures/ref/vjepa2-vitl-fpc64-256 --samples archery_f16,bowling_f16 --threads 32
build/test-predictor --lewm models/gguf/lewm-pusht-{f32,f16,q8_0}.gguf \
    --ref tests/fixtures/ref/lewm-pusht --threads 32
```

### V-JEPA 2 ViT-L masked predictor (ctx = tgt = all 2048 tokens of a 16-frame clip)

| ftype | sample | rows | cos mean | cos min | rel_max | ms t=32 | PyTorch t=32 |
|---|---|---|---|---|---|---|---|
| f32 | archery | 2048 | 1.0000000 | 1.0000000 | 3.4e-06 | 409 | 2524⁵ |
| f32 | bowling | 2048 | 1.0000000 | 1.0000000 | 3.9e-06 | 329 | 2524⁵ |
| f16 | archery | 2048 | 0.9999997 | 0.9999968 | 1.3e-03 | 452 | 2524⁵ |
| f16 | bowling | 2048 | 0.9999996 | 0.9999961 | 1.7e-03 | 326 | 2524⁵ |
| q8_0 | archery | 2048 | 0.9998458 | 0.9961310 | 4.2e-02 | 388 | 2524⁵ |
| q8_0 | bowling | 2048 | 0.9998402 | 0.9990631 | 2.4e-02 | 290 | 2524⁵ |

The predictor is 12 layers of 384 dims over 4096 rows (2048 context + 2048 mask tokens) — ~5.6×
faster than the PyTorch predictor at f16, and 40–55 % of the cost of the ViT-L encoder pass on the
same clip (24 layers of 1024 dims over 2048 rows: 827 ms at f16). The cosines are deterministic; the
ms are the lower of two identical runs on an **idle** box (the two runs agree to within 1.3 %,
against the 15–40 % spread the earlier shared-box measurement saw), and the *archery* rows carry the
first-touch page-in of the run, which is why *bowling* is consistently faster at the same shape.
`rel_max` at f16 (1.3–1.7e-3 on values reaching ±12) is weight rounding: the same run at f32 is
exact to 4e-6. **Correction to an earlier draft of this table:** the bowling f16 row is the plain-f16
measurement above; the 0.9999999 / 6.2e-04 that circulated for it was a `--kv-f32` run, not the default
K/V policy.

### V-JEPA 2-AC (`jepa.pred.kind = "ac"`) — the action-conditioned world model

The fixture is Meta's own: `tests/fixtures/ref/vjepa2-ac-vitg` is produced by running the
`vjepa2_ac_vit_giant` hub entry (encoder + `VisionTransformerPredictorAC`), its
`notebooks/utils/{mpc_utils,world_model_wrapper}.py` world-model loop and its
`notebooks/franka_example_traj.npz` demo trajectory, all in float32 on the CPU
(`scripts/dump_reference.py --model vjepa2-ac-vitg`). Five samples: the encoder latents of the first
and last trajectory frames, a one-step prediction, a two-frame prediction, and a 4-candidate × 2-step
rollout with its L1 planning energies.

```bash
build/test-predictor --ac models/gguf/vjepa2-ac-vitg-{f32,f16,q8_0,q4_0,q4_k}.gguf \
    --ref tests/fixtures/ref/vjepa2-ac-vitg --threads 16
build-cuda/test-predictor --ac models/gguf/vjepa2-ac-vitg-f16.gguf \
    --ref tests/fixtures/ref/vjepa2-ac-vitg --gpu 1
```

`step1` is one predictor call over a single context frame (256 rows out); `step2` is two frames
(512 rows, every row block); the rollout rows are 4 candidates × 256 rows at that step.

| ftype | backend | step1 cos mean / min | step2 cos mean / min | rollout step 0 mean / min | rollout step 1 mean / min | energy rel | argmin |
|---|---|---|---|---|---|---|---|
| **f32** | CPU | **1.0000000** / **1.0000000** | **1.0000000** / **1.0000000** | **1.0000000** / **1.0000000** | **1.0000000** / **1.0000000** | **7.5e-07** | same |
| f16 | CPU | 0.9999997 / 0.9999879 | 0.9999998 / 0.9999879 | 0.9999997 / 0.9999792 | 0.9999852 / 0.9925298 | 4.3e-04 | same |
| q8_0 | CPU | 0.9997186 / 0.9941231 | 0.9996091 / 0.9941229 | 0.9997422 / 0.9847434 | 0.9989531 / 0.9368142 | 1.9e-03 | same |
| q4_0 | CPU | 0.9887799 / 0.8211261 | 0.9872608 / 0.8211251 | 0.9894599 / 0.8120557 | 0.9667215 / 0.5763850 | 6.0e-02 | same |
| q4_k | CPU | 0.9919827 / 0.8529575 | 0.9890511 / 0.8529583 | 0.9927472 / 0.8529011 | 0.9735382 / 0.5428584 | 5.6e-02 | same |
| f32 | CUDA | 0.9999997 / 0.9999956 | 0.9999998 / 0.9999892 | 0.9999996 / 0.9999819 | 0.9999938 / 0.9972536 | 3.5e-04 | same |
| f16 | CUDA | 0.9999997 / 0.9999937 | 0.9999997 / 0.9999646 | 0.9999996 / 0.9999761 | 0.9999834 / 0.9973585 | 7.0e-04 | same |
| q8_0 | CUDA | 0.9997900 / 0.9958302 | 0.9996120 / 0.9920831 | 0.9997561 / 0.9828506 | 0.9986458 / 0.9586072 | 6.4e-03 | same |
| q4_0 | CUDA | 0.9888310 / 0.8204811 | 0.9872460 / 0.8203875 | 0.9895106 / 0.8099474 | 0.9667593 / 0.5926768 | 6.1e-02 | same |
| q4_k | CUDA | 0.9914772 / 0.8281933 | 0.9890609 / 0.8299207 | 0.9925835 / 0.8192582 | 0.9738240 / 0.6924390 | 5.3e-02 | **FLIPS** |

`energy rel` is the relative error of `jepa_ac_energy` — Meta's `l1(pred, goal)` — against the
reference's own energies, and `argmin` is whether the planner would pick the same candidate. **The
last row is the finding that matters for planning: at q4_k on CUDA the ranking flips** (our argmin is
candidate 1, the reference's is 2), i.e. the model chooses a different action. q4_0/q4_k keep the
ranking on the CPU, but their worst rollout row is already at cosine 0.54–0.58. **f16 is the dtype to
plan with**; q8_0 is usable for a single step and marginal over a horizon; q4_* is encoder-grade only.

Four invariants beyond the reference comparison, all gated by `test-predictor --ac`:

| invariant | CPU (every dtype) | CUDA f32 | CUDA f16 |
|---|---|---|---|
| `jepa_ac_predict(T)` == the last row block of `jepa_ac_predict_all(T)` | max\|Δ\| **0** | max\|Δ\| **0** | max\|Δ\| **0** |
| block-causality: `predict(T=1)` == row block 0 of `predict(T=2)` | max\|Δ\| **0** | rel 3.020e-03 | rel 5.076e-03 |
| block-causality: perturbing frame 1's context | max\|Δ\| **0** on block 0, 3.56 on block 1 | same | same |
| **batched K == K sequential rollouts**, step 0 | max\|Δ\| **0** (bit-identical, f32 → q4_k) | rel 3.224e-03 | rel 5.140e-03 |
| **batched K == K sequential rollouts**, step 1 | max\|Δ\| **0** | rel 7.957e-02 | rel 6.092e-02 |
| `jepa_ac_next_state` vs scipy's `compute_new_pose` | max\|Δ\| **0** | max\|Δ\| **0** | max\|Δ\| **0** |
| n_seed = 2 rollout vs Meta's predictor (f32 CPU) | cos **1.0000000** / **1.0000000**, rel 4.5e-06 (step 0), 1.7e-05 (step 1) | — | — |
| cached context == explicit context | max\|Δ\| **0** | max\|Δ\| **0** | max\|Δ\| **0** |
| `jepa_ac_plan` vs Meta's `cem()` on its own draws (f32 CPU) | max\|Δ\| **2.98e-08** over the 2 × 7 plan | — | — |

Putting the K candidates of a planning step on the graph's batch axis therefore changes nothing on
the CPU: the items share a graph but never a reduction. On a GPU the GEMM tiling does depend on the
batch shape, and step 1 of a rollout inherits step 0's deviation, so that row is bounded by 4× the
batched-vs-reference deviation of the same step rather than by a fixed number (`test-predictor` also
checks the sequential run against the reference at the same tier, so a self-scaled bar cannot hide a
batching bug).

The numpy spec reproduces the same reference from the GGUF alone
(`selftest.py::ac_predictor_forward` on `vjepa2-ac-vitg-f32.gguf`): `pred_next` cos 1.0000000 /
1.0000000, rel 5.3e-06; `pred_seq` (T = 2) cos 1.0000000 / 1.0000000, rel 7.0e-06. That is what pins
the token layout, the RoPE id trick and the block-causal mask independently of the C++.

The AC bundle's **encoder** (its own frozen ViT-g, see docs/gguf-schema.md) is checked separately on
the two Franka frames, 256 tokens each: **PASS on every dtype and both backends**, f32 CPU cos
mean/median/worst all 1.000000 with rel 2.0e-04. Its own-preprocessing pass agrees with Meta's
transform to `max|Δ| = 2.38e-07` — one float32 ulp, the fused-versus-sequential normalisation
difference.

### V-JEPA 2 ViT-g/16 (the first 22-head, ffn-48/11 encoder here)

`facebook/vjepa2-vitg-fpc64-256`: 40 layers × 1408 dims, 22 heads (head_dim 64), ffn 6144 = 1408·48/11.
Nothing in the converter or the graph is model-specific — both are metadata-driven — but this is the
first file to exercise those numbers, so it gets its own fixtures and its own row here.

**The conversion is exact.** The numpy spec run on the GGUF's own weights against the HF model
(`vjepa2_numpy_ref.py --gguf models/gguf/vjepa2-vitg-fpc64-256-f32.gguf --hf models/facebook/vjepa2-vitg-fpc64-256
--frames 4 --size 256`) gives encoder `last_hidden_state` cos **1.0000000**, rel **2.893e-05**
(max |Δ| 1.543e-03), and the masked predictor cos **1.0000000**, rel **1.794e-06**. The input is
`torch.randn` under `torch.manual_seed(0)`, so the figure is deterministic and reproduces exactly.

| ftype | backend | sample set | tokens | cos mean | cos med | cos min | rel_max | pooled | verdict |
|---|---|---|---|---|---|---|---|---|---|
| f32 | CPU | archery × 16 f | 2048 | 1.000000 | 1.000000 | 1.000000 | 4.3e-04 | 1.000000 | pass |
| f32 | CPU | bowling × 16 f | 2048 | 1.000000 | 1.000000 | 0.999975 | **2.3e-03** | 1.000000 | over the bound |
| f32 | CPU | archery × 64 f | 8192 | 1.000000 | 1.000000 | 0.999999 | 3.0e-04 | 1.000000 | pass |
| f32 | CPU | bowling × 64 f | 8192 | 1.000000 | 1.000000 | 0.999976 | 1.7e-03 | 1.000000 | pass |
| f16 | CPU | bowling × 16 f | 2048 | 0.995983 | 0.999858 | 0.3005 | 5.6e-01 | 0.999993 | **PASS** |
| q8_0 | CPU | bowling × 16 f | 2048 | 0.961773 | 0.997127 | 0.1746 | 7.6e-01 | 0.999925 | **PASS** |
| q8_0 | CPU | bowling × 64 f | 8192 | 0.954860 | 0.995389 | 0.0870 | 6.9e-01 | 0.999956 | **PASS** |
| q4_k | CPU | bowling × 16 f | 2048 | 0.909267 | 0.976313 | 0.0752 | 9.5e-01 | 0.996920 | **PASS** |
| q4_k | CPU | bowling × 64 f | 8192 | 0.911208 | 0.972787 | 0.0181 | 7.8e-01 | 0.996801 | **PASS** |
| f32 | CUDA | bowling × 16 f | 2048 | 0.995088 | 0.999814 | 0.3698 | **5.0e-01** | 0.999991 | over the bound (0.5) |
| f16 | CUDA | bowling × 64 f | 8192 | 0.989901 | 0.999481 | 0.3291 | 7.3e-01 | 0.999991 | mean under 0.99 |
| q8_0 / q4_0 / q4_k | CUDA | 2 clips × 16/64 f | 2048/8192 | — | — | — | — | — | **PASS** |

Two honest non-passes, both of a bar fitted on ViT-L, neither a conversion or graph error:

1. **CPU f32, `rel_max` 2.30e-03 on one clip against a 1e-03 bound.** `REL(N) = 1e-3 · max(1, √(N/2048))`
   was calibrated on the 24-layer / 1024-dim ViT-L, where the same clip measures 7.51e-04. Six
   calibration clips at 2048 tokens on ViT-g measure 2.59e-04 (high_jump2), 3.82e-04 (high_jump),
   4.33e-04 (archery), 4.94e-04 (flying_kite), 5.49e-04 (marching) and **2.30e-03** (bowling) — five
   well inside the bound and one outlier, the same single ill-conditioned
   token that still reads cosine 0.999975 while the mean and median are 1.000000 and the pooled
   feature is exact. `--no-flash` gives 2.11e-03, so it is not the attention kernel; the numpy
   cross-check above puts the file itself at 2.893e-05. It is ggml's float32 accumulation over 40
   blocks of 1408 dims. A √-in-depth term predicts 1.5e-03 and a linear one 2.0e-03 — neither covers
   the outlier, so **the shared bound is left as it is** rather than widened to fit one measurement;
   this table is the record instead, and the registered ctest entry is the f16 one.
   Those six figures come from a calibration dump that is **not** part of the committed fixture set
   (the published `tests/fixtures/ref/vjepa2-vitg-fpc64-256` carries the two clips at 16 and 64
   frames, as every other video model does). Regenerate them with, from the repo root:

   ```bash
   # the six clips are the ones scripts/download_fixtures.sh puts in tests/fixtures/media
   .venv/bin/python scripts/dump_reference.py --model vjepa2-vitg-fpc64-256 \
       --n-clips 6 --frames 16 --out tmp/vitg-calib
   build/test-parity models/gguf/vjepa2-vitg-fpc64-256-f32.gguf tmp/vitg-calib/vjepa2-vitg-fpc64-256 \
       --threads 16 --samples archery_f16,bowling_f16,flying_kite_f16,high_jump_f16,high_jump2_f16,marching_f16
   ```

2. **CUDA f32/f16 sit on the GPU video tier's bars** (`rel ≤ 0.5·√(N/2048)`, mean ≥ 0.99): 5.03e-01
   against 5.0e-01 at 2048 tokens, and mean 0.989901 against 0.99 on the 64-frame bowling clip. The
   worst-token tail of a ViT-g at f16 is genuinely longer than a ViT-L's (0.33 against 0.51 at the
   same tier) — depth and width, not a defect. Every quantized tier passes on both backends, because
   their bars are the wider ones.

**Which of these ctest actually gates.** Only the **f16** ViT-g configuration is registered
(`parity-vjepa2-vitg-fpc64-256-clips`, the two 16-frame clips) — the f32 row above does not clear a
bound calibrated on ViT-L, and the 64-frame samples take ~18 s each at 8 threads. Note what the f16
video bar is: token-map **mean ≥ 0.99 and median ≥ 0.999**, with the worst token reported and not
gated. For a 1 B-parameter encoder that is a wide bar — this model's worst token is 0.30 and it
passes. The gate is a regression tripwire, not a statement that every token is faithful; the table
above is where the per-token truth is, and `pooled_mean ≥ 0.9999` is what holds across the board.

### V-JEPA 2.1 ViT-B predictor — image vs video modality (vs the numpy spec)

The 2.1 predictor adds a modality vector to *every* row, and it has two: `pred.mod_embed_video` and
`pred.mod_embed_img`. `jepa_predict()` / `jepa_predict_ex()` always use the video one (the HF/Meta
default); `jepa_predict_mod(..., JEPA_MODALITY_IMAGE)` selects the image one for the 1×16×16 image path
(`JEPA_MODALITY_AUTO` picks it when the token ids span a single temporal slice). Cross-check on the
576-token COCO image `coco_000000000139` (`predictor_forward(..., mode="image")`), ctx = tgt = 576:

| ftype | modality | rows | cos mean | cos min | rel_max | ms t=32 |
|---|---|---|---|---|---|---|
| f32 | **image** | 576 | 1.0000000 | 1.0000000 | 7.6e-05 | 84 |
| f16 | **image** | 576 | 0.9999955 | 0.9999434 | 7.7e-03 | 91 |
| q8_0 | **image** | 576 | 0.9996050 | 0.9907839 | 8.3e-02 | 77 |
| f32 | video (wrong vector) | 576 | 0.8624148 | 0.6550520 | 8.1e-01 | 82 |
| f16 | video (wrong vector) | 576 | 0.8624867 | 0.6554096 | 8.1e-01 | 89 |
| f16 | video, 4608-token clip archery (correct) | 4608 | 0.9999955 | 0.9999133 | 7.5e-03 | 1498 |
| f32 | video, 4608-token clip archery | 4608 | 1.0000000 | 0.9999958 | 8.3e-04 | 1409 |
| f32 | video, 4608-token clip bowling | 4608 | 1.0000000 | 0.9999944 | **1.07e-03** | 1442 |

i.e. the image path is exact at f32 once the right modality vector is used, and using the video vector
on an image costs two digits of *mean* cosine and a third of the worst row — a silent error the video
default would have shipped. The two f32 clip rows are the second case that needs the length-aware f32
`rel_max` bound: bowling sits at 1.07e-3 with cosine 1.0000000 on the mean and 0.9999944 on the worst
of 4608 rows, i.e. a false FAIL under a flat 1e-3 and a comfortable pass under REL(4608) = 1.5e-3.

Regenerating the case dirs (they live in git-ignored `tmp/`; the same snippet with `mode="video"` and a
clip sample writes the video case):

```bash
.venv/bin/python - <<'PY'
import os, sys, numpy as np
sys.path.insert(0, "scripts/jepa_convert")
import vjepa2_numpy_ref as ref
kv, W = ref.load_gguf("models/gguf/vjepa2_1-vitb-384-f16.gguf")          # same ftype as the file under test
enc = np.load("tests/fixtures/ref/vjepa2_1-vitb-384/coco_000000000139.last_hidden_state.npy")
ids = np.arange(enc.shape[0]); g = kv["jepa.pred.grid_size"]
pred, _ = ref.predictor_forward(kv, W, enc, ids, ids, (g, g), 1, "image")
os.makedirs("tmp/case-2_1-image", exist_ok=True)
for n, a in [("enc", enc), ("ctx_idx", ids.astype(np.int32)), ("tgt_idx", ids.astype(np.int32)),
             ("pred", np.asarray(pred, np.float32))]:
    np.save(f"tmp/case-2_1-image/{n}.npy", a)
PY
build/test-predictor --vjepa2 models/gguf/vjepa2_1-vitb-384-f16.gguf --case tmp/case-2_1-image \
    --modality image --threads 32
```

The case has to be generated from the **same GGUF** that is tested: the numpy spec runs the weights of
the file it is given, so an f16 case judged against the f32 file only measures the dtype gap
(rel 3.0e-3 — a false FAIL at the f32 bar).

### LeWM world model (D = 192, 3-frame causal window)

| ftype | check | rows | cos mean | cos min | rel_max | ms t=32 | PyTorch t=32 |
|---|---|---|---|---|---|---|---|
| f32 | `pred_next` (T = 1) | 1 | 1.0000000 | 1.0000000 | 3.5e-07 | 0.66 | 3.64⁵ |
| f32 | `pred_seq` (T = 3) | 3 | 1.0000000 | 1.0000000 | 3.5e-07 | 1.23 | 1.94⁵ |
| f16 | `pred_next` (T = 1) | 1 | 0.9999999 | 0.9999999 | 3.4e-04 | 0.65 | 3.64⁵ |
| f16 | `pred_seq` (T = 3) | 3 | 0.9999999 | 0.9999999 | 3.1e-04 | 1.22 | 1.94⁵ |
| q8_0 | `pred_next` (T = 1) | 1 | 0.9999397 | 0.9999397 | 1.4e-02 | 0.65 | 3.64⁵ |
| q8_0 | `pred_seq` (T = 3) | 3 | 0.9999573 | 0.9999397 | 9.6e-03 | 0.99 | 1.94⁵ |

(`ms` = the steady-state call — the second `pred_next` sample of the run, best of two runs; the
*first* predictor call of a process costs 2.3–10.0 ms because the weights are paged in and the graph
allocator sizes its buffer, and the T = 3 graph pays that once more because it is a different shape.
LeWM is a 192-dim, 6-layer predictor over ≤ 3 rows — it is launch-bound, so f16/q8_0 buy nothing over
f32 and PyTorch's 3.64 ms for T = 1 is mostly framework overhead as well.)

plus, at every dtype, the structural checks: row *t* of the T = 3 run is bit-identical to the T = *t*+1
prefix run (T ≥ 2 exactly; the T = 1 prefix is a different graph — one query row and no causal mask, so
at f16/q8_0 it matches to dtype round-off, max|d| 2.4e-4, see `tests/test-predictor.cpp`), perturbing
frame T−1 **non-uniformly** (`emb[T-1][0] += 5`, `emb[T-1][1] -= 3`; a uniform shift is absorbed by the
non-affine LayerNorm of the adaLN path, which would make the test vacuous) leaves rows 0..T−2
bit-identical while moving row T−1 by 0.99, and `jepa_lewm_rollout` step 0/1 equals
`jepa_lewm_predict(T = 1)` / the last row of `(T = 2)` exactly. `jepa-worldmodel --ref-check` is the
tool-level twin (cosine 1.0000000 on `emb`, `emb_seq`, `pred_next`, `pred_seq` at f32).

⁵ PyTorch predictor baselines (torch 2.13.0+cpu, 32 threads) measured in the predictor phase and not
re-measured here: 2524 ms for the 2048-token V-JEPA 2 predictor call (34810 ms for the 8192-token
64-frame one) and 3.64 / 1.94 ms for LeWM T = 1 / T = 3. Every jepa.cpp ms in this section is a single
`ggml_backend_graph_compute` call on the llamafile build at 32 threads, **re-measured on an idle box**
as the lower of two identical `test-predictor` runs; the earlier values (415/335, 457/332, 395/295 for
the ViT-L predictor; 87/90/79, 101/108, 1440/1408/1712 for 2.1; 0.62/1.28, 0.64/1.26, 0.63/0.95 for
LeWM) were taken with a second agent active. The cosines and `rel_max` reproduced to the last printed
digit in every one of the 20 rows, which is what one expects of load-independent numbers.

## Thresholds (per backend × model family × file-type tier)

`test-parity` judges two classes of tensor separately, and does it **per family**: the long low-cosine
tail described above is a property of the V-JEPA 2 *video* encoders at f16/q8_0 — the image ViTs
(I-JEPA, LeJEPA/hfvit, LeWM) reproduce the reference on every token at every dtype, so they keep the
hard bars. LeVJEPA is judged with the video rows and needs none of the slack they give: it clears them
by two to three digits in every tier (previous section). The table below is the CPU half of the `POLICY` table of `tests/test-parity.cpp`, which prints the
row it used in its header line; the GPU half is under "Parity on a GPU" below.

| family class | tier | token map (`last_hidden_state`) | derived (`pooled_mean`, `pooled`, `cls`, `emb`, `logits`) |
|---|---|---|---|
| image (`ijepa`, `hfvit`, `lewm`) | f32 | mean & worst token ≥ 0.9999, `rel_max` ≤ REL(N) | ≥ 0.9999, `rel_max` ≤ REL(N) |
| image | f16 | **mean ≥ 0.9999**, worst ≥ 0.99 | ≥ 0.9995 |
| image | q8 (≥ 8 bits/weight) | mean ≥ 0.98 | mean ≥ 0.999, worst row ≥ 0.98 |
| video (`vjepa2`, `vjepa2_1`, `levjepa`) | f32 | mean & median & worst ≥ 0.9999, `rel_max` ≤ REL(N) | ≥ 0.9999, `rel_max` ≤ REL(N) |
| video | f16 | **median ≥ 0.999**, mean ≥ 0.99 | ≥ 0.9995 |
| video | q8 | **median ≥ 0.99**, mean ≥ 0.95 | ≥ 0.995 |
| either | low-bit (< 8 bits/weight) | *reported, not gated* | ≥ 0.99 |

Plus, in every tier: a classifier has to reproduce the reference **top-1 exactly** and ≥ 4 of its top-5
(top-1 only in the low-bit tier), and the own-preprocessing pass uses the same rules with **no bar
stricter than 0.99** and no worst-token / `rel_max` bound (it carries JPEG-decoder noise on top unless
`--rgb-dir` supplies the reference pixels).

**REL(N) = max(1e-3, 1e-3·√(N/2048))** — the f32 `rel_max` bound, widened with the number of compared
rows. `rel_max` is a max-abs difference, and max|a−b| grows with the length of the reductions feeding
it (~√N for attention over N tokens), while the cosine does not. At the 2048-token reference point
(the 16-frame ViT-L clip, 7.5e-4) the bound is the historical 1e-3; the 8192-token clip measures
1.22e-3 with cosine 1.000000 on *every* token, and the 4608-row V-JEPA 2.1 predictor case 1.07e-3
(bowling; 8.3e-4 on archery) at cosine 1.0000000 / worst row 0.9999944 — both were false FAILs under a
flat 1e-3. Derived tensors have N = 1, so their bound stays exactly 1e-3.
The bar is still ~40× the observed f32 noise floor: perturbing one encoder weight tensor of
`vjepa2_1-vitb-384-f32` by +1 % pushes the 4608-token clip to `rel_max` 2.09e-3 → FAIL, and +0.1 % on
the final `enc.norm.weight` fails on the (unwidened) pooled bound, while a clean run sits at 4.0e-5.

**Tiers come from `general.file_type`**, which carries a `GGML_FTYPE_*` value: 0 → f32, 1/24 → f16 tier,
otherwise the bits per stored weight decide (`ggml_type_size / ggml_blck_size`): q8_0 is 8.5 bits →
q8 tier; q4_0/q4_1/q4_k/q5_*/q6_k/iq* are below 8 bits → **low-bit**, where `test-parity` prints
"below the recommended quantization for parity" and gates only the derived tensors (≥ 0.99) and top-1.
`docs/quantization.md` has the per-model numbers behind that recommendation.

The lossy bars sit just under the worst fixture value (video f16 derived: SSv2 pooler 0.999897; video
q8_0 derived: SSv2 q8_0 pooler 0.996645 and logits 0.998501, with top-1/top-5 still exact; image q8_0
derived: LeWM `emb_seq` 0.999895 and I-JEPA pooled 0.999748; image q8_0 token map: I-JEPA worst sample
mean 0.987843), and the median is the gate for the *video* token map because it is insensitive to the
f16/q8_0 tail while still collapsing for a real graph bug: a wrong RoPE layout alone gives cosine ~0.63
(V-JEPA 2) / ~0.91 (2.1) on *every* token (`docs/architecture.md`, `VJEPA_NOTES.md` §6). Every f32 file
keeps the hard 0.9999-on-every-token bar.

Sensitivity note: at f16/q8 the video token map has no rel_max and no worst-token gate, so its floor for a weight-level error is ~1–5 % (a +1 % scale on one `attn_out` matrix of the SSv2 f16 file still passes: mean 0.9995, median 0.99997, logits 0.99998); the f32 file is the sensitive configuration (REL(N) ≈ 37× its noise floor) and every model ships one, which is why every family is anchored at f32.

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

## Parity on a GPU (`--gpu`)

Everything above is the CPU backend. A CUDA build (`-DJEPA_CUDA=ON`, `docs/getting-started.md`) runs the
same graphs on a GPU, and `test-parity` / `test-predictor` take `--gpu [N]` to judge them there.
Box: one NVIDIA RTX 4500 Ada Generation (compute 8.9, 24 GB), CUDA 13.0.88, driver 580.173.02,
ggml @ 36da5713, `GGML_PREC_F32` on every `mul_mat` — which is what four of the six families
default to, and what `$JEPA_GPU_PREC=f32` pins for the other two.

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DJEPA_CUDA=ON && cmake --build build-cuda -j 16
build-cuda/test-parity models/gguf/<model>.gguf tests/fixtures/ref/<ref> --gpu 0 [--json out.json]
build-cuda/test-predictor --vjepa2 models/gguf/<model>.gguf --ref tests/fixtures/ref/<ref> --gpu 0
build-cuda/test-backend models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf   # ctest "backend"
```

### There is no f32 tier on a GPU

Three backend differences, none of which jepa.cpp can turn off:

| | CPU | GPU (ggml-CUDA) |
|---|---|---|
| `mul_mat`, f32 weights | strict FP32, 1.2e-07 | **TF32**, 3.2e-04 — ggml passes `CUBLAS_GEMM_DEFAULT_TENSOR_OP` to every `cublasGemmEx` and never calls `cublasSetMathMode`; `ggml_mul_mat_set_prec` only picks the *compute type*, so it cannot undo this |
| `mul_mat`, f16 weights | 3.0e-04 | 2.6e-05 **with** `GGML_PREC_F32`, 4.6e-03 without |
| `flash_attn_ext` | F32 K/V honoured, rel ≤ 6e-07 | K/V always converted to F16 and the PV accumulator is `half2`; rel 5.2e-03–1.5e-02, and `ggml_flash_attn_ext_set_prec` is a no-op there |
| `ggml_norm` | centred two-pass variance | one-pass `E[x²] − mean²` |

So **an f32 GGUF on a GPU is judged with its family's f16 bars**, and `test-parity` says so in its
header. f16 and quantized files keep their own bars: with `GGML_PREC_F32` the f16 matmul term is
*better* than the CPU's.

Every GPU cell additionally gates **`rel_max`**, which the CPU tiers only do at f32. That is the
one thing cosine cannot see: a wrong `ggml_norm` variance is a per-row *scale* error, and
CUDA has been measured at relative error 5.7 with per-row cosine still reading
1.0000000. The bars below are set just outside the worst measured fixture value, so they are loose
in the quantized tiers (where weight rounding dominates `rel_max` anyway) and tight on the derived
tensors — but any of them collapses a norm-scale failure.

| family class | tier | token map (`last_hidden_state`) | derived (`pooled_mean`, `pooled`, `cls`, `emb`, `logits`) |
|---|---|---|---|
| image (`ijepa`, `hfvit`, `lewm`) | f32 → judged as f16 | mean ≥ 0.999, median ≥ 0.9999, worst ≥ 0.90, `rel_max` ≤ 0.15·REL | ≥ 0.9995, `rel_max` ≤ 1e-2 |
| image | f16 | same as above | same as above |
| image | q8 | mean ≥ 0.98, median ≥ 0.99, `rel_max` ≤ 0.6·REL | mean ≥ 0.999, worst ≥ 0.98, `rel_max` ≤ 5e-2 |
| video (`vjepa2`, `vjepa2_1`) | f32 → judged as f16 | median ≥ 0.999, mean ≥ 0.99, `rel_max` ≤ 0.5·REL | ≥ 0.9995, `rel_max` ≤ 3e-2 |
| video | f16 | same as above | same as above |
| video | q8 | median ≥ 0.99, mean ≥ 0.95, `rel_max` ≤ 0.8·REL | ≥ 0.995, `rel_max` ≤ 0.3 |
| either | low-bit (< 8 bits/weight) | *reported, not gated* | ≥ 0.99 |

`REL` is the same length-aware widening as on the CPU, `max(1, √(N/2048))`.

Two bars had to be *loosened* rather than mapped across, both in the image families' token map:
the **worst-token** floor, 0.99 → 0.90, and the **mean**, 0.9999 → 0.999. Both are one model:
I-JEPA ViT-H/14's worst token goes from 0.9976 on the CPU at f16 to **0.9613** on a GPU (0.9723
with `--no-flash`, i.e. F32 attention) and its mean to 0.999788. That is the F16 PV accumulator of
every CUDA flash kernel plus TF32 landing on token 8 of `coco_000000219578` — a **high-norm outlier
row** (‖ref row‖ 39.23 against a 32.85 mean; this is a *different* token from the CPU-f16
low-norm cluster discussed under "Deviation from the protocol"). LeJEPA and LeWM stay at
0.9998 / 0.99999 on the same backend, and I-JEPA's *median* stays at 0.999996.

Sensitivity note for the GPU image tier: the loosening has a measured cost. A +0.5 % scale on one
`attn_out` weight matrix of the I-JEPA f16 file **passes** this tier (cos_min 0.9838, rel 0.075)
where the CPU tier fails it; at +1 % the tier fails — but through `rel_max` (0.1530 vs the 0.150
bar), not the cosine floors. `rel_max` is the load-bearing gate here, which is why every GPU tier
carries it; treat the GPU image tier as ~2× less sensitive to weight-level errors than the CPU one,
with the f32-on-CPU run remaining the sensitive configuration for any real investigation.

### Results — encoders on CUDA0

Worst sample per file, stored-input pass, all fixture samples. `derived` names the worst of
`pooled_mean` / `pooled` / `cls` / `emb` / `logits`; its `rel` column is the worst over all of them.

| model | ftype | samples | tokens | cos mean | cos med | cos min | rel_max | derived (worst) | derived rel |
|---|---|---:|---:|---|---|---|---|---|---|
| lejepa-vits16 | f32 | 8 | 197 | 0.999994 | 0.999996 | 0.9998 | 6.0e-03 | pooled_mean 1.000000 | 9.6e-04 |
| lejepa-vits16 | f16 | 8 | 197 | 0.999994 | 0.999996 | 0.9998 | 5.8e-03 | pooled_mean 1.000000 | 8.2e-04 |
| lejepa-vits16 | q8_0 | 8 | 197 | 0.999277 | 0.999426 | 0.9966 | 3.5e-02 | cls 0.999808 | 2.5e-02 |
| lejepa-vits16 | q4_k | 8 | 197 | 0.964043 | 0.965076 | 0.8547 | 1.7e-01 | cls 0.985080 | 1.7e-01 |
| lewm-pusht | f32 | 2 | 257 | 1.000000 | 1.000000 | 1.0000 | 5.9e-04 | emb 1.000000 | 6.2e-04 |
| lewm-pusht | f16 | 2 | 257 | 1.000000 | 1.000000 | 1.0000 | 7.1e-04 | emb 1.000000 | 7.3e-04 |
| lewm-pusht | q8_0 | 2 | 257 | 0.999978 | 0.999980 | 0.9999 | 8.9e-03 | emb 0.999923 | 2.0e-02 |
| lewm-pusht | q4_k | 2 | 257 | 0.998698 | 0.998798 | 0.9966 | 7.1e-02 | emb 0.991171 | 1.2e-01 |
| ijepa-vith14-1k | f32 | 8 | 256 | 0.999853 | 0.999997 | 0.9730 | 7.4e-02 | pooled_mean 0.999996 | 2.1e-03 |
| ijepa-vith14-1k | f16 | 8 | 256 | 0.999788 | 0.999996 | 0.9613 | 9.1e-02 | pooled_mean 0.999994 | 2.7e-03 |
| ijepa-vith14-1k | q8_0 | 8 | 256 | 0.990211 | 0.999443 | 0.6453 | 3.8e-01 | pooled_mean 0.999725 | 1.6e-02 |
| ijepa-vith14-1k | q4_k | 8 | 256 | 0.907524 | 0.977606 | 0.0352 | 1.0e+00 | pooled_mean 0.992122 | 1.1e-01 |
| vjepa2-vitl-fpc64-256 | f32 | 4 | 8192 | 0.994774 | 0.999698 | 0.3549 | 4.7e-01 | pooled_mean 0.999984 | 5.2e-03 |
| vjepa2-vitl-fpc64-256 | f16 | 4 | 8192 | 0.994793 | 0.999696 | 0.3557 | 4.7e-01 | pooled_mean 0.999984 | 5.0e-03 |
| vjepa2-vitl-fpc64-256 | q8_0 | 4 | 8192 | 0.967114 | 0.996680 | 0.1938 | 6.5e-01 | pooled_mean 0.999889 | 6.6e-03 |
| vjepa2-vitl-fpc64-256 | q4_k | 4 | 8192 | 0.895503 | 0.964412 | 0.1435 | 9.4e-01 | pooled_mean 0.995438 | 4.6e-02 |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 2 | 2048 | 0.997086 | 0.999871 | 0.4917 | 3.6e-01 | pooled 0.999896 | 1.5e-02 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 2 | 2048 | 0.997052 | 0.999870 | 0.4352 | 3.6e-01 | pooled 0.999899 | 1.4e-02 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 2 | 2048 | 0.967114 | 0.997084 | 0.1938 | 6.5e-01 | pooled 0.996336 | 1.6e-01 |
| vjepa2-vitl-fpc16-256-ssv2 | q4_k | 2 | 2048 | 0.895503 | 0.964412 | 0.1944 | 9.4e-01 | logits 0.978056 | 1.8e-01 |
| vjepa2_1-vitb-384 | f32 | 4 | 4608 | 0.999949 | 0.999985 | 0.9853 | 4.0e-02 | pooled_mean 0.999999 | 5.2e-04 |
| vjepa2_1-vitb-384 | f16 | 4 | 4608 | 0.999951 | 0.999985 | 0.9769 | 5.1e-02 | pooled_mean 0.999999 | 5.2e-04 |
| vjepa2_1-vitb-384 | q8_0 | 4 | 4608 | 0.999073 | 0.999568 | 0.8396 | 1.3e-01 | pooled_mean 0.999983 | 1.3e-03 |
| vjepa2_1-vitb-384 | q4_k | 4 | 4608 | 0.978335 | 0.984305 | 0.6029 | 2.7e-01 | pooled_mean 0.997691 | 1.1e-02 |
| levjepa-vitl16 | f32 | 4 | 3137 | 0.999996 | 0.999998 | 0.9999 | 4.1e-03 | cls 0.999999 | 6.4e-04 |
| levjepa-vitl16 | f16 | 4 | 3137 | 0.999996 | 0.999998 | 0.9999 | 4.1e-03 | cls 0.999999 | 6.6e-04 |
| levjepa-vitl16 | q8_0 | 4 | 3137 | 0.999785 | 0.999864 | 0.9914 | 2.3e-02 | pooled_mean 0.999982 | 2.3e-03 |
| levjepa-vitl16 | q4_0 | 4 | 3137 | 0.997961 | 0.998148 | 0.9836 | 3.8e-02 | pooled_mean 0.999547 | 8.6e-03 |
| levjepa-vitl16 | q4_k | 4 | 3137 | 0.994222 | 0.997310 | 0.8844 | 1.6e-01 | cls 0.999343 | 1.8e-02 |

**27 of the 29 files PASS.** The two that do not — `lejepa-vits16-q4_k` (derived `cls` 0.9851 against
the low-bit tier's 0.99) and `vjepa2-vitl-fpc16-256-ssv2-q4_k` (`logits` 0.9781, `pooled` 0.9807) —
**fail on the CPU too, identically**: q4_k below 8 bits per weight is not a parity configuration for
these two models and `docs/quantization.md` already says so. Every q4_k row is otherwise a *pass on
the GPU*, which matters because q4_k is the fastest GPU path (`docs/performance.md`).

The classifier rows reproduce the reference **top-1 and top-5 exactly** at f32/f16/q8_0. The
own-preprocessing pass passes everywhere the stored-input pass does.

Reading the table: the two encoder-only V-JEPA 2 ViT-L rows carry the model's own f16 low-cosine
tail (`cos min` 0.35), which is *not* a GPU effect — the CPU f16 rows in the video table above show
the same 0.51/0.60 and the f32 CPU rows do not, because the tail comes from rounding activations to
F16 inside an f16 `mul_mat`. On the GPU the f32 file behaves like the f16 one (0.3549 vs 0.3557),
which is the same statement as "there is no f32 tier here", measured end to end.

LeVJEPA is the family whose GPU rows come out *better* than its CPU ones at f32/f16 (worst token
0.9999 against the CPU f16's 0.99982), and its q4_k row is the one place where the backend matters:
0.8844 on the GPU against 0.9575 on the CPU, `rel_max` 1.6e-1 against 5.9e-2, with the CLS feature
still at 0.9993. That row also carries the practical answer to the one open question this family
raised — the block-causal mask had never been exercised on a CUDA *flash* kernel before (LeWM's
causal predictor takes the naive head-32 path). It needs no padding at this ggml commit: the MMA
kernel wraps mask rows with `fastmodulo(j0 + j, ne01)` and clamps the key axis of the final tile
against `k_VKQ_sup`, so an unpadded `[3137, 3137]` F16 buffer is read correctly. Graph validation —
mandatory on a GPU — accepts every node, and the measured agreement with PyTorch settles it: a
silently dropped mask would read 0.945 on the CLS, not 1.000000.

### Accumulation precision, per family

The table above is `GGML_PREC_F32` on every `mul_mat`. That is what four of the six families ask a
GPU context for; `hfvit` and `levjepa` ask for f16 accumulation instead, because they clear the same
tiers with it and are 1.11× and 1.06× faster
([performance.md](performance.md#accumulation-precision-on-a-gpu) has the milliseconds).
`scripts/gpu_prec_sweep.sh` runs every family at every dtype at **both** settings and writes
`tests/results/gpu-prec.json`; the bars are the ones above and are untouched.

Only f16 weights move. A quantized file takes `mmq` and never reaches cuBLAS, ggml's f32 path is
TF32 either way, and the flash-attention accumulator is `GGML_PREC_F32` in both — the small
movements in the quantized rows are the f16 and f32 tensors a quantized file still carries.

Where the two settings diverge, worst sample per file, stored-input pass:

| model | ftype | cos mean | cos med | cos min | rel_max | verdict at f16 accumulation |
|---|---|---|---|---|---|---|
| lejepa-vits16 | f16 | 0.999988 (0.999994) | 0.999991 (0.999996) | 0.9998 (0.9998) | 6.3e-03 (5.8e-03) | **passes** — the shipped default |
| levjepa-vitl16 | f16 | 0.999964 (0.999996) | 0.999990 (0.999998) | 0.9933 (0.9999) | 1.6e-02 (4.1e-03) | **passes** — the shipped default |
| ijepa-vith14-1k | f16 | 0.999240 (0.999788) | 0.999982 (0.999996) | **0.8938** (0.9613) | **2.0e-01** (9.1e-02) | **fails**: worst token under the 0.90 floor and `rel_max` over the 0.15 bar, both on `coco_000000039769` |
| vjepa2-vitl-fpc64-256 | f16 | **0.9887** (0.9948) | 0.999170 (0.999696) | 0.3026 (0.3557) | **7.0e-01** (4.7e-01) | **fails**: token-map mean under 0.99 and `rel_max` over 0.5·√(N/2048), on *bowling* |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 0.9887 | 0.999253 | 0.3383 | 7.0e-01 | **fails** the same two bars |
| vjepa2_1-vitb-384 | f16 | 0.999718 (0.999951) | 0.999919 (0.999985) | 0.9599 (0.9769) | 7.6e-02 (5.1e-02) | passes, but its clips are 2–4 % slower, so it keeps `GGML_PREC_F32` |
| lewm-pusht | f16 | 0.999999 (1.000000) | 0.999999 (1.000000) | 1.0000 (1.0000) | 1.9e-03 (7.1e-04) | passes, with no time to gain, so it keeps `GGML_PREC_F32` |

Values in brackets are the same file at `GGML_PREC_F32`, i.e. the row of the table above.
V-JEPA 2 ViT-g/16 fails its f16 tier at **both** settings and for the same reason it does in the
[ViT-g section](#v-jepa-2-vit-g16-the-first-22-head-ffn-4811-encoder-here) — a bound fitted on ViT-L —
so nothing about it turns on this choice; its quantized tiers pass at both.

```bash
scripts/gpu_prec_sweep.sh 1 --timing-repeat 4
JEPA_GPU_PREC=f16 build-cuda/test-parity models/gguf/ijepa_vith14_1k-f16.gguf \
    tests/fixtures/ref/ijepa-vith14-1k --gpu 1
```

### Results — predictors on CUDA0

`test-predictor --gpu 0`, against the same PyTorch dumps. The V-JEPA 2 predictor has head_dim 32,
for which **no CUDA flash-attention kernel exists**, so it runs the naive `mul_mat + soft_max_ext`
path — fully F32, one graph split, and a 0.75 GiB score matrix at these row counts
(`docs/architecture.md` "GPU backend"). `test-predictor` uses the GPU thresholds (f32 judged as f16, plus
`rel_max` ≤ 2e-2 at f32/f16 and ≤ 8e-2 at q8).

| model | ftype | check | rows | cos mean | cos min | rel_max | ms (GPU) | ms (CPU t=32) |
|---|---|---|---:|---|---|---|---:|---:|
| vjepa2-vitl-fpc64-256 | f32 | archery | 2048 | 0.9999996 | 0.9999886 | 2.8e-03 | 152 | 409 |
| vjepa2-vitl-fpc64-256 | f32 | bowling | 2048 | 0.9999995 | 0.9999955 | 1.7e-03 | 113 | 329 |
| vjepa2-vitl-fpc64-256 | f16 | archery | 2048 | 0.9999996 | 0.9999904 | 2.5e-03 | 154 | 452 |
| vjepa2-vitl-fpc64-256 | f16 | bowling | 2048 | 0.9999995 | 0.9999959 | 1.7e-03 | 113 | 326 |
| vjepa2-vitl-fpc64-256 | q8_0 | archery | 2048 | 0.9998456 | 0.9967527 | 5.1e-02 | 158 | 388 |
| vjepa2-vitl-fpc64-256 | q8_0 | bowling | 2048 | 0.9998407 | 0.9987917 | 2.6e-02 | 112 | 290 |
| lewm-pusht | f32 | `pred_next` / `pred_seq` | 1 / 3 | 1.0000000 | 1.0000000 | 3.1e-05 | 0.45 / 0.57 | 0.66 / 1.23 |
| lewm-pusht | f16 | `pred_next` / `pred_seq` | 1 / 3 | 0.9999999 | 0.9999999 | 4.6e-04 | 0.41 / 3.22 | 0.65 / 1.22 |
| lewm-pusht | q8_0 | `pred_next` / `pred_seq` | 1 / 3 | 0.9999340 | 0.9999340 | 1.3e-02 | 0.45 / 0.57 | 0.65 / 0.99 |

`ms (GPU)` is the median of three `test-predictor --gpu 0` launches per file, re-measured on
2026-09-01 with the sweep behind `tests/results/benchmarks-gpu.json` (raw log in
`tmp/bench-gpu/test-predictor-gpu.log`); it replaces an earlier session's single-launch column,
which read 8–30 % higher on the *archery* and LeWM rows and within half a per cent on *bowling*.
Those are the rows a single launch measures worst: *archery* is the first sample of its run and
carries the first-touch page-in, and the LeWM graphs are sub-millisecond and launch-bound.
Every cosine and `rel_max` above reproduced to the digit, so what moved is timing, not numerics.
`ms (CPU t=32)` is unchanged.

Every predictor row **PASSES**, and the naive path is *more* accurate than flash would be: it is
genuinely F32 end to end, so the f32 and f16 files land at `rel_max` 1.7e-03–2.8e-03 against the
PyTorch dump — the same order as the CPU's 1.3e-03–1.7e-03, not the 5e-03–1.5e-02 a CUDA flash
kernel costs. The 2.5–2.9× speed-up (rather than the encoders' 20×) is the price of that path
running at ~3 TFLOP/s instead of flash's 50–70.

LeWM's structural self-consistency checks (causal-prefix equality, rollout-vs-predict) are
**bit-identical on the GPU too**: `max|d| 0.000e+00` for every T ≥ 2 prefix and both rollout steps,
at all three dtypes.

### f16 overflow and the one-pass variance, measured on real activations

Both CUDA numerics questions above turn on how large real activations get, so they were measured
directly: forward hooks on the reference PyTorch models over the fixture inputs (1.66 M LayerNorm
rows):

* **f16 GEMM accumulation cannot overflow these models.** The largest linear-layer output anywhere
  is **414** (V-JEPA 2.1 ViT-B, layer 0 fused qkv); I-JEPA ViT-H — the model the docs used to
  describe as reaching ~2e4 — peaks at **95**, with a largest residual value of 151 and a largest
  unscaled q·k of 201. The number that actually decides overflow is the running partial sum inside
  the GEMM: measured in float64 it peaks at **412**, and the unattainable all-same-sign ceiling
  `Σ|x_k w_k|` over *every* element of *every* linear is **600** — 109× below `half`'s 65504.
  `GGML_PREC_F32` is kept for **mantissa**, not range.
* **CUDA's one-pass `ggml_norm` variance is not a risk for these models.** Its failure is governed
  by |row mean| / row σ (1e-7 at ratio 0, 8.9e-4 at 100, catastrophic at 2000). Over 1.66 M real
  pre-LN rows the maximum ratio is **2.72** (I-JEPA ViT-H, layer 0, one token), the 99.9th
  percentile 2.30 and the median 0.06–1.35; no row anywhere exceeds 10. At that ratio both backends
  match a double-precision two-pass reference to ~1.8e-07. The `rel_max` gates above stay, because
  they cost nothing and this is the only failure mode cosine is blind to — but they are insurance,
  not a live concern.

## Flash attention K/V dtype

`ggml_flash_attn_ext` with K/V cast to F16 costs ~3 digits of worst-token cosine on ViT-H/14 f32
(cos min 0.9910 vs 1.000000, rel_max 3.5e-2 vs 8.6e-5). That is F16's 10-bit mantissa, not its range:
measured on the real forward, I-JEPA ViT-H's largest linear output is 95 and its largest residual
value 151 ("f16 overflow and the one-pass variance" above), nowhere near F16's 65504.
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

The `-i` frames of a clip may have different source sizes — each is resized/cropped on its own and the
planes are concatenated, which is bit-identical to preprocessing an equal-sized stack in one call
(verified against `<sample>.input.npy`: max|d| 0, 100 % of values equal) and lets the repo's
differently-sized fixture JPEGs (640×426 and 586×640) form one clip.
A single image given to a tubelet-2 model is repeated to fill the tubelet (what the HF processor does);
V-JEPA 2.1 instead takes its native 1-frame image path (`enc.patch_embed_img` + `img_mod_embed`), which
is what the `coco_*` reference samples use.

## Timing notes

* ms/item is the wall time of `ggml_backend_graph_compute` per image/clip (graph build + alloc excluded;
  `docs/benchmarks.md` measures that overhead — `wall_ms_mean − ms_mean` — at 0.3–0.8 ms for the
  image models and 3–99 ms from 2048 to 18432 tokens, where it scales with the token count because it
  is the host-side patchify and the output copy, not graph build).
  PyTorch baseline = `timing_s.forward_s` from the manifest (32 threads), summarised with the
  drop-first rule described under the image table, and with the caveats ² and ³ above for the video
  models.
* I-JEPA ViT-H/14 (with `GGML_LLAMAFILE=ON`): f32 185 ms, f16 156 ms, q8_0 139 ms at 32 threads and
  f16 122 ms at 96 threads vs PyTorch 249.8 ms (32 t, drop-first median) — 1.3–2.0× faster than the
  PyTorch CPU baseline; f16 halves the weight memory (1.2 GiB peak vs 2.4 GiB), q8_0 uses 671 MiB.
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
