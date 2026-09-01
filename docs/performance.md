# Performance

Speed and memory, measured. Every figure on this page is copied from a measurement artifact, and
each table names its source. The CPU tables have a committed machine-readable twin,
`tests/results/benchmarks.json`; **the GPU tables do not** — `scripts/gen_benchmarks_md.py` drops
`--gpu` runs, because [benchmarks.md](benchmarks.md) is keyed by thread count, so this page is the
document of record for them and the reproduce block below is how to regenerate them.

## Environment

| | |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, AVX-512, 251 GB RAM |
| Build | gcc 13.3.0, `-O3 -march=native`, ggml `36da5713` (v0.22.0), `GGML_LLAMAFILE=ON`, kernel 6.17.0-1032-oem |
| GPU | 2 × NVIDIA RTX 4500 Ada Generation, 24 GB each, compute 8.9, 210 W board limit; every measurement uses device 0 alone |
| CUDA | runtime 13.0, `nvcc` 13.0.88, driver 580.173.02 |
| PyTorch (CPU) | torch 2.13.0+cpu, transformers 5.16.1, float32, 32 threads |
| PyTorch (GPU) | torch 2.13.0+cu130, transformers 5.16.1, `attn_implementation="sdpa"`, batch 1, TF32 off |
| Date | 2026-08-31, idle box |

`ms` is the wall time of `ggml_backend_graph_compute` for one item — model load, graph build, the
host-side patchify and the output copy are excluded on both backends. On the CPU the full
`jepa_encode` call runs 0.3–0.8 ms above that up to 1024 tokens and 3.4–98.9 ms above it beyond,
dominated by the patchify and the output copy rather than by graph build. On a GPU the same two
stages become PCIe transfers and the gap is 3–14 %: 16.0 against 15.5 ms for I-JEPA at 256 tokens,
49.4 against 46.5 for the 16-frame ViT-L clip, and 349 against 306 for the 64-frame one, where 8 192
patch rows of 1 536 floats are built on the host and copied across.

Every GPU row is the best of 5 runs after 2 warmups, and the warmups are not a formality: ggml's CUDA
backend captures a CUDA graph once it has seen the same topology and the same tensor addresses twice
in a row, so from the third call the encoder is one graph launch instead of roughly 700 kernel
launches. The first two calls of a process are materially slower, and that matters most on the
small-N models where per-launch overhead is a real fraction of the forward.

## CPU encoder

| model | shape | tokens | f32 t=32 | f16 t=32 | q8_0 t=32 | f16 t=96 | PyTorch t=32 | f16 speedup |
|---|---|---:|---:|---:|---:|---:|---:|---|
| I-JEPA ViT-H/14 | 224² | 256 | 174 | 147 | 129 | 113 | 250 | **1.70×** (2.21× @96) |
| LeJEPA ViT-S/16 | 224² | 197 | 13.1 | 12.8 | 11.3 | – | 15.4 | **1.20×** |
| LeWM ViT-Ti/14 | 224² | 257 | 9.2 | 9.8 | 9.1 | – | 16.8 ᶜ | **1.72×** ᶜ |
| V-JEPA 2 ViT-L SSv2 | 16 f 256² | 2 048 | 943 | 823 | 793 | 564 | 1051 ᵃ | n/a ᵃ |
| V-JEPA 2 ViT-L fpc64 | 16 f 256² | 2 048 | 941 | 821 | 794 | 567 | 1293 ᵇ | n/a ᵇ |
| V-JEPA 2 ViT-L fpc64 | 64 f 256² | 8 192 | 7020 | 6388 | 6482 | 4027 | 10114 ᵇ | n/a ᵇ |
| V-JEPA 2.1 ViT-B/384 | 384² | 576 | 70.0 | 60.3 | 58.5 | 52.7 | 110 | **1.82×** (2.09× @96) |
| V-JEPA 2.1 ViT-B/384 | 16 f 384² | 4 608 | 826 | 853 | 914 | 636 | 908 | **1.06×** |
| V-JEPA 2.1 ViT-B/384 | 64 f 384² | 18 432 | 9050 | 9036 | 9487 | 5040 | – | – |
| LeVJEPA ViT-L/16 | 16 f 224² | 3 137 | 1512 | 1480 | 1508 | 882 | 1752 | **1.18×** (1.99× @96) |

ᵃ the SSv2 reference forward is encoder + attentive pooler + classifier; the like-for-like comparison
is the end-to-end table below.
ᵇ the fpc64 reference forward always runs the predictor as well, so it is an upper bound and no
speedup is claimed against it.
ᶜ the LeWM reference forward is encode + projector + one 1-frame predictor call; the two extra graphs
are ~1 ms of it, so this speedup is a slight over-estimate.

The PyTorch column is the mean over the reference samples of the same frame count, or the median after
a cold first sample where that first sample is ≥ 1.2× the median of the rest (I-JEPA: 331.8 ms first
against a steady 250; LeJEPA: 72.2 against 15.4).

Those samples are single-shot forwards, which is the noisier of the two ways to time a reference.
[parity.md](parity.md#levjepa-vitl16--cls-block-causal-attention-tubelet-1) divides by a **warm loop**
instead — 1 warmup then 5 forwards on one clip — and for LeVJEPA the two differ by 8 %: 1752 ms here
against 1904 ms median (1883 ms minimum) there, so the same f16 file reads 1.18× on this page and 1.2×
on that one. The column keeps the single-shot rule so that every row divides by the same thing.

*Source: [benchmarks.md](benchmarks.md#encoder), `tools/jepa-bench` on synthetic deterministic input,
1 warmup + 3 measured runs, 32-thread sessions 2026-08-31 11:32 and 23:02 UTC and 96-thread sessions
11:44 and 23:05 UTC, all starting on an idle box; PyTorch column from the fixture manifests'
`timing_s.forward_s`. Machine-readable twin: `tests/results/benchmarks.json`.*

## GPU encoder

Optional build (`-DJEPA_CUDA=ON`, then `--gpu [N]`). Best of 5 runs after 2 warmups, `GGML_PREC_F32`
on every `mul_mat` (the GPU default). The `CPU f16 t=32` column repeats the 32-thread f16 row of the
encoder table above, i.e. 96 Zen 4 cores' worth of machine against one workstation card.

| model | shape | tokens | f32 | f16 | q8_0 | q4_k | CPU f16 t=32 | **f16 speed-up** |
|---|---|---:|---:|---:|---:|---:|---:|---|
| I-JEPA ViT-H/14 | 224² | 256 | 11.8 | 15.5 | 8.0 | **7.8** | 147 | **9.5×** |
| LeJEPA ViT-S/16 | 224² | 197 | – | **1.1** | – | – | 12.8 | **11.6×** |
| LeWM ViT-Ti/14 | 224² | 257 | – | **0.8** | – | – | 9.8 | **12.3×** |
| V-JEPA 2 ViT-L fpc64 | 16 f 256² | 2 048 | 44.3 | 46.5 | **34.4** | 34.8 | 821 | **17.6×** |
| V-JEPA 2 ViT-L fpc64 | 64 f 256² | 8 192 | 303 | 306 | **281** | 282 | 6388 | **20.9×** |
| V-JEPA 2.1 ViT-B/384 | 384² | 576 | – | **4.4** | – | 3.5 | 60.3 | **13.7×** |
| V-JEPA 2.1 ViT-B/384 | 16 f 384² | 4 608 | – | **43.0** | – | 37.3 | 853 | **19.8×** |
| V-JEPA 2.1 ViT-B/384 | 64 f 384² | 18 432 | – | **424** | – | – | 9036 | **21.3×** |
| LeVJEPA ViT-L/16 | 16 f 224² | 3 137 | 85.4 | 87.2 | **70.8** | 71.2 | 1480 | **17.0×** |

Against 96 CPU threads the same rows read 7.3× (I-JEPA), 12.2× and 13.2× (V-JEPA 2 ViT-L at 16 and 64
frames), 12.0×, 14.8× and 11.9× (V-JEPA 2.1 at 576, 4 608 and 18 432 tokens) and 10.1× (LeVJEPA).

The ratio grows with the sequence, and two effects separate the ends of it. The long clips keep the
card busy. The small image models are held back both by launch overhead — LeJEPA's whole forward is
1.1 ms — and, more than that, by `GGML_PREC_F32`, which costs 1.76× at I-JEPA's 256 tokens against
1.06× at 8 192; that is why the image models land at 9.5–13.7× rather than in the long clips'
twenties.

*Source: `tools/jepa-bench --gpu 0` on the box above, `GGML_PREC_F32` on. No committed
machine-readable twin exists for these rows — see the reproduce block at the foot of the page.*

![Milliseconds per image or clip, one row per model and shape, four bars each: PyTorch on 32 CPU
threads, jepa.cpp on 32 CPU threads, PyTorch on one GPU and jepa.cpp on one GPU, on a log
scale](assets/results-latency.svg)

*Both tables above in one picture: the bar is the millisecond, the white mark inside a jepa.cpp CPU
bar is the same run at 96 threads, and a hatched PyTorch bar is a forward that does more work than
ours, i.e. an upper bound. `scripts/gen_results_figure.py --split` redraws it.*

### Predictor, head and world model on a GPU

These graphs are not encoder rows and do not share the encoder table's harness, so they are reported
with their own shapes and sources.

| graph | shape | CUDA f16 | CPU f16 t=32 | ratio |
|---|---|---:|---:|---|
| V-JEPA 2 ViT-L masked predictor, *archery* | 4 096 rows (2 048 context + 2 048 mask) | 169 | 452 | 2.7× |
| V-JEPA 2 ViT-L masked predictor, *bowling* | 4 096 rows | 113 | 326 | 2.9× |
| SSv2 attentive-pool head | 2 048 tokens | 5.7 | 96 ᵈ | 16.8× |
| LeWM predictor, `pred_next` | 1 row | 0.53 | 0.65 | 1.2× |
| LeWM predictor, `pred_seq` | 3 rows | 3.80 | 1.22 | **0.3× — slower on the GPU** |

The masked-predictor and LeWM rows come from one `test-predictor` run per backend on the same fixture
clips, so each line is like-for-like; *archery* carries the first-touch page-in of its run, which is
why *bowling* is faster at the same shape on both backends.

The masked predictor gains **2.7–2.9×, where the ViT-L encoder on the very same clip gains 17.6×**,
and that is the price of the naive attention path it takes at `head_dim` 32 — accurate, genuinely
F32, and ~3 TFLOP/s against flash's 50–70. The LeWM predictor is the one graph that is *slower* on a GPU at its real shape: three
rows of 192 dimensions is far below the size at which a launch pays for itself.

ᵈ 96 ms is `jepa-classify --time` on the fixture clip; the synthetic `head` mode of `jepa-bench`
measures the same graph at 99.0 ms (see the end-to-end table below). `jepa-bench --mode lewm-step`
likewise reads 0.918 ms for the full 3-frame window against `test-predictor`'s 1.22 ms, on synthetic
rather than reference state.

*Source: [parity.md](parity.md#results-predictors-on-cuda0) and
[parity.md](parity.md#v-jepa-2-vit-l-masked-predictor-ctx-tgt-all-2048-tokens-of-a-16-frame-clip) for
the predictor and LeWM rows; `jepa-bench --gpu 0` / `jepa-classify --time` for the head row.*

## GPU against PyTorch on the same card

`VJEPA2Model(skip_predictor=True)` / `IJepaModel` / `LeVJEPAModel`, 3 warmup + 7 timed forwards with
`cuda.synchronize()` around each.

| shape | jepa.cpp CPU t=32 | **jepa.cpp CUDA** | with `--gpu-prec f16` | torch fp16 | torch fp32 | ggml / torch fp16 |
|---|---:|---:|---:|---:|---:|---:|
| I-JEPA ViT-H, 224² | 147.0 | **15.5** | 8.8 | 5.91 | 24.26 | 2.6× / 1.5× |
| V-JEPA 2 ViT-L, 16 f | 820.7 | **46.5** | 37.4 | 28.74 | 115.62 | 1.6× / 1.3× |
| V-JEPA 2 ViT-L, 64 f | 6388.1 | **306** | 304 | 147.5 | 838.38 | 2.1× / 2.1× |
| LeVJEPA ViT-L, 16 f | 1480 | **87.2** | 82.9 | 58.51 | 222.64 | 1.5× / 1.4× |

jepa.cpp-CUDA lands at **38–67 % of PyTorch's throughput on the same GPU** at its default precision
and 48–77 % with `--gpu-prec f16`, while being 9–21× faster than the CPU engine. LeVJEPA is the closest
of the four (67 % / 71 %), and the mask is why: its explicit attention mask disqualifies PyTorch's flash
SDPA kernel, which its own model card warns about, so the reference gives up more than jepa.cpp does. `--gpu-prec f16` is
bench-only — it is not exposed in the runtime tools and is not parity-gated, so those cells are a
measured upper bound rather than a shipping configuration.

For scale, the card's own ceilings computed from what `nvidia-smi` reports — 7 680 CUDA cores at a
max SM clock of 3 105 MHz under a 210 W board limit — are **47.7 TFLOP/s FP32** and, at the 2× dense
rate of Ada's 4th-generation tensor cores, **~95 TFLOP/s FP16 with FP32 accumulate**. Sustained
clocks under that power cap are lower, so both are upper bounds; PyTorch reaches 56–77 TFLOP/s at
fp16 and ~14 TFLOP/s at fp32 on this card, which is the practical ceiling to read the percentages
against.

The remaining gap is everything that is neither a GEMM nor an attention: 48 unfused LayerNorms per
ViT-L forward (ggml-CUDA fuses `{RMS_NORM, MUL, ADD}` but has no `{NORM, MUL, ADD}` pattern, and ViTs
use LayerNorm), `gelu_erf` over the FFN hidden twice per layer, and the per-layer F32→F16 K/V casts.
At the 64-frame shape the component rates account for 182 ms of the 306 measured.

Memory on the card: torch's fp16 peak GPU memory is 1.19 GiB (I-JEPA at 256 tokens), 0.67 GiB
(ViT-L at 2 048) and 1.15 GiB (LeVJEPA at 3 137, where the `3137²` mask is 20 MiB per layer input on
its own), against 0.83 GiB at 8 192.

*Source: the torch-GPU baseline as re-measured after the port. Every row reproduced to within 3 % of
the earlier session except I-JEPA fp16, which moved +6.1 % — jitter at that scale, since the row has
a 0.34 ms run-to-run σ at 5.91 ms and its minimum, 5.62 ms, is within 1 % of the earlier
figure.*

## End-to-end video classification

V-JEPA 2 ViT-L SSv2, 16-frame 256² clip, 2 048 tokens. The reference forward here is
`VJEPA2ForVideoClassification` — encoder + attentive pooler + classifier with the predictor skipped —
so this comparison is like-for-like.

| ftype | threads | encoder ms | head ms | total ms | PyTorch ms | speedup |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 32 | 943 | 107 | 1050 | 1051 | 1.00× |
| f16 | 32 | 823 | 99.0 | 922 | 1051 | **1.14×** |
| f16 | 96 | 564 | 67.1 | 631 | 1051 | **1.66×** |
| q8_0 | 32 | 793 | 98.2 | 891 | 1051 | **1.18×** |

*Source: [benchmarks.md](benchmarks.md#attentive-pool-head).*

## Predictors and world model

Predictor worst case: context = target = *every* token, i.e. 2 × tokens through the 12-layer 384-d
predictor.

| model | tokens | f32 t=32 | f16 t=32 | q8_0 t=32 | f16 t=96 |
|---|---:|---:|---:|---:|---:|
| V-JEPA 2 ViT-L SSv2 | 2 048 | 333.1 | 340.8 | 336.8 | 208.0 |
| V-JEPA 2 ViT-L fpc64 | 2 048 | 338.3 | 343.8 | 332.7 | 207.4 |
| V-JEPA 2.1 ViT-B/384 | 4 608 | 1296.7 | 1303.6 | 1272.6 | 854.4 |

On the real fixture clips the same 2 048-token V-JEPA 2 predictor runs 452 / 326 ms at f16 against a
PyTorch predictor call of 2524 ms — **~5.6×** — and costs 40–55 % of the encoder pass on the same clip.

The LeWM Push-T world model (D = 192, 3-frame causal window) is launch-bound: one full-window
`jepa_lewm_predict` takes 0.911 / 0.918 / 0.739 ms at f32 / f16 / q8_0, and `jepa_lewm_rollout` runs
0.817 / 0.865 / 0.744 ms **per step** — 1225 / 1156 / 1344 steps per second, encoder excluded.

*Source: [benchmarks.md](benchmarks.md#masked-predictor) and
[benchmarks.md](benchmarks.md#lewm-world-model); the fixture-clip predictor timings from
[parity.md](parity.md#results-predictors-masked-v-jepa-2-21-predictor-lewm-world-model).*

## Batched image encoding

`ms` is `jepa-bench --batch B` on synthetic input; `img/s` is `jepa-embed` over 561 Imagenette-160
validation JPEGs with decode, preprocessing and one GGUF load inside the number, best of two passes.

| model | dtype | ms/image b=1 | b=8 | b=32 | img/s b=1 → 8 → 32 | PyTorch batch 32 | peak RSS b=1 → 32 |
|---|---|---:|---:|---:|---|---:|---|
| LeJEPA ViT-S/16 | f16 | 12.6 | 8.1 | **7.4** | 67.0 → 94.4 → **94.5** | 86.4–89.2 | 52 → 148 MiB |
| LeWM ViT-Ti/14 | f16 | 9.7 | 4.8 | **4.4** | 95.4 → 148.5 → **159.0** | 189.7–206.2 | 48 → 107 MiB |
| I-JEPA ViT-H/14 | f16 | 148.8 | **137.9** | 140.3 | 6.16 → **6.92** → 6.77 | 5.45–5.51 | 1230 → 1597 MiB |
| I-JEPA ViT-H/14 | q8_0 | 139.4 | **129.8** | 129.8 | 6.94 → **7.60** → 7.38 | 5.45–5.51 | 659 → 993 MiB |

The PyTorch baseline is f32 in every row.

The win is inversely proportional to model size. Batching amortises what does not scale with the
matmuls — thread launch, the per-layer norm and GELU passes, weight streaming — so the two small
models gain 1.7× and 2.2× of encoder time while ViT-H, whose 1.2 GiB of weights already keep 32
threads busy on one image, gains 1.08×. B = 32 is also *worse* than B = 8 there (140.3 against
137.9 ms): peak RSS grows to 1.6 GiB — 1.2 GiB of weights plus a graph arena that has gone from
~24 MiB at B = 1 to ~391 MiB — and the weights stop fitting alongside it in cache.
LeJEPA passes PyTorch's batch-32 throughput, LeWM reaches 0.77–0.84× of it, and video is not batched.

*Source: `tests/results/batching.json`, 32 threads, every timed pass run alone;
[accuracy-image.md](accuracy-image.md#throughput-32-threads-end-to-end-jpeg-decode-preprocess-encode)
has the per-split throughput rows.*

## Memory

| model | f32 | f16 | q8_0 | q4_k | peak RSS, f16 at its largest shape |
|---|---:|---:|---:|---:|---:|
| I-JEPA ViT-H/14 | 2406 | 1206 | 644 | 344 | 1230 (256 tok) |
| LeJEPA ViT-S/16 | 83 | 42 | 23 | 13 | 52 (197 tok) |
| LeWM Push-T | 69 | 38 | 23 | 15 | 47 (257 tok) |
| V-JEPA 2 ViT-L SSv2 | 1432 | 717 | 383 | 205 | 808 (2 048 tok) |
| V-JEPA 2 ViT-L fpc64 | 1243 | 622 | 333 | 178 | 1034 (8 192 tok) |
| V-JEPA 2.1 ViT-B/384 | 419 | 210 | 113 | 62 | 948 (18 432 tok) |
| LeVJEPA ViT-L/16 | 1156 | 579 | 310 | 166 | 779 (3 137 tok) |

Weights resident, MiB (`jepa_model_n_bytes()`); peak RSS additionally covers the graph arena, the
host-side patch buffer and the output rows, so it grows with the token count — linearly, with one
exception. A `block_causal` file (LeVJEPA) also holds an `N × N` F16 attention mask, on the host and
in the arena: 39 MiB of the 779 above at 3 137 rows, but **600 MiB at 64 frames** (12 545 rows), where
it becomes the largest single allocation in the process. `$JEPA_MAX_GRAPH_MIB` bounds it and
`jepa_encode` refuses a clip over that ceiling rather than allocating
([architecture.md](architecture.md#block-causal-attention)). Across these seven
models q8_0 holds **0.53–0.61×** the resident f16 weights and q4 **0.29–0.40×** (LeWM is the high end
of both, because a larger share of its file is the F32 remainder — adaLN, action embedder, position
tables — that no type touches). The clean 0.53× / 0.29× ratios quoted for *file* sizes, from the 8.5
and 4.5 bits per stored weight, are in [quantization.md](quantization.md#file-sizes).

*Source: [benchmarks.md](benchmarks.md#memory).*

## Thread scaling

| model | mode | shape | tokens | ms t=32 | ms t=96 | speedup |
|---|---|---|---:|---:|---:|---:|
| I-JEPA ViT-H/14 | encoder | 224² | 256 | 147 | 113 | 1.30× |
| V-JEPA 2 ViT-L SSv2 | encoder | 16 f 256² | 2 048 | 823 | 564 | 1.46× |
| V-JEPA 2 ViT-L SSv2 | head | 16 f 256² | 2 048 | 99.0 | 67.1 | 1.48× |
| V-JEPA 2 ViT-L SSv2 | predictor | 16 f 256² | 2 048 | 341 | 208 | 1.64× |
| V-JEPA 2 ViT-L fpc64 | encoder | 64 f 256² | 8 192 | 6388 | 4027 | 1.59× |
| V-JEPA 2.1 ViT-B/384 | encoder | 384² | 576 | 60.3 | 52.7 | 1.15× |
| V-JEPA 2.1 ViT-B/384 | encoder | 16 f 384² | 4 608 | 853 | 636 | 1.34× |
| V-JEPA 2.1 ViT-B/384 | encoder | 64 f 384² | 18 432 | 9036 | 5040 | 1.79× |
| V-JEPA 2.1 ViT-B/384 | predictor | 16 f 384² | 4 608 | 1304 | 854 | 1.53× |
| LeVJEPA ViT-L/16 | encoder | 16 f 224² | 3 137 | 1480 | 882 | 1.68× |

All rows f16. Tripling the threads never triples the throughput: the 32-thread runs already saturate a
good part of the memory bandwidth, and both the LayerNorm/GELU passes and the graph launch overhead
scale poorly. The gain is largest where a single matmul or attention tile is big enough to keep 96
workers busy.

*Source: [benchmarks.md](benchmarks.md#thread-scaling-32-96-threads).*

## Quantization and speed

**On the CPU, quantization buys memory, not time.** q8_0 lands at 0.93–1.14× of f16, and q4_k is a
loss on the wide matmuls: llamafile's accelerated sgemm covers F32, F16 and Q8_0, while the K-quants
fall back to ggml's generic vector dot product.

| model | shape | f16 ms | q8_0 ms | q4_0 ms | q4_k ms | f16 MiB | q4 MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| I-JEPA ViT-H/14 | 224² | 147 | 129 | 140 | 198 | 1206 | 344 |
| LeJEPA ViT-S/16 | 224² | 12.8 | 11.3 | 12.0 | 12.9 | 42 | 13 |
| V-JEPA 2 ViT-L fpc64 | 16 f 256² | 821 | 794 | 844 | 1096 | 622 | 178 |
| V-JEPA 2 ViT-L fpc64 | 64 f 256² | 6388 | 6482 | 6616 | 7405 | 622 | 178 |
| V-JEPA 2.1 ViT-B/384 | 384² | 60.3 | 58.5 | 59.7 | 90.7 | 210 | 62 |
| V-JEPA 2.1 ViT-B/384 | 16 f 384² | 853 | 914 | 895 | 1148 | 210 | 62 |
| LeVJEPA ViT-L/16 | 16 f 224² | 1480 | 1508 | 1537 | 1851 | 579 | 166 |

**On CUDA the ordering inverts.** Every type jepa.cpp ships takes `mmq`, a real INT8 tensor-core
kernel, so q4_k ties q8_0 and both beat f16 — while also being a quarter and a half of the weight
bytes. q8_0 is the faster of the two on the long clips and q4_k on the short shapes; the two are
within 3 % of each other everywhere both were measured.

| model / shape | f32 | f16 | q8_0 | q4_k | CPU f16 | CPU q4_k |
|---|---:|---:|---:|---:|---:|---:|
| I-JEPA ViT-H, 224² | 11.8 | 15.5 | 8.0 | **7.8** | 147 | 198 |
| V-JEPA 2 ViT-L, 16 f | 44.3 | 46.5 | **34.4** | 34.8 | 821 | 1096 |
| V-JEPA 2 ViT-L, 64 f | 303 | 306 | **281** | 282 | 6388 | 7405 |
| V-JEPA 2.1 ViT-B, 384² | – | 4.4 | – | **3.5** | 60.3 | 90.7 |
| V-JEPA 2.1 ViT-B, 16 f | – | 43.0 | – | **37.3** | 853 | 1148 |
| LeVJEPA ViT-L, 16 f | 85.4 | 87.2 | **70.8** | 71.2 | 1480 | 1851 |

The f32 GPU column is not slower than f16 because ggml's CUDA F32 path is TF32 while the f16 path pays
for `GGML_PREC_F32` accumulation. Accuracy per type does *not* invert with the backend; see
[Accuracy → which dtype to ship](accuracy.md#which-dtype-to-ship).

*Source: CPU rows from [benchmarks.md](benchmarks.md#sub-8-bit-weights-what-q4-costs-and-what-it-buys-encoder-t32)
and its dtype table; GPU rows from the same `jepa-bench --gpu 0` sweep as the GPU encoder table.*

![Two small charts, one per model: latency against the same backend's f16 file, on the CPU and on
CUDA, over the resident weights of each dtype, with the PyTorch CPU and PyTorch CUDA
levels](assets/results-quantization.svg)

*The trade-off per dtype, for the two models measured over the full sweep on both backends: the x
axis is what the file weighs, the y axis is its latency against the same backend's own f16, so a
point below 1× is faster than f16 and a point to the left is smaller.
`scripts/gen_results_figure.py --split` redraws it.*

## Reproduce

```bash
# the whole CPU matrix at 32 threads, then the q4 rows and the 96-thread rows, into tmp/bench/
scripts/bench_all.sh 32
scripts/bench_all.sh 32 --keep --include-quants --only '\-(q4_0|q4_k)$' --modes encoder
scripts/bench_all.sh 96 --keep --only 'ijepa.*-f16|vjepa2-vitl.*-f16|vjepa2_1.*-f16'

# regenerate the raw report and its JSON twin
scripts/gen_benchmarks_md.py --bench-dir tmp/bench --ref-dir tests/fixtures/ref \
    --parity docs/parity.md -o docs/benchmarks.md --results-json tests/results/benchmarks.json

# one CPU configuration by hand
build/jepa-bench -m models/gguf/vjepa2-vitl-fpc64-256-f16.gguf --frames 64 --threads 32,96 --md
```

`bench_all.sh` writes one JSON per (file, mode, shape) plus a `meta.json`; `tmp/bench/` is
git-ignored and `tests/results/benchmarks.json` is the committed twin.

The GPU tables are produced one configuration at a time — `bench_all.sh` has no `--gpu` mode and
`gen_benchmarks_md.py` drops `--gpu` runs, so nothing aggregates them into a committed JSON:

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DJEPA_CUDA=ON && cmake --build build-cuda -j 16

# encoder rows: one invocation per (file, shape), best of 5 after 2 warmups
build-cuda/jepa-bench -m models/gguf/ijepa_vith14_1k-f16.gguf      --gpu 0 --warmup 2 --repeat 5 --md
build-cuda/jepa-bench -m models/gguf/vjepa2-vitl-fpc64-256-q8_0.gguf --frames 64 --gpu 0 --warmup 2 --repeat 5 --md
build-cuda/jepa-bench -m models/gguf/levjepa-vitl16-q8_0.gguf --frames 16 --gpu 0 --warmup 2 --repeat 5 --md

# head and predictor rows
build-cuda/jepa-bench -m models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf --mode head      --gpu 0 --md
build-cuda/test-predictor --vjepa2 models/gguf/vjepa2-vitl-fpc64-256-f16.gguf \
    --ref tests/fixtures/ref/vjepa2-vitl-fpc64-256 --samples archery_f16,bowling_f16 --gpu 0

# the precision opt-out behind the --gpu-prec f16 column (bench-only, not parity-gated)
build-cuda/jepa-bench -m models/gguf/ijepa_vith14_1k-f16.gguf --gpu 0 --gpu-prec f16 --md
```
