# jepa.cpp — measured benchmarks

Every number here comes from `tools/jepa-bench` on the box described below, on **synthetic but deterministic** input (a seeded uint8 stream put through the model's own `jepa.pre.*` normalisation), so the tables can be reproduced without the fixture media or a Python environment. They are cross-checked against `docs/parity.md`, which times the same graphs on the real reference clips (see the cross-check table below).

## How to reproduce

```bash
git submodule update --init ggml
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 32

# the whole matrix at 32 threads (every f32/f16/q8_0 GGUF in models/gguf, every mode it supports)
scripts/bench_all.sh 32

# the big configurations at 96 threads, appended to the same tmp/bench directory
scripts/bench_all.sh 96 --keep --only 'ijepa.*-f16|vjepa2-vitl.*-f16|vjepa2_1.*-f16'

# a single configuration by hand
build/jepa-bench -m models/gguf/vjepa2-vitl-fpc64-256-f16.gguf --frames 64 --threads 32,96 --md
```

`bench_all.sh` writes one JSON per (file, mode, shape) into `tmp/bench/` plus a `meta.json`, then rebuilds this document with `scripts/gen_benchmarks_md.py --bench-dir tmp/bench --ref-dir tests/fixtures/ref -o docs/benchmarks.md`. Add `--include-quants` to sweep the q4/q5/q6 files as well (skipped by default).

## Box and build

| setting | value |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 7995WX 96-Cores — 192 hardware threads, AVX-512 |
| RAM | 251 GB |
| Kernel | 6.17.0-1032-oem |
| Compiler | c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, `-O3 -march=native` (`JEPA_NATIVE=ON`) |
| ggml | `36da5713`, **`GGML_LLAMAFILE=ON`** |
| Attention | `ggml_flash_attn_ext`; K/V dtype auto (F32 for f32 files, F16 otherwise) |
| Thread counts | 32, 96 |

Measurement sessions (one `bench_all.sh` invocation each):

| threads | warmup + measured | start | end | 1-min load avg | note |
|---:|---|---|---|---|---|
| 32 | 1 + 3 | 2026-08-30 21:29 UTC | 2026-08-30 21:34 UTC | 14.62 → 33.02 | shared box: a second implementation agent was running jobs during part of this session |
| 96 | 1 + 3 | 2026-08-30 21:34 UTC | 2026-08-30 21:35 UTC | 24.35 → 64.15 | the single budgeted 96-thread session, run back-to-back with the 32-thread one; the rising load average is this session's own 96 workers |
| 32 | 1 + 3 | 2026-08-30 21:36 UTC | 2026-08-30 21:37 UTC | 25.61 → 27.82 | re-measurement of the vjepa2_1 f16 encoder rows: their first 32-thread run overlapped a burst from the other agent |

The box is shared with other agents, so the 1-minute load average is recorded per session (out of 192 hardware threads; a session's own run contributes its thread count). Where a session ran against a busy box, `ms min` — the least contended of the measured runs — is the better estimate of the uncontended cost, and the tables print it next to the mean.

**What the milliseconds are.** `ms` is the wall time of `ggml_backend_graph_compute` for the named graph (`jepa_context_last_compute_ms`) — model load, graph build/allocation, the host-side patchify and the output copy are excluded; measured as `wall_ms_mean − ms_mean` on an idle box at 32 threads they add 0.4–0.7 ms for the image models and 6–13 ms at 2048–4608 tokens (dominated by the patchify and the output copy, not by graph build). The JSONs keep the full API-call time as `wall_ms_mean`. `tokens/s` is `tokens / ms_mean`. `peak RSS` is the process `VmHWM` after the run, i.e. weights + the largest graph allocation, not a per-graph figure. For `lewm-rollout` the reported ms is **per rollout step** (the K graphs of one `jepa_lewm_rollout` call divided by K).

## Encoder

| model | ftype | shape | tokens | threads | ms mean | ms min | tokens/s | PyTorch ms | speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | f32 | 224x224 | 256 | 32 | 178.2 | 177.4 | 1436 | 263 | 1.48x |
| ijepa_vith14_1k | f16 | 224x224 | 256 | 32 | 151.5 | 150.0 | 1689 | 263 | 1.74x |
| ijepa_vith14_1k | f16 | 224x224 | 256 | 96 | 116.6 | 116.0 | 2195 | 263 | 2.26x |
| ijepa_vith14_1k | q8_0 | 224x224 | 256 | 32 | 135.6 | 133.5 | 1887 | 263 | 1.94x |
| lejepa-vits16-pretrain-in1k | f32 | 224x224 | 197 | 32 | 13.2 | 13.0 | 14917 | 22.6 | 1.71x |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 197 | 32 | 12.8 | 12.8 | 15406 | 22.6 | 1.77x |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | 197 | 32 | 11.2 | 11.1 | 17658 | 22.6 | 2.02x |
| lewm-pusht | f32 | 224x224 | 257 | 32 | 9.2 | 9.2 | 27918 | 16.8<sup>lewm</sup> | 1.82x |
| lewm-pusht | f16 | 224x224 | 257 | 32 | 9.7 | 9.6 | 26502 | 16.8<sup>lewm</sup> | 1.73x |
| lewm-pusht | q8_0 | 224x224 | 257 | 32 | 8.8 | 8.8 | 29112 | 16.8<sup>lewm</sup> | 1.90x |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 2 048 | 32 | 954.9 | 952.3 | 2145 | 1051<sup>ssv2</sup> | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 32 | 851.4 | 849.8 | 2405 | 1051<sup>ssv2</sup> | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 96 | 585.3 | 562.0 | 3499 | 1051<sup>ssv2</sup> | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 2 048 | 32 | 809.1 | 805.6 | 2531 | 1051<sup>ssv2</sup> | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 2 048 | 32 | 1055.5 | 1020.4 | 1940 | 1293<sup>fpc64</sup> | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 32 | 899.8 | 886.6 | 2276 | 1293<sup>fpc64</sup> | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 96 | 580.6 | 561.2 | 3527 | 1293<sup>fpc64</sup> | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | 32 | 824.2 | 814.0 | 2485 | 1293<sup>fpc64</sup> | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f32 | 64f 256x256 | 8 192 | 32 | 6511.7 | 6255.1 | 1258 | 10114<sup>fpc64</sup> | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 8 192 | 32 | 6571.5 | 6508.1 | 1247 | 10114<sup>fpc64</sup> | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 8 192 | 96 | 4074.4 | 4039.8 | 2011 | 10114<sup>fpc64</sup> | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | q8_0 | 64f 256x256 | 8 192 | 32 | 6532.8 | 6484.1 | 1254 | 10114<sup>fpc64</sup> | n/a<sup>fpc64</sup> |
| vjepa2_1-vitb-384 | f32 | 384x384 | 576 | 32 | 71.3 | 70.8 | 8083 | 110 | 1.54x |
| vjepa2_1-vitb-384 | f16 | 384x384 | 576 | 32 | 61.5 | 61.4 | 9363 | 110 | 1.79x |
| vjepa2_1-vitb-384 | f16 | 384x384 | 576 | 96 | 53.2 | 52.9 | 10826 | 110 | 2.07x |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | 576 | 32 | 59.7 | 58.8 | 9650 | 110 | 1.84x |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 4 608 | 32 | 846.9 | 836.9 | 5441 | 908 | 1.07x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 32 | 892.4 | 878.1 | 5164 | 908 | 1.02x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 96 | 641.7 | 629.9 | 7181 | 908 | 1.41x |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 4 608 | 32 | 909.2 | 906.3 | 5068 | 908 | 1.00x |
| vjepa2_1-vitb-384 | f32 | 64f 384x384 | 18 432 | 32 | 9215.4 | 9139.9 | 2000 | – | – |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 18 432 | 32 | 9363.8 | 9328.1 | 1968 | – | – |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 18 432 | 96 | 5192.7 | 5191.5 | 3550 | – | – |
| vjepa2_1-vitb-384 | q8_0 | 64f 384x384 | 18 432 | 32 | 9917.1 | 9675.5 | 1859 | – | – |

PyTorch baseline = the mean `timing_s.forward_s` over the reference samples with the same frame count in `tests/fixtures/ref/<model>/manifest.json` — the same box, CPU float32, torch 2.13.0+cpu, transformers 5.16.1, 32 threads. It is the model forward alone (no decode, no preprocessing). The speedup column is filled only where that forward is the same work as our encoder; see the footnotes for the three models where it is not.

### Encoder throughput summary (tokens/s)

| model | shape | tokens | f32 t=32 | f16 t=32 | q8_0 t=32 | f16 t=96 |
|---|---|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | 224x224 | 256 | 1436 | 1689 | 1887 | 2195 |
| lejepa-vits16-pretrain-in1k | 224x224 | 197 | 14917 | 15406 | 17658 | – |
| lewm-pusht | 224x224 | 257 | 27918 | 26502 | 29112 | – |
| vjepa2-vitl-fpc16-256-ssv2 | 16f 256x256 | 2 048 | 2145 | 2405 | 2531 | 3499 |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | 2 048 | 1940 | 2276 | 2485 | 3527 |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | 8 192 | 1258 | 1247 | 1254 | 2011 |
| vjepa2_1-vitb-384 | 384x384 | 576 | 8083 | 9363 | 9650 | 10826 |
| vjepa2_1-vitb-384 | 16f 384x384 | 4 608 | 5441 | 5164 | 5068 | 7181 |
| vjepa2_1-vitb-384 | 64f 384x384 | 18 432 | 2000 | 1968 | 1859 | 3550 |

### Effect of the weight dtype (encoder, t=32)

| model | shape | tokens | f32 ms | f16 ms | q8_0 ms | f32 → f16 | f16 → q8_0 |
|---|---|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | 224x224 | 256 | 178 | 152 | 136 | 1.18x | 1.12x |
| lejepa-vits16-pretrain-in1k | 224x224 | 197 | 13.2 | 12.8 | 11.2 | 1.03x | 1.15x |
| lewm-pusht | 224x224 | 257 | 9.2 | 9.7 | 8.8 | 0.95x | 1.10x |
| vjepa2-vitl-fpc16-256-ssv2 | 16f 256x256 | 2 048 | 955 | 851 | 809 | 1.12x | 1.05x |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | 2 048 | 1056 | 900 | 824 | 1.17x | 1.09x |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | 8 192 | 6512 | 6571 | 6533 | 0.99x | 1.01x |
| vjepa2_1-vitb-384 | 384x384 | 576 | 71.3 | 61.5 | 59.7 | 1.16x | 1.03x |
| vjepa2_1-vitb-384 | 16f 384x384 | 4 608 | 847 | 892 | 909 | 0.95x | 0.98x |
| vjepa2_1-vitb-384 | 64f 384x384 | 18 432 | 9215 | 9364 | 9917 | 0.98x | 0.94x |

Quantisation buys memory, not reliably time. The matmuls do get faster (`docs/ggml-notes.md` §5 measures 2.2 → 3.3 TFLOP/s going F32 → F16, and ~3.6-4.1 for Q8_0 on this box), but q8_0 pays for quantising the *activations* on every matmul, and on the long clips the flash-attention time — F32 work whatever the weights are — dominates the layer (`docs/ggml-notes.md` §3: 158 ms per ViT-L layer at 8192 tokens against ~62 ms of matmul). The small image models are launch- and LayerNorm-bound and barely move at all. Pick the dtype on the accuracy tables in `docs/parity.md` and `docs/quantization.md` and on the memory table below, not on these milliseconds.

### Thread scaling (32 → 96 threads)

| model | ftype | mode | shape | tokens | ms t=32 | ms t=96 | speedup |
|---|---|---|---|---:|---:|---:|---:|
| ijepa_vith14_1k | f16 | encoder | 224x224 | 256 | 152 | 117 | 1.30x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | encoder | 16f 256x256 | 2 048 | 851 | 585 | 1.45x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | head | 16f 256x256 | 2 048 | 102 | 69.0 | 1.47x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | predictor | 16f 256x256 | 2 048 | 349 | 208 | 1.67x |
| vjepa2-vitl-fpc64-256 | f16 | encoder | 16f 256x256 | 2 048 | 900 | 581 | 1.55x |
| vjepa2-vitl-fpc64-256 | f16 | encoder | 64f 256x256 | 8 192 | 6571 | 4074 | 1.61x |
| vjepa2-vitl-fpc64-256 | f16 | predictor | 16f 256x256 | 2 048 | 384 | 222 | 1.73x |
| vjepa2_1-vitb-384 | f16 | encoder | 384x384 | 576 | 61.5 | 53.2 | 1.16x |
| vjepa2_1-vitb-384 | f16 | encoder | 16f 384x384 | 4 608 | 892 | 642 | 1.39x |
| vjepa2_1-vitb-384 | f16 | encoder | 64f 384x384 | 18 432 | 9364 | 5193 | 1.80x |
| vjepa2_1-vitb-384 | f16 | predictor | 16f 384x384 | 4 608 | 1331 | 818 | 1.63x |

Tripling the threads never triples the throughput: the 32-thread runs already saturate a good part of the memory bandwidth, and both the LayerNorm/GELU passes and the graph launch overhead scale poorly. The gain is largest where a single matmul or flash-attention tile is big enough to keep 96 workers busy.

### Cross-check against `docs/parity.md`

`docs/parity.md` times the *same* encoder graphs on the real preprocessed fixture inputs, reporting the second sample of a run (after the weights are paged in) — effectively a best-of figure, so `ms min` is what it should be compared against. The synthetic input used here has the same shape and scale, so the two agree to within run-to-run noise; `docs/parity.md` itself puts that at ±10-15 % on this shared box.

| model | ftype | shape | threads | bench ms min | parity.md ms | delta |
|---|---|---|---:|---:|---:|---:|
| ijepa_vith14_1k | f32 | 224x224 | 32 | 177 | 185 | -4.1 % |
| ijepa_vith14_1k | f16 | 224x224 | 32 | 150 | 156 | -3.8 % |
| ijepa_vith14_1k | f16 | 224x224 | 96 | 116 | 122 | -4.9 % |
| ijepa_vith14_1k | q8_0 | 224x224 | 32 | 133 | 138 | -3.3 % |
| lejepa-vits16-pretrain-in1k | f32 | 224x224 | 32 | 13.0 | 14.0 | -6.9 % |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 32 | 12.8 | 13.4 | -4.8 % |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | 32 | 11.1 | 12.6 | -12.1 % |
| lewm-pusht | f32 | 224x224 | 32 | 9.2 | 10.3 | -10.8 % |
| lewm-pusht | f16 | 224x224 | 32 | 9.6 | 10.7 | -10.2 % |
| lewm-pusht | q8_0 | 224x224 | 32 | 8.8 | 9.7 | -9.8 % |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 32 | 952 | 924 | +3.1 % |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 32 | 850 | 814 | +4.4 % |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 32 | 806 | 771 | +4.5 % |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 32 | 887 | 827 | +7.2 % |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 32 | 814 | 852 | -4.5 % |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 32 | 6508 | 6386 | +1.9 % |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 96 | 4040 | 4076 | -0.9 % |
| vjepa2-vitl-fpc64-256 | q8_0 | 64f 256x256 | 32 | 6484 | 7203 | -10.0 % |
| vjepa2_1-vitb-384 | f32 | 384x384 | 32 | 70.8 | 70.0 | +1.1 % |
| vjepa2_1-vitb-384 | f16 | 384x384 | 32 | 61.4 | 63.0 | -2.6 % |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | 32 | 58.8 | 59.0 | -0.3 % |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 32 | 837 | 909 | -7.9 % |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 32 | 878 | 875 | +0.3 % |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 32 | 906 | 878 | +3.2 % |

## Memory

| model | ftype | shape | weights MiB | peak RSS MiB | model load ms |
|---|---|---|---:|---:|---:|
| ijepa_vith14_1k | f32 | 224x224 | 2406 | 2434 | 1172 |
| ijepa_vith14_1k | f16 | 224x224 | 1206 | 1230 | 531 |
| ijepa_vith14_1k | q8_0 | 224x224 | 644 | 660 | 289 |
| lejepa-vits16-pretrain-in1k | f32 | 224x224 | 83 | 94 | 46 |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 42 | 52 | 27 |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | 23 | 33 | 16 |
| lewm-pusht | f32 | 224x224 | 69 | 80 | 40 |
| lewm-pusht | f16 | 224x224 | 38 | 47 | 23 |
| lewm-pusht | q8_0 | 224x224 | 23 | 33 | 15 |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 1432 | 1522 | 645 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 717 | 810 | 327 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 383 | 472 | 186 |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 1243 | 1320 | 657 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 622 | 717 | 294 |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 333 | 415 | 164 |
| vjepa2-vitl-fpc64-256 | f32 | 64f 256x256 | 1243 | 1603 | 569 |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 622 | 1037 | 325 |
| vjepa2-vitl-fpc64-256 | q8_0 | 64f 256x256 | 333 | 714 | 174 |
| vjepa2_1-vitb-384 | f32 | 384x384 | 419 | 435 | 192 |
| vjepa2_1-vitb-384 | f16 | 384x384 | 210 | 226 | 108 |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | 113 | 127 | 62 |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 419 | 570 | 246 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 210 | 386 | 134 |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 113 | 272 | 76 |
| vjepa2_1-vitb-384 | f32 | 64f 384x384 | 419 | 1066 | 206 |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 210 | 958 | 107 |
| vjepa2_1-vitb-384 | q8_0 | 64f 384x384 | 113 | 816 | 62 |

`weights MiB` is `jepa_model_n_bytes()` (the tensor bytes resident after the load, i.e. the GGUF payload); `peak RSS` additionally covers the graph allocation, the host-side patch buffer and the output rows, so it grows with the token count — the same weights are listed once per shape so that growth is visible. The loader `fread`s every tensor into its own buffer (no `mmap`), so `model load ms` tracks the file size and the page-cache state; these numbers are all warm-cache. Compare the weight column with the GGUF file sizes in `docs/quantization.md`.

## Attentive-pool head

| model | ftype | shape | tokens | threads | ms mean | ms min | encoder ms | peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 2 048 | 32 | 108.0 | 107.8 | 952.2 | 1511 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 32 | 101.7 | 100.9 | 836.3 | 817 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 96 | 69.0 | 67.2 | 558.5 | 771 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 2 048 | 32 | 101.5 | 98.7 | 812.7 | 464 |

The classifier head (3 self-attention blocks over the tokens + one cross-attention query + MLP + linear) on top of the encoder output of the same clip, so the end-to-end classification cost is encoder + head. `encoder ms` is the pass that produced this row's input — a **single** graph (the second of two, so the weights are warm), not an average of `repeat` runs like the `ms` columns, so read the Encoder table for the encoder cost proper.

End-to-end classification against the reference — this *is* like-for-like, because the manifest's forward is `VJEPA2ForVideoClassification` (encoder + attentive pooler + classifier, predictor skipped):

| model | ftype | shape | threads | encoder ms | head ms | total ms | PyTorch ms | speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 32 | 955 | 108 | 1063 | 1051 | 0.99x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 32 | 851 | 102 | 953 | 1051 | 1.10x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 96 | 585 | 69.0 | 654 | 1051 | 1.61x |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 32 | 809 | 101 | 911 | 1051 | 1.15x |

## Masked predictor

| model | ftype | shape | tokens | threads | ms mean | ms min | encoder ms | peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 2 048 | 32 | 341.1 | 334.3 | 951.0 | 1521 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 32 | 348.7 | 347.7 | 837.5 | 818 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 96 | 208.3 | 197.6 | 560.1 | 780 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 2 048 | 32 | 338.5 | 336.2 | 806.5 | 469 |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 2 048 | 32 | 358.4 | 355.8 | 1005.6 | 1340 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 32 | 383.7 | 375.9 | 904.1 | 728 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 96 | 221.9 | 216.7 | 614.3 | 680 |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | 32 | 346.7 | 345.9 | 824.1 | 416 |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 4 608 | 32 | 1266.9 | 1235.3 | 838.5 | 584 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 32 | 1331.3 | 1329.4 | 913.7 | 413 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 96 | 818.0 | 806.6 | 587.0 | 391 |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 4 608 | 32 | 1370.6 | 1292.3 | 1148.1 | 283 |

Worst case for the predictor: context = target = **every** token, i.e. a sequence of 2 x tokens through the 12-layer 384-d predictor. `encoder ms` is the pass that produced this row's input — a **single** graph (the second of two, so the weights are warm), not an average of `repeat` runs like the `ms` columns, so read the Encoder table for the encoder cost proper.

## LeWM world model

| model | ftype | mode | shape | threads | ms mean | ms min | steps/s |
|---|---|---|---|---:|---:|---:|---:|
| lewm-pusht | f32 | lewm-step | 3f x 192d | 32 | 0.894 | 0.847 | 1119 |
| lewm-pusht | f16 | lewm-step | 3f x 192d | 32 | 0.924 | 0.898 | 1082 |
| lewm-pusht | q8_0 | lewm-step | 3f x 192d | 32 | 0.721 | 0.713 | 1386 |
| lewm-pusht | f32 | lewm-rollout | rollout K=20 | 32 | 0.825 | 0.816 | 1212 |
| lewm-pusht | f16 | lewm-rollout | rollout K=20 | 32 | 0.852 | 0.849 | 1174 |
| lewm-pusht | q8_0 | lewm-rollout | rollout K=20 | 32 | 0.735 | 0.717 | 1361 |

`lewm-step` is one `jepa_lewm_predict` over the predictor's full 3-frame window; `lewm-rollout` is `jepa_lewm_rollout` and its ms is **per step** (the growing window means the first steps are cheaper than the last). Neither includes the encoder or the projector — see the encoder table for `lewm-pusht` for the cost of turning an image into a world-model state.

## Footnotes

<sup>fpc64</sup> the `vjepa2-vitl-fpc64-256` manifest times one `VJEPA2Model` forward, which always runs the **predictor** as well (its `predictor_last_hidden_state` comes from the same call), so it is an upper bound on the encoder and no speedup is claimed against it.

<sup>ssv2</sup> the SSv2 manifest times `VJEPA2ForVideoClassification`, i.e. encoder + attentive pooler + classifier with the predictor skipped. It is therefore not comparable with the encoder row alone; the end-to-end table under *Attentive-pool head* adds our encoder and head and makes the comparison there.

<sup>lewm</sup> the LeWM manifest times encode + projector + one 1-frame predictor call; the two extra graphs are ~1 ms of it (see the world-model table), so the speedup is a slight over-estimate.

---

Generated by `scripts/gen_benchmarks_md.py` from 56 runs in `tmp/bench`. Cross-check against `docs/parity.md` (same graphs, real fixture inputs) and `docs/quantization.md` (accuracy per dtype).
