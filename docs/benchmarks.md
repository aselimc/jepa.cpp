# jepa.cpp — measured benchmarks

*Raw measurement report — the curated view is [Benchmarks → Performance](performance.md).*

Every number here comes from `tools/jepa-bench` on the box described below, on **synthetic but deterministic** input (a seeded uint8 stream put through the model's own `jepa.pre.*` normalisation), so the tables can be reproduced without the fixture media or a Python environment. They are cross-checked against `docs/parity.md`, which times the same graphs on the real reference clips (see the cross-check table below).

## How to reproduce

```bash
git submodule update --init ggml
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 32

# the whole matrix at 32 threads (every f32/f16/q8_0 GGUF in models/gguf, every mode it supports)
scripts/bench_all.sh 32

# the q4 encoder rows, appended (the size/speed trade-off; --include-quants alone adds q5/q6 too)
scripts/bench_all.sh 32 --keep --include-quants --only '\-(q4_0|q4_k)$' --modes encoder

# the big configurations at 96 threads, appended to the same tmp/bench directory
scripts/bench_all.sh 96 --keep --only 'ijepa.*-f16|vjepa2-vitl.*-f16|vjepa2_1.*-f16'

# a single configuration by hand
build/jepa-bench -m models/gguf/vjepa2-vitl-fpc64-256-f16.gguf --frames 64 --threads 32,96 --md
```

`bench_all.sh` writes one JSON per (file, mode, shape) into `tmp/bench/` plus a `meta.json`, then rebuilds this document with `scripts/gen_benchmarks_md.py --bench-dir tmp/bench --ref-dir tests/fixtures/ref --parity docs/parity.md -o docs/benchmarks.md --results-json tests/results/benchmarks.json`. `tmp/bench/` is git-ignored; the committed `tests/results/benchmarks.json` is the machine-readable twin of every table below, one row per configuration, so a number quoted elsewhere in the repo can be traced without it. It passes each file's `--ftype-label` from the filename: `general.file_type` records the most common stored tensor type, and a q4_k mix falls back to q4_0 for every tensor whose rows are not a multiple of the 256-element super-block, so the small models' q4_k files read back as q4_0 and would otherwise be tabulated as such (`ftype_gguf` in the JSONs keeps what the file actually says).

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
| 32 | 1 + 3 | 2026-08-31 11:32 UTC | 2026-08-31 11:37 UTC | 0.04 → 30.84 | official re-sweep, idle box (previous agents finished) |
| 32 | 1 + 3 | 2026-08-31 11:39 UTC | 2026-08-31 11:42 UTC | 3.24 → 30.08 | q4_0/q4_k encoder rows only (size/speed trade-off), idle box |
| 96 | 1 + 3 | 2026-08-31 11:44 UTC | 2026-08-31 11:46 UTC | 3.14 → 65.00 | the single budgeted 96-thread session, idle box |
| 32 | 1 + 3 | 2026-08-31 23:02 UTC | 2026-08-31 23:03 UTC | 1.41 → 10.16 | levjepa f32/f16/q8_0 encoder, idle box |
| 96 | 1 + 3 | 2026-08-31 23:05 UTC | 2026-08-31 23:05 UTC | 1.81 → 9.35 | levjepa 96-thread row, idle box |
| 32 | 1 + 3 | 2026-08-31 23:13 UTC | 2026-08-31 23:13 UTC | 0.82 → 7.80 | levjepa q4_0/q4_k encoder rows, idle box |

The 1-minute load average is recorded per session, before the first run and after the last (out of 192 hardware threads; a session's own run contributes its thread count, which is what the end-of-session figure mostly is). Every session here started on an **idle** box — the highest starting load average is 3.24 — so `ms mean` is a fair figure and `ms min` sits within a per cent or two of it.

**What the milliseconds are.** `ms` is the wall time of `ggml_backend_graph_compute` for the named graph (`jepa_context_last_compute_ms`) — model load, graph build/allocation, the host-side patchify and the output copy are excluded. Measured as `wall_ms_mean − ms_mean` over this sweep's own encoder runs at 32 threads they add 0.3–0.8 ms up to 1024 tokens and 3.4–98.9 ms above it (dominated by the patchify and the output copy, not by graph build). The JSONs keep the full API-call time as `wall_ms_mean`. `tokens/s` is `tokens / ms_mean`. `peak RSS` is the process `VmHWM` after the run, i.e. weights + the largest graph allocation, not a per-graph figure. For `lewm-rollout` the reported ms is **per rollout step** (the K graphs of one `jepa_lewm_rollout` call divided by K, so its rate column is steps/s).

## Encoder

| model | ftype | shape | tokens | threads | ms mean | ms min | tokens/s | PyTorch mean ms | PyTorch median ms | speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | f32 | 224x224 | 256 | 32 | 174.4 | 173.4 | 1468 | 263 | **250** | 1.43x |
| ijepa_vith14_1k | f16 | 224x224 | 256 | 32 | 147.0 | 145.9 | 1741 | 263 | **250** | 1.70x |
| ijepa_vith14_1k | f16 | 224x224 | 256 | 96 | 113.1 | 112.0 | 2264 | 263 | **250** | 2.21x |
| ijepa_vith14_1k | q8_0 | 224x224 | 256 | 32 | 129.1 | 128.5 | 1983 | 263 | **250** | 1.94x |
| ijepa_vith14_1k | q4_k | 224x224 | 256 | 32 | 197.8 | 197.5 | 1295 | 263 | **250** | 1.26x |
| ijepa_vith14_1k | q4_0 | 224x224 | 256 | 32 | 140.2 | 139.4 | 1826 | 263 | **250** | 1.78x |
| lejepa-vits16-pretrain-in1k | f32 | 224x224 | 197 | 32 | 13.1 | 13.1 | 15019 | 22.6 | **15.4** | 1.17x |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 197 | 32 | 12.8 | 12.7 | 15383 | 22.6 | **15.4** | 1.20x |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | 197 | 32 | 11.3 | 11.2 | 17379 | 22.6 | **15.4** | 1.36x |
| lejepa-vits16-pretrain-in1k | q4_k | 224x224 | 197 | 32 | 12.9 | 12.8 | 15301 | 22.6 | **15.4** | 1.20x |
| lejepa-vits16-pretrain-in1k | q4_0 | 224x224 | 197 | 32 | 12.0 | 11.9 | 16411 | 22.6 | **15.4** | 1.28x |
| levjepa-vitl16 | f32 | 16f 224x224 | 3 137 | 32 | 1519.5 | 1512.0 | 2065 | **1752** | 1735 | 1.15x |
| levjepa-vitl16 | f16 | 16f 224x224 | 3 137 | 32 | 1496.2 | 1479.9 | 2097 | **1752** | 1735 | 1.17x |
| levjepa-vitl16 | f16 | 16f 224x224 | 3 137 | 96 | 894.7 | 882.2 | 3506 | **1752** | 1735 | 1.96x |
| levjepa-vitl16 | q8_0 | 16f 224x224 | 3 137 | 32 | 1547.3 | 1508.4 | 2027 | **1752** | 1735 | 1.13x |
| levjepa-vitl16 | q4_k | 16f 224x224 | 3 137 | 32 | 1874.6 | 1851.4 | 1673 | **1752** | 1735 | 0.93x |
| levjepa-vitl16 | q4_0 | 16f 224x224 | 3 137 | 32 | 1551.0 | 1536.7 | 2023 | **1752** | 1735 | 1.13x |
| lewm-pusht | f32 | 224x224 | 257 | 32 | 9.2 | 9.2 | 28022 | **16.8**<sup>lewm</sup> | 16.0 | 1.83x |
| lewm-pusht | f16 | 224x224 | 257 | 32 | 9.8 | 9.7 | 26356 | **16.8**<sup>lewm</sup> | 16.0 | 1.72x |
| lewm-pusht | q8_0 | 224x224 | 257 | 32 | 9.1 | 8.9 | 28212 | **16.8**<sup>lewm</sup> | 16.0 | 1.84x |
| lewm-pusht | q4_k | 224x224 | 257 | 32 | 10.5 | 10.3 | 24561 | **16.8**<sup>lewm</sup> | 16.0 | 1.60x |
| lewm-pusht | q4_0 | 224x224 | 257 | 32 | 9.1 | 9.0 | 28243 | **16.8**<sup>lewm</sup> | 16.0 | 1.84x |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 2 048 | 32 | 943.5 | 938.6 | 2171 | **1051**<sup>ssv2</sup> | 1046 | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 32 | 822.8 | 820.7 | 2489 | **1051**<sup>ssv2</sup> | 1046 | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 96 | 564.4 | 560.8 | 3629 | **1051**<sup>ssv2</sup> | 1046 | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 2 048 | 32 | 793.3 | 789.7 | 2582 | **1051**<sup>ssv2</sup> | 1046 | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc16-256-ssv2 | q4_k | 16f 256x256 | 2 048 | 32 | 1092.6 | 1087.0 | 1874 | **1051**<sup>ssv2</sup> | 1046 | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc16-256-ssv2 | q4_0 | 16f 256x256 | 2 048 | 32 | 853.3 | 837.4 | 2400 | **1051**<sup>ssv2</sup> | 1046 | n/a<sup>ssv2</sup> |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 2 048 | 32 | 941.3 | 938.0 | 2176 | **1293**<sup>fpc64</sup> | 1296 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 32 | 820.7 | 817.1 | 2495 | **1293**<sup>fpc64</sup> | 1296 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 96 | 566.6 | 552.3 | 3614 | **1293**<sup>fpc64</sup> | 1296 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | 32 | 794.1 | 781.2 | 2579 | **1293**<sup>fpc64</sup> | 1296 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | q4_k | 16f 256x256 | 2 048 | 32 | 1096.1 | 1095.7 | 1868 | **1293**<sup>fpc64</sup> | 1296 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | q4_0 | 16f 256x256 | 2 048 | 32 | 844.4 | 835.2 | 2425 | **1293**<sup>fpc64</sup> | 1296 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f32 | 64f 256x256 | 8 192 | 32 | 7020.1 | 6973.4 | 1167 | **10114**<sup>fpc64</sup> | 9694 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 8 192 | 32 | 6388.1 | 6368.7 | 1282 | **10114**<sup>fpc64</sup> | 9694 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 8 192 | 96 | 4026.9 | 3992.3 | 2034 | **10114**<sup>fpc64</sup> | 9694 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | q8_0 | 64f 256x256 | 8 192 | 32 | 6482.4 | 6474.6 | 1264 | **10114**<sup>fpc64</sup> | 9694 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | q4_k | 64f 256x256 | 8 192 | 32 | 7405.5 | 7401.7 | 1106 | **10114**<sup>fpc64</sup> | 9694 | n/a<sup>fpc64</sup> |
| vjepa2-vitl-fpc64-256 | q4_0 | 64f 256x256 | 8 192 | 32 | 6616.3 | 6537.0 | 1238 | **10114**<sup>fpc64</sup> | 9694 | n/a<sup>fpc64</sup> |
| vjepa2_1-vitb-384 | f32 | 384x384 | 576 | 32 | 70.0 | 69.8 | 8228 | **110** | 109 | 1.57x |
| vjepa2_1-vitb-384 | f16 | 384x384 | 576 | 32 | 60.3 | 60.1 | 9547 | **110** | 109 | 1.82x |
| vjepa2_1-vitb-384 | f16 | 384x384 | 576 | 96 | 52.7 | 52.2 | 10935 | **110** | 109 | 2.09x |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | 576 | 32 | 58.5 | 57.9 | 9846 | **110** | 109 | 1.88x |
| vjepa2_1-vitb-384 | q4_k | 384x384 | 576 | 32 | 90.7 | 90.5 | 6351 | **110** | 109 | 1.21x |
| vjepa2_1-vitb-384 | q4_0 | 384x384 | 576 | 32 | 59.7 | 58.6 | 9654 | **110** | 109 | 1.84x |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 4 608 | 32 | 825.9 | 819.0 | 5579 | **908** | 918 | 1.10x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 32 | 853.5 | 849.8 | 5399 | **908** | 918 | 1.06x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 96 | 636.0 | 621.5 | 7246 | **908** | 918 | 1.43x |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 4 608 | 32 | 913.6 | 899.6 | 5044 | **908** | 918 | 0.99x |
| vjepa2_1-vitb-384 | q4_k | 16f 384x384 | 4 608 | 32 | 1148.5 | 1144.5 | 4012 | **908** | 918 | 0.79x |
| vjepa2_1-vitb-384 | q4_0 | 16f 384x384 | 4 608 | 32 | 895.2 | 893.3 | 5147 | **908** | 918 | 1.01x |
| vjepa2_1-vitb-384 | f32 | 64f 384x384 | 18 432 | 32 | 9050.4 | 8885.2 | 2037 | – | – | – |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 18 432 | 32 | 9036.1 | 8973.9 | 2040 | – | – | – |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 18 432 | 96 | 5040.2 | 5008.8 | 3657 | – | – | – |
| vjepa2_1-vitb-384 | q8_0 | 64f 384x384 | 18 432 | 32 | 9487.1 | 9425.4 | 1943 | – | – | – |
| vjepa2_1-vitb-384 | q4_k | 64f 384x384 | 18 432 | 32 | 10234.3 | 10212.9 | 1801 | – | – | – |
| vjepa2_1-vitb-384 | q4_0 | 64f 384x384 | 18 432 | 32 | 9526.9 | 9392.3 | 1935 | – | – | – |

PyTorch baseline = `timing_s.forward_s` over the reference samples with the same frame count in `tests/fixtures/ref/<model>/manifest.json` — the same box, CPU float32, torch 2.13.0+cpu, transformers 5.16.1, 32 threads. It is the model forward alone (no decode, no preprocessing). Two summaries are given: the **mean** over every such sample, and the **median of the samples after the first**, because a manifest's first forward of a frame group can be a cold one (weights paged in, kernels selected) that the mean then carries into every row. The bold column is the one the speedup divides by — the drop-first median wherever that first sample is ≥ 1.2x the median of the rest, the mean otherwise. The speedup column is filled only where the reference forward is the same work as our encoder; see the footnotes for the three models where it is not.

<details><summary>Which manifest samples feed each baseline</summary>

| model | manifest | frames | samples (manifest order) | first sample ms | mean ms | median ms (drop-first) | used for the speedup |
|---|---|---:|---|---:|---:|---:|---|
| ijepa_vith14_1k | `ijepa-vith14-1k` | 1 | coco_000000000139, … (8 samples) | 331.8 | 263 | 250 | median (cold first sample) |
| lejepa-vits16-pretrain-in1k | `lejepa-vits16` | 1 | coco_000000000139, … (8 samples) | 72.2 | 22.6 | 15.4 | median (cold first sample) |
| levjepa-vitl16 | `levjepa-vitl16` | 16 | archery_f16, bowling_f16, coco_000000000139, coco_000000000285 | 1682.3 | 1752 | 1735 | mean |
| lewm-pusht | `lewm-pusht` | 1 | coco_000000000139, coco_000000000285 | 17.5 | 16.8 | 16.0 | mean |
| lewm-pusht | `lewm-pusht` | 3 | seq | 20.9 | 20.9 | – | mean |
| vjepa2-vitl-fpc16-256-ssv2 | `vjepa2-vitl-fpc16-256-ssv2` | 16 | archery_f16, bowling_f16 | 1056.0 | 1051 | 1046 | mean |
| vjepa2-vitl-fpc64-256 | `vjepa2-vitl-fpc64-256` | 16 | archery_f16, bowling_f16 | 1289.6 | 1293 | 1296 | mean |
| vjepa2-vitl-fpc64-256 | `vjepa2-vitl-fpc64-256` | 64 | archery_f64, bowling_f64 | 10533.4 | 10114 | 9694 | mean |
| vjepa2_1-vitb-384 | `vjepa2_1-vitb-384` | 1 | coco_000000000139, coco_000000000285 | 111.2 | 110 | 109 | mean |
| vjepa2_1-vitb-384 | `vjepa2_1-vitb-384` | 16 | archery_f16, bowling_f16 | 897.4 | 908 | 918 | mean |

A frame group is matched to an encoder row by frame count, so a sample that is not a plain encoder forward never reaches one: `lewm-pusht`'s 3-frame `seq` sample times encode + projector + a 3-frame predictor call and forms its own group, and only the two 1-frame samples feed the 224x224 LeWM rows above. Groups of one sample have no drop-first median and always use the mean.

</details>

### Encoder throughput summary (tokens/s)

| model | shape | tokens | f32 t=32 | f16 t=32 | q8_0 t=32 | q4_k t=32 | q4_0 t=32 | f16 t=96 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | 224x224 | 256 | 1468 | 1741 | 1983 | 1295 | 1826 | 2264 |
| lejepa-vits16-pretrain-in1k | 224x224 | 197 | 15019 | 15383 | 17379 | 15301 | 16411 | – |
| levjepa-vitl16 | 16f 224x224 | 3 137 | 2065 | 2097 | 2027 | 1673 | 2023 | 3506 |
| lewm-pusht | 224x224 | 257 | 28022 | 26356 | 28212 | 24561 | 28243 | – |
| vjepa2-vitl-fpc16-256-ssv2 | 16f 256x256 | 2 048 | 2171 | 2489 | 2582 | 1874 | 2400 | 3629 |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | 2 048 | 2176 | 2495 | 2579 | 1868 | 2425 | 3614 |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | 8 192 | 1167 | 1282 | 1264 | 1106 | 1238 | 2034 |
| vjepa2_1-vitb-384 | 384x384 | 576 | 8228 | 9547 | 9846 | 6351 | 9654 | 10935 |
| vjepa2_1-vitb-384 | 16f 384x384 | 4 608 | 5579 | 5399 | 5044 | 4012 | 5147 | 7246 |
| vjepa2_1-vitb-384 | 64f 384x384 | 18 432 | 2037 | 2040 | 1943 | 1801 | 1935 | 3657 |

### Effect of the weight dtype (encoder, t=32)

| model | shape | tokens | f32 ms | f16 ms | q8_0 ms | f32 → f16 | f16 → q8_0 |
|---|---|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | 224x224 | 256 | 174 | 147 | 129 | 1.19x | 1.14x |
| lejepa-vits16-pretrain-in1k | 224x224 | 197 | 13.1 | 12.8 | 11.3 | 1.02x | 1.13x |
| levjepa-vitl16 | 16f 224x224 | 3 137 | 1519 | 1496 | 1547 | 1.02x | 0.97x |
| lewm-pusht | 224x224 | 257 | 9.2 | 9.8 | 9.1 | 0.94x | 1.07x |
| vjepa2-vitl-fpc16-256-ssv2 | 16f 256x256 | 2 048 | 943 | 823 | 793 | 1.15x | 1.04x |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | 2 048 | 941 | 821 | 794 | 1.15x | 1.03x |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | 8 192 | 7020 | 6388 | 6482 | 1.10x | 0.99x |
| vjepa2_1-vitb-384 | 384x384 | 576 | 70.0 | 60.3 | 58.5 | 1.16x | 1.03x |
| vjepa2_1-vitb-384 | 16f 384x384 | 4 608 | 826 | 853 | 914 | 0.97x | 0.93x |
| vjepa2_1-vitb-384 | 64f 384x384 | 18 432 | 9050 | 9036 | 9487 | 1.00x | 0.95x |

Quantisation buys memory, not reliably time. The matmuls do get faster (`docs/ggml-notes.md` §5 measures 2.2 → 3.3 TFLOP/s going F32 → F16, and ~3.6-4.1 for Q8_0 on this box), but q8_0 pays for quantising the *activations* on every matmul, and on the long clips the flash-attention time — F32 work whatever the weights are — dominates the layer (`docs/ggml-notes.md` §3: 158 ms per ViT-L layer at 8192 tokens against ~62 ms of matmul). The small image models are launch- and LayerNorm-bound and barely move at all. Pick the dtype on the accuracy tables in `docs/parity.md` and `docs/quantization.md` and on the memory table below, not on these milliseconds.

### Sub-8-bit weights: what q4 costs and what it buys (encoder, t=32)

| model | shape | tokens | f16 MiB | q4 MiB | of f16 | f16 ms | q4_0 ms | q4_k ms | q4_0 vs f16 | q4_k vs q4_0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | 224x224 | 256 | 1206 | 344 | 28 % | 147 | 140 | 198 | 1.05x | 0.71x |
| lejepa-vits16-pretrain-in1k | 224x224 | 197 | 42 | 13 | 31 % | 12.8 | 12.0 | 12.9 | 1.07x | 0.93x |
| levjepa-vitl16 | 16f 224x224 | 3 137 | 579 | 166 | 29 % | 1496 | 1551 | 1875 | 0.96x | 0.83x |
| lewm-pusht | 224x224 | 257 | 38 | 15 | 41 % | 9.8 | 9.1 | 10.5 | 1.07x | 0.87x |
| vjepa2-vitl-fpc16-256-ssv2 | 16f 256x256 | 2 048 | 717 | 205 | 29 % | 823 | 853 | 1093 | 0.96x | 0.78x |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | 2 048 | 622 | 178 | 29 % | 821 | 844 | 1096 | 0.97x | 0.77x |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | 8 192 | 622 | 178 | 29 % | 6388 | 6616 | 7405 | 0.97x | 0.89x |
| vjepa2_1-vitb-384 | 384x384 | 576 | 210 | 62 | 30 % | 60.3 | 59.7 | 90.7 | 1.01x | 0.66x |
| vjepa2_1-vitb-384 | 16f 384x384 | 4 608 | 210 | 62 | 30 % | 853 | 895 | 1148 | 0.95x | 0.78x |
| vjepa2_1-vitb-384 | 64f 384x384 | 18 432 | 210 | 62 | 30 % | 9036 | 9527 | 10234 | 0.95x | 0.93x |

q4 is a **memory** win and, at best, time-neutral: the weights fall to 28–41 % of f16, `q4_0` lands within 7 % of f16's time either way, and `q4_k` costs another 1.07–1.52x on top of q4_0 — ggml's Q4_K vec-dot does more work per block than Q4_0's, and nothing in these graphs is weight-bandwidth-bound enough to pay that back. Where a model's rows are not a multiple of the 256-element super-block the k-quant falls back to q4_0 tensor by tensor, so the small models' q4_k files are mostly q4_0 and their two rows nearly coincide. **Neither is a parity configuration**: `docs/parity.md` puts every file below 8 bits per weight in the advisory tier, and `docs/quantization.md` has the per-model cosines. These rows are here so the size/speed side of that trade-off is measured rather than assumed.

### Thread scaling (32 → 96 threads)

| model | ftype | mode | shape | tokens | ms t=32 | ms t=96 | speedup |
|---|---|---|---|---:|---:|---:|---:|
| ijepa_vith14_1k | f16 | encoder | 224x224 | 256 | 147 | 113 | 1.30x |
| levjepa-vitl16 | f16 | encoder | 16f 224x224 | 3 137 | 1496 | 895 | 1.67x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | encoder | 16f 256x256 | 2 048 | 823 | 564 | 1.46x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | head | 16f 256x256 | 2 048 | 99.0 | 67.1 | 1.48x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | predictor | 16f 256x256 | 2 048 | 341 | 208 | 1.64x |
| vjepa2-vitl-fpc64-256 | f16 | encoder | 16f 256x256 | 2 048 | 821 | 567 | 1.45x |
| vjepa2-vitl-fpc64-256 | f16 | encoder | 64f 256x256 | 8 192 | 6388 | 4027 | 1.59x |
| vjepa2-vitl-fpc64-256 | f16 | predictor | 16f 256x256 | 2 048 | 344 | 207 | 1.66x |
| vjepa2_1-vitb-384 | f16 | encoder | 384x384 | 576 | 60.3 | 52.7 | 1.15x |
| vjepa2_1-vitb-384 | f16 | encoder | 16f 384x384 | 4 608 | 853 | 636 | 1.34x |
| vjepa2_1-vitb-384 | f16 | encoder | 64f 384x384 | 18 432 | 9036 | 5040 | 1.79x |
| vjepa2_1-vitb-384 | f16 | predictor | 16f 384x384 | 4 608 | 1304 | 854 | 1.53x |

Tripling the threads never triples the throughput: the 32-thread runs already saturate a good part of the memory bandwidth, and both the LayerNorm/GELU passes and the graph launch overhead scale poorly. The gain is largest where a single matmul or flash-attention tile is big enough to keep 96 workers busy.

### Cross-check against `docs/parity.md`

`docs/parity.md` times the *same* encoder graphs on the real preprocessed fixture inputs, reporting the second sample of a run (after the weights are paged in) — effectively a best-of figure, so `ms min` is what it should be compared against. The synthetic input used here has the same shape and scale, so what is left is run-to-run and input-dependent noise: every row below agrees to within ±12 %, and the two rows where `docs/parity.md` was itself re-measured on this idle box (the f32 fpc64 clips) to within 3.3 %. The right-hand column is **parsed** out of `docs/parity.md` (its `ms/item t=N` / `ms/clip t=N` columns), so the two documents cannot drift apart without this table saying so.

| model | ftype | shape | threads | bench ms min | parity.md ms | delta |
|---|---|---|---:|---:|---:|---:|
| ijepa_vith14_1k | f32 | 224x224 | 32 | 173 | 185 | -6.3 % |
| ijepa_vith14_1k | f16 | 224x224 | 32 | 146 | 156 | -6.5 % |
| ijepa_vith14_1k | f16 | 224x224 | 96 | 112 | 122 | -8.2 % |
| ijepa_vith14_1k | q8_0 | 224x224 | 32 | 129 | 138 | -6.9 % |
| lejepa-vits16-pretrain-in1k | f32 | 224x224 | 32 | 13.1 | 14.0 | -6.6 % |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 32 | 12.7 | 13.4 | -5.2 % |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | 32 | 11.2 | 12.6 | -11.0 % |
| levjepa-vitl16 | f32 | 16f 224x224 | 32 | 1512 | 1661 | -9.0 % |
| levjepa-vitl16 | f16 | 16f 224x224 | 32 | 1480 | 1542 | -4.0 % |
| levjepa-vitl16 | f16 | 16f 224x224 | 96 | 882 | 881 | +0.1 % |
| levjepa-vitl16 | q8_0 | 16f 224x224 | 32 | 1508 | 1609 | -6.3 % |
| levjepa-vitl16 | q4_k | 16f 224x224 | 32 | 1851 | 1935 | -4.3 % |
| levjepa-vitl16 | q4_0 | 16f 224x224 | 32 | 1537 | 1610 | -4.6 % |
| lewm-pusht | f32 | 224x224 | 32 | 9.2 | 10.3 | -11.1 % |
| lewm-pusht | f16 | 224x224 | 32 | 9.7 | 10.7 | -9.6 % |
| lewm-pusht | q8_0 | 224x224 | 32 | 8.9 | 9.7 | -8.7 % |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 32 | 939 | 924 | +1.6 % |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 32 | 821 | 814 | +0.8 % |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 32 | 790 | 771 | +2.4 % |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 32 | 938 | 970 | -3.3 % |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 32 | 817 | 827 | -1.2 % |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 32 | 781 | 852 | -8.3 % |
| vjepa2-vitl-fpc64-256 | f32 | 64f 256x256 | 32 | 6973 | 6845 | +1.9 % |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 32 | 6369 | 6386 | -0.3 % |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 96 | 3992 | 4076 | -2.1 % |
| vjepa2-vitl-fpc64-256 | q8_0 | 64f 256x256 | 32 | 6475 | 7203 | -10.1 % |
| vjepa2_1-vitb-384 | f32 | 384x384 | 32 | 69.8 | 70.0 | -0.2 % |
| vjepa2_1-vitb-384 | f16 | 384x384 | 32 | 60.1 | 63.0 | -4.6 % |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | 32 | 57.9 | 59.0 | -1.8 % |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 32 | 819 | 909 | -9.9 % |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 32 | 850 | 875 | -2.9 % |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 32 | 900 | 878 | +2.5 % |

## Memory

| model | ftype | shape | weights MiB | peak RSS MiB | model load ms |
|---|---|---|---:|---:|---:|
| ijepa_vith14_1k | f32 | 224x224 | 2406 | 2432 | 1150 |
| ijepa_vith14_1k | f16 | 224x224 | 1206 | 1230 | 513 |
| ijepa_vith14_1k | q8_0 | 224x224 | 644 | 660 | 279 |
| ijepa_vith14_1k | q4_k | 224x224 | 344 | 360 | 160 |
| ijepa_vith14_1k | q4_0 | 224x224 | 344 | 360 | 157 |
| lejepa-vits16-pretrain-in1k | f32 | 224x224 | 83 | 94 | 49 |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 42 | 52 | 27 |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | 23 | 33 | 22 |
| lejepa-vits16-pretrain-in1k | q4_k | 224x224 | 13 | 23 | 10 |
| lejepa-vits16-pretrain-in1k | q4_0 | 224x224 | 13 | 23 | 10 |
| levjepa-vitl16 | f32 | 16f 224x224 | 1156 | 1335 | 547 |
| levjepa-vitl16 | f16 | 16f 224x224 | 579 | 779 | 276 |
| levjepa-vitl16 | q8_0 | 16f 224x224 | 310 | 490 | 147 |
| levjepa-vitl16 | q4_k | 16f 224x224 | 166 | 348 | 96 |
| levjepa-vitl16 | q4_0 | 16f 224x224 | 166 | 354 | 86 |
| lewm-pusht | f32 | 224x224 | 69 | 79 | 37 |
| lewm-pusht | f16 | 224x224 | 38 | 47 | 23 |
| lewm-pusht | q8_0 | 224x224 | 23 | 33 | 14 |
| lewm-pusht | q4_k | 224x224 | 15 | 25 | 11 |
| lewm-pusht | q4_0 | 224x224 | 15 | 25 | 12 |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 1432 | 1522 | 635 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 717 | 808 | 332 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 383 | 466 | 184 |
| vjepa2-vitl-fpc16-256-ssv2 | q4_k | 16f 256x256 | 205 | 297 | 111 |
| vjepa2-vitl-fpc16-256-ssv2 | q4_0 | 16f 256x256 | 205 | 291 | 106 |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 1243 | 1336 | 581 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 622 | 720 | 292 |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 333 | 426 | 158 |
| vjepa2-vitl-fpc64-256 | q4_k | 16f 256x256 | 178 | 272 | 90 |
| vjepa2-vitl-fpc64-256 | q4_0 | 16f 256x256 | 178 | 265 | 97 |
| vjepa2-vitl-fpc64-256 | f32 | 64f 256x256 | 1243 | 1599 | 567 |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 622 | 1034 | 295 |
| vjepa2-vitl-fpc64-256 | q8_0 | 64f 256x256 | 333 | 713 | 179 |
| vjepa2-vitl-fpc64-256 | q4_k | 64f 256x256 | 178 | 564 | 95 |
| vjepa2-vitl-fpc64-256 | q4_0 | 64f 256x256 | 178 | 561 | 106 |
| vjepa2_1-vitb-384 | f32 | 384x384 | 419 | 435 | 197 |
| vjepa2_1-vitb-384 | f16 | 384x384 | 210 | 226 | 97 |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | 113 | 127 | 60 |
| vjepa2_1-vitb-384 | q4_k | 384x384 | 62 | 75 | 36 |
| vjepa2_1-vitb-384 | q4_0 | 384x384 | 62 | 75 | 35 |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 419 | 569 | 246 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 210 | 376 | 111 |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 113 | 270 | 72 |
| vjepa2_1-vitb-384 | q4_k | 16f 384x384 | 62 | 218 | 49 |
| vjepa2_1-vitb-384 | q4_0 | 16f 384x384 | 62 | 220 | 35 |
| vjepa2_1-vitb-384 | f32 | 64f 384x384 | 419 | 1060 | 198 |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 210 | 948 | 105 |
| vjepa2_1-vitb-384 | q8_0 | 64f 384x384 | 113 | 809 | 62 |
| vjepa2_1-vitb-384 | q4_k | 64f 384x384 | 62 | 759 | 39 |
| vjepa2_1-vitb-384 | q4_0 | 64f 384x384 | 62 | 764 | 37 |

`weights MiB` is `jepa_model_n_bytes()` (the tensor bytes resident after the load, i.e. the GGUF payload); `peak RSS` additionally covers the graph allocation, the host-side patch buffer and the output rows, so it grows with the token count — the same weights are listed once per shape so that growth is visible. The loader `fread`s every tensor into its own buffer (no `mmap`), so `model load ms` tracks the file size and the page-cache state; these numbers are all warm-cache. Compare the weight column with the GGUF file sizes in `docs/quantization.md`.

## Attentive-pool head

| model | ftype | shape | tokens | threads | ms mean | ms min | encoder ms | peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 2 048 | 32 | 106.9 | 106.8 | 935.5 | 1512 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 32 | 99.0 | 98.9 | 815.8 | 815 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 96 | 67.1 | 67.0 | 573.5 | 773 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 2 048 | 32 | 98.2 | 98.2 | 793.3 | 467 |

The classifier head (3 self-attention blocks over the tokens + one cross-attention query + MLP + linear) on top of the encoder output of the same clip, so the end-to-end classification cost is encoder + head. `encoder ms` is the pass that produced this row's input — the **faster of two warm** encoder graphs (a third, cold one runs first and is discarded), not an average of `repeat` runs like the `ms` columns, so read the Encoder table for the encoder cost proper.

End-to-end classification against the reference — this *is* like-for-like, because the manifest's forward is `VJEPA2ForVideoClassification` (encoder + attentive pooler + classifier, predictor skipped):

| model | ftype | shape | threads | encoder ms | head ms | total ms | PyTorch ms | speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 32 | 943 | 107 | 1050 | 1051 | 1.00x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 32 | 823 | 99.0 | 922 | 1051 | 1.14x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 96 | 564 | 67.1 | 631 | 1051 | 1.66x |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 32 | 793 | 98.2 | 891 | 1051 | 1.18x |

## Masked predictor

| model | ftype | shape | tokens | threads | ms mean | ms min | encoder ms | peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 2 048 | 32 | 333.1 | 328.9 | 937.4 | 1531 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 32 | 340.8 | 340.5 | 814.6 | 822 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 96 | 208.0 | 196.5 | 593.4 | 773 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 2 048 | 32 | 336.8 | 335.7 | 810.6 | 472 |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 2 048 | 32 | 338.3 | 330.1 | 942.9 | 1344 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 32 | 343.8 | 341.0 | 827.6 | 724 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 96 | 207.4 | 197.8 | 557.0 | 681 |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | 32 | 332.7 | 332.3 | 796.8 | 416 |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 4 608 | 32 | 1296.7 | 1286.0 | 824.1 | 593 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 32 | 1303.6 | 1291.2 | 871.2 | 410 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 96 | 854.4 | 813.6 | 630.3 | 347 |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 4 608 | 32 | 1272.6 | 1269.9 | 901.1 | 295 |

Worst case for the predictor: context = target = **every** token, i.e. a sequence of 2 x tokens through the 12-layer 384-d predictor. `encoder ms` is the pass that produced this row's input — the **faster of two warm** encoder graphs (a third, cold one runs first and is discarded), not an average of `repeat` runs like the `ms` columns, so read the Encoder table for the encoder cost proper.

## LeWM world model

| model | ftype | mode | shape | threads | ms mean | ms min | steps/s |
|---|---|---|---|---:|---:|---:|---:|
| lewm-pusht | f32 | lewm-step | 3f x 192d | 32 | 0.911 | 0.863 | 1097 |
| lewm-pusht | f16 | lewm-step | 3f x 192d | 32 | 0.918 | 0.904 | 1089 |
| lewm-pusht | q8_0 | lewm-step | 3f x 192d | 32 | 0.739 | 0.718 | 1354 |
| lewm-pusht | f32 | lewm-rollout | rollout K=20 | 32 | 0.817 | 0.809 | 1225 |
| lewm-pusht | f16 | lewm-rollout | rollout K=20 | 32 | 0.865 | 0.850 | 1156 |
| lewm-pusht | q8_0 | lewm-rollout | rollout K=20 | 32 | 0.744 | 0.729 | 1344 |

`lewm-step` is one `jepa_lewm_predict` over the predictor's full 3-frame window; `lewm-rollout` is `jepa_lewm_rollout` and its ms is **per step** (the growing window means the first steps are cheaper than the last). Neither includes the encoder or the projector — see the encoder table for `lewm-pusht` for the cost of turning an image into a world-model state.

## Footnotes

<sup>fpc64</sup> the `vjepa2-vitl-fpc64-256` manifest times one `VJEPA2Model` forward, which always runs the **predictor** as well (its `predictor_last_hidden_state` comes from the same call), so it is an upper bound on the encoder and no speedup is claimed against it.

<sup>ssv2</sup> the SSv2 manifest times `VJEPA2ForVideoClassification`, i.e. encoder + attentive pooler + classifier with the predictor skipped. It is therefore not comparable with the encoder row alone; the end-to-end table under *Attentive-pool head* adds our encoder and head and makes the comparison there.

<sup>lewm</sup> the LeWM manifest times encode + projector + one 1-frame predictor call; the two extra graphs are ~1 ms of it (see the world-model table), so the speedup is a slight over-estimate.

---

Generated by `scripts/gen_benchmarks_md.py` from 80 runs in `tmp/bench`. Cross-check against `docs/parity.md` (same graphs, real fixture inputs) and `docs/quantization.md` (accuracy per dtype).
