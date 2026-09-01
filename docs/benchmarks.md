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
| 32 | 1 + 3 | 2026-09-01 19:11 UTC | 2026-09-01 19:26 UTC | 0.23 → 31.45 | — |

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
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 197 | 32 | 12.8 | 12.7 | 15382 | 22.6 | **15.4** | 1.20x |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | 197 | 32 | 11.3 | 11.2 | 17380 | 22.6 | **15.4** | 1.36x |
| lejepa-vits16-pretrain-in1k | q4_k | 224x224 | 197 | 32 | 12.9 | 12.8 | 15301 | 22.6 | **15.4** | 1.20x |
| lejepa-vits16-pretrain-in1k | q4_0 | 224x224 | 197 | 32 | 12.0 | 11.9 | 16411 | 22.6 | **15.4** | 1.28x |
| levjepa-vitl16 | f32 | 16f 224x224 | 3 137 | 32 | 1519.5 | 1512.0 | 2065 | **1752** | 1735 | 1.15x |
| levjepa-vitl16 | f16 | 16f 224x224 | 3 137 | 32 | 1496.2 | 1479.9 | 2097 | **1752** | 1735 | 1.17x |
| levjepa-vitl16 | f16 | 16f 224x224 | 3 137 | 96 | 894.7 | 882.2 | 3506 | **1752** | 1735 | 1.96x |
| levjepa-vitl16 | q8_0 | 16f 224x224 | 3 137 | 32 | 1547.3 | 1508.4 | 2027 | **1752** | 1735 | 1.13x |
| levjepa-vitl16 | q4_k | 16f 224x224 | 3 137 | 32 | 1874.6 | 1851.4 | 1673 | **1752** | 1735 | 0.93x |
| levjepa-vitl16 | q4_0 | 16f 224x224 | 3 137 | 32 | 1551.0 | 1536.7 | 2023 | **1752** | 1735 | 1.13x |
| lewm-pusht | f32 | 224x224 | 257 | 32 | 9.2 | 9.2 | 28023 | **16.8**<sup>lewm</sup> | 16.0 | 1.83x |
| lewm-pusht | f16 | 224x224 | 257 | 32 | 9.8 | 9.7 | 26356 | **16.8**<sup>lewm</sup> | 16.0 | 1.72x |
| lewm-pusht | q8_0 | 224x224 | 257 | 32 | 9.1 | 8.9 | 28211 | **16.8**<sup>lewm</sup> | 16.0 | 1.84x |
| lewm-pusht | q4_k | 224x224 | 257 | 32 | 10.5 | 10.3 | 24560 | **16.8**<sup>lewm</sup> | 16.0 | 1.60x |
| lewm-pusht | q4_0 | 224x224 | 257 | 32 | 9.1 | 9.0 | 28242 | **16.8**<sup>lewm</sup> | 16.0 | 1.84x |
| vjepa2-ac-vitg | f32 | 16f 256x256 | 2 048 | 32 | 2683.3 | 2675.9 | 763 | – | – | – |
| vjepa2-ac-vitg | f16 | 16f 256x256 | 2 048 | 32 | 2227.7 | 2219.0 | 919 | – | – | – |
| vjepa2-ac-vitg | q8_0 | 16f 256x256 | 2 048 | 32 | 2320.7 | 2309.6 | 882 | – | – | – |
| vjepa2-ac-vitg | f32 | 64f 256x256 | 8 192 | 32 | 15773.9 | 15619.6 | 519 | – | – | – |
| vjepa2-ac-vitg | f16 | 64f 256x256 | 8 192 | 32 | 15747.2 | 15538.5 | 520 | – | – | – |
| vjepa2-ac-vitg | q8_0 | 64f 256x256 | 8 192 | 32 | 16618.2 | 16577.4 | 493 | – | – | – |
| vjepa2-vitg-fpc64-256 | f32 | 16f 256x256 | 2 048 | 32 | 2696.6 | 2685.1 | 759 | – | – | – |
| vjepa2-vitg-fpc64-256 | f16 | 16f 256x256 | 2 048 | 32 | 2384.1 | 2379.3 | 859 | – | – | – |
| vjepa2-vitg-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | 32 | 2293.1 | 2287.2 | 893 | – | – | – |
| vjepa2-vitg-fpc64-256 | f32 | 64f 256x256 | 8 192 | 32 | 15876.7 | 15730.8 | 516 | – | – | – |
| vjepa2-vitg-fpc64-256 | f16 | 64f 256x256 | 8 192 | 32 | 16285.6 | 16236.7 | 503 | – | – | – |
| vjepa2-vitg-fpc64-256 | q8_0 | 64f 256x256 | 8 192 | 32 | 16622.0 | 16557.8 | 493 | – | – | – |
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
| lejepa-vits16-pretrain-in1k | 224x224 | 197 | 15019 | 15382 | 17380 | 15301 | 16411 | – |
| levjepa-vitl16 | 16f 224x224 | 3 137 | 2065 | 2097 | 2027 | 1673 | 2023 | 3506 |
| lewm-pusht | 224x224 | 257 | 28023 | 26356 | 28211 | 24560 | 28242 | – |
| vjepa2-ac-vitg | 16f 256x256 | 2 048 | 763 | 919 | 882 | – | – | – |
| vjepa2-ac-vitg | 64f 256x256 | 8 192 | 519 | 520 | 493 | – | – | – |
| vjepa2-vitg-fpc64-256 | 16f 256x256 | 2 048 | 759 | 859 | 893 | – | – | – |
| vjepa2-vitg-fpc64-256 | 64f 256x256 | 8 192 | 516 | 503 | 493 | – | – | – |
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
| vjepa2-ac-vitg | 16f 256x256 | 2 048 | 2683 | 2228 | 2321 | 1.20x | 0.96x |
| vjepa2-ac-vitg | 64f 256x256 | 8 192 | 15774 | 15747 | 16618 | 1.00x | 0.95x |
| vjepa2-vitg-fpc64-256 | 16f 256x256 | 2 048 | 2697 | 2384 | 2293 | 1.13x | 1.04x |
| vjepa2-vitg-fpc64-256 | 64f 256x256 | 8 192 | 15877 | 16286 | 16622 | 0.97x | 0.98x |
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
| lewm-pusht | q4_k | 224x224 | 15 | 25 | 12 |
| lewm-pusht | q4_0 | 224x224 | 15 | 25 | 12 |
| vjepa2-ac-vitg | f32 | 16f 256x256 | 5025 | 5131 | 2489 |
| vjepa2-ac-vitg | f16 | 16f 256x256 | 2519 | 2644 | 1007 |
| vjepa2-ac-vitg | q8_0 | 16f 256x256 | 1344 | 1460 | 570 |
| vjepa2-ac-vitg | f32 | 64f 256x256 | 5025 | 5511 | 2521 |
| vjepa2-ac-vitg | f16 | 64f 256x256 | 2519 | 3092 | 1026 |
| vjepa2-ac-vitg | q8_0 | 64f 256x256 | 1344 | 1876 | 576 |
| vjepa2-vitg-fpc64-256 | f32 | 16f 256x256 | 3947 | 4053 | 1915 |
| vjepa2-vitg-fpc64-256 | f16 | 16f 256x256 | 1975 | 2108 | 810 |
| vjepa2-vitg-fpc64-256 | q8_0 | 16f 256x256 | 1057 | 1178 | 449 |
| vjepa2-vitg-fpc64-256 | f32 | 64f 256x256 | 3947 | 4435 | 1907 |
| vjepa2-vitg-fpc64-256 | f16 | 64f 256x256 | 1975 | 2546 | 819 |
| vjepa2-vitg-fpc64-256 | q8_0 | 64f 256x256 | 1057 | 1580 | 463 |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 1432 | 1522 | 635 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 717 | 808 | 332 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 383 | 466 | 184 |
| vjepa2-vitl-fpc16-256-ssv2 | q4_k | 16f 256x256 | 205 | 297 | 110 |
| vjepa2-vitl-fpc16-256-ssv2 | q4_0 | 16f 256x256 | 205 | 291 | 106 |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 1244 | 1336 | 581 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 622 | 720 | 292 |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 333 | 426 | 158 |
| vjepa2-vitl-fpc64-256 | q4_k | 16f 256x256 | 178 | 272 | 90 |
| vjepa2-vitl-fpc64-256 | q4_0 | 16f 256x256 | 178 | 265 | 97 |
| vjepa2-vitl-fpc64-256 | f32 | 64f 256x256 | 1244 | 1599 | 568 |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 622 | 1034 | 294 |
| vjepa2-vitl-fpc64-256 | q8_0 | 64f 256x256 | 333 | 713 | 179 |
| vjepa2-vitl-fpc64-256 | q4_k | 64f 256x256 | 178 | 564 | 95 |
| vjepa2-vitl-fpc64-256 | q4_0 | 64f 256x256 | 178 | 561 | 106 |
| vjepa2_1-vitb-384 | f32 | 384x384 | 418 | 435 | 197 |
| vjepa2_1-vitb-384 | f16 | 384x384 | 210 | 226 | 97 |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | 113 | 127 | 60 |
| vjepa2_1-vitb-384 | q4_k | 384x384 | 62 | 75 | 36 |
| vjepa2_1-vitb-384 | q4_0 | 384x384 | 62 | 75 | 35 |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 418 | 569 | 246 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 210 | 376 | 111 |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 113 | 270 | 72 |
| vjepa2_1-vitb-384 | q4_k | 16f 384x384 | 62 | 218 | 49 |
| vjepa2_1-vitb-384 | q4_0 | 16f 384x384 | 62 | 220 | 35 |
| vjepa2_1-vitb-384 | f32 | 64f 384x384 | 418 | 1060 | 198 |
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
| vjepa2-vitg-fpc64-256 | f32 | 16f 256x256 | 2 048 | 32 | 362.3 | 358.9 | 2728.9 | 4059 |
| vjepa2-vitg-fpc64-256 | f16 | 16f 256x256 | 2 048 | 32 | 354.7 | 354.3 | 2479.0 | 2119 |
| vjepa2-vitg-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | 32 | 349.0 | 348.3 | 2313.3 | 1182 |
| vjepa2-vitl-fpc16-256-ssv2 | f32 | 16f 256x256 | 2 048 | 32 | 333.1 | 328.9 | 937.4 | 1531 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 32 | 340.8 | 340.5 | 814.6 | 822 |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | 16f 256x256 | 2 048 | 96 | 208.0 | 196.4 | 593.4 | 773 |
| vjepa2-vitl-fpc16-256-ssv2 | q8_0 | 16f 256x256 | 2 048 | 32 | 336.8 | 335.7 | 810.6 | 472 |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 2 048 | 32 | 338.3 | 330.1 | 942.9 | 1344 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 32 | 343.8 | 341.0 | 827.6 | 724 |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 96 | 207.4 | 197.8 | 557.0 | 681 |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | 32 | 332.7 | 332.3 | 796.8 | 416 |
| vjepa2_1-vitb-384 | f32 | 16f 384x384 | 4 608 | 32 | 1296.7 | 1286.0 | 824.1 | 594 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 32 | 1303.6 | 1291.2 | 871.2 | 410 |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 96 | 854.4 | 813.6 | 630.3 | 347 |
| vjepa2_1-vitb-384 | q8_0 | 16f 384x384 | 4 608 | 32 | 1272.6 | 1269.9 | 901.1 | 295 |

Worst case for the predictor: context = target = **every** token, i.e. a sequence of 2 x tokens through the 12-layer 384-d predictor. `encoder ms` is the pass that produced this row's input — the **faster of two warm** encoder graphs (a third, cold one runs first and is discarded), not an average of `repeat` runs like the `ms` columns, so read the Encoder table for the encoder cost proper.

## V-JEPA 2-AC world model and planner

| model | ftype | mode | shape | threads | ms mean | ms min | ms / candidate | peak RSS MiB |
|---|---|---|---|---:|---:|---:|---:|---:|
| vjepa2-ac-vitg | f32 | ac | 1f x 256tok, K=1 | 32 | 106.8 | 106.6 | 106.79 | 5061 |
| vjepa2-ac-vitg | f16 | ac | 1f x 256tok, K=1 | 32 | 94.9 | 94.4 | 94.87 | 2543 |
| vjepa2-ac-vitg | q8_0 | ac | 1f x 256tok, K=1 | 32 | 83.6 | 83.4 | 83.59 | 1368 |
| vjepa2-ac-vitg | f32 | ac | 1f x 256tok, K=16 | 32 | 1476.1 | 1472.2 | 92.25 | 5188 |
| vjepa2-ac-vitg | f16 | ac | 1f x 256tok, K=16 | 32 | 1337.1 | 1334.3 | 83.57 | 2710 |
| vjepa2-ac-vitg | q8_0 | ac | 1f x 256tok, K=16 | 32 | 1229.8 | 1216.8 | 76.86 | 1515 |
| vjepa2-ac-vitg | f32 | ac | 1f x 256tok, K=64 | 32 | 6544.7 | 6538.3 | 102.26 | 5722 |
| vjepa2-ac-vitg | f16 | ac | 1f x 256tok, K=64 | 32 | 5411.2 | 5377.1 | 84.55 | 3338 |
| vjepa2-ac-vitg | q8_0 | ac | 1f x 256tok, K=64 | 32 | 4936.3 | 4920.9 | 77.13 | 2093 |
| vjepa2-ac-vitg | f32 | ac-rollout | rollout H=2, K=1 | 32 | 170.1 | 169.5 | 170.13 | 5071 |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=2, K=1 | 32 | 138.9 | 137.5 | 138.88 | 2547 |
| vjepa2-ac-vitg | q8_0 | ac-rollout | rollout H=2, K=1 | 32 | 133.3 | 127.7 | 133.34 | 1372 |
| vjepa2-ac-vitg | f32 | ac-rollout | rollout H=2, K=16 | 32 | 2295.6 | 2294.8 | 143.48 | 5397 |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=2, K=16 | 32 | 2049.1 | 2044.4 | 128.07 | 2974 |
| vjepa2-ac-vitg | q8_0 | ac-rollout | rollout H=2, K=16 | 32 | 1874.3 | 1869.9 | 117.14 | 1760 |
| vjepa2-ac-vitg | f32 | ac-rollout | rollout H=2, K=64 | 32 | 9966.2 | 9926.8 | 155.72 | 6564 |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=2, K=64 | 32 | 8242.9 | 8216.7 | 128.80 | 4298 |
| vjepa2-ac-vitg | q8_0 | ac-rollout | rollout H=2, K=64 | 32 | 7508.9 | 7417.9 | 117.33 | 3008 |

`--batch` is the candidate count K, which is the axis a planner scales on: the K action sequences of one CEM iteration share a single graph per horizon step, so `ms / candidate` is what a candidate actually costs. **`ac`** is one `jepa_ac_predict` call; **`ac-rollout`** is `jepa_ac_rollout` over H steps and its ms is **per step**; **`ac-plan`** is `jepa_ac_plan` and its ms is **per CEM iteration**, i.e. what a planner pays per decision. None of them run the encoder — a planner encodes once and scores thousands of candidates against that one encode, which is the whole point of the cached context (see [architecture](architecture.md#the-cached-planning-context)). The rollout modes default to the cached-context entry point; `--no-cached` measures the explicit one, and the two agree to within run-to-run noise.

## LeWM world model

| model | ftype | mode | shape | threads | ms mean | ms min | steps/s |
|---|---|---|---|---:|---:|---:|---:|
| lewm-pusht | f32 | lewm-step | 3f x 192d | 32 | 0.911 | 0.863 | 1098 |
| lewm-pusht | f16 | lewm-step | 3f x 192d | 32 | 0.918 | 0.904 | 1089 |
| lewm-pusht | q8_0 | lewm-step | 3f x 192d | 32 | 0.739 | 0.718 | 1353 |
| lewm-pusht | f32 | lewm-rollout | rollout K=20 | 32 | 0.817 | 0.809 | 1224 |
| lewm-pusht | f16 | lewm-rollout | rollout K=20 | 32 | 0.865 | 0.850 | 1156 |
| lewm-pusht | q8_0 | lewm-rollout | rollout K=20 | 32 | 0.744 | 0.729 | 1344 |

`lewm-step` is one `jepa_lewm_predict` over the predictor's full 3-frame window; `lewm-rollout` is `jepa_lewm_rollout` and its ms is **per step** (the growing window means the first steps are cheaper than the last). Neither includes the encoder or the projector — see the encoder table for `lewm-pusht` for the cost of turning an image into a world-model state.

## GPU (CUDA)

The same `tools/jepa-bench`, the same synthetic input, one CUDA device instead of the CPU backend (`-DJEPA_CUDA=ON`, then `--gpu N`). These tables are keyed by device and accumulation precision where the ones above are keyed by thread count, which is why they have an artifact of their own: `tests/results/benchmarks-gpu.json`, written by `scripts/bench_gpu.sh` and read back by this generator.

Every row is the best of 5 runs after 2 warmups, and the warmups are not a formality: ggml's CUDA backend captures a CUDA graph once it has seen the same topology and the same tensor addresses twice in a row, so from the third call the encoder is one graph launch instead of hundreds of kernel launches. `ms sd` is the spread of the measured runs and is the width to read a difference between two rows against.

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DJEPA_CUDA=ON \
  && cmake --build build-cuda -j 32

# the PyTorch baseline of the last table (needs a CUDA-enabled torch)
python scripts/torch_gpu_baseline.py --device 0 -o tmp/bench-gpu/torch-gpu.json

# every configuration in scripts/bench_gpu.grid on device 0, then this section and its JSON
scripts/bench_gpu.sh 0
```

The configurations live in `scripts/bench_gpu.grid`, one line per (model, mode, shape, dtypes), and are the ones [performance.md](performance.md) publishes. Without `--gpu-dir` the generator rebuilds this section straight out of `tests/results/benchmarks-gpu.json`, so the document survives the loss of `tmp/bench-gpu/` — which is git-ignored — without the card.

### Card and build

| setting | value |
|---|---|
| GPU | NVIDIA RTX 4500 Ada Generation, 24570 MiB, compute 8.9, 210.00 W board limit |
| Device | NVIDIA RTX 4500 Ada Generation — every run below has the card to itself; the device index is per row (`device` in the artifact), because the sweeps that built this table did not all use the same card |
| Driver | 580.173.02 (CUDA 13.0 driver API) |
| Toolkit | `nvcc` 13.0.88 |
| Kernel | 6.17.0-1032-oem |
| Host compiler | c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| ggml | `36da5713`, **`GGML_LLAMAFILE=ON`** (a host-side path, unused here) |
| jepa.cpp | `b98cfa1a` |
| Precision | the accumulation precision the model's family defaults to (`src/jepa.cpp`, `jepa_gpu_prec_f32_default`) unless the row's `prec` column says otherwise; K/V F16 in flash attention for every file but f32 |

Measurement sessions (one `bench_gpu.sh` invocation each):

| device | warmup + measured | start | end | 1-min load avg | foreign cores | note |
|---|---|---|---|---|---:|---|
| CUDA0 | 2 + 5 | 2026-09-01 09:13 UTC | 2026-09-01 09:14 UTC | 0.68 → 1.09 | 0.24 | GPU twin sweep, idle box, device 0 |
| CUDA1 | 2 + 5 | 2026-09-01 17:06 UTC | 2026-09-01 17:07 UTC | 0.18 → 0.72 | 0.22 | V-JEPA 2-AC + ViT-g rows, plan item 7.1 |
| CUDA1 | 2 + 5 | 2026-09-01 17:10 UTC | 2026-09-01 17:10 UTC | 0.45 → 0.67 | 0.23 | — |
| CUDA1 | 2 + 5 | 2026-09-01 19:38 UTC | 2026-09-01 19:41 UTC | 0.75 → 1.00 | 0.22 | V-JEPA 2-AC planning KPIs (7.2) |
| CUDA1 | 2 + 5 | 2026-09-01 20:56 UTC | 2026-09-01 21:02 UTC | 0.47 → 1.00 | 0.19 | AC planning grid incl. H=4 (7.2 rework) |
| CUDA1 | 2 + 5 | 2026-09-01 22:40 UTC | 2026-09-01 22:41 UTC | 0.28 → 0.72 | 0.22 | accumulation-precision pairs and batched image/clip encoding, device 1 |

`foreign cores` is the CPU time the whole machine spent out of idle over the session minus the CPU time this sweep's own processes spent, divided by the wall clock: how much of the box belonged to somebody else while the card was timed. The highest here is **0.24** of one core out of 192, i.e. an idle box. A GPU row is host-idle by construction, so the load average alone would not have caught a second tenant.

### GPU encoder

| model | ftype | shape | tokens | device | ms mean | ms min | ms sd | tokens/s | peak RSS MiB | CPU f16 t=32 ms | vs CPU f16 t=32 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | f32 | 224x224 | 256 | CUDA0 | 11.86 | 11.81 | 0.074 | 21580 | 356 | 147.0 | 12.4x |
| ijepa_vith14_1k | f16 | 224x224 | 256 | CUDA0 | 15.52 | 15.45 | 0.044 | 16500 | 359 | 147.0 | 9.5x |
| ijepa_vith14_1k | f16 | 224x224 | 256 | CUDA1 | 16.07 | 16.07 | 0.008 | 15925 | 359 | 147.0 | 9.2x |
| ijepa_vith14_1k | f16 | 224x224 x8 | 2 048 | CUDA1 | 82.62 | 82.58 | 0.034 | 24788 | 360 | – | – |
| ijepa_vith14_1k | f16 | 224x224 x32 | 8 192 | CUDA1 | 358.35 | 358.11 | 0.151 | 22860 | 412 | – | – |
| ijepa_vith14_1k | q8_0 | 224x224 | 256 | CUDA0 | 7.99 | 7.99 | 0.002 | 32034 | 346 | 147.0 | 18.4x |
| ijepa_vith14_1k | q8_0 | 224x224 | 256 | CUDA1 | 8.12 | 8.11 | 0.004 | 31537 | 346 | 147.0 | 18.1x |
| ijepa_vith14_1k | q8_0 | 224x224 x8 | 2 048 | CUDA1 | 48.03 | 47.83 | 0.127 | 42639 | 363 | – | – |
| ijepa_vith14_1k | q8_0 | 224x224 x32 | 8 192 | CUDA1 | 288.99 | 288.95 | 0.051 | 28347 | 414 | – | – |
| ijepa_vith14_1k | q4_k | 224x224 | 256 | CUDA0 | 7.77 | 7.77 | 0.002 | 32947 | 347 | 147.0 | 18.9x |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 197 | CUDA0 | 1.08 | 1.08 | 0.000 | 181734 | 358 | 12.8 | 11.8x |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 197 | CUDA1 | 0.99 | 0.99 | 0.005 | 198469 | 428 | 12.8 | 12.9x |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 x8 | 1 576 | CUDA1 | 3.29 | 3.28 | 0.004 | 479129 | 423 | – | – |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 x32 | 6 304 | CUDA1 | 13.42 | 13.30 | 0.124 | 469691 | 457 | – | – |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | 197 | CUDA1 | 1.29 | 1.29 | 0.001 | 152796 | 361 | 12.8 | 9.9x |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 x8 | 1 576 | CUDA1 | 3.20 | 3.20 | 0.003 | 492254 | 356 | – | – |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 x32 | 6 304 | CUDA1 | 10.94 | 10.88 | 0.083 | 576392 | 390 | – | – |
| levjepa-vitl16 | f32 | 16f 224x224 | 3 137 | CUDA0 | 85.89 | 85.72 | 0.120 | 36524 | 399 | 1496.2 | 17.4x |
| levjepa-vitl16 | f16 | 16f 224x224 | 3 137 | CUDA0 | 87.64 | 87.36 | 0.277 | 35794 | 390 | 1496.2 | 17.1x |
| levjepa-vitl16 | f16 | 16f 224x224 | 3 137 | CUDA1 | 83.99 | 83.80 | 0.168 | 37351 | 394 | 1496.2 | 17.8x |
| levjepa-vitl16 | q8_0 | 16f 224x224 | 3 137 | CUDA0 | 71.39 | 70.88 | 0.402 | 43942 | 391 | 1496.2 | 21.0x |
| levjepa-vitl16 | q4_k | 16f 224x224 | 3 137 | CUDA0 | 71.23 | 71.00 | 0.198 | 44038 | 393 | 1496.2 | 21.0x |
| lewm-pusht | f16 | 224x224 | 257 | CUDA0 | 0.86 | 0.85 | 0.001 | 300409 | 358 | 9.8 | 11.4x |
| lewm-pusht | f16 | 224x224 | 257 | CUDA1 | 0.85 | 0.85 | 0.001 | 300937 | 358 | 9.8 | 11.4x |
| lewm-pusht | f16 | 224x224 x8 | 2 056 | CUDA1 | 1.99 | 1.99 | 0.001 | 1033218 | 352 | – | – |
| lewm-pusht | f16 | 224x224 x32 | 8 224 | CUDA1 | 6.38 | 6.37 | 0.005 | 1288624 | 384 | – | – |
| lewm-pusht | q8_0 | 224x224 | 257 | CUDA1 | 1.09 | 1.09 | 0.002 | 234853 | 362 | 9.8 | 8.9x |
| lewm-pusht | q8_0 | 224x224 x8 | 2 056 | CUDA1 | 2.02 | 2.01 | 0.002 | 1019791 | 355 | – | – |
| lewm-pusht | q8_0 | 224x224 x32 | 8 224 | CUDA1 | 6.49 | 6.48 | 0.004 | 1266848 | 387 | – | – |
| vjepa2-ac-vitg | f16 | 2f 256x256 | 256 | CUDA1 | 27.83 | 27.80 | 0.018 | 9197 | 347 | – | – |
| vjepa2-ac-vitg | q8_0 | 2f 256x256 | 256 | CUDA1 | 14.04 | 14.03 | 0.006 | 18230 | 350 | – | – |
| vjepa2-ac-vitg | q4_k | 2f 256x256 | 256 | CUDA1 | 13.37 | 13.37 | 0.002 | 19150 | 356 | – | – |
| vjepa2-vitg-fpc64-256 | f32 | 16f 256x256 | 2 048 | CUDA1 | 133.24 | 133.03 | 0.141 | 15371 | 386 | 2384.1 | 17.9x |
| vjepa2-vitg-fpc64-256 | f16 | 16f 256x256 | 2 048 | CUDA1 | 142.84 | 142.57 | 0.141 | 14338 | 389 | 2384.1 | 16.7x |
| vjepa2-vitg-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | CUDA1 | 101.74 | 101.61 | 0.100 | 20129 | 381 | 2384.1 | 23.4x |
| vjepa2-vitg-fpc64-256 | q4_k | 16f 256x256 | 2 048 | CUDA1 | 100.70 | 99.54 | 0.589 | 20337 | 387 | 2384.1 | 23.7x |
| vjepa2-vitg-fpc64-256 | f32 | 64f 256x256 | 8 192 | CUDA1 | 889.62 | 888.80 | 0.518 | 9208 | 475 | 16285.6 | 18.3x |
| vjepa2-vitg-fpc64-256 | f16 | 64f 256x256 | 8 192 | CUDA1 | 905.12 | 903.64 | 0.834 | 9051 | 480 | 16285.6 | 18.0x |
| vjepa2-vitg-fpc64-256 | q8_0 | 64f 256x256 | 8 192 | CUDA1 | 831.24 | 830.81 | 0.321 | 9855 | 485 | 16285.6 | 19.6x |
| vjepa2-vitg-fpc64-256 | q4_k | 64f 256x256 | 8 192 | CUDA1 | 832.29 | 832.17 | 0.073 | 9843 | 491 | 16285.6 | 19.6x |
| vjepa2-vitl-fpc64-256 | f32 | 16f 256x256 | 2 048 | CUDA0 | 43.57 | 42.94 | 0.338 | 47006 | 382 | 820.7 | 18.8x |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | CUDA0 | 46.47 | 46.29 | 0.096 | 44068 | 374 | 820.7 | 17.7x |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | CUDA1 | 47.02 | 46.98 | 0.048 | 43554 | 374 | 820.7 | 17.4x |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 x8 | 16 384 | CUDA1 | 381.08 | 379.93 | 0.837 | 42994 | 509 | – | – |
| vjepa2-vitl-fpc64-256 | q8_0 | 16f 256x256 | 2 048 | CUDA0 | 33.86 | 32.87 | 0.595 | 60488 | 379 | 820.7 | 24.2x |
| vjepa2-vitl-fpc64-256 | q4_k | 16f 256x256 | 2 048 | CUDA0 | 34.66 | 34.22 | 0.302 | 59095 | 380 | 820.7 | 23.7x |
| vjepa2-vitl-fpc64-256 | f32 | 64f 256x256 | 8 192 | CUDA0 | 302.59 | 302.18 | 0.259 | 27073 | 465 | 6388.1 | 21.1x |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 8 192 | CUDA0 | 305.43 | 305.22 | 0.223 | 26821 | 468 | 6388.1 | 20.9x |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 8 192 | CUDA1 | 309.43 | 308.73 | 0.526 | 26475 | 469 | 6388.1 | 20.6x |
| vjepa2-vitl-fpc64-256 | q8_0 | 64f 256x256 | 8 192 | CUDA0 | 280.42 | 280.35 | 0.071 | 29214 | 473 | 6388.1 | 22.8x |
| vjepa2-vitl-fpc64-256 | q4_k | 64f 256x256 | 8 192 | CUDA0 | 281.38 | 281.28 | 0.090 | 29113 | 476 | 6388.1 | 22.7x |
| vjepa2_1-vitb-384 | f16 | 384x384 | 576 | CUDA0 | 4.38 | 4.38 | 0.004 | 131411 | 346 | 60.3 | 13.8x |
| vjepa2_1-vitb-384 | f16 | 384x384 | 576 | CUDA1 | 4.47 | 4.46 | 0.004 | 128920 | 346 | 60.3 | 13.5x |
| vjepa2_1-vitb-384 | f16 | 384x384 x8 | 4 608 | CUDA1 | 36.33 | 36.03 | 0.191 | 126825 | 370 | – | – |
| vjepa2_1-vitb-384 | f16 | 384x384 x32 | 18 432 | CUDA1 | 144.06 | 143.91 | 0.098 | 127949 | 445 | – | – |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | 576 | CUDA1 | 3.45 | 3.44 | 0.004 | 167126 | 352 | 60.3 | 17.5x |
| vjepa2_1-vitb-384 | q8_0 | 384x384 x8 | 4 608 | CUDA1 | 27.74 | 27.69 | 0.039 | 166122 | 375 | – | – |
| vjepa2_1-vitb-384 | q8_0 | 384x384 x32 | 18 432 | CUDA1 | 111.26 | 111.17 | 0.107 | 165666 | 452 | – | – |
| vjepa2_1-vitb-384 | q4_k | 384x384 | 576 | CUDA0 | 3.44 | 3.44 | 0.002 | 167252 | 352 | 60.3 | 17.5x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | CUDA0 | 42.46 | 42.09 | 0.467 | 108535 | 410 | 853.5 | 20.1x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | CUDA1 | 43.38 | 43.23 | 0.098 | 106234 | 410 | 853.5 | 19.7x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 x8 | 36 864 | CUDA1 | 349.01 | 348.63 | 0.244 | 105624 | 690 | – | – |
| vjepa2_1-vitb-384 | q4_k | 16f 384x384 | 4 608 | CUDA0 | 37.35 | 37.13 | 0.146 | 123366 | 416 | 853.5 | 22.9x |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 18 432 | CUDA0 | 424.21 | 420.98 | 1.70 | 43450 | 618 | 9036.1 | 21.3x |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 18 432 | CUDA1 | 429.07 | 426.13 | 1.99 | 42958 | 619 | 9036.1 | 21.1x |

`peak RSS` is **host** memory (the process `VmHWM`), not device memory: the weights are uploaded and the host copy is released, so it says little beyond the size of the graph arena and the patch buffer. The speed-up column divides the 32-thread f16 run of the same graph and shape — from the Encoder table above, i.e. 96 Zen 4 cores' worth of machine against one workstation card — by this row, whatever this row's dtype is.

### Effect of the weight dtype on a GPU (encoder)

| model | shape | tokens | f32 ms | f16 ms | q8_0 ms | q4_k ms | f16 → q8_0 | f16 → q4_k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | 224x224 | 256 | 11.9 | 16.1 | 8.1 | 7.8 | 1.98x | 2.07x |
| ijepa_vith14_1k | 224x224 x8 | 2 048 | – | 82.6 | 48.0 | – | 1.72x | – |
| ijepa_vith14_1k | 224x224 x32 | 8 192 | – | 358 | 289 | – | 1.24x | – |
| lejepa-vits16-pretrain-in1k | 224x224 | 197 | – | 1.0 | 1.3 | – | 0.77x | – |
| lejepa-vits16-pretrain-in1k | 224x224 x8 | 1 576 | – | 3.3 | 3.2 | – | 1.03x | – |
| lejepa-vits16-pretrain-in1k | 224x224 x32 | 6 304 | – | 13.4 | 10.9 | – | 1.23x | – |
| levjepa-vitl16 | 16f 224x224 | 3 137 | 85.9 | 84.0 | 71.4 | 71.2 | 1.18x | 1.18x |
| lewm-pusht | 224x224 | 257 | – | 0.9 | 1.1 | – | 0.78x | – |
| lewm-pusht | 224x224 x8 | 2 056 | – | 2.0 | 2.0 | – | 0.99x | – |
| lewm-pusht | 224x224 x32 | 8 224 | – | 6.4 | 6.5 | – | 0.98x | – |
| vjepa2-ac-vitg | 2f 256x256 | 256 | – | 27.8 | 14.0 | 13.4 | 1.98x | 2.08x |
| vjepa2-vitg-fpc64-256 | 16f 256x256 | 2 048 | 133 | 143 | 102 | 101 | 1.40x | 1.42x |
| vjepa2-vitg-fpc64-256 | 64f 256x256 | 8 192 | 890 | 905 | 831 | 832 | 1.09x | 1.09x |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | 2 048 | 43.6 | 47.0 | 33.9 | 34.7 | 1.39x | 1.36x |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | 8 192 | 303 | 309 | 280 | 281 | 1.10x | 1.10x |
| vjepa2_1-vitb-384 | 384x384 | 576 | – | 4.5 | 3.4 | 3.4 | 1.30x | 1.30x |
| vjepa2_1-vitb-384 | 384x384 x8 | 4 608 | – | 36.3 | 27.7 | – | 1.31x | – |
| vjepa2_1-vitb-384 | 384x384 x32 | 18 432 | – | 144 | 111 | – | 1.29x | – |
| vjepa2_1-vitb-384 | 16f 384x384 | 4 608 | – | 43.4 | – | 37.4 | – | 1.16x |

**The CPU ordering inverts here.** Every type jepa.cpp ships takes `mmq`, a real INT8 tensor-core kernel, so q8_0 and q4_k both beat f16 while being half and a quarter of the weight bytes — where on the CPU the k-quants fall off llamafile's accelerated sgemm and lose. The f32 column is not slower than f16 because ggml's CUDA F32 path is TF32, while the f16 path pays for `GGML_PREC_F32` accumulation. Accuracy per type does not invert with the backend: `docs/parity.md` *Results — encoders on CUDA0* has the cosines.

### What `GGML_PREC_F32` costs (against f16 accumulation)

| model | ftype | shape | tokens | `GGML_PREC_F32` ms | f16 accumulation ms | cost of F32 accumulation |
|---|---|---|---:|---:|---:|---:|
| ijepa_vith14_1k | f16 | 224x224 | 256 | 16.1 | 8.9 | 1.80x |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | 197 | 1.1 | 1.0 | 1.11x |
| levjepa-vitl16 | f16 | 16f 224x224 | 3 137 | 89.2 | 84.0 | 1.06x |
| lewm-pusht | f16 | 224x224 | 257 | 0.9 | 0.8 | 1.02x |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | 2 048 | 47.0 | 38.0 | 1.24x |
| vjepa2-vitl-fpc64-256 | f16 | 64f 256x256 | 8 192 | 309 | 305 | 1.01x |
| vjepa2_1-vitb-384 | f16 | 384x384 | 576 | 4.5 | 3.4 | 1.30x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | 4 608 | 43.4 | 44.4 | 0.98x |
| vjepa2_1-vitb-384 | f16 | 64f 384x384 | 18 432 | 429 | 446 | 0.96x |

f16 accumulation hands the `mul_mat`s cuBLAS' own f16 compute type instead of forcing F32. Which of the two a family defaults to is decided per family by the parity sweep `scripts/gpu_prec_sweep.sh` and by these milliseconds together (`tests/results/gpu-prec.json`, and `docs/performance.md` *Accumulation precision on a GPU*); `--gpu-prec` and `$JEPA_GPU_PREC` select either one for any run. The cost is a strong function of the shape — it is what holds the small image models back against the long clips' twenties in the speed-up column above — and it is largest on I-JEPA, whose tier the faster setting does not clear.

### Predictor, head and world model on a GPU

| model | ftype | mode | shape | tokens | ms mean | ms min | ms sd | encoder ms | CPU f16 t=32 ms | vs CPU f16 t=32 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc16-256-ssv2 | f16 | head | 16f 256x256 | 2 048 | 5.612 | 5.497 | 0.089 | 46.9 | 98.97 | 17.63x |
| vjepa2-vitg-fpc64-256 | f16 | predictor | 16f 256x256 | 2 048 | 112.242 | 112.172 | 0.056 | 143.7 | 354.68 | 3.16x |
| vjepa2-vitl-fpc16-256-ssv2 | f16 | predictor | 16f 256x256 | 2 048 | 112.744 | 112.654 | 0.060 | 45.6 | 340.76 | 3.02x |
| vjepa2-vitl-fpc64-256 | f16 | predictor | 16f 256x256 | 2 048 | 112.804 | 112.721 | 0.063 | 46.3 | 343.84 | 3.05x |
| lewm-pusht | f16 | lewm-step | 3f x 192d | 3 | 0.450 | 0.441 | 0.006 | – | 0.92 | 2.04x |
| lewm-pusht | f16 | lewm-rollout | rollout K=20 | 1 | 0.448 | 0.439 | 0.013 | – | 0.86 | 1.93x |
| vjepa2-ac-vitg | f16 | ac | 1f x 256tok, K=1 | 256 | 8.984 | 8.896 | 0.072 | – | 94.87 | 10.56x |
| vjepa2-ac-vitg | f16 | ac | 1f x 256tok, K=16 | 4 096 | 113.118 | 112.886 | 0.191 | – | 1337.13 | 11.82x |
| vjepa2-ac-vitg | f16 | ac | 1f x 256tok, K=64 | 16 384 | 533.225 | 533.008 | 0.138 | – | 5411.15 | 10.15x |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=2, K=1 | 256 | 11.867 | 11.857 | 0.007 | – | 138.88 | 11.70x |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=2, K=16 | 256 | 175.968 | 175.884 | 0.066 | – | 2049.10 | 11.64x |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=4, K=16 | 256 | 321.001 | 320.864 | 0.100 | – | – | – |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=2, K=64 | 256 | 787.574 | 786.909 | 0.390 | – | 8242.94 | 10.47x |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=4, K=64 | 256 | 1341.136 | 1340.755 | 0.290 | – | – | – |
| vjepa2-ac-vitg | f16 | ac-rollout | rollout H=2, K=256 | 256 | 3173.207 | 3167.872 | 3.27 | – | – | – |
| vjepa2-ac-vitg | q8_0 | ac-rollout | rollout H=2, K=16 | 256 | 143.976 | 143.940 | 0.055 | – | 2049.10 | 14.23x |
| vjepa2-ac-vitg | q8_0 | ac-rollout | rollout H=4, K=16 | 256 | 283.755 | 283.715 | 0.038 | – | – | – |
| vjepa2-ac-vitg | q8_0 | ac-rollout | rollout H=2, K=64 | 256 | 750.660 | 750.453 | 0.114 | – | 8242.94 | 10.98x |
| vjepa2-ac-vitg | q8_0 | ac-rollout | rollout H=4, K=64 | 256 | 1331.563 | 1331.518 | 0.028 | – | – | – |
| vjepa2-ac-vitg | f16 | ac-plan | plan K=16, H=2, it=1 | 512 | 357.957 | 357.811 | 0.128 | – | – | – |
| vjepa2-ac-vitg | f16 | ac-plan | plan K=16, H=4, it=1 | 1 024 | 1285.918 | 1285.186 | 0.417 | – | – | – |
| vjepa2-ac-vitg | f16 | ac-plan | plan K=64, H=2, it=1 | 512 | 1590.445 | 1589.309 | 0.592 | – | – | – |
| vjepa2-ac-vitg | f16 | ac-plan | plan K=64, H=4, it=1 | 1 024 | 5368.995 | 5367.895 | 0.959 | – | – | – |
| vjepa2-ac-vitg | f16 | ac-plan | plan K=256, H=2, it=1 | 512 | 6371.005 | 6366.964 | 3.54 | – | – | – |
| vjepa2-ac-vitg | q8_0 | ac-plan | plan K=16, H=2, it=1 | 512 | 290.904 | 290.706 | 0.133 | – | – | – |
| vjepa2-ac-vitg | q8_0 | ac-plan | plan K=64, H=2, it=1 | 512 | 1503.825 | 1503.678 | 0.083 | – | – | – |

These are the synthetic-input graphs of the *Masked predictor*, *Attentive-pool head* and *LeWM world model* tables above, run on the card. The masked predictor is the one encoder-sized graph that does **not** gain twentyfold: at `head_dim` 32 no CUDA flash-attention kernel exists, so it takes the naive `mul_mat + soft_max_ext` path — genuinely F32, and about 3 TFLOP/s against flash's 50–70 (`docs/architecture.md` "GPU backend"). The LeWM graphs are the opposite end: three rows of 192 dimensions is far below the size at which a kernel launch pays for itself, and `docs/parity.md` *Results — predictors on CUDA0* times the same two graphs on the real fixture state.

### Batched encoding on a GPU (`--batch B`, one graph)

| model | ftype | shape | device | prec | ms/item b=1 | b=8 | b=32 | items/s b=1 | b=8 | b=32 | best gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | f16 | 224x224 | CUDA1 | f32 | 16.075 | 10.328 | 11.199 | 62.2 | 96.8 | 89.3 | 1.56x |
| ijepa_vith14_1k | q8_0 | 224x224 | CUDA1 | f32 | 8.117 | 6.004 | 9.031 | 123.2 | 166.6 | 110.7 | 1.35x |
| lejepa-vits16-pretrain-in1k | f16 | 224x224 | CUDA1 | f16 | 0.993 | 0.411 | 0.419 | 1007.0 | 2432.4 | 2384.1 | 2.42x |
| lejepa-vits16-pretrain-in1k | q8_0 | 224x224 | CUDA1 | f16 | 1.289 | 0.400 | 0.342 | 775.8 | 2498.4 | 2925.8 | 3.77x |
| lewm-pusht | f16 | 224x224 | CUDA1 | f32 | 0.854 | 0.249 | 0.199 | 1171.0 | 4020.1 | 5014.1 | 4.28x |
| lewm-pusht | q8_0 | 224x224 | CUDA1 | f32 | 1.094 | 0.252 | 0.203 | 914.1 | 3968.3 | 4929.1 | 5.39x |
| vjepa2-vitl-fpc64-256 | f16 | 16f 256x256 | CUDA1 | f32 | 47.023 | 47.635 | – | 21.3 | 21.0 | – | 1.00x |
| vjepa2_1-vitb-384 | f16 | 384x384 | CUDA1 | f32 | 4.468 | 4.542 | 4.502 | 223.8 | 220.2 | 222.1 | 1.00x |
| vjepa2_1-vitb-384 | f16 | 16f 384x384 | CUDA1 | f32 | 43.376 | 43.626 | – | 23.1 | 22.9 | – | 1.00x |
| vjepa2_1-vitb-384 | q8_0 | 384x384 | CUDA1 | f32 | 3.446 | 3.467 | 3.477 | 290.2 | 288.4 | 287.6 | 1.00x |

`--batch B` puts B items through **one** graph on the batch dimension, so the row is one `ggml_backend_graph_compute` and `ms/item` is it divided by B. What batching amortises is everything that does not scale with the matmuls — kernel launches, the per-layer norm and activation passes, weight streaming — so the gain is largest where a single item leaves the card idle and smallest where one item already fills it.

### PyTorch on the same card

| model | shape | tokens | jepa.cpp CUDA f16 ms | torch fp16 ms | torch fp32 ms | ggml / torch fp16 | torch fp16 peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | 224x224 | 256 | 15.5 | 5.51 | 23.39 | 2.8x | 1.19 |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | 2 048 | 46.5 | 28.92 | 117.9 | 1.6x | 0.67 |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | 8 192 | 305 | 148.1 | 836.4 | 2.1x | 0.83 |
| levjepa-vitl16 | 16f 224x224 | 3 137 | 87.6 | 58.75 | 224.3 | 1.5x | 0.68 |

`scripts/torch_gpu_baseline.py` on the same device: torch 2.13.0+cu130, transformers 5.16.1, batch 1, TF32 off, 3 warmup + 7 timed forwards, cuda.synchronize() around each, on the stored preprocessed tensor of a reference fixture — the same pixels, not merely the same shape. `VJEPA2Model` runs with `skip_predictor=True`, so its forward is the encoder alone. `torch fp16 peak GiB` is `max_memory_allocated` after the warmups, one model per precision, so it is the steady-state device footprint of that precision and nothing else.

### PyTorch batched, eager against `torch.compile`

| model | shape | precision | runtime | ms/item b=1 | b=8 | b=32 | items/s b=1 | b=8 | b=32 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ijepa_vith14_1k | 224x224 | fp16 | eager | 5.893 | 4.160 | 4.245 | 169.7 | 240.4 | 235.6 |
| ijepa_vith14_1k | 224x224 | fp16 | compile | 7.177 | 5.688 | 6.389 | 139.3 | 175.8 | 156.5 |
| ijepa_vith14_1k | 224x224 | fp32 | eager | 23.624 | 21.621 | 20.436 | 42.3 | 46.2 | 48.9 |
| ijepa_vith14_1k | 224x224 | fp32 | compile | 23.825 | 21.552 | 20.299 | 42.0 | 46.4 | 49.3 |
| lejepa-vits16-pretrain-in1k | 224x224 | fp16 | eager | 2.228 | 0.302 | 0.259 | 448.9 | 3312.2 | 3858.7 |
| lejepa-vits16-pretrain-in1k | 224x224 | fp16 | compile | 0.920 | 0.238 | 0.212 | 1086.8 | 4199.7 | 4711.6 |
| lejepa-vits16-pretrain-in1k | 224x224 | fp32 | eager | 1.983 | 0.720 | 0.816 | 504.2 | 1388.3 | 1226.0 |
| lejepa-vits16-pretrain-in1k | 224x224 | fp32 | compile | 1.578 | 0.704 | 0.775 | 633.8 | 1419.8 | 1290.2 |
| levjepa-vitl16 | 16f 224x224 | fp16 | eager | 60.468 | 65.617 | – | 16.5 | 15.2 | – |
| levjepa-vitl16 | 16f 224x224 | fp16 | compile | 54.915 | 52.179 | – | 18.2 | 19.2 | – |
| levjepa-vitl16 | 16f 224x224 | fp32 | eager | 231.570 | 240.379 | – | 4.3 | 4.2 | – |
| levjepa-vitl16 | 16f 224x224 | fp32 | compile | 222.148 | 216.436 | – | 4.5 | 4.6 | – |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | fp16 | eager | 30.302 | 30.086 | – | 33.0 | 33.2 | – |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | fp16 | compile | 24.338 | 24.064 | – | 41.1 | 41.6 | – |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | fp32 | eager | 119.532 | 125.055 | – | 8.4 | 8.0 | – |
| vjepa2-vitl-fpc64-256 | 16f 256x256 | fp32 | compile | 111.087 | 109.947 | – | 9.0 | 9.1 | – |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | fp16 | eager | 153.977 | – | – | 6.5 | – | – |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | fp16 | compile | 155.030 | – | – | 6.5 | – | – |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | fp32 | eager | 861.161 | – | – | 1.2 | – | – |
| vjepa2-vitl-fpc64-256 | 64f 256x256 | fp32 | compile | 868.312 | – | – | 1.1 | – | – |

`scripts/torch_gpu_baseline.py --device 1 --batch 1,8,32 --compile`: the fixture tensor repeated along the batch axis, which is the axis `jepa-bench --batch` sweeps, so the two engines are timed on the same work. `torch.compile` is warmed up before timing, so its compilation is not in these milliseconds — a served model pays it once. This is a session of its own on the second card, kept apart from the batch-1 baseline above rather than merged into it.

Configurations the baseline could not measure, and why:

| model | precision | runtime | batch | reason |
|---|---|---|---|---|
| vjepa2-vitl-fpc64-256 | fp32 | eager | 32 | 65536 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp32 | compile | 32 | 65536 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp16 | eager | 32 | 65536 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp16 | compile | 32 | 65536 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp32 | eager | 8 | 65536 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp32 | eager | 32 | 262144 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp32 | compile | 8 | 65536 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp32 | compile | 32 | 262144 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp16 | eager | 8 | 65536 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp16 | eager | 32 | 262144 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp16 | compile | 8 | 65536 rows is over the --max-batch-tokens 32768 ceiling |
| vjepa2-vitl-fpc64-256 | fp16 | compile | 32 | 262144 rows is over the --max-batch-tokens 32768 ceiling |
| levjepa-vitl16 | fp32 | eager | 32 | 100384 rows is over the --max-batch-tokens 32768 ceiling |
| levjepa-vitl16 | fp32 | compile | 32 | 100384 rows is over the --max-batch-tokens 32768 ceiling |
| levjepa-vitl16 | fp16 | eager | 32 | 100384 rows is over the --max-batch-tokens 32768 ceiling |
| levjepa-vitl16 | fp16 | compile | 32 | 100384 rows is over the --max-batch-tokens 32768 ceiling |

## Footnotes

<sup>fpc64</sup> the `vjepa2-vitl-fpc64-256` manifest times one `VJEPA2Model` forward, which always runs the **predictor** as well (its `predictor_last_hidden_state` comes from the same call), so it is an upper bound on the encoder and no speedup is claimed against it.

<sup>ssv2</sup> the SSv2 manifest times `VJEPA2ForVideoClassification`, i.e. encoder + attentive pooler + classifier with the predictor skipped. It is therefore not comparable with the encoder row alone; the end-to-end table under *Attentive-pool head* adds our encoder and head and makes the comparison there.

<sup>lewm</sup> the LeWM manifest times encode + projector + one 1-frame predictor call; the two extra graphs are ~1 ms of it (see the world-model table), so the speedup is a slight over-estimate.

---

Generated by `scripts/gen_benchmarks_md.py` from 113 runs in `tmp/bench` and 104 GPU runs in `tmp/bench-gpu`. Cross-check against `docs/parity.md` (same graphs, real fixture inputs) and `docs/quantization.md` (accuracy per dtype).
