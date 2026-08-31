# Accuracy — video k-NN on the UCF101 subset (PyTorch vs jepa.cpp)

Frozen-feature evaluation, 2026-08-31. **Inference only** — nothing is trained: the encoders are frozen and both metrics are look-ups over their pooled clip features.

- **Dataset** `data/ucf101-subset/UCF101_subset` — 10 classes, gallery = train (300 clips), queries = test (75) / val (30) / val+test (105).
- **Clip** 16 frames, `idx = round(linspace(0, T_total-1, 16)) over all PyAV rgb24 frames`, decoded once with PyAV to a THWC uint8 `.npy` that *both* backends read, so the two see identical pixels.
- **Feature** mean over encoder tokens (pooled_mean), L2-normalized.
- **k-NN** k = 20, cosine similarity, DINO-style weighted vote (`exp(sim / 0.07)`). **Centroid** = nearest L2-normalized class mean of the gallery (no hyper-parameters).
- **Agreement** = fraction of query clips where the jepa.cpp prediction equals the PyTorch one (k-NN and centroid separately); **feat cos** = mean per-clip cosine between the two backends' feature vectors.
- 32 threads everywhere (AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, AVX-512), gcc 13.3.0, ggml @ 36da5713, torch 2.13.0+cpu / transformers 5.16.1. Protocol implemented in `scripts/_knn_video.py`.
- Chance is 10 % (10 classes); the largest query class holds 13.3 % of the clips.

## V-JEPA 2 ViT-L/16 (fpc64-256) — `vjepa2-vitl-fpc64-256`

| backend | dtype | query split | k-NN top-1 % | centroid top-1 % | k-NN agreement % | centroid agreement % | feat cos | clips/s |
|---|---|---|--:|--:|--:|--:|--:|--:|
| pytorch | f32 | test (75) | 88.0 | 94.7 | ref | ref | ref | 0.45 |
| pytorch | f32 | val (30) | 90.0 | 96.7 | ref | ref | ref | 0.45 |
| pytorch | f32 | val+test (105) | 88.6 | 95.2 | ref | ref | ref | 0.45 |
| jepa.cpp | f16 | test (75) | 89.3 | 94.7 | 98.7 | 100.0 | 0.999996 | 0.99 |
| jepa.cpp | f16 | val (30) | 90.0 | 96.7 | 100.0 | 100.0 | 0.999995 | 0.99 |
| jepa.cpp | f16 | val+test (105) | 89.5 | 95.2 | 99.0 | 100.0 | 0.999996 | 0.99 |
| jepa.cpp | q8_0 | test (75) | 89.3 | 94.7 | 98.7 | 100.0 | 0.999889 | 0.97 |
| jepa.cpp | q8_0 | val (30) | 90.0 | 96.7 | 100.0 | 100.0 | 0.999880 | 0.97 |
| jepa.cpp | q8_0 | val+test (105) | 89.5 | 95.2 | 99.0 | 100.0 | 0.999886 | 0.97 |

`clips/s` is end-to-end over all 405 clips (frame `.npy` -> preprocess -> encode -> pool), one clip at a time, excluding the model load. Agreement and feat cos are against the PyTorch row of the same model.

Feature fidelity over **all** 405 clips (gallery + queries), against the PyTorch vectors:

| dtype | mean cos | worst clip cos | max abs component diff |
|---|--:|--:|--:|
| f16 | 0.9999946 | 0.9999526 | 7.08e-02 |
| q8_0 | 0.9998754 | 0.9996434 | 3.03e-01 |

## V-JEPA 2.1 ViT-B/16 @384 — `vjepa2_1-vitb-384`

| backend | dtype | query split | k-NN top-1 % | centroid top-1 % | k-NN agreement % | centroid agreement % | feat cos | clips/s |
|---|---|---|--:|--:|--:|--:|--:|--:|
| pytorch | f32 | test (75) | 88.0 | 85.3 | ref | ref | ref | 0.74 |
| pytorch | f32 | val (30) | 90.0 | 90.0 | ref | ref | ref | 0.74 |
| pytorch | f32 | val+test (105) | 88.6 | 86.7 | ref | ref | ref | 0.74 |
| jepa.cpp | f32 | test (75) | 88.0 | 85.3 | 100.0 | 100.0 | 1.000000 | 0.91 |
| jepa.cpp | f32 | val (30) | 90.0 | 90.0 | 100.0 | 100.0 | 1.000000 | 0.91 |
| jepa.cpp | f32 | val+test (105) | 88.6 | 86.7 | 100.0 | 100.0 | 1.000000 | 0.91 |
| jepa.cpp | f16 | test (75) | 88.0 | 85.3 | 100.0 | 100.0 | 1.000000 | 0.85 |
| jepa.cpp | f16 | val (30) | 93.3 | 90.0 | 96.7 | 100.0 | 1.000000 | 0.85 |
| jepa.cpp | f16 | val+test (105) | 89.5 | 86.7 | 99.0 | 100.0 | 1.000000 | 0.85 |
| jepa.cpp | q8_0 | test (75) | 88.0 | 85.3 | 100.0 | 100.0 | 0.999988 | 0.83 |
| jepa.cpp | q8_0 | val (30) | 93.3 | 90.0 | 96.7 | 100.0 | 0.999987 | 0.83 |
| jepa.cpp | q8_0 | val+test (105) | 89.5 | 86.7 | 99.0 | 100.0 | 0.999988 | 0.83 |

`clips/s` is end-to-end over all 405 clips (frame `.npy` -> preprocess -> encode -> pool), one clip at a time, excluding the model load. Agreement and feat cos are against the PyTorch row of the same model.

Feature fidelity over **all** 405 clips (gallery + queries), against the PyTorch vectors:

| dtype | mean cos | worst clip cos | max abs component diff |
|---|--:|--:|--:|
| f32 | 1.0000000 | 1.0000000 | 7.13e-05 |
| f16 | 0.9999999 | 0.9999990 | 1.13e-02 |
| q8_0 | 0.9999874 | 0.9999731 | 5.52e-02 |

## SSv2 classification head — backend fidelity

`facebook/vjepa2-vitl-fpc16-256-ssv2` (attentive pooler + linear head, 174 Something-Something-v2 classes) run on the same 105 query clips. **This is not a task accuracy**: SSv2 labels have nothing to do with the UCF101 classes, so an SSv2 prediction on a UCF clip is meaningless as a label. What it measures is whether jepa.cpp's head reaches the *same* decision as PyTorch's on real, out-of-distribution video — 105 independent 174-way argmaxes over a full encoder + pooler + classifier stack, which a parity fixture of a handful of clips cannot cover.

| backend | dtype | top-1 agreement % | top-5 overlap % | PyTorch top-1 in top-5 % | max abs logit diff | logit cos | clips/s |
|---|---|--:|--:|--:|--:|--:|--:|
| pytorch | f32 | ref | ref | ref | ref | ref | 0.70 |
| jepa.cpp | f16 | 99.0 | 99.4 | 100.0 | 0.1885 | 0.999970 | 0.82 |
| jepa.cpp | q8_0 | 94.3 | 97.0 | 100.0 | 1.0421 | 0.998922 | 0.82 |

`top-5 overlap` is the mean size of the intersection of the two top-5 label sets divided by 5.

## What the numbers say

**The f32 anchor is exact end to end.** On all 405 clips, V-JEPA 2.1 ViT-B/16 @384 at f32 reproduces the PyTorch pooled vector to cosine 1.0000000 (worst clip 1.0000000, largest single-component difference 7.1e-05) and every k-NN and centroid prediction is identical. That covers the *whole* pipeline, not just the encoder: jepa.cpp decodes nothing, but it does its own resize, centre crop and normalisation from the same uint8 frames, so the match confirms `jepa.pre.*` reproduces the HF video processor's pixels as well as the graph reproduces the weights. `docs/parity.md` shows the same at token level on two fixture clips; this is 405 real clips of pooled output.

**f16 and q8_0 do not move the accuracy.** Across both models and every query split the k-NN and centroid top-1 numbers are within one clip of the PyTorch row, and the single worst clip out of all 405 at any quantisation tested here still matches the PyTorch pooled vector to cosine 0.999643 (`vjepa2-vitl-fpc64-256` q8_0). Where a jepa.cpp row reads *higher* than PyTorch — 89.5 vs 88.6 % on val+test — that is a single clip out of 105, i.e. 0.95 pp of quantisation noise landing on the right side. It is not an improvement, and the doc reports it rather than hiding it because the reverse would have been equally likely.

**Every disagreement is a tie in the neighbour set, not a feature error.** The k-NN predictions that differ between the backends are:

| model | dtype | clip | true | PyTorch | jepa.cpp | PyTorch top-2 vote ratio | shared neighbours | cos gap 20th↔21st | backend cos shift | vote shift |
|---|---|---|---|---|---|--:|--:|--:|--:|--:|
| vjepa2-vitl-fpc64-256 | f16 | `v_ApplyEyeMakeup_g23_c04` (test) | ApplyEyeMakeup | ApplyLipstick | ApplyEyeMakeup | 1.1021 | 19/20 | 4.7e-04 | 1.0e-03 | -5.4 % |
| vjepa2-vitl-fpc64-256 | q8_0 | `v_ApplyEyeMakeup_g23_c04` (test) | ApplyEyeMakeup | ApplyLipstick | ApplyEyeMakeup | 1.1021 | 19/20 | 4.7e-04 | 4.8e-03 | -3.9 % |
| vjepa2_1-vitb-384 | f16 | `v_Basketball_g20_c02` (val) | Basketball | BabyCrawling | Basketball | 1.0030 | 19/20 | 1.3e-05 | 7.0e-05 | -15.6 % |
| vjepa2_1-vitb-384 | q8_0 | `v_Basketball_g20_c02` (val) | Basketball | BabyCrawling | Basketball | 1.0030 | 19/20 | 1.3e-05 | 4.8e-04 | -15.6 % |

In each case the two backends agree on 19 of the 20 nearest gallery clips and differ only at the last one — and the final columns say why: the 20th- and 21st-ranked gallery clips are separated by *less* cosine than the two backends' similarities to that query differ, by 2.2x to 36x, so which of the two lands inside the neighbourhood is decided by round-off. That last neighbour is not a rounding term in the vote: its `exp(sim / 0.07)` weight is 0.18 and 0.61 of the top neighbour's, so one swap moves the leading class total by 4–16 %, which is enough to decide a vote that was already a 1.003 / 1.102 near-tie. The tell is the parameter-free metric: **nearest-class-centroid agreement is 100 % for every model and every dtype** — with no k and no neighbour set, there is nothing for a 1e-6 perturbation to reshuffle. Read the k-NN agreement column as a property of k-NN at k = 20 on a 300-clip gallery, not as a fidelity measure of the backend; `feat cos` and the centroid column are the fidelity measures.

**The SSv2 head is where q8_0 finally costs something.** f16 reaches the same 174-way argmax as PyTorch on 99.0 % of the 105 clips (logit cosine 0.999970); q8_0 drops to 94.3 % (6 clips of 105) with logit cosine 0.998922 and a largest logit error of 1.04. The PyTorch top-1 stays inside the jepa.cpp top-5 on 100.0 % of clips at both dtypes, so the ranking is intact and only near-ties at the top move. This is the sharpest measurement in the document: an argmax over 174 classes has no averaging to hide behind, unlike a pooled 1024-vector whose cosine stays at 0.9999 — which is exactly why `docs/parity.md`'s advice to prefer f16 over q8_0 for head/classifier work, and q8_0 only for pooled retrieval features, holds up on 105 real clips.

**Throughput.** jepa.cpp is faster than PyTorch on the same 32 threads in every configuration measured here: V-JEPA 2 ViT-L/16 (fpc64-256) 0.99 clips/s at f16 vs 0.45 (2.21x); V-JEPA 2.1 ViT-B/16 @384 0.91 clips/s at f32 vs 0.74 (1.23x). Both sides include preprocessing and run one clip per process with no batching. Within jepa.cpp the dtype barely moves the clock — ViT-L f16 0.99 vs q8_0 0.97 clips/s, and on V-JEPA 2.1 ViT-B/384 the **f32** file is actually the fastest of the three (0.91 vs 0.85 f16 vs 0.83 q8_0). `docs/parity.md` sees the same absence of a dtype speedup on its two fixture clips (1073 / 1125 / 1067 ms per clip for f32 / f16 / q8_0). These encoders are compute-bound at 32 threads, so q8_0 buys resident weights (332.8 vs 622.5 MiB for ViT-L, 113.3 vs 209.6 MiB for 2.1 ViT-B — 0.53x and 0.54x), not speed.

**Practical reading.** For frozen-feature video retrieval and k-NN, f16 is the default and q8_0 costs nothing measurable — both land within one clip of PyTorch on 405 clips, at 0.53x the weights for q8_0. Use f32 only when you need bit-level agreement with a PyTorch reference. For the classification head, use f16: q8_0 moves 6 of 105 top-1 decisions.

## Reproduce

```bash
export PATH=$HOME/.local/bin:$PATH
git submodule update --init ggml
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32 --target jepa-embed jepa-classify jepa-info
# batch driver (out-of-tree; see the note below)
g++ -O2 -std=c++17 -I include -I src -I third_party -I ggml/include \
    scripts/jepa_embed_clips.cpp build/libjepa.a -L build/ggml/src \
    -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build/ggml/src -o build/jepa-embed-clips

PY=.venv/bin/python                     # torch 2.13 CPU, transformers 5.16, av, numpy
export HF_HOME=$PWD/tmp/hf-home TORCH_HOME=$PWD/tmp OMP_NUM_THREADS=32

$PY scripts/video_frames.py --data data/ucf101-subset/UCF101_subset \
      --out tmp/frames --frames 16 --jobs 32
$PY scripts/bench_accuracy_video.py lists --index tmp/frames/index.json
for m in vjepa2-vitl-fpc64-256 vjepa2_1-vitb-384; do
  $PY scripts/bench_accuracy_video.py torch --model $m
done
$PY scripts/bench_accuracy_video.py cpp --model vjepa2-vitl-fpc64-256 --dtype f16
$PY scripts/bench_accuracy_video.py cpp --model vjepa2-vitl-fpc64-256 --dtype q8_0
for d in f32 f16 q8_0; do
  $PY scripts/bench_accuracy_video.py cpp --model vjepa2_1-vitb-384 --dtype $d
done
$PY scripts/bench_accuracy_video.py ssv2-torch
$PY scripts/bench_accuracy_video.py ssv2-cpp --dtype f16
$PY scripts/bench_accuracy_video.py ssv2-cpp --dtype q8_0
$PY scripts/bench_accuracy_video.py report \
      --out-json tests/results/accuracy-video.json --out-md docs/accuracy-video.md
```

**Wall time at 32 threads**, measured: frame decode 4.8 s for all 405 clips (32 processes), then vjepa2-vitl-fpc64-256 torch 903 s, vjepa2-vitl-fpc64-256 f16 408 s, vjepa2-vitl-fpc64-256 q8_0 418 s, vjepa2_1-vitb-384 torch 548 s, vjepa2_1-vitb-384 f32 446 s, vjepa2_1-vitb-384 f16 475 s, vjepa2_1-vitb-384 q8_0 489 s, ssv2 f16 128 s, ssv2 q8_0 127 s, ssv2 pytorch 151 s — 68 min of compute in total, serial because the 32-thread budget is shared. `report` takes a few seconds.

Every stage writes into `tmp/accuracy-video/` (git-ignored) and can be re-run alone; `lists` fixes the clip order once in `tmp/accuracy-video/clips.json`, which every feature `.npy` is indexed by, and `tmp/frames/index.json` records the sampled frame indices per clip.

`build/jepa-embed-clips` exists because `tools/jepa-embed` takes one `--frames-npy` per process: a 405-clip sweep would re-load the weights 405 times and write 405 one-row `.npy` files. The driver links `libjepa` directly and does exactly what `jepa-embed --frames-npy F --pool mean` does per clip — same `jepa_preprocess_frames_rgb`, `jepa_encode`, `jepa_pool_mean` — verified bit-for-bit on `v_Archery_g01_c04` (both print `|x| = 48.7494`, first four components `-0.18624, -0.27707, 0.20074, -0.71871`). What `jepa-embed` is missing for this job is a `--frames-list`/multi-clip mode with one `[n_clips, D]` output.
