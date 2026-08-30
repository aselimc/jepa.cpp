# ggml notes: flash attention & ops for the video graph (CPU backend)

Everything here was measured with `tests/test-attn.cpp` on this box (AMD Threadripper PRO 7995WX, 96 cores /
192 threads, AVX-512 + avx512_bf16, 250 GB RAM) against ggml `36da5713` (v0.22.0), Release, `-march=native`,
OpenMP threading, `GGML_LLAMAFILE=OFF` (ggml's default; see §6). Re-run with

```bash
cmake --build build -j 32 --target test-attn
./build/test-attn -t 32                      # attention matrix + mask semantics (exit 1 on regression), ~8 min
./build/test-attn -t 32 --n 256,2048          # the quick version registered with ctest ("attn")
./build/test-attn bench -t 32,96              # matmul GFLOP/s + norm/gelu accuracy
```

All error numbers are against a double-precision softmax(q kᵀ·scale)·v computed on the host from the *same F32*
q/k/v (so "F16 K/V error" is the error of rounding K/V to F16 plus the kernel's own arithmetic).

## 1. What `ggml_flash_attn_ext` wants (read from `ggml.h` + `ggml-cpu/ops.cpp` at this commit)

```
q:    [head_dim, N_q,  n_head,    1]   F32 (the CPU kernel asserts q->type == F32 for the fast paths)
k:    [head_dim, N_kv, n_head_kv, 1]   F16 or F32 (also quantized types, slow path)
v:    [head_dim, N_kv, n_head_kv, 1]   same type as k  — NOT transposed (unlike the naive mul_mat path)
mask: [N_kv,     N_q,  1, 1]           F16, contiguous, values 0 (attend) / -INF (blocked); optional (NULL)
res:  [head_dim, n_head, N_q, 1]       F32, contiguous  — heads are the middle dim, i.e. already "[N, H*D]"
```

* Only `ne0` (the head_dim row) has to be contiguous: `nb0 == type_size`. The strides over N and heads are free,
  so **q can be a permuted view straight out of the fused qkv projection, no `ggml_cont`**. Same for F32 K/V.
* `ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)`: `scale = 1/sqrt(head_dim)`,
  `max_bias = 0` (ALiBi off), `logit_softcap = 0`.
* **Mask**: F16 only (`GGML_ASSERT(mask->type == GGML_TYPE_F16)`), contiguous, row `i` (= `mask + i*nb[1]`) holds the
  N_kv entries for query `i`; `ne[2]` may be 1 (broadcast over heads: `q->ne[2] % mask->ne[2] == 0`).
  The value is *added* to the scaled score; entries equal to `-INFINITY` are skipped entirely. **There is no
  `GGML_KQ_MASK_PAD` any more at this commit** (older ggml required `mask->ne[1]` padded to a multiple of 32/64);
  `mask->ne[1] == N_q` is fine, the kernels only read rows `< N_q`. Build it with
  `ggml_fp32_to_fp16(0.0f)` / `ggml_fp32_to_fp16(-INFINITY)` (F16 has a proper -inf).
* A **fully masked query row yields an all-zero output row** (online-softmax sum S == 0 → 0), not NaN.
  The naive `ggml_soft_max_ext` path returns NaN for such a row (verified).
* **`ggml_flash_attn_ext_set_prec(t, GGML_PREC_F32)` is a no-op on the CPU backend**: `GGML_PREC_DEFAULT` and
  `GGML_PREC_F32` dispatch to the same function (`ggml_compute_forward_flash_attn_ext`, ops.cpp). Set it anyway so
  the graph does the right thing on Metal/CUDA later (there it selects F32 accumulation).
* Output layout `[D, H, N]` is exactly what the output projection wants: `ggml_reshape_2d(res, D*H, N)` then
  `ggml_mul_mat(attn_out_w, ...)` — no permute/cont.

### Which kernel runs (ops.cpp, `ggml_compute_forward_flash_attn_ext_f16`)

| condition | kernel | precision |
|---|---|---|
| `N_q >= 64` (GGML_FA_TILE_Q), q F32, K/V both F16 or both F32, `head_dim % 16 == 0` (AVX-512 `GGML_F32_EPR`; `% 8` on AVX2) | **tiled** 64×64 GEMM path | Q stays F32, K/V tiles are converted to F32, scores/softmax/accumulator all F32 |
| otherwise (`N_q < 64`, quantized K/V, head_dim not a multiple of 16) | per-row `one_chunk` path | K F16 ⇒ **q is rounded to F16** for the dot; V F16 ⇒ **the PV accumulator is F16** (`ggml_vec_mad_f16`) |
| `N_q == 1` and `N_kv >= 512` | split-KV decode path | F32 |

head_dim 32 / 64 / 80 all hit the tiled path on this box (all multiples of 16). Consequence for the numbers below:
with the tiled kernel the F16-K/V error is purely the storage rounding of K and V (~5e-4 abs, 2.5e-4..1.3e-3 relative
to max|out|, cosine ≥ 0.9999996); the `one_chunk` path (tiny N, e.g. the LeWM predictor with N = 3 frames) is ~3× worse
(2e-3 abs) because of the F16 q and F16 accumulator — use **F32 K/V there**, it is free at that size.

## 2. Recipe for the video graph (copy this)

```cpp
// x: [C, N] F32 (C = D*H), qkv_w: [C, 3C], qkv_b: [3C]; rows of the fused projection are [q_0..q_{H-1} | k_.. | v_..]
struct ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, qkv_w, x), qkv_b);          // [3C, N] F32
const size_t es = ggml_element_size(qkv);                                                // 4
// [D, H, N] views: nb1 = D*es (next head), nb2 = qkv->nb[1] (next token), offset = 0 / C / 2C elements
struct ggml_tensor * q = ggml_view_3d(ctx, qkv, D, H, N, D*es, qkv->nb[1], 0*C*es);
struct ggml_tensor * k = ggml_view_3d(ctx, qkv, D, H, N, D*es, qkv->nb[1], 1*C*es);
struct ggml_tensor * v = ggml_view_3d(ctx, qkv, D, H, N, D*es, qkv->nb[1], 2*C*es);
// (V-JEPA 2 / 2.1: q = jepa_rope3d_apply(ctx, q, cos, sin); k = jepa_rope3d_apply(ctx, k, cos, sin); — returns
//  contiguous [D, H, N] F32, the permute below still applies)
q = ggml_permute(ctx, q, 0, 2, 1, 3);                                                    // [D, N, H] strided view — OK as is
k = ggml_permute(ctx, k, 0, 2, 1, 3);
v = ggml_permute(ctx, v, 0, 2, 1, 3);
if (kv_f16) {                       // optional: halves K/V bytes, costs one F32->F16 pass per layer, adds ~5e-4 error
    k = ggml_cast(ctx, k, GGML_TYPE_F16);     // ggml_cast of a permuted view = contiguous [D, N, H] F16 (generic dup)
    v = ggml_cast(ctx, v, GGML_TYPE_F16);
}                                   // else: F32 K/V views straight from qkv, no cast, no cont, exact (rel 2e-6)
struct ggml_tensor * a = ggml_flash_attn_ext(ctx, q, k, v, mask /*F16 [N, N] or NULL*/, 1.0f/sqrtf((float) D), 0.0f, 0.0f);
ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);                                          // no-op on CPU, right thing elsewhere
struct ggml_tensor * o = ggml_reshape_2d(ctx, a, C, N);                                  // [C, N] — a is contiguous [D, H, N]
o = ggml_add(ctx, ggml_mul_mat(ctx, out_w, o), out_b);
```

`tests/test-attn.cpp::run_flash_fused` is this exact recipe (with the fused tensor filled from the host) and is
checked against the double reference for every (D, H, N) in the matrix (`fused-f16kv` rows) and for the masked cases.

### Block-causal mask (V-JEPA 2-AC predictor, `jepa.pred.frame_causal`; LeWM predictor is the B = 1 special case)

Token `i` belongs to frame `i / B` (B tokens per frame, T-major order). Query `i` may attend key `j` iff
`frame(j) <= frame(i)`:

```cpp
// host side, once per (T, B); upload to a GGML_TYPE_F16 tensor of shape [N_kv = N, N_q = N]
std::vector<ggml_fp16_t> m((size_t) N * N);
const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
for (int i = 0; i < N; ++i)              // query (row of the mask, ne[1])
    for (int j = 0; j < N; ++j)          // key   (column, ne[0])
        m[(size_t) i * N + j] = (j / B) <= (i / B) ? zero : ninf;
struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, N);   // then ggml_backend_tensor_set(mask, m.data(), ...)
```

The same tensor works for the naive path (`ggml_soft_max_ext(kq, mask, scale, 0.0f)` accepts F16 or F32 masks with
`mask->ne[1] >= N_q`). Verified on both CPU kernels (N = 48 → one_chunk, N = 256 → tiled) against the double reference
with the same boolean mask, including a fully masked row (flash → zeros, naive → NaN). The tiled kernel skips whole
64-key tiles whose mask is all -INF (`can_skip` in ops.cpp), so a block-causal mask also *saves* flash time rather
than costing any.

## 3. Attention accuracy and speed (32 threads, best of 2 runs after a warm-up)

Full matrix: head_dim ∈ {32, 64, 80} × n_head ∈ {12, 16} × N ∈ {256, 2048, 4608, 8192, 18432} — 30 cases,
none skipped (biggest: 20.4 GB naive score matrix, 11.5 s double reference; budget was 60 GB / 120 s per case).
Production shapes: **N=2048** = ViT-L 16f@256 (D=64, H=16), **N=8192** = ViT-L 64f@256, **N=4608** = 2.1 ViT-B
16f@384 (D=64, H=12), **N=18432** = 2.1 ViT-B 64f@384, **N=256** = I-JEPA ViT-H @224 (D=80, H=16), D=32 = the
predictors. The box was shared with two other agents during the runs, so treat timings as ±10-15 %.

### Wall time (ms, 32 threads)

| head_dim | n_head | N | naive-f32 ms (GFLOP/s) | flash F32 K/V ms | flash F16 K/V ms | fused recipe ms (incl. casts) | speed-up vs naive |
|---|---|---|---|---|---|---|---|
| 32 | 12 | 256 | 0.6 (172) | 0.2 | 0.2 | 0.2 | 2.6x |
| 32 | 16 | 256 | 0.8 (174) | 0.2 | 0.2 | 0.2 | 3.9x |
| 64 | 12 | 256 | 0.6 (360) | 0.4 | 0.4 | 0.5 | 1.3x |
| 64 | 16 | 256 | 0.7 (370) | 0.4 | 0.3 | 0.4 | 2.1x |
| 80 | 12 | 256 | 0.7 (356) | 0.5 | 0.5 | 0.5 | 1.4x |
| 80 | 16 | 256 | 0.9 (360) | 0.4 | 0.4 | 0.5 | 2.3x |
| 32 | 12 | 2048 | 34.5 (187) | 3.6 | 3.8 | 3.9 | 9.2x |
| 32 | 16 | 2048 | 45.6 (188) | 5.0 | 5.0 | 5.5 | 9.0x |
| 64 | 12 | 2048 | 26.4 (488) | 7.4 | 7.3 | 7.7 | 3.6x |
| 64 | 16 | 2048 | 33.7 (509) | 9.7 | 9.7 | 10.1 | 3.5x |
| 80 | 12 | 2048 | 36.0 (448) | 8.6 | 8.6 | 8.9 | 4.2x |
| 80 | 16 | 2048 | 47.4 (453) | 11.3 | 11.3 | 11.6 | 4.2x |
| 32 | 12 | 4608 | 160.6 (203) | 19.4 | 20.0 | 20.3 | 8.0x |
| 32 | 16 | 4608 | 213.2 (204) | 29.7 | 30.6 | 31.1 | 7.0x |
| 64 | 12 | 4608 | 127.0 (514) | 38.6 | 38.7 | 39.2 | 3.3x |
| 64 | 16 | 4608 | 168.0 (518) | 49.1 | 49.0 | 49.6 | 3.4x |
| 80 | 12 | 4608 | 174.3 (468) | 45.7 | 51.1 | 46.6 | 3.4x |
| 80 | 16 | 4608 | 256.5 (424) | 61.4 | 67.2 | 76.1 | 3.8x |
| 32 | 12 | 8192 | 485.8 (212) | 59.4 | 61.3 | 62.5 | 7.9x |
| 32 | 16 | 8192 | 648.2 (212) | 79.5 | 90.5 | 82.3 | 7.2x |
| 64 | 12 | 8192 | 398.9 (517) | 117.0 | 119.1 | 119.7 | 3.4x |
| 64 | 16 | 8192 | 534.3 (514) | 157.9 | 184.1 | 160.4 | 2.9x |
| 80 | 12 | 8192 | 556.8 (463) | 139.9 | 168.3 | 143.4 | 3.3x |
| 80 | 16 | 8192 | 742.0 (463) | 186.9 | 188.2 | 225.3 | 3.9x |
| 32 | 12 | 18432 | 2465.1 (212) | 308.3 | 321.7 | 316.9 | 7.7x |
| 32 | 16 | 18432 | 3294.0 (211) | 428.2 | 461.2 | 426.1 | 7.1x |
| 64 | 12 | 18432 | 2073.1 (503) | 616.0 | 608.9 | 622.9 | 3.4x |
| 64 | 16 | 18432 | 2861.1 (486) | 882.4 | 916.1 | 891.3 | 3.1x |
| 80 | 12 | 18432 | 3058.5 (426) | 797.3 | 793.3 | 818.1 | 3.9x |
| 80 | 16 | 18432 | 4137.0 (420) | 1134.2 | 1096.4 | 1082.5 | 3.8x |

* flash runs at **1.5-1.9 TFLOP/s** regardless of N (attention flops = 4·N²·D·H); the naive path is stuck at
  0.2 TFLOP/s for D=32 and ~0.5 TFLOP/s for D=64/80 (batched F32 `ggml_mul_mat` + softmax + the `cont(transpose(v))`
  copy). Net speed-up 3-9×, biggest exactly where it matters (small head_dim / long N).
* **F16 vs F32 K/V makes no speed difference on CPU** (the tiled kernel up-converts K/V tiles to F32 either way);
  differences in the table are run-to-run noise. The fused-recipe column shows the two `ggml_cast`s cost ~1-4 %.
* The `fused` recipe was also re-run against a `GGML_LLAMAFILE=ON` build: flash timings are identical (llamafile only
  replaces `ggml_mul_mat`); the naive path improves to ~0.67 TFLOP/s (25.6 ms for D=64 H=16 N=2048) — still 2.5× slower
  than flash.
* Per-layer cost at the production shapes (flash, 32 threads): 16f@256 ViT-L **10 ms**, 64f@256 ViT-L **158 ms**,
  16f@384 ViT-B **39 ms**, 64f@384 ViT-B **616 ms** — × 24 (L) / 12 (B) layers of attention alone.

### Accuracy vs the double-precision reference

| head_dim | n_head | N | naive-f32 max abs / rel | flash F32 K/V max abs / rel / min cos | flash F16 K/V max abs / rel / min cos |
|---|---|---|---|---|---|
| 32 | 12 | 256 | 3.2e-07 / 4.1e-07 | 4.9e-07 / 6.2e-07 / 1.0000000 | 4.6e-04 / 5.9e-04 / 0.9999997 |
| 32 | 16 | 256 | 2.8e-07 / 3.5e-07 | 4.2e-07 / 5.4e-07 / 1.0000000 | 5.4e-04 / 6.9e-04 / 0.9999996 |
| 64 | 12 | 256 | 1.6e-07 / 2.4e-07 | 6.1e-07 / 9.5e-07 / 1.0000000 | 3.7e-04 / 5.8e-04 / 0.9999998 |
| 64 | 16 | 256 | 1.9e-07 / 2.1e-07 | 7.2e-07 / 8.2e-07 / 1.0000000 | 4.1e-04 / 4.6e-04 / 0.9999998 |
| 80 | 12 | 256 | 5.1e-07 / 5.4e-07 | 5.8e-07 / 6.3e-07 / 1.0000000 | 5.3e-04 / 5.7e-04 / 0.9999999 |
| 80 | 16 | 256 | 4.3e-07 / 5.1e-07 | 6.5e-07 / 7.8e-07 / 1.0000000 | 3.3e-04 / 4.0e-04 / 0.9999999 |
| 32 | 12 | 2048 | 1.5e-07 / 5.4e-07 | 5.0e-07 / 1.8e-06 / 1.0000000 | 2.2e-04 / 8.0e-04 / 0.9999997 |
| 32 | 16 | 2048 | 2.1e-07 / 5.7e-07 | 9.3e-07 / 2.6e-06 / 1.0000000 | 4.8e-04 / 1.3e-03 / 0.9999997 |
| 64 | 12 | 2048 | 9.4e-08 / 3.3e-07 | 5.6e-07 / 2.0e-06 / 1.0000000 | 2.1e-04 / 7.2e-04 / 0.9999998 |
| 64 | 16 | 2048 | 1.2e-07 / 4.4e-07 | 6.5e-07 / 2.4e-06 / 1.0000000 | 1.4e-04 / 5.1e-04 / 0.9999997 |
| 80 | 12 | 2048 | 2.4e-07 / 8.1e-07 | 7.4e-07 / 2.5e-06 / 1.0000000 | 1.4e-04 / 4.6e-04 / 0.9999998 |
| 80 | 16 | 2048 | 4.2e-07 / 6.7e-07 | 1.2e-06 / 1.9e-06 / 1.0000000 | 1.6e-04 / 2.5e-04 / 0.9999998 |
| 32 | 12 | 4608 | 3.3e-07 / 1.6e-06 | 4.6e-07 / 2.2e-06 / 1.0000000 | 1.1e-04 / 5.1e-04 / 0.9999996 |
| 32 | 16 | 4608 | 1.8e-07 / 7.2e-07 | 4.8e-07 / 1.9e-06 / 1.0000000 | 2.5e-04 / 1.0e-03 / 0.9999997 |
| 64 | 12 | 4608 | 8.5e-08 / 4.3e-07 | 4.9e-07 / 2.4e-06 / 1.0000000 | 1.3e-04 / 6.6e-04 / 0.9999998 |
| 64 | 16 | 4608 | 1.1e-07 / 5.0e-07 | 5.3e-07 / 2.4e-06 / 1.0000000 | 8.8e-05 / 4.1e-04 / 0.9999998 |
| 80 | 12 | 4608 | 2.0e-07 / 1.0e-06 | 5.7e-07 / 3.0e-06 / 1.0000000 | 7.5e-05 / 3.9e-04 / 0.9999998 |
| 80 | 16 | 4608 | 1.4e-07 / 8.5e-07 | 6.0e-07 / 3.7e-06 / 1.0000000 | 1.1e-04 / 6.9e-04 / 0.9999998 |
| 32 | 12 | 8192 | 1.0e-07 / 5.8e-07 | 7.7e-07 / 4.4e-06 / 1.0000000 | 1.7e-04 / 9.6e-04 / 0.9999996 |
| 32 | 16 | 8192 | 1.3e-07 / 9.1e-07 | 5.1e-07 / 3.7e-06 / 1.0000000 | 1.8e-04 / 1.3e-03 / 0.9999997 |
| 64 | 12 | 8192 | 1.0e-07 / 6.2e-07 | 6.0e-07 / 3.6e-06 / 1.0000000 | 6.8e-05 / 4.1e-04 / 0.9999997 |
| 64 | 16 | 8192 | 7.8e-08 / 6.6e-07 | 5.0e-07 / 4.2e-06 / 1.0000000 | 5.6e-05 / 4.7e-04 / 0.9999997 |
| 80 | 12 | 8192 | 1.1e-07 / 8.0e-07 | 4.6e-07 / 3.3e-06 / 1.0000000 | 7.2e-05 / 5.1e-04 / 0.9999998 |
| 80 | 16 | 8192 | 1.4e-07 / 7.8e-07 | 4.4e-07 / 2.4e-06 / 1.0000000 | 8.5e-05 / 4.6e-04 / 0.9999998 |
| 32 | 12 | 18432 | 1.6e-07 / 1.1e-06 | 8.2e-07 / 5.7e-06 / 1.0000000 | 1.9e-04 / 1.3e-03 / 0.9999996 |
| 32 | 16 | 18432 | 1.1e-07 / 8.6e-07 | 6.3e-07 / 4.7e-06 / 1.0000000 | 1.1e-04 / 7.9e-04 / 0.9999995 |
| 64 | 12 | 18432 | 8.1e-08 / 8.4e-07 | 4.4e-07 / 4.6e-06 / 1.0000000 | 7.1e-05 / 7.3e-04 / 0.9999998 |
| 64 | 16 | 18432 | 4.5e-08 / 5.4e-07 | 4.8e-07 / 5.8e-06 / 1.0000000 | 7.2e-05 / 8.7e-04 / 0.9999998 |
| 80 | 12 | 18432 | 7.1e-08 / 8.9e-07 | 4.3e-07 / 5.5e-06 / 1.0000000 | 5.4e-05 / 6.8e-04 / 0.9999998 |
| 80 | 16 | 18432 | 1.8e-07 / 1.4e-06 | 5.2e-07 / 4.2e-06 / 1.0000000 | 3.9e-05 / 3.2e-04 / 0.9999998 |

* naive-f32 and flash-f32kv are both exact to F32 round-off (≤ 1.4e-6 relative; flash's online softmax is slightly
  worse than the two-pass softmax but far below anything visible).
* flash-f16kv error is the F16 rounding of K/V: max abs ≤ 5.4e-4, relative ≤ 1.3e-3, per-row cosine ≥ **0.9999995**,
  independent of N. That is ~100× below the parity threshold for logits (cosine 0.9999) but *at* the max-abs ≤ 1e-3
  relative threshold of docs/architecture.md — fine for one attention, and empirically the f16 GGUFs (which also
  round the weights) still hit cosine ≥ 0.99999 end-to-end (selftest.py), so F16 K/V is safe. If a parity run ever
  lands marginal, switching K/V to F32 costs nothing on CPU (see above) and removes this term entirely.
* The block-causal mask cases (§2) pass on both kernels: flash vs double reference ≤ 1.5e-3 rel (F16 K/V, one_chunk
  worst case) / ≤ 6e-7 (F32 K/V); a fully masked query row returns exactly 0 from flash, NaN from the naive path.

## 4. Memory

`graph` = `ggml_gallocr` compute buffer (all intermediates; softmax reuses the score matrix in place),
`work` = CPU backend work buffer (`ggml_graph_plan(...).work_size`, per-thread flash tiles). Inputs (q/k/v/mask) not
included — identical for both paths. Selected rows (full data in the test output):

| head_dim | n_head | N | naive graph buffer (score matrix) | flash graph buffer | flash work buffer | fused recipe graph buffer (qkv views + F16 K/V) |
|---|---|---|---|---|---|---|
| 80 | 16 | 256 | 0.006 GB | 0.001 GB | 0.003 GB | 0.002 GB |
| 32 | 12 | 2048 | 0.193 GB | 0.003 GB | 0.002 GB | 0.006 GB |
| 32 | 16 | 2048 | 0.258 GB | 0.004 GB | 0.002 GB | 0.008 GB |
| 64 | 16 | 2048 | 0.266 GB | 0.008 GB | 0.003 GB | 0.016 GB |
| 64 | 12 | 4608 | 0.976 GB | 0.013 GB | 0.003 GB | 0.026 GB |
| 32 | 12 | 8192 | 3.023 GB | 0.012 GB | 0.002 GB | 0.023 GB |
| 64 | 16 | 8192 | 4.062 GB | 0.031 GB | 0.003 GB | 0.062 GB |
| 64 | 12 | 18432 | 15.293 GB | 0.053 GB | 0.003 GB | 0.105 GB |

Rule of thumb: naive needs `4·N²·H` bytes (the F32 score matrix — **20.4 GB** for 64f@384 ViT-B with 16 heads, and it
also pays a full extra `cont` of V); flash needs `2·(D·H·N)` bytes for the cast K/V copies plus a ~3 MB work buffer —
**~300× less** at N=18432. The `fused` column includes the F16 K/V copies the casts produce. With F32 K/V views straight
into the qkv tensor, flash allocates only the output (graph buffer halves again).

## 5. Matmul throughput of this box (what a ViT-L/H layer can expect)

`test-attn bench`: `Y = ggml_mul_mat(W, X)`, W = [4096×1024] (a ViT-L ffn_up / 4·1024 attention block),
X = [1024×N] F32, best of ≥3 runs; `max rel err` vs a double reference sampled over the first min(N, 64) output columns (Q8_0's ~8e-3 is the quantisation of W *and*
of X to Q8_0 for the dot products — Q8_0 activations are quantised on the fly by the CPU backend).

ggml default (`GGML_LLAMAFILE=OFF`, what `build/` currently produces), 32 threads:

| W type | N=256 | N=2048 | N=8192 | max rel err |
|---|---|---|---|---|
| F32  | 1381 GFLOP/s | 1355 | 1575 | 1.2e-07 |
| F16  | 1049 GFLOP/s | 1042 | 1260 | 3.0e-04 |
| Q8_0 | 1703 GFLOP/s | 1620 | 1952 | 7.6e-03 |

Same source, `-DGGML_LLAMAFILE=ON` (llamafile sgemm kernels for mul_mat; **off by default in stock ggml**, llama.cpp
turns it on):

| W type | N=256, 32t | N=2048, 32t | N=8192, 32t | N=256, 96t | N=2048, 96t | N=8192, 96t |
|---|---|---|---|---|---|---|
| F32  | 2204 | 2210 | 2181 | 4042 | 4452 | 3430 |
| F16  | 3327 | 3178 | 3350 | 5579 | 5929 | 4955 |
| Q8_0 | 4098 | 3664 | 3575 | 4074 | 3634 | 3865 |

* **Turn `GGML_LLAMAFILE` ON in the top-level CMakeLists** (`set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)` next to the
  other ggml options): identical f16/q8_0 numerics (bit-identical bench outputs; F32 differs only at round-off ~2e-7 — different kernels run), roughly **1.3–1.6× (F32) / 2.6–3.2× (F16) / 1.9–2.2× (Q8_0)** faster matmuls (the spread reflects background load between measurement runs; both ends re-measured),
  which dominate ViT inference. The realistic budget for this box is then **~3.3 TFLOP/s F16 at 32 threads,
  ~5-6 TFLOP/s at 96** (Q8_0 tops out ~4 TFLOP/s and stops scaling past 32 threads — activation-quantisation bound).
* Back-of-envelope with llamafile F16 at 32 threads: one ViT-L layer has ≈ 24·1024²·N matmul FLOPs (qkv 3C² + out C²
  + ffn 8C² MACs) → 64f@256 (N=8192) ≈ 206 GFLOP ≈ 62 ms of matmul + 158 ms of flash attention per layer — attention
  dominates the long-video encoder; for images (N=256, 6.4 GFLOP ≈ 2 ms vs 0.4 ms flash) the matmuls dominate.

Elementwise ops (32 threads, 1M elements, §6 for the `ggml_gelu` trap):

| op | max err vs double |
|---|---|
| `ggml_norm` (rows of 1024, eps 1e-6) | 5.2e-07 |
| `ggml_gelu_erf` (inputs in [-12,12]) | 4.5e-07 |
| `ggml_gelu` (tanh approx + F16 lookup) vs erf-GELU | 3.9e-03 |

## 6. Gotchas collected on the way

* **`ggml_flash_attn_ext` output is `[D, H, N]`, not `[D, N, H]`** — the doc comment says "!! permuted !!". Reshape to
  `[D*H, N]` directly; do not permute it back.
* **V is not transposed** for flash attention (`[D, N_kv, H]`, same layout as K). The naive path needs
  `ggml_cont(ggml_transpose(v))` → `[N_kv, D, H]` for the second `ggml_mul_mat` — that `cont` is an extra N×D×H copy per
  layer and the reason the naive path is even slower than its FLOP count suggests.
* **Mask dtype**: F16 only for `ggml_flash_attn_ext` (asserted in `ggml.c`), F16 or F32 for `ggml_soft_max_ext`. Use one
  F16 mask for both. It must be contiguous (`ggml_is_contiguous`), so build it on the host, not as a view.
* **No mask padding** at this ggml commit (`GGML_KQ_MASK_PAD` is gone; older llama.cpp code that does
  `GGML_PAD(n_tokens, GGML_KQ_MASK_PAD)` is obsolete). `mask->ne[1] == N_q` works.
* **Views**: `ggml_permute(view, 0, 2, 1, 3)` of a `[D, H, N]` slice of the fused qkv tensor is accepted directly by
  the kernel for q (and for F32 K/V) — the kernel only asserts `nb0 == type_size`. `ggml_cast(permuted_view, F16)`
  produces a *contiguous* `[D, N, H]` F16 tensor (generic strided `dup`), no `ggml_cont` needed; casting before the
  permute (`ggml_permute(ggml_cast(view))`) is equally valid and equally fast (§3 probe).
* **`ggml_flash_attn_ext_set_prec` does nothing on CPU** (both precisions run the F32-accumulator code), and the
  CPU kernel ignores `ggml_prec` for `ggml_mul_mat` as well. Set it for other backends.
* **head_dim must be a multiple of 16** (AVX-512) / 8 (AVX2) to get the tiled kernel — 32 / 64 / 80 are all fine.
  For N_q < 64 the slow per-row kernel is used, and with F16 K/V it rounds q to F16 and accumulates PV in F16.
* **Fully masked rows**: flash → zeros, `ggml_soft_max_ext` → NaN. If the AC predictor ever produces a query that
  attends nothing (it does not with a block-causal mask, the diagonal block is always allowed), only flash is safe.
* **`ggml_gelu` is the tanh approximation and runs through an F16 lookup table on CPU** (max error 3.9e-3 vs erf-GELU);
  all Meta JEPAs use `gelu_erf` → use **`ggml_gelu_erf`**, which is exact to 4.5e-7 (`erff` per element).
* **`ggml_norm` has no affine part**: `ggml_norm(x, eps)` then `ggml_mul(., w)` + `ggml_add(., b)`. Error 5e-7 on
  1024-wide rows (eps 1e-6).
* **Softmax memory is reused in place** by `ggml_gallocr` (SOFT_MAX is in `ggml_op_can_inplace`), so the naive path
  costs one N×N×H F32 score matrix, not two — still 15.3 GB for the 64f@384 ViT-B case (12 heads) vs 0.05 GB for flash.
* **Threads**: `ggml_backend_cpu_set_n_threads` applies to the next `ggml_backend_graph_compute`; the flash kernel is
  parallelised over query rows in chunks of 4×n_threads (`ggml_threadpool_chunk_add`), so it scales fine to 96 threads
  but the `ref`/`naive` numbers above are 32-thread numbers.
* `ggml_graph_plan(gf, n_threads, NULL).work_size` gives the CPU work buffer the backend will allocate internally —
  3 MB for flash (per-thread 64×64 tiles), essentially zero for the naive path.
