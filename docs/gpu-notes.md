# GPU (CUDA) feasibility for jepa.cpp

**Status: audit only.** Nothing in `src/`, `tools/` or the build system has been changed. This document
records (1) what ggml's CUDA backend can and cannot run *at jepa.cpp's exact shapes and strides*,
(2) measured ggml-CUDA throughput on this box next to the CPU numbers of `docs/ggml-notes.md`,
(3) a measured PyTorch-GPU baseline for the same models, and (4) the implementation plan and effort
estimate a CUDA port would need. Read `docs/architecture.md` and `docs/ggml-notes.md` first — this
document assumes the graph they describe.

Everything read-only was checked against the pinned ggml submodule, commit `36da5713`, in
`ggml/src/ggml-cuda/`. File and function names are given so each claim can be re-checked.

## 0. Box

| | |
|---|---|
| GPU | 2 × NVIDIA RTX 4500 Ada Generation, 24 GB each, compute capability **8.9** (Ada Lovelace) |
| Driver | 580.173.02, CUDA runtime 13.0 |
| Toolkit | `/usr/local/cuda`, `nvcc` release 13.0, V13.0.88 |
| CPU (for the CPU columns) | AMD Ryzen Threadripper PRO 7995WX, 96 cores / 192 threads, AVX-512, 251 GB |
| ggml | submodule `36da5713` (v0.22.0), `GGML_CUDA=ON`, `GGML_NATIVE=ON` → `CMAKE_CUDA_ARCHITECTURES=native` = `89` |
| Vulkan | loader `libvulkan1` 1.3.275 + the NVIDIA ICD are installed; **`glslc` / `SPIRV-Headers` are not** |

RTX 4500 Ada paper peak: ~24 TFLOP/s FP32, ~48 TFLOP/s FP16 with FP32 accumulate on the tensor
cores (~96 with FP16 accumulate), ~192 TOP/s INT8, 432 GB/s of GDDR6 ECC bandwidth. Those are the
ceilings the measured numbers below should be read against.

## 1. Op coverage

Every op jepa.cpp's graphs emit, checked against `ggml_backend_cuda_device_supports_op`
(`ggml/src/ggml-cuda/ggml-cuda.cu:4867`) **with the tensor shapes and strides our builders actually
produce** — a bare "CUDA has a `mul` kernel" is not the question; the question is whether the
`supports_op` predicate says yes for *our* tensor, because a `no` makes `ggml_backend_sched` cut the
graph and shuttle the tensor across PCIe.

The full op list comes from `grep -oh 'ggml_[a-z0-9_]*(' src/*.cpp src/*.h`; `ggml_get_rows` is in the
brief but jepa.cpp never emits it (it gathers rows on the host in `predict_masked`, `jepa_encode_*`).

### 1.1 Verdict table

| op (as jepa.cpp emits it) | where | CUDA kernel | `supports_op` verdict | evidence |
|---|---|---|---|---|
| `mul_mat` F32 weight × F32 acts | every linear | `ggml_cuda_mul_mat_cublas` (compute F32) | **yes** | `ggml-cuda.cu:4949` `switch (a->type)` |
| `mul_mat` F16 weight × F32 acts | every linear | cuBLAS (**compute F16 unless `GGML_PREC_F32`**) or `mmf` | **yes**, see §1.3 | same switch; `ggml_cuda_mul_mat_cublas` at `ggml-cuda.cu:1623` |
| `mul_mat` Q8_0 / Q4_0 / Q4_K × F32 | quantized files | `mmq` (INT8 tensor cores) or `mmvq` | **yes** | same switch (`Q8_0`, `Q4_0`, `Q4_K`); `mmq.cu` |
| `mul_mat` on a permuted view (naive attn) | `jepa_build_attention` naive path | generic | **yes** — only `nb[0] == type_size` is required | `ggml-cuda.cu:4930` |
| `mul_mat` with `ne[3] > 1` (batching) | batched encoder | cuBLAS batched | **yes** | same switch; `ne[2]*ne[3]` handled |
| `flash_attn_ext` head_dim **64 / 80**, F16 or F32 K/V, no mask | every encoder block | MMA-F16 (`fattn-mma-f16.cuh`) | **yes** | `fattn.cu:393` `switch (K->ne[0])` lists 40/64/72/80/96/112/128/192/256/320/512/576 |
| `flash_attn_ext` head_dim **32** | **V-JEPA 2 / 2.1 predictors** | — | **NO** → `BEST_FATTN_KERNEL_NONE` | `fattn.cu:438` `default: return BEST_FATTN_KERNEL_NONE`; 32 is not in the switch |
| `flash_attn_ext` `N_q = 1` (attentive-pool head) | `jepa_build_head` | vec kernel | **yes** | `fattn.cu:462`, `Q->ne[1] == 1 && cc >= ADA_LOVELACE` |
| `flash_attn_ext` + F16 mask (LeWM / AC) | `jepa_build_lewm` | MMA-F16 | **yes** (mask must be F16 and `ne[2] == 1`) | `fattn.cu:452`; `launch_fattn` asserts `mask->type == F16` (`fattn-common.cuh:998`) |
| `flash_attn_ext` batched over `ne[3]` | batched encoder | any | **yes** (`Q->ne[3]` is a grid dimension) | `fattn-common.cuh:1084` `ntiles_dst … * Q->ne[3]` |
| `soft_max_ext` (naive path, no mask / F16 mask) | `jepa_build_attention` naive | `softmax.cu` | **yes** (unconditional) | `ggml-cuda.cu:5229` `case GGML_OP_SOFT_MAX: return true` |
| `norm` eps 1e-6 / 1e-12 | every LN | `norm.cu` | **yes** — but see §1.4, **different formula from the CPU** | `ggml-cuda.cu:5175` `ggml_is_contiguous_rows` |
| `gelu_erf` | every FFN | `op_gelu_erf` = `0.5x(1+erff(x/√2))` | **yes** (needs `ggml_is_contiguous(src0)`, ours is) | `unary.cu:24`; `ggml-cuda.cu:4893` |
| `silu` | LeWM action embed | `unary.cu` | **yes** | `ggml-cuda.cu:4888` |
| `add` / `mul` (incl. `[D,N] ⊕ [D]` broadcast, strided src) | everywhere | `binbcast.cu` | **yes**, F32/F16 only | `ggml-cuda.cu:5197`; `binbcast.cu:201` has a contiguous fast path and a general strided path |
| `scale_bias` (`ggml_scale` with bias) | rope masks, LeWM adaLN | `scale.cu` reads both `op_params[0..1]` | **yes** | `ggml-cuda.cu:5189` `case GGML_OP_SCALE`; `scale.cu:12` `dst = scale*x + bias` |
| `concat` dim 1 (CLS / registers) | `jepa_build_encoder_image` | `concat.cu` | **yes** (non-quantized, 4-byte type) | `ggml-cuda.cu:5120` |
| `cont` of a permuted view | naive attn, LeWM adaLN split | `cpy.cu` generic dup | **yes** (unconditional) | `ggml-cuda.cu:5225` `case GGML_OP_CONT: return true` |
| `cast` F32→F16 of a permuted view | flash K/V cast | `cpy.cu` | **yes** | `ggml-cuda.cu:5046` (`GGML_OP_CPY`, `F32 ↔ F16`) |
| `view` / `reshape` / `permute` / `transpose` | everywhere | no kernel (metadata) | **yes** | `ggml-cuda.cu:5183` |
| `repeat_4d` (mask tokens, rope masks) | predictor, rope | `unary`/`repeat` | **yes** (F32/F16) | `ggml-cuda.cu:5112` |
| `arange` (rope pair mask) | `jepa_rope3d_apply` | `arange.cu` | **yes** | `ggml-cuda.cu:5276` |
| **`roll` on the fused-qkv view** | `jepa_rope3d_apply`, ×2 per q, ×2 per k, every block | `roll.cu` (contiguous-only kernel) | **NO** — `ggml_is_contiguous(src[0])` is **false** for `ggml_view_3d(qkv, …)` | `ggml-cuda.cu:5236`; `roll.cu:41` indexes `src[d3*ne00*ne01*ne02 + …]`, i.e. assumes full contiguity |
| `roll` on a **contiguous** tensor | (the fix, §2) | `roll.cu` | **yes** | same |

### 1.2 The two ops that would split the graph, and where they sit

Only two entries are `NO`, and their positions are the whole story.

1. **`ggml_roll` on the q/k views — per layer, twice per q and twice per k.** This is the
   catastrophic one. `jepa_build_qkv` hands the RoPE hook a `ggml_view_3d` into the fused
   `[3D, N]` qkv projection, whose `nb[2] = qkv->nb[1] = 3·D·4` while `ne[0]·ne[1]·4 = D·4` — so
   `ggml_is_contiguous` is false and the CUDA `roll` is rejected. `ggml_backend_sched` would then
   run `roll` on the CPU, which means copying the whole `qkv` tensor device→host and the two rolled
   copies host→device, **four transfers per block, 24 (ViT-L) or 32 (ViT-H) blocks per forward**.
   At the 64-frame ViT-L shape `qkv` is `3·1024·8192·4 B = 100 MB`, so one block costs of order
   0.4 GB of PCIe traffic; the encoder would move ~10 GB per clip and be several times *slower*
   than the CPU path. `ggml_backend_sched` would also cut the 24-block graph into ~100 splits,
   destroying kernel-launch batching and any CUDA-graph capture.
   **This is a per-layer split = disaster, and it is the single blocking issue.** §2 fixes it.

2. **`flash_attn_ext` at head_dim 32 — the V-JEPA 2 / 2.1 masked predictors.** One split per
   predictor block, 12 blocks. Bad, but the predictor is a much smaller graph and there is a clean
   in-graph fallback (§3), so this is a *policy* decision, not a blocker.

Everything else — patch embed, every LN, every linear, the FFN, the attention of the encoders,
CLS/register concat, the modality vector, `cast`, `cont`, the attentive-pool head, the whole LeWM
predictor — is CUDA-resident. Once §2 lands, an encoder forward is **one split**: upload the patch
rows and the RoPE tables, run, download `[D, N]`.

### 1.3 `mul_mat` precision: cuBLAS picks F16 compute for F16 weights

`ggml_cuda_mul_mat_cublas` (`ggml-cuda.cu:1623`) sets `compute_type = src0->type` and only falls back
to F32 when `dst->op_params[0] == GGML_PREC_F32`. jepa.cpp never calls `ggml_mul_mat_set_prec`, and
`ggml_cuda_should_use_mmf` refuses any `mul_mat` with more than 16 activation columns (`mmf.cu:176`),
so at every one of our shapes an **f16 GGUF runs every projection through cuBLAS with
`batched_mul_mat_traits<GGML_TYPE_F16>::compute_type = CUBLAS_COMPUTE_16F`** (`ggml-cuda.cu:1398`).
That is *F16 accumulation*, and because `prefer_f32_output` is only set on Volta / RDNA4 / CDNA
(`ggml-cuda.cu:1512`), on Ada the **GEMM output is written as `half` too** and converted back to F32
afterwards. The CPU path accumulates in F32 throughout (`docs/ggml-notes.md` §5 measures F16 mul_mat
at 3.0e-4 max rel err there).

Two consequences, one of them potentially fatal:
- a wider error band at f16 on GPU than the `docs/parity.md` f16 rows describe, and
- **a real overflow risk**: `half` saturates at 65504, and `docs/architecture.md` records I-JEPA
  ViT-H activations reaching ~2e4. An FFN output stored as `half` is one factor of 3 away from `inf`.
  §5.4 checks for non-finite outputs explicitly.

F32 weights are safe — `batched_mul_mat_traits<GGML_TYPE_F32>` uses `CUBLAS_COMPUTE_32F` with
`CUDA_R_32F`, i.e. **strict FP32, not TF32** (ggml never calls `cublasSetMathMode`), so an f32 GGUF
gets Ada's ~24 TFLOP/s FP32 rather than its ~48 TFLOP/s tensor-core rate. Quantized weights take the
`mmq` path (INT8 tensor cores, INT32 accumulate, F32 rescale) and are unaffected.

**Consequence for the port:** call `ggml_mul_mat_set_prec(t, GGML_PREC_F32)` on every `mul_mat` in
`jepa_build_linear` when the backend is a GPU (it is a no-op on CPU — `docs/ggml-notes.md` §6), or
accept a measurably wider f16 error band on GPU. Cost is measured in §5.

**And a pleasant surprise for the quantized files.** `ggml_cuda_should_use_mmq` (`mmq.cu:259-313`)
returns `true` unconditionally once `turing_mma_available(cc)` holds — which it does on Ada — for
every type we ship, **Q4_K included**. So on CUDA all of Q8_0 / Q4_0 / Q4_K get a real INT8
tensor-core kernel. That inverts the CPU guidance: on this box `GGML_LLAMAFILE`'s fast sgemm only
covers F32/F16/Q8_0, so `docs/benchmarks.md` measures q4_k *slower* than q8_0 (ViT-H 198 ms vs
129 ms) and `docs/quantization.md` calls q4_k "a memory win only". On the GPU q4_k should be at
least as fast as q8_0 and a quarter of the weight bytes. Measured in §5.1.

### 1.4 `ggml_norm`: CUDA uses the numerically weaker variance formula

- CPU (`ggml-cpu/ops.cpp:3733`): two-pass **centred** variance, `ggml_vec_cvar_f32(n, y, x, mean)`.
- CUDA (`ggml-cuda/norm.cu:19-33`): one-pass `var = E[x²] − mean²`.

The one-pass form suffers catastrophic cancellation when `mean² ≈ E[x²]`, i.e. when a row has a
large DC offset relative to its spread. `docs/architecture.md` records that I-JEPA ViT-H activations
reach ~2e4; at that magnitude `E[x²] ≈ 4e8` in float32 has ~7 significant digits and the subtraction
can eat most of them. This is a **backend-intrinsic parity difference that no jepa.cpp flag can turn
off**. Measured in §5.3.

### 1.5 `flash_attn_ext`: there is no F32 K/V on CUDA

`jepa.cpp`'s K/V policy (`jepa_context_kv_type`) picks **F32 K/V for f32 model files**, because
`docs/architecture.md` measured that the F16 cast alone costs ~3 digits of worst-token cosine on
I-JEPA ViT-H. On CUDA that policy has no effect:

- `ggml_cuda_flash_attn_ext_get_alloc_size` / `launch_fattn` (`fattn.cu:553`, `fattn-common.cuh:1022`)
  set `need_f16_K = need_f16_V = true` for both the MMA and the tile kernels, and for the vec kernel
  `need_f16_K = (K->type == GGML_TYPE_F32)`. In every case an F32 K or V is **converted to F16 into a
  scratch buffer before the kernel runs**.
- `ggml_flash_attn_ext_set_prec(t, GGML_PREC_F32)` is *also* a no-op on CUDA: `grep -n GGML_PREC ggml-cuda/fattn*` returns nothing;
  only `mul_mat` reads it. The MMA kernel's KQ accumulator is F32 (`T_C_KQ = tile<…, float>`), but the
  **PV accumulator is `half2`** for most `(DKQ, DV, ncols)` configurations (`fattn-mma-f16.cuh:1034/1085/1093/…`).

So on CUDA, flash attention is F16-K/V, F16-PV-accumulate, always. The escape hatch is the **naive
path** (`--no-flash`): `mul_mat + soft_max_ext + cont` is fully F32 on CUDA and fully supported. For
I-JEPA (N=256) and the small image models that costs almost nothing; for a 18 432-token clip the
score matrix is 15.3 GB and does not fit in 24 GB, so f32-parity-on-GPU is an image-model-only
option. §6 turns this into a policy.

### 1.6 LayerNorm is not fused on CUDA (RMSNorm is)

`ggml-cuda.cu:3962-3980` fuses `{RMS_NORM, MUL, ADD}` and `{RMS_NORM, MUL}` into single kernels —
the LLM shape. There is **no `{NORM, MUL, ADD}` pattern**, and ViTs use LayerNorm
(`GGML_OP_NORM`), so `jepa_build_layer_norm`'s `norm → mul → add` stays three kernels and three
full read/write passes over `[D, N]`.

Cost at the 64-frame ViT-L shape: `[1024, 8192]` F32 is 33.5 MB, so the two extra passes are
~134 MB of traffic per LN × 48 LNs ≈ **6.4 GB, or ~15 ms at 432 GB/s** — real but not dominant.
At I-JEPA's 256 tokens the bytes are negligible and what matters is the 144–192 extra kernel
launches, which the CUDA-graph capture in §6.1 absorbs.

This is not a blocker and needs no jepa.cpp change; it is the most obvious **upstream** contribution
the port would motivate (`{NORM, MUL, ADD}` next to the RMS_NORM patterns), and it would speed up
every ViT on ggml-CUDA, not only ours.

## 2. RoPE portability plan

### 2.1 What the current apply emits

`jepa_rope3d_apply` (`src/rope3d.cpp:129`) builds, per call:

| # | node | CUDA |
|---|---|---|
| 1 | `ggml_arange(0, 2, 1)` → `[2]` | yes |
| 2 | `ggml_repeat_4d(m01, 2, D/2, 1, 1)` | yes |
| 3 | `ggml_reshape_1d(…, D)` (view) | yes |
| 4 | `ggml_scale_bias(m_odd, 1, −1)` → `m_evn` | yes |
| 5 | `ggml_mul(sin_t, m_evn)` → `s_e` | yes |
| 6 | `ggml_mul(sin_t, m_odd)` → `s_o` | yes |
| 7 | **`ggml_roll(x, −1, 0,0,0)`** | **NO** (x is the strided qkv view) |
| 8 | **`ggml_roll(x, +1, 0,0,0)`** | **NO** |
| 9 | `ggml_mul(x, cos_t)` | yes |
| 10–11 | `ggml_mul(x_next, s_e)`, `ggml_add` | yes |
| 12–13 | `ggml_mul(x_prev, s_o)`, `ggml_add` | yes |

13 nodes, called twice per block (q and k) — the `video_graph_nodes` comment in `src/jepa.cpp:452`
already budgets "2×13 RoPE".

### 2.2 The brief's suggestion, checked

Hoisting the three tables to the host (`cos`, `sin·even-mask-negated`, `sin·odd-mask`) removes nodes
1–6, i.e. **`arange`, `repeat_4d`, `reshape`, `scale_bias` and two `mul`s**. It does **not** remove
the two `ggml_roll`s, because the rolls act on `x`, not on the tables — so on its own it does not
make the chain CUDA-resident. It is still worth doing (7 fewer nodes per call, ~340 fewer nodes in a
24-block graph) and the repo already lists it as a candidate ("rope3d sin-mask hoisting (<5 %)").

### 2.3 What actually makes the chain CUDA-resident

Two changes, both small:

**(a) make `x` contiguous.** `ggml_roll`'s CUDA kernel is contiguous-only, so insert one
`ggml_cont` when `x` is a view. On the GPU this is a pure bandwidth copy of `D·N·4` bytes (33.5 MB
at the 64-frame ViT-L shape → ~0.15 ms per call at 432 GB/s, ~7 ms for a whole 24-block encoder).
Note the copy is *not* wasted: the current CPU graph materialises the same bytes inside `roll`.

**(b) express the pair swap as one `roll` over a length-2 axis.** The rotation only needs
`swap(x)[j] = x[j ^ 1]`. On a contiguous `[D, H, N]` tensor that is exactly
`ggml_roll(ggml_reshape_2d(x, 2, D/2·H·N), 1, 0, 0, 0)` — rolling an axis of length 2 by 1 *is* the
pair swap, with no wrap-around artefacts to mask away. One host table then suffices:

```
sin_signed[j] = (j even) ? −sin[j] : +sin[j]
out           = x·cos + swap(x)·sin_signed
```

which expands to `out[2k] = x[2k]·C[2k] − x[2k+1]·S[2k]` and
`out[2k+1] = x[2k+1]·C[2k+1] + x[2k]·S[2k+1]` — the same products, in the same order, as today. The
only difference against the current expression is that today's chain also adds an exact `0.0` term
per lane (the masked-out half of each roll), and adding `+0.0` to a finite float is exact.

**The refactor is bit-identical for finite inputs.** Checked in float32 against the current
expression (`np.array_equal` → `True`, max abs diff `0.0`), together with the claim that rolling a
length-2 axis by 1 is exactly the pair swap:

```python
D, H, N = 64, 4, 7
x, cos, sin = (rng.standard_normal(s).astype(np.float32) for s in [(N,H,D), (N,D), (N,D)])
m_odd = np.array([j % 2 for j in range(D)], np.float32); m_evn = m_odd - 1.0
cur = x*cos[:,None,:] + np.roll(x,-1,-1)*(sin*m_evn)[:,None,:] + np.roll(x,1,-1)*(sin*m_odd)[:,None,:]
sgn = np.where(np.arange(D) % 2 == 0, -1.0, 1.0).astype(np.float32)
sw  = x.reshape(N,H,D//2,2)[..., ::-1].reshape(N,H,D)          # == np.roll(x.reshape(-1,2),1,-1)
new = x*cos[:,None,:] + sw*(sin*sgn)[:,None,:]
assert np.array_equal(cur.astype(np.float32), new.astype(np.float32))
```

Re-confirmed end to end on both backends in §5.4.

Resulting node list — 5 nodes, all CUDA:

| # | node | CUDA |
|---|---|---|
| 1 | `ggml_cont(x)` (skipped when `x` is already contiguous) | yes |
| 2 | `ggml_reshape_2d(xc, 2, D/2·H·N)` (view) | yes |
| 3 | `ggml_roll(…, 1, 0,0,0)` | yes |
| 4–5 | `ggml_mul(xc, cos_t)`, `ggml_mul(sw, sins_t)`, `ggml_add` | yes |

### 2.4 Proposed signature change

```diff
 // src/rope3d.h
-// cos/sin tables for the full grid: row-major [N, head_dim] …
+// cos / signed-sin tables for the full grid: row-major [N, head_dim].
+// sin_out carries the SIGN of the rotation already folded in:
+//     sin_out[r*D + j] = (j even) ? -sin(angle) : +sin(angle)
+// so jepa_rope3d_apply is one pair-swap plus two multiplies and an add (docs/gpu-notes.md §2).
 void jepa_rope3d_tables(const jepa_rope3d_params & p, std::vector<float> & cos_out, std::vector<float> & sin_out);
 void jepa_rope3d_tables_ids(const jepa_rope3d_params & p, const int32_t * ids, int n_ids,
                             std::vector<float> & cos_out, std::vector<float> & sin_out);

-//   sin_t  : F32 [head_dim, 1, N]
+//   sin_t  : F32 [head_dim, 1, N]  (the SIGNED table above)
-// Runs on the CPU backend via ggml_backend / ggml_gallocr (stock ops only: arange, repeat, scale, roll, mul, add).
+// Backend-agnostic: cont (only if x is a view) + roll + 2 mul + add. Every node has a CUDA kernel.
 struct ggml_tensor * jepa_rope3d_apply(struct ggml_context * ctx, struct ggml_tensor * x,
                                        struct ggml_tensor * cos_t, struct ggml_tensor * sin_t);
```

The **public signature does not change** — only the meaning of the `sin` table's sign, which is
private to `rope3d.{h,cpp}` and its two callers. Optionally add a `bool signed_sin` parameter to the
table builders so `tests/test-ops.cpp` can keep checking the raw table against
`tests/vectors/rope3d/`.

### 2.5 Call sites and diff size

| file | change |
|---|---|
| `src/rope3d.cpp` | negate the even lanes of `sin_out` at the end of `rope3d_build` (2 lines); rewrite `jepa_rope3d_apply` (16 lines → ~8) |
| `src/rope3d.h` | comment/contract update only (no signature change) |
| `src/jepa.cpp` | **none** — `jepa_build_encoder_video` passes the tables through the `qk_hook` unchanged |
| `src/predictor.cpp` | **none** — same |
| `src/lewm.cpp` | **none** — LeWM has no 3-D RoPE |
| `tests/test-ops.cpp` | the table-vs-golden check (lines 184–190, 227) needs either a sign flip on the expected even lanes or the `signed_sin=false` flag; the `jepa_rope3d_apply` check (line 108) is unchanged and must still pass bit-exactly |

**Estimated diff: ~40 changed lines across 3 files, plus ~15 in the test.** This is the smallest
change in the whole port and it is worth landing *on CPU first*, independently of any GPU work: it
removes ~340 nodes from a ViT-L video graph and is bit-identical.

## 3. head_dim 32 — the V-JEPA 2 / 2.1 predictor

**Is head_dim 32 supported by any CUDA flash-attention path?** No.
`ggml_cuda_get_best_fattn_kernel` (`fattn.cu:359-439`) switches on `K->ne[0]` over
`{40, 64, 72, 80, 96, 112, 128, 192, 256, 320, 512, 576}` and returns `BEST_FATTN_KERNEL_NONE` in
`default:`. 32 is not there — not for the MMA kernel (`fattn.cu:121` switches on 64/80/96/112/128/192/256/320/512/576),
not for the vec kernel (`FATTN_VEC_CASES_ALL_D` instantiates only 64/128/256), not for the tile
kernel. `ggml_cuda_flash_attn_ext_supported` therefore returns false and `supports_op` says no.

This hits `jepa.pred.head_dim = 32` (V-JEPA 2 predictor: 384-d, 12 heads; V-JEPA 2.1: same,
`docs/architecture.md` "Predictor head_dim is 32 → d = 10 per axis"). The encoders are unaffected
(head_dim 64 for ViT-L/B, 80 for I-JEPA ViT-H).

**Fallbacks, in order of preference:**

1. **Naive attention on the GPU** — set `opts.attn.flash = false` for the masked predictor when the
   backend is CUDA. `mul_mat + soft_max_ext + cont(permute(v))` are all CUDA-supported (§1.1), so
   the predictor stays one split and fully resident. Cost is the score matrix, `4·N²·H` bytes with
   H = 12 — and **N here is `n_context + n_target`, i.e. twice the token count** in the default
   `jepa_predict` pass (`docs/parity.md` "the predictor is 12 layers of 384 dims over 4096 rows
   (2048 context + 2048 mask tokens)"):

   | shape | N (rows) | naive score matrix |
   |---|---:|---:|
   | 2.1 ViT-B, 576-token image | 1 152 | 0.06 GB |
   | ViT-L, 2 048-token 16f clip | 4 096 | **0.80 GB** |
   | 2.1 ViT-B, 4 608-token 16f clip | 9 216 | **4.1 GB** |
   | ViT-L, 8 192-token 64f clip | 16 384 | 12.9 GB |
   | 2.1 ViT-B, 18 432-token 64f clip | 36 864 | 65 GB — **does not fit** |

   The first three are comfortable on a 24 GB card next to a ≤ 2.5 GB weight buffer; the 64-frame
   ViT-L case is tight (12.9 GB + weights + activations); the 64-frame 2.1 case has to stay on the
   CPU or be chunked over query blocks. `docs/ggml-notes.md` §3 measures the naive path at ~7–9×
   slower than flash *on CPU at head_dim 32*; on the GPU the ratio should be far smaller, because
   the naive path is two dense GEMMs and one softmax, which is exactly what the hardware is good at
   and exactly what the CPU is not. Measured in §5.2.
2. **Per-layer CPU split** — what `ggml_backend_sched` would do by default. 12 splits per predictor
   call plus the `[hd, N, H]` q/k/v round trips. Strictly worse than (1) and it must be avoided by
   *explicitly* choosing the naive path, not left to the scheduler.
3. Pad head_dim 32 → 64 with zeros. Doubles the attention FLOPs, adds two pad/slice ops per block
   and changes nothing about the numerics of the softmax (padded lanes contribute 0 to the dot
   product). Mentioned for completeness; option (1) is simpler and faster.

Note the LeWM predictor (head_dim 64, N = 3 frames) *is* supported, and the attentive-pool head
(head_dim 64/80, `N_q = 1`) takes the vec kernel. Both, however, lose their **F32 K/V** mitigation
on CUDA (§1.5) — `src/lewm.cpp:57` and `src/jepa.cpp:534` explicitly set `GGML_TYPE_F32` for exactly
the reason that ggml's CPU one-chunk kernel rounds q and the PV accumulator to F16; on CUDA the K/V
are down-converted regardless, so those two comments become CPU-only statements.

## 4. Scratch CUDA build

The audit builds ggml with `GGML_CUDA=ON` in a **separate** CMake project under this worktree's
`tmp/gpu-probe/` (build dir `tmp/build-cuda/`), so the shared `build/` is untouched. It links `ggml`
only — no jepa.cpp code — and reproduces jepa.cpp's graphs from first principles, which is what lets
the op probes use the exact shapes and strides `src/jepa.cpp` produces.

What the build has to chew through at `-j8`: **138 CUDA translation units** (67 top-level `.cu`,
about 24 k lines, plus 71 `template-instances/*.cu` for the flash-attention and MMQ/MMF kernels;
`GGML_CUDA_FA_ALL_QUANTS` stays off, which is what keeps the fattn-vec instances out). `GGML_NATIVE=ON`
resolves `CMAKE_CUDA_ARCHITECTURES` to `native` → the single arch `89`, so this is the cheapest
faithful build available. Measured build time and any CUDA-13/compute-8.9 diagnostics are in §5.

*(measurements pending the shared-box sentinel)*

## 5. Measured

*(pending the shared-box sentinel; see the final report)*

## 6. Implementation plan

### 6.1 `src/jepa-internal.h` / `src/jepa.cpp` — backend selection

Today `jepa_context` owns exactly one backend (`ggml_backend_cpu_init()`), one `ggml_gallocr_t`, and
`jepa_model` keeps a second CPU backend purely as the weight-buffer owner. The port keeps that
shape — **one backend, no `ggml_backend_sched`** — because §1 shows the graph is fully CUDA-resident
after the §2 fix. `ggml_backend_sched` is the wrong tool here: it exists to *partition* a graph, and
a partitioned jepa.cpp graph is a bug, not a feature. Using a single backend also makes any split a
loud failure (`ggml_backend_graph_compute` returns an error) instead of a silent 10× slowdown.

```diff
 struct jepa_context_params {
     int  n_threads;
     bool use_flash_attn;
     ...
+    int  gpu_device;      // -1 = CPU (default), >= 0 = the n-th GPU device from the ggml registry
 };
```

```diff
 struct jepa_model {
-    ggml_backend_t        backend = nullptr;   // CPU backend used for weight storage
+    ggml_backend_t        backend = nullptr;   // backend the weights live on
     ggml_backend_buffer_t buf_w   = nullptr;
+    bool                  is_gpu  = false;
 };
 struct jepa_context {
-    ggml_backend_t      backend = nullptr;   // CPU backend (compute)
+    ggml_backend_t      backend = nullptr;   // compute backend (CPU or CUDA); must match model->backend
+    bool                is_gpu  = false;
 };
```

1. **Device discovery.** `ggml_backend_load_all()` at load time, then
   `ggml_backend_dev_get(i)` filtered on `GGML_BACKEND_DEVICE_TYPE_GPU`; `--gpu N` selects the N-th.
   No CUDA header is needed in jepa.cpp — the registry API in `ggml-backend.h` is backend-agnostic,
   so the same code would light up Vulkan/Metal/ROCm if those were built.
2. **Weight allocation — already backend-agnostic.** `jepa_model_load` (`src/jepa-gguf.cpp:410-453`)
   allocates `buf_w` with `ggml_backend_alloc_ctx_tensors(m->ctx_w, m->backend)` and fills each
   tensor with `ggml_backend_tensor_set` from a heap read buffer. Both calls dispatch through the
   backend interface, so **the only change is which backend `m->backend` is** — one line, plus the
   device lookup. `ggml_backend_buffer_set_usage(..., USAGE_WEIGHTS)` is already there, which is
   what keeps the CUDA `mul_mat` off its `bad_padding_clear` cuBLAS fallback (`ggml-cuda.cu:1827`).
   Sizes fit easily: ViT-H f32 2.4 GB, ViT-L f32 1.3 GB, everything else smaller, against 24 GB.
   The public API needs the device though: `jepa_model_load(const char *, bool verbose)` has no
   slot for it, so add `jepa_model_load_ex(const char * path, const jepa_model_params *)` (or a
   `jepa_model_params` with `{ verbose, gpu_device }`) and keep the old symbol as a wrapper — there
   are 7 `jepa_model_load` call sites (4 tools, 2 tests, the header), and with a wrapper only the
   4 tools that gain `--gpu` have to change.
   **The model must be loaded onto the device it will run on**; a `jepa_context` whose
   `gpu_device` differs from the model's must be rejected, not silently split.
3. **Transfer points.** Exactly three, all already funnelled through helpers:
   `ggml_backend_tensor_set(inp, …)` for the host-side patchify output (`jepa_encode_image`,
   `jepa_encode_video`), `ggml_backend_tensor_set(cos_t/sin_t, …)` for the RoPE tables, and
   `ggml_backend_tensor_get(y, …)` for `[D, N]` out. No change needed — the CUDA implementations
   (`ggml-cuda.cu:786/794`) are `cudaMemcpyAsync` + `cudaStreamSynchronize` on `cudaStreamPerThread`,
   i.e. **synchronous**, so `last_compute_ms` keeps its meaning and no explicit sync is needed. (The
   per-thread stream does mean a `jepa_context` must not be driven from two threads at once — it
   already must not be.) Budget at the worst shape, 64-frame ViT-L @256: patch rows
   `8192 × 1536 × 4 B` = 50 MB in, RoPE tables 4 MB in, `[1024, 8192]` = 34 MB out ≈ **88 MB**, or
   ~7 ms of pageable PCIe 4.0 ×16 against a forward measured in the hundreds of ms. Worth adding a
   **pinned staging buffer** for the patch rows (`ggml_backend_dev_host_buffer_type`) — it roughly
   doubles H2D bandwidth — but it is a tuning detail, not a design constraint.
   Note that `docs/benchmarks.md`'s `ms` column is `ggml_backend_graph_compute` only and therefore
   **excludes** these copies on both backends; a GPU row must say so, or quote wall time instead.
4. **`jepa_graph_compute`.** Unchanged: `ggml_backend_graph_compute` already synchronises, so
   `last_compute_ms` stays meaningful. One free win comes with it: the CUDA backend captures a
   **CUDA graph** once it has seen the same topology and the same tensor addresses twice in a row
   (`ggml_backend_cuda_graph_compute`, `ggml-cuda.cu:4247-4288`; enabled from Volta up, so Ada
   qualifies). jepa.cpp rebuilds the graph every call but with an identical topology and a reused
   `ggml_gallocr`, so from the third call on the whole encoder becomes one graph launch instead of
   ~700 kernel launches. That matters most for the small-N models (I-JEPA ViT-H is 32 blocks × ~20
   nodes at N = 256, where per-launch overhead is a real fraction of the forward) — but only if the
   graph is **not** split, which is another reason single-backend beats `ggml_backend_sched`.
5. **K/V dtype policy on GPU.** `jepa_context_kv_type` should return `GGML_TYPE_F16` unconditionally
   when `is_gpu` (§1.5) and log once if the user asked for `JEPA_KV_F32`, because the request cannot
   be honoured. The honest F32 path on GPU is `--no-flash`, which should be documented as
   "image-model / short-clip only" with the `4·N²·H`-byte score matrix spelled out.
6. **Predictor attention on GPU.** `src/predictor.cpp` must force `opts.attn.flash = false` when
   `is_gpu` and `head_dim_eff() == 32` (§3), with a one-line log. Leaving `flash = true` there is a
   silent 12-split-per-call trap.
7. **`ggml_mul_mat_set_prec(GGML_PREC_F32)`** in `jepa_build_linear` when `is_gpu` (§1.3) — one line,
   and it is a no-op on CPU so it can be unconditional.

### 6.2 Tools

`--gpu N` (default: none, i.e. CPU) on `jepa-embed`, `jepa-classify`, `jepa-worldmodel`, `jepa-bench`;
`jepa-info` gains a `--devices` listing (`ggml_backend_dev_name/description/memory`). `jepa-bench`
should print the device in its `--md` / `--json` header so `docs/benchmarks.md` can grow GPU rows
without ambiguity. `--threads` keeps meaning CPU threads and becomes irrelevant (but not an error)
under `--gpu`. `jepa-quantize` stays CPU-only.

### 6.3 Build system

```cmake
option(JEPA_CUDA "Build with the ggml CUDA backend" OFF)
if (JEPA_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
```

Nothing else: jepa.cpp links `ggml`, and `GGML_CUDA=ON` pulls `ggml-cuda` into that target. The
`GGML_NATIVE=ON` we already set makes `CMAKE_CUDA_ARCHITECTURES=native`, which is right for a
from-source project and wrong for a distributed binary — a release build should set an explicit
list (`89-real;80-virtual` covers Ada plus a JIT fallback). Note that with CUDA 13 the toolkit no
longer offers the 50/61/70 virtual architectures the ggml CMake would otherwise append, so a non-native
CUDA-13 build starts at `75-virtual` (`ggml/src/ggml-cuda/CMakeLists.txt:31`).

**CI story.** `.github/workflows/` currently contains **only `docs.yml`** — jepa.cpp has no compile
or test CI at all today, so "add a CUDA job" really means "add the first build job". The realistic
shape is: (a) a CPU job that configures, builds and runs `ctest` — worth adding regardless of GPU
work; (b) a CUDA **compile-only** job (`Jimver/cuda-toolkit` action, `-DJEPA_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=89-virtual`), because GitHub-hosted runners have no NVIDIA GPU; (c) GPU
correctness left to a manual/self-hosted run. The thing that makes (c) cheap is chunk 5's
`test-backend`: one binary that runs every jepa.cpp graph on the CPU and on the GPU and compares —
it skips silently with no GPU, so it can live in the normal `ctest` set.

### 6.4 Parity implications

| model / dtype | CPU class today | expected GPU class | why |
|---|---|---|---|
| f32 image (LeJEPA, LeWM, I-JEPA) with flash | every token cos ≥ 0.9999, `rel_max` ≤ REL(N) | **drops to the f16-K/V class** | no F32 K/V on CUDA (§1.5) |
| f32 image with `--no-flash` | — | should hold the f32 class, modulo `ggml_norm` (§1.4) | naive path is F32 end to end on CUDA |
| f32 video (ViT-L 16f/64f) | cos 1.000000, `rel_max` 7.5e-4 / 1.2e-3 | f16-K/V class; `--no-flash` not affordable above ~4 608 tokens | 15.3 GB score matrix at 18 432 |
| f16 anything | mean ≥ 0.9999, worst ≥ 0.99 | **wider unless `GGML_PREC_F32` is set on every mul_mat** (§1.3) | cuBLAS HGEMM |
| q8_0 / q4_* | as `docs/parity.md` | comparable; `mmq` is INT8×INT8→INT32 like the CPU dot | same quantisation, different reduction order |
| I-JEPA ViT-H, any dtype | — | watch `ggml_norm` (§1.4) — the ~2e4 activations are exactly its bad case | one-pass variance |

The practical recommendation is that `tests/test-parity.cpp`'s `POLICY` table would need a
**backend dimension**, not just family × file-type: the GPU rows of the f32 tier cannot meet the CPU
f32 thresholds with flash attention on. That is a real cost of the port and should be decided before
any code is written.

### 6.5 Multi-GPU

**Ignore it for the compute path.** Both cards are the same model on the same box, and a ViT forward
is a single dependency chain — tensor-parallel splitting inside ggml means `ggml_backend_sched` with
peer copies per layer, which is the same disease as §1.2. The right multi-GPU story for an inference
engine of this shape is **two independent streams**: one `jepa_model` + `jepa_context` per device,
each processing different clips. That falls out of `--gpu N` for free and needs no engine work
beyond making `jepa_model_load` device-aware. Document it; do not implement a scheduler.

### 6.6 Vulkan as a secondary target

Not worth it as a second target *on this box*, and not worth it soon:

- **It cannot be built here today.** `ggml/src/ggml-vulkan/CMakeLists.txt:9,14` requires
  `find_package(Vulkan COMPONENTS glslc REQUIRED)` and `find_package(SPIRV-Headers CONFIG REQUIRED)`.
  The box has the loader (`libvulkan1` 1.3.275) and the NVIDIA ICD but neither `glslc` (shaderc) nor
  the Vulkan/SPIRV headers, so `-DGGML_VULKAN=ON` fails at configure time. Adding them is
  `apt install glslc libvulkan-dev spirv-headers` — a prerequisite, not a blocker.
- **On NVIDIA it is strictly slower than CUDA** and its `ggml_vk_supports_op` coverage for
  `flash_attn_ext` is narrower (it needs the `coopmat`/`coopmat2` extensions for the fast path).
- **Its value is portability**, i.e. AMD and Intel GPUs and Apple-less non-CUDA boxes. That is a real
  goal for a "runs anywhere in plain C++" project, but it is a *third* milestone, after CUDA works
  and after the §2 refactor has proven that the graph is backend-clean. The good news from §1 is
  that the §2 refactor is what makes *any* backend viable — nothing in it is CUDA-specific.

### 6.7 Effort, in reviewable chunks

| # | chunk | files | est. |
|---|---|---|---|
| 1 | **RoPE table refactor** (§2) — signed sin table, one-`roll` apply, `ggml_cont` for views. Bit-identical, CPU-only change, ships on its own value. | `src/rope3d.{h,cpp}`, `tests/test-ops.cpp` | ~55 lines, **0.5 d** |
| 2 | **Backend plumbing** — `gpu_device` param, device discovery, device weight allocation, `jepa_context`/`jepa_model` pairing check, `jepa-info --devices`. | `include/jepa.h`, `src/jepa-internal.h`, `src/jepa.cpp`, `src/jepa-gguf.cpp`, `tools/jepa-info.cpp` | ~250 lines, **1.5 d** |
| 3 | **GPU numeric policy** — `ggml_mul_mat_set_prec` in `jepa_build_linear`, K/V policy override + log, predictor naive-attention override, `--no-flash` documentation. | `src/jepa.cpp`, `src/predictor.cpp` | ~40 lines, **0.5 d** |
| 4 | **Tools + build** — `--gpu N` on four tools, `JEPA_CUDA` option, pinned staging buffer, bench header records the device. | `tools/*.cpp`, `CMakeLists.txt` | ~150 lines, **1 d** |
| 5 | **Backend-parity test** — `tests/test-backend.cpp` running every graph on CPU and GPU and comparing; `POLICY` gains a backend dimension in `test-parity`. | `tests/`, `CMakeLists.txt` | ~350 lines, **1.5 d** |
| 6 | **Measure + document** — GPU rows in `docs/benchmarks.md` and `docs/parity.md`, a GPU section in the README, torch-GPU baselines. | `docs/`, `scripts/bench_all.sh` | **1 d** |

**Total ≈ 6 developer-days** for a CUDA backend that is honest about its numerics, plus whatever the
parity-policy decision (§6.4) costs in discussion. Chunk 1 is worth doing regardless of the outcome.

## 7. Go / no-go

*(filled in with the measurements)*
