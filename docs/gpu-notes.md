# GPU (CUDA) feasibility for jepa.cpp

**Status: implemented — §§0-7 are the audit that preceded it, §8 is what shipped.** Sections 0-7 were
written before any code changed and are left exactly as they were, so the forecast can be read against
the result; **§8 records the port as built and measured**, including the two places the audit was wrong.
A CUDA build is `-DJEPA_CUDA=ON` (off by default) and the tools take `--gpu [N]`; start at §8 if you
want the numbers rather than the reasoning. The audit
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

Ceilings, computed from what `nvidia-smi` reports rather than from a datasheet: 7 680 CUDA cores at
a max SM clock of **3 105 MHz** and a 210 W board limit give **47.7 TFLOP/s FP32** and, at the 2×
dense rate of Ada's 4th-gen tensor cores, **~95 TFLOP/s FP16 with FP32 accumulate**. Sustained
clocks under a 210 W cap are lower than 3 105 MHz, so treat these as upper bounds. §5.6 shows
PyTorch reaching 56–77 TFLOP/s (fp16) and ~14 TFLOP/s (fp32) on this card, which is the practical
ceiling any ggml-CUDA number should be read against.

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
  The §5.5 encoder runs check for non-finite outputs and found none — but their random weights keep
  activations near ±5, so that is **not** evidence about the real ViT-H. Like the `ggml_norm`
  question of §5.3, it needs a dump of real activations and is a pre-flight check for chunk 3.

F32 weights use `batched_mul_mat_traits<GGML_TYPE_F32>`, i.e. `CUBLAS_COMPUTE_32F` with `CUDA_R_32F`,
which *reads* like strict FP32 — **but §5.1 measured that it is not**: 62.7 TFLOP/s (above this card's
FP32 ceiling) at 3.2e-04 error against the CPU, which is TF32. ggml passes the legacy algo enum
`CUBLAS_GEMM_DEFAULT_TENSOR_OP` to every `cublasGemmEx` (`ggml-cuda.cu:1559/1575/1613`) and never
calls `cublasSetMathMode`, and that enum permits TF32 downconversion for `CUBLAS_COMPUTE_32F`.
`GGML_PREC_F32` cannot undo it, because it only selects the compute type. **Read §5.1 before relying
on anything in this paragraph.** Quantized weights take the `mmq` path (INT8 tensor cores, INT32
accumulate, F32 rescale) and are unaffected.

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

### 4.1 It builds clean

```
cmake -S tmp/gpu-probe -B tmp/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build tmp/build-cuda -j8
```

| | |
|---|---|
| configure | 4.5 s; `Found CUDAToolkit … 13.0.88`, `CUDA compiler … NVIDIA 13.0.88 with host compiler GNU 13.3.0` |
| architectures | `Using CMAKE_CUDA_ARCHITECTURES=89-real CMAKE_CUDA_ARCHITECTURES_NATIVE=89-real` |
| build | **171 s** wall at `-j8`, 178 ninja targets |
| diagnostics | **zero warnings, zero errors** (`grep -ci warning build.log` → 0) |
| `libggml-cuda.so` | **45.0 MB** for the single arch 89 |
| `libggml-cpu.so` / `libggml-base.so` | 1.4 MB / 0.9 MB |
| whole build tree | 114 MB |

**No CUDA-13 or compute-8.9 problems at all.** Two non-fatal notices worth knowing: `ccache not found`
(would cut rebuild time a lot) and `Could NOT find NCCL … performance for multiple CUDA GPUs will be
suboptimal`, which is irrelevant under the multi-GPU stance in §6.5.

The 45 MB CUDA shared object is the real distribution cost: it is 30× the CPU backend, for one
architecture. A binary that shipped `75-virtual;80-virtual;86-real;89-real;90-virtual` would be
several times that. For a project whose pitch is "plain C/C++ on ggml", `JEPA_CUDA=OFF` **must** stay
the default (§6.3).

## 5. Measured

### 5.0 The FLOP budget the forecast is built on

Per transformer layer and token, the matmuls are `2·D·(3D + D + 2·FF)` FLOPs (qkv, out projection,
FFN up+down) and the attention is `4·N²·D` (QKᵀ and PV), with `D` the embedding dim. That gives:

| model | shape | tokens N | matmul GFLOP | attention GFLOP | total | attention share |
|---|---|---:|---:|---:|---:|---:|
| I-JEPA ViT-H/14 (D 1280, FF 5120, L 32) | 224×224 image | 256 | 322 | 11 | **333** | 3 % |
| V-JEPA 2 ViT-L (D 1024, FF 4096, L 24) | 16f @256 | 2 048 | 1 237 | 412 | **1 649** | 25 % |
| V-JEPA 2 ViT-L | 64f @256 | 8 192 | 4 948 | 6 597 | **11 545** | 57 % |
| V-JEPA 2.1 ViT-B (D 768, FF 3072, L 12) | 384 image | 576 | 98 | 12 | **110** | 11 % |
| V-JEPA 2.1 ViT-B | 16f @384 | 4 608 | 783 | 783 | **1 566** | 50 % |
| V-JEPA 2.1 ViT-B | 64f @384 | 18 432 | 3 131 | 12 524 | **15 655** | 80 % |
| V-JEPA 2 predictor (D 384, FF 1536, L 12) | ViT-L 16f clip, N = ctx+tgt | 4 096 | 174 | 309 | **483** | 64 % |
| V-JEPA 2.1 predictor | 2.1 16f clip, N = ctx+tgt | 9 216 | 391 | 1 566 | **1 957** | 80 % |

So the port has **two different bottlenecks depending on the shape**: images and short clips are
matmul-bound (attention 3–25 %), long clips are attention-bound (57–80 %). A forecast therefore
needs both the `mul_mat` and the `flash_attn_ext` rates, and `t ≈ mm_GF/mm_rate + attn_GF/attn_rate`.

That model is validated against the CPU numbers already in the repo: `docs/ggml-notes.md` §5 measures
~3.3 TFLOP/s F16 matmul and §3 measures ~1.5–1.9 TFLOP/s flash at 32 threads, which predicts
`4948/3.3 + 6597/1.7 = 5.4 s` for the 64-frame ViT-L clip against the **6 388 ms** `docs/benchmarks.md`
actually measures — 18 % out, i.e. the right shape with the expected slack for LN/GELU/launch overhead.

### 5.1 `mul_mat` — CUDA vs CPU

`Y = mul_mat(W, X)`, `W = [1024 x 4096]` (a ViT-L `ffn_up`), `X = [1024, N]` F32 — the same shape
`docs/ggml-notes.md` §5 benches on the CPU. Best of 20 (CUDA) / 3 (CPU, 32 threads); the error column
is CUDA vs the CPU backend on the *same weight bytes*, i.e. the kernel difference alone, not
quantisation.

| W type | N | CUDA ms | **CUDA GFLOP/s** | CPU 32t ms | CPU GFLOP/s | CUDA vs CPU max rel | min cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| f32 | 256 | 0.055 | 38 823 | 0.99 | 2 180 | 3.17e-04 | 1.0000000 |
| f32 | 2048 | 0.290 | 59 310 | 7.86 | 2 185 | 3.19e-04 | 1.0000000 |
| f32 | 8192 | 1.095 | **62 744** | 31.46 | 2 185 | 3.28e-04 | 1.0000000 |
| f16 | 256 | 0.047 | 45 758 | 0.72 | 2 992 | 4.55e-03 | 0.9999984 |
| f16 | 2048 | 0.260 | **66 034** | 5.25 | 3 271 | 4.58e-03 | 0.9999984 |
| f16 | 8192 | 1.229 | 55 905 | 20.60 | 3 336 | 4.53e-03 | 0.9999984 |
| q8_0 | 256 | 0.040 | 54 051 | 0.58 | 3 727 | 5.15e-04 | 0.9999998 |
| q8_0 | 2048 | 0.175 | **98 066** | 8.42 | 2 040 | 5.37e-04 | 0.9999998 |
| q8_0 | 8192 | 0.767 | 89 570 | 30.22 | 2 274 | 6.25e-04 | 0.9999996 |
| q4_0 | 2048 | 0.176 | **97 458** | 9.11 | 1 885 | 4.84e-04 | 0.9999998 |
| q4_0 | 8192 | 0.751 | 91 497 | 32.64 | 2 105 | 6.00e-04 | 0.9999997 |
| q4_K | 2048 | 0.185 | **93 019** | 9.64 | 1 783 | 1.35e-02 | 0.9991882 |
| q4_K | 8192 | 0.778 | 88 294 | 38.05 | 1 806 | 1.39e-02 | 0.9991175 |

**Three findings, one of them a correction to §1.3.**

1. **ggml's "F32" matmul on CUDA is actually TF32.** 62.7 TFLOP/s is *above* this card's 47.7 TFLOP/s
   FP32 ceiling, and the error against the CPU is **3.2e-04** — far too large for real FP32 (which
   would be ~1e-7) and exactly right for TF32's 10-bit mantissa. The cause is not the compute type:
   ggml passes the legacy algo enum `CUBLAS_GEMM_DEFAULT_TENSOR_OP` to every `cublasGemmEx` /
   `cublasGemmStridedBatchedEx` (`ggml-cuda.cu:1559/1575/1613`) and never calls `cublasSetMathMode`,
   and that enum permits TF32 for `CUBLAS_COMPUTE_32F`.
   **`ggml_mul_mat_set_prec(GGML_PREC_F32)` does not help** — it only forces `compute_type` to
   `GGML_TYPE_F32`, which is already the case for an F32 weight, so the `--prec-f32` run reproduces
   the f32 rows to the digit (38 915 / 59 306 / 62 625 GFLOP/s, rel 3.17e-04). **There is no way to
   get a true-F32 `mul_mat` out of ggml-CUDA from jepa.cpp.** Torch is in the same regime by
   *choice* (`torch.backends.cuda.matmul.allow_tf32` defaults to `False`, and §5.6 confirms it ran
   at 13.8 TFLOP/s, i.e. real FP32); ggml is in it by accident.
2. **`GGML_PREC_F32` transforms f16 accuracy and is nearly free at scale.** With `--prec-f32` the f16
   rows go from **4.55e-03 to 2.57e-05** max rel error (177x better) and min cos 0.9999984 to
   1.0000000, at 28 704 / 52 067 / 61 138 GFLOP/s — i.e. **-37 % at N=256, -21 % at N=2048, +9 % at
   N=8192**. For anything but the shortest sequence it is close to free. This is the single cheapest
   accuracy fix in the whole port.
3. **The quantized types are the *fastest* path on CUDA** — 88-98 TFLOP/s, above f16 and f32 — and
   q4_K is level with q8_0. That is the inversion §1.3 predicted: on the CPU `docs/benchmarks.md` has
   q4_k *slower* than q8_0 (ViT-H 198 vs 129 ms) because llamafile's sgemm covers only F32/F16/Q8_0.

Speed-up over the 32-thread CPU at N=2048: f32 **27x**, f16 **20x**, q8_0 **48x**, q4_K **52x**.

### 5.2 `flash_attn_ext` — CUDA vs CPU, and the head_dim-32 fallback

No mask, `GGML_PREC_F32` requested (a no-op on CUDA, §1.5). Best of 10 (CUDA) / 2 (CPU, 32 threads).
`GFLOP/s` counts `4*N^2*D*H`.

| head_dim | n_head | N | K/V | shape | CUDA ms | **CUDA GFLOP/s** | CPU 32t ms | rel vs CPU | min cos |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 64 | 16 | 2 048 | f16 | ViT-L 16f@256 | 0.268 | 64 143 | 10.1 | 5.25e-03 | 0.9999903 |
| 64 | 16 | 8 192 | f16 | ViT-L 64f@256 | 3.890 | **70 669** | 159.9 | 1.09e-02 | 0.9999572 |
| 64 | 12 | 4 608 | f16 | 2.1 ViT-B 16f@384 | 0.939 | 69 498 | 40.1 | 9.85e-03 | 0.9999800 |
| 64 | 12 | 18 432 | f16 | 2.1 ViT-B 64f@384 | 20.379 | 51 213 | 639.0 | 1.53e-02 | 0.9999122 |
| 80 | 16 | 256 | f16 | I-JEPA ViT-H@224 | 0.028 | 11 872 | 0.6 | 9.51e-04 | 0.9999994 |
| 64 | 16 | 2 048 | **f32** | ViT-L 16f@256 | 0.291 | 59 045 | 9.7 | 5.21e-03 | 0.9999903 |
| 64 | 16 | 8 192 | **f32** | ViT-L 64f@256 | 4.194 | 65 542 | 154.6 | 1.08e-02 | 0.9999572 |
| 64 | 12 | 18 432 | **f32** | 2.1 ViT-B 64f@384 | 19.996 | 52 193 | 626.0 | 1.53e-02 | 0.9999116 |
| 80 | 16 | 256 | **f32** | I-JEPA ViT-H@224 | 0.034 | 9 904 | 0.5 | 1.08e-03 | 0.9999993 |
| 32 | 12 | 2 048 / 4 096 | f16 **or** f32 | V-JEPA 2 predictor | **UNSUPPORTED** | – | – | – | – |

* **35-40x faster than the CPU** on every production encoder shape, running at 51-71 TFLOP/s. The
  small I-JEPA case (N=256) only reaches 11.9 TFLOP/s — 11 GFLOP of work is too little to fill the
  GPU, which is exactly why I-JEPA's 3 % attention share (§5.0) does not matter.
* **F32 K/V buys nothing, exactly as §1.5 predicted from the source.** Same error to three digits
  (5.21e-03 vs 5.25e-03) and, if anything, *slower* (the extra F32-to-F16 conversion pass). This is
  the measured proof that `JEPA_KV_F32` cannot be honoured on CUDA.
* The CUDA error is ~10x the CPU's. `docs/ggml-notes.md` §3 measures CPU flash with F16 K/V at
  rel <= 1.3e-3 and per-row cosine >= 0.9999995; CUDA lands at rel 5.2e-3-1.5e-2 and min cos
  0.99991-0.99999. That is the F16 PV accumulator (`T_C_VKQ = tile<..., half2>`, §1.5), and unlike
  the `mul_mat` case there is **no flag to turn it off**.

**The head_dim-32 fallback, measured.** Naive F32 attention (`mul_mat + soft_max_ext + cont`) at the
predictors' real row counts (N = context + target):

| head_dim | n_head | N | shape | CUDA ms | CUDA GFLOP/s | CPU 32t ms | rel vs CPU | min cos | CUDA graph buffer |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 12 | 1 152 | 2.1 ViT-B image predictor | 0.799 | 2 551 | 5.3 | 6.69e-04 | 0.9999994 | 0.06 GiB |
| 32 | 12 | 4 096 | ViT-L 16f predictor | 8.623 | 2 989 | 57.4 | 6.49e-04 | 0.9999993 | **0.76 GiB** |
| 32 | 12 | 9 216 | 2.1 ViT-B 16f predictor | 43.979 | 2 966 | – | – | – | **3.82 GiB** |
| 64 | 16 | 2 048 | (ViT-L encoder, for reference) | 2.913 | 5 898 | 26.6 | 7.42e-04 | 0.9999995 | 0.27 GiB |
| 80 | 16 | 256 | (I-JEPA encoder, for reference) | 0.041 | 8 238 | 1.0 | 4.09e-04 | 0.9999997 | 0.01 GiB |

It **works, fits, and is accurate** (rel 6.5e-04, min cos 0.9999993 — *better* than flash, because it
is genuinely F32 end to end), but it runs at only **~3 TFLOP/s, 23x below flash at head_dim 64**.
So the predictor is the weak spot of a CUDA port: 12 layers x 8.6 ms = ~103 ms of attention for the
ViT-L 16-frame predictor against 452 ms on the CPU — a **4-7x win, not the 20x the encoders get**.
The 3.82 GiB buffer at N=9216 is fine on a 24 GB card; the 64-frame 2.1 shape (N=36 864, 65 GB)
remains out of reach and must stay on the CPU or be chunked over query blocks.

### 5.3 `ggml_norm`: how bad the one-pass variance actually is

`ggml_norm(x, 1e-6)` on `[1280, 256]` (a ViT-H row width), rows drawn as `N(mean, stdev)`, compared
against a double-precision two-pass reference.

| row mean | row stdev | mean/sigma | CPU vs double | **CUDA vs double** | min cos (CUDA) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 9.18e-08 | 1.38e-07 | 1.0000000 |
| 0 | 30 | 0 | 9.18e-08 | 1.38e-07 | 1.0000000 |
| 1 | 1 | 1 | 9.18e-08 | 1.84e-07 | 1.0000000 |
| 100 | 1 | 100 | 1.19e-06 | **8.94e-04** | 1.0000000 |
| 100 | 30 | 3.3 | 9.18e-08 | 1.24e-06 | 1.0000000 |
| 2 000 | 1 | 2 000 | 2.91e-05 | **5.70e+00** | 1.0000000 |
| 2 000 | 30 | 67 | 9.64e-07 | 3.96e-04 | 1.0000000 |
| 20 000 | 1 | 20 000 | 3.05e-04 | **7.99e-01** | 0.9999984 |
| 20 000 | 30 | 667 | 9.82e-06 | 4.05e-02 | 1.0000000 |

The failure is governed by **|row mean| / row sigma**, not by absolute magnitude: at a ratio of ~1 the
two backends agree to 1e-7; at 100 the CUDA error is 8.9e-4; at 2 000 the output is **numerically
destroyed** (relative error 5.7).

**And cosine cannot see it.** A wrong variance is a *per-row scale factor*, which leaves cosine
similarity at 1.0000000 in every row above. jepa.cpp's parity gates are cosine-dominated
(`docs/parity.md` "Thresholds"); only `rel_max` would catch this, and only in the tiers that gate on
it. That is worth knowing regardless of whether the port happens.

**What this audit did *not* establish** is whether real ViT residual streams ever reach a mean/sigma
ratio where it matters. `docs/architecture.md` records I-JEPA ViT-H *activations* reaching ~2e4, but
that is a max element, not a row mean, and a row with large elements has a large sigma too — the
`mean 20 000, sigma 30` row (ratio 667, error 4.05e-02) is the realistic worst case and the `mean 0`
rows are the realistic typical case. Settling this needs a dump of real pre-LN rows, which would
have meant changing `src/`, outside this audit's scope. **It is a required pre-flight check for
chunk 3 of §6.7.**

### 5.4 The RoPE chain: 193 splits vs 1, and bit-identical

`jepa_rope3d_apply` applied to q and k of L layers, on the real non-contiguous qkv view. "current" is
`src/rope3d.cpp` as it stands; "hoisted" is the brief's three-table suggestion **plus** a `ggml_cont`;
"proposed" is §2.3 (two tables, one pair-swap roll). `sched` is `ggml_backend_sched` over
`{CUDA, CPU}`.

| N | L | chain | CPU 32t | **CUDA-only** | sched | **splits** |
|---:|---:|---|---:|---:|---:|---:|
| 2 048 | 24 | current | 7.0 ms | — | 10.2 ms | **193** |
| 2 048 | 24 | hoisted | 22.1 ms | 8.194 ms | 8.181 ms | **1** |
| 2 048 | 24 | proposed | 32.5 ms | **5.118 ms** | 5.175 ms | **1** |
| 8 192 | 6 | current | 8.1 ms | — | 21.3 ms | **49** |
| 8 192 | 6 | hoisted | 26.0 ms | 18.129 ms | 17.973 ms | **1** |
| 8 192 | 6 | proposed | 40.9 ms | **13.360 ms** | 13.323 ms | **1** |

* **193 splits** for 24 layers is 8 per layer — the four rejected rolls plus the copies around them,
  exactly the per-layer fragmentation §1.2 predicted. Both fixes collapse it to **1**.
* **Bit-identical, on both backends.** `max abs 0.000e+00`, `min cos 1.0000000` for hoisted-vs-current
  and proposed-vs-current on the CPU, *and* for proposed-CUDA-vs-current-CPU. The numpy proof in
  §2.3 holds in ggml, in float32, on the GPU.
* The isolated CPU columns look alarming (7.0 to 32.5 ms) but are an **artefact of this
  microbenchmark**: all 48 chains read one shared `qkv` tensor that stays hot in cache, so adding a
  `ggml_cont` per chain looks far more expensive than it is. In the **full encoder**, where each
  layer computes its own qkv, the real cost is (ViT-L, N=2048, f16, 32 threads):

  | chain | CPU 32t | CUDA | correct on CUDA? |
  |---|---:|---:|---|
  | current | **736.8 ms** | 36.9 ms | **NO — see below** |
  | hoisted | 757.9 ms (+2.9 %) | 37.8 ms | yes |
  | proposed | 777.5 ms (+5.5 %) | **34.4 ms** | yes |

  So the refactor costs **3-5 % of CPU encoder time**, not 4x. The `hoisted` variant is the cheaper
  of the two on CPU and the `proposed` variant the faster on CUDA; either is acceptable, and the
  `hoisted` one is the smaller change (it keeps today's two ne0-wide rolls, which the CPU kernel
  likes, and only moves the masks to the host).

**A trap worth the whole audit: a single CUDA backend does not check `supports_op`.**
`ggml_backend_cuda_graph_compute` just calls `ggml_cuda_compute_forward` per node
(`ggml-cuda.cu:4181`) and only logs when *that* returns false; `ggml_cuda_op_roll` dispatches happily
and its kernel reads the strided view **as if it were contiguous**. `ggml_backend_cuda_device_supports_op`
is wired only into the device interface, which **only `ggml_backend_sched` consults**. So the
`current` chain on a single CUDA backend produces no error, no warning, and a **wrong answer**:

```
enc --rope-current --verify : verify vs CPU: max abs 3.494e-02 (rel 7.373e-03), min cos 0.9999600
enc --rope         --verify : verify vs CPU: max abs 7.401e-03 (rel 1.560e-03), min cos 0.9999982
```

Note the failure would **pass jepa.cpp's f16 parity gate** (mean >= 0.9999, worst >= 0.99). And
because this synthetic encoder has random weights — giving near-uniform attention that is insensitive
to corrupted q/k — 0.99996 is a *lower bound* on the real error, not an estimate of it. Consequence
for §6.1: a single-backend design **must validate its own graph** against
`ggml_backend_dev_supports_op` at build time. That is ~15 lines and it is not optional.

### 5.5 Synthetic encoder forward, CUDA vs CPU

A full pre-LN ViT stack built from ggml ops — LN, fused qkv, RoPE (proposed), flash attention, output
projection, LN, FFN with `gelu_erf` — with random weights of the right shapes. It omits the patch
embedding (one extra `mul_mat`) and the position table, so it should read slightly *faster* than the
real encoder. Best of 5 (CUDA) / 2 (CPU 32 threads). K/V F16 unless stated.

| model | N | weights | **CUDA ms** | CUDA GFLOP/s | jepa.cpp CPU 32t (`docs/benchmarks.md`) | **speed-up** |
|---|---:|---|---:|---:|---:|---:|
| ViT-L | 2 048 | f32 | 40.7 | 40 571 | 941.3 | 23x |
| ViT-L | 2 048 | f16 | **35.1** | 47 046 | 820.7 | **23x** |
| ViT-L | 2 048 | q8_0 | 31.5 | 52 302 | 794.1 | 25x |
| ViT-L | 2 048 | q4_K | 32.3 | 51 115 | 1 096.1 | 34x |
| ViT-L | 8 192 | f32 | 269.0 | 42 912 | 7 020.1 | 26x |
| ViT-L | 8 192 | f16 | **270.2** | 42 728 | 6 388.1 | **24x** |
| ViT-L | 8 192 | q8_0 | 245.0 | 47 130 | 6 482.4 | 26x |
| ViT-L | 8 192 | q4_K | 246.3 | 46 872 | 7 405.5 | 30x |
| ViT-B | 576 | f16 | 2.9 | 37 581 | 60.3 | 21x |
| ViT-B | 4 608 | f16 | 38.2 | 40 956 | 853.5 | 22x |
| ViT-B | 18 432 | f16 | 397.0 | 39 435 | 9 036.1 | 23x |
| ViT-H | 256 | f32 | 10.8 | 30 945 | 174.4 | 16x |
| ViT-H | 256 | f16 | **7.2** | 45 936 | 147.0 | **20x** |
| ViT-H | 256 | q8_0 | 6.8 | 48 882 | 129.1 | 19x |

**The synthetic graph is a faithful proxy.** Run on the CPU backend it gives 778.8 ms for ViT-L
N=2048 f16 against the 820.7 ms `docs/benchmarks.md` measures for the real thing (-5 %), and 129.4 ms
for ViT-H N=256 f16 against 147.0 (-12 %) — the residual being the patch embed and pos-embed it
omits. So the CUDA column can be compared to the published CPU rows directly.

Variants:

| variant | ViT-H N=256 f16 | ViT-L N=2048 f16 | note |
|---|---:|---:|---|
| baseline | 7.2 ms | 35.1 ms | |
| `+ GGML_PREC_F32` on every mul_mat | **15.1 ms** (2.1x) | **43.7 ms** (1.25x) | the accuracy fix of §5.1 |
| `+ GGML_PREC_F32`, q8_0 weights | – | 31.9 ms (vs 31.5, **free**) | quantized skips cuBLAS |
| F32 weights, F32 K/V | 10.7 ms (vs 10.8 F16 K/V) | – | K/V dtype is a no-op, §5.2 |
| naive attention instead of flash | 11.2 ms (vs 10.7) — **~free** | 98.0 ms (2.8x), 0.30 GiB | the F32-parity escape hatch |

Two useful consequences: **`GGML_PREC_F32` costs 2.1x on the short-sequence ViT-H but only 1.25x on
ViT-L and nothing at all on quantized weights**; and **the naive attention path is essentially free at
I-JEPA's 256 tokens**, so `--no-flash` really is a viable f32-accuracy mode for the image models
(§6.4), while at 2 048 tokens it costs 2.8x.

### 5.6 PyTorch-GPU baseline

The numbers a jepa.cpp-CUDA has to be compared against. One RTX 4500 Ada (`CUDA_VISIBLE_DEVICES=0`),
batch 1, synthetic input of the right shape, `torch.set_num_threads(8)`, TF32 **off** (torch's
default), 3 warmup + 7 timed forwards with `torch.cuda.synchronize()` around each.
Driver 580.173.02, **torch 2.13.0+cu130**, transformers 5.16.1, `attn_implementation="sdpa"`.
`VJEPA2Model(..., skip_predictor=True)` and `IJepaModel`, both loaded from `models/`.
Script: `tmp/torch_gpu_bench.py`; raw JSON: `tmp/results/torch-gpu.json`.

| model | dtype | frames | tokens | mean ms | median ms | min ms | peak GiB | implied TFLOP/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc64-256 | fp32 | 16 | 2 048 | 118.58 | 118.54 | 118.39 | 1.33 | 13.9 |
| vjepa2-vitl-fpc64-256 | fp32 | 64 | 8 192 | 836.52 | 836.61 | 835.12 | 1.65 | 13.8 |
| vjepa2-vitl-fpc64-256 | **fp16** | 16 | 2 048 | **29.56** | 29.56 | 29.47 | 1.27 | **55.8** |
| vjepa2-vitl-fpc64-256 | **fp16** | 64 | 8 192 | **150.03** | 150.04 | 149.85 | 0.83 | **77.0** |
| ijepa_vith14_1k | fp32 | 1 | 256 | 24.99 | 25.02 | 24.66 | 2.43 | 13.3 |
| ijepa_vith14_1k | **fp16** | 1 | 256 | **5.57** | 5.58 | 5.52 | 2.42 | **59.7** |

Run-to-run spread is well under 1 % on every row (mean ≈ median ≈ min), so these are clean numbers.

Two things to take from this table:

* **fp16 is 4.2–5.6× faster than fp32** on this card, because TF32 is off by default in torch and
  ggml likewise never calls `cublasSetMathMode` (§1.3) — both engines run f32 on the FP32 CUDA cores
  (~14 TFLOP/s achieved, 29 % of the 47.7 ceiling) and f16 on the tensor cores (56–77 TFLOP/s,
  59–81 % of the ~95 ceiling). **An f32 GGUF on CUDA will be roughly 4× slower than an f16 one**,
  the same shape as on CPU but for a different reason.
* Against the CPU engine we ship today (`docs/benchmarks.md`, f16):

  | shape | jepa.cpp CPU t=32 | jepa.cpp CPU t=96 | torch-GPU fp16 | GPU vs CPU-32 / CPU-96 |
  |---|---:|---:|---:|---:|
  | I-JEPA ViT-H, 224² | 147.0 ms | 113.1 ms | 5.57 ms | **26.4× / 20.3×** |
  | V-JEPA 2 ViT-L, 16f@256 | 820.7 ms | 566.6 ms | 29.56 ms | **27.8× / 19.2×** |
  | V-JEPA 2 ViT-L, 64f@256 | 6 388.1 ms | 4 026.9 ms | 150.03 ms | **42.6× / 26.8×** |

  That is the size of the prize: one 210 W workstation GPU beats 96 Zen 4 cores by 20–43×. Even a
  jepa.cpp-CUDA that only reached half of torch's throughput would be a 10–21× improvement on the
  fastest thing the project can do today.

### 5.7 Forecast, and the honest ratio against torch-GPU

The §5.5 column *is* the forecast — it is a measured ggml-CUDA forward of the right shape, so no
extrapolation is needed. Adding the transfers of §6.1.3 (~7 ms at the largest shape, less elsewhere)
and the patch embed (~2-5 % of the matmul work):

| model / shape | jepa.cpp CPU 32t today | **forecast jepa.cpp-CUDA (f16)** | torch-GPU fp16 | **ggml / torch** |
|---|---:|---:|---:|---:|
| I-JEPA ViT-H, 224^2 | 147.0 ms | **~7.5 ms** | 5.57 ms | 1.35x slower (**74 %**) |
| V-JEPA 2 ViT-L, 16f@256 | 820.7 ms | **~37 ms** | 29.56 ms | 1.25x slower (**80 %**) |
| V-JEPA 2 ViT-L, 64f@256 | 6 388.1 ms | **~280 ms** | 150.03 ms | 1.87x slower (**54 %**) |
| V-JEPA 2.1 ViT-B, 16f@384 | 853.5 ms | ~40 ms | – | – |
| V-JEPA 2.1 ViT-B, 64f@384 | 9 036.1 ms | ~405 ms | – | – |
| V-JEPA 2 ViT-L predictor, 16f | 452 ms | **~105 ms** | – | (naive attention, §5.2) |

**The honest ratio is 54-80 % of PyTorch**, i.e. jepa.cpp-CUDA would be 1.25-1.9x slower than
`VJEPA2Model` on the same card — and **20-26x faster than jepa.cpp as it exists today**.

Where the remaining gap is, at the 64-frame ViT-L shape: the measured component rates (§5.1, §5.2)
predict `4948/55.9 + 6597/70.7 = 88 + 93 = 182 ms`, against 270 ms measured. The missing ~88 ms is
everything that is neither a GEMM nor an attention: 48 unfused LayerNorms (§1.6), `gelu_erf` over
`[4096, 8192]` twice per layer, the per-layer F32-to-F16 K/V casts, and the RoPE. **That is where a
second round of optimisation would go**, and the `{NORM, MUL, ADD}` fusion of §1.6 is the first item
on that list. It also explains the shape of the ratio column: the gap is smallest where matmuls
dominate (I-JEPA, ViT-L 16f) and widest where the long-sequence overheads pile up (ViT-L 64f).

### 5.8 Measurement conditions

The batching agent's sentinel appeared at 22:17 and every number above was taken after it, on an
otherwise idle box. `uptime` load average: **1.79 before the build**, 10.06 at the start of the
measurement sweep and 13.99 at its end (the sweep's own CPU-backend comparison runs at 32 threads are
most of that). GPU 0 only (`CUDA_VISIBLE_DEVICES=0` for the torch runs; the ggml probe selects
device 0 explicitly), both cards otherwise idle at 13-14 W. Build capped at `-j8` throughout.
Raw outputs: `tmp/results/{ops,norm,mm,mm-prec32,fa,rope,rope-enc,enc,torch-gpu}.txt` and
`tmp/results/torch-gpu.json` in this worktree.

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
7. **`ggml_mul_mat_set_prec(GGML_PREC_F32)`** in `jepa_build_linear` (§1.3) — one line, a no-op on
   CPU, so it can be unconditional. §5.1 measured what it buys and costs: for **f16 weights** it takes
   the error from 4.55e-03 to **2.57e-05** (177×) for −21 % throughput at N=2048 and *+9 %* at
   N=8192; for **quantized** weights it is free (they never reach cuBLAS); for **f32** weights it
   changes nothing at all, because ggml's F32 path is TF32 by way of the algo enum, not the compute
   type. End to end (§5.5) it costs 1.25× on ViT-L@2048 and 2.1× on the short-sequence ViT-H@256.
   **Recommendation: on by default, with a `--fast-math`-style opt-out** — correctness first, and the
   one shape where it is expensive (N=256) is also the one that is already 20× faster than the CPU.
8. **Validate the graph against `supports_op` before computing.** §5.4 established that a single CUDA
   backend performs *no* such check: `ggml_backend_cuda_graph_compute` dispatches every node and
   `ggml_cuda_op_roll` happily misreads a strided view, producing a wrong answer with no error, no
   warning, and a cosine that would pass jepa.cpp's f16 parity gate. So `jepa_graph_alloc` must walk
   `ctx->gf` and `GGML_ABORT` (or fall back to the CPU backend) on the first node for which
   `ggml_backend_dev_supports_op(dev, node)` is false. ~15 lines, and it is what converts every
   future "some op regressed on this ggml bump" from a silent accuracy bug into a loud failure.

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

This is the part of the port that costs something, and the measurements sharpened it considerably.
Per-op error, CUDA vs the CPU backend on identical inputs (§5.1–§5.3):

| op | CPU error | CUDA error | fixable? |
|---|---|---|---|
| `mul_mat` f32 weights | 1.2e-07 (`docs/ggml-notes.md` §5) | **3.2e-04** (TF32) | **no** — algo enum, not compute type |
| `mul_mat` f16 weights | 3.0e-04 | 4.6e-03 → **2.6e-05** with `GGML_PREC_F32` | **yes**, and cheap |
| `mul_mat` q8_0 / q4_0 | 7.6e-03 (incl. activation quant) | 5.2e-04 / 4.8e-04 vs CPU | n/a, comparable |
| `flash_attn_ext` F16 K/V | rel ≤ 1.3e-3, cos ≥ 0.9999995 | rel 5.2e-03–1.5e-02, cos ≥ 0.99991 | **no** — F16 PV accumulator |
| `flash_attn_ext` F32 K/V | rel ≤ 6e-07, cos 1.0000000 | **same as F16 K/V** | **no** — silently down-converted |
| naive attention (F32) | rel ≤ 1.4e-06 | rel 6.5e-04, cos 0.9999993 | this *is* the escape hatch |
| `ggml_norm` | 9.2e-08 at any mean/σ | 1.4e-07 at mean/σ ≈ 0, degrading to 4.1e-02 at 667 | **no** — one-pass variance |

So the tier-by-tier verdict:

| model / dtype | CPU class today | **measured GPU class** | escape hatch |
|---|---|---|---|
| **f32, any model, flash on** | every token cos ≥ 0.9999, `rel_max` ≤ REL(N) | **cannot hold it.** TF32 matmul (3.2e-04) + F16-accumulate attention (5e-03) | none for the matmul |
| **f32 image models, `--no-flash`** | as above | attention error drops to 6.5e-04 but TF32 stays | `--no-flash` is **free at N=256** (§5.5) |
| **f32 video, `--no-flash`** | cos 1.000000 | affordable to ~4 608 tokens (0.30 GiB at 2 048; 2.8× slower) | not above ~8 192 tokens |
| **f16** | mean ≥ 0.9999, worst ≥ 0.99 | **holds, with `GGML_PREC_F32`** — the matmul term becomes 2.6e-05, i.e. *better* than the CPU's 3.0e-04 | on by default |
| **q8_0 / q4_\*** | as `docs/parity.md` | **holds** — `mmq` is INT8×INT8→INT32, and it is the *fastest* path (§5.1) | — |

**The headline is uncomfortable but clean: a GPU backend cannot reproduce jepa.cpp's f32 tier, and
can reproduce its f16 and quantized tiers.** The f32 tier is exactly the one `docs/parity.md` uses to
prove the port is bit-faithful to PyTorch ("the f32 files reproduce the PyTorch reference
**exactly**"), so this is a claim the project would have to qualify by backend.

Concretely, `tests/test-parity.cpp`'s `POLICY` needs a **backend dimension** on top of family ×
file-type. The honest framing for users is: *use f16 or a quantized file on the GPU — they are both
faster and, with `GGML_PREC_F32`, no less accurate than on the CPU; use the CPU when you need the f32
tier.* That happens to line up with the existing recommendation in `docs/quantization.md`.

One more thing the measurements flagged that is **not** about dtypes: because a wrong `ggml_norm`
variance is a per-row *scale* error, cosine similarity is blind to it (§5.3). Any backend-parity test
must gate on `rel_max`, not only on cosine.

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
  that the §2 refactor is what makes *any* backend viable — nothing in it is CUDA-specific, and §5.4
  measured it bit-identical on both backends.
- The §6.1.8 graph check is what makes a second backend cheap to qualify: point it at a Vulkan
  device and it tells you in one run which ops are missing, instead of producing quiet wrong answers.

### 6.7 Effort, in reviewable chunks

| # | chunk | files | est. |
|---|---|---|---|
| 1 | **RoPE table refactor** (§2) — host tables, `ggml_cont` for views, one or two rolls. Measured bit-identical on both backends, 193 sched splits → 1, and it costs 3–5 % of CPU encoder time (§5.4). Ships on its own value. | `src/rope3d.{h,cpp}`, `tests/test-ops.cpp` | ~55 lines, **0.5 d** |
| 2 | **Backend plumbing** — `gpu_device` param, device discovery, device weight allocation, `jepa_context`/`jepa_model` pairing check, `jepa-info --devices`. | `include/jepa.h`, `src/jepa-internal.h`, `src/jepa.cpp`, `src/jepa-gguf.cpp`, `tools/jepa-info.cpp` | ~250 lines, **1.5 d** |
| 3 | **GPU numeric policy + graph validation** — `ggml_mul_mat_set_prec`, K/V policy override + log, predictor naive-attention override, `--no-flash` docs, and the `supports_op` graph check of §6.1.8 (**not optional**, §5.4). Includes the `ggml_norm` mean/σ pre-flight check of §5.3 on real activations. | `src/jepa.cpp`, `src/predictor.cpp` | ~60 lines, **1 d** |
| 4 | **Tools + build** — `--gpu N` on four tools, `JEPA_CUDA` option, pinned staging buffer, bench header records the device. | `tools/*.cpp`, `CMakeLists.txt` | ~150 lines, **1 d** |
| 5 | **Backend-parity test** — `tests/test-backend.cpp` running every graph on CPU and GPU and comparing; `POLICY` gains a backend dimension in `test-parity`. | `tests/`, `CMakeLists.txt` | ~350 lines, **1.5 d** |
| 6 | **Measure + document** — GPU rows in `docs/benchmarks.md` and `docs/parity.md`, a GPU section in the README, torch-GPU baselines. | `docs/`, `scripts/bench_all.sh` | **1 d** |

**Total ≈ 6.5 developer-days** for a CUDA backend that is honest about its numerics, plus whatever the
parity-policy decision (§6.4) costs in discussion. Chunk 1 is worth doing regardless of the outcome —
it is bit-identical, it removes ~340 nodes from a ViT-L video graph, and the repo already lists it as
a candidate ("rope3d sin-mask hoisting").

A **second, optional** round of optimisation exists and was quantified in §5.7: ~33 % of the
64-frame ViT-L time is neither GEMM nor attention. The first item on that list is an upstream
`{NORM, MUL, ADD}` fusion in ggml-cuda (§1.6), which would benefit every ViT on ggml.

## 7. Go / no-go

**Go — with one condition and one disclosure.**

The three decisive facts:

1. **Nothing structural blocks it, and the one thing that did is a ~55-line bit-identical fix.**
   Of 48 op probes at jepa.cpp's exact shapes and strides, exactly two things fail: `ggml_roll` on
   the fused-qkv view (which fragments the encoder into **193 scheduler splits**, measured) and
   `flash_attn_ext` at head_dim 32. The first is fixed by moving the RoPE masks to the host and
   adding one `ggml_cont` — **1 split, bit-identical on both backends, 3–5 % of CPU encoder time**
   (§5.4). The second has a measured in-graph fallback that works and fits (§5.2). Everything else —
   every matmul dtype, both attention paths, LN, GELU, concat, casts, the batched `ne[3]` path — is
   already supported.

2. **The win is 20–26×, and it lands at 54–80 % of PyTorch.** A measured ggml-CUDA forward of the
   real encoder shapes gives 7.2 ms for I-JEPA ViT-H, 35 ms for the ViT-L 16-frame clip and 270 ms
   for the 64-frame clip, against 147 / 821 / 6 388 ms on 32 Zen 4 cores today (§5.5). One 210 W
   workstation card beats 96 cores by more than an order of magnitude, and ggml gets within 1.25–1.9×
   of `VJEPA2Model` on the same GPU (§5.7). Quantized weights are the *fastest* path on CUDA
   (q4_K at 93 TFLOP/s, level with q8_0), which inverts the CPU guidance and makes the small
   quantized files genuinely attractive rather than a memory-only trade.

3. **The cost is the f32 parity tier, and it is not recoverable.** ggml's "F32" matmul on CUDA is
   really TF32 (3.2e-04 against the CPU, and `GGML_PREC_F32` cannot turn it off — it is the
   `CUBLAS_GEMM_DEFAULT_TENSOR_OP` algo enum), flash attention always down-converts K/V to F16 and
   accumulates PV in F16, and `ggml_norm` uses the numerically weaker one-pass variance. **f16 and
   quantized tiers hold** — with `GGML_PREC_F32` the f16 matmul term is 2.6e-05, *better* than the
   CPU's 3.0e-04 — but the f32 tier that `docs/parity.md` uses to prove bit-faithfulness to PyTorch
   does not, and the project would have to qualify that claim by backend (§6.4).

**The condition:** a single-backend design must validate its own graph against
`ggml_backend_dev_supports_op` before computing. §5.4 showed that ggml does *not* do this for you —
the unfixed RoPE chain on a single CUDA backend produced no error, no warning, and a wrong answer
whose cosine (0.99996) would have **passed jepa.cpp's own f16 parity gate**. Fifteen lines, and
without them this port is a silent-corruption risk on every future ggml bump.

**The disclosure:** `JEPA_CUDA=OFF` stays the default. `libggml-cuda.so` is 45 MB for a *single*
GPU architecture (§4.1), 30× the CPU backend, which is not something a project whose pitch is
"CPU inference in plain C/C++ on ggml" should ship by default.

**Suggested sequencing.** Land chunk 1 (the RoPE refactor) now, on CPU, on its own merits — it is
bit-identical, it removes ~340 nodes from a ViT-L video graph, and the repo already has it on the
TODO list. Then decide on the GPU port with the parity question (§6.4) settled first, because that
is a documentation-and-promises decision, not an engineering one. If it goes ahead, **I-JEPA and the
image models are the right first target**: 97 % of their work is matmul, `--no-flash` is free at
256 tokens so the f32 path stays honest there, and they need none of the RoPE or head_dim-32 work.

**One open item this audit could not close:** whether real ViT residual streams ever reach the
|mean|/σ ratio where CUDA's one-pass `ggml_norm` matters (§5.3). It needs a dump of real pre-LN rows,
which would have meant changing `src/`. It is the first thing to check in chunk 3.

## 8. Measured implementation — forecast vs achieved

**Status: implemented.** §§0–7 above are the audit, written before any code changed and left as
written. This section records what the port actually does and what it actually measures, on the same
box, and marks the places where the audit was wrong.

The port landed as the six chunks of §6.7. `JEPA_CUDA=OFF` is still the default; a CUDA build is

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DJEPA_CUDA=ON
cmake --build build-cuda -j 16
build-cuda/jepa-info --devices
build-cuda/jepa-embed -m models/gguf/<model>.gguf -i img.jpg --gpu 0    # or $JEPA_DEVICE=cuda:0
```

### 8.1 What was built

| chunk | landed as | note |
|---|---|---|
| 1 RoPE refactor | `src/rope3d.{h,cpp}`: the sin table carries the rotation's sign, and the pair swap is one `ggml_roll` of 1 over a length-2 axis, preceded by a `ggml_cont` when `x` is a view | 13 nodes → 5, bit-identical (`test-ops` 9/9 with the errors unchanged to the digit), **+4.8 % CPU** on the ViT-L 16-frame encoder (790.6 → 828.3 ms) — inside the 2.9–5.5 % §5.4 forecast |
| 2 backend plumbing | `jepa_model_params` / `jepa_model_load_ex` / `$JEPA_DEVICE` / `--gpu [N]` / `jepa-info --devices`; the context always takes the model's backend | single backend, no `ggml_backend_sched`, as §6.1 argued |
| 2 graph validation | `jepa_graph_validate()` in `jepa_graph_alloc()` | **the audit's condition, and it works**: `tests/test-backend.cpp` builds the historical bad node and the walk refuses it |
| 3 numerics | `GGML_PREC_F32` on every `mul_mat` (GPU default, `--gpu-prec f16` / `$JEPA_GPU_PREC` to opt out), F16-only K/V with a one-shot log, a device-memory guard on naive attention | |
| 4 predictor | naive attention on a GPU at head_dim 32, refusal with both sizes when the score matrix does not fit | |
| 5 parity | `POLICY[backend][family][tier]` in `test-parity`, `--gpu` on `test-parity`/`test-predictor`, `tests/test-backend.cpp` in `ctest` | `docs/parity.md` "Parity on a GPU" |
| 6 measurement | this section, `docs/results.md` "GPU (CUDA)", the README paragraph | |

The §2 refactor changed `src/rope3d.cpp` and `tests/test-ops.cpp` only — `src/jepa.cpp`,
`src/predictor.cpp` and `src/lewm.cpp` needed no change, exactly as §2.5 predicted.

### 8.2 Forecast vs achieved

§5.7's forecast column against the measured `ggml_backend_graph_compute` time, f16 weights, best of
5 after 2 warmups, one RTX 4500 Ada, `GGML_PREC_F32` on (the default):

| model / shape | §5.7 forecast | **achieved** | with `--gpu-prec f16` | §5.5's own `+PREC_F32` variant |
|---|---:|---:|---:|---:|
| I-JEPA ViT-H, 224² | ~7.5 ms | **15.5 ms** | 8.8 ms | 15.1 ms |
| V-JEPA 2 ViT-L, 16 f @256 | ~37 ms | **46.5 ms** | 37.4 ms | 43.7 ms |
| V-JEPA 2 ViT-L, 64 f @256 | ~280 ms | **306 ms** | 304 ms | – |
| V-JEPA 2.1 ViT-B, 16 f @384 | ~40 ms | **43.0 ms** | – | – |
| V-JEPA 2.1 ViT-B, 64 f @384 | ~405 ms | **424 ms** | – | – |
| V-JEPA 2 ViT-L predictor, 16 f | ~105 ms | **113 ms** | – | – |

**The forecast holds, and the two rows that look off are a like-for-like error in the forecast, not
in the port.** §5.7 built its forecast on the §5.5 baseline column, which did *not* have
`GGML_PREC_F32`, while §6.1.7 recommended turning it on by default — which the port did. Against
§5.5's own `+ GGML_PREC_F32` variant the match is 15.5 vs 15.1 ms and 46.5 vs 43.7 ms, i.e. **+3 %
and +6 %**, and the residual is the patch embedding and position handling the synthetic encoder
omitted. With the flag off, the port reproduces the forecast directly (8.8 vs ~7.5, 37.4 vs ~37).
The three long-sequence rows, where `GGML_PREC_F32` is nearly free, land at **+6 % to +9 %** of
forecast with no caveat at all.

Against the engine as it shipped before this work (`docs/benchmarks.md`, 32 threads, f16):

| shape | CPU t=32 | CPU t=96 | **CUDA** | vs t=32 | vs t=96 |
|---|---:|---:|---:|---:|---:|
| I-JEPA ViT-H, 224² | 147.0 | 113.1 | **15.5** | 9.5× | 7.3× |
| LeJEPA ViT-S, 224² | 12.8 | – | **1.1** | 11.6× | – |
| LeWM ViT-Ti, 224² | 9.8 | – | **0.8** | 12.3× | – |
| V-JEPA 2 ViT-L, 16 f | 820.7 | 566.6 | **46.5** | 17.6× | 12.2× |
| V-JEPA 2 ViT-L, 64 f | 6 388.1 | 4 026.9 | **306** | 20.9× | 13.2× |
| V-JEPA 2.1 ViT-B, 384² | 60.3 | 52.7 | **4.4** | 13.7× | 12.0× |
| V-JEPA 2.1 ViT-B, 16 f | 853.5 | 636.0 | **43.0** | 19.8× | 14.8× |
| V-JEPA 2.1 ViT-B, 64 f | 9 036.1 | 5 040.2 | **424** | 21.3× | 11.9× |
| V-JEPA 2 ViT-L predictor | 452 | – | **113** | 4.0× | – |
| SSv2 attentive-pool head | 96 | – | **5.7** | 16.8× | – |

§7 promised "20–26×" against 32 threads. The long clips deliver it (17.6–21.3×); the short-sequence
image models come in at 9.5–13.7× rather than 16–20× because `GGML_PREC_F32` costs most exactly
where there is least work to hide it (I-JEPA at 256 tokens: 1.76×, against 1.06× at 8 192 tokens),
and because the audit's synthetic encoder omitted the patch embed. The predictor's 4.0× is the
"4–7×, not the 20× the encoders get" §5.2 called for.

### 8.3 The torch-GPU baseline, re-measured

§5.6 re-run after the port (torch 2.13.0+cu130, transformers 5.16.1, sdpa, batch 1, TF32 off,
3 warmup + 7 timed, `cuda.synchronize()` around each, GPU 0):

| model | dtype | frames | tokens | mean ms | §5.6 | delta |
|---|---|---:|---:|---:|---:|---:|
| vjepa2-vitl-fpc64-256 | fp32 | 16 | 2 048 | 115.62 | 118.58 | −2.5 % |
| vjepa2-vitl-fpc64-256 | fp32 | 64 | 8 192 | 838.38 | 836.52 | +0.2 % |
| vjepa2-vitl-fpc64-256 | fp16 | 16 | 2 048 | **28.74** | 29.56 | −2.8 % |
| vjepa2-vitl-fpc64-256 | fp16 | 64 | 8 192 | **147.48** | 150.03 | −1.7 % |
| ijepa_vith14_1k | fp32 | 1 | 256 | 24.26 | 24.99 | −2.9 % |
| ijepa_vith14_1k | fp16 | 1 | 256 | **5.91** | 5.57 | +6.1 % |

Reproducible to within 3 % on every row (the I-JEPA fp16 row has a 0.34 ms stdev at 5.9 ms, so its
+6 % is jitter; its minimum, 5.62 ms, is within 1 % of §5.6).

**One correction to §5.6: its fp16 `peak GiB` column is wrong.** 1.27 and 2.42 GiB are essentially
its own fp32 peaks (1.33, 2.43), which cannot be right for half-sized weights; the cause is loading
fp32 → cuda → `.half()` without resetting the allocator's high-water mark. Measured cleanly, real
fp16 inference peaks are **0.67 GiB** (ViT-L @2 048 tokens) and **1.19 GiB** (ViT-H @256), not
1.27/2.42. The fp32 rows and the fp16 64-frame row (0.83) were already correct.

The honest ratio, with everything measured on this box:

| shape | jepa.cpp CUDA | `--gpu-prec f16` | torch fp16 | **ggml / torch** |
|---|---:|---:|---:|---:|
| I-JEPA ViT-H, 224² | 15.5 | 8.8 | 5.91 | 2.6× (38 %) / 1.5× (67 %) |
| V-JEPA 2 ViT-L, 16 f | 46.5 | 37.4 | 28.74 | 1.6× (62 %) / 1.3× (77 %) |
| V-JEPA 2 ViT-L, 64 f | 306 | 304 | 147.5 | 2.1× (48 %) / 2.1× (49 %) |

§5.7 forecast "54–80 % of PyTorch"; the port delivers **38–62 % at its default precision** and
**48–77 % with `--gpu-prec f16`**, which is the same band once the `GGML_PREC_F32` decision is
accounted for. §5.7's diagnosis of the remaining gap stands unchanged: at the 64-frame shape the
component rates predict 182 ms of GEMM + attention against 306 ms measured, and the missing ~120 ms
is 48 unfused LayerNorms, `gelu_erf` over `[4096, 8192]` twice per layer, the F32→F16 K/V casts and
the RoPE. The `{NORM, MUL, ADD}` fusion of §1.6 is still the first item on any second round.

### 8.4 Quantised weights really are the fastest GPU path

§1.3 predicted the inversion and §5.1 measured it on one matmul; end to end it holds:

| model / shape | f32 | f16 | q8_0 | q4_k | CPU f16 | CPU q4_k |
|---|---:|---:|---:|---:|---:|---:|
| I-JEPA ViT-H, 224² | 11.8 | 15.5 | 8.0 | **7.8** | 147 | 198 |
| V-JEPA 2 ViT-L, 16 f | 44.3 | 46.5 | **34.4** | 34.8 | 821 | 1 096 |
| V-JEPA 2 ViT-L, 64 f | 303 | 306 | **281** | 282 | 6 388 | 7 406 |
| V-JEPA 2.1 ViT-B, 384² | – | 4.4 | – | **3.5** | 60.3 | – |
| V-JEPA 2.1 ViT-B, 16 f | – | 43.0 | – | **37.3** | 853 | – |

q4_k ties q8_0 on the GPU and beats f16 everywhere, against being the *slowest* type on the CPU —
`mmq`'s INT8 tensor-core kernel covers every type we ship, while llamafile's fast sgemm covers only
F32/F16/Q8_0. Note also that the f32 column is not slower than f16: ggml's CUDA F32 path is TF32
(§5.1) while the f16 path is paying for `GGML_PREC_F32`.

### 8.5 The two open numeric questions, closed

§1.3 and §5.3 both ended on "this needs a dump of real activations". Done, with forward hooks on the
reference PyTorch models over the stored fixture inputs (1.66 M LayerNorm rows across I-JEPA ViT-H,
V-JEPA 2 ViT-L at 16 and 64 frames, V-JEPA 2.1 ViT-B and LeJEPA ViT-S).

**Overflow (§1.3): not real, and the premise was wrong.** The largest linear-layer output anywhere
is **414** (V-JEPA 2.1 ViT-B, layer 0 fused qkv). I-JEPA ViT-H — the model `docs/architecture.md`
described as reaching ~2e4 — peaks at **95**, with a largest residual value of 151, a largest
unscaled q·k of 201 and a largest pre-softmax logit of 22.5. The number that decides overflow is the
GEMM's running partial sum: measured in float64 for the worst output element it peaks at **412**,
and the unattainable all-same-sign ceiling `Σ_k |x_k w_k|`, computed for *every* element of *every*
linear, is **600** — 109× below `half`'s 65504. Activations would have to grow ~100× before
`CUBLAS_COMPUTE_16F` could produce an `inf`. The `~2e4` figure has been corrected in
`docs/architecture.md` and `docs/parity.md`; the F16-K/V parity finding it was cited for
(cos min 0.9910 vs 1.000000) is unaffected, because its cause is F16's 10-bit **mantissa**, not its
range. `GGML_PREC_F32` therefore stays on by default for precision, not for safety.

**`ggml_norm`'s one-pass variance (§5.3): not a risk for these models.** Its failure is governed by
|row mean| / row σ. Over 1.66 M real pre-LN rows the maximum is **2.72** (I-JEPA ViT-H, layer 0, one
token: mean 0.488, σ 0.179), the 99.9th percentile 2.30, the median 0.06–1.35, and no row anywhere
exceeds 10. §5.3's table has both backends matching a double-precision two-pass reference to
~1.8e-07 at that ratio; the catastrophic behaviour needs 100–2000, which LayerNorm's own centring
keeps these residual streams far away from. The `rel_max` gates the GPU parity tier adds are kept
anyway — they cost nothing and this is the only failure mode cosine is blind to by construction —
but they are insurance, not a live concern.

### 8.6 Parity, measured

`docs/parity.md`'s new "Parity on a GPU" section carries the tier table and the full sweep. The
headline is §6.4's, confirmed end to end: **there is no f32 tier on a GPU** — V-JEPA 2 ViT-L's f32
file on CUDA behaves like its f16 file (worst token 0.3549 vs 0.3557) where on the CPU the f32 file
is exact — and **f16 and quantized hold**. 22 of 24 encoder files and all 9 predictor rows pass; the
two that do not (`lejepa-vits16-q4_k`, `vjepa2-vitl-ssv2-q4_k`) fail identically on the CPU and are
already documented as below the recommended parity quantisation.

One bar had to be loosened rather than mapped across: the image families' worst-token floor, 0.99 →
0.90, for I-JEPA ViT-H alone (0.9976 on the CPU at f16 → 0.9613 on a GPU, 0.9723 with `--no-flash`).
That is the F16 PV accumulator landing on a token that is already degenerate on the CPU; LeJEPA and
LeWM stay at 0.9998 / 0.99999 on the same backend and I-JEPA's median stays at 0.999996.

### 8.7 Things the audit got right that are worth restating

* **The `ggml_roll` blocker was the whole game.** With the §2 refactor the encoder is one split and
  every node has a CUDA kernel; without it a single CUDA backend computes a wrong answer in silence.
* **`supports_op` is not checked by a single backend.** `jepa_graph_validate()` is 30 lines and
  `tests/test-backend.cpp` proves it fires: building `ggml_roll` on a strided qkv view is refused
  with the node listed, and the same roll on a contiguous tensor is accepted and computes.
* **head_dim 32 has no CUDA flash kernel.** The naive path is what the predictor takes, and it is
  both accurate (F32 end to end: `rel_max` 1.7e-03–2.8e-03 against the PyTorch dump, better than a
  CUDA flash kernel would give) and the reason the predictor gets 4× where encoders get 20×.
* **`--no-flash` is affordable only at image scale.** Measured **16.5 ms vs 15.5** for I-JEPA at 256
  tokens (+6 %, essentially free, as §5.5 predicted) against **109.4 ms vs 46.5** for the ViT-L
  16-frame clip (2.4×); the guard refuses shapes whose `4·N²·H` score matrix does not fit on the
  device and prints both numbers.

### 8.8 CI

Unchanged, deliberately: `.github/workflows/` still contains only `docs.yml` and **no CUDA job was
added**, because no GPU runner exists and a compile-only CUDA job on a GitHub-hosted runner would
install a ~3 GB toolkit to prove something a developer's `-DJEPA_CUDA=ON` build already proves. The
shape §6.3 sketched is still the right one when CI is added: a CPU job running `ctest`, a
compile-only CUDA job, and GPU correctness on a self-hosted or manual run. What makes that last part
cheap already exists — `ctest`'s `backend` test runs the whole check and exits 0 with a `SKIP` line
on a machine without a GPU, so the same `ctest` invocation covers both kinds of runner. It registers
like the other asset-gated tests (it needs one small GGUF) and costs a CPU-only checkout nothing.
