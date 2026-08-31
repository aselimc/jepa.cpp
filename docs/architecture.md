# Architecture

jepa.cpp is an inference engine and nothing else: C++17, ggml (submodule at `ggml/`), no other runtime
dependency, no training code. Python appears only in conversion and reference dumping.

## The engine in one page

A model is a **GGUF bundle**: one file holding an encoder, optionally a predictor and optionally a
head, together with the metadata that describes them — dimensions, depth, head count, patch and
tubelet geometry, positional-encoding scheme, preprocessing recipe, class labels, licence. The
[GGUF schema](gguf-schema.md) is the contract; the converter, the loader and the graph builder all
implement exactly it.

The graph is **built from that metadata, not from per-model code**. `jepa_model_load` reads the
hyperparameters and the tensor table; `jepa_context_new` allocates a compute arena; `jepa_encode`
builds a ggml graph for the requested shape and runs it on the context's backend. A new checkpoint of
a known family therefore needs no C++ change — the converter writes the metadata and the loader picks
the branches, which is why six models of five families share one builder.

Around that sit four layers: preprocessing (pixels to a normalised tensor, host-side), the shared ViT
graph, the optional heads, and pooling. The public surface is one C header, `include/jepa.h`
([reference](api.md)).

## The shared ViT graph

Every family runs the same encoder graph:

```
tokens = patch_embed(rearranged pixels) [+ pos_embed] [+ cls/reg] [+ modality vec]
for each block:  x += attn(ln1(x)) ;  x += ffn(ln2(x))     # pre-LN, GELU(erf), qkv bias, optional layer-scale
x = norm(x)
```

The patch "convolution" is a host-side rearrangement of the pixels into `[N, C·T·P·P]` followed by one
`ggml_mul_mat` — mathematically identical to the strided convolution of the reference implementations
and far friendlier to a matmul-shaped backend. Attention always goes through `ggml_flash_attn_ext`
unless it is explicitly disabled. Positions are added either as a baked table or as rotary factors
applied inside the attention, depending on the family.

Heads are separate graphs over the encoder output: the attentive pooler plus 174-way classifier for
SSv2, the masked predictor for V-JEPA 2 / 2.1, and the action-conditioned predictor for LeWorldModel.

## Family matrix

| family | tokenizer | positions | extras |
|---|---|---|---|
| `ijepa` | 2-D patch 14/16 | `sincos2d` table, baked by the converter | no CLS; the feature is the mean of the patch tokens |
| `vjepa` | 2×16×16 tubelet | `sincos3d` table | — |
| `vjepa2` | 2×16×16 tubelet | `rope3d`, tiled layout | predictor 384-d / 12 L / 10 mask tokens; attentive-pool head |
| `vjepa2_1` | 2×16×16 tubelet **and** 1×16×16 image embed | `rope3d`, interleaved layout, plus `interpolate_rope` | modality vectors; per-hierarchy-layer norms (inference uses the last); 8 mask tokens |
| `hfvit` | 2-D patch | `learned`, with CLS and optional registers | DINOv2-style; optional layer-scale; the LeJEPA community checkpoint and the LeWM encoder |
| `lewm` | `hfvit` encoder (ViT-Ti/14) | inherited from `hfvit` | predictor 192-d / 6 L over 3 frames plus an action embedding (10 → 192); BatchNorm MLPs folded at conversion |

## 3-D RoPE

The V-JEPA 2 and 2.1 encoders and predictors use a three-axis rotary embedding that must match
Hugging Face's `VJEPA2RopeAttention` exactly.

- Per axis, `d = 2 · ((head_dim // 3) // 2)` dimensions are rotated — 20/20/20 for `head_dim` 64, with
  the remaining 4 dimensions untouched. The predictors have `head_dim` 32, so `d = 10` per axis.
- Token *i* maps to `t = i // (gh·gw)`, `h = (i % (gh·gw)) // gw`, `w = i % gw`.
- Frequencies are `omega_k = 1 / 10000**(2k/d)` for `k` in `[0, d/2)`.
- The rotation acts on **interleaved pairs** `(x[2k], x[2k+1])`: `x·cos + rotate90(x)·sin`, where
  `rotate90(y1, y2) = (−y2, y1)`.
- It is applied to q and k (not v), after the qkv projection and before scaling.

**Two cos/sin table layouts exist, and the family selects one** (`jepa.enc.rope_freq_layout`,
`jepa_rope3d_params::variant`):

| layout | table | used by |
|---|---|---|
| tiled | `table[j] = f(pos · omega_{j mod d/2})` — the two members of a pair see *different* angles | V-JEPA 2 (both the Hugging Face and the Meta implementation) |
| interleaved | `table[2k] = table[2k+1] = f(pos · omega_k)` — a true rotation | V-JEPA 2.1 |

The tiled layout is a quirk Meta's own code documents and keeps for weight compatibility. It is not
optional and not interchangeable: applying the wrong one gives per-token cosine ≈ 0.63 against the
V-JEPA 2 reference and ≈ 0.91 against the 2.1 reference — the graph runs, the numbers look plausible,
and the output is wrong. The full derivation with source excerpts is in the
[V-JEPA tensor and RoPE notes](vjepa-notes.md); the numpy reference is
`scripts/jepa_convert/vjepa2_numpy_ref.py`.

The V-JEPA 2.1 **encoder** additionally rescales `h` and `w` (not `t`) by
`(rope_ref_grid − 1) / (grid − 1)` — `interpolate_rope`, with `rope_ref_grid = 256 / patch_size`, i.e.
16 for the released 384-px checkpoints and *not* `img_size / patch`. Its predictor does not rescale.

Implementation: ggml's `ggml_rope_multi` uses half-split rotation and a single theta scale and is
therefore unusable here. `src/rope3d.cpp` precomputes `[N, head_dim]` cos/sin tables on the host and
applies them with stock ops — one `ggml_roll` for the pair swap plus a multiply and an add, with the
sign of the rotation carried in the sin table. `jepa_rope3d_tables_ids` produces rows for masked or
subsampled token ids, which is what the predictors need. `tests/test-ops.cpp` checks both layouts
against `tests/vectors/rope3d/`, generated by `scripts/gen_rope_ref.py`.

## Preprocessing

`src/preprocess.cpp` turns pixels into the normalised tensor the graph expects, driven by the
`jepa.pre.*` metadata: shortest-edge resize, centre crop, scale to `[0, 1]`, then subtract the
per-channel mean and divide by the per-channel std. `jepa_resize_antialias_u8` is a faithful port of
the integer antialiased path that `torchvision.transforms.v2.functional.resize(antialias=True)` runs
on x86 CPUs, so the tensor entering the network is **bit-exact** against the reference pipeline given
the same decoded pixels ([measured](parity.md#preprocessing-parity)).

The caveat is decoding, not resizing. jepa.cpp decodes JPEGs with `stb_image`, which differs from
PIL/libjpeg by ±1–2 levels on about 2 % of pixels. That decoder floor is `1 − cos` of 6.3e-6 to
1.7e-5 on the final features — larger than what f16 weights add. Video frames are decoded outside the
engine and handed in as uint8, so the video path has no decoder floor at all.

## Attention and precision

Attention runs through `ggml_flash_attn_ext`. The K/V dtype is **auto**: F32 K/V for f32 model files,
F16 K/V for f16 and quantized files.

- The F16 cast alone costs roughly three digits of worst-token cosine on I-JEPA ViT-H at f32
  (0.9910 against 1.000000, `rel_max` 3.5e-2 against 8.6e-5). The cause is F16's 10-bit mantissa, not
  its range: on the real forward the model's largest linear output is 95 and its largest residual
  value 151, against F16's ceiling of 65504.
- For f16 and quantized files weight rounding dominates anyway, and F16 K/V is pure storage rounding —
  cosine ≥ 0.9999995 against a double-precision reference.
- The attentive-pool cross-attention always uses F32 K/V: it has a single query row, so ggml takes its
  per-row kernel, which with F16 K/V would round q and the PV accumulator to F16. At one query row it
  costs nothing. The same reasoning applies to the LeWM predictor's short sequences.
- `--no-flash` selects the naive `mul_mat` + `soft_max_ext` path. It is fully F32 and ~15–30 % slower
  on ViT-H, and its score matrix is `4·N²·H` bytes — 15.3 GB at 18 432 tokens against ~0.05–0.1 GB for
  flash, so it is a debugging and short-sequence option, not a general one.

Overrides: `JEPA_KV_F16` / `JEPA_KV_F32` (`--kv-f16` / `--kv-f32` in the tools).

On the CUDA backend the picture differs and jepa.cpp cannot change it:

| | CPU | CUDA |
|---|---|---|
| `mul_mat`, f32 weights | strict FP32 | **TF32** — ggml passes `CUBLAS_GEMM_DEFAULT_TENSOR_OP` to every `cublasGemmEx` and never calls `cublasSetMathMode`, and `ggml_mul_mat_set_prec` only selects the compute type |
| `mul_mat`, f16 weights | F32 accumulation | F16 accumulation unless `GGML_PREC_F32` is set, which jepa.cpp sets by default on a GPU |
| `flash_attn_ext` | F32 K/V honoured | K/V always converted to F16, PV accumulator `half2`; `ggml_flash_attn_ext_set_prec` is a no-op |
| `ggml_norm` | centred two-pass variance | one-pass `E[x²] − mean²` |

`GGML_PREC_F32` on every `mul_mat` is the GPU default and takes the f16 matmul error from 4.6e-03 to
2.6e-05, i.e. below the CPU's own 3.0e-04. `$JEPA_GPU_PREC=f16` opts out; on the CPU the call is a
no-op. The one-pass variance is a per-row *scale* error that cosine is structurally blind to, which is
why every GPU parity tier gates `rel_max` as well; over 1.66 M real pre-LayerNorm rows the ratio that
governs its failure peaks at 2.72, far from the 100–2000 where it degrades.

## Batching

`jepa_encode` folds up to `jepa_context_max_batch()` image items (default 32, `$JEPA_MAX_BATCH`,
`jepa-embed --batch`) into **one** graph on ggml's batch dimension: activations are `[D, N, B]`,
attention tensors `[head_dim, n_head, N, B]`, and both `ggml_flash_attn_ext` and `ggml_mul_mat` walk
`ne[3]` as an outer loop, so items never mix.

Guarantees:

- On the CPU the batched rows are **bit-identical** to the per-item path at every dtype measured
  (`tests/test-batch`, ctest entry `batch`).
- On CUDA batched and per-item runs agree to ~1e-7 cosine but are not bit-identical, because GEMM
  tiling varies with the batch shape.

Video is deliberately excluded. What batching buys is the fixed per-call cost, ~5 ms on the image
models; a V-JEPA 2 clip runs 908–9 000 ms, so the saving is under 1 %, while a batched clip graph
would multiply the largest arena in the engine. `jepa_encode` keeps a per-clip loop for video, and
`n_batch = 2` is two calls' worth and bit-identical to two `n_batch = 1` calls
(`tests/test-batch --video`). Inputs larger than the cap are encoded in several graphs rather than one
giant one.

## GPU backend

A CUDA build (`-DJEPA_CUDA=ON`) runs the same graphs on a device. The design has three load-bearing
properties.

**One backend, never a split graph.** A context owns exactly one backend and the model's weights live
on it; `ggml_backend_sched` is not used. Its job is to *partition* a graph, and a partitioned jepa.cpp
graph is a defect, not a feature — one unsupported node per transformer block would shuttle the fused
qkv tensor across PCIe four times per block. Device selection goes through the backend-agnostic ggml
registry (`ggml_backend_load_all`, then the *N*-th device of type GPU), so no CUDA header appears in
jepa.cpp and the same code path would light up another backend if one were built. A context whose
device differs from its model's is rejected rather than silently split.

**Graph validation is mandatory.** A single CUDA backend performs no `supports_op` check of its own:
it dispatches every node, and a kernel handed a tensor it cannot really run returns a *wrong answer
with no error and no warning* — an answer that can still pass an f16 cosine gate.
`jepa_graph_validate()` therefore walks the graph before every compute and refuses the first node for
which `ggml_backend_dev_supports_op` is false, naming it. `tests/test-backend.cpp` builds a node known
to have that property and checks the refusal fires. `$JEPA_VALIDATE_GRAPH=0` disables the walk and is
a debugging switch only.

**The predictors take the naive attention path.** No CUDA flash-attention kernel exists for
`head_dim` 32, which is what the V-JEPA 2 and 2.1 masked predictors have, so `src/predictor.cpp`
selects `mul_mat + soft_max_ext` there. That path is fully F32 and stays one graph split — it is
*more* accurate than a CUDA flash kernel would be — but it runs at roughly 3 TFLOP/s against flash's
50–70, which is why the predictor gains 4× on a GPU where the encoders gain 20×. Its score matrix is
`4·N²·H` bytes over `n_context + n_target` rows; the engine checks that against device memory and
refuses the shape with both sizes printed rather than failing inside the allocator.

**There is no f32 tier on a GPU.** TF32 matmuls and F16-accumulating flash attention are backend
properties, so an f32 GGUF on CUDA behaves like its f16 file and is judged with the f16 bars. f16 and
quantized files hold their own bars there, and quantized weights are additionally the fastest GPU
path, because ggml's CUDA `mmq` INT8 tensor-core kernel covers every type jepa.cpp ships. Use the CPU
when f32 exactness is the requirement.

## Runtime switches

| variable | tool flag | effect |
|---|---|---|
| `JEPA_DEVICE=cuda:N` \| `cpu` \| `N` | `--gpu [N]` | select the compute device; default CPU |
| `JEPA_GPU_PREC=f16` | `--gpu-prec f16` (`jepa-bench`) | opt out of the default `GGML_PREC_F32` matmuls on a GPU. Bench-only and not parity-gated: read its numbers as a measured upper bound, not a shipping configuration |
| `JEPA_VALIDATE_GRAPH=0` | — | disable the pre-compute graph validation. Debugging only — without it an unsupported node on a single CUDA backend computes a silently wrong answer |
| `JEPA_MAX_BATCH` | `--batch B` | image items per encoder graph; default 32 |
| `JEPA_MAX_GRAPH_MIB` | — | cap on the compute arena; larger inputs are split across graphs |
| `JEPA_KV_F16` / `JEPA_KV_F32` | `--kv-f16` / `--kv-f32` | override the automatic flash-attention K/V dtype (CPU; on CUDA F16 is forced and the request is logged once) |

`jepa_context_set_mul_mat_prec_f32(ctx, false)` is the API form of `JEPA_GPU_PREC=f16`.

## Repository layout

```
include/jepa.h        public C API (opaque handles, plain structs)
src/jepa.cpp          model struct, GGUF load, graph build, run (encoder / predictor / head)
src/jepa-gguf.cpp     GGUF metadata + tensor lookup helpers, hparam parsing
src/preprocess.cpp    image/video -> normalised float tensors (stb_image; resize matching PIL/torchvision)
src/rope3d.cpp        3-D RoPE: cos/sin table generation + graph application
src/predictor.cpp     masked predictors (V-JEPA 2 / 2.1)
src/lewm.cpp          LeWorldModel projector and action-conditioned predictor
src/npy.h             header-only .npy reader/writer (fixtures, tool output)
third_party/          nlohmann/json 3.12 (manifests, hparam dumps); stb_image*.h for decode/resize

tools/jepa-info       print GGUF hparams/tensors, list devices
tools/jepa-embed      image/video -> features (.npy / text)
tools/jepa-classify   video -> top-k labels (attentive-pool head)
tools/jepa-worldmodel LeWM: image -> state -> K-step action rollout; --ref-check against fixtures
tools/jepa-quantize   f32/f16 GGUF -> q8_0 / q4_k / ...
tools/jepa-bench      timing: encoder / head / predictor / lewm-step / lewm-rollout, --md / --json

tests/test-parity     replay the golden dumps: cosine / max-abs / top-k, non-zero exit on regression
tests/test-predictor  the same for the three predictors, against the reference encoder tokens
tests/test-batch      batched vs per-item bit-exactness
tests/test-attn       flash vs naive attention against a double-precision reference; K/V policy; timing
tests/test-ops        rope3d and friends against tests/vectors/
tests/test-backend    GPU graph validation and CPU/GPU agreement; skips cleanly without a GPU

scripts/convert.py          HF safetensors / torch.hub .pt -> GGUF
scripts/dump_reference.py   PyTorch golden outputs -> tests/fixtures/ref/<model>/
scripts/compare.py          .npy / ref-dir comparison (cosine, max-abs, rel, top-k)
scripts/knn_eval.py         the frozen-feature k-NN protocol shared by both accuracy benchmarks
scripts/bench_all.sh + gen_benchmarks_md.py       the benchmark sweep and its document generator
scripts/bench_accuracy_{image,video}.py           the Imagenette / UCF-101 sweeps
```

Scratch files go in `tmp/` and weights in `models/`; both are git-ignored.

## Testing and parity methodology

Correctness is measured against PyTorch, per model, per dtype, per backend.

**Golden dumps.** `scripts/dump_reference.py --model <name>` runs the reference implementation and
writes `tests/fixtures/ref/<name>/manifest.json` plus one float32 C-order `.npy` per tensor per
sample: the preprocessed input (layout recorded in the manifest), `last_hidden_state`, the pooled
feature, and where a head exists the pooler output and logits. Video samples also carry the raw
sampled frames as uint8, so the engine's own preprocessing can be exercised from identical pixels.
The per-model tensor lists are in [fixtures](fixtures.md).

**Two passes per file.** `test-parity <model.gguf> <ref dir>` first feeds the **stored preprocessed
tensor**, bypassing jepa.cpp's preprocessor, and reports per-token cosine, `rel_max` and top-1/top-5
agreement; then it runs its own preprocessor from the stored pixels and reports the same. The first
pass isolates the graph; the second adds the preprocessing and, for images, the JPEG-decoder floor.

**Thresholds are a table, not a constant.** `POLICY` in `tests/test-parity.cpp` is indexed by
**backend × family class × file-type tier**, and `test-parity` prints the row it judged with. Family
class matters because the long low-cosine token tail of the f16 and quantized *video* encoders is a
property those checkpoints have and the image ViTs do not; tier comes from `general.file_type`, with
anything below 8 bits per weight treated as advisory — reported, with only the derived tensors and the
top-1 label gated. `tests/test-predictor.cpp` uses the image-family rows. The full tables, CPU and
GPU, are in [parity](parity.md#thresholds-per-backend-model-family-file-type-tier).

**`REL(N) = max(1e-3, 1e-3·√(N/2048))`** is the f32 `rel_max` bound. `rel_max` is a max-abs
difference, and max|a−b| grows with the length of the reductions feeding it (~√N over N tokens) while
the cosine does not. The 8192-token V-JEPA 2 clip measures 1.22e-3 with cosine 1.000000 on *every*
token; a flat 1e-3 bound would call that a failure. The widened bound still sits ~40× above the
observed f32 noise floor: perturbing one encoder weight tensor by +1 % pushes a clean 4.0e-5 run to
2.09e-3 and fails.

**Video tiers gate the median.** For f16 and quantized *video* files the token map is judged on the
median per-token cosine rather than the worst token, because the tail is a checkpoint property that no
implementation can remove, while the median still collapses for a real graph defect — a wrong RoPE
layout puts *every* token at ~0.63. The strict bars live on the pooled and logit outputs, which is
where the tail does not reach. Every f32 file keeps the hard "every token ≥ 0.9999" bar, and every
family ships one, so each family stays anchored at a configuration with no slack.

**GPU tiers gate `rel_max` in every cell**, not only at f32. A wrong `ggml_norm` variance is a per-row
scale error, and cosine cannot see it: a relative error of 5.7 has been measured with per-row cosine
still reading 1.0000000. The GPU bars are otherwise set just outside the worst measured fixture value,
and two of the image-family bars are looser than their CPU counterparts, which makes the GPU image
tier roughly half as sensitive to weight-level errors as the CPU one — the f32-on-CPU run stays the
sensitive configuration for any real investigation.

**What else the tests cover.** `test-batch` checks batched-versus-per-item bit-exactness, `test-ops`
the RoPE tables against generated vectors, `test-attn` flash attention against a double-precision
reference including the K/V dtype policy, and `test-backend` the GPU graph validation plus CPU/GPU
agreement. `test-predictor` adds structural checks the reference cannot provide: causal-prefix
equality and rollout-versus-predict identity on LeWorldModel, which are bit-exact on both backends.

Results: [Accuracy](accuracy.md) for the curated view, [parity](parity.md) for every row.
