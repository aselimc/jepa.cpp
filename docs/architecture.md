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
the branches, which is why nine models of six families share one builder.

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
| `vjepa2` | 2×16×16 tubelet | `rope3d`, tiled layout | predictor 384-d / 12 L / 10 mask tokens; attentive-pool head. ViT-g/16 (1408 d, 40 L, **22 heads**, ffn 1408·48/11 = 6144) is the widest member |
| `vjepa2` + `jepa.pred.kind = ac` | 2×16×16 tubelet | `rope3d`, tiled layout | V-JEPA 2-AC: the same encoder plus an **action-conditioned** predictor (1024-d / 24 L / 16 heads) that takes a 7-d action and a 7-d pose per frame and predicts the next frame's latents. Block-causal over frames; no mask tokens, no position table |
| `vjepa2_1` | 2×16×16 tubelet **and** 1×16×16 image embed | `rope3d`, interleaved layout, plus `interpolate_rope` | modality vectors; per-hierarchy-layer norms (inference uses the last); 8 mask tokens |
| `hfvit` | 2-D patch | `learned`, with CLS and optional registers | DINOv2-style; optional layer-scale; the LeJEPA community checkpoint and the LeWM encoder |
| `lewm` | `hfvit` encoder (ViT-Ti/14) | inherited from `hfvit` | predictor 192-d / 6 L over 3 frames plus an action embedding (10 → 192); BatchNorm MLPs folded at conversion |
| `levjepa` | 1×16×16 tubelet (one token per frame per patch) | `rope3d`, tiled layout; CLS gets none | CLS token, block-causal attention mask; no predictor, no head |

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

That five-node form replaced a thirteen-node one, bit-identically, and it is a deliberate trade: it
costs **+4.8 % on the CPU** at the ViT-L 16-frame encoder shape (790.6 → 828.3 ms) and it is what
keeps the chain resident on a GPU. The older form emitted a `ggml_roll` over a strided view of the
fused qkv tensor, which no CUDA kernel accepts — see the graph-validation paragraph under
[GPU backend](#gpu-backend) for what a backend does with such a node when nothing checks.

## Block-causal attention

`jepa.enc.attn_mode` selects how the encoder's attention is masked. Every family but one leaves it at
`full`, where the key is absent from the graph. LeVJEPA sets `block_causal`, and for that family the
mask is part of the model, not a tuning knob: run the same weights unmasked and the CLS feature drops
to cosine 0.945 against the reference with a worst patch token of 0.834.

The rule is one line — a patch query may attend a patch key iff `frame_id(query) >= frame_id(key)`,
where `frame_id` of a token is `id // (gh·gw)` — so attention is bidirectional inside a temporal slot
and causal across slots. The CLS prefix is a read-only sink: its row is open (it reads the whole clip)
and its column is closed to every patch query, which is what stops layer *l* information about the last
frame from reaching a first-frame token at layer *l+1*. At the released 16-frame 224² shape that leaves
53.1 % of the 3137 × 3137 grid open.

`jepa_block_causal_mask_f16` builds it host-side as an additive F16 `[N, N]` buffer of zeros and
`-INFINITY`, once per `jepa_encode` call — shared by all 24 layers and by every clip of that call, and
uploaded into each clip's graph — and passes it through `jepa_attn_opts::mask`. F16 serves both
consumers — `ggml_flash_attn_ext` requires it and `ggml_soft_max_ext` accepts it — so one buffer covers
the flash path, the `--no-flash` path and both backends. No padding is needed at this ggml commit: the
CUDA MMA kernel wraps mask rows with `fastmodulo(j0 + j, ne01)` and clamps the key axis of its final
tile, so an unpadded 3137-row buffer is read correctly there too (measured in
[parity](parity.md#results-encoders-on-cuda0)).

**The mask is the one term in this graph that grows with N².** Everything else — activations, RoPE
tables, the patch buffer — is linear in the token count, so the memory of a longer clip is otherwise
predictable. The mask is `N × N` F16 held twice, once on the host and once in the graph arena: 39 MiB
at the released 16-frame shape (3137 rows), **600 MiB at 64 frames** (12 545 rows), 2.4 GiB at 128.
`jepa_encode` therefore budgets a video graph the way it budgets a batched image graph — the
`$JEPA_MAX_GRAPH_MIB` ceiling of the Runtime switches below, counting the mask — and refuses a clip
that exceeds it instead of allocating; above 64 MiB of mask it also says so once. There is no batch to
shrink on the video path, so over-budget is a refusal rather than a smaller graph.

A CLS token in a RoPE model raises a second question, and the reference answers it: `RoPEAttention`
rotates `q/k[..., 1:, :]` only. The runtime therefore prepends an identity row (cos 1, sin 0) to the
host cos/sin tables for every prefix token, and `jepa_rope3d_apply` is untouched.

**A one-frame clip is a one-frame clip.** `jepa_encode` runs `in->n_frames` frames as given for every
video family; it never invents frames. LeVJEPA's model card feeds a still image as that frame repeated
to `jepa.enc.n_frames`, and `jepa-embed` does the repeat (and says so on stderr) — the library entry
point does not, so a caller handing it `n_frames = 1` gets a 197-row clip whose mask spans one temporal
slot. `jepa-embed` also notes any other clip length that differs from the trained one.

## Preprocessing

`src/preprocess.cpp` turns pixels into the normalised tensor the graph expects, driven by the
`jepa.pre.*` metadata: shortest-edge resize, centre crop, scale to `[0, 1]`, then subtract the
per-channel mean and divide by the per-channel std. `jepa_resize_antialias_u8` is a faithful port of
the integer antialiased path that `torchvision.transforms.v2.functional.resize(antialias=True)` runs
on x86 CPUs, so the tensor entering the network is **bit-exact** against the reference pipeline given
the same decoded pixels ([measured](parity.md#preprocessing-parity)).

The caveat is decoding, not resizing. jepa.cpp decodes JPEGs with `stb_image`, which differs from
PIL/libjpeg by ±1–2 levels on about 2 % of pixels. That decoder floor is `1 − cos` of 6.3e-6 to
1.7e-5 on the final features — larger than what f16 weights add.

**Video ingest.** The library is frames-in: `jepa_encode()` takes an NCTHW tensor and nothing in
`src/` opens a container. Turning a `.mp4` into frames is the *tools'* job, and `tools/video-decode.cpp`
does it by running `ffmpeg` as a subprocess —

```
ffmpeg -nostdin -v error -noautorotate -i CLIP -map 0:v:0 -an -sn -dn \
       -fps_mode passthrough -f rawvideo -pix_fmt rgb24 -
```

— and keeping `n` of the frames it writes, sampled uniformly over the whole clip with the formula the
PyTorch side uses, `idx = round(linspace(0, T_total − 1, n))` (numpy's ties-to-even rounding and its
pinned last sample included; a clip shorter than `n` repeats frames rather than failing). `T_total`
comes from `ffprobe` — the container's `nb_frames` where it has one, a `-count_frames` decode pass
otherwise — and is re-checked against the frames the decode actually produced, so a container that
lies about its length is resampled rather than mis-sampled.

Three details of that command line carry the parity. `-fps_mode passthrough` turns off ffmpeg's
constant-frame-rate conversion, which would otherwise duplicate frames of a variable-rate file (a
5-frame Something-Something-v2 `.webm` comes out as 59 frames without it) and sample entirely
different pixels; `-noautorotate` keeps ffmpeg from applying a display matrix PyAV's `to_ndarray()`
ignores; and **no** `-sws_flags` is passed, because libswscale's default yuv420p → rgb24 conversion is
already bit-identical to PyAV's reformatter (nothing is scaled on either side) while forcing
`full_chroma_int+accurate_rnd` changes 86 % of pixels by up to 44 levels. `ffmpeg` older than 5.1 gets
`-vsync 0` instead, decided once per process by asking the binary.

The result is that `jepa-embed --video clip.mp4` and `jepa-embed --frames-npy` on
`scripts/video_frames.py`'s output of the same clip build the **same uint8 tensor, byte for byte** —
so the video path has no decoder floor at all, the way it did when frames could only arrive as a
`.npy`. `tests/test-video.cpp` (`ctest -R video`) checks that against every video sample of the
reference dumps; it was also measured over the six fixture clips and 40 SSv2 clips, 20 of them shorter
than the requested 16 frames.

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
  flash, so it is a debugging and short-sequence option on the CPU.
  **On a GPU it is more than that: it is the only F32 attention path there**, since CUDA flash
  attention converts K/V to F16 unconditionally. Its cost scales with the sequence — 16.5 ms against
  15.5 for I-JEPA at 256 tokens, i.e. essentially free, but 109.4 ms against 46.5 for the ViT-L
  16-frame clip, a factor of 2.4 — so it is affordable at image scale and progressively less so
  beyond it.

Overrides: `JEPA_KV_F16` / `JEPA_KV_F32` (`--kv-f16` / `--kv-f32` in the tools).

On the CUDA backend the picture differs and jepa.cpp cannot change it:

| | CPU | CUDA |
|---|---|---|
| `mul_mat`, f32 weights | strict FP32 | **TF32** — ggml passes `CUBLAS_GEMM_DEFAULT_TENSOR_OP` to every `cublasGemmEx` and never calls `cublasSetMathMode`, and `ggml_mul_mat_set_prec` only selects the compute type |
| `mul_mat`, f16 weights | F32 accumulation | F16 accumulation unless `GGML_PREC_F32` is set, which jepa.cpp sets by default for four of its six families |
| `flash_attn_ext` | F32 K/V honoured | K/V always converted to F16, PV accumulator `half2`; `ggml_flash_attn_ext_set_prec` is a no-op |
| `ggml_norm` | centred two-pass variance | one-pass `E[x²] − mean²` |

`GGML_PREC_F32` on every `mul_mat` takes the f16 matmul error from 4.6e-03 to
2.6e-05, i.e. below the CPU's own 3.0e-04. It is not free, and what it costs depends on how much work
there is to hide it behind: on one matmul, **−21 % throughput at N = 2048 and +9 % at N = 8192**; end
to end, **1.78× at I-JEPA's 256 tokens against 0.96× at 18 432**. Quantized weights never reach cuBLAS,
so for them it costs nothing.

**Which of the two a GPU context picks is decided per family** (`jepa_gpu_prec_f32_default` in
`src/jepa.cpp`), by two measurements: whether the family still clears its GPU parity tier with f16
accumulation, and whether f16 accumulation is faster at its shapes.
`hfvit` and `levjepa` pass and gain, so they accumulate in f16; `ijepa` and `vjepa2` gain the most
and fail the tier, `lewm` and `vjepa2_1` pass and gain nothing at their shapes, so all four keep
`GGML_PREC_F32`. [performance.md](performance.md#accumulation-precision-on-a-gpu) is the table and
`tests/results/gpu-prec.json` the artifact. `$JEPA_GPU_PREC` overrides either way; on the CPU the
call is a no-op. The one-pass variance is a per-row *scale* error that cosine is structurally blind to, which is
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
50–70, which is why the masked predictor gains 2.7–2.9× on a GPU where the encoder of the same
model on the same clip gains 17.6×.

Its score matrix is `4·N²·H` bytes over `n_context + n_target` rows, and that sets a hard ceiling:
0.80 GB for the ViT-L 16-frame clip and 4.1 GB for the V-JEPA 2.1 16-frame clip both fit comfortably
on a 24 GB card, 12.9 GB for the ViT-L 64-frame clip is tight next to the weights, and **the V-JEPA
2.1 64-frame predictor needs ~65 GB and cannot run on this card at all** — that shape belongs on the
CPU. The engine checks the requirement against free device memory before building and refuses with
both sizes printed, rather than failing inside the allocator.

**There is no f32 tier on a GPU.** TF32 matmuls and F16-accumulating flash attention are backend
properties, so an f32 GGUF on CUDA behaves like its f16 file and is judged with the f16 bars. f16 and
quantized files hold their own bars there, and quantized weights are additionally the fastest GPU
path, because ggml's CUDA `mmq` INT8 tensor-core kernel covers every type jepa.cpp ships. Use the CPU
when f32 exactness is the requirement.

**One device per process.** `--gpu N` selects a single device; there is no tensor-parallel split
across cards, and none is planned — splitting a ViT forward, which is one dependency chain, would
mean a partitioned graph with peer copies per layer, the very thing the single-backend design exists
to avoid. Two cards are two independent streams: one model and one context per device, each
processing different items. That falls out of `--gpu N` with no engine work.

**The first two calls are slower.** ggml's CUDA backend captures a CUDA graph once it has seen the
same topology at the same tensor addresses twice in a row. jepa.cpp rebuilds its graph on every call
but with identical topology and a reused allocator, so from the third call the encoder is one graph
launch instead of roughly 700 kernel launches. Capture requires an unsplit graph, which is another
reason a single backend beats a scheduler; every measurement on this site warms up twice for it.

## Runtime switches

| variable | tool flag | effect |
|---|---|---|
| `JEPA_DEVICE=cuda:N` \| `cpu` \| `N` | `--gpu [N]` | select the compute device; default CPU |
| `JEPA_GPU_PREC=f16` \| `f32` | `--gpu-prec f16\|f32` (`jepa-bench`) | override the family's GPU accumulation precision for the process; both settings are parity-measured per family in `tests/results/gpu-prec.json` |
| `JEPA_VALIDATE_GRAPH=0` | — | disable the pre-compute graph validation. Debugging only — without it an unsupported node on a single CUDA backend computes a silently wrong answer |
| `JEPA_MAX_BATCH` | `--batch B` | image items per encoder graph; default 32 |
| `JEPA_MAX_GRAPH_MIB` | — | cap on the compute arena; larger inputs are split across graphs |
| `JEPA_KV_F16` / `JEPA_KV_F32` | `--kv-f16` / `--kv-f32` | override the automatic flash-attention K/V dtype (CPU; on CUDA F16 is forced and the request is logged once) |

`jepa_context_set_mul_mat_prec_f32(ctx, false)` is the API form of `JEPA_GPU_PREC=f16`, and
`jepa_context_mul_mat_prec_f32()` reports what a context ended up with.

## The cached planning context

A planner does not run the world model once. It encodes the current frame, then scores hundreds or
thousands of candidate action sequences against that one encode, over several CEM iterations, and
does the whole thing again after every observation. `jepa_ac_context` is the object that makes that
shape explicit: it owns the observed frames' latents **on the compute device**, and the AC graph
takes its context in two pieces —

```
shared : [enc_dim, n_observed*256, 1]   the frames every candidate has in common (the handle's
                                        tensor, referenced directly by the graph)
tail   : [enc_dim, n_predicted*256, K]  the frames the rollout produced, one set per candidate
```

— so the observed frames are neither replicated across the K candidates on the host nor re-uploaded
per step, and `pred.embed` runs on them once per graph instead of K times. `jepa_ac_context_update`
appends a newly observed frame (the receding-horizon step) and `jepa_ac_context_trim` slides the
window; capacity is `jepa.pred.n_frames`, allocated once.

**What that is worth, measured: nothing in time, and that is the honest answer.** Back-to-back in one
session on CUDA1 at f16, cached against explicit context is +0.11 % at K = 16 / H = 2, +0.07 % at
K = 64, +0.14 % at K = 256, −0.18 % at K = 16 / H = 4 and −0.20 % at K = 64 / H = 4: **±0.2 %,
straddling zero**. The shared-prefix broadcast against a fully replicated context is the same story — within 1 % in
its own controlled session (527.95 vs 529.43 ms on the GPU, 2181 vs 2200 ms on the CPU at 16
threads; not benchmark-artifact rows). The reason is arithmetic:
`pred.embed` is one [1408 → 1024] matmul over 256 rows, against 24 blocks of 1024-d attention and FFN
over K × 258 rows, and the upload is 1.44 MB against a half-second graph. The handle earns its place
as an **API**, not as a speed-up — one object held across CEM iterations and receding-horizon steps,
one device allocation for the observed frames — and the parity suite gates it as bit-identical to the
explicit path, including a six-update receding-horizon loop
([performance](performance.md#planning-what-a-cem-decision-costs) has the table).

The planner on top of it, `jepa_ac_plan`, is a port of Meta's `mpc_utils.py::cem`. Two details of
that loop decide whether an implementation is right: only four of the seven action dimensions are
sampled (translation and the gripper; rotation is hard zeros), and `selected.std(0)` is torch's
**unbiased** estimator, a factor 1.155 at topk = 4 applied every iteration. Replaying the random
draws their loop made, jepa.cpp returns the same plan to max |Δ| 2.98e-08
([parity](parity.md#v-jepa-2-ac-jepapredkind-ac-the-action-conditioned-world-model)).

## Robustness

A GGUF is a file someone downloaded, and an image is bytes off a network. Both are parsed before
anything about them is known, so the loader and the preprocessor treat every number in them as
hostile until it is in range. Nothing is clamped or defaulted silently: an input that does not
describe a model this engine can run is refused, by name, on `stderr` and in
[`jepa_error_text()`](api.md).

**What the loader validates.** The metadata and the tensor extents come first, before a byte of
weight is allocated — that ordering is the point of the extent check, since the weight buffer is
sized from the header's claims and not from the file's length. The dtype, shape and
required-tensor checks run once the tensors are resident and before any graph is built:

| check | refused when |
|---|---|
| `general.architecture`, `jepa.family` | not `jepa`; not one of the seven families |
| every `jepa.*` integer | outside its range — `embed_dim` 1–65536, `n_layer` 1–1024, `n_head` 1–1024, `ffn_dim` 1–2²⁰, `patch_size`/`tubelet_size` 1–1024, `img_size`/`n_frames` 1–65536, `in_chans` 1–1024, token counts ≤ 2²² (`JEPA_LIMIT_*` in `src/jepa-internal.h`) |
| every `jepa.*` float | not finite, or outside its range — `ln_eps` ∈ [0, 1], `rope_theta` > 0, `jepa.pre.std` ≠ 0 |
| enum-valued strings | unknown `jepa.enc.act`, `jepa.enc.attn_mode`, `jepa.pred.act`, `jepa.pred.action_act`, `jepa.pred.proj_act` |
| `attn_mode = block_causal` | on a family whose graph has no mask, where it would run unmasked |
| derived invariants | `embed_dim % n_head`; an odd head width on a 3-D RoPE family (it rotates pairs); `rope_interpolate` with no `rope_ref_grid` to rescale from; a masked predictor with no position grid; a LeWM predictor with no `action_dim` or `n_frames`; an **AC** predictor with no `action_dim`, `state_dim`, `n_cond_tokens`, `grid_size` or `n_frames`, or an odd head width (it rotates pairs too) |
| tensor extent | any tensor whose bytes are not inside the file (a truncated download; the header alone can promise terabytes) |
| tensor dtype | anything that is not f32, f16, bf16 or a quantized type; and, for the operands the graph adds, multiplies or concatenates — norms, biases, layer scales, CLS and register tokens, position tables, mask tokens, modality vectors — anything that is not f32 |
| tensor shape | every weight, bias and vector of every encoder, predictor and head block against the hparams; the AC conditioning projections (`pred.action_embed` / `pred.state_embed` against `action_dim` / `state_dim` and the predictor width, since both feed a `ggml_concat`); and every table a metadata count indexes into (`pred.mask_tokens` against `n_mask_tokens`, `pred.pos_embed` against `n_frames`) |
| required tensors | a predictor or head the metadata promises and the tensors do not deliver |

**What is refused at call time.** Non-positive or overflowing input shapes; an encoder output header
that describes nothing; token ids off the predictor's grid; and any graph whose estimated activation
bytes exceed `$JEPA_MAX_GRAPH_MIB` (8 GiB by default) — the encoder shrinks its batch first and only
errors when a single item still will not fit, the video encoder, the masked predictor, the LeWM
rollout and both AC entry points refuse outright (`jepa_ac_predict` also refuses a context longer
than `jepa.pred.n_frames`, the frame slots the block-causal mask was built for, and
`jepa_ac_rollout` refuses a horizon that would reach past them). The image pipeline additionally caps the intermediate of the shortest-edge
resize at 64 megapixels, which is what stops a 16384×1 image from asking for gigabytes. The cap is on the
resized intermediate, so in terms of the input it is an aspect-ratio limit of 64 Mpx / `resize_short`²
— about 1337:1 for a 224-pixel model and 350:1 for V-JEPA 2.1 at 438 — and a refusal names the input
geometry and the bound.

**Thread contract.** Stated in full in [`include/jepa.h`](api.md) and checked by
`tests/test-threads.cpp`: a `jepa_model` is immutable after load and may be shared by any number of
threads; a `jepa_context` belongs to one thread; `jepa_error_reset` / `jepa_error_text` are
thread-local; preprocessing is re-entrant; the "said once" warnings are published atomically.
Concurrent encodes through per-thread contexts are bit-identical to the same work run serially.
`n_threads` is the width of the pool *inside* one graph — a context per thread parallelises calls,
and doing both oversubscribes the machine. The suite forges its own model when given none, so it is
a real check on a runner with no weights; run it under TSan with
`-fsanitize=thread -DGGML_OPENMP=OFF` (TSan cannot see through libgomp's own synchronisation and
reports its pool as a race in any program that uses it). On a kernel with 32-bit ASLR entropy
TSan aborts at start-up with `unexpected memory mapping`; run the suite under `setarch -R` or set
`vm.mmap_rnd_bits=28`.

**Running the fuzzer.** `tests/fuzz/fuzz-gguf-load.cpp` feeds arbitrary bytes to `jepa_model_load`
and, when they load, to a shallow encode/pool/predict pass. It is off by default; CI builds it and
never runs it.

```bash
scripts/make_fuzz_corpus.py                       # needs gguf-py; writes tests/fuzz/corpus/
cmake -S . -B build-fuzz -DJEPA_FUZZ=ON -DGGML_OPENMP=OFF \
      -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fsanitize-recover=undefined -g" \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fsanitize-recover=undefined -g" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-fuzz -j --target fuzz-gguf-load
build-fuzz/fuzz-gguf-load --corpus tests/fuzz/corpus --out findings --fork --seconds 3600
build-fuzz/fuzz-gguf-load --file findings/crash-0000-sig11-exit-1.gguf     # replay one input
```

The seed corpus is generated from the two small GGUFs: a shrunk-but-loadable copy of each (D=8,
2 layers, 8×8 images), truncations of both those and the real files, and one-key mutations. With
clang the same source is a libFuzzer target (`-fsanitize=fuzzer`, `LLVMFuzzerTestOneInput`); with
any other compiler it is the deterministic mutation loop above, reproducible from `--seed`.
`--fork` runs each input in a child, so a crash or a hang (`--timeout`, default 10 s) is written to
`--out` and the run continues.

UBSan is *recoverable* in that build on purpose. The GGUF reader in the ggml submodule loads a type
tag into an uninitialised `gguf_type` on its own failure path and computes tensor sizes before
range-checking them (`ggml/src/gguf.cpp:575`, `:584`, `:714`, `ggml/src/ggml.c:1288`); those are
upstream's and they fire on most malformed inputs. Making UBSan non-fatal there keeps every check
enabled and turns triage into "which source file does the report name" — a report outside `ggml/` is
ours. The sanitizer CI job, which feeds only valid files, runs with `halt_on_error=1` and no
exclusions.

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
tools/video-decode.*  --video: an ffmpeg subprocess -> sampled THWC uint8 frames (tools only)
tools/jepa-worldmodel LeWM: image -> state -> K-step action rollout; --ref-check against fixtures
tools/jepa-quantize   f32/f16 GGUF -> q8_0 / q4_k / ...
tools/jepa-bench      timing: encoder / head / predictor / lewm-step / lewm-rollout, --md / --json

tests/test-parity     replay the golden dumps: cosine / max-abs / top-k, non-zero exit on regression
                      (--gpu [N] judges the same run with the GPU threshold column; SKIPs without a GPU)
tests/test-predictor  the same for the three predictors, against the reference encoder tokens
tests/test-batch      batched vs per-item bit-exactness
tests/test-attn       flash vs naive attention against a double-precision reference; K/V policy; timing
tests/test-ops        rope3d against tests/vectors/, and the block-causal mask on both attention paths
tests/test-backend    GPU graph validation and CPU/GPU agreement; skips cleanly without a GPU
tests/test-video      --video decode+sampling vs the reference dumps' PyAV frames; skips without ffmpeg
tests/test-errors     the failure paths: forged and truncated GGUFs, bad shapes, budget refusals
tests/test-threads    one model, N threads, a context each: outputs bit-identical to single-threaded
tests/forge-gguf.h    builds a tiny loadable GGUF, one knob per thing that can be wrong with it
tests/fuzz/           the GGUF loader fuzz target (-DJEPA_FUZZ=ON) and its corpus generator

scripts/convert.py          HF safetensors / torch.hub .pt -> GGUF
scripts/dump_reference.py   PyTorch golden outputs -> tests/fixtures/ref/<model>/
scripts/compare.py          .npy / ref-dir comparison (cosine, max-abs, rel, top-k)
scripts/knn_eval.py         the frozen-feature k-NN protocol shared by both accuracy benchmarks
scripts/make_fuzz_corpus.py the fuzz seed corpus: shrunk minis, truncations, one-key mutations
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
the RoPE tables against generated vectors and the block-causal mask (entry by entry, then through
`jepa_build_attention` on both attention paths against a double-precision masked softmax, with an
unmasked control that must disagree), `test-attn` flash attention against a double-precision
reference including the K/V dtype policy, and `test-backend` the GPU graph validation plus CPU/GPU
agreement. The two `parity-*-gpu` entries add GPU parity for the two families whose `mul_mat`
accumulation default is f16 rather than `GGML_PREC_F32` — `hfvit` (LeJEPA) and `levjepa` — running
them at that default, so the shipped setting is gated rather than only measured
([parity](parity.md#accumulation-precision-per-family)); like `backend` they exit 0 with a SKIP line
on a build with no GPU. `test-predictor` adds structural checks the reference cannot provide:
causal-prefix equality and rollout-versus-predict identity on LeWorldModel, which are bit-exact on
both backends.

Results: [Accuracy](accuracy.md) for the curated view, [parity](parity.md) for every row.
