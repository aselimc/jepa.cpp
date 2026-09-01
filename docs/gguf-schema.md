# jepa.cpp GGUF schema (v1)

One GGUF file is one *model bundle*: an encoder, and optionally a predictor and/or a head. The
converter, the loader and the graph builder all implement exactly this document — change it here
first. The graph these keys describe is in [Architecture](architecture.md); the Python side that
writes them is in [Converter](converter.md).

## General metadata

| key | type | notes |
|---|---|---|
| `general.architecture` | str | always `"jepa"` |
| `general.name` | str | e.g. `"vjepa2-vitl-fpc64-256"` |
| `general.license` | str | `"mit"`, `"cc-by-nc-4.0"`, `"apache-2.0"` — copied from the source |
| `general.source_url` | str | HF repo or download URL |
| `general.file_type` | u32 | ggml ftype (0 = f32, 1 = f16, 7 = q8_0, ...) |
| `jepa.schema_version` | u32 | `1` |
| `jepa.family` | str | `ijepa` · `vjepa` · `vjepa2` · `vjepa2_1` · `levjepa` · `hfvit` (LeJEPA-style / LeWM encoder) · `lewm` |
| `jepa.modality` | str | `image` · `video` · `image+video` |

## Encoder (`jepa.enc.*`)

| key | type | notes |
|---|---|---|
| `jepa.enc.embed_dim` | u32 | 1024 for ViT-L |
| `jepa.enc.n_layer` | u32 | |
| `jepa.enc.n_head` | u32 | V-JEPA 2 ViT-g = **22**, I-JEPA ViT-g = 16 |
| `jepa.enc.ffn_dim` | u32 | explicit, not a ratio (6144 for g) |
| `jepa.enc.patch_size` | u32 | 14 or 16 |
| `jepa.enc.tubelet_size` | u32 | 1 for image models, 2 for video |
| `jepa.enc.img_size` | u32 | training crop (224 / 256 / 384); inference may differ if pos scheme allows |
| `jepa.enc.n_frames` | u32 | training frames (1 for image, 16 or 64 for video) |
| `jepa.enc.in_chans` | u32 | 3 |
| `jepa.enc.ln_eps` | f32 | 1e-6 (HF I-JEPA config says 1e-6 too) |
| `jepa.enc.act` | str | `gelu_erf` (all Meta JEPAs) · `gelu_tanh` · `silu` |
| `jepa.enc.pos_type` | str | `sincos2d` · `sincos3d` · `learned` · `rope3d` |
| `jepa.enc.rope_theta` | f32 | 10000 |
| `jepa.enc.rope_interpolate` | bool | V-JEPA 2.1 `interpolate_rope`. The "pretrained grid" it rescales h/w to is **not** `img_size / patch_size`: Meta hard-codes `256 / patch_size` (16 for patch 16, 18 for patch 14) — the loader derives `train_grid_h = train_grid_w = 256 / patch_size` = `jepa.enc.rope_ref_grid` (see `src/rope3d.h`). `family == vjepa2_1` also selects the interleaved cos/sin layout (`variant = 1`); `vjepa2` uses the tiled HF layout (`variant = 0`). |
| `jepa.enc.rope_freq_layout` | str | `tiled` (V-JEPA 2: per-axis cos/sin table is `[f_0..f_{d/2-1}, f_0..f_{d/2-1}]`, the Meta/HF quirk) · `interleaved` (V-JEPA 2.1: `[f_0,f_0,f_1,f_1,...]`, textbook RoPE). See `scripts/jepa_convert/VJEPA_NOTES.md` |
| `jepa.enc.rope_ref_grid` | u32 | V-JEPA 2.1 only: spatial RoPE positions are rescaled `h *= (ref_grid-1)/(gh-1)` (16 for patch 16) |
| `jepa.enc.cls_token` | bool | true for `hfvit` (DINOv2-style) and `levjepa` (the only video family with one), false for Meta JEPAs |
| `jepa.enc.n_registers` | u32 | 0 |
| `jepa.enc.qkv_fused` | bool | true → `attn_qkv.*` tensor with rows [q;k;v]; false → separate q/k/v |
| `jepa.enc.modality_embed` | bool | V-JEPA 2.1 |
| `jepa.enc.image_patch_embed` | bool | V-JEPA 2.1 separate 1×16×16 image tokenizer |
| `jepa.enc.hier_layers` | [u32] | V-JEPA 2.1 hierarchical layer ids (per-layer norms exist for these) |
| `jepa.enc.layer_scale` | bool | DINOv2-style `ls1/ls2` present |
| `jepa.enc.attn_mode` | str | `full` (every existing file: the key is absent and the loader defaults to it) · `block_causal` (`levjepa`: key *j* is visible to query *i* iff `frame_id(i) >= frame_id(j)`, with the CLS row open and the CLS column closed — see the levjepa family note) |
| `jepa.enc.proj_act` | str | `lewm` only: activation of the `enc.proj.*` MLP (`gelu_erf`) |

Encoder tensors (ggml dim order is reversed from PyTorch; PyTorch row-major is stored as-is, i.e. `ne[0]` = last PyTorch dim):

```
enc.patch_embed.weight        [embed_dim, in_chans * tubelet * patch * patch]   (conv flattened to a matmul, C-T-H-W order)
enc.patch_embed.bias          [embed_dim]
enc.patch_embed_img.weight    [embed_dim, in_chans * 1 * patch * patch]        (2.1 only)
enc.patch_embed_img.bias
enc.pos_embed                 [n_tokens, embed_dim]                            (sincos*/learned; rows = [CLS?; patches] — registers are appended AFTER the pos add and have no pos rows)
enc.cls_token                 [embed_dim]
enc.reg_tokens                [n_registers, embed_dim]
enc.mod_embed_img             [embed_dim]                                      (2.1)
enc.mod_embed_video           [embed_dim]                                      (2.1)
enc.blk.{i}.ln1.weight / .bias
enc.blk.{i}.attn_qkv.weight   [3*embed_dim, embed_dim]  (+ .bias)   -- or attn_q / attn_k / attn_v
enc.blk.{i}.attn_out.weight   [embed_dim, embed_dim]    (+ .bias)
enc.blk.{i}.ln2.weight / .bias
enc.blk.{i}.ffn_up.weight     [ffn_dim, embed_dim]      (+ .bias)
enc.blk.{i}.ffn_down.weight   [embed_dim, ffn_dim]      (+ .bias)
enc.blk.{i}.ls1 / .ls2        [embed_dim]               (layer_scale only)
enc.norm.weight / .bias                                  (final LN; for 2.1 this is norms_block[-1])
enc.hier_norm.{k}.weight/.bias                           (2.1: one per hier layer, k = index into hier_layers)
enc.proj.0.weight/.bias       [proj_hidden, embed_dim]  (lewm: projector MLP on the CLS token, BatchNorm folded)
enc.proj.2.weight/.bias       [embed_dim, proj_hidden]  (lewm; act between = jepa.enc.proj_act)
```

Sincos tables are **precomputed at conversion** for the training grid (`enc.pos_embed`); the runtime bicubic/trilinear-interpolates for other grids.

## Predictor (`jepa.pred.*`) — optional

| key | notes |
|---|---|
| `jepa.pred.kind` | `masked` (I-JEPA / V-JEPA / V-JEPA 2 / 2.1) · `ac` (V-JEPA 2-AC, see the family note) · `lewm` |
| `jepa.pred.embed_dim`, `n_layer`, `n_head`, `ffn_dim` | |
| `jepa.pred.n_mask_tokens` | 10 (V-JEPA 2), 8 (2.1) |
| `jepa.pred.out_dim` | width of `pred.proj` output (enc_dim for V-JEPA 2; teacher_dim / n_hier_in = 1664 for 2.1 ViT-B) |
| `jepa.pred.rope_freq_layout`, `jepa.pred.rope_interpolate`, `jepa.pred.rope_ref_grid`, `jepa.pred.grid_size` | predictor RoPE convention (2.1: `interleaved`, **not** interpolated, mask ids decoded on the fixed `img_size/patch` grid = `grid_size`) |
| `jepa.pred.n_hier_in` | 2.1: number of hierarchical encoder outputs concatenated into `pred.embed` (1 for ViT-B/L, 4 for g/G) |
| `jepa.pred.modality_embed`, `jepa.pred.context_proj` | 2.1: `pred.mod_embed_*` / `pred.proj_context.*` present |
| `jepa.pred.action_dim`, `jepa.pred.state_dim` | 7 / 7 for V-JEPA 2-AC; 10 / 0 for LeWM |
| `jepa.pred.n_cond_tokens` | u32, AC only: conditioning rows prepended to **each frame's** patch tokens (2 = action, state; Meta's `use_extrinsics` would make it 3). It multiplies the sequence length, so the loader bounds it |
| `jepa.pred.cond_order` | str, AC only: the order of those rows, `"action,state"` |
| `jepa.pred.normalize_reps` | bool, AC only: the released world-model loop passes every latent through a **non-affine** LayerNorm before and after the predictor (`F.layer_norm(h, (D,))`). `jepa_ac_rollout` applies it between steps; `jepa_ac_normalize` is the same thing for a caller feeding latents in by hand |
| `jepa.pred.norm_reps_eps` | f32, AC only: eps of that LayerNorm (1e-5, torch's default) |
| `jepa.pred.frame_causal` | bool (AC, LeWM) |
| `jepa.pred.n_frames` | LeWM: 3. AC: the frame slots the block-causal mask was built for (32 = `num_frames / tubelet_size`); it caps a context and a rollout |
| `jepa.pred.head_dim` | u32, only when `n_head * head_dim != embed_dim` (LeWM: 16 × 64 in a 192-d model) |
| `jepa.pred.ln_eps` | f32, eps of the affine LayerNorms in the predictor (LeWM: 1e-5) |
| `jepa.pred.adaln_eps` | f32, eps of the non-affine adaLN norms (LeWM: 1e-6) |
| `jepa.pred.act` | str, FFN activation (`gelu_erf`) |
| `jepa.pred.qkv_bias` | bool (LeWM: false → no `attn_qkv.bias`) |
| `jepa.pred.action_act` | str, activation inside the 2-layer `pred.action_embed` MLP (LeWM: `silu`) |
| `jepa.pred.proj_act` | str, activation inside the 2-layer `pred.proj` MLP (LeWM: `gelu_erf`) |

```
pred.embed.weight/bias        [pred_dim, enc_dim]        (context projection; single Linear when jepa.pred.n_hier_in == 1 (V-JEPA 2, 2.1 ViT-B/L); 2.1 with n_hier_in > 1 (ViT-g/G): 2-layer MLP pred.embed.0 / pred.embed.2 over the concatenated hier features)
pred.mask_tokens              [n_mask_tokens, pred_dim]
pred.pos_embed                [n_tokens, pred_dim]        (sincos models only)
pred.blk.{i}.*                same layout as enc.blk
pred.norm.weight/bias
pred.proj.weight/bias         [enc_dim (or teacher_dim), pred_dim]
pred.action_embed.weight/bias [pred_dim, action_dim]        (AC: single linear)
pred.action_embed.0 / .2      [4*pred_dim, action_dim] / [pred_dim, 4*pred_dim]   (lewm: 2-layer MLP, act = jepa.pred.action_act)
pred.proj_context.weight/bias [teacher_dim, pred_dim]     (2.1 return_all_tokens: projection of the *context* tokens)
pred.mod_embed_img / pred.mod_embed_video  [pred_dim]      (2.1: added to every predictor token)
pred.action_embed.weight/bias [pred_dim, action_dim]                (ac: single Linear, 7 -> 1024)
pred.state_embed.weight/bias  [pred_dim, state_dim]                (ac: single Linear, 7 -> 1024)
pred.blk.{i}.adaln.weight/bias [6*pred_dim, pred_dim]      (lewm: adaLN-zero modulation, see the lewm section)
pred.proj.0 / .2              [proj_hidden, pred_dim] / [out_dim, proj_hidden]    (lewm: 2-layer MLP, BatchNorm folded, act = jepa.pred.proj_act)
```

## Head (`jepa.head.*`) — optional

| key | notes |
|---|---|
| `jepa.head.kind` | `attentive_pool` (V-JEPA 2 classifiers) · `linear_cls` · `none` |
| `jepa.head.n_classes` | |
| `jepa.head.n_pool_layers` | 3 |
| `jepa.head.labels` | [str] id2label, in order |

```
head.blk.{i}.*                self-attention blocks over all tokens, run FIRST (same layout as enc.blk)
head.query                    [1, embed_dim]            (raw query token = residual of the cross-attn)
head.xattn.ln_kv.weight/bias                            (HF layer_norm1: applied to keys/values only, NOT to the query)
head.xattn.q / .k / .v        .weight [embed_dim, embed_dim] + .bias   (no output projection in HF)
head.xattn.ln2.weight/bias                              (HF layer_norm2, on query + attn)
head.xattn.ffn_up / ffn_down  MLP after the cross-attention (fc1/fc2)
head.cls.weight/bias          [n_classes, embed_dim]    (no final norm: classifier acts directly on the pooled token)
```

## Preprocessing metadata (`jepa.pre.*`)

| key | notes |
|---|---|
| `jepa.pre.mean`, `jepa.pre.std` | [f32 ×3], ImageNet (0.485,0.456,0.406)/(0.229,0.224,0.225) for V-JEPA, LeJEPA, LeWM. **The HF I-JEPA checkpoint ships 0.5/0.5/0.5** (`ViTImageProcessor`) and that is what the file carries — parity is against the HF processor |
| `jepa.pre.resize_short` | short-side resize before crop (V-JEPA: crop×256/224; LeJEPA: 256; LeWM: 224). For `resize_mode=squash` it is the target square size (I-JEPA HF: 224) |
| `jepa.pre.crop` | center crop size |
| `jepa.pre.resample` | `bilinear` · `bicubic` (must match the reference processor) |
| `jepa.pre.resize_mode` | optional — loader defaults to `shortest_edge` when absent (video-family files omit it). `shortest_edge` (resize short side to `resize_short`, keep aspect, centre-crop `crop`) · `squash` (resize directly to `crop`×`crop`, aspect not preserved — HF I-JEPA `ViTImageProcessor` with `size={height,width}`) |
| `jepa.pre.rescale` | f32, pixel scale before normalisation (1/255) |

**Reference resampler.** The HF 5.x processors (`ViTImageProcessor`, `BitImageProcessor`, `VJEPA2VideoProcessor`) are torchvision-backed: `torchvision.transforms.v2.functional.resize` on the **uint8** CHW tensor with `antialias=True`, rounded back to uint8, then center-crop, /255, normalize. PIL resampling differs by up to 1 uint8 LSB on ~0.1–0.3 % of pixels; the C++ preprocessor targets the torchvision result (see `tests/fixtures/README.md`).

## Token order

Video tokens are **T-major, then H, then W** (`i = t*gh*gw + h*gw + w`). Image tokens are H-major then W. `hfvit` and `levjepa` prepend CLS (and, for `hfvit`, registers); a prepended CLS carries no position information at all — no pos-embed row and no RoPE rotation — so the grid ids above are those of the patch tokens only, shifted by the prefix.

## Quantization rules

Quantize only 2-D weight matrices of `*.attn_qkv` / `attn_q` / `attn_k` / `attn_v` / `attn_out`, `*.ffn_up` / `ffn_down`, `pred.proj*` (incl. `pred.proj_context`), `pred.embed*`, `enc.proj.*`, `head.blk.*`, `head.xattn.{q,k,v}`, `head.cls`. `pred.action_embed` / `pred.state_embed` are matmul weights too, but their 7-wide rows are not a multiple of 32, so every quantizer falls back to F16 for them. K-quants need `ne[0] % 256 == 0`; fall back per tensor to q8_0 (or the q4_0/q5_0 sibling) otherwise.

Everything else stays **F32**, and this is a requirement rather than a convention: norms, biases, layer scales, position tables, CLS / register / mask / modality tokens and `pred.blk.*.adaln` biases reach ggml as the right-hand side of an `add`, a `mul` or a `concat`, and those ops take an f32 operand and assert on anything else. The loader checks it and refuses a file that gets it wrong (docs/architecture.md ["Robustness"](architecture.md#robustness)). Patch embeddings and `pred.action_embed.*` are matmul weights and so *may* be F16, but the converter writes them F32 today.

Converter dtype rule for `--ftype f16`: the quantizable set above is written as F16, everything else as F32.

## Family notes

### ijepa (facebook/ijepa_vith14_1k)

* The HF checkpoint stores the position table as a parameter (`embeddings.position_embeddings`, [1,256,1280]).
  It is a 2-D sincos table, but with the halves ordered `[emb_w | emb_h]` — bit-identical to
  `sincos_2d(16, 16, 1280, w_first=True)` in `scripts/jepa_convert/common.py` (max abs diff 0.0), while the
  `[emb_h | emb_w]` ordering of `vjepa2/src/models/utils/pos_embs.py` differs by up to 2.0. The converter stores the
  checkpoint table verbatim in `enc.pos_embed`; the runtime interpolates it for other grids (as HF does).
* No CLS token, no predictor in the HF release; pooled feature = mean over the 256 patch tokens after `enc.norm`.
* `general.license = cc-by-nc-4.0`.

### levjepa (galilai-group/LeVJEPA-VideoMix-Large)

ViT-L/16 video encoder (`modeling_levjepa.py`, shipped with the weights; the file is the reference
implementation and `trust_remote_code=True` loads it). Three things separate it from `vjepa2`, and the
schema says all three:

* **`tubelet_size = 1`** with `n_frames = 16`, so a 16×224×224 clip is `16·14·14 = 3136` patch tokens
  in the usual T-major/H/W order.
* **`cls_token = true`** — the first video family with one. `enc.cls_token` is prepended *after* the
  patch embedding and receives **no position information**: there is no `enc.pos_embed` (RoPE model)
  and RoPE skips it (`RoPEAttention.forward` rotates `q/k[..., 1:, :]`), so the runtime's cos/sin
  tables get an identity row (cos 1, sin 0) at index 0. Total 3137 tokens. Pooled feature = CLS after
  `enc.norm`, i.e. `last_hidden_state[:, 0]`.
* **`attn_mode = "block_causal"`** — an additive attention mask, not an option: with full attention the
  same weights give CLS cosine 0.945 and a worst patch token of 0.834 against the reference
  (`archery_f16`). Key *j* is visible to query *i* iff
  `frame_id(i) >= frame_id(j)`, where `frame_id` of a patch token is `id // (gh*gw)`; the CLS **row**
  is fully open (it reads the whole clip) and the CLS **column** is closed to every patch query (it is
  a read-only sink, so nothing routes last-frame information into a first-frame token). Density 53.1 %
  at 3137 tokens. Everything else follows `vjepa2`: 3-D RoPE θ 10000 in the **tiled** layout
  (`rope_freq_layout = tiled`; `rotate_queries_or_keys` uses Meta's `repeat(1,1,1,2)` expansion — the
  interleaved 2.1 layout scores CLS cosine 0.841 on these weights), no `interpolate_rope`, LN eps 1e-6,
  GELU(erf), fused qkv with bias, pre-LN, no layer-scale.

The checkpoint is the EMA encoder only: no predictor, no head, no projector. `jepa.pre.*` comes from the
model card (there is no `preprocessor_config.json`): ImageNet mean/std, short side → 224, centre crop 224,
**bicubic**, rescale 1/255. A still image is fed as a 16-frame clip by repeating the frame, which is what
the model card's ImageNet probe does. `general.license = cc-by-nc-4.0`.

### vjepa2_ac (V-JEPA 2-AC, `jepa.pred.kind = "ac"`)

`jepa.family` stays `vjepa2` — the encoder is an ordinary V-JEPA 2 ViT-g/16 and the whole delta is in
the predictor. One bundle carries both. `scripts/jepa_convert/vjepa2_ac.py` writes it from Meta's
single `vjepa2-ac-vitg.pt`; `src/vjepa2_ac.cpp` builds the graph and
`scripts/jepa_convert/selftest.py::ac_predictor_forward` is the executable spec.

**The encoder in that checkpoint is not the HF `facebook/vjepa2-vitg-fpc64-256` release.** Measured
per tensor over all **484** encoder tensors, the cosine is **median 0.99995**, p05 0.99896, **min
0.99647** (`blocks.0.norm1.weight`), with 38 of 484 below 0.999 — the low tail is the early blocks'
attention projections (`blocks.1.attn.proj.weight` 0.99767, `blocks.0.attn.proj.weight` 0.99827) and
max |Δ| reaches 2.1e-2 on values up to 1.35. Close, in other words, but not the same weights.
Meanwhile `encoder` and `target_encoder` inside the AC checkpoint are **bit-identical** — the encoder
was frozen during AC training. `src/hub/backbones.py::_make_vjepa2_ac_model` loads `state_dict["encoder"]`, so that is
what the bundle ships, and it gets its own parity fixtures.

Predictor (`src/models/ac_predictor.py::VisionTransformerPredictorAC`, 24 blocks × 1024 dims,
16 heads × 64, ffn 4096, GELU(erf), LN eps 1e-6, fused qkv with bias, no mask tokens and no
position table):

```
x   = pred.embed(context_latents)                 # [T*HW, 1024]   HW = grid_size^2 = 256
a_t = pred.action_embed(action_t)                 # [T, 1024]      action_dim 7
s_t = pred.state_embed(state_t)                   # [T, 1024]      state_dim 7
seq = per frame t: [a_t, s_t, x_{t,0} .. x_{t,HW-1}]                # T * (2 + HW) rows
24 pre-LN blocks, full-width attention under the block-causal mask, 3-D RoPE (tiled) on q and k
seq = seq minus the 2 conditioning rows of every frame                # [T*HW, 1024]
out = pred.proj(pred.norm(seq))                                       # [T*HW, 1408]
```

Two details decide whether an implementation is right:

* **The mask is block-causal over whole frames**, not over tokens: every row of frame *t* attends
  every row of frames 0..*t*, conditioning rows included
  (`build_action_block_causal_attention_mask` fills whole `N_T × N_T` blocks, `N_T = 2 + HW`). It is
  built on the host and passed through `jepa_attn_opts::mask`, as `src/lewm.cpp` does. Row block *t*
  of the output is therefore the prediction of frame *t+1* given frames 0..*t*: one call with *T*
  frames also answers every shorter prefix.
* **The action and state tokens are rotated on the depth axis only**, with `pos = t`, and their
  height/width lanes are left alone (`ACRoPEAttention` rotates `q[..., :d_dim]` with
  `arange(T)` and passes `q[..., d_dim:]` through). A grid id of `t·grid²` has `h = w = 0`, whose
  cos/sin rows are exactly 1 and 0 — the identity — so **one id list** feeds the ordinary
  `jepa_rope3d_tables_ids` for every row:

  | row | grid id |
  |---|---|
  | action / state of frame *t* | `t·grid²` |
  | patch (*t*, *h*, *w*) | `t·grid² + h·grid + w` |

  head_dim is 64 here, so `d = 2·⌊⌊64/3⌋/2⌋ = 20` per axis: lanes 0–59 rotate, 60–63 are untouched.
  `ACRoPEAttention` also rescales `h`/`w` by `grid_size / H`; the released checkpoint runs at its own
  16×16 grid, where that factor is 1, and `jepa_ac_predict` decodes the ids on `jepa.pred.grid_size`.

`extrinsics_encoder.*` is instantiated unconditionally by the reference module but only read when
`use_extrinsics=True`, which the released hub entry never sets; the converter skips it and says so.

Preprocessing is **not** the V-JEPA 2 video-processor pipeline. The demo builds
`app/vjepa_droid/transforms.py::make_transforms` with `random_resize_scale=(1,1)` and
`random_resize_aspect_ratio=(1,1)`, which degenerates `random_resized_crop` to a centre crop of the
largest square followed by `torch.nn.functional.interpolate(bilinear, align_corners=False, no
antialias)` to 256×256, then `(x − 255·mean) / (255·std)`. On the square DROID/Franka renders the
model consumes, both the crop and the resize are the identity, which is what `resize_short = crop =
256` reproduces; the file records that, and `docs/parity.md` measures the residual (2.4e-07, one
float32 ulp). `general.license = mit` (the facebookresearch/vjepa2 LICENSE).

### hfvit (OK-AI/lejepa-vits16-pretrain-in1k)

DINOv2-style ViT-S/16 from `Open-Knowledge-AI/lite_ssl` ("ViTv2"): LN eps 1e-6, GELU(erf), qkv/proj/ffn biases,
learned `pos_embed` with the CLS slot at row 0, no registers, no LayerScale (`init_values=null`). The checkpoint's
`backbone.cva_module_proj.*` (loss-side DINOHead) is not converted. Pooled feature = CLS after `enc.norm`.

### lewm (quentinll/lewm-pusht)

Encoder = HF `ViTModel` tiny/14 written in the `hfvit` layout (**LN eps 1e-12**, ViTConfig default). The world-model
state is `emb = enc.proj(CLS)` (2-layer MLP, BatchNorm folded, GELU). Predictor (`jepa.pred.kind = lewm`,
192-d, 6 layers, 16 heads × 64, ffn 2048, `n_frames` 3, `action_dim` 10, causal over frames):

```
a_t = action_embed.2( silu( action_embed.0( action_t ) ) )              # (T, 192)   [Conv1d(k=1) folded into .0]
x   = emb + pred.pos_embed[:T]
for each block i:
    sh_a, sc_a, g_a, sh_m, sc_m, g_m = split6( adaln( silu(a) ) )       # each (T, 192)
    h = LN(x, no affine, eps=adaln_eps) * (1 + sc_a) + sh_a
    h = ln1(h)                                                           # affine, eps=ln_eps
    q,k,v = attn_qkv(h)  (no bias) ; causal softmax(q kᵀ / 8) v over the T frames, 16 heads × 64
    x = x + g_a * attn_out(...)
    h = LN(x, no affine, eps=adaln_eps) * (1 + sc_m) + sh_m
    x = x + g_m * ffn_down( gelu( ffn_up( ln2(h) ) ) )
x = pred.norm(x)
pred_t = pred.proj.2( gelu( pred.proj.0( x_t ) ) )                     # next-state embedding in the same space as emb
```
Rollout: append `pred_{T-1}` as the next `emb`, slide a window of `n_frames`. Tensor names: `enc.proj.{0,2}`,
`pred.pos_embed`, `pred.blk.{i}.{adaln,ln1,attn_qkv,attn_out,ln2,ffn_up,ffn_down}`, `pred.norm`,
`pred.action_embed.{0,2}`, `pred.proj.{0,2}`. Preprocessing: ImageNet mean/std, bilinear resize to 224 (square renders).
`scripts/jepa_convert/selftest.py::lewm_predictor_forward` is the executable reference of this graph.
