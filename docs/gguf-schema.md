# jepa.cpp GGUF schema (v1)

One GGUF file = one *model bundle*: an encoder, and optionally a predictor and/or a head.
All agents (converter, loader, graph builder) implement exactly this. Change it here first.

## General metadata

| key | type | notes |
|---|---|---|
| `general.architecture` | str | always `"jepa"` |
| `general.name` | str | e.g. `"vjepa2-vitl-fpc64-256"` |
| `general.license` | str | `"mit"`, `"cc-by-nc-4.0"`, `"apache-2.0"` — copied from the source |
| `general.source_url` | str | HF repo or download URL |
| `general.file_type` | u32 | ggml ftype (0 = f32, 1 = f16, 7 = q8_0, ...) |
| `jepa.schema_version` | u32 | `1` |
| `jepa.family` | str | `ijepa` · `vjepa` · `vjepa2` · `vjepa2_1` · `hfvit` (LeJEPA-style / LeWM encoder) · `lewm` |
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
| `jepa.enc.rope_interpolate` | bool | V-JEPA 2.1 `interpolate_rope` |
| `jepa.enc.rope_freq_layout` | str | `tiled` (V-JEPA 2: per-axis cos/sin table is `[f_0..f_{d/2-1}, f_0..f_{d/2-1}]`, the Meta/HF quirk) · `interleaved` (V-JEPA 2.1: `[f_0,f_0,f_1,f_1,...]`, textbook RoPE). See `scripts/jepa_convert/VJEPA_NOTES.md` |
| `jepa.enc.rope_ref_grid` | u32 | V-JEPA 2.1 only: spatial RoPE positions are rescaled `h *= (ref_grid-1)/(gh-1)` (16 for patch 16) |
| `jepa.enc.cls_token` | bool | true for `hfvit` (DINOv2-style), false for Meta JEPAs |
| `jepa.enc.n_registers` | u32 | 0 |
| `jepa.enc.qkv_fused` | bool | true → `attn_qkv.*` tensor with rows [q;k;v]; false → separate q/k/v |
| `jepa.enc.modality_embed` | bool | V-JEPA 2.1 |
| `jepa.enc.image_patch_embed` | bool | V-JEPA 2.1 separate 1×16×16 image tokenizer |
| `jepa.enc.hier_layers` | [u32] | V-JEPA 2.1 hierarchical layer ids (per-layer norms exist for these) |
| `jepa.enc.layer_scale` | bool | DINOv2-style `ls1/ls2` present |

Encoder tensors (ggml dim order is reversed from PyTorch; we store PyTorch row-major as-is, i.e. `ne[0]` = last PyTorch dim):

```
enc.patch_embed.weight        [embed_dim, in_chans * tubelet * patch * patch]   (conv flattened to a matmul, C-T-H-W order)
enc.patch_embed.bias          [embed_dim]
enc.patch_embed_img.weight    [embed_dim, in_chans * 1 * patch * patch]        (2.1 only)
enc.patch_embed_img.bias
enc.pos_embed                 [n_tokens, embed_dim]                            (sincos*/learned; includes CLS slot if cls_token)
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
```

Sincos tables are **precomputed at conversion** for the training grid (`enc.pos_embed`); the runtime bicubic/trilinear-interpolates for other grids.

## Predictor (`jepa.pred.*`) — optional

| key | notes |
|---|---|
| `jepa.pred.kind` | `masked` (I-JEPA / V-JEPA / V-JEPA 2 / 2.1) · `ac` (V-JEPA 2-AC) · `lewm` |
| `jepa.pred.embed_dim`, `n_layer`, `n_head`, `ffn_dim` | |
| `jepa.pred.n_mask_tokens` | 10 (V-JEPA 2), 8 (2.1) |
| `jepa.pred.out_dim` | width of `pred.proj` output (enc_dim for V-JEPA 2; teacher_dim / n_hier_in = 1664 for 2.1 ViT-B) |
| `jepa.pred.rope_freq_layout`, `jepa.pred.rope_interpolate`, `jepa.pred.rope_ref_grid`, `jepa.pred.grid_size` | predictor RoPE convention (2.1: `interleaved`, **not** interpolated, mask ids decoded on the fixed `img_size/patch` grid = `grid_size`) |
| `jepa.pred.n_hier_in` | 2.1: number of hierarchical encoder outputs concatenated into `pred.embed` (1 for ViT-B/L, 4 for g/G) |
| `jepa.pred.modality_embed`, `jepa.pred.context_proj` | 2.1: `pred.mod_embed_*` / `pred.proj_context.*` present |
| `jepa.pred.action_dim`, `jepa.pred.state_dim` | 7 / 7 for V-JEPA 2-AC; 10 / 0 for LeWM |
| `jepa.pred.frame_causal` | bool (AC) |
| `jepa.pred.n_frames` | LeWM: 3 |

```
pred.embed.weight/bias        [pred_dim, enc_dim]        (context projection; 2.1: over concatenated hier features → 2-layer MLP: pred.embed.0 / pred.embed.2)
pred.mask_tokens              [n_mask_tokens, pred_dim]
pred.pos_embed                [n_tokens, pred_dim]        (sincos models only)
pred.blk.{i}.*                same layout as enc.blk
pred.norm.weight/bias
pred.proj.weight/bias         [enc_dim (or teacher_dim), pred_dim]
pred.proj_context.weight/bias [teacher_dim, pred_dim]     (2.1 return_all_tokens: projection of the *context* tokens)
pred.mod_embed_img / pred.mod_embed_video  [pred_dim]      (2.1: added to every predictor token)
pred.action_embed.weight/bias [pred_dim, action_dim]
pred.state_embed.weight/bias  [pred_dim, state_dim]
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
| `jepa.pre.mean`, `jepa.pre.std` | [f32 ×3], ImageNet (0.485,0.456,0.406)/(0.229,0.224,0.225) for all Meta models |
| `jepa.pre.resize_short` | short-side resize before crop (I-JEPA HF: 224 via resize to 224×224? → check `preprocessor_config.json`; V-JEPA: crop×256/224) |
| `jepa.pre.crop` | center crop size |
| `jepa.pre.resample` | `bilinear` · `bicubic` (must match the reference processor) |
| `jepa.pre.rescale` | f32, pixel scale before normalisation (1/255) |

## Token order

Video tokens are **T-major, then H, then W** (`i = t*gh*gw + h*gw + w`). Image tokens are H-major then W. `hfvit` prepends CLS (and registers).

## Quantization rules

Quantize only `*.attn_*`, `*.ffn_*`, `pred.proj`, `head.cls` weights. Keep patch embeddings, norms, biases, pos tables, tokens, and `pred.embed` in F32/F16.
