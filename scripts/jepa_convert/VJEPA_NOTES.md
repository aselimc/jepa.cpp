# V-JEPA 2 / V-JEPA 2.1 conversion notes

Companion to `vjepa2.py` (HF safetensors) and `vjepa2_1.py` (Meta torch.hub pickle).
Everything here was read off the actual model code and **verified numerically** with
`vjepa2_numpy_ref.py` (a pure-numpy forward of the GGUF compared against PyTorch, see §6).
Sources:

* HF: `transformers/models/vjepa2/modeling_vjepa2.py` (transformers 5.16.1)
* Meta: `https://github.com/facebookresearch/vjepa2` —
  V-JEPA 2: `src/models/utils/modules.py`, V-JEPA 2.1: `app/vjepa_2_1/models/{vision_transformer,predictor}.py`,
  `app/vjepa_2_1/models/utils/{modules,patch_embed}.py`, factory `src/hub/backbones.py`.

Converted files (all `--ftype f16`; 2-D `*.weight` in F16, everything else F32):

| GGUF | source | tensors | dtypes | tensor data |
|---|---|---|---|---|
| `models/gguf/vjepa2-vitl-fpc64-256-f16.gguf` | `facebook/vjepa2-vitl-fpc64-256` (VJEPA2Model, 587 tensors) | 443 | 147 F16 / 296 F32 | 622.5 MiB |
| `models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf` | `facebook/vjepa2-vitl-fpc16-256-ssv2` (VJEPA2ForVideoClassification, 652 tensors) | 496 | 165 F16 / 331 F32 | 717.0 MiB |
| `models/gguf/vjepa2_1-vitb-384-f16.gguf` | `vjepa2_1_vitb_dist_vitG_384.pt` (`ema_encoder` 158 + `predictor` 162 tensors) | 315 | 101 F16 / 214 F32 | 209.6 MiB |

(HF tensor count drops because q/k/v are fused: 587 = 1 + 2 + 24·16 + ... ; every source key is
consumed — the converters raise if anything is left unmapped.)

---

## 1. Hyper-parameters written

| key | vjepa2-vitl-fpc64-256 | vjepa2-vitl-fpc16-256-ssv2 | vjepa2_1-vitb-384 |
|---|---|---|---|
| `jepa.family` / `modality` | `vjepa2` / `video` | `vjepa2` / `video` | `vjepa2_1` / `image+video` |
| `enc.embed_dim / n_layer / n_head / ffn_dim` | 1024 / 24 / 16 / 4096 | same | 768 / 12 / 12 / 3072 |
| `enc.patch_size / tubelet_size` | 16 / 2 | 16 / 2 | 16 / 2 |
| `enc.img_size / n_frames` | 256 / 64 | 256 / 16 | 384 / 64 |
| `enc.ln_eps / act` | 1e-6 / `gelu_erf` | same | same |
| `enc.pos_type / rope_theta` | `rope3d` / 10000 | same | same |
| `enc.rope_freq_layout` | **`tiled`** | **`tiled`** | **`interleaved`** |
| `enc.rope_interpolate` / `rope_ref_grid` | false / – | false / – | **true / 16** |
| `enc.qkv_fused` | true | true | true |
| `enc.cls_token / n_registers / layer_scale` | false / 0 / false | same | same |
| `enc.modality_embed / image_patch_embed` | false / false | false / false | **true / true** |
| `enc.hier_layers` | – | – | [2, 5, 8, 11] |
| `pred.kind / embed_dim / n_layer / n_head / ffn_dim` | masked / 384 / 12 / 12 / 1536 | same | same |
| `pred.n_mask_tokens / out_dim` | 10 / 1024 | 10 / 1024 | **8 / 1664** |
| `pred.rope_freq_layout / rope_interpolate` | tiled / – | tiled / – | interleaved / **false** (`pred.grid_size` = 24) |
| `pred.n_hier_in / modality_embed / context_proj` | – | – | 1 / true / true |
| `head.kind` | none | `attentive_pool`, 174 classes, 3 pool layers, labels from `id2label` | none |
| `pre.resize_short / crop / resample` | 292 / 256 / bilinear | 292 / 256 / bilinear | 438 / 384 / bilinear |
| `pre.mean / std / rescale` | ImageNet / 1/255 | same | same |

Head dim is 64 everywhere in the encoders (ViT-L 1024/16, ViT-B 768/12) and **32 in the predictors**
(384/12) — the RoPE split therefore differs: `d = 2*((64//3)//2) = 20` per axis (60 rotated, 4 pass-through)
for encoders, `d = 2*((32//3)//2) = 10` per axis (30 rotated, 2 pass-through) for predictors.

`hidden_act = "gelu"` in the HF config maps to `ACT2FN["gelu"]` = exact erf GELU; Meta uses `nn.GELU()`
(erf). `use_SiLU=false` for all released checkpoints (the converter refuses SwiGLU configs).

---

## 2. Key mapping — HF V-JEPA 2 (`vjepa2.py`)

`P` = `""` for `VJEPA2Model`, `"vjepa2."` for `VJEPA2ForVideoClassification`.

| HF key | GGUF tensor | transform |
|---|---|---|
| `{P}encoder.embeddings.patch_embeddings.proj.weight` `[D,3,2,16,16]` | `enc.patch_embed.weight` `[D, 1536]` | `reshape(D, C*T*P*P)` — row-major, i.e. feature order **C, T, H, W** |
| `…proj.bias` | `enc.patch_embed.bias` | |
| `{P}encoder.layer.{i}.norm1.{weight,bias}` | `enc.blk.{i}.ln1.*` | |
| `{P}encoder.layer.{i}.attention.{query,key,value}.weight` | `enc.blk.{i}.attn_qkv.weight` `[3D, D]` | `concat([Wq, Wk, Wv], dim=0)` (rows `[q; k; v]`, same layout as Meta's `attn.qkv`) |
| `…attention.{query,key,value}.bias` | `enc.blk.{i}.attn_qkv.bias` `[3D]` | concat |
| `…attention.proj.*` | `enc.blk.{i}.attn_out.*` | |
| `…norm2.*` | `enc.blk.{i}.ln2.*` | |
| `…mlp.fc1.*` / `…mlp.fc2.*` | `enc.blk.{i}.ffn_up.*` / `ffn_down.*` | |
| `{P}encoder.layernorm.*` | `enc.norm.*` | |
| `{P}predictor.embeddings.predictor_embeddings.*` | `pred.embed.*` `[384, 1024]` | |
| `{P}predictor.embeddings.mask_tokens` `[10,1,1,384]` | `pred.mask_tokens` `[10, 384]` | reshape |
| `{P}predictor.layer.{i}.*` | `pred.blk.{i}.*` | as encoder blocks (q/k/v fused) |
| `{P}predictor.layernorm.*` | `pred.norm.*` | |
| `{P}predictor.proj.*` | `pred.proj.*` `[1024, 384]` | |
| `pooler.query_tokens` `[1,1,1024]` | `head.query` `[1, 1024]` | reshape |
| `pooler.self_attention_layers.{i}.layer_norm1.*` | `head.blk.{i}.ln1.*` | i = 0..2 |
| `…self_attn.{q,k,v}_proj.*` | `head.blk.{i}.attn_qkv.*` | fused `[q;k;v]` |
| `…self_attn.out_proj.*` | `head.blk.{i}.attn_out.*` | |
| `…layer_norm2.*`, `…mlp.fc1/fc2.*` | `head.blk.{i}.ln2.*`, `ffn_up/ffn_down.*` | |
| `pooler.cross_attention_layer.layer_norm1.*` | `head.xattn.ln_kv.*` | applied to keys/values **only** |
| `pooler.cross_attention_layer.cross_attn.{q,k,v}_proj.*` | `head.xattn.{q,k,v}.*` | not fused (q sees 1 token, k/v see N) |
| `pooler.cross_attention_layer.layer_norm2.*` | `head.xattn.ln2.*` | |
| `pooler.cross_attention_layer.mlp.fc1/fc2.*` | `head.xattn.ffn_up/ffn_down.*` | |
| `classifier.*` | `head.cls.*` `[174, 1024]` | |

Nothing is dropped; the HF safetensors contain no buffers beyond the parameters above.
The predictor is exported for the classifier checkpoint too (it is present in the safetensors).

### Attentive-pool head — exact forward (HF `VJEPA2AttentivePooler` + `classifier`)

```
x = enc_out                              # [N, D], after enc.norm
for i in 0..n_pool_layers-1: x = x + attn(ln1(x)); x = x + mlp(ln2(x))      # head.blk.i, NO RoPE
q0 = head.query                          # [1, D]   (raw token, NOT normalised)
kv = ln_kv(x)                            # head.xattn.ln_kv
q = q0 Wq^T + bq ; k = kv Wk^T + bk ; v = kv Wv^T + bv
o = concat_heads( softmax(q_h k_h^T / sqrt(64)) v_h )   # no output projection
y = q0 + o                               # residual is the raw query
y = y + ffn_down(gelu(ffn_up(ln2(y))))   # head.xattn.ln2 / ffn_up / ffn_down
logits = y Wcls^T + bcls                 # [n_classes]; no final norm
```
Self-attention layers run **before** the cross-attention (HF order), all with head_dim 64, 16 heads.

---

## 3. Key mapping — Meta V-JEPA 2.1 (`vjepa2_1.py`)

Pickle top-level keys: `encoder`, `predictor`, `opt`, `scaler`, `ema_encoder`, `epoch`, `loss`,
`batch_size`, `world_size`, `lr`. Parameter names are `module.backbone.<name>`; the converter strips
`module.` and `backbone.` exactly like `src/hub/backbones.py::_clean_backbone_key`.

The hub factory `vjepa2_1_vit_base_384` loads `checkpoint_key="ema_encoder"` (the EMA / target
encoder, which is the distilled model) — so does the converter (`--encoder-key` to override).

**Intentionally dropped** (not model weights, or not used by the hub model):

| pickle key | why dropped |
|---|---|
| `encoder` (158 tensors) | the *student* (online) encoder; the released/hub model is `ema_encoder`. Convert with `--encoder-key encoder` if ever needed. |
| `opt` | optimizer state |
| `scaler` | AMP grad-scaler state |
| `epoch` (40), `loss`, `batch_size` (72), `world_size` (512), `lr` | training bookkeeping scalars |

Every tensor of `ema_encoder` and `predictor` is mapped (the converter raises otherwise):

| pickle key (prefix stripped) | GGUF tensor | transform |
|---|---|---|
| `patch_embed.proj.weight` `[768,3,2,16,16]` / `.bias` | `enc.patch_embed.*` `[768, 1536]` | reshape, C-T-H-W |
| `patch_embed_img.proj.weight` `[768,3,1,16,16]` / `.bias` | `enc.patch_embed_img.*` `[768, 768]` | reshape (tubelet 1) |
| `img_mod_embed` / `video_mod_embed` `[1,1,768]` | `enc.mod_embed_img` / `enc.mod_embed_video` `[768]` | reshape |
| `blocks.{i}.norm1/norm2.*` | `enc.blk.{i}.ln1/ln2.*` | |
| `blocks.{i}.attn.qkv.{weight,bias}` `[2304,768]` | `enc.blk.{i}.attn_qkv.*` | already fused `[q;k;v]` |
| `blocks.{i}.attn.proj.*` | `enc.blk.{i}.attn_out.*` | |
| `blocks.{i}.mlp.fc1/fc2.*` | `enc.blk.{i}.ffn_up/ffn_down.*` | |
| `norms_block.{k}.*`, k = 0..3 | `enc.hier_norm.{k}.*` | k indexes `jepa.enc.hier_layers = [2,5,8,11]` |
| `norms_block.3.*` (the last) | **also** `enc.norm.*` | inference output norm (`VisionTransformer.forward`: `x = self.norms_block[-1](x)`) |
| `predictor_embed.{weight,bias}` `[384,768]` | `pred.embed.*` | single Linear because `n_output_distillation=1` (ViT-g/G would have `predictor_embed.0/.2` → `pred.embed.0/.2`, handled) |
| `mask_tokens.{0..7}` `[1,1,384]` | `pred.mask_tokens` `[8, 384]` | stacked in index order |
| `img_mod_embed` / `video_mod_embed` `[1,1,384]` (predictor's own) | `pred.mod_embed_img` / `pred.mod_embed_video` `[384]` | |
| `predictor_blocks.{i}.*` | `pred.blk.{i}.*` | as encoder |
| `predictor_norm.*` | `pred.norm.*` | |
| `predictor_proj.*` `[1664, 384]` | `pred.proj.*` | output head → teacher dim (1664 = ViT-G width, `n_output_distillation=1`) |
| `predictor_proj_context.*` `[1664, 384]` | `pred.proj_context.*` | `return_all_tokens=True`: projection of the context tokens |

There is no `pos_embed` anywhere (RoPE only), no CLS, no registers.

### 2.1 encoder forward (what the graph must do)

```
video  x[C, T, H, W]  (T multiple of 2) : tokens = patchify(x, 2, 16) Wpe^T + bpe ; tokens += enc.mod_embed_video
image  x[C, 1, H, W]                    : tokens = patchify(x, 1, 16) Wpe_img^T + bpe_img ; tokens += enc.mod_embed_img
blocks with 3-D RoPE (interleaved, interpolated, §4.3)
out = enc.norm(x)            # == norms_block[3]; hier features = norms_block[k](x after block hier_layers[k])
```
Meta's code selects the image path by `x.shape[2] == img_temporal_dim_size (1)`; images must be fed
as a 5-D tensor with one frame (a 4-D image tensor would hit the video Conv3d and fail).

### 2.1 predictor forward

```
ctx = enc_out[ids_ctx] Wemb^T + bemb                       # [N_ctx, 384]
tgt = mask_tokens[mask_index % 8] repeated for ids_tgt      # default mask_index = 1
# NOTE (verified on both released checkpoints): only mask_tokens[0] is non-zero (norm 0.77 in V-JEPA 2 ViT-L, 1.10 in 2.1 ViT-B);
# mask_tokens[1..] are exactly zero, so the default mask_index=1 selects an all-zero token and mask_index tests are numerically insensitive.
x = concat(ctx, tgt) ; x += pred.mod_embed_video (or _img)   # every token
blocks with 3-D RoPE from ids (interleaved, NOT interpolated, grid = img_size/patch = 24)
x = pred.norm(x)
pred     = x[N_ctx:] Wproj^T + bproj              # [N_tgt, 1664]
ctx_out  = x[:N_ctx] Wproj_ctx^T + bproj_ctx       # [N_ctx, 1664]
```
Meta sorts the concatenated tokens by id before the blocks and unsorts after; attention is
permutation-equivariant so the C++ side may skip the sort.

---

## 4. The 3-D RoPE — exact conventions (READ THIS, kernel author)

Common to both families:

* `head_dim = 64` (encoder) or `32` (predictor); `d = 2 * ((head_dim // 3) // 2)` per axis
  (20 / 10). Dims `[0, d)` rotate with the **frame** position, `[d, 2d)` with **height**, `[2d, 3d)`
  with **width**; dims `[3d, head_dim)` (4 / 2 of them) are passed through unchanged.
* Token id → positions (T-major, then H, then W):
  `t = i // (gh*gw)`, `h = (i mod gh*gw) // gw`, `w = i mod gw`. `t` counts **tubelets**, not frames.
  V-JEPA 2 (HF) hard-codes `gh = gw = crop_size/patch` from the config; Meta 2.1 uses the actual
  `H_patches`, `W_patches` of the input (non-square works, verified).
* Frequencies per axis: `omega_k = theta ** (-(2k)/d) = 1 / 10000 ** (k / (d/2))`, `k = 0..d/2-1`.
  (`omega = arange(d/2) / (d/2); omega = 1 / 10000**omega`.)
* Rotation acts on **adjacent, interleaved pairs** `(x[2k], x[2k+1])` within each axis segment:
  `out = x * COS + rot90(x) * SIN` with `rot90(x)[2k] = -x[2k+1]`, `rot90(x)[2k+1] = x[2k]`.
* Applied to **q and k** (not v), after the qkv projection + bias, before the 1/sqrt(head_dim) scaling.
  Positions are per token and identical for every head and layer.

The two families differ **only** in how the length-`d/2` table `f_k = pos * omega_k` is expanded to
length `d` — and this is not a detail: swapping the layout gives cos 0.63 (V-JEPA 2) / 0.91 (2.1)
against the reference (§6).

### 4.1 V-JEPA 2 (HF `VJEPA2RopeAttention` and Meta `src/models/utils/modules.py`): "tiled"

```python
# HF transformers/models/vjepa2/modeling_vjepa2.py::rotate_queries_or_keys
omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
omega /= D / 2.0
omega = 1.0 / 10000**omega                       # (D/2,)
freq = pos.unsqueeze(-1) * omega                 # (..., N, D/2)
emb_sin = freq.sin().repeat(1, 1, 1, 2)          # (..., N, D)   <-- TILE: [f0..f9, f0..f9]
emb_cos = freq.cos().repeat(1, 1, 1, 2)
y = x.unflatten(-1, (-1, 2)); y1, y2 = y.unbind(dim=-1)
y = torch.stack((-y2, y1), dim=-1).flatten(-2)   # rot90 on interleaved pairs
return (x * emb_cos) + (y * emb_sin)
```
Meta's V-JEPA 2 code is identical and carries this comment:
```python
# -- NOTE: This expansion has a subtle bug where frequencies are duplicated across the vector pair.
# -- Fixing the bug would break compatibility with the pretrained model, but the fix can be applied by
# -- commenting out the two lines below, and uncommenting the following two lines.
# -- Thanks to @echosprint, original PR: https://github.com/facebookresearch/vjepa2/pull/15
emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
# emb_sin = emb_sin.repeat_interleave(2, dim=-1)  # (..., N, D)
# emb_cos = emb_cos.repeat_interleave(2, dim=-1)  # (..., N, D)
```
So with `d = 20`, `j = 0..19`, **`COS[j] = cos(pos * omega_{j mod 10})`**, and the pair
`(x[2k], x[2k+1])` is combined with two *different* angles:

```
out[2k]   = x[2k]   * cos(pos*omega_{(2k)   mod 10}) - x[2k+1] * sin(pos*omega_{(2k)   mod 10})
out[2k+1] = x[2k+1] * cos(pos*omega_{(2k+1) mod 10}) + x[2k]   * sin(pos*omega_{(2k+1) mod 10})
```
This is **not** a rotation (not orthogonal), it is simply what the weights were trained with.
Pairs k = 0..4 use omega_0..omega_9, pairs k = 5..9 use omega_0..omega_9 again. Metadata:
`jepa.enc.rope_freq_layout = "tiled"` (also `jepa.pred.rope_freq_layout`).

Implementation hint: build per-token tables `COS, SIN [N, head_dim]` on the host exactly as above
(ones/zeros in the pass-through dims) and apply `q*COS + rot90(q)*SIN`; `rot90` is a view
`(-q[...,1::2], q[...,0::2])`. ggml's `ggml_rope_multi` cannot express the tiled table.

### 4.2 V-JEPA 2.1 (Meta `app/vjepa_2_1/models/utils/modules.py`): "interleaved" (textbook)

```python
def rotate_queries_or_keys(x, pos, n_registers, has_cls_first):
    ...
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin()
    emb_cos = freq.cos()
    emb_sin = emb_sin.repeat_interleave(2, dim=-1)     # <-- [f0,f0,f1,f1,...,f9,f9]
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)
    y = x_ctx.unflatten(-1, (-1, 2)); y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)
```
i.e. **`COS[2k] = COS[2k+1] = cos(pos * omega_k)`** — a proper 2-D rotation of each pair by
`pos * omega_k` (GPT-J / "interleaved" RoPE style, *not* the half-split/NeoX style that ggml's
`ggml_rope` mode 0 uses on pairs `(x[k], x[k+d/2])`). Metadata: `rope_freq_layout = "interleaved"`.
This is the "corrected RoPE" of the 2.1 release: the diff to 4.1 is exactly
`repeat(1,1,1,2)` → `repeat_interleave(2, dim=-1)`; the `n_registers`/`has_cls_first` slicing is
inert for the released checkpoints (0 / False). Positions are cast to float (`1.0 * frame_ids`).

### 4.3 `interpolate_rope` (2.1 encoder only) — what it does mathematically

```python
# RoPEAttention.__init__:  pretrained_grid_size = 256 // patch_size   (16 for patch 16, 18 for patch 14)
# RoPEAttention.forward:
if self.interpolate_rope:
    h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
    w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)
```
Spatial positions are **linearly rescaled so the last row/column always lands on 15** (patch 16):
`h' = h * 15 / (gh - 1)`, `w' = w * 15 / (gw - 1)`; the temporal position is *not* rescaled.
`pretrained_grid_size` is hard-coded from the 256-px V-JEPA 2 initialisation and does **not** change
with `img_size=384` — at the native 384 input the model runs with `h' = h * 15/23`, i.e. fractional
positions. Any input resolution therefore maps onto the same [0, 15] span (a 1×1 grid would divide by
zero; guard `gh > 1`). Stored as `jepa.enc.rope_interpolate = true`, `jepa.enc.rope_ref_grid = 16`.

The **2.1 predictor does not interpolate** (`src/hub/backbones.py` passes `interpolate_rope=True`
only in `vit_encoder_kwargs`); its blocks get `T=H_patches=W_patches=None` and decode the mask ids on
the fixed `grid_size = img_size // patch = 24` grid with raw integer positions
(`jepa.pred.rope_interpolate = false`, `jepa.pred.grid_size = 24`).

### 4.4 Predictor position ids (both families)

The predictor receives explicit ids (`context_mask` / `masks_x`, `target_mask` / `masks_y`) which
index the **full** encoder grid `T/2 × gh × gw` (HF: `crop_size/patch`, 2.1: `img_size/patch`);
positions are decoded from those ids exactly as in §4. Context tokens and mask tokens with the same
id get the same position (HF default: both = `arange(N)` → 2N tokens).

---

## 5. Preprocessing references

* HF `VJEPA2VideoProcessor` (torchvision backend): RGB → `resize(shortest_edge = int(crop*256/224) = 292,
  BILINEAR, antialias=True)` → center-crop 256×256 → `/255` → ImageNet mean/std. Tensor layout
  `[B, T, C, H, W]`; the model permutes to `[B, C, T, H, W]` before the Conv3d. If `T < tubelet`
  the frames are repeated (HF), i.e. a single image becomes a 2-frame clip.
* Meta eval transforms (V-JEPA 2 and 2.1, `evals/*/eval.py`): video → `Resize(short side
  int(img*256/224), bilinear)` (cv2 `INTER_LINEAR`) → `CenterCrop(img)` → ImageNet normalisation;
  images → torchvision `Resize(int(img*256/224))` (bilinear) → `CenterCrop(img)` → normalise. For
  the 2.1 384-px models: short side 438, crop 384 (`jepa.pre.*`).
* 2.1 **image** inference uses `patch_embed_img` with a 1-frame 5-D tensor (`[B, C, 1, H, W]`). The
  Meta ImageNet probe configs instead replicate the image 16/18× and go through the video tokenizer
  (`img_as_video_nframes`, `evals/image_classification_frozen`); both are valid encodings, they are
  not equivalent. jepa.cpp should expose the native image path (`jepa.enc.image_patch_embed`).

---

## 6. Verification (numbers a reviewer can reproduce)

Convert (from the worktree root; `.venv` = `/home/overseer2/workdir/jepa.cpp/.venv`):
```
python scripts/jepa_convert/vjepa2.py   --src models/facebook/vjepa2-vitl-fpc64-256      --out models/gguf/vjepa2-vitl-fpc64-256-f16.gguf      --ftype f16
python scripts/jepa_convert/vjepa2.py   --src models/facebook/vjepa2-vitl-fpc16-256-ssv2 --out models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf --ftype f16
python scripts/jepa_convert/vjepa2_1.py --src models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt --out models/gguf/vjepa2_1-vitb-384-f16.gguf --ftype f16
python scripts/jepa_convert/vjepa2.py --info models/gguf/vjepa2-vitl-fpc64-256-f16.gguf --tensors
```

Parity of the numpy forward (`vjepa2_numpy_ref.py`) against PyTorch, random-normal inputs, f32
math. `rel` = max|diff| / max|ref|:

| GGUF | case | cosine | max abs | rel |
|---|---|---|---|---|
| fpc64 **f32** | encoder, 4×256×256 (512 tokens) | 1.0000000 | 4.9e-4 | 1.1e-5 |
| fpc64 f32 | predictor (ctx = tgt = all 512) | 1.0000000 | 9.1e-6 | 1.2e-6 |
| ssv2 f32 | encoder / predictor | 1.0000000 | 4.9e-4 / 9.1e-6 | 1.1e-5 / 1.2e-6 |
| ssv2 f32 | head logits (174) | 1.0000000 | 6.3e-6 | 1.6e-6, top-1 45 = 45 |
| 2.1 **f32** | video 2×384×384 (576 tok) encoder | 1.0000000 | 3.7e-4 | 1.4e-5 |
| 2.1 f32 | predictor targets / context proj | 1.0000000 | 5.6e-4 / 2.7e-3 | 4.1e-5 / 1.7e-4 |
| 2.1 f32 | image 1×384×384 (patch_embed_img) | 1.0000000 | 2.7e-4 | 9.9e-6 |
| 2.1 f32 | video 4×320×320 (rope interp 15/19) | 1.0000000 | 2.1e-3 | 7.9e-5 |
| 2.1 f32 | video 2×256×320 (non-square 16×20) | 1.0000000 | 8.3e-5 | 3.1e-6 |
| 2.1 **f16** (deliverable) | encoder video / image / interp / non-square | 0.9999999–1.0000000 | ≤ 1.3e-2 | ≤ 4.8e-4 |
| 2.1 f16 | predictor targets / context | 0.9999998 / 0.9999999 | 2.1e-2 / 2.2e-2 | 1.5e-3 / 1.4e-3 |
| fpc64 / ssv2 **f16** (deliverables) | encoder 4×256×256 | 0.9999990 | 2.8e-1 | 6.3e-3 (24 ViT-L layers of f16 weights) |
| fpc64 / ssv2 f16 | predictor | 0.9999999 | 1.8e-3 | 2.5e-4 |
| ssv2 f16 | head logits | 0.9999999 | 1.9e-3 | 5.0e-4, top-1 45 = 45 |

| 2.1 **q8_0** (`--ftype q8_0`, 98 Q8_0 + 3 F16 + 214 F32 tensors, 113.6 MiB) | encoder video / image / interp / non-square | 0.99994–0.99997 | ≤ 2.3e-1 | ≤ 8.8e-3 |
| 2.1 q8_0 | predictor targets / context | 0.99988 / 0.99987 | 1.1e-1 / 1.5e-1 | 8.2e-3 / 9.5e-3 |

(fpc64 and ssv2 give bit-identical encoder/predictor numbers: all 587 backbone + predictor tensors of
`vjepa2-vitl-fpc16-256-ssv2` are `array_equal` to `vjepa2-vitl-fpc64-256` — the SSv2 checkpoint is
the frozen pretrained ViT-L plus a trained attentive pooler + classifier.)

Negative control (same script, `--rope-layout` override):

| model | wrong layout | encoder cosine | predictor cosine |
|---|---|---|---|
| vjepa2-vitl-fpc64-256 | `interleaved` | **0.634** | 0.839 |
| vjepa2_1-vitb-384 | `tiled` | **0.914** | 0.911 |

Commands:
```
python scripts/jepa_convert/vjepa2_numpy_ref.py --gguf models/gguf/vjepa2-vitl-fpc64-256-f16.gguf      --hf models/facebook/vjepa2-vitl-fpc64-256      --frames 4 --size 256
python scripts/jepa_convert/vjepa2_numpy_ref.py --gguf models/gguf/vjepa2-vitl-fpc16-256-ssv2-f16.gguf --hf models/facebook/vjepa2-vitl-fpc16-256-ssv2 --frames 4 --size 256
python scripts/jepa_convert/vjepa2_numpy_ref.py --gguf models/gguf/vjepa2_1-vitb-384-f16.gguf --meta models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt --vjepa2-src tmp/vjepa2-src
python scripts/jepa_convert/vjepa2_numpy_ref.py --gguf ... --hf ... --rope-layout interleaved     # expect FAIL
```
