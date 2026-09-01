# jepa_convert — checkpoint → GGUF converters

Python side of jepa.cpp. Turns the reference PyTorch checkpoints into the GGUF layout
defined in [`docs/gguf-schema.md`(https://github.com/aselimc/jepa.cpp/blob/main/docs/gguf-schema.md). Inference only; no training code.

```
scripts/convert.py              CLI entry point (dispatches on --family)
scripts/jepa_convert/common.py  JepaWriter, sincos tables, qkv fusing, BN folding, preprocessing keys
scripts/jepa_convert/ijepa.py   facebook/ijepa_*             (HF IJepaModel safetensors)
scripts/jepa_convert/hfvit.py   OK-AI/lejepa-*, DINOv2-style (timm layout safetensors)  + HF-ViT source mapper
scripts/jepa_convert/lewm.py    quentinll/lewm-*             (weights.pt: HF ViT encoder + adaLN predictor)
scripts/jepa_convert/selftest.py numpy forward straight from a GGUF vs. the PyTorch reference
scripts/jepa_convert/vjepa2.py, vjepa2_1.py   standalone video-family converters (separate owner)
scripts/jepa_convert/levjepa.py galilai-group/LeVJEPA-*      (custom HF architecture; reuses vjepa2.py's writer)
```

## Usage

```bash
PY=.venv/bin/python
$PY scripts/convert.py --family ijepa --src models/facebook/ijepa_vith14_1k            --ftype f16
$PY scripts/convert.py --family hfvit --src models/OK-AI/lejepa-vits16-pretrain-in1k    --ftype f32
$PY scripts/convert.py --family lewm  --src models/quentinll/lewm-pusht                 --ftype f16
$PY scripts/convert.py --family levjepa --src models/galilai-group/LeVJEPA-VideoMix-Large \
                       --out models/gguf/levjepa-vitl16-f32.gguf --ftype f32
# default output: models/gguf/<basename of --src>-<ftype>.gguf ; override with --out
# --name overrides general.name ; --allow-unmapped tolerates unknown source tensors (default: hard error)
```

`--ftype f32` writes every tensor as F32. `--ftype f16` writes the *quantizable* weights
(`*.attn_*`, `*.ffn_*`, `enc.proj.*`, `pred.proj*`, `head.cls`) as F16 and everything else
(patch embed, norms, biases, position tables, tokens, adaLN, action embed) as F32. Q8_0 / Q4_K
come from `tools/jepa-quantize`, never from Python.

Every converter fails loudly if a source tensor is neither mapped nor on its explicit skip list,
and cross-checks every hparam against tensor shapes (layer count, ffn dim, patch kernel, pos rows).

### Self-test

```bash
$PY scripts/jepa_convert/selftest.py --gguf models/gguf/lejepa-vits16-pretrain-in1k-f32.gguf --src models/OK-AI/lejepa-vits16-pretrain-in1k
$PY scripts/jepa_convert/selftest.py --gguf models/gguf/lewm-pusht-f32.gguf --src models/quentinll/lewm-pusht \
      --lewm-src tmp/<branch>/stable-worldmodel      # git clone --depth 1 https://github.com/galilai-group/stable-worldmodel
$PY scripts/jepa_convert/selftest.py --gguf models/gguf/ijepa_vith14_1k-f16.gguf --src models/facebook/ijepa_vith14_1k --n-images 2
$PY scripts/jepa_convert/selftest.py --gguf models/gguf/levjepa-vitl16-f32.gguf --src models/galilai-group/LeVJEPA-VideoMix-Large --n-images 1
```

It reads only the GGUF (tensors + `jepa.*` keys), runs the architecture.md graph in numpy on
fixture images, and compares against timm / transformers / stable-worldmodel. The numpy code in
`selftest.py` (`encoder_forward`, `lewm_predictor_forward`, `levjepa_encoder_forward`) is the
executable spec of what the C++ graph builder has to do. Observed: f32 files agree to ~1e-6 relative, f16 to cosine ≥ 0.99999.

## common.py

| helper | what it does |
|---|---|
| `JepaWriter(path, ftype, name=, family=, modality=, license=, source_url=)` | `gguf.GGUFWriter(arch="jepa")` + `general.*` + `jepa.schema_version/family/modality` |
| `.add_hparams(dict)` | writes every `jepa.*` key with the type from `HPARAM_TYPES` (u32 / f32 / bool / str / [u32] / [f32] / [str]); unknown keys are inferred with a warning |
| `.add_tensor(name, arr, quantizable)` | dtype rule above; rejects NaN/Inf and scalars |
| `.add_linear / .add_norm` | weight + optional bias shortcuts |
| `sincos_2d(gh, gw, dim, w_first=False)` | bit-identical to `get_2d_sincos_pos_embed` (vjepa2 `pos_embs.py`); `w_first=True` gives the `[emb_w \| emb_h]` variant that the HF I-JEPA checkpoints contain |
| `sincos_3d(gt, gh, gw, dim, uniform_power)` | bit-identical to `get_3d_sincos_pos_embed` |
| `fuse_qkv(q, k, v)` | rows `[q; k; v]` for weights and biases |
| `flatten_patch_conv(w)` | `[D,C,(T,)P,P] → [D, C*T*P*P]`, C-T-H-W order |
| `fold_batchnorm(w, b, γ, β, mean, var, eps)` | eval-mode BN folded into the preceding Linear (float64 math) |
| `fold_linear_pair(w1, b1, w2, b2)` | `Linear2∘Linear1` → one Linear |
| `preproc_from_hf(preprocessor_config.json)` | `jepa.pre.mean/std/resize_short/crop/resample/resize_mode` |
| `SourceTensors` | dict-like checkpoint view that records consumed keys (`take`, `take_opt`, `drop`, `unused`, `layer_count`) |

## Family: `ijepa` (facebook/ijepa_vith14_1k)

| source (safetensors) | GGUF | note |
|---|---|---|
| `embeddings.patch_embeddings.projection.weight` `[1280,3,14,14]` | `enc.patch_embed.weight` `[1280,588]` | flattened |
| `embeddings.patch_embeddings.projection.bias` | `enc.patch_embed.bias` | |
| `embeddings.position_embeddings` `[1,256,1280]` | `enc.pos_embed` `[256,1280]` | stored verbatim; it **is** a 2-D sincos table (bit-identical to `sincos_2d(16,16,1280, w_first=True)`, max diff 0.0; the `[emb_h\|emb_w]` ordering of `vjepa2/pos_embs.py` differs by 2.0) |
| `encoder.layer.{i}.layernorm_before.*` | `enc.blk.{i}.ln1.*` | |
| `encoder.layer.{i}.attention.attention.{query,key,value}.*` | `enc.blk.{i}.attn_qkv.*` | fused `[3840,1280]` |
| `encoder.layer.{i}.attention.output.dense.*` | `enc.blk.{i}.attn_out.*` | |
| `encoder.layer.{i}.layernorm_after.*` | `enc.blk.{i}.ln2.*` | |
| `encoder.layer.{i}.intermediate.dense.*` | `enc.blk.{i}.ffn_up.*` | `[5120,1280]` |
| `encoder.layer.{i}.output.dense.*` | `enc.blk.{i}.ffn_down.*` | |
| `layernorm.*` | `enc.norm.*` | |

transformers ≥ 5 names (`layers.{i}.attention.q_proj`, `mlp.fc1`, …) are accepted too.
hparams: D 1280, 32 layers, 16 heads, ffn 5120, patch 14, img 224, LN eps 1e-6, `gelu_erf`,
`pos_type=sincos2d`, no CLS, `qkv_fused`. Features = mean of the 256 patch tokens.
Preprocessing **as shipped by HF** (`ViTImageProcessor`): direct resize to 224×224 (`resize_mode=squash`),
bilinear, mean/std **0.5/0.5/0.5** — not the ImageNet statistics. `general.license = cc-by-nc-4.0`.

## Family: `hfvit` (OK-AI/lejepa-vits16-pretrain-in1k)

Source is the OK-AI "ViTv2" (DINOv2 ViT without registers, from `Open-Knowledge-AI/lite_ssl`),
stored with a `backbone.` prefix in timm layout. Verified from the source: `LayerNorm(eps=1e-6)`,
`nn.GELU` (erf), `qkv_bias=proj_bias=ffn_bias=True`, learned `pos_embed` with the CLS slot at row 0,
`num_register_tokens=0`, `init_values=null` → no LayerScale.

| source | GGUF | note |
|---|---|---|
| `backbone.patch_embed.proj.{weight,bias}` `[384,3,16,16]` | `enc.patch_embed.*` `[384,768]` | |
| `backbone.cls_token` `[1,1,384]` | `enc.cls_token` `[384]` | |
| `backbone.pos_embed` `[1,197,384]` | `enc.pos_embed` `[197,384]` | row 0 = CLS |
| `backbone.register_tokens` | `enc.reg_tokens` | absent here |
| `backbone.blocks.{i}.norm1 / norm2` | `enc.blk.{i}.ln1 / ln2` | |
| `backbone.blocks.{i}.attn.qkv.*` `[1152,384]` | `enc.blk.{i}.attn_qkv.*` | already fused (q,k,v row blocks) |
| `backbone.blocks.{i}.attn.proj.*` | `enc.blk.{i}.attn_out.*` | |
| `backbone.blocks.{i}.mlp.fc1 / fc2` | `enc.blk.{i}.ffn_up / ffn_down` | `[1536,384]` / `[384,1536]` |
| `backbone.blocks.{i}.ls1.gamma / ls2.gamma` | `enc.blk.{i}.ls1 / ls2` | only when `layer_scale` |
| `backbone.norm.*` | `enc.norm.*` | |
| `backbone.cva_module_proj.mlp.{0,2,4}.*` | — | DINOHead of the SIGReg/CVA loss; listed and skipped |

150 of 156 source tensors mapped; the 6 unmapped are exactly `cva_module_proj.*`.
Preprocessing (`BitImageProcessor`): short side 256 bicubic, centre crop 224, ImageNet mean/std.
Features: CLS token (`latent`) or patch tokens (`patch_latent`), both after `enc.norm`.

`map_hf_vit()` in the same file maps a transformers `ViTModel` state dict
(`embeddings.cls_token`, `encoder.layer.{i}.attention.attention.query`, …) to the identical
`enc.*` layout; it is used by the LeWM converter.

## Family: `lewm` (quentinll/lewm-pusht)

`weights.pt` is a plain `LeWM` state dict. Sub-modules and their GGUF names:

| source | GGUF | note |
|---|---|---|
| `encoder.*` (HF `ViTModel`, tiny/14: D 192, 12 L, 3 heads, ffn 768, 257 tokens) | `enc.*` via `map_hf_vit` | LN eps **1e-12** (ViTConfig default), `gelu_erf`, CLS + learned pos |
| `projector.net.0` Linear 192→2048 + `net.1` BatchNorm1d(2048) | `enc.proj.0.{weight,bias}` `[2048,192]` | BN folded (eps 1e-5) |
| `projector.net.3` Linear 2048→192 | `enc.proj.2.{weight,bias}` | GELU between (`jepa.enc.proj_act`) |
| `predictor.pos_embedding` `[1,3,192]` | `pred.pos_embed` `[3,192]` | |
| `predictor.transformer.layers.{i}.adaLN_modulation.1.*` `[1152,192]` | `pred.blk.{i}.adaln.*` | applied to `silu(action_emb)`; chunks = shift/scale/gate (attn), shift/scale/gate (mlp) |
| `predictor.transformer.layers.{i}.attn.norm.*` | `pred.blk.{i}.ln1.*` | affine LN eps 1e-5 |
| `predictor.transformer.layers.{i}.attn.to_qkv.weight` `[3072,192]` | `pred.blk.{i}.attn_qkv.weight` | no bias; 16 heads × 64 |
| `predictor.transformer.layers.{i}.attn.to_out.0.*` `[192,1024]` | `pred.blk.{i}.attn_out.*` | |
| `predictor.transformer.layers.{i}.mlp.net.0.*` | `pred.blk.{i}.ln2.*` | affine LN eps 1e-5 |
| `predictor.transformer.layers.{i}.mlp.net.1 / net.4` | `pred.blk.{i}.ffn_up / ffn_down` | `[2048,192]` / `[192,2048]` |
| `predictor.transformer.norm.*` | `pred.norm.*` | |
| `action_encoder.patch_embed` Conv1d(10,10,k=1) + `embed.0` Linear 10→768 | `pred.action_embed.0.*` `[768,10]` | Conv1d folded into the Linear |
| `action_encoder.embed.2` Linear 768→192 | `pred.action_embed.2.*` | SiLU between (`jepa.pred.action_act`) |
| `pred_proj.net.0` + `net.1` BatchNorm1d | `pred.proj.0.*` `[2048,192]` | BN folded |
| `pred_proj.net.3` | `pred.proj.2.*` `[192,2048]` | GELU between (`jepa.pred.proj_act`) |

The non-affine adaLN LayerNorms (`norm1/norm2`, eps 1e-6, `jepa.pred.adaln_eps`) have no tensors.
Full inference graph: see the `lewm` subsection of `docs/gguf-schema.md` and
`lewm_predictor_forward` in `selftest.py`. Preprocessing (training pipeline): ImageNet mean/std,
bilinear resize to 224 (PushT renders are square).

## Family: `levjepa` (galilai-group/LeVJEPA-VideoMix-Large)

`model.safetensors` is the EMA copy of the encoder and nothing else: 293 tensors, all F32, no
predictor, no head, no projector. The architecture is custom and ships with the weights
(`modeling_levjepa.py` + `configuration_levjepa.py`), so `trust_remote_code=True` is what loads the
reference — and `levjepa.py` reuses the writer, the safetensors reader and the dtype policy of
`vjepa2.py`, since the rules are identical.

| source | GGUF | note |
|---|---|---|
| `encoder.patch_embed.proj.weight` `[1024,3,1,16,16]` | `enc.patch_embed.weight` `[1024,768]` | Conv3d with **tubelet 1**, flattened C-T-H-W like the other video families |
| `encoder.patch_embed.proj.bias` | `enc.patch_embed.bias` | |
| `encoder.cls_token` `[1,1,1024]` | `enc.cls_token` `[1024]` | prepended after the patch embedding; no pos-embed row and no RoPE |
| `encoder.blocks.{i}.norm1/norm2.*` | `enc.blk.{i}.ln1/ln2.*` | LN eps **1e-6** (hard-coded in `LeVJEPAModel`, not in the config) |
| `encoder.blocks.{i}.attn.qkv.*` `[3072,1024]` | `enc.blk.{i}.attn_qkv.*` | already fused as `[q; k; v]`, nothing to concatenate |
| `encoder.blocks.{i}.attn.proj.*` | `enc.blk.{i}.attn_out.*` | |
| `encoder.blocks.{i}.mlp.fc1/fc2.*` | `enc.blk.{i}.ffn_up/ffn_down.*` | `[4096,1024]` / `[1024,4096]` |
| `encoder.norm.*` | `enc.norm.*` | |

Two config keys carry into the graph and one does not. `attn_mode = "block_causal"` becomes
`jepa.enc.attn_mode` and is mandatory — the same weights under full attention give CLS cosine 0.945
against the reference. `use_rope = True` selects the tiled 3-D RoPE of `vjepa2`
(`rotate_queries_or_keys` uses Meta's `repeat(1, 1, 1, 2)` expansion verbatim). `token_drop_rate` is a
training-time regulariser that `eval()` disables, so it is not converted; the converter refuses
`use_rope=False` and `qkv_bias=False` rather than guessing.

There is no `preprocessor_config.json`, so `jepa.pre.*` comes from the model card: ImageNet mean/std,
short side → 224, centre crop 224, **bicubic**, rescale 1/255. `general.license = cc-by-nc-4.0`.
Full inference graph: the `levjepa` section of `docs/gguf-schema.md` and `levjepa_encoder_forward` in
`selftest.py`.
