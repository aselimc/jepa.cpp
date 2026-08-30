# jepa.cpp — architecture brief

Inference engine only. No training code, ever. C++17, ggml (submodule at `ggml/`), no other runtime deps.
Scratch files go in `tmp/` (git-ignored). Weights go in `models/` (git-ignored). Python is used only for
conversion and reference dumps (`.venv/`, created with `uv`).

## Layout

```
include/jepa.h        public C API (opaque handles, plain structs)
src/jepa.cpp          model struct, GGUF load, graph build, run (encoder / predictor / head)
src/jepa-gguf.cpp     GGUF metadata + tensor lookup helpers, hparam parsing
src/preprocess.cpp    image/video → normalized float tensors (stb_image; bilinear/bicubic resize matching PIL/torchvision)
src/rope3d.cpp        V-JEPA 3D RoPE: cos/sin table generation + graph application
tools/jepa-info       print GGUF hparams/tensors
tools/jepa-embed      image/video → features (.npy / text)
tools/jepa-classify   video → top-k labels (attentive-pool head)
tools/jepa-quantize   f32/f16 GGUF → q8_0 / q4_k ...
tools/jepa-bench      timing
tests/test-parity     load fixtures/ref/*.npz, run model, report cosine / max-abs / top-k agreement, exit non-zero on regression
scripts/convert.py    HF safetensors / torch.hub .pt → GGUF (docs/gguf-schema.md)
scripts/dump_reference.py   PyTorch golden outputs for tests/fixtures/media/* → tests/fixtures/ref/*.npz
scripts/compare.py    optional python-side comparison of .npy outputs
```

## The shared graph

```
tokens = patch_embed(rearranged pixels) [+ pos_embed] [+ cls/reg] [+ modality vec]
for each block: x += attn(ln1(x)) ; x += ffn(ln2(x))         # pre-LN, GELU(erf), qkv bias, optional layer-scale
x = norm(x)
```
Attention always goes through `ggml_flash_attn_ext` (F16 K/V) — sequences reach 18k tokens.
Patch "conv" is a host-side rearrangement into `[N, C*T*P*P]` followed by one `ggml_mul_mat`.

## Family deltas

| family | tokenizer | positions | extras |
|---|---|---|---|
| `ijepa` | 2D patch 14/16 | `sincos2d` table (converter bakes it) | no CLS; features = mean of patch tokens |
| `vjepa` | 2×16×16 tubelet | `sincos3d` table | |
| `vjepa2` | 2×16×16 | `rope3d` (custom, see below) | predictor 384-d/12L/10 mask tokens; attentive pooler head |
| `vjepa2_1` | 2×16×16 video **and** 1×16×16 image embed | `rope3d` + `interpolate_rope` | modality vectors; per-hier-layer norms (inference: last one); 8 mask tokens |
| `hfvit` | 2D patch | `learned` (+CLS, optional registers) | DINOv2-style; optional layer-scale; used by community LeJEPA + LeWM encoder |
| `lewm` | `hfvit` encoder (ViT-tiny/14) | | predictor 192-d/6L over 3 frames + action embed (10→192); BatchNorm MLPs folded at conversion |

### V-JEPA 2 3D RoPE (must match HF `VJEPA2RopeAttention` exactly)
- `d = 2 * ((head_dim // 3) // 2)` per axis (20/20/20 for head_dim 64; remaining 4 dims untouched).
- token i → `t = i // (gh*gw)`, `h = (i % (gh*gw)) // gw`, `w = i % gw`.
- per axis, frequencies `omega_k = 1 / 10000**(2k/d)`, k in [0, d/2).
- rotation on **interleaved pairs** `(x[2k], x[2k+1])` : `x*cos + rotate90(x)*sin` where `rotate90(y1,y2) = (-y2, y1)`.
- **cos/sin table layout differs per family** (`jepa.enc.rope_freq_layout`): V-JEPA 2 (HF + Meta) *tiles* the
  d/2 frequencies, `table[j] = f(pos*omega_{j mod d/2})`, so the two halves of a pair see different angles
  (Meta's "subtle bug", kept for checkpoint compatibility); V-JEPA 2.1 uses `repeat_interleave`,
  `table[2k] = table[2k+1] = f(pos*omega_k)` (true rotation). Using the wrong one gives cosine 0.63 / 0.91
  against the reference. Full derivation + code excerpts: `scripts/jepa_convert/VJEPA_NOTES.md` §4;
  numpy reference: `scripts/jepa_convert/vjepa2_numpy_ref.py`.
- V-JEPA 2.1 encoder additionally rescales `h, w` by `(rope_ref_grid-1)/(grid-1)` (`interpolate_rope`, ref grid 16);
  its predictor does not. Predictor head_dim is 32 → d = 10 per axis.
- applied to q and k (not v), after the qkv projection, before scaling.
- ggml's `ggml_rope_multi` uses half-split rotation and one theta scale — do **not** use it. Precompute cos/sin tables `[N, head_dim]` on the host (`rope3d.cpp`) and apply with `ggml_mul` + `ggml_add` on views, or with a small custom op.

## Parity protocol (every phase ends with this)
1. `scripts/dump_reference.py --model <name>` writes `tests/fixtures/ref/<name>.npz` containing, per fixture: preprocessed input tensor, `last_hidden_state`, pooled feature, and (if a head exists) logits.
2. `test-parity <model.gguf> <ref.npz>` feeds the **same preprocessed tensor** (bypassing our preprocessor) and reports per-sample cosine similarity, max abs error; then runs our own preprocessor and reports the same, plus top-1/top-5 agreement for heads.
3. Pass thresholds: F32 cosine ≥ 0.9999, max-abs ≤ 1e-3 relative; Q8_0 cosine ≥ 0.999; top-1 agreement 100 % on fixtures for F32, ≥ 90 % for Q8_0.
4. Numbers get written to `docs/parity.md` (model, dtype, cosine, max-abs, top-k, time per sample on this box).
