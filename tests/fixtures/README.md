# Parity fixtures

Everything the parity tests need: small media inputs (`media/`) and PyTorch golden outputs (`ref/`).
Only the eight COCO images are tracked in git; clips and reference dumps are git-ignored and (re)generated with:

```bash
scripts/download_fixtures.sh                     # 8 COCO val2017 jpgs + 6 Kinetics-mini clips -> tests/fixtures/media
scripts/dump_reference.py --model all            # golden outputs -> tests/fixtures/ref/<model>/   (~1 min on 32 cores)
scripts/compare.py OUT_DIR tests/fixtures/ref/<model>   # jepa.cpp output vs reference, non-zero exit on regression
```

`dump_reference.py` needs the project venv (`.venv`, torch CPU + transformers + av + pillow + timm + einops), the
checkpoints from `scripts/download_models.sh` under `models/`, and for `vjepa2_1-vitb-384` a clone of
`facebookresearch/vjepa2` at `tmp/vjepa2-src` (cloned automatically if absent). For `lejepa-vits16` the model directory
must also contain the `hf_src/` package of the HF repo (its `modelling_vitv2.py` imports it). Use `--root` to point at another checkout; caches go to `<root>/tmp/hf-home` and `<root>/tmp/torch-home`.
Everything runs in float32 eval mode, 32 threads (`--threads`), no autocast.

## media/

| file | content | notes |
|---|---|---|
| `coco_*.jpg` (8) | COCO val2017 images, various sizes (e.g. 640x480, 480x640) | tracked in git |
| `archery.mp4` | Kinetics-mini val, 300 frames, 480x360, 29.97 fps, h264 yuv420p | primary clip |
| `bowling.mp4` | Kinetics-mini val, 150 frames, 480x270, 15 fps | primary clip |
| `flying_kite.mp4`, `high_jump.mp4`, `marching.mp4`, `high_jump2.mp4` | more Kinetics-mini clips (10 s each) | spare; `--n-clips N` |

## ref/<model>/ layout

One directory per model with a `manifest.json` and one `.npy` per tensor per sample, named `<sample>.<tensor>.npy`.
All arrays are float32 C-order except `frames_u8` (uint8) and `top5_idx` (int64). Shapes have **no batch dim** unless the
tensor is the model input (`input`), which is stored exactly as fed to the model (batch = 1).

Manifest schema (abridged):

```json
{
 "model": "vjepa2-vitl-fpc64-256", "source": "/abs/path/models/facebook/vjepa2-vitl-fpc64-256", "hf_id": "...",
 "dtype": "float32", "framework": {"torch": "2.13.0+cpu", "transformers": "5.16.1", "threads": 32, ...},
 "hparams": {"embed_dim": 1024, "n_layer": 24, "n_head": 16, "patch_size": 16, "tubelet_size": 2, "ln_eps": 1e-6, ...},
 "preprocessing": {"description": "...exact pipeline...", "resize": {...}, "center_crop": 256, "mean": [...], "std": [...],
                   "processor": {"class": "VJEPA2VideoProcessor", "backend": "torchvision", "config": {...}}},
 "outputs": {"last_hidden_state": "what this tensor is", ...},
 "labels": ["..."],                               // classifiers only (id2label in order)
 "timing_s": {"model_load": 2.1, "forward_total": 9.8, "forward_mean": 2.4, "wall_total": 14.0},
 "samples": [
   {"name": "archery_f64", "media": "archery.mp4", "frames": 64, "frame_indices": [0, 5, ...], "n_frames_total": 300, "fps": 29.97,
    "timing_s": {"decode_s": 0.4, "preprocess_s": 0.2, "forward_s": 7.9},
    "tensors": {"input": {"file": "archery_f64.input.npy", "shape": [1, 64, 3, 256, 256], "dtype": "float32", "layout": "NTCHW"}, ...}}
 ]
}
```

`timing_s.forward_s` is the PyTorch CPU wall time of the model forward alone (no decode / preprocessing) and is the
baseline for speed comparisons in `docs/parity.md`.

## Per-model tensors

Token order is always t-major, then h, then w (images: h-major then w); `hfvit`-style models put CLS first.

| model | samples | tensors (shape) |
|---|---|---|
| `ijepa-vith14-1k` | 8 images | `input` [1,3,224,224] NCHW · `last_hidden_state` [256,1280] (after final LN, no CLS) · `pooled_mean` [1280] |
| `lejepa-vits16` | 8 images | `input` [1,3,224,224] · `last_hidden_state` [197,384] = [CLS; 196 patches] after the final LN (`ViTv2.forward_backbone`) · `cls` [384] · `cls_raw` [384] (CLS before the final LN) · `pooled_mean` [384] (mean of the 196 patch tokens) |
| `lewm-pusht` | 2 images + `seq` | per image: `input` [1,3,224,224] · `last_hidden_state` [257,192] (HF ViT, CLS first, after final LN) · `cls` [192] · `emb` [192] = projector(cls) · `action` [10] · `act_emb` [192] · `pred_next` [192] (one predictor step, T=1). `seq`: `input` [3,3,224,224] · `emb_seq`/`act_emb_seq`/`pred_seq` [3,192] · `action_seq` [3,10] (T=3 causal rollout, prediction at every step) |
| `vjepa2-vitl-fpc64-256` | archery/bowling x {64, 16} frames | `frames_u8` [T,H,W,3] uint8 · `input` [1,T,3,256,256] **NTCHW** · `last_hidden_state` [T/2·256, 1024] · `pooled_mean` [1024] · `predictor_last_hidden_state` [T/2·256, 1024] (default masks: full context, predict every token) |
| `vjepa2-vitl-fpc16-256-ssv2` | archery/bowling x 16 frames | `frames_u8` · `input` [1,16,3,256,256] · `last_hidden_state` [2048,1024] · `pooled` [1024] (attentive-pooler output = classifier input, via forward hook) · `logits` [174] · `top5_idx` [5] int64; `labels` (id2label) in the manifest |
| `vjepa2_1-vitb-384` | archery x 16 frames + 2 images | `frames_u8` · `input` [1,3,16,384,384] **NCTHW** (video) / [1,3,1,384,384] (image path: `patch_embed_img` + `img_mod_embed`) · `last_hidden_state` [4608,768] / [576,768] = `norms_block[-1](x)` (the encoder's default inference output) · `pooled_mean` [768] |

## Preprocessing actually applied (also in each manifest)

All resizing is `torchvision.transforms.v2.functional.resize` **on the uint8 CHW tensor with `antialias=True`** (rounded back
to uint8), which is what every transformers>=5 image/video processor does (`TorchvisionBackend`); verified against a
re-implementation to 2.4e-7, whereas PIL resampling differs by up to 1.8e-2 (1-2 uint8 levels). Center crop uses
`top = int((H-c)/2)`, `left = int((W-c)/2)`; short-side resize keeps the aspect ratio with the long side **floored**: `other = int(s * other / short)` (transformers `get_resize_output_image_size`; e.g. 640×426 → 256 short gives 384, not 385).

| model | pipeline |
|---|---|
| `ijepa-vith14-1k` | `ViTImageProcessor`: resize to exactly 224x224 (aspect not kept), bilinear; x/255; mean = std = 0.5 |
| `lejepa-vits16` | `BitImageProcessor`: short side -> 256, bicubic; center crop 224; x/255; ImageNet mean/std |
| `lewm-pusht` | resize to exactly 224x224 (aspect not kept), bilinear; x/255; ImageNet mean/std (upstream trains/evals on 224x224 renders, so the resize only exists to feed COCO images) |
| `vjepa2-*` | `VJEPA2VideoProcessor`: per frame, short side -> 292 = int(256*256/224), bilinear; center crop 256; x/255; ImageNet mean/std; output [B,T,3,H,W] |
| `vjepa2_1-vitb-384` | same processor with crop 384 (short side -> 438), then permuted to [B,3,T,H,W]; images are 1-frame videos |

Frame sampling for clips: all frames decoded with PyAV (`rgb24`), then `idx = round(linspace(0, T_total-1, n))`; the indices are
stored per sample so the C++ side can be fed `frames_u8` directly (there is no video decoder in jepa.cpp).

The parity test feeds the stored `input` tensor first (bypassing preprocessing) and only then runs our own preprocessor,
so a preprocessing mismatch shows up separately from a graph bug.

## LeWM: what the predictor consumes

Documented in `ref/lewm-pusht/manifest.json` (`predictor_semantics`), in short: the encoder is a stock HF `ViTModel`
ViT-Ti/14 @224 (257 tokens, learned pos-embed incl. CLS slot, **LayerNorm eps 1e-12**, GELU-erf). Only the CLS token is used:
`emb = projector(cls)` with projector = Linear(192->2048) -> BatchNorm1d (eval: running stats, eps 1e-5) -> GELU -> Linear(2048->192).
The action (10 = frameskip 5 x 2, z-scored at train time; here N(0,1) with seed 0) goes through Embedder = Conv1d(10->10, k=1)
-> Linear(10->768) -> SiLU -> Linear(768->192). The predictor adds `pos_embedding[:T]` (T<=3) and runs 6 AdaLN-zero blocks
conditioned per step on `act_emb` (SiLU -> Linear(192->1152) -> shift/scale/gate for attn and MLP; non-affine LN eps 1e-6),
each with an inner affine LayerNorm before the qkv (16 heads x 64, no qkv bias, **causal**) and before the MLP (192->2048->192,
GELU), then a final LayerNorm(192). `pred_next = pred_proj(out[:, -1])` (same MLP shape as the projector) is the predicted
*projected* embedding of the next frame. `weights.pt` loads strictly into this re-implementation.

## compare.py

```
scripts/compare.py A B [--topk 5] [--tensors t1,t2] [--skip frames_u8] [--min-cos 0.9999] [--max-rel 1e-3] [--max-abs X]
                       [--min-top1 1.0] [--min-topk F] [--json rows.json] [--quiet]
```

`A` is the candidate, `B` the reference; both `.npy` files or both directories (samples matched by name via `manifest.json`,
or by `<sample>.<tensor>.npy` file names). Per tensor it prints the per-token cosine (mean and worst), max abs error,
`rel_max` = max|a-b| / max|b|, `rel_fro` = ||a-b|| / ||b||, and for `logits` (name substring, `--logits`) the top-1 match and
top-k overlap; integer tensors (`top5_idx`) are compared as sets. Exit status 1 if any threshold is violated. The defaults
are the F32 thresholds of `docs/architecture.md` (cosine >= 0.9999, relative max-abs <= 1e-3, top-1 = 100 %); for Q8_0 use
`--min-cos 0.999 --max-rel -1 --min-top1 0`. `compare_arrays()` / `compare_dirs()` are importable for the C++ test's
Python twin.
