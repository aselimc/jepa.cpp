#!/usr/bin/env python3
"""V-JEPA 2-AC (action-conditioned world model) -> jepa.cpp GGUF (standalone).

Source: the ``vjepa2_ac_vit_giant`` torch.hub entry of https://github.com/facebookresearch/vjepa2,
i.e. the single checkpoint ``https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`` (11.8 GB).
It holds ``{"encoder", "predictor", "target_encoder", "opt", "scaler", ...}``; the two we need are

  * ``encoder``   - the ViT-g/16 the AC predictor was trained against.  **It is not the HF
                    facebook/vjepa2-vitg-fpc64-256 release**: the two agree only to cosine ~0.998
                    per tensor (measured, see docs/parity.md).  ``encoder`` and ``target_encoder``
                    are bit-identical in the checkpoint, i.e. the encoder was frozen during AC
                    training, and ``src/hub/backbones.py::_make_vjepa2_ac_model`` loads
                    ``state_dict["encoder"]`` - so that is what this converter ships.
  * ``predictor`` - ``src/models/ac_predictor.py::VisionTransformerPredictorAC``, 24 blocks of
                    1024 dims / 16 heads, ffn 4096, GELU(erf), LN eps 1e-6, fused qkv with bias.

Graph (``jepa.pred.kind = "ac"``, docs/gguf-schema.md "vjepa2_ac"; the executable spec is
``scripts/jepa_convert/selftest.py::ac_predictor_forward``)::

    x   = pred.embed(context_latents)                 # [T*HW, 1024]
    a_t = pred.action_embed(action_t)                 # [T, 1024]
    s_t = pred.state_embed(state_t)                   # [T, 1024]
    seq = concat per frame: [a_t, s_t, x_t0 .. x_t(HW-1)]        # T * (2 + HW) rows
    24 pre-LN blocks, 3-D RoPE (tiled) + an action-block-causal attention mask
    seq = seq without the 2 conditioning rows per frame            # [T*HW, 1024]
    out = pred.proj(pred.norm(seq))                                # [T*HW, 1408]

Usage:
    python scripts/jepa_convert/vjepa2_ac.py --src models/vjepa2_ac/vjepa2-ac-vitg.pt \\
        --out models/gguf/vjepa2-ac-vitg-f16.gguf --ftype f16
    python scripts/jepa_convert/vjepa2_ac.py --info models/gguf/vjepa2-ac-vitg-f16.gguf

Only numpy, torch (to read the pickle) and the ``gguf`` package are imported.  ``convert(src, out,
ftype)`` is exposed at module level so scripts/convert.py can dispatch to it.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

import numpy as np

import gguf
from gguf import GGMLQuantizationType

SCHEMA_VERSION = 1
FTYPES = {"f32": 0, "f16": 1, "q8_0": 7}

# Only these weights may be quantized (docs/gguf-schema.md "Quantization rules").
QUANTIZABLE = re.compile(
    r"(\.attn_(qkv|q|k|v|out)\.weight"
    r"|\.ffn_(up|down)\.weight"
    r"|^pred\.proj\.weight"
    r"|^pred\.embed\.weight"
    r"|^head\.cls\.weight)$"
)

# ---- the architecture, as `vit_ac_predictor` builds it -------------------------------------
# src/hub/backbones.py::_make_vjepa2_ac_model passes only img_size / patch_size / num_frames /
# tubelet_size / embed_dim, so every other predictor hparam is a VisionTransformerPredictorAC
# default (src/models/ac_predictor.py:20-47) and `vit_ac_predictor` (ibid. 193-200).
AC_DEFAULTS = dict(
    img_size=256,          # backbones.py:201 (vjepa2_ac_vit_giant -> img_size=256)
    patch_size=16,         # backbones.py:40
    tubelet_size=2,        # backbones.py:41
    num_frames=64,         # backbones.py:42
    predictor_embed_dim=1024,  # ac_predictor.py:27
    depth=24,              # ac_predictor.py:28
    num_heads=16,          # ac_predictor.py:29
    mlp_ratio=4.0,         # vit_ac_predictor(), ac_predictor.py:195
    action_embed_dim=7,    # ac_predictor.py:44
    is_frame_causal=True,  # ac_predictor.py:41
    use_extrinsics=False,  # ac_predictor.py:45
    ln_eps=1e-6,           # vit_ac_predictor(): norm_layer=partial(nn.LayerNorm, eps=1e-6)
)
# The world-model loop of notebooks/utils/world_model_wrapper.py normalises the encoder latents and
# every predicted frame with a NON-AFFINE LayerNorm before they re-enter the predictor
# (`F.layer_norm(h, (h.size(-1),))`, torch default eps 1e-5).  It is part of the released world
# model, not of the predictor module, so the file records it for the runtime.
NORMALIZE_REPS = True
NORMALIZE_REPS_EPS = 1e-5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SOURCE_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"


class JepaGGUFWriter:
    """Same dtype policy as scripts/jepa_convert/vjepa2.py (docs/gguf-schema.md)."""

    def __init__(self, path: str, ftype: str):
        if ftype not in FTYPES:
            raise ValueError(f"unknown ftype {ftype!r}; choose from {sorted(FTYPES)}")
        self.path = path
        self.ftype = ftype
        self.gw = gguf.GGUFWriter(path, "jepa")
        self.gw.add_file_type(FTYPES[ftype])
        self.tensor_names: list[str] = []
        self.dtype_counts: dict[str, int] = {}
        self.n_bytes = 0

    def u32(self, k: str, v: int) -> None:
        self.gw.add_uint32(k, int(v))

    def f32(self, k: str, v: float) -> None:
        self.gw.add_float32(k, float(v))

    def bool(self, k: str, v: bool) -> None:
        self.gw.add_bool(k, bool(v))

    def str(self, k: str, v: str) -> None:
        self.gw.add_string(k, str(v))

    def arr(self, k: str, v: list) -> None:
        if len(v) == 0:
            raise ValueError(f"refusing to write empty array for {k}")
        self.gw.add_array(k, list(v))

    def tensor(self, name: str, arr: np.ndarray) -> None:
        if name in self.tensor_names:
            raise ValueError(f"duplicate tensor {name}")
        arr = np.ascontiguousarray(arr)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        is_matrix_weight = arr.ndim >= 2 and name.endswith(".weight")
        if self.ftype == "q8_0" and QUANTIZABLE.search(name) and arr.shape[-1] % 32 == 0:
            q = gguf.quants.quantize(arr, GGMLQuantizationType.Q8_0)
            self.gw.add_tensor(name, q, raw_dtype=GGMLQuantizationType.Q8_0)
            dt, nbytes = "Q8_0", q.nbytes
        elif self.ftype in ("f16", "q8_0") and is_matrix_weight:
            arr = arr.astype(np.float16)
            self.gw.add_tensor(name, arr)
            dt, nbytes = "F16", arr.nbytes
        else:
            self.gw.add_tensor(name, arr)
            dt, nbytes = "F32", arr.nbytes
        self.tensor_names.append(name)
        self.dtype_counts[dt] = self.dtype_counts.get(dt, 0) + 1
        self.n_bytes += nbytes

    def close(self) -> None:
        self.gw.write_header_to_file()
        self.gw.write_kv_data_to_file()
        self.gw.write_tensors_to_file(progress=False)
        self.gw.close()


def _np(t) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def convert(src: str, out: str, ftype: str = "f16", name: str | None = None,
            source_url: str | None = None, quiet: bool = False) -> dict[str, Any]:
    """Convert the vjepa2-ac-vitg.pt checkpoint `src` to the GGUF bundle `out`."""
    import torch

    log = (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr))
    src = os.path.abspath(str(src))
    model_name = name or "vjepa2-ac-vitg"
    source_url = source_url or SOURCE_URL

    sd = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    for k in ("encoder", "predictor"):
        if k not in sd:
            raise ValueError(f"{src} has no '{k}' state dict (top level: {sorted(sd)})")

    def clean(state):  # backbones.py::_clean_backbone_key
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}

    enc = clean(sd["encoder"])
    pred = clean(sd["predictor"])

    # ---- shapes read back off the checkpoint (never assumed) --------------------------------
    pw = _np(enc["patch_embed.proj.weight"])            # [D, C, tubelet, P, P]
    D, C, TUB, P, P2 = pw.shape
    if P != P2:
        raise ValueError(f"non-square patch {P}x{P2}")
    L = 1 + max(int(m.group(1)) for k in enc if (m := re.match(r"blocks\.(\d+)\.", k)))
    ffn = _np(enc["blocks.0.mlp.fc1.weight"]).shape[0]
    qkv0 = _np(enc["blocks.0.attn.qkv.weight"]).shape[0]
    if qkv0 != 3 * D:
        raise ValueError(f"encoder qkv rows {qkv0} != 3*{D}")
    # ViT-g: 1408 / 22 = 64.  vision_transformer.py's vit_giant_xformers uses num_heads=22.
    n_head = {1408: 22, 1280: 16, 1024: 16, 768: 12}.get(D)
    if n_head is None:
        raise ValueError(f"unknown encoder width {D}: cannot infer num_heads, pass --n-head")

    Dp = _np(pred["predictor_embed.weight"]).shape[0]
    if _np(pred["predictor_embed.weight"]).shape[1] != D:
        raise ValueError("predictor_embed input width != encoder width")
    Lp = 1 + max(int(m.group(1)) for k in pred if (m := re.match(r"predictor_blocks\.(\d+)\.", k)))
    ffn_p = _np(pred["predictor_blocks.0.mlp.fc1.weight"]).shape[0]
    n_head_p = AC_DEFAULTS["num_heads"]
    action_dim = _np(pred["action_encoder.weight"]).shape[1]
    state_dim = _np(pred["state_encoder.weight"]).shape[1]
    out_dim = _np(pred["predictor_proj.weight"]).shape[0]

    crop = AC_DEFAULTS["img_size"]
    grid = crop // P                       # predictor grid_height/grid_width (ac_predictor.py:68-69)
    n_frames = AC_DEFAULTS["num_frames"]
    grid_depth = n_frames // TUB           # ac_predictor.py:111, the mask's frame axis (32)

    log(f"[vjepa2_ac] enc D={D} L={L} H={n_head} ffn={ffn} patch={P} tubelet={TUB} crop={crop} frames={n_frames}"
        f" | pred D={Dp} L={Lp} H={n_head_p} ffn={ffn_p} action={action_dim} state={state_dim} out={out_dim}"
        f" | grid {grid}x{grid}, {grid_depth} frame slots")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    w = JepaGGUFWriter(out, ftype)

    # ---------------- general ----------------
    w.str("general.name", model_name)
    w.str("general.license", "mit")   # facebookresearch/vjepa2 LICENSE (MIT); README "License"
    w.str("general.source_url", source_url)
    w.str("general.description",
          "V-JEPA 2-AC (action-conditioned world model): the frozen ViT-g/16 encoder and the "
          "24-layer AC predictor of vjepa2-ac-vitg.pt, converted by scripts/jepa_convert/vjepa2_ac.py")
    w.u32("jepa.schema_version", SCHEMA_VERSION)
    w.str("jepa.family", "vjepa2")
    w.str("jepa.modality", "video")

    # ---------------- encoder (identical schema to the vjepa2 family) ----------------
    w.u32("jepa.enc.embed_dim", D)
    w.u32("jepa.enc.n_layer", L)
    w.u32("jepa.enc.n_head", n_head)
    w.u32("jepa.enc.ffn_dim", ffn)
    w.u32("jepa.enc.patch_size", P)
    w.u32("jepa.enc.tubelet_size", TUB)
    w.u32("jepa.enc.img_size", crop)
    w.u32("jepa.enc.n_frames", n_frames)
    w.u32("jepa.enc.in_chans", C)
    w.f32("jepa.enc.ln_eps", AC_DEFAULTS["ln_eps"])
    w.str("jepa.enc.act", "gelu_erf")
    w.str("jepa.enc.pos_type", "rope3d")
    w.f32("jepa.enc.rope_theta", 10000.0)
    w.bool("jepa.enc.rope_interpolate", False)
    w.str("jepa.enc.rope_freq_layout", "tiled")
    w.bool("jepa.enc.cls_token", False)
    w.u32("jepa.enc.n_registers", 0)
    w.bool("jepa.enc.qkv_fused", True)
    w.bool("jepa.enc.modality_embed", False)
    w.bool("jepa.enc.image_patch_embed", False)
    w.bool("jepa.enc.layer_scale", False)

    # ---------------- predictor ----------------
    w.str("jepa.pred.kind", "ac")
    w.u32("jepa.pred.embed_dim", Dp)
    w.u32("jepa.pred.n_layer", Lp)
    w.u32("jepa.pred.n_head", n_head_p)
    w.u32("jepa.pred.ffn_dim", ffn_p)
    w.u32("jepa.pred.out_dim", out_dim)
    w.u32("jepa.pred.action_dim", action_dim)
    w.u32("jepa.pred.state_dim", state_dim)
    w.u32("jepa.pred.n_cond_tokens", 3 if AC_DEFAULTS["use_extrinsics"] else 2)
    w.str("jepa.pred.cond_order", "action,state")
    w.bool("jepa.pred.frame_causal", AC_DEFAULTS["is_frame_causal"])
    w.u32("jepa.pred.n_frames", grid_depth)
    w.u32("jepa.pred.grid_size", grid)
    w.str("jepa.pred.rope_freq_layout", "tiled")
    w.bool("jepa.pred.rope_interpolate", False)
    w.f32("jepa.pred.ln_eps", AC_DEFAULTS["ln_eps"])
    w.str("jepa.pred.act", "gelu_erf")
    w.bool("jepa.pred.qkv_bias", True)
    w.bool("jepa.pred.normalize_reps", NORMALIZE_REPS)
    w.f32("jepa.pred.norm_reps_eps", NORMALIZE_REPS_EPS)

    w.str("jepa.head.kind", "none")

    # ---------------- preprocessing ----------------
    # The AC eval transform is NOT the V-JEPA 2 video-processor pipeline.  The demo builds
    # app/vjepa_droid/transforms.py::make_transforms with random_resize_scale=(1,1) and
    # random_resize_aspect_ratio=(1,1) (notebooks/energy_landscape_example.ipynb, cell 2), which turns
    # `random_resized_crop` (src/datasets/utils/video/transforms.py:510-542) into: centre-crop to the
    # largest square, then torch F.interpolate(bilinear, align_corners=False, NO antialias) to
    # crop x crop, then (x - 255*mean) / (255*std).  On the square 256x256 DROID/Franka renders the
    # model actually consumes, the crop and the resize are both the identity, which is what
    # resize_short = crop = 256 below reproduces; a non-square source would differ from Meta's
    # resampler (docs/parity.md, "V-JEPA 2-AC").
    w.arr("jepa.pre.mean", [float(x) for x in IMAGENET_MEAN])
    w.arr("jepa.pre.std", [float(x) for x in IMAGENET_STD])
    w.u32("jepa.pre.resize_short", crop)
    w.u32("jepa.pre.crop", crop)
    w.str("jepa.pre.resample", "bilinear")
    w.f32("jepa.pre.rescale", 1.0 / 255.0)

    # ============================ tensors ============================
    w.tensor("enc.patch_embed.weight", pw.reshape(D, C * TUB * P * P))
    w.tensor("enc.patch_embed.bias", _np(enc["patch_embed.proj.bias"]))
    for i in range(L):
        b = f"blocks.{i}."
        w.tensor(f"enc.blk.{i}.ln1.weight", _np(enc[b + "norm1.weight"]))
        w.tensor(f"enc.blk.{i}.ln1.bias", _np(enc[b + "norm1.bias"]))
        w.tensor(f"enc.blk.{i}.attn_qkv.weight", _np(enc[b + "attn.qkv.weight"]))
        w.tensor(f"enc.blk.{i}.attn_qkv.bias", _np(enc[b + "attn.qkv.bias"]))
        w.tensor(f"enc.blk.{i}.attn_out.weight", _np(enc[b + "attn.proj.weight"]))
        w.tensor(f"enc.blk.{i}.attn_out.bias", _np(enc[b + "attn.proj.bias"]))
        w.tensor(f"enc.blk.{i}.ln2.weight", _np(enc[b + "norm2.weight"]))
        w.tensor(f"enc.blk.{i}.ln2.bias", _np(enc[b + "norm2.bias"]))
        w.tensor(f"enc.blk.{i}.ffn_up.weight", _np(enc[b + "mlp.fc1.weight"]))
        w.tensor(f"enc.blk.{i}.ffn_up.bias", _np(enc[b + "mlp.fc1.bias"]))
        w.tensor(f"enc.blk.{i}.ffn_down.weight", _np(enc[b + "mlp.fc2.weight"]))
        w.tensor(f"enc.blk.{i}.ffn_down.bias", _np(enc[b + "mlp.fc2.bias"]))
    w.tensor("enc.norm.weight", _np(enc["norm.weight"]))
    w.tensor("enc.norm.bias", _np(enc["norm.bias"]))

    w.tensor("pred.embed.weight", _np(pred["predictor_embed.weight"]))
    w.tensor("pred.embed.bias", _np(pred["predictor_embed.bias"]))
    w.tensor("pred.action_embed.weight", _np(pred["action_encoder.weight"]))
    w.tensor("pred.action_embed.bias", _np(pred["action_encoder.bias"]))
    w.tensor("pred.state_embed.weight", _np(pred["state_encoder.weight"]))
    w.tensor("pred.state_embed.bias", _np(pred["state_encoder.bias"]))
    for i in range(Lp):
        b = f"predictor_blocks.{i}."
        w.tensor(f"pred.blk.{i}.ln1.weight", _np(pred[b + "norm1.weight"]))
        w.tensor(f"pred.blk.{i}.ln1.bias", _np(pred[b + "norm1.bias"]))
        w.tensor(f"pred.blk.{i}.attn_qkv.weight", _np(pred[b + "attn.qkv.weight"]))
        w.tensor(f"pred.blk.{i}.attn_qkv.bias", _np(pred[b + "attn.qkv.bias"]))
        w.tensor(f"pred.blk.{i}.attn_out.weight", _np(pred[b + "attn.proj.weight"]))
        w.tensor(f"pred.blk.{i}.attn_out.bias", _np(pred[b + "attn.proj.bias"]))
        w.tensor(f"pred.blk.{i}.ln2.weight", _np(pred[b + "norm2.weight"]))
        w.tensor(f"pred.blk.{i}.ln2.bias", _np(pred[b + "norm2.bias"]))
        w.tensor(f"pred.blk.{i}.ffn_up.weight", _np(pred[b + "mlp.fc1.weight"]))
        w.tensor(f"pred.blk.{i}.ffn_up.bias", _np(pred[b + "mlp.fc1.bias"]))
        w.tensor(f"pred.blk.{i}.ffn_down.weight", _np(pred[b + "mlp.fc2.weight"]))
        w.tensor(f"pred.blk.{i}.ffn_down.bias", _np(pred[b + "mlp.fc2.bias"]))
    w.tensor("pred.norm.weight", _np(pred["predictor_norm.weight"]))
    w.tensor("pred.norm.bias", _np(pred["predictor_norm.bias"]))
    w.tensor("pred.proj.weight", _np(pred["predictor_proj.weight"]))
    w.tensor("pred.proj.bias", _np(pred["predictor_proj.bias"]))

    # `extrinsics_encoder.*` is instantiated unconditionally by VisionTransformerPredictorAC
    # (ac_predictor.py:56) but only read when use_extrinsics=True, which the released hub entry
    # never sets.  Deliberately not converted; recorded so the count below stays honest.
    skipped = sorted(k for k in pred if k.startswith("extrinsics_encoder."))
    expected = set(pred) - set(skipped)
    mapped = 6 + 12 * Lp + 2 + 2      # embed/action/state (6) + blocks + pred.norm (2) + pred.proj (2)
    if len(expected) != mapped:
        raise RuntimeError(f"predictor: mapped {mapped} tensors but the checkpoint has {len(expected)} "
                           f"(skipped {skipped}); unexpected: {sorted(expected)[:8]} ...")
    if len(enc) != 2 + 12 * L + 2:
        raise RuntimeError(f"encoder: {len(enc)} tensors, expected {2 + 12 * L + 2}")

    w.close()
    summary = dict(out=os.path.abspath(out), ftype=ftype, name=model_name, n_tensors=len(w.tensor_names),
                   dtype_counts=w.dtype_counts, tensor_bytes=w.n_bytes, skipped=skipped,
                   enc=dict(embed_dim=D, n_layer=L, n_head=n_head, ffn_dim=ffn),
                   pred=dict(embed_dim=Dp, n_layer=Lp, n_head=n_head_p, ffn_dim=ffn_p,
                             action_dim=action_dim, state_dim=state_dim, out_dim=out_dim,
                             grid_size=grid, n_frames=grid_depth))
    log(f"[vjepa2_ac] wrote {out}: {summary['n_tensors']} tensors {w.dtype_counts}, "
        f"{w.n_bytes / 2**20:.1f} MiB of tensor data (skipped {len(skipped)}: {skipped})")
    return summary


def describe(path: str, show_tensors: bool = False) -> dict[str, Any]:
    from vjepa2 import describe as _d  # same reader, keep one implementation
    return _d(path, show_tensors)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="vjepa2-ac-vitg.pt (torch.hub checkpoint)")
    ap.add_argument("--out", help="output .gguf path")
    ap.add_argument("--ftype", default="f16", choices=sorted(FTYPES))
    ap.add_argument("--name", default=None, help="general.name override (default vjepa2-ac-vitg)")
    ap.add_argument("--source-url", default=None)
    ap.add_argument("--info", metavar="GGUF", help="print metadata + tensor summary and exit")
    ap.add_argument("--tensors", action="store_true")
    args = ap.parse_args(argv)
    if args.info:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        describe(args.info, show_tensors=args.tensors)
        return 0
    if not args.src or not args.out:
        ap.error("--src and --out are required")
    convert(args.src, args.out, args.ftype, name=args.name, source_url=args.source_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
