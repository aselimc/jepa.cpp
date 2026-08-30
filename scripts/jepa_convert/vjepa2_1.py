#!/usr/bin/env python3
"""Meta V-JEPA 2.1 torch.hub checkpoint -> jepa.cpp GGUF converter (standalone).

Input: a ``vjepa2_1_*.pt`` pickle from https://dl.fbaipublicfiles.com/vjepa2/ with top-level keys
``encoder`` (student), ``ema_encoder`` (EMA/target encoder = the one torch.hub loads for the
distilled ViT-B/ViT-L), ``predictor``, ``opt``, ``scaler``, ``epoch``, ``loss``, ``batch_size``,
``world_size``, ``lr``.  Parameter names carry a ``module.backbone.`` prefix which is stripped.

Model code: app/vjepa_2_1/models/{vision_transformer,predictor}.py and src/hub/backbones.py of
https://github.com/facebookresearch/vjepa2 (factory ``vjepa2_1_vit_base_384``).

Usage:
    python scripts/jepa_convert/vjepa2_1.py --src models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \
        --out models/gguf/vjepa2_1-vitb-384-f16.gguf --ftype f16
    python scripts/jepa_convert/vjepa2_1.py --info models/gguf/vjepa2_1-vitb-384-f16.gguf

Only numpy, torch (to unpickle) and the ``gguf`` python package are imported.
``convert(src, out, ftype)`` is exposed at module level for a package-level dispatcher.
See VJEPA_NOTES.md for the key map, the dropped keys and the RoPE convention.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

import numpy as np
import torch

import gguf
from gguf import GGMLQuantizationType

SCHEMA_VERSION = 1
FTYPES = {"f32": 0, "f16": 1, "q8_0": 7}
QUANTIZABLE = re.compile(
    r"(\.attn_(qkv|q|k|v|out)\.weight"
    r"|\.ffn_(up|down)\.weight"
    r"|^pred\.proj(_context)?\.weight"
    r"|^head\.cls\.weight"
    r"|^head\.xattn\.(q|k|v)\.weight)$"
)

# ---- constants of the V-JEPA 2.1 release (src/hub/backbones.py::_make_vjepa2_1_model) ----
HUB_IMG_SIZE = 384
HUB_PATCH = 16
HUB_TUBELET = 2
HUB_NUM_FRAMES = 64
HUB_PRED_HEADS = 12
HUB_TEACHER_DIM = 1664
ROPE_THETA = 10000.0
LN_EPS = 1e-6
# app/vjepa_2_1/models/utils/modules.py::RoPEAttention: `pretrained_grid_size = 256 // patch`
# (patch 16 -> 16, patch 14 -> 18); spatial RoPE positions are rescaled to [0, ref_grid-1].
ROPE_REF_GRID = {16: 16, 14: 18}
# app/vjepa_2_1/models/vision_transformer.py: depth -> hierarchical (per-layer-norm) layers
ENC_HIER_LAYERS = {12: [2, 5, 8, 11], 24: [5, 11, 17, 23], 40: [9, 19, 29, 39], 48: [11, 23, 37, 47]}
# app/vjepa_2_1/models/predictor.py: depth -> all_hierarchical_layers (only its length matters)
PRED_HIER_LAYERS = {4: [0, 1, 2, 3], 8: [1, 3, 5, 7], 12: [2, 5, 8, 11], 20: [4, 9, 14, 19],
                    24: [4, 11, 17, 23], 40: [9, 19, 29, 39]}
# every V-JEPA 2/2.1 ViT uses head_dim 64 (vit_base 768/12, vit_large 1024/16, vit_giant_xformers
# 1408/22, vit_gigantic_xformers 1664/26)
HEAD_DIM = 64
ARCH_BY_DIM = {768: "vit_base", 1024: "vit_large", 1408: "vit_giant_xformers", 1664: "vit_gigantic_xformers"}

# keys of the pickle that are intentionally NOT converted (training state / student encoder)
DROPPED_TOPLEVEL = ("encoder", "opt", "scaler", "epoch", "loss", "batch_size", "world_size", "lr")


class JepaGGUFWriter:
    """Same dtype policy as vjepa2.py (f32 / f16 for 2-D ``*.weight`` / q8_0 for QUANTIZABLE)."""

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


class StateDict:
    """numpy view of a torch state dict with prefix stripping + consumption tracking."""

    def __init__(self, sd: dict[str, torch.Tensor], label: str):
        self.label = label
        self.t: dict[str, np.ndarray] = {}
        for k, v in sd.items():
            k2 = k.replace("module.", "").replace("backbone.", "")  # src/hub/backbones.py::_clean_backbone_key
            if k2 in self.t:
                raise ValueError(f"prefix stripping collides on {k2}")
            self.t[k2] = v.detach().to(torch.float32).cpu().numpy()
        self.used: set[str] = set()

    def has(self, k: str) -> bool:
        return k in self.t

    def get(self, k: str) -> np.ndarray:
        if k not in self.t:
            raise KeyError(f"{k} not in {self.label}")
        self.used.add(k)
        return self.t[k]

    def count(self, pattern: str) -> int:
        rx = re.compile(pattern)
        ids = {int(m.group(1)) for k in self.t for m in [rx.match(k)] if m}
        return (max(ids) + 1) if ids else 0

    def unused(self) -> list[str]:
        return sorted(set(self.t) - self.used)


def _write_block(w: JepaGGUFWriter, sd: StateDict, dst: str, src: str) -> None:
    """Meta ``Block``: norm1, attn.qkv (already fused, rows [q;k;v]), attn.proj, norm2, mlp.fc1/fc2."""
    w.tensor(f"{dst}.ln1.weight", sd.get(f"{src}.norm1.weight"))
    w.tensor(f"{dst}.ln1.bias", sd.get(f"{src}.norm1.bias"))
    w.tensor(f"{dst}.attn_qkv.weight", sd.get(f"{src}.attn.qkv.weight"))
    w.tensor(f"{dst}.attn_qkv.bias", sd.get(f"{src}.attn.qkv.bias"))
    w.tensor(f"{dst}.attn_out.weight", sd.get(f"{src}.attn.proj.weight"))
    w.tensor(f"{dst}.attn_out.bias", sd.get(f"{src}.attn.proj.bias"))
    w.tensor(f"{dst}.ln2.weight", sd.get(f"{src}.norm2.weight"))
    w.tensor(f"{dst}.ln2.bias", sd.get(f"{src}.norm2.bias"))
    w.tensor(f"{dst}.ffn_up.weight", sd.get(f"{src}.mlp.fc1.weight"))
    w.tensor(f"{dst}.ffn_up.bias", sd.get(f"{src}.mlp.fc1.bias"))
    w.tensor(f"{dst}.ffn_down.weight", sd.get(f"{src}.mlp.fc2.weight"))
    w.tensor(f"{dst}.ffn_down.bias", sd.get(f"{src}.mlp.fc2.bias"))


def convert(src: str, out: str, ftype: str = "f16", name: str | None = None, source_url: str | None = None,
            encoder_key: str = "ema_encoder", img_size: int = HUB_IMG_SIZE, n_frames: int = HUB_NUM_FRAMES,
            n_head: int | None = None, quiet: bool = False) -> dict[str, Any]:
    """Convert V-JEPA 2.1 pickle `src` to GGUF `out`.  Returns a summary dict."""
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr))
    src = os.path.abspath(src)
    ck = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or encoder_key not in ck or "predictor" not in ck:
        raise ValueError(f"{src}: expected a dict with {encoder_key!r} and 'predictor' keys, got {type(ck)}"
                         f" {list(ck)[:12] if isinstance(ck, dict) else ''}")
    enc = StateDict(ck[encoder_key], encoder_key)
    prd = StateDict(ck["predictor"], "predictor")
    toplevel = sorted(ck.keys())
    del ck

    # ---------------- derive hparams from tensor shapes ----------------
    pw = enc.get("patch_embed.proj.weight")  # [D, C, T, P, P]
    D, C, T, P, P2 = pw.shape
    if P != P2:
        raise ValueError(f"non-square patch {pw.shape}")
    L = enc.count(r"blocks\.(\d+)\.")
    ffn = enc.get("blocks.0.mlp.fc1.weight").shape[0]
    if n_head is None:
        if D % HEAD_DIM:
            raise ValueError(f"embed_dim {D} not divisible by head_dim {HEAD_DIM}; pass --n-head")
        n_head = D // HEAD_DIM
    arch = ARCH_BY_DIM.get(D, f"vit_{D}")
    hier = ENC_HIER_LAYERS.get(L)
    if hier is None:
        raise ValueError(f"no hierarchical-layer table for depth {L}")
    n_norms = enc.count(r"norms_block\.(\d+)\.")
    if n_norms != len(hier):
        raise ValueError(f"{n_norms} norms_block entries but {len(hier)} hierarchical layers")
    has_img_pe = enc.has("patch_embed_img.proj.weight")
    has_mod = enc.has("img_mod_embed")
    if has_img_pe:
        pwi = enc.get("patch_embed_img.proj.weight")
        if pwi.shape != (D, C, 1, P, P):
            raise ValueError(f"patch_embed_img shape {pwi.shape}")
    if P not in ROPE_REF_GRID:
        raise ValueError(f"no RoPE reference grid known for patch size {P}")

    Lp = prd.count(r"predictor_blocks\.(\d+)\.")
    Dp = prd.get("predictor_norm.weight").shape[0]
    ffn_p = prd.get("predictor_blocks.0.mlp.fc1.weight").shape[0]
    n_mask = prd.count(r"mask_tokens\.(\d+)$")
    Hp = HUB_PRED_HEADS
    if Dp % Hp:
        raise ValueError(f"predictor dim {Dp} not divisible by {Hp} heads")
    # predictor_embed is a single Linear when n_output_distillation == 1, else a 2-layer MLP over the
    # concatenation of the hierarchical features
    pred_embed_mlp = prd.has("predictor_embed.0.weight")
    n_hier_in = (prd.get("predictor_embed.0.weight").shape[1] // D) if pred_embed_mlp else 1
    out_dim_total = prd.get("predictor_proj.weight").shape[0]
    if out_dim_total % n_hier_in:
        raise ValueError(f"predictor_proj out dim {out_dim_total} not divisible by n_hier_in={n_hier_in}")
    out_dim = out_dim_total // n_hier_in
    has_ctx_proj = prd.has("predictor_proj_context.weight")
    pred_has_mod = prd.has("img_mod_embed")

    model_name = name or os.path.splitext(os.path.basename(src))[0]
    source_url = source_url or f"https://dl.fbaipublicfiles.com/vjepa2/{os.path.basename(src)}"

    log(f"[vjepa2_1] {arch} from '{encoder_key}': D={D} L={L} H={n_head} ffn={ffn} patch={P} tubelet={T}"
        f" img={img_size} frames={n_frames} hier={hier} img_pe={has_img_pe} mod_embed={has_mod}"
        f" | pred D={Dp} L={Lp} H={Hp} ffn={ffn_p} mask_tokens={n_mask} hier_in={n_hier_in} out={out_dim}"
        f" ctx_proj={has_ctx_proj}")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    w = JepaGGUFWriter(out, ftype)

    # ---------------- general ----------------
    w.str("general.name", model_name)
    w.str("general.license", "mit")
    w.str("general.source_url", source_url)
    w.str("general.description",
          f"V-JEPA 2.1 {arch} ({encoder_key} + predictor) converted from the torch.hub pickle by "
          f"scripts/jepa_convert/vjepa2_1.py")
    w.u32("jepa.schema_version", SCHEMA_VERSION)
    w.str("jepa.family", "vjepa2_1")
    w.str("jepa.modality", "image+video" if has_img_pe else "video")

    # ---------------- encoder ----------------
    w.u32("jepa.enc.embed_dim", D)
    w.u32("jepa.enc.n_layer", L)
    w.u32("jepa.enc.n_head", n_head)
    w.u32("jepa.enc.ffn_dim", ffn)
    w.u32("jepa.enc.patch_size", P)
    w.u32("jepa.enc.tubelet_size", T)
    w.u32("jepa.enc.img_size", img_size)
    w.u32("jepa.enc.n_frames", n_frames)
    w.u32("jepa.enc.in_chans", C)
    w.f32("jepa.enc.ln_eps", LN_EPS)
    w.str("jepa.enc.act", "gelu_erf")
    w.str("jepa.enc.pos_type", "rope3d")
    w.f32("jepa.enc.rope_theta", ROPE_THETA)
    w.bool("jepa.enc.rope_interpolate", True)
    w.u32("jepa.enc.rope_ref_grid", ROPE_REF_GRID[P])
    # 2.1 uses repeat_interleave -> proper pairwise rotation ("interleaved"); see VJEPA_NOTES.md
    w.str("jepa.enc.rope_freq_layout", "interleaved")
    w.bool("jepa.enc.cls_token", False)
    w.u32("jepa.enc.n_registers", 0)
    w.bool("jepa.enc.qkv_fused", True)
    w.bool("jepa.enc.modality_embed", has_mod)
    w.bool("jepa.enc.image_patch_embed", has_img_pe)
    w.arr("jepa.enc.hier_layers", hier)
    w.bool("jepa.enc.layer_scale", False)

    # ---------------- predictor ----------------
    w.str("jepa.pred.kind", "masked")
    w.u32("jepa.pred.embed_dim", Dp)
    w.u32("jepa.pred.n_layer", Lp)
    w.u32("jepa.pred.n_head", Hp)
    w.u32("jepa.pred.ffn_dim", ffn_p)
    w.u32("jepa.pred.n_mask_tokens", n_mask)
    w.u32("jepa.pred.out_dim", out_dim)
    w.u32("jepa.pred.n_hier_in", n_hier_in)
    w.bool("jepa.pred.modality_embed", pred_has_mod)
    w.bool("jepa.pred.context_proj", has_ctx_proj)
    # NOTE: src/hub/backbones.py passes interpolate_rope=True to the ENCODER kwargs only; the
    # predictor keeps the default (False) and decodes mask ids on the fixed img_size//patch grid.
    w.bool("jepa.pred.rope_interpolate", False)
    w.u32("jepa.pred.grid_size", img_size // P)
    w.str("jepa.pred.rope_freq_layout", "interleaved")

    w.str("jepa.head.kind", "none")

    # ---------------- preprocessing (Meta eval transform, evals/*/eval.py) ----------------
    # video : Resize(short side = int(img*256/224), bilinear) -> CenterCrop(img) -> ImageNet norm
    # image : torchvision Resize(int(img*256/224)) (bilinear) -> CenterCrop(img) -> ImageNet norm
    w.arr("jepa.pre.mean", [0.485, 0.456, 0.406])
    w.arr("jepa.pre.std", [0.229, 0.224, 0.225])
    w.u32("jepa.pre.resize_short", int(img_size * 256 / 224))
    w.u32("jepa.pre.crop", img_size)
    w.str("jepa.pre.resample", "bilinear")
    w.f32("jepa.pre.rescale", 1.0 / 255.0)

    # ============================ encoder tensors ============================
    w.tensor("enc.patch_embed.weight", pw.reshape(D, C * T * P * P))  # C-T-H-W flatten order
    w.tensor("enc.patch_embed.bias", enc.get("patch_embed.proj.bias"))
    if has_img_pe:
        w.tensor("enc.patch_embed_img.weight", pwi.reshape(D, C * 1 * P * P))
        w.tensor("enc.patch_embed_img.bias", enc.get("patch_embed_img.proj.bias"))
    if has_mod:
        w.tensor("enc.mod_embed_img", enc.get("img_mod_embed").reshape(D))
        w.tensor("enc.mod_embed_video", enc.get("video_mod_embed").reshape(D))
    for i in range(L):
        _write_block(w, enc, f"enc.blk.{i}", f"blocks.{i}")
    for k in range(len(hier)):
        w.tensor(f"enc.hier_norm.{k}.weight", enc.get(f"norms_block.{k}.weight"))
        w.tensor(f"enc.hier_norm.{k}.bias", enc.get(f"norms_block.{k}.bias"))
    # inference output norm = norms_block[-1] (VisionTransformer.forward, non-training path)
    w.tensor("enc.norm.weight", enc.get(f"norms_block.{len(hier) - 1}.weight"))
    w.tensor("enc.norm.bias", enc.get(f"norms_block.{len(hier) - 1}.bias"))

    # ============================ predictor tensors ============================
    if pred_embed_mlp:
        w.tensor("pred.embed.0.weight", prd.get("predictor_embed.0.weight"))
        w.tensor("pred.embed.0.bias", prd.get("predictor_embed.0.bias"))
        w.tensor("pred.embed.2.weight", prd.get("predictor_embed.2.weight"))
        w.tensor("pred.embed.2.bias", prd.get("predictor_embed.2.bias"))
    else:
        w.tensor("pred.embed.weight", prd.get("predictor_embed.weight"))
        w.tensor("pred.embed.bias", prd.get("predictor_embed.bias"))
    mts = [prd.get(f"mask_tokens.{i}").reshape(Dp) for i in range(n_mask)]
    w.tensor("pred.mask_tokens", np.stack(mts, axis=0))  # [n_mask, Dp]
    if pred_has_mod:
        w.tensor("pred.mod_embed_img", prd.get("img_mod_embed").reshape(Dp))
        w.tensor("pred.mod_embed_video", prd.get("video_mod_embed").reshape(Dp))
    for i in range(Lp):
        _write_block(w, prd, f"pred.blk.{i}", f"predictor_blocks.{i}")
    w.tensor("pred.norm.weight", prd.get("predictor_norm.weight"))
    w.tensor("pred.norm.bias", prd.get("predictor_norm.bias"))
    w.tensor("pred.proj.weight", prd.get("predictor_proj.weight"))
    w.tensor("pred.proj.bias", prd.get("predictor_proj.bias"))
    if has_ctx_proj:
        w.tensor("pred.proj_context.weight", prd.get("predictor_proj_context.weight"))
        w.tensor("pred.proj_context.bias", prd.get("predictor_proj_context.bias"))

    for sd in (enc, prd):
        un = sd.unused()
        if un:
            raise RuntimeError(f"{len(un)} tensors of '{sd.label}' were not mapped: {un[:10]} ...")

    w.close()
    dropped = [k for k in toplevel if k not in (encoder_key, "predictor")]
    summary = dict(out=os.path.abspath(out), ftype=ftype, arch=arch, name=model_name, encoder_key=encoder_key,
                   n_tensors=len(w.tensor_names), dtype_counts=w.dtype_counts, tensor_bytes=w.n_bytes,
                   n_source_tensors=len(enc.t) + len(prd.t), dropped_toplevel=dropped)
    log(f"[vjepa2_1] wrote {out}: {summary['n_tensors']} tensors {w.dtype_counts}, "
        f"{w.n_bytes / 2**20:.1f} MiB of tensor data; dropped top-level keys: {dropped}")
    return summary


def describe(path: str, show_tensors: bool = False) -> dict[str, Any]:
    r = gguf.GGUFReader(path)
    kv: dict[str, Any] = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        v = f.contents()
        if isinstance(v, list) and len(v) > 8:
            v = v[:8] + [f"... ({len(v)} items)"]
        kv[k] = v
    counts: dict[str, int] = {}
    for t in r.tensors:
        counts[t.tensor_type.name] = counts.get(t.tensor_type.name, 0) + 1
    for k, v in kv.items():
        print(f"{k:36s} = {v}")
    print(f"tensors: {len(r.tensors)}  {counts}")
    if show_tensors:
        for t in r.tensors:
            print(f"  {t.name:40s} {[int(x) for x in t.shape]} {t.tensor_type.name}")
    return dict(path=path, n_kv=len(kv), n_tensors=len(r.tensors), dtype_counts=counts, kv=kv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="V-JEPA 2.1 .pt checkpoint")
    ap.add_argument("--out", help="output .gguf path")
    ap.add_argument("--ftype", default="f16", choices=sorted(FTYPES))
    ap.add_argument("--name", default=None, help="general.name override (default: checkpoint basename)")
    ap.add_argument("--source-url", default=None)
    ap.add_argument("--encoder-key", default="ema_encoder", choices=["ema_encoder", "encoder", "target_encoder"],
                    help="which encoder state dict to export (hub uses ema_encoder for the distilled B/L models)")
    ap.add_argument("--img-size", type=int, default=HUB_IMG_SIZE, help="training crop (hub: 384)")
    ap.add_argument("--n-frames", type=int, default=HUB_NUM_FRAMES, help="training clip length (hub: 64)")
    ap.add_argument("--n-head", type=int, default=None, help="encoder heads (default: embed_dim / 64)")
    ap.add_argument("--info", metavar="GGUF", help="print metadata + tensor summary of an existing GGUF and exit")
    ap.add_argument("--tensors", action="store_true", help="with --info: also list every tensor")
    args = ap.parse_args(argv)
    if args.info:
        describe(args.info, show_tensors=args.tensors)
        return 0
    if not args.src or not args.out:
        ap.error("--src and --out are required")
    convert(args.src, args.out, args.ftype, name=args.name, source_url=args.source_url,
            encoder_key=args.encoder_key, img_size=args.img_size, n_frames=args.n_frames, n_head=args.n_head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
