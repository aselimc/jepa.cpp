#!/usr/bin/env python3
"""LeVJEPA -> jepa.cpp GGUF converter.

Handles ``galilai-group/LeVJEPA-VideoMix-Large`` (``LeVJEPAModel``, custom architecture shipped
with the weights as ``modeling_levjepa.py`` / ``configuration_levjepa.py``): a ViT-L/16 video
encoder with tubelet 1, 3-D RoPE in the V-JEPA 2 *tiled* layout, a CLS token that takes no RoPE,
and **block-causal** attention.  There is no predictor and no head in the checkpoint.

Usage:
    python scripts/jepa_convert/levjepa.py --src models/galilai-group/LeVJEPA-VideoMix-Large \
        --out models/gguf/levjepa-vitl16-f32.gguf --ftype f32
    python scripts/jepa_convert/levjepa.py --info models/gguf/levjepa-vitl16-f32.gguf

Tensor names and metadata follow docs/gguf-schema.md (see its "levjepa" family note).  The dtype
policy, the safetensors reader and the block writer are shared with the V-JEPA 2 converter
(``vjepa2.py``) — same rules, same quantizable set, one implementation.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

try:                                     # as a package module (scripts/convert.py)
    from .vjepa2 import FTYPES, SCHEMA_VERSION, JepaGGUFWriter, SourceTensors, _read_json, _readme_license, describe
except ImportError:                      # run directly as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jepa_convert.vjepa2 import (FTYPES, SCHEMA_VERSION, JepaGGUFWriter, SourceTensors,
                                     _read_json, _readme_license, describe)

SOURCE_URL = "https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large"
DEFAULT_NAME = "levjepa-vitl16"

# ImageNet statistics; the model card states them explicitly (there is no preprocessor_config.json).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def convert(src: str, out: str, ftype: str = "f16", name: str | None = None,
            source_url: str | None = None, quiet: bool = False) -> dict[str, Any]:
    """Convert a LeVJEPA checkpoint directory `src` to the GGUF file `out`.  Returns a summary."""
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr))

    src = os.path.abspath(src)
    cfg = _read_json(os.path.join(src, "config.json"))
    if cfg.get("model_type") != "levjepa":
        raise ValueError(f"model_type={cfg.get('model_type')!r} is not levjepa")
    arch = (cfg.get("architectures") or ["LeVJEPAModel"])[0]
    if arch != "LeVJEPAModel":
        raise ValueError(f"unsupported architecture {arch}")
    st_path = os.path.join(src, "model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"{st_path} not found (sharded checkpoints are not supported)")
    st = SourceTensors(st_path)

    D = int(cfg["embed_dim"])
    L = int(cfg["depth"])
    H = int(cfg["num_heads"])
    ffn = int(D * float(cfg.get("mlp_ratio", 4.0)))
    P = int(cfg["patch_size"])
    tub = int(cfg.get("tubelet_size", 1))
    C = int(cfg.get("in_chans", 3))
    crop = int(cfg.get("img_size", 224))
    n_frames = int(cfg.get("num_frames", 16))
    # modeling_levjepa.LeVJEPAModel hard-codes norm_layer=partial(nn.LayerNorm, eps=1e-6).
    eps = 1e-6
    attn_mode = str(cfg.get("attn_mode", "full"))
    if attn_mode not in ("full", "block_causal"):
        raise ValueError(f"unknown attn_mode {attn_mode!r} (expected full | block_causal)")
    if not cfg.get("use_rope", True):
        raise NotImplementedError("use_rope=False (sincos3d tables) is not implemented for levjepa")
    if not cfg.get("qkv_bias", True):
        raise NotImplementedError("qkv_bias=False is not implemented for levjepa")
    if D % H:
        raise ValueError("embed_dim not divisible by num_heads")

    model_name = name or DEFAULT_NAME
    source_url = source_url or SOURCE_URL
    n_tokens = 1 + (n_frames // tub) * (crop // P) ** 2

    log(f"[levjepa] {arch}: D={D} L={L} H={H} ffn={ffn} patch={P} tubelet={tub} crop={crop} "
        f"frames={n_frames} attn={attn_mode} -> {n_tokens} tokens (CLS + patches)")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    w = JepaGGUFWriter(out, ftype)

    # ---------------- general ----------------
    w.str("general.name", model_name)
    w.str("general.license", _readme_license(src, "cc-by-nc-4.0"))
    w.str("general.source_url", source_url)
    w.str("general.description", f"LeVJEPA ({arch}) converted from HF safetensors by scripts/jepa_convert/levjepa.py")
    w.u32("jepa.schema_version", SCHEMA_VERSION)
    w.str("jepa.family", "levjepa")
    w.str("jepa.modality", "video")

    # ---------------- encoder ----------------
    w.u32("jepa.enc.embed_dim", D)
    w.u32("jepa.enc.n_layer", L)
    w.u32("jepa.enc.n_head", H)
    w.u32("jepa.enc.ffn_dim", ffn)
    w.u32("jepa.enc.patch_size", P)
    w.u32("jepa.enc.tubelet_size", tub)
    w.u32("jepa.enc.img_size", crop)
    w.u32("jepa.enc.n_frames", n_frames)
    w.u32("jepa.enc.in_chans", C)
    w.f32("jepa.enc.ln_eps", eps)
    w.str("jepa.enc.act", "gelu_erf")
    w.str("jepa.enc.pos_type", "rope3d")
    w.f32("jepa.enc.rope_theta", 10000.0)
    w.bool("jepa.enc.rope_interpolate", False)
    # rotate_queries_or_keys() expands the per-axis table with `.repeat(1, 1, 1, 2)` -- the tiled
    # V-JEPA 2 layout, kept verbatim from Meta's code ("Match V-JEPA2's pretrained-compatible
    # frequency expansion").  The interleaved 2.1 layout gives cosine 0.666 on these weights.
    w.str("jepa.enc.rope_freq_layout", "tiled")
    # The CLS token is prepended AFTER the patch embedding and is excluded from RoPE
    # (RoPEAttention.forward rotates q/k[..., 1:, :] only).
    w.bool("jepa.enc.cls_token", True)
    w.u32("jepa.enc.n_registers", 0)
    w.bool("jepa.enc.qkv_fused", True)
    w.bool("jepa.enc.modality_embed", False)
    w.bool("jepa.enc.image_patch_embed", False)
    w.bool("jepa.enc.layer_scale", False)
    # The one key this family adds: bidirectional inside a temporal slot, causal across slots,
    # CLS a read-only sink (build_block_causal_mask in modeling_levjepa.py).
    w.str("jepa.enc.attn_mode", attn_mode)

    w.str("jepa.head.kind", "none")

    # ---------------- preprocessing ----------------
    # No preprocessor_config.json in the repo; the model card gives ImageNet statistics and a
    # 224 resize/crop, and the reference notebook resizes the SHORT side with bicubic.
    w.arr("jepa.pre.mean", [float(x) for x in IMAGENET_MEAN])
    w.arr("jepa.pre.std", [float(x) for x in IMAGENET_STD])
    w.u32("jepa.pre.resize_short", crop)
    w.u32("jepa.pre.crop", crop)
    w.str("jepa.pre.resample", "bicubic")
    w.str("jepa.pre.resize_mode", "shortest_edge")
    w.f32("jepa.pre.rescale", 1.0 / 255.0)

    # ============================ tensors ============================
    enc = "encoder"
    pw = st.get(f"{enc}.patch_embed.proj.weight")            # Conv3d [D, C, tubelet, P, P]
    if pw.shape != (D, C, tub, P, P):
        raise ValueError(f"unexpected patch embed shape {pw.shape}, expected {(D, C, tub, P, P)}")
    w.tensor("enc.patch_embed.weight", pw.reshape(D, C * tub * P * P))   # C-T-H-W flatten order
    w.tensor("enc.patch_embed.bias", st.get(f"{enc}.patch_embed.proj.bias"))
    cls = st.get(f"{enc}.cls_token")                          # [1, 1, D]
    if cls.size != D:
        raise ValueError(f"unexpected cls_token shape {cls.shape}")
    w.tensor("enc.cls_token", cls.reshape(D))
    for i in range(L):
        b, s = f"enc.blk.{i}", f"{enc}.blocks.{i}"
        w.tensor(f"{b}.ln1.weight", st.get(f"{s}.norm1.weight"))
        w.tensor(f"{b}.ln1.bias", st.get(f"{s}.norm1.bias"))
        # nn.Linear(dim, 3*dim): rows are already [q; k; v], no concatenation needed.
        qkv = st.get(f"{s}.attn.qkv.weight")
        if qkv.shape != (3 * D, D):
            raise ValueError(f"unexpected qkv shape {qkv.shape}")
        w.tensor(f"{b}.attn_qkv.weight", qkv)
        w.tensor(f"{b}.attn_qkv.bias", st.get(f"{s}.attn.qkv.bias"))
        w.tensor(f"{b}.attn_out.weight", st.get(f"{s}.attn.proj.weight"))
        w.tensor(f"{b}.attn_out.bias", st.get(f"{s}.attn.proj.bias"))
        w.tensor(f"{b}.ln2.weight", st.get(f"{s}.norm2.weight"))
        w.tensor(f"{b}.ln2.bias", st.get(f"{s}.norm2.bias"))
        w.tensor(f"{b}.ffn_up.weight", st.get(f"{s}.mlp.fc1.weight"))
        w.tensor(f"{b}.ffn_up.bias", st.get(f"{s}.mlp.fc1.bias"))
        w.tensor(f"{b}.ffn_down.weight", st.get(f"{s}.mlp.fc2.weight"))
        w.tensor(f"{b}.ffn_down.bias", st.get(f"{s}.mlp.fc2.bias"))
    w.tensor("enc.norm.weight", st.get(f"{enc}.norm.weight"))
    w.tensor("enc.norm.bias", st.get(f"{enc}.norm.bias"))

    unused = st.unused()
    if unused:
        raise RuntimeError(f"{len(unused)} source tensors were not mapped: {unused[:10]} ...")

    w.close()
    summary = dict(out=os.path.abspath(out), ftype=ftype, arch=arch, name=model_name,
                   n_tensors=len(w.tensor_names), dtype_counts=w.dtype_counts, tensor_bytes=w.n_bytes,
                   n_source_tensors=len(st.keys), n_tokens=n_tokens, attn_mode=attn_mode)
    log(f"[levjepa] wrote {out}: {summary['n_tensors']} tensors {w.dtype_counts}, "
        f"{w.n_bytes / 2**20:.1f} MiB of tensor data")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="HF model directory (config.json + model.safetensors)")
    ap.add_argument("--out", help="output .gguf path")
    ap.add_argument("--ftype", default="f16", choices=sorted(FTYPES), help="weight storage type (default f16)")
    ap.add_argument("--name", default=None, help=f"general.name override (default {DEFAULT_NAME})")
    ap.add_argument("--source-url", default=None, help="general.source_url override")
    ap.add_argument("--info", metavar="GGUF", help="print metadata + tensor summary of an existing GGUF and exit")
    ap.add_argument("--tensors", action="store_true", help="with --info: also list every tensor")
    args = ap.parse_args(argv)
    if args.info:
        describe(args.info, show_tensors=args.tensors)
        return 0
    if not args.src or not args.out:
        ap.error("--src and --out are required")
    convert(args.src, args.out, args.ftype, name=args.name, source_url=args.source_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
