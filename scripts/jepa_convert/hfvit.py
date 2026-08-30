"""DINOv2 / timm-style ViT checkpoints (OK-AI LeJEPA "ViTv2") -> jepa.cpp GGUF, family "hfvit".

Two *source* layouts are understood, both written to the same `enc.*` GGUF layout:

  timm / DINOv2 (`map_timm_vit`)                 HF ViTModel (`map_hf_vit`, used by lewm.py)
  ------------------------------                 -------------------------------------------
  patch_embed.proj.{weight,bias}                 embeddings.patch_embeddings.projection.*
  cls_token [1,1,D]                              embeddings.cls_token
  pos_embed [1,1+N,D]                            embeddings.position_embeddings [1,1+N,D]
  register_tokens [1,R,D]                        -
  blocks.i.norm1 / norm2                         encoder.layer.i.layernorm_before / _after
  blocks.i.attn.qkv (fused) / attn.proj          encoder.layer.i.attention.attention.{query,key,value} / attention.output.dense
  blocks.i.mlp.fc1 / fc2                         encoder.layer.i.intermediate.dense / output.dense
  blocks.i.ls1.gamma / ls2.gamma                 -
  norm                                           layernorm

OK-AI/lejepa-vits16-pretrain-in1k facts (verified against lite_ssl/model/image/vitv2 + layers):
  LayerNorm eps 1e-6 (hard-coded `partial(nn.LayerNorm, eps=1e-6)`), nn.GELU (erf) MLP,
  qkv_bias=proj_bias=ffn_bias=True, learned pos_embed with a CLS slot at row 0, no registers,
  init_values=None -> no LayerScale.  `backbone.cva_module_proj.*` is the DINOHead used by the
  SIGReg/CVA loss; it is not part of the encoder and is skipped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (JepaWriter, SourceTensors, check_unmapped, flatten_patch_conv, fuse_qkv, log,
                     preproc_from_hf, read_json)


def map_timm_vit(st: SourceTensors, w: JepaWriter, prefix: str, *, n_layer: int, embed_dim: int,
                 ffn_dim: int, n_registers: int, layer_scale: bool) -> None:
    """Emit enc.* from a timm/DINOv2-style state dict rooted at `prefix`."""
    D = embed_dim
    pw = st.take(prefix + "patch_embed.proj.weight")
    if pw.ndim != 4 or pw.shape[0] != D:
        raise SystemExit(f"patch_embed.proj.weight has shape {pw.shape}")
    w.add_tensor("enc.patch_embed.weight", flatten_patch_conv(pw), False)
    w.add_tensor("enc.patch_embed.bias", st.take(prefix + "patch_embed.proj.bias"), False)

    cls = st.take(prefix + "cls_token").reshape(-1)
    if cls.shape != (D,):
        raise SystemExit(f"cls_token has shape {cls.shape}")
    w.add_tensor("enc.cls_token", cls, False)

    pos = st.take(prefix + "pos_embed")
    pos = pos.reshape(-1, D)
    w.add_tensor("enc.pos_embed", pos, False)  # row 0 = CLS slot

    reg = st.take_opt(prefix + "register_tokens")
    if (reg is not None) != (n_registers > 0):
        raise SystemExit(f"config says {n_registers} registers but register_tokens {'present' if reg is not None else 'absent'}")
    if reg is not None:
        reg = reg.reshape(-1, D)
        if reg.shape[0] != n_registers:
            raise SystemExit(f"register_tokens has {reg.shape[0]} rows, config says {n_registers}")
        w.add_tensor("enc.reg_tokens", reg, False)

    for i in range(n_layer):
        s = f"{prefix}blocks.{i}."
        b = f"enc.blk.{i}."
        w.add_norm(b + "ln1", st.take(s + "norm1.weight"), st.take(s + "norm1.bias"))
        qkv_w = st.take(s + "attn.qkv.weight")
        if qkv_w.shape != (3 * D, D):
            raise SystemExit(f"block {i}: attn.qkv.weight shape {qkv_w.shape} != {(3 * D, D)}")
        w.add_linear(b + "attn_qkv", qkv_w, st.take_opt(s + "attn.qkv.bias"), True)
        w.add_linear(b + "attn_out", st.take(s + "attn.proj.weight"), st.take_opt(s + "attn.proj.bias"), True)
        w.add_norm(b + "ln2", st.take(s + "norm2.weight"), st.take(s + "norm2.bias"))
        fc1 = st.take(s + "mlp.fc1.weight")
        if fc1.shape != (ffn_dim, D):
            raise SystemExit(f"block {i}: mlp.fc1.weight shape {fc1.shape} != {(ffn_dim, D)}")
        w.add_linear(b + "ffn_up", fc1, st.take_opt(s + "mlp.fc1.bias"), True)
        w.add_linear(b + "ffn_down", st.take(s + "mlp.fc2.weight"), st.take_opt(s + "mlp.fc2.bias"), True)
        ls1 = st.take_opt(s + "ls1.gamma")
        ls2 = st.take_opt(s + "ls2.gamma")
        if (ls1 is not None) != layer_scale or (ls2 is not None) != layer_scale:
            raise SystemExit(f"block {i}: layer-scale tensors {'present' if ls1 is not None else 'absent'} "
                             f"but config layer_scale={layer_scale}")
        if layer_scale:
            w.add_tensor(b + "ls1", ls1, False)
            w.add_tensor(b + "ls2", ls2, False)

    w.add_norm("enc.norm", st.take(prefix + "norm.weight"), st.take(prefix + "norm.bias"))


def map_hf_vit(st: SourceTensors, w: JepaWriter, prefix: str, *, n_layer: int, embed_dim: int,
               ffn_dim: int) -> None:
    """Emit enc.* from a HF ViTModel state dict (transformers <= 4.x on-disk names) rooted at `prefix`."""
    D = embed_dim
    pw = st.take(prefix + "embeddings.patch_embeddings.projection.weight")
    if pw.ndim != 4 or pw.shape[0] != D:
        raise SystemExit(f"patch embed weight has shape {pw.shape}")
    w.add_tensor("enc.patch_embed.weight", flatten_patch_conv(pw), False)
    w.add_tensor("enc.patch_embed.bias", st.take(prefix + "embeddings.patch_embeddings.projection.bias"), False)
    w.add_tensor("enc.cls_token", st.take(prefix + "embeddings.cls_token").reshape(-1), False)
    w.add_tensor("enc.pos_embed", st.take(prefix + "embeddings.position_embeddings").reshape(-1, D), False)
    st.drop(prefix + "embeddings.mask_token")  # iBOT-style mask token, unused at inference

    for i in range(n_layer):
        s = f"{prefix}encoder.layer.{i}."
        b = f"enc.blk.{i}."
        a = s + "attention.attention."
        w.add_norm(b + "ln1", st.take(s + "layernorm_before.weight"), st.take(s + "layernorm_before.bias"))
        w.add_linear(b + "attn_qkv",
                     fuse_qkv(st.take(a + "query.weight"), st.take(a + "key.weight"), st.take(a + "value.weight")),
                     fuse_qkv(st.take(a + "query.bias"), st.take(a + "key.bias"), st.take(a + "value.bias")),
                     True)
        w.add_linear(b + "attn_out", st.take(s + "attention.output.dense.weight"),
                     st.take(s + "attention.output.dense.bias"), True)
        w.add_norm(b + "ln2", st.take(s + "layernorm_after.weight"), st.take(s + "layernorm_after.bias"))
        fc1 = st.take(s + "intermediate.dense.weight")
        if fc1.shape != (ffn_dim, D):
            raise SystemExit(f"layer {i}: intermediate.dense.weight shape {fc1.shape} != {(ffn_dim, D)}")
        w.add_linear(b + "ffn_up", fc1, st.take(s + "intermediate.dense.bias"), True)
        w.add_linear(b + "ffn_down", st.take(s + "output.dense.weight"), st.take(s + "output.dense.bias"), True)

    w.add_norm("enc.norm", st.take(prefix + "layernorm.weight"), st.take(prefix + "layernorm.bias"))


def convert(src: str | Path, out: str | Path, ftype: str, *, name: str | None = None,
            allow_unmapped: bool = False) -> Path:
    src = Path(src)
    cfg = read_json(src / "config.json")
    if cfg.get("model_type") not in ("vitv2", "vitv3", None):
        raise SystemExit(f"{src}/config.json model_type={cfg.get('model_type')!r}; hfvit expects the OK-AI ViTv2 config")
    st = SourceTensors.from_safetensors(src / "model.safetensors")
    pre = preproc_from_hf(src / "preprocessor_config.json")

    prefix = "backbone." if "backbone.patch_embed.proj.weight" in st else ""
    D = int(cfg["embed_dim"])
    L = int(cfg["depth"])
    H = int(cfg["num_heads"])
    F = int(round(D * float(cfg.get("mlp_ratio", 4))))
    P = int(cfg["patch_size"])
    img = int(cfg.get("img_size", cfg.get("image_size", 224)))
    R = int(cfg.get("num_register_tokens", 0))
    init_values = cfg.get("init_values")
    layer_scale = bool(init_values)  # ViTv2: `LayerScale(...) if init_values else nn.Identity()`
    eps = 1e-6                       # ViTv2: norm_layer = partial(nn.LayerNorm, eps=1e-6)
    act = "gelu_erf"                 # ViTv2: act_layer=nn.GELU

    n_found = st.layer_count(prefix + "blocks.{i}.norm1.weight")
    if n_found != L:
        raise SystemExit(f"config depth={L} but checkpoint has {n_found} blocks")
    pw = st.peek(prefix + "patch_embed.proj.weight")
    C = int(pw.shape[1])
    if pw.shape[2:] != (P, P):
        raise SystemExit(f"patch conv kernel {pw.shape[2:]} != patch_size {P}")
    n_pos = st.peek(prefix + "pos_embed").shape[1]
    if n_pos != 1 + (img // P) ** 2:
        raise SystemExit(f"pos_embed has {n_pos} rows, expected 1 + ({img}/{P})^2 = {1 + (img // P) ** 2}")

    model_name = name or src.name
    w = JepaWriter(out, ftype, name=model_name, family="hfvit", modality="image",
                   license="apache-2.0", source_url=f"https://huggingface.co/OK-AI/{src.name}",
                   description="LeJEPA ViT (DINOv2-style ViTv2) encoder from Open-Knowledge-AI lite_ssl")
    hp = {
        "jepa.enc.embed_dim": D,
        "jepa.enc.n_layer": L,
        "jepa.enc.n_head": H,
        "jepa.enc.ffn_dim": F,
        "jepa.enc.patch_size": P,
        "jepa.enc.tubelet_size": 1,
        "jepa.enc.img_size": img,
        "jepa.enc.n_frames": 1,
        "jepa.enc.in_chans": C,
        "jepa.enc.ln_eps": eps,
        "jepa.enc.act": act,
        "jepa.enc.pos_type": "learned",
        "jepa.enc.cls_token": True,
        "jepa.enc.n_registers": R,
        "jepa.enc.qkv_fused": True,
        "jepa.enc.layer_scale": layer_scale,
        "jepa.head.kind": "none",
    }
    hp.update(pre)
    w.add_hparams(hp)
    log(f"hfvit: D={D} L={L} H={H} F={F} P={P} img={img} registers={R} layer_scale={layer_scale} "
        f"eps={eps} act={act} pre={pre['jepa.pre.resize_short']}->{pre['jepa.pre.crop']}/{pre['jepa.pre.resample']}")

    map_timm_vit(st, w, prefix, n_layer=L, embed_dim=D, ffn_dim=F, n_registers=R, layer_scale=layer_scale)
    check_unmapped(st, (prefix + "cva_module_proj.", prefix + "head.", prefix + "projection_head.",
                        prefix + "mask_token"), allow_unmapped)
    return w.write()
