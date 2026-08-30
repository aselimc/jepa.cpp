"""facebook/ijepa_* (HF IJepaModel safetensors) -> jepa.cpp GGUF, family "ijepa".

Source layout (transformers <= 4.x on-disk names, also accepts the 5.x `layers.N.*` names):
  embeddings.patch_embeddings.projection.{weight,bias}   [D,3,P,P]
  embeddings.position_embeddings                         [1,N,D]   fixed 2-D sincos table
  encoder.layer.{i}.attention.attention.{query,key,value}.{weight,bias}
  encoder.layer.{i}.attention.output.dense.{weight,bias}
  encoder.layer.{i}.layernorm_before / layernorm_after
  encoder.layer.{i}.intermediate.dense  (fc1) / output.dense (fc2)
  layernorm.{weight,bias}                                final norm

No CLS token, no predictor in the HF release; features = mean of the patch tokens.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (ACT_NAMES, JepaWriter, SourceTensors, check_unmapped, flatten_patch_conv,
                     fuse_qkv, log, preproc_from_hf, read_json, sincos_2d)

# transformers <= 4.x on-disk naming
LAYOUT_OLD = {
    "q": "encoder.layer.{i}.attention.attention.query",
    "k": "encoder.layer.{i}.attention.attention.key",
    "v": "encoder.layer.{i}.attention.attention.value",
    "o": "encoder.layer.{i}.attention.output.dense",
    "ln1": "encoder.layer.{i}.layernorm_before",
    "ln2": "encoder.layer.{i}.layernorm_after",
    "up": "encoder.layer.{i}.intermediate.dense",
    "down": "encoder.layer.{i}.output.dense",
}
# transformers >= 5 naming
LAYOUT_NEW = {
    "q": "layers.{i}.attention.q_proj",
    "k": "layers.{i}.attention.k_proj",
    "v": "layers.{i}.attention.v_proj",
    "o": "layers.{i}.attention.o_proj",
    "ln1": "layers.{i}.layernorm_before",
    "ln2": "layers.{i}.layernorm_after",
    "up": "layers.{i}.mlp.fc1",
    "down": "layers.{i}.mlp.fc2",
}


def _find_prefix(st: SourceTensors) -> str:
    for p in ("", "ijepa.", "model."):
        if p + "embeddings.patch_embeddings.projection.weight" in st:
            return p
    raise SystemExit("not an IJepaModel checkpoint (no embeddings.patch_embeddings.projection.weight)")


def convert(src: str | Path, out: str | Path, ftype: str, *, name: str | None = None,
            allow_unmapped: bool = False) -> Path:
    src = Path(src)
    cfg = read_json(src / "config.json")
    if cfg.get("model_type") != "ijepa":
        raise SystemExit(f"{src}/config.json model_type={cfg.get('model_type')!r}, expected 'ijepa'")
    st = SourceTensors.from_safetensors(src / "model.safetensors")
    pre = preproc_from_hf(src / "preprocessor_config.json")

    D = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    H = int(cfg["num_attention_heads"])
    F = int(cfg["intermediate_size"])
    P = int(cfg["patch_size"])
    img = int(cfg["image_size"])
    C = int(cfg.get("num_channels", 3))
    eps = float(cfg.get("layer_norm_eps", 1e-6))
    act = ACT_NAMES[cfg.get("hidden_act", "gelu")]
    if not cfg.get("qkv_bias", True):
        raise SystemExit("qkv_bias=false is not handled")
    gh = gw = img // P

    prefix = _find_prefix(st)
    layout = LAYOUT_OLD if prefix + LAYOUT_OLD["q"].format(i=0) + ".weight" in st else LAYOUT_NEW
    n_found = st.layer_count(prefix + layout["q"] + ".weight")
    if n_found != L:
        raise SystemExit(f"config says {L} layers but checkpoint has {n_found}")

    model_name = name or src.name
    w = JepaWriter(out, ftype, name=model_name, family="ijepa", modality="image",
                   license="cc-by-nc-4.0", source_url=f"https://huggingface.co/facebook/{src.name}",
                   description="I-JEPA ViT encoder (Assran et al. 2023), converted from the HF checkpoint")
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
        "jepa.enc.pos_type": "sincos2d",
        "jepa.enc.cls_token": False,
        "jepa.enc.n_registers": 0,
        "jepa.enc.qkv_fused": True,
        "jepa.enc.layer_scale": False,
        "jepa.head.kind": "none",
    }
    hp.update(pre)
    w.add_hparams(hp)
    log(f"ijepa: D={D} L={L} H={H} F={F} P={P} img={img} grid={gh}x{gw} eps={eps} act={act} "
        f"pre={pre['jepa.pre.resize_mode']}/{pre['jepa.pre.resample']} mean={pre['jepa.pre.mean']}")

    # patch embed
    pw = st.take(prefix + "embeddings.patch_embeddings.projection.weight")
    if pw.shape != (D, C, P, P):
        raise SystemExit(f"patch embed weight has shape {pw.shape}, expected {(D, C, P, P)}")
    w.add_tensor("enc.patch_embed.weight", flatten_patch_conv(pw), False)
    w.add_tensor("enc.patch_embed.bias", st.take(prefix + "embeddings.patch_embeddings.projection.bias"), False)

    # position table: the HF checkpoint stores the (fixed) sincos table as a parameter.
    pos = st.take(prefix + "embeddings.position_embeddings")
    pos = pos.reshape(-1, D)
    if pos.shape[0] != gh * gw:
        raise SystemExit(f"position table has {pos.shape[0]} rows, expected {gh * gw}")
    ref_hw = sincos_2d(gh, gw, D, w_first=False)
    ref_wh = sincos_2d(gh, gw, D, w_first=True)
    d_hw = float(np.abs(pos - ref_hw).max())
    d_wh = float(np.abs(pos - ref_wh).max())
    log(f"  pos_embed vs sincos_2d: max|diff| [emb_h|emb_w]={d_hw:.3e}  [emb_w|emb_h]={d_wh:.3e}"
        f"  -> checkpoint table {'IS' if min(d_hw, d_wh) == 0.0 else 'is NOT'} bit-identical to a generated table")
    if min(d_hw, d_wh) > 1e-5:
        log("  [warn] checkpoint position table is not a Meta sincos table; storing it verbatim anyway")
    w.add_tensor("enc.pos_embed", pos, False)  # verbatim checkpoint values (bit-exact parity with HF)

    # blocks
    for i in range(L):
        k = {n: prefix + fmt.format(i=i) for n, fmt in layout.items()}
        b = f"enc.blk.{i}."
        w.add_norm(b + "ln1", st.take(k["ln1"] + ".weight"), st.take(k["ln1"] + ".bias"))
        w.add_linear(b + "attn_qkv",
                     fuse_qkv(st.take(k["q"] + ".weight"), st.take(k["k"] + ".weight"), st.take(k["v"] + ".weight")),
                     fuse_qkv(st.take(k["q"] + ".bias"), st.take(k["k"] + ".bias"), st.take(k["v"] + ".bias")),
                     True)
        w.add_linear(b + "attn_out", st.take(k["o"] + ".weight"), st.take(k["o"] + ".bias"), True)
        w.add_norm(b + "ln2", st.take(k["ln2"] + ".weight"), st.take(k["ln2"] + ".bias"))
        up_w = st.take(k["up"] + ".weight")
        if up_w.shape != (F, D):
            raise SystemExit(f"layer {i}: fc1 shape {up_w.shape} != {(F, D)}")
        w.add_linear(b + "ffn_up", up_w, st.take(k["up"] + ".bias"), True)
        w.add_linear(b + "ffn_down", st.take(k["down"] + ".weight"), st.take(k["down"] + ".bias"), True)

    w.add_norm("enc.norm", st.take(prefix + "layernorm.weight"), st.take(prefix + "layernorm.bias"))

    # pooler (only present with add_pooling_layer=True) / classifier heads are not used for features
    check_unmapped(st, (prefix + "pooler.", "classifier."), allow_unmapped)
    return w.write()
