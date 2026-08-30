"""quentinll/lewm-* (LeWorldModel, weights.pt + config.json) -> jepa.cpp GGUF, family "lewm".

Source (stable_worldmodel.wm.lewm.LeWM state dict):
  encoder.*          HF ViTModel (stable_pretraining.backbone.utils.vit_hf: size=tiny -> D=192,
                     12 layers, 3 heads, ffn 768; patch 14, img 224 -> 256 patches + CLS;
                     ViTConfig defaults: LayerNorm eps 1e-12, GELU(erf), qkv bias; no pooler)
  projector.net.*    MLP  Linear(192,2048) -> BatchNorm1d(2048) -> GELU -> Linear(2048,192)
  predictor.*        Predictor: pos_embedding [1,3,192] + Transformer of 6 ConditionalBlocks
                     (adaLN-zero, 16 heads x 64, mlp 2048), final LayerNorm(192)
  action_encoder.*   Embedder: Conv1d(10,10,k=1) -> Linear(10,768) -> SiLU -> Linear(768,192)
  pred_proj.net.*    MLP  Linear(192,2048) -> BatchNorm1d(2048) -> GELU -> Linear(2048,192)

Inference graph (LeWM.encode / LeWM.predict, eval mode):
  emb_t   = enc.proj( ViT(pixels_t)[CLS] )                       -> (T, 192)
  a_t     = pred.action_embed( action_t )                        -> (T, 192)
  x       = emb + pred.pos_embed[:T]
  per block (c = a):
      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split6( adaln( silu(c) ) )
      h = LN_noaffine(x, eps=1e-6) * (1 + scale_msa) + shift_msa
      h = ln1(h)                                                  (affine, eps 1e-5)
      q,k,v = attn_qkv(h)  (no bias; 16 heads x 64)   ; causal softmax(q k^T / 8) v
      x = x + gate_msa * attn_out(...)
      h = LN_noaffine(x, eps=1e-6) * (1 + scale_mlp) + shift_mlp
      x = x + gate_mlp * ffn_down( gelu( ffn_up( ln2(h) ) ) )    (ln2 affine, eps 1e-5)
  x       = pred.norm(x)                                          (affine, eps 1e-5)
  pred_t  = pred.proj(x_t)                                        (BN folded, GELU)
The rollout appends pred_{T} as the next emb and slides a window of n_frames=3.

Folds performed here (all exact up to float rounding, verified by selftest.py):
  * BatchNorm1d (eval) folded into the preceding Linear of both MLPs
  * Conv1d(k=1) of the action Embedder folded into the following Linear
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (JepaWriter, SourceTensors, check_unmapped, fold_batchnorm, fold_linear_pair, log,
                     read_json)
from .hfvit import map_hf_vit

VIT_SIZES = {  # stable_pretraining.backbone.utils.vit_hf
    "tiny": (192, 12, 3), "small": (384, 12, 6), "base": (768, 12, 12), "large": (1024, 24, 16), "huge": (1280, 32, 16),
}
BN_EPS = 1e-5  # torch.nn.BatchNorm1d default (config passes no eps)

# LeWM training pipeline (stable-worldmodel scripts/train/lewm.py get_img_preprocessor):
#   ToImage(uint8 -> float [0,1], Normalize(ImageNet)) then torchvision v2 Resize(224, bilinear, antialias)
LEWM_PRE = {
    "jepa.pre.mean": [0.485, 0.456, 0.406],
    "jepa.pre.std": [0.229, 0.224, 0.225],
    "jepa.pre.resize_short": 224,
    "jepa.pre.crop": 224,
    "jepa.pre.resample": "bilinear",
    "jepa.pre.resize_mode": "shortest_edge",
}


def _cfg_get(d: dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def _fold_mlp(st: SourceTensors, prefix: str, expect_in: int, expect_hidden: int, expect_out: int):
    """MLP net.0 Linear -> net.1 BatchNorm1d -> net.2 act -> net.3 Linear  => two Linears."""
    w0 = st.take(prefix + "net.0.weight")
    b0 = st.take(prefix + "net.0.bias")
    if w0.shape != (expect_hidden, expect_in):
        raise SystemExit(f"{prefix}net.0.weight shape {w0.shape} != {(expect_hidden, expect_in)}")
    wf, bf = fold_batchnorm(w0, b0,
                            st.take(prefix + "net.1.weight"), st.take(prefix + "net.1.bias"),
                            st.take(prefix + "net.1.running_mean"), st.take(prefix + "net.1.running_var"), BN_EPS)
    st.drop(prefix + "net.1.num_batches_tracked")
    w3 = st.take(prefix + "net.3.weight")
    b3 = st.take(prefix + "net.3.bias")
    if w3.shape != (expect_out, expect_hidden):
        raise SystemExit(f"{prefix}net.3.weight shape {w3.shape} != {(expect_out, expect_hidden)}")
    return (wf, bf), (w3, b3)


def convert(src: str | Path, out: str | Path, ftype: str, *, name: str | None = None,
            allow_unmapped: bool = False) -> Path:
    src = Path(src)
    cfg = read_json(src / "config.json")
    st = SourceTensors.from_torch(src / "weights.pt")

    # ---- encoder (HF ViTModel) -------------------------------------------------------------
    enc_cfg = cfg.get("encoder", {})
    if "vit_hf" not in str(_cfg_get(enc_cfg, "_target_", "")):
        raise SystemExit(f"encoder _target_={_cfg_get(enc_cfg, '_target_')!r}; only stable_pretraining vit_hf is supported")
    D, L, H = VIT_SIZES[_cfg_get(enc_cfg, "size", "tiny")]
    F = 4 * D
    P = int(_cfg_get(enc_cfg, "patch_size", 16))
    img = int(_cfg_get(enc_cfg, "image_size", 224))
    n_found = st.layer_count("encoder.encoder.layer.{i}.layernorm_before.weight")
    if n_found != L:
        raise SystemExit(f"vit_hf size implies {L} layers, checkpoint has {n_found}")
    pw = st.peek("encoder.embeddings.patch_embeddings.projection.weight")
    if pw.shape != (D, 3, P, P):
        raise SystemExit(f"encoder patch conv shape {pw.shape} != {(D, 3, P, P)}")
    n_pos = st.peek("encoder.embeddings.position_embeddings").shape[1]
    if n_pos != 1 + (img // P) ** 2:
        raise SystemExit(f"encoder pos table has {n_pos} rows, expected {1 + (img // P) ** 2}")

    # ---- predictor / action encoder / projector configs -----------------------------------
    pcfg = cfg["predictor"]
    PD = int(pcfg["input_dim"])
    if int(pcfg["hidden_dim"]) != PD or int(pcfg.get("output_dim", PD)) != PD:
        raise SystemExit("predictor input/hidden/output dims differ; input_proj/output_proj would be needed")
    PL = int(pcfg["depth"])
    PH = int(pcfg["heads"])
    PHD = int(pcfg.get("dim_head", 64))
    PF = int(pcfg["mlp_dim"])
    NF = int(pcfg["num_frames"])
    acfg = cfg["action_encoder"]
    A = int(acfg["input_dim"])
    if int(acfg["emb_dim"]) != PD:
        raise SystemExit("action_encoder emb_dim != predictor dim")
    pj = cfg.get("projector", {})
    pp = cfg.get("pred_proj", {})
    PJH = int(_cfg_get(pj, "hidden_dim", 2048))
    PPH = int(_cfg_get(pp, "hidden_dim", 2048))
    for nm, c in (("projector", pj), ("pred_proj", pp)):
        if "BatchNorm1d" not in str(_cfg_get(_cfg_get(c, "norm_fn", {}), "_target_", "")):
            raise SystemExit(f"{nm}.norm_fn is not BatchNorm1d; folding rule does not apply")
    if PD != D:
        raise SystemExit(f"encoder dim {D} != predictor dim {PD}")

    model_name = name or src.name
    w = JepaWriter(out, ftype, name=model_name, family="lewm", modality="image",
                   license="mit", source_url=f"https://huggingface.co/quentinll/{src.name}",
                   description="LeWorldModel (LeWM) ViT-tiny/14 encoder + adaLN action-conditioned predictor")
    hp = {
        "jepa.enc.embed_dim": D,
        "jepa.enc.n_layer": L,
        "jepa.enc.n_head": H,
        "jepa.enc.ffn_dim": F,
        "jepa.enc.patch_size": P,
        "jepa.enc.tubelet_size": 1,
        "jepa.enc.img_size": img,
        "jepa.enc.n_frames": 1,
        "jepa.enc.in_chans": 3,
        "jepa.enc.ln_eps": 1e-12,      # transformers ViTConfig default
        "jepa.enc.act": "gelu_erf",    # ViTConfig hidden_act="gelu"
        "jepa.enc.pos_type": "learned",
        "jepa.enc.cls_token": True,
        "jepa.enc.n_registers": 0,
        "jepa.enc.qkv_fused": True,
        "jepa.enc.layer_scale": False,
        "jepa.enc.proj_act": "gelu_erf",
        "jepa.pred.kind": "lewm",
        "jepa.pred.embed_dim": PD,
        "jepa.pred.n_layer": PL,
        "jepa.pred.n_head": PH,
        "jepa.pred.head_dim": PHD,
        "jepa.pred.ffn_dim": PF,
        "jepa.pred.n_frames": NF,
        "jepa.pred.action_dim": A,
        "jepa.pred.state_dim": 0,
        "jepa.pred.frame_causal": True,
        "jepa.pred.ln_eps": 1e-5,       # nn.LayerNorm default: attn.norm, mlp.net.0, transformer.norm
        "jepa.pred.adaln_eps": 1e-6,    # ConditionalBlock.norm1/norm2 (elementwise_affine=False)
        "jepa.pred.act": "gelu_erf",
        "jepa.pred.qkv_bias": False,
        "jepa.pred.action_act": "silu",
        "jepa.pred.proj_act": "gelu_erf",
        "jepa.head.kind": "none",
    }
    hp.update(LEWM_PRE)
    w.add_hparams(hp)
    log(f"lewm: encoder ViT D={D} L={L} H={H} F={F} P={P} img={img} | predictor D={PD} L={PL} H={PH}x{PHD} F={PF} "
        f"frames={NF} action_dim={A} | projector hidden {PJH}, pred_proj hidden {PPH}")

    # ---- encoder ---------------------------------------------------------------------------
    map_hf_vit(st, w, "encoder.", n_layer=L, embed_dim=D, ffn_dim=F)

    # ---- projector: enc.proj.0 (BN folded) / enc.proj.2 ----------------------------------------
    (w0, b0), (w3, b3) = _fold_mlp(st, "projector.", D, PJH, PD)
    w.add_linear("enc.proj.0", w0, b0, True)
    w.add_linear("enc.proj.2", w3, b3, True)

    # ---- predictor ---------------------------------------------------------------------------
    pos = st.take("predictor.pos_embedding").reshape(-1, PD)
    if pos.shape[0] != NF:
        raise SystemExit(f"predictor.pos_embedding has {pos.shape[0]} rows, num_frames={NF}")
    w.add_tensor("pred.pos_embed", pos, False)
    inner = PH * PHD
    n_pl = st.layer_count("predictor.transformer.layers.{i}.attn.to_qkv.weight")
    if n_pl != PL:
        raise SystemExit(f"predictor depth {PL} but checkpoint has {n_pl} layers")
    for i in range(PL):
        s = f"predictor.transformer.layers.{i}."
        b = f"pred.blk.{i}."
        ada_w = st.take(s + "adaLN_modulation.1.weight")
        if ada_w.shape != (6 * PD, PD):
            raise SystemExit(f"{s}adaLN_modulation.1.weight shape {ada_w.shape} != {(6 * PD, PD)}")
        w.add_linear(b + "adaln", ada_w, st.take(s + "adaLN_modulation.1.bias"), False)
        w.add_norm(b + "ln1", st.take(s + "attn.norm.weight"), st.take(s + "attn.norm.bias"))
        qkv = st.take(s + "attn.to_qkv.weight")
        if qkv.shape != (3 * inner, PD):
            raise SystemExit(f"{s}attn.to_qkv.weight shape {qkv.shape} != {(3 * inner, PD)}")
        if s + "attn.to_qkv.bias" in st:
            raise SystemExit("predictor to_qkv has a bias; jepa.pred.qkv_bias would have to be true")
        w.add_linear(b + "attn_qkv", qkv, None, True)
        ow = st.take(s + "attn.to_out.0.weight")
        if ow.shape != (PD, inner):
            raise SystemExit(f"{s}attn.to_out.0.weight shape {ow.shape} != {(PD, inner)}")
        w.add_linear(b + "attn_out", ow, st.take(s + "attn.to_out.0.bias"), True)
        w.add_norm(b + "ln2", st.take(s + "mlp.net.0.weight"), st.take(s + "mlp.net.0.bias"))
        up = st.take(s + "mlp.net.1.weight")
        if up.shape != (PF, PD):
            raise SystemExit(f"{s}mlp.net.1.weight shape {up.shape} != {(PF, PD)}")
        w.add_linear(b + "ffn_up", up, st.take(s + "mlp.net.1.bias"), True)
        w.add_linear(b + "ffn_down", st.take(s + "mlp.net.4.weight"), st.take(s + "mlp.net.4.bias"), True)
    w.add_norm("pred.norm", st.take("predictor.transformer.norm.weight"), st.take("predictor.transformer.norm.bias"))

    # ---- action embedder: Conv1d(k=1) folded into embed.0 --------------------------------------
    cw = st.take("action_encoder.patch_embed.weight")  # [smoothed, A, 1]
    cb = st.take("action_encoder.patch_embed.bias")
    if cw.ndim != 3 or cw.shape[2] != 1 or cw.shape[1] != A:
        raise SystemExit(f"action_encoder.patch_embed.weight shape {cw.shape}, expected [*, {A}, 1]")
    e0w = st.take("action_encoder.embed.0.weight")
    e0b = st.take("action_encoder.embed.0.bias")
    fw, fb = fold_linear_pair(cw[:, :, 0], cb, e0w, e0b)  # [4*PD, A]
    w.add_linear("pred.action_embed.0", fw, fb, False)
    e2w = st.take("action_encoder.embed.2.weight")
    if e2w.shape != (PD, e0w.shape[0]):
        raise SystemExit(f"action_encoder.embed.2.weight shape {e2w.shape}")
    w.add_linear("pred.action_embed.2", e2w, st.take("action_encoder.embed.2.bias"), False)

    # ---- pred_proj: pred.proj.0 (BN folded) / pred.proj.2 ------------------------------------
    (w0, b0), (w3, b3) = _fold_mlp(st, "pred_proj.", PD, PPH, PD)
    w.add_linear("pred.proj.0", w0, b0, True)
    w.add_linear("pred.proj.2", w3, b3, True)

    check_unmapped(st, ("value_function.", "encoder.pooler."), allow_unmapped)
    return w.write()
