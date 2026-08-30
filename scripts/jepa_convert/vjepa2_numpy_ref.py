#!/usr/bin/env python3
"""Pure-numpy reference forward for V-JEPA 2 / 2.1 GGUF files + parity check against PyTorch.

This is the executable form of VJEPA_NOTES.md: it reads a GGUF written by vjepa2.py / vjepa2_1.py,
runs encoder / predictor / attentive-pool head in numpy *exactly the way the C++ graph should*
(host-side patchify -> matmul, pre-LN blocks, 3-D RoPE with the per-family cos/sin layout, ...),
and compares against the original PyTorch implementation.

    # HF V-JEPA 2 (encoder + predictor [+ pooler head])
    python scripts/jepa_convert/vjepa2_numpy_ref.py --gguf models/gguf/vjepa2-vitl-fpc64-256-f16.gguf \
        --hf models/facebook/vjepa2-vitl-fpc64-256 --frames 4 --size 256
    # Meta V-JEPA 2.1 (video, image and RoPE-interpolated / non-square grids)
    python scripts/jepa_convert/vjepa2_numpy_ref.py --gguf models/gguf/vjepa2_1-vitb-384-f16.gguf \
        --meta models/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt --vjepa2-src tmp/vjepa2-src

Exit code 1 if any cosine similarity is below --min-cos.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

import gguf
from gguf import GGMLQuantizationType, GGUFReader

try:  # erf: scipy if available, else torch, else a 1.5e-7 polynomial
    from scipy.special import erf as _erf  # type: ignore
except Exception:  # pragma: no cover
    try:
        import torch as _torch

        def _erf(x):
            return _torch.erf(_torch.from_numpy(np.ascontiguousarray(x))).numpy()
    except Exception:
        def _erf(x):  # Abramowitz-Stegun 7.1.26
            s = np.sign(x)
            a = np.abs(x)
            t = 1.0 / (1.0 + 0.3275911 * a)
            y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                       + 0.254829592) * t * np.exp(-a * a)
            return s * y


# ----------------------------------------------------------------------------------------------
# GGUF loading
# ----------------------------------------------------------------------------------------------
def load_gguf(path: str):
    r = GGUFReader(path)
    kv = {k: f.contents() for k, f in r.fields.items() if not k.startswith("GGUF.")}
    W: dict[str, np.ndarray] = {}
    for t in r.tensors:
        shape = [int(x) for x in reversed(list(t.shape))]  # ggml ne -> PyTorch order
        data = np.asarray(t.data)
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            a = data.astype(np.float32).reshape(shape)
        else:
            a = gguf.quants.dequantize(data, t.tensor_type).astype(np.float32).reshape(shape)
        W[t.name] = a
    return kv, W


# ----------------------------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------------------------
def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu_erf(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def linear(x, w, b=None):
    y = x @ w.T
    return y if b is None else y + b


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def patchify(x, tubelet: int, patch: int) -> np.ndarray:
    """x: [C, T, H, W] -> [Tt*gh*gw, C*tubelet*patch*patch], token order T-major, then H, then W,
    feature order C-T-H-W (matches Conv3d weight [D, C, T, P, P] flattened row-major)."""
    C, T, H, W = x.shape
    Tt, gh, gw = T // tubelet, H // patch, W // patch
    x = x[:, : Tt * tubelet, : gh * patch, : gw * patch]
    x = x.reshape(C, Tt, tubelet, gh, patch, gw, patch).transpose(1, 3, 5, 0, 2, 4, 6)
    return np.ascontiguousarray(x).reshape(Tt * gh * gw, C * tubelet * patch * patch)


# ----------------------------------------------------------------------------------------------
# 3-D RoPE (see VJEPA_NOTES.md)
# ----------------------------------------------------------------------------------------------
def rope_positions(ids: np.ndarray, gh: int, gw: int):
    """token id -> (t, h, w) for T-major/H/W token order."""
    ids = np.asarray(ids, dtype=np.int64)
    tpf = gh * gw
    t = ids // tpf
    rem = ids - t * tpf
    h = rem // gw
    w = rem - h * gw
    return t.astype(np.float64), h.astype(np.float64), w.astype(np.float64)


def rope_tables(pos_t, pos_h, pos_w, head_dim: int, theta: float, layout: str):
    """cos/sin tables [N, head_dim].  Dims [0:d) rotate with t, [d:2d) with h, [2d:3d) with w,
    d = 2*((head_dim//3)//2); the last head_dim-3d dims are untouched (cos=1, sin=0).

    layout == "interleaved" (V-JEPA 2.1): table[2k] = table[2k+1] = f(pos*omega_k)  (proper RoPE)
    layout == "tiled"       (V-JEPA 2)  : table[j] = f(pos*omega_{j mod d/2})         (Meta/HF quirk)
    """
    N = pos_t.shape[0]
    d = 2 * ((head_dim // 3) // 2)
    cos = np.ones((N, head_dim), dtype=np.float32)
    sin = np.zeros((N, head_dim), dtype=np.float32)
    omega = 1.0 / theta ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
    for ax, pos in enumerate((pos_t, pos_h, pos_w)):
        freq = pos[:, None] * omega[None, :]  # [N, d/2]
        c, s = np.cos(freq), np.sin(freq)
        if layout == "tiled":
            c, s = np.concatenate([c, c], -1), np.concatenate([s, s], -1)
        elif layout == "interleaved":
            c, s = np.repeat(c, 2, -1), np.repeat(s, 2, -1)
        else:
            raise ValueError(layout)
        cos[:, ax * d:(ax + 1) * d] = c
        sin[:, ax * d:(ax + 1) * d] = s
    return cos, sin


def apply_rope(x, cos, sin):
    """x: [H, N, hd]; rotates interleaved pairs (x[2k], x[2k+1]) -> x*cos + rot90(x)*sin,
    rot90(y1, y2) = (-y2, y1)."""
    rot = np.empty_like(x)
    rot[..., 0::2] = -x[..., 1::2]
    rot[..., 1::2] = x[..., 0::2]
    return x * cos[None] + rot * sin[None]


# ----------------------------------------------------------------------------------------------
# transformer pieces
# ----------------------------------------------------------------------------------------------
def attention(x, W, p, n_head, rope=None):
    N, D = x.shape
    hd = D // n_head
    qkv = linear(x, W[f"{p}.attn_qkv.weight"], W[f"{p}.attn_qkv.bias"])
    q = qkv[:, :D].reshape(N, n_head, hd).transpose(1, 0, 2)
    k = qkv[:, D:2 * D].reshape(N, n_head, hd).transpose(1, 0, 2)
    v = qkv[:, 2 * D:].reshape(N, n_head, hd).transpose(1, 0, 2)
    if rope is not None:
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
    att = softmax((q @ k.transpose(0, 2, 1)) / math.sqrt(hd))
    o = (att @ v).transpose(1, 0, 2).reshape(N, D)
    return linear(o, W[f"{p}.attn_out.weight"], W[f"{p}.attn_out.bias"])


def block(x, W, p, n_head, eps, rope=None):
    h = layer_norm(x, W[f"{p}.ln1.weight"], W[f"{p}.ln1.bias"], eps)
    x = x + attention(h, W, p, n_head, rope)
    h = layer_norm(x, W[f"{p}.ln2.weight"], W[f"{p}.ln2.bias"], eps)
    h = gelu_erf(linear(h, W[f"{p}.ffn_up.weight"], W[f"{p}.ffn_up.bias"]))
    return x + linear(h, W[f"{p}.ffn_down.weight"], W[f"{p}.ffn_down.bias"])


def encoder_forward(kv, W, x, mode="video"):
    """x: [C, T, H, W] normalized pixels (T == 1 with mode='image' for V-JEPA 2.1).
    Returns (tokens [N, D], (Tt, gh, gw))."""
    D = kv["jepa.enc.embed_dim"]
    P = kv["jepa.enc.patch_size"]
    tub = kv["jepa.enc.tubelet_size"] if mode == "video" else 1
    n_head = kv["jepa.enc.n_head"]
    eps = kv["jepa.enc.ln_eps"]
    C, T, H, Wd = x.shape
    toks = patchify(x, tub, P)
    if mode == "video":
        h = linear(toks, W["enc.patch_embed.weight"], W["enc.patch_embed.bias"])
        if kv.get("jepa.enc.modality_embed"):
            h = h + W["enc.mod_embed_video"]
    else:
        h = linear(toks, W["enc.patch_embed_img.weight"], W["enc.patch_embed_img.bias"])
        if kv.get("jepa.enc.modality_embed"):
            h = h + W["enc.mod_embed_img"]
    Tt, gh, gw = T // tub, H // P, Wd // P
    N = Tt * gh * gw
    pt, ph, pw = rope_positions(np.arange(N), gh, gw)
    if kv.get("jepa.enc.rope_interpolate"):
        ref = kv["jepa.enc.rope_ref_grid"]
        ph = ph * (ref - 1) / (gh - 1)
        pw = pw * (ref - 1) / (gw - 1)
    rope = rope_tables(pt, ph, pw, D // n_head, kv["jepa.enc.rope_theta"], kv["jepa.enc.rope_freq_layout"])
    for i in range(kv["jepa.enc.n_layer"]):
        h = block(h, W, f"enc.blk.{i}", n_head, eps, rope)
    h = layer_norm(h, W["enc.norm.weight"], W["enc.norm.bias"], eps)
    return h, (Tt, gh, gw)


def predictor_forward(kv, W, enc_tokens, ids_ctx, ids_tgt, grid_hw, mask_index=1, mode="video"):
    """Masked predictor with explicit context / target token ids on grid (gh, gw).
    Returns (pred [N_tgt, out_dim], ctx_out [N_ctx, out_dim] or None)."""
    Dp = kv["jepa.pred.embed_dim"]
    n_head = kv["jepa.pred.n_head"]
    eps = kv["jepa.enc.ln_eps"]
    if "pred.embed.0.weight" in W:
        ctx = gelu_erf(linear(enc_tokens, W["pred.embed.0.weight"], W["pred.embed.0.bias"]))
        ctx = linear(ctx, W["pred.embed.2.weight"], W["pred.embed.2.bias"])
    else:
        ctx = linear(enc_tokens, W["pred.embed.weight"], W["pred.embed.bias"])
    mt = W["pred.mask_tokens"][mask_index % kv["jepa.pred.n_mask_tokens"]]
    tgt = np.repeat(mt[None], len(ids_tgt), 0)
    x = np.concatenate([ctx, tgt], 0)
    if kv.get("jepa.pred.modality_embed"):
        x = x + (W["pred.mod_embed_video"] if mode == "video" else W["pred.mod_embed_img"])
    ids = np.concatenate([ids_ctx, ids_tgt])
    gh, gw = grid_hw
    pt, ph, pw = rope_positions(ids, gh, gw)
    if kv.get("jepa.pred.rope_interpolate"):
        ref = kv["jepa.pred.rope_ref_grid"]
        ph = ph * (ref - 1) / (gh - 1)
        pw = pw * (ref - 1) / (gw - 1)
    rope = rope_tables(pt, ph, pw, Dp // n_head, kv["jepa.enc.rope_theta"], kv["jepa.pred.rope_freq_layout"])
    for i in range(kv["jepa.pred.n_layer"]):
        x = block(x, W, f"pred.blk.{i}", n_head, eps, rope)
    x = layer_norm(x, W["pred.norm.weight"], W["pred.norm.bias"], eps)
    n_ctx = len(ids_ctx)
    pred = linear(x[n_ctx:], W["pred.proj.weight"], W["pred.proj.bias"])
    ctx_out = None
    if "pred.proj_context.weight" in W:
        ctx_out = linear(x[:n_ctx], W["pred.proj_context.weight"], W["pred.proj_context.bias"])
    return pred, ctx_out


def head_forward(kv, W, enc_tokens):
    """HF VJEPA2AttentivePooler + classifier: self-attn blocks over all tokens, then one query
    token cross-attends (no output projection), residual on the *raw* query, LN2 + MLP, linear."""
    D = kv["jepa.enc.embed_dim"]
    n_head = kv["jepa.enc.n_head"]
    eps = kv["jepa.enc.ln_eps"]
    x = enc_tokens
    for i in range(kv["jepa.head.n_pool_layers"]):
        x = block(x, W, f"head.blk.{i}", n_head, eps, None)
    q0 = W["head.query"].reshape(1, D)
    kvn = layer_norm(x, W["head.xattn.ln_kv.weight"], W["head.xattn.ln_kv.bias"], eps)
    q = linear(q0, W["head.xattn.q.weight"], W["head.xattn.q.bias"])
    k = linear(kvn, W["head.xattn.k.weight"], W["head.xattn.k.bias"])
    v = linear(kvn, W["head.xattn.v.weight"], W["head.xattn.v.bias"])
    hd = D // n_head
    N = x.shape[0]
    qh = q.reshape(1, n_head, hd).transpose(1, 0, 2)
    kh = k.reshape(N, n_head, hd).transpose(1, 0, 2)
    vh = v.reshape(N, n_head, hd).transpose(1, 0, 2)
    att = softmax((qh @ kh.transpose(0, 2, 1)) / math.sqrt(hd))
    o = (att @ vh).transpose(1, 0, 2).reshape(1, D)
    y = q0 + o
    h = layer_norm(y, W["head.xattn.ln2.weight"], W["head.xattn.ln2.bias"], eps)
    h = gelu_erf(linear(h, W["head.xattn.ffn_up.weight"], W["head.xattn.ffn_up.bias"]))
    y = y + linear(h, W["head.xattn.ffn_down.weight"], W["head.xattn.ffn_down.bias"])
    return linear(y, W["head.cls.weight"], W["head.cls.bias"])[0]


# ----------------------------------------------------------------------------------------------
# comparison helpers
# ----------------------------------------------------------------------------------------------
def compare(name, a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    mad = float(np.abs(a - b).max())
    rel = mad / (float(np.abs(b).max()) + 1e-30)
    print(f"  {name:44s} cos={cos:.7f}  max|d|={mad:.3e}  rel={rel:.3e}  (n={a.size})")
    return cos


def check_hf(args, kv, W):
    import torch
    from transformers import VJEPA2ForVideoClassification, VJEPA2Model

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    has_head = kv.get("jepa.head.kind") == "attentive_pool"
    cls = VJEPA2ForVideoClassification if has_head else VJEPA2Model
    m = cls.from_pretrained(args.hf, dtype=torch.float32, attn_implementation="eager").eval()
    T, S = args.frames, args.size
    x = torch.randn(1, T, 3, S, S)  # HF layout [B, T, C, H, W]
    cos_all = []
    with torch.no_grad():
        t0 = time.time()
        if has_head:
            core = m.vjepa2
            out = core(pixel_values_videos=x)
            logits_ref = m.classifier(m.pooler(out.last_hidden_state))[0].numpy()
        else:
            out = m(pixel_values_videos=x)
        enc_ref = out.last_hidden_state[0].numpy()
        pred_ref = out.predictor_output.last_hidden_state[0].numpy()
        print(f"  torch reference done in {time.time() - t0:.1f}s; tokens={enc_ref.shape[0]}")
    t0 = time.time()
    xn = x[0].permute(1, 0, 2, 3).numpy()  # [C, T, H, W]
    enc, (Tt, gh, gw) = encoder_forward(kv, W, xn)
    print(f"  numpy encoder done in {time.time() - t0:.1f}s; grid=({Tt},{gh},{gw})")
    cos_all.append(compare("encoder last_hidden_state", enc, enc_ref))
    N = enc.shape[0]
    t0 = time.time()
    grid_hw = (kv["jepa.enc.img_size"] // kv["jepa.enc.patch_size"],) * 2  # HF uses the config grid
    pred, _ = predictor_forward(kv, W, enc, np.arange(N), np.arange(N), grid_hw, mask_index=1)
    print(f"  numpy predictor done in {time.time() - t0:.1f}s")
    cos_all.append(compare("predictor output (ctx=all, tgt=all)", pred, pred_ref))
    if has_head:
        logits = head_forward(kv, W, enc)
        cos_all.append(compare("head logits", logits, logits_ref))
        top_np, top_ref = int(np.argmax(logits)), int(np.argmax(logits_ref))
        print(f"  top-1: numpy={top_np} ref={top_ref} {'OK' if top_np == top_ref else 'MISMATCH'}")
        if top_np != top_ref:
            cos_all.append(0.0)
    return cos_all


def check_meta(args, kv, W):
    import torch

    sys.path.insert(0, os.path.abspath(args.vjepa2_src))
    from app.vjepa_2_1.models import predictor as vit_predictor, vision_transformer as vit_encoder
    from src.hub.backbones import _clean_backbone_key

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    P = kv["jepa.enc.patch_size"]
    img = kv["jepa.enc.img_size"]
    nf = kv["jepa.enc.n_frames"]
    arch = {768: "vit_base", 1024: "vit_large", 1408: "vit_giant_xformers", 1664: "vit_gigantic_xformers"}[
        kv["jepa.enc.embed_dim"]]
    enc = vit_encoder.__dict__[arch](
        patch_size=P, img_size=(img, img), num_frames=nf, tubelet_size=kv["jepa.enc.tubelet_size"],
        use_sdpa=False, use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
        img_temporal_dim_size=1, interpolate_rope=True, n_output_distillation=1,
    )
    prd = vit_predictor.vit_predictor(
        img_size=(img, img), patch_size=P, use_mask_tokens=True, embed_dim=kv["jepa.enc.embed_dim"],
        predictor_embed_dim=kv["jepa.pred.embed_dim"], teacher_embed_dim=1664, num_frames=nf,
        tubelet_size=kv["jepa.enc.tubelet_size"], depth=kv["jepa.pred.n_layer"], num_heads=kv["jepa.pred.n_head"],
        num_mask_tokens=kv["jepa.pred.n_mask_tokens"], use_rope=True, uniform_power=False, use_sdpa=False,
        use_silu=False, wide_silu=True, n_output_distillation=1, return_all_tokens=True, img_temporal_dim_size=1,
    )
    ck = torch.load(args.meta, map_location="cpu", weights_only=True)
    print("  encoder load:", enc.load_state_dict(_clean_backbone_key(ck[args.encoder_key]), strict=True))
    print("  predictor load:", prd.load_state_dict(_clean_backbone_key(ck["predictor"]), strict=True))
    del ck
    enc.eval()
    prd.eval()
    cos_all = []

    cases = [("video 2x%dx%d" % (img, img), 2, img, img, "video"),
             ("image 1x%dx%d" % (img, img), 1, img, img, "image"),
             ("video 4x320x320 (rope interp 15/19)", 4, 320, 320, "video"),
             ("video 2x256x320 (non-square)", 2, 256, 320, "video")]
    for label, T, H, Wd, mode in cases:
        x = torch.randn(1, 3, T, H, Wd)
        with torch.no_grad():
            t0 = time.time()
            ref = enc(x)[0].numpy()
            print(f"  [{label}] torch encoder {time.time() - t0:.1f}s, tokens={ref.shape[0]}")
        t0 = time.time()
        out, (Tt, gh, gw) = encoder_forward(kv, W, x[0].numpy(), mode=mode)
        print(f"  [{label}] numpy encoder {time.time() - t0:.1f}s, grid=({Tt},{gh},{gw})")
        cos_all.append(compare(f"[{label}] encoder", out, ref))
        if mode == "video" and H == img and Wd == img:
            N = out.shape[0]
            ids = torch.arange(N)[None]
            with torch.no_grad():
                t0 = time.time()
                p_ref, c_ref = prd(torch.from_numpy(ref)[None], [ids], [ids], mod="video", mask_index=1)
                print(f"  [{label}] torch predictor {time.time() - t0:.1f}s")
            t0 = time.time()
            gp = kv["jepa.pred.grid_size"]
            pred, ctx = predictor_forward(kv, W, ref, np.arange(N), np.arange(N), (gp, gp), 1, "video")
            print(f"  [{label}] numpy predictor {time.time() - t0:.1f}s")
            cos_all.append(compare(f"[{label}] predictor targets", pred, p_ref[0].numpy()))
            cos_all.append(compare(f"[{label}] predictor context proj", ctx, c_ref[0].numpy()))
    return cos_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf", help="HF model dir to compare against (V-JEPA 2)")
    ap.add_argument("--meta", help="V-JEPA 2.1 .pt checkpoint to compare against")
    ap.add_argument("--vjepa2-src", default="tmp/vjepa2-src", help="clone of facebookresearch/vjepa2")
    ap.add_argument("--encoder-key", default="ema_encoder")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--rope-layout", choices=["tiled", "interleaved"], help="override (to demonstrate the quirk)")
    ap.add_argument("--min-cos", type=float, default=0.999)
    args = ap.parse_args(argv)

    kv, W = load_gguf(args.gguf)
    print(f"{args.gguf}: family={kv['jepa.family']} tensors={len(W)} rope_layout={kv['jepa.enc.rope_freq_layout']}")
    if args.rope_layout:
        kv["jepa.enc.rope_freq_layout"] = args.rope_layout
        kv["jepa.pred.rope_freq_layout"] = args.rope_layout
        print(f"  !! overriding rope layout -> {args.rope_layout}")
    if args.hf:
        cos = check_hf(args, kv, W)
    elif args.meta:
        cos = check_meta(args, kv, W)
    else:
        ap.error("need --hf or --meta")
    ok = all(c >= args.min_cos for c in cos)
    print("RESULT:", "PASS" if ok else "FAIL", f"(min cos {min(cos):.6f}, threshold {args.min_cos})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
