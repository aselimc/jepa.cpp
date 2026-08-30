#!/usr/bin/env python3
"""Numerical self-test for the image-family converters.

Runs a *numpy* forward pass using only the GGUF file (tensors + jepa.* hparams, following
docs/gguf-schema.md and docs/architecture.md literally) and compares it with the PyTorch
reference model on real fixture images.  A wrong tensor mapping, a wrong fold, or a wrong
hparam shows up here as a large error.  The numpy code doubles as an executable spec of
the graph the C++ side has to build.

  selftest.py --gguf models/gguf/lejepa-vits16-pretrain-in1k-f32.gguf --src models/OK-AI/lejepa-vits16-pretrain-in1k
  selftest.py --gguf models/gguf/lewm-pusht-f32.gguf --src models/quentinll/lewm-pusht --lewm-src tmp/<branch>/stable-worldmodel
  selftest.py --gguf models/gguf/ijepa_vith14_1k-f16.gguf --src models/facebook/ijepa_vith14_1k --n-images 2

Reference models: hfvit -> timm VisionTransformer (same block math as DINOv2 ViTv2, loaded from
the safetensors); lewm -> transformers ViTModel + stable_worldmodel.wm.lewm.module (Predictor,
Embedder, MLP) loaded from weights.pt; ijepa -> transformers IJepaModel.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

import gguf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MEDIA = ROOT / "tests" / "fixtures" / "media"


# ----------------------------------------------------------------------------- GGUF access
class Model:
    def __init__(self, path: str | Path):
        self.r = gguf.GGUFReader(str(path))
        self.hp = {k: f.contents() for k, f in self.r.fields.items() if k.startswith(("jepa.", "general."))}
        self.t = {t.name: t for t in self.r.tensors}
        self._cache: dict[str, np.ndarray] = {}  # dequantized / widened tensors (quantized files are read repeatedly)

    def __contains__(self, name: str) -> bool:
        return name in self.t

    def get(self, name: str) -> np.ndarray:
        a = self._cache.get(name)
        if a is not None:
            return a
        t = self.t[name]
        a = np.asarray(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            return a  # numpy shape == PyTorch shape (gguf reverses ne for us)
        if t.tensor_type == gguf.GGMLQuantizationType.F16:
            a = a.astype(np.float32)
        else:  # q8_0 / q4_0 / q4_k / ... written by tools/jepa-quantize: dequantize, restore the PyTorch shape
            shape = [int(x) for x in reversed(list(t.shape))]
            a = gguf.quants.dequantize(a, t.tensor_type).astype(np.float32).reshape(shape)
        self._cache[name] = a
        return a

    def tensor_types(self) -> dict[str, int]:
        """{ggml type name: tensor count}, e.g. {'F32': 102, 'Q8_0': 48}."""
        out: dict[str, int] = {}
        for t in self.r.tensors:
            out[t.tensor_type.name] = out.get(t.tensor_type.name, 0) + 1
        return out

    def opt(self, name: str):
        return self.get(name) if name in self.t else None


# ----------------------------------------------------------------------------- numpy ops (float32)
def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    if w is not None:
        y = y * w
    if b is not None:
        y = y + b
    return y


def _erf(x):
    import torch  # torch.erf is used only as a vectorised erf; the GELU math itself is explicit
    return torch.erf(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()


def gelu_erf(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def act_fn(name):
    if name == "gelu_erf":
        return gelu_erf
    if name == "silu":
        return silu
    if name == "gelu_tanh":
        return lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    raise ValueError(name)


def linear(x, w, b=None):
    y = x @ w.T
    return y if b is None else y + b


def attention(q, k, v, n_head, causal=False):
    """q,k,v: [N, H*hd] -> [N, H*hd]; softmax(q k^T / sqrt(hd)) v per head."""
    N, inner = q.shape
    hd = inner // n_head
    q = q.reshape(N, n_head, hd).transpose(1, 0, 2)
    k = k.reshape(N, n_head, hd).transpose(1, 0, 2)
    v = v.reshape(N, n_head, hd).transpose(1, 0, 2)
    s = (q @ k.transpose(0, 2, 1)) / math.sqrt(hd)
    if causal:
        mask = np.triu(np.ones((N, N), dtype=bool), 1)
        s = np.where(mask[None], -np.inf, s)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(-1, keepdims=True)
    o = p @ v  # [H, N, hd]
    return o.transpose(1, 0, 2).reshape(N, inner)


def patchify(img: np.ndarray, P: int) -> np.ndarray:
    """img [C,H,W] float -> [N, C*P*P] rows in h-major token order, C-H-W order inside a row."""
    C, H, W = img.shape
    gh, gw = H // P, W // P
    x = img[:, : gh * P, : gw * P].reshape(C, gh, P, gw, P)
    x = x.transpose(1, 3, 0, 2, 4)  # gh, gw, C, P, P
    return np.ascontiguousarray(x.reshape(gh * gw, C * P * P))


# ----------------------------------------------------------------------------- encoder graph
def encoder_forward(m: Model, img: np.ndarray):
    """docs/architecture.md shared graph.  img: [C,H,W] already normalised.  Returns [N_tok, D]."""
    hp = m.hp
    D = hp["jepa.enc.embed_dim"]
    P = hp["jepa.enc.patch_size"]
    H = hp["jepa.enc.n_head"]
    eps = hp["jepa.enc.ln_eps"]
    act = act_fn(hp["jepa.enc.act"])
    x = linear(patchify(img, P), m.get("enc.patch_embed.weight"), m.get("enc.patch_embed.bias"))
    toks = [x]
    if hp.get("jepa.enc.n_registers", 0):
        toks.insert(0, m.get("enc.reg_tokens"))
    if hp["jepa.enc.cls_token"]:
        toks.insert(0, m.get("enc.cls_token")[None])
    x = np.concatenate(toks, 0)
    pos = m.get("enc.pos_embed")
    if pos.shape[0] != x.shape[0]:
        raise SystemExit(f"pos table {pos.shape} vs tokens {x.shape}: interpolation not implemented in selftest")
    x = x + pos
    for i in range(hp["jepa.enc.n_layer"]):
        b = f"enc.blk.{i}."
        h = layer_norm(x, m.get(b + "ln1.weight"), m.get(b + "ln1.bias"), eps)
        qkv = linear(h, m.get(b + "attn_qkv.weight"), m.opt(b + "attn_qkv.bias"))
        q, k, v = np.split(qkv, 3, axis=-1)
        a = linear(attention(q, k, v, H), m.get(b + "attn_out.weight"), m.opt(b + "attn_out.bias"))
        if hp.get("jepa.enc.layer_scale"):
            a = a * m.get(b + "ls1")
        x = x + a
        h = layer_norm(x, m.get(b + "ln2.weight"), m.get(b + "ln2.bias"), eps)
        f = linear(act(linear(h, m.get(b + "ffn_up.weight"), m.opt(b + "ffn_up.bias"))),
                   m.get(b + "ffn_down.weight"), m.opt(b + "ffn_down.bias"))
        if hp.get("jepa.enc.layer_scale"):
            f = f * m.get(b + "ls2")
        x = x + f
    return layer_norm(x, m.get("enc.norm.weight"), m.get("enc.norm.bias"), eps)


def mlp2(m: Model, prefix: str, x: np.ndarray, act):
    return linear(act(linear(x, m.get(prefix + ".0.weight"), m.get(prefix + ".0.bias"))),
                  m.get(prefix + ".2.weight"), m.get(prefix + ".2.bias"))


def lewm_predictor_forward(m: Model, emb: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """emb [T, D] (projector outputs), actions [T, A] -> pred.proj(predictor(...)) [T, D]."""
    hp = m.hp
    D = hp["jepa.pred.embed_dim"]
    H = hp["jepa.pred.n_head"]
    eps = hp["jepa.pred.ln_eps"]
    aeps = hp["jepa.pred.adaln_eps"]
    act = act_fn(hp["jepa.pred.act"])
    aact = act_fn(hp["jepa.pred.action_act"])
    T = emb.shape[0]
    c = mlp2(m, "pred.action_embed", actions, aact)  # [T, D]
    x = emb + m.get("pred.pos_embed")[:T]
    for i in range(hp["jepa.pred.n_layer"]):
        b = f"pred.blk.{i}."
        mod = linear(silu(c), m.get(b + "adaln.weight"), m.get(b + "adaln.bias"))
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = np.split(mod, 6, axis=-1)
        h = layer_norm(x, None, None, aeps) * (1 + sc_a) + sh_a
        h = layer_norm(h, m.get(b + "ln1.weight"), m.get(b + "ln1.bias"), eps)
        qkv = linear(h, m.get(b + "attn_qkv.weight"), m.opt(b + "attn_qkv.bias"))
        q, k, v = np.split(qkv, 3, axis=-1)
        a = linear(attention(q, k, v, H, causal=hp["jepa.pred.frame_causal"]),
                   m.get(b + "attn_out.weight"), m.get(b + "attn_out.bias"))
        x = x + g_a * a
        h = layer_norm(x, None, None, aeps) * (1 + sc_m) + sh_m
        h = layer_norm(h, m.get(b + "ln2.weight"), m.get(b + "ln2.bias"), eps)
        f = linear(act(linear(h, m.get(b + "ffn_up.weight"), m.get(b + "ffn_up.bias"))),
                   m.get(b + "ffn_down.weight"), m.get(b + "ffn_down.bias"))
        x = x + g_m * f
    x = layer_norm(x, m.get("pred.norm.weight"), m.get("pred.norm.bias"), eps)
    return mlp2(m, "pred.proj", x, act_fn(hp["jepa.pred.proj_act"]))


# ----------------------------------------------------------------------------- references
def load_images(n: int, size: int = 224):
    from PIL import Image
    files = sorted(MEDIA.glob("*.jpg"))[:n]
    if not files:
        raise SystemExit(f"no fixture images in {MEDIA}")
    return files


def preprocess(m: Model, path: Path) -> np.ndarray:
    """Apply jepa.pre.* with PIL (the same ops the HF processors use)."""
    from PIL import Image
    hp = m.hp
    im = Image.open(path).convert("RGB")
    rs = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[hp["jepa.pre.resample"]]
    short, crop = hp["jepa.pre.resize_short"], hp["jepa.pre.crop"]
    if hp["jepa.pre.resize_mode"] == "squash":
        im = im.resize((short, short), rs)
    else:
        w, h = im.size
        s = short / min(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), rs)
        w, h = im.size
        l, t = (w - crop) // 2, (h - crop) // 2
        im = im.crop((l, t, l + crop, t + crop))
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array(hp["jepa.pre.mean"], np.float32)) / np.array(hp["jepa.pre.std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def report(name, ours, ref, tol):
    ours = np.asarray(ours, np.float64).ravel()
    ref = np.asarray(ref, np.float64).ravel()
    cos = float(ours @ ref / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-30))
    mad = float(np.abs(ours - ref).max())
    rel = mad / (float(np.abs(ref).max()) + 1e-30)
    ok = rel <= tol
    print(f"  {name:40s} cos={cos:.7f} max|diff|={mad:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def ref_hfvit(src: Path, m: Model):
    import torch, timm
    from functools import partial
    from safetensors.torch import load_file
    hp = m.hp
    net = timm.models.vision_transformer.VisionTransformer(
        img_size=hp["jepa.enc.img_size"], patch_size=hp["jepa.enc.patch_size"], embed_dim=hp["jepa.enc.embed_dim"],
        depth=hp["jepa.enc.n_layer"], num_heads=hp["jepa.enc.n_head"], mlp_ratio=hp["jepa.enc.ffn_dim"] / hp["jepa.enc.embed_dim"],
        qkv_bias=True, class_token=True, global_pool="", num_classes=0,
        norm_layer=partial(torch.nn.LayerNorm, eps=hp["jepa.enc.ln_eps"]))
    sd = load_file(str(src / "model.safetensors"))
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.") and "cva_module_proj" not in k}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    net.eval()

    def run(img):
        with torch.no_grad():
            return net.forward_features(torch.from_numpy(img)[None])[0].numpy()
    return run


def ref_ijepa(src: Path, m: Model):
    import torch
    from transformers import IJepaModel
    net = IJepaModel.from_pretrained(str(src), torch_dtype=torch.float32).eval()

    def run(img):
        with torch.no_grad():
            return net(pixel_values=torch.from_numpy(img)[None]).last_hidden_state[0].numpy()
    return run


def ref_lewm(src: Path, m: Model, lewm_src: Path | None):
    import torch
    from transformers import ViTConfig, ViTModel
    hp = m.hp
    sd = torch.load(str(src / "weights.pt"), map_location="cpu", weights_only=False)
    cfg = ViTConfig(hidden_size=hp["jepa.enc.embed_dim"], num_hidden_layers=hp["jepa.enc.n_layer"],
                    num_attention_heads=hp["jepa.enc.n_head"], intermediate_size=hp["jepa.enc.ffn_dim"],
                    image_size=hp["jepa.enc.img_size"], patch_size=hp["jepa.enc.patch_size"])
    vit = ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)
    ren = {}
    for k, v in sd.items():
        if not k.startswith("encoder."):
            continue
        k = k[len("encoder."):]
        k = (k.replace("encoder.layer.", "layers.")
              .replace("attention.attention.query", "attention.q_proj")
              .replace("attention.attention.key", "attention.k_proj")
              .replace("attention.attention.value", "attention.v_proj")
              .replace("attention.output.dense", "attention.o_proj")
              .replace("intermediate.dense", "mlp.fc1")
              .replace("output.dense", "mlp.fc2"))
        ren[k] = v
    missing, unexpected = vit.load_state_dict(ren, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    vit.eval()

    def run_enc(img):
        with torch.no_grad():
            return vit(pixel_values=torch.from_numpy(img)[None], interpolate_pos_encoding=True).last_hidden_state[0].numpy()

    if lewm_src is None:
        return run_enc, None, None
    # load module.py directly: the package __init__ pulls in gymnasium & co. which we do not need
    import importlib.util
    mod_path = Path(lewm_src) / "stable_worldmodel" / "wm" / "lewm" / "module.py"
    spec = importlib.util.spec_from_file_location("lewm_module", mod_path)
    lewm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lewm_module)
    Predictor, Embedder, MLP = lewm_module.Predictor, lewm_module.Embedder, lewm_module.MLP
    import torch.nn as nn
    proj = MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
    pproj = MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
    pred = Predictor(num_frames=hp["jepa.pred.n_frames"], depth=hp["jepa.pred.n_layer"], heads=hp["jepa.pred.n_head"],
                     mlp_dim=hp["jepa.pred.ffn_dim"], input_dim=192, hidden_dim=192, output_dim=192,
                     dim_head=hp["jepa.pred.head_dim"], dropout=0.1)
    emb = Embedder(input_dim=hp["jepa.pred.action_dim"], emb_dim=192)
    for mod, pre in ((proj, "projector."), (pproj, "pred_proj."), (pred, "predictor."), (emb, "action_encoder.")):
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        mod.load_state_dict(sub, strict=True)
        mod.eval()

    def run_proj(cls):
        with torch.no_grad():
            return proj(torch.from_numpy(cls)).numpy()

    def run_pred(embs, actions):
        with torch.no_grad():
            x = pred(torch.from_numpy(embs)[None], emb(torch.from_numpy(actions)[None]))[0]
            return pproj(x).numpy()
    return run_enc, run_proj, run_pred


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--src", required=True, help="source checkpoint directory (for the PyTorch reference)")
    ap.add_argument("--lewm-src", default=None, help="checkout of stable-worldmodel (for the LeWM predictor reference)")
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--tol", type=float, default=None, help="max relative error (default 1e-4 f32 / 2e-2 f16)")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    m = Model(args.gguf)
    fam = m.hp["jepa.family"]
    types = m.tensor_types()
    is_q = any(k not in ("F32", "F16") for k in types)  # quantized by tools/jepa-quantize
    is_f16 = "F16" in types
    tol = args.tol if args.tol is not None else (1e-1 if is_q else 2e-2 if is_f16 else 1e-4)
    print(f"{args.gguf}: family={fam} tensor types={types} tol(rel)={tol}")
    src = Path(args.src)
    ok = True

    if fam == "hfvit":
        run = ref_hfvit(src, m)
        for f in load_images(args.n_images):
            img = preprocess(m, f)
            ours, ref = encoder_forward(m, img), run(img)
            ok &= report(f"{f.name} all tokens", ours, ref, tol)
            ok &= report(f"{f.name} CLS", ours[0], ref[0], tol)
    elif fam == "ijepa":
        run = ref_ijepa(src, m)
        for f in load_images(args.n_images):
            img = preprocess(m, f)
            ours, ref = encoder_forward(m, img), run(img)
            ok &= report(f"{f.name} all tokens", ours, ref, tol)
            ok &= report(f"{f.name} mean-pooled", ours.mean(0), ref.mean(0), tol)
    elif fam == "lewm":
        run_enc, run_proj, run_pred = ref_lewm(src, m, Path(args.lewm_src) if args.lewm_src else None)
        files = load_images(max(args.n_images, m.hp["jepa.pred.n_frames"]))
        cls_ours, cls_ref = [], []
        for f in files:
            img = preprocess(m, f)
            ours, ref = encoder_forward(m, img), run_enc(img)
            ok &= report(f"{f.name} all tokens", ours, ref, tol)
            cls_ours.append(ours[0]); cls_ref.append(ref[0])
        if run_proj is not None:
            cls_ours = np.stack(cls_ours); cls_ref = np.stack(cls_ref)
            emb_ours = mlp2(m, "enc.proj", cls_ours, act_fn(m.hp["jepa.enc.proj_act"]))
            emb_ref = run_proj(cls_ref)
            ok &= report("projector(CLS) [BN folded]", emb_ours, emb_ref, tol)
            T = m.hp["jepa.pred.n_frames"]
            rng = np.random.default_rng(0)
            actions = rng.uniform(-1, 1, size=(T, m.hp["jepa.pred.action_dim"])).astype(np.float32)
            pred_ours = lewm_predictor_forward(m, emb_ref[:T], actions)
            pred_ref = run_pred(emb_ref[:T], actions)
            ok &= report("predictor+pred_proj (3 frames)", pred_ours, pred_ref, tol)
            ok &= report("predictor last frame", pred_ours[-1], pred_ref[-1], tol)
            # causality check: changing the last action must not change earlier predictions
            actions2 = actions.copy(); actions2[-1] += 0.5
            p2 = lewm_predictor_forward(m, emb_ref[:T], actions2)
            ok &= report("frame-causal (frames 0..T-2 unchanged)", p2[:-1], pred_ours[:-1], 1e-6)
        else:
            print("  (pass --lewm-src to also verify projector / predictor / action embedder)")
    else:
        raise SystemExit(f"family {fam} not covered by this self-test")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
