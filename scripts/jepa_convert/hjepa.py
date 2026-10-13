"""jepa-world-models/h-jepa (H-JEPA / HWM / flat LeWM `*_object.ckpt`) -> jepa.cpp GGUF, family "hjepa".

Source: `torch.save(HJEPA)` of github.com/kevinghst/H-JEPA (arXiv 2610.06805). The file is a pickled
module object, so it is read with torch_object.py, which never runs the code the pickle names.
`HJEPA.jepas[k-1]` is level k (1-based below); every level is one `JEPA`:

  encoder         level 1  ProjectedEncoder(HF ViTModel, projector)                     "vit"
                           FusionEncoder(pixel = ProjectedEncoder(ViT, MLP),
                                         proprio = ProjectedEncoder(MLP, Identity))     "vit_proprio"
                  level 2+ ProjectedEncoder(ResidualLatentMLP, Identity)                "latent_mlp"
                           ProjectedEncoder(Identity, Identity)   (the HWM baseline)    "identity"
  action_encoder  level 1  Identity                                                     "none"
                  level 2+ SequenceEncoder  (1 pre-LN transformer layer over a chunk)   "seq"
                           SequenceMLPEncoder (flatten the chunk, 2-layer MLP)          "mlp"
  action_embed    Embedder (Conv1d(k=1) -> Linear -> SiLU -> Linear) or Identity
  predictor       ProjectedPredictor(ARPredictor, MLP)                                  "adaln"
                  CausalTransformerPredictor (DROID; own action embedder, no inner LN)  "adaln_bare"

Inference graph per level k (window_size 1, the only value the released checkpoints use):

  level 1   pix   = enc.proj( ViT(pixels)[CLS] )          # MLP (BatchNorm folded) | LayerNorm | identity
            s1    = pix                       ("vit")
            s1    = concat(pix, proprio_mlp(proprio))     ("vit_proprio")
  level k   sk_T  = enc_k( s{k-1}_{T*stride} )            # LN -> fc1 -> GELU -> fc2 [-> LN], or identity
  actions   e1_t  = act_embed_1( a_t )                    # a_t = frameskip stacked raw actions, z-scored
            ak_T  = act_enc_k( e{k-1}_{T*stride .. T*stride+stride-1} )   # the level's latent macro-action
            ek_T  = act_embed_k( ak_T )
  predict   x = in_proj( sk + pos[:T] ) ; c = cond_proj( ek )          ("adaln";  c = pred.act_embed(ek) for "adaln_bare")
            per block: six = adaln( silu(c) ); x += g_a * attn( [ln1]( LN0(x)*(1+sc_a)+sh_a ) ) ; same for the MLP half
            y = [out_proj]( norm(x) ) ; next = [proj]( y )             # causal over the T steps

Folds performed here (exact up to float rounding, checked by selftest.py against the authors' modules):
  * eval-mode BatchNorm1d of every `mlp` projector folded into the Linear before it
  * the Conv1d(k=1) of an Embedder folded into the Linear after it
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import JepaWriter, SourceTensors, check_unmapped, fold_batchnorm, fold_linear_pair, log
from .hfvit import map_hf_vit
from .torch_object import ModuleInfo, load_object, module_tree, root_attr

HF_REPO = "jepa-world-models/h-jepa"
LN_EPS = 1e-5      # nn.LayerNorm default: every affine LayerNorm outside the ViT
ADALN_EPS = 1e-6   # ConditionalBlock.norm1/norm2 (elementwise_affine=False)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _Src:
    """The checkpoint as (tensors, modules), with the assertions the converter leans on."""

    def __init__(self, path: Path):
        self.obj = load_object(path)
        tensors, self.mods = module_tree(self.obj)
        self.st = SourceTensors.from_dict({k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
                                           for k, v in tensors.items()})

    def mod(self, name: str) -> ModuleInfo:
        if name not in self.mods:
            raise SystemExit(f"checkpoint has no module {name!r}")
        return self.mods[name]

    def kind(self, name: str) -> str:
        return self.mods[name].kind if name in self.mods else "missing"

    def has(self, key: str) -> bool:
        return key in self.st

    def expect_kind(self, name: str, *kinds: str) -> str:
        k = self.kind(name)
        if k not in kinds:
            raise SystemExit(f"{name}: module is {k}, expected one of {kinds}")
        return k

    def expect_eps(self, name: str, eps: float, affine: bool = True) -> None:
        a = self.mod(name).attrs
        if self.kind(name) != "LayerNorm" or abs(float(a.get("eps", -1)) - eps) > eps * 1e-3 \
                or bool(a.get("elementwise_affine", True)) != affine:
            raise SystemExit(f"{name}: expected LayerNorm(eps={eps}, affine={affine}), got {self.kind(name)} {a}")

    def expect_gelu(self, name: str) -> None:
        k = self.kind(name)
        if k == "GELUActivation":
            return
        if k != "GELU" or self.mod(name).attrs.get("approximate", "none") != "none":
            raise SystemExit(f"{name}: expected GELU(approximate='none'), got {k} {self.mod(name).attrs}")


def _linear(src: _Src, key: str, n_in: int | None = None, n_out: int | None = None):
    w = src.st.take(key + ".weight")
    b = src.st.take(key + ".bias")
    if w.ndim != 2 or (n_in is not None and w.shape[1] != n_in) or (n_out is not None and w.shape[0] != n_out):
        raise SystemExit(f"{key}.weight has shape {w.shape}, expected [{n_out}, {n_in}]")
    return w, b


def _projector(src: _Src, w: JepaWriter, name: str, dst: str, n_in: int) -> tuple[str, int]:
    """`build_projector` output at module `name` -> `dst.*`. Returns (kind, output dim)."""
    kind = src.kind(name)
    if kind == "Identity":
        return "identity", n_in
    if kind == "LayerNorm":
        src.expect_eps(name, LN_EPS)
        w.add_norm(dst + "_ln", src.st.take(name + ".weight"), src.st.take(name + ".bias"))
        return "layernorm", n_in
    if kind != "MLP":
        raise SystemExit(f"{name}: unsupported projector module {kind}")
    src.expect_kind(name + ".net.1", "BatchNorm1d")
    src.expect_gelu(name + ".net.2")
    src.expect_kind(name + ".final_norm", "Identity")
    bn = src.mod(name + ".net.1").attrs
    w0, b0 = _linear(src, name + ".net.0", n_in=n_in)
    wf, bf = fold_batchnorm(w0, b0, src.st.take(name + ".net.1.weight"), src.st.take(name + ".net.1.bias"),
                            src.st.take(name + ".net.1.running_mean"), src.st.take(name + ".net.1.running_var"),
                            float(bn.get("eps", 1e-5)))
    src.st.drop(name + ".net.1.num_batches_tracked")
    w3, b3 = _linear(src, name + ".net.3", n_in=w0.shape[0])
    w.add_linear(dst + ".0", wf, bf, True)
    w.add_linear(dst + ".2", w3, b3, True)
    return "mlp", int(w3.shape[0])


def _embedder(src: _Src, w: JepaWriter, name: str, dst: str) -> tuple[int, int]:
    """Embedder -> `dst.0` / `dst.2` (SiLU between). Returns (input dim, output dim)."""
    src.expect_kind(name + ".embed.1", "SiLU")
    e0w, e0b = _linear(src, name + ".embed.0")
    if src.has(name + ".patch_embed.weight"):
        cw = src.st.take(name + ".patch_embed.weight")   # Conv1d(k=1): [smoothed, in, 1]
        cb = src.st.take(name + ".patch_embed.bias")
        if cw.ndim != 3 or cw.shape[2] != 1 or cw.shape[0] != e0w.shape[1]:
            raise SystemExit(f"{name}.patch_embed.weight has shape {cw.shape}")
        e0w, e0b = fold_linear_pair(cw[:, :, 0], cb, e0w, e0b)
    e2w, e2b = _linear(src, name + ".embed.2", n_in=e0w.shape[0])
    w.add_linear(dst + ".0", e0w, e0b, False)
    w.add_linear(dst + ".2", e2w, e2b, False)
    return int(e0w.shape[1]), int(e2w.shape[0])


def _vit(src: _Src, w: JepaWriter, name: str) -> dict:
    """HF ViTModel at module `name` -> enc.* and the jepa.enc.* keys."""
    p = name + "."
    L = src.st.layer_count(p + "layers.{i}.layernorm_before.weight") or \
        src.st.layer_count(p + "encoder.layer.{i}.layernorm_before.weight")
    pw = src.st.peek(p + "embeddings.patch_embeddings.projection.weight")
    D, C, P = int(pw.shape[0]), int(pw.shape[1]), int(pw.shape[2])
    emb = src.mod(name + ".embeddings").attrs
    img = emb.get("image_size", (0, 0))
    img = int(img[0] if isinstance(img, (tuple, list)) else img)
    n_pos = src.st.peek(p + "embeddings.position_embeddings").shape[1]
    if pw.shape[2] != pw.shape[3] or img <= 0 or n_pos != 1 + (img // P) ** 2:
        raise SystemExit(f"{name}: patch {tuple(pw.shape)} / image {img} / {n_pos} position rows do not agree")
    attn = name + (".layers.0.attention" if src.has(p + "layers.0.attention.q_proj.weight")
                   else ".encoder.layer.0.attention.attention")
    H = int(src.mod(attn).attrs.get("num_attention_heads", 0))
    if H <= 0 or D % H:
        raise SystemExit(f"{attn}: num_attention_heads={H} does not divide {D}")
    src.expect_eps(name + ".layernorm", 1e-12)
    F = int(src.st.peek(p + ("layers.0.mlp.fc1.weight" if src.has(p + "layers.0.mlp.fc1.weight")
                             else "encoder.layer.0.intermediate.dense.weight")).shape[0])
    map_hf_vit(src.st, w, p, n_layer=L, embed_dim=D, ffn_dim=F)
    return {
        "jepa.enc.embed_dim": D, "jepa.enc.n_layer": L, "jepa.enc.n_head": H, "jepa.enc.ffn_dim": F,
        "jepa.enc.patch_size": P, "jepa.enc.tubelet_size": 1, "jepa.enc.img_size": img, "jepa.enc.n_frames": 1,
        "jepa.enc.in_chans": C, "jepa.enc.ln_eps": 1e-12, "jepa.enc.act": "gelu_erf", "jepa.enc.pos_type": "learned",
        "jepa.enc.cls_token": True, "jepa.enc.n_registers": 0, "jepa.enc.qkv_fused": True, "jepa.enc.layer_scale": False,
    }


def _level1_encoder(src: _Src, w: JepaWriter, name: str, lv: dict) -> dict:
    kind = src.expect_kind(name, "ProjectedEncoder", "FusionEncoder")
    if kind == "ProjectedEncoder":
        src.expect_kind(name + ".encoder", "ViTModel")
        hp = _vit(src, w, name + ".encoder")
        pk, d = _projector(src, w, name + ".projector", "lvl.1.enc.proj", hp["jepa.enc.embed_dim"])
        lv.update(enc_kind="vit", proj_kind=pk, embed_dim=d, pixel_dim=d, proprio_dim=0)
        return hp
    # FusionEncoder: pixel stream + proprio stream, concatenated
    px, pr = name + ".pixel_encoder", name + ".proprio_encoder"
    src.expect_kind(px, "ProjectedEncoder")
    src.expect_kind(px + ".encoder", "ViTModel")
    src.expect_kind(pr, "ProjectedEncoder")
    src.expect_kind(pr + ".projector", "Identity")
    src.expect_kind(name + ".projector", "Identity")
    hp = _vit(src, w, px + ".encoder")
    pk, d_pix = _projector(src, w, px + ".projector", "lvl.1.enc.proj", hp["jepa.enc.embed_dim"])
    m = pr + ".encoder"                       # MLP: Linear -> LayerNorm -> GELU -> Linear
    src.expect_kind(m, "MLP")
    src.expect_eps(m + ".net.1", LN_EPS)
    src.expect_gelu(m + ".net.2")
    src.expect_kind(m + ".final_norm", "Identity")
    w0, b0 = _linear(src, m + ".net.0")
    w3, b3 = _linear(src, m + ".net.3", n_in=w0.shape[0])
    w.add_linear("lvl.1.enc.proprio.0", w0, b0, False)
    w.add_norm("lvl.1.enc.proprio.ln", src.st.take(m + ".net.1.weight"), src.st.take(m + ".net.1.bias"))
    w.add_linear("lvl.1.enc.proprio.2", w3, b3, False)
    lv.update(enc_kind="vit_proprio", proj_kind=pk, embed_dim=d_pix + int(w3.shape[0]), pixel_dim=d_pix,
              proprio_dim=int(w0.shape[1]))
    return hp


def _upper_encoder(src: _Src, w: JepaWriter, name: str, k: int, d_in: int, lv: dict) -> None:
    src.expect_kind(name, "ProjectedEncoder")
    src.expect_kind(name + ".projector", "Identity")
    kind = src.expect_kind(name + ".encoder", "ResidualLatentMLP", "Identity")
    if kind == "Identity":
        lv.update(enc_kind="identity", embed_dim=d_in)
        return
    m = name + ".encoder"
    if src.mod(m).attrs.get("use_residual", False):
        raise SystemExit(f"{m}: use_residual=True is not supported")
    src.expect_eps(m + ".input_norm", LN_EPS)
    src.expect_gelu(m + ".act")
    p = f"lvl.{k}.enc."
    w.add_norm(p + "in_norm", src.st.take(m + ".input_norm.weight"), src.st.take(m + ".input_norm.bias"))
    w1, b1 = _linear(src, m + ".fc1", n_in=d_in)
    w2, b2 = _linear(src, m + ".fc2", n_in=w1.shape[0])
    w.add_linear(p + "fc1", w1, b1, True)
    w.add_linear(p + "fc2", w2, b2, True)
    if src.kind(m + ".final_norm") == "LayerNorm":
        src.expect_eps(m + ".final_norm", LN_EPS)
        w.add_norm(p + "out_norm", src.st.take(m + ".final_norm.weight"), src.st.take(m + ".final_norm.bias"))
    else:
        src.expect_kind(m + ".final_norm", "Identity")
    lv.update(enc_kind="latent_mlp", embed_dim=int(w2.shape[0]))


def _action_encoder(src: _Src, w: JepaWriter, name: str, k: int, d_in: int, stride: int, lv: dict) -> None:
    kind = src.expect_kind(name, "SequenceEncoder", "SequenceMLPEncoder")
    p = f"lvl.{k}.act_enc."
    if kind == "SequenceMLPEncoder":
        m = name + ".net"                     # MLP(norm_fn=None, act=ReLU): Linear -> Identity -> ReLU -> Linear
        src.expect_kind(m + ".net.1", "Identity")
        src.expect_kind(m + ".net.2", "ReLU")
        w0, b0 = _linear(src, m + ".net.0", n_in=stride * d_in)
        w3, b3 = _linear(src, m + ".net.3", n_in=w0.shape[0])
        w.add_linear(p + "fc1", w0, b0, False)
        w.add_linear(p + "fc2", w3, b3, False)
        if src.kind(m + ".final_norm") == "LayerNorm":
            src.expect_eps(m + ".final_norm", LN_EPS)
            w.add_norm(p + "out_norm", src.st.take(m + ".final_norm.weight"), src.st.take(m + ".final_norm.bias"))
        lv.update(act_enc_kind="mlp", action_dim=int(w3.shape[0]), act_enc_dim=int(w0.shape[0]),
                  act_enc_n_layer=0, act_enc_n_head=0, act_enc_max_chunk=stride)
        return
    a = src.mod(name).attrs
    for flag, want in (("stochastic", False), ("use_cls", True), ("use_step_mlp", True)):
        if a.get(flag, want) != want:
            raise SystemExit(f"{name}: {flag}={a[flag]} is not supported")
    sw, sb = _linear(src, name + ".step_mlp.fc1", n_in=d_in)
    d = int(sw.shape[0])
    pos = src.st.take(name + ".pos")
    max_chunk = int(a.get("max_chunk", pos.shape[0] - 1))
    if pos.shape != (max_chunk + 1, d) or stride > max_chunk:
        raise SystemExit(f"{name}.pos has shape {pos.shape}; max_chunk={max_chunk}, stride={stride}")
    w.add_linear(p + "step", sw, sb, False)
    w.add_tensor(p + "pos", pos, False)       # row 0 = CLS token, rows 1.. = step positions
    n_layer = src.st.layer_count(name + ".encoder.layers.{i}.self_attn.in_proj_weight")
    if src.mod(name + ".encoder").attrs.get("num_layers", n_layer) != n_layer or n_layer < 1:
        raise SystemExit(f"{name}.encoder: layer count mismatch")
    if src.kind(name + ".encoder.norm") not in ("missing", "NoneType"):
        raise SystemExit(f"{name}.encoder has a final norm; the graph has none")
    n_head = 0
    for i in range(n_layer):
        s = f"{name}.encoder.layers.{i}"
        la = src.mod(s).attrs
        if not la.get("norm_first", False) or la.get("activation_relu_or_gelu", 1) != 1:
            raise SystemExit(f"{s}: expected a norm_first TransformerEncoderLayer with ReLU, got {la}")
        ma = src.mod(s + ".self_attn").attrs
        n_head = int(ma["num_heads"])
        if int(ma["embed_dim"]) != d or d % n_head or not ma.get("batch_first", False) or ma.get("add_zero_attn", False):
            raise SystemExit(f"{s}.self_attn: unsupported configuration {ma}")
        src.expect_eps(s + ".norm1", LN_EPS)
        src.expect_eps(s + ".norm2", LN_EPS)
        b = f"{p}blk.{i}."
        w.add_norm(b + "ln1", src.st.take(s + ".norm1.weight"), src.st.take(s + ".norm1.bias"))
        qkv = src.st.take(s + ".self_attn.in_proj_weight")
        if qkv.shape != (3 * d, d):
            raise SystemExit(f"{s}.self_attn.in_proj_weight has shape {qkv.shape}")
        w.add_linear(b + "attn_qkv", qkv, src.st.take(s + ".self_attn.in_proj_bias"), False)
        ow, ob = _linear(src, s + ".self_attn.out_proj", n_in=d, n_out=d)
        w.add_linear(b + "attn_out", ow, ob, False)
        w.add_norm(b + "ln2", src.st.take(s + ".norm2.weight"), src.st.take(s + ".norm2.bias"))
        uw, ub = _linear(src, s + ".linear1", n_in=d)
        dw, db = _linear(src, s + ".linear2", n_in=uw.shape[0], n_out=d)
        w.add_linear(b + "ffn_up", uw, ub, False)
        w.add_linear(b + "ffn_down", dw, db, False)
    # head: Linear -> GELU -> LayerNorm -> Linear, then final_ln (eps 1e-6)
    src.expect_gelu(name + ".head.1")
    src.expect_eps(name + ".head.2", LN_EPS)
    src.expect_eps(name + ".final_ln", 1e-6)
    h0w, h0b = _linear(src, name + ".head.0", n_in=d)
    h3w, h3b = _linear(src, name + ".head.3", n_in=h0w.shape[0])
    w.add_linear(p + "head.0", h0w, h0b, False)
    w.add_norm(p + "head.ln", src.st.take(name + ".head.2.weight"), src.st.take(name + ".head.2.bias"))
    w.add_linear(p + "head.2", h3w, h3b, False)
    w.add_norm(p + "out_norm", src.st.take(name + ".final_ln.weight"), src.st.take(name + ".final_ln.bias"))
    lv.update(act_enc_kind="seq", action_dim=int(h3w.shape[0]), act_enc_dim=d, act_enc_n_layer=n_layer,
              act_enc_n_head=n_head, act_enc_max_chunk=max_chunk)


def _predictor(src: _Src, w: JepaWriter, name: str, k: int, d_state: int, d_cond: int, lv: dict) -> None:
    kind = src.expect_kind(name, "ProjectedPredictor", "CausalTransformerPredictor")
    p = f"lvl.{k}.pred."
    bare = kind == "CausalTransformerPredictor"
    core = name if bare else name + ".predictor"
    if not bare:
        src.expect_kind(core, "ARPredictor")
    tr = core + ".transformer"
    pos = src.st.take(core + ".pos_embedding")
    if pos.ndim != 3 or pos.shape[2] != d_state:
        raise SystemExit(f"{core}.pos_embedding has shape {pos.shape}, the level's state is {d_state}-d")
    w.add_tensor(p + "pos_embed", pos.reshape(-1, d_state), False)
    if bare:   # the predictor embeds the raw (latent) action itself: Linear -> SiLU -> Linear
        src.expect_kind(core + ".action_embedder.1", "SiLU")
        a0w, a0b = _linear(src, core + ".action_embedder.0", n_in=d_cond)
        a2w, a2b = _linear(src, core + ".action_embedder.2", n_in=a0w.shape[0], n_out=d_state)
        w.add_linear(p + "act_embed.0", a0w, a0b, False)
        w.add_linear(p + "act_embed.2", a2w, a2b, False)
        d_cond = d_state
    elif d_cond != d_state:
        raise SystemExit(f"{name}: action embedding is {d_cond}-d but the state is {d_state}-d")
    D = int(src.st.peek(tr + ".norm.weight").shape[0])       # transformer width
    for nm in ("input_proj", "cond_proj"):
        if src.kind(f"{tr}.{nm}") == "Linear":
            pw_, pb_ = _linear(src, f"{tr}.{nm}", n_in=d_state, n_out=D)
            w.add_linear(p + ("in_proj" if nm == "input_proj" else "cond_proj"), pw_, pb_, True)
        elif D != d_state:
            raise SystemExit(f"{tr}.{nm} is missing but the transformer is {D}-d and the state {d_state}-d")
    n_layer = src.st.layer_count(tr + ".layers.{i}.attn.to_qkv.weight")
    heads = int(src.mod(tr + ".layers.0.attn").attrs["heads"])
    inner = int(src.st.peek(tr + ".layers.0.attn.to_qkv.weight").shape[0]) // 3
    if n_layer < 1 or heads <= 0 or inner % heads:
        raise SystemExit(f"{tr}: {n_layer} layers, {heads} heads, attention width {inner}")
    ffn = 0
    for i in range(n_layer):
        s = f"{tr}.layers.{i}"
        b = f"{p}blk.{i}."
        src.expect_eps(s + ".norm1", ADALN_EPS, affine=False)
        src.expect_eps(s + ".norm2", ADALN_EPS, affine=False)
        src.expect_kind(s + ".adaLN_modulation.0", "SiLU")
        aw, ab = _linear(src, s + ".adaLN_modulation.1", n_in=D, n_out=6 * D)
        w.add_linear(b + "adaln", aw, ab, False)
        if int(src.mod(s + ".attn").attrs["heads"]) != heads:
            raise SystemExit(f"{s}.attn: head count differs between layers")
        if not bare:
            src.expect_eps(s + ".attn.norm", LN_EPS)
            w.add_norm(b + "ln1", src.st.take(s + ".attn.norm.weight"), src.st.take(s + ".attn.norm.bias"))
        qkv = src.st.take(s + ".attn.to_qkv.weight")
        if qkv.shape != (3 * inner, D) or src.has(s + ".attn.to_qkv.bias"):
            raise SystemExit(f"{s}.attn.to_qkv: shape {qkv.shape} / unexpected bias")
        w.add_linear(b + "attn_qkv", qkv, None, True)
        ow, ob = _linear(src, s + ".attn.to_out.0", n_in=inner, n_out=D)
        w.add_linear(b + "attn_out", ow, ob, True)
        up, dn = (".mlp.net.0", ".mlp.net.3") if bare else (".mlp.net.1", ".mlp.net.4")
        src.expect_gelu(s + (".mlp.net.1" if bare else ".mlp.net.2"))
        if not bare:
            src.expect_eps(s + ".mlp.net.0", LN_EPS)
            w.add_norm(b + "ln2", src.st.take(s + ".mlp.net.0.weight"), src.st.take(s + ".mlp.net.0.bias"))
        uw, ub = _linear(src, s + up, n_in=D)
        dw, db = _linear(src, s + dn, n_in=uw.shape[0], n_out=D)
        ffn = int(uw.shape[0])
        w.add_linear(b + "ffn_up", uw, ub, True)
        w.add_linear(b + "ffn_down", dw, db, True)
    src.expect_eps(tr + ".norm", LN_EPS)
    w.add_norm(p + "norm", src.st.take(tr + ".norm.weight"), src.st.take(tr + ".norm.bias"))
    d_out = D
    if src.kind(tr + ".output_proj") == "Linear":
        ow, ob = _linear(src, tr + ".output_proj", n_in=D)
        w.add_linear(p + "out_proj", ow, ob, True)
        d_out = int(ow.shape[0])
    if not bare:
        _, d_out = _projector(src, w, name + ".projector", p + "proj", d_out)
    if d_out != d_state:
        raise SystemExit(f"{name}: predicts {d_out}-d states, the level's state is {d_state}-d")
    lv.update(pred_kind="adaln_bare" if bare else "adaln", pred_dim=D, pred_n_layer=n_layer, pred_n_head=heads,
              pred_head_dim=inner // heads, pred_ffn_dim=ffn, pred_n_frames=int(pos.shape[1]))


def default_name(src: Path) -> str:
    """ant_hjepa_l3_object.ckpt under .../seed42/ -> "ant_hjepa_l3-seed42" (the authors' run name + seed)."""
    stem = src.stem[: -len("_object")] if src.stem.endswith("_object") else src.stem
    seed = src.parent.name if src.parent.name.startswith("seed") else ""
    return stem + (f"-{seed}" if seed else "")


def convert(src: str | Path, out: str | Path, ftype: str, *, name: str | None = None,
            allow_unmapped: bool = False) -> Path:
    src = Path(src)
    if src.is_dir():
        found = sorted(src.glob("*_object.ckpt"))
        if len(found) != 1:
            raise SystemExit(f"{src}: expected exactly one *_object.ckpt, found {len(found)}")
        src = found[0]
    s = _Src(src)
    if s.kind("") != "HJEPA" or s.kind("jepas") != "ModuleList":
        raise SystemExit(f"{src}: root module is {s.kind('')}, expected the H-JEPA `HJEPA` container")
    n_levels = len(s.mod("jepas").children)
    model_name = name or default_name(src)
    flat = n_levels == 1
    w = JepaWriter(out, ftype, name=model_name, family="hjepa", modality="image", license="mit",
                   source_url=f"https://huggingface.co/{HF_REPO}",
                   description=("flat LeWM baseline (1 level)" if flat else f"{n_levels}-level hierarchical JEPA world model")
                               + " from H-JEPA (arXiv 2610.06805)")

    levels: list[dict] = []
    enc_hp: dict = {}
    for k in range(1, n_levels + 1):
        j = f"jepas.{k - 1}"
        a = s.mod(j).attrs
        s.expect_kind(j, "JEPA")
        if int(a.get("level", k)) != k:
            raise SystemExit(f"{j}: level attribute is {a.get('level')}, expected {k}")
        stride, window = int(a.get("temporal_stride", 1)), int(a.get("temporal_window_size", 1))
        if window != 1:
            raise SystemExit(f"{j}: temporal_window_size={window}; only 1 is supported (no released checkpoint uses more)")
        lv: dict = {"stride": stride, "window": window, "target_length": int(a.get("target_length") or 0)}
        below = levels[-1] if levels else None
        if k == 1:
            if stride != 1:
                raise SystemExit(f"{j}: level 1 has temporal_stride={stride}")
            enc_hp = _level1_encoder(s, w, j + ".encoder", lv)
            s.expect_kind(j + ".action_encoder", "Identity")
            lv.update(act_enc_kind="none", act_enc_dim=0, act_enc_n_layer=0, act_enc_n_head=0, act_enc_max_chunk=0)
        else:
            _upper_encoder(s, w, j + ".encoder", k, below["embed_dim"], lv)
            _action_encoder(s, w, j + ".action_encoder", k, below["act_embed_dim"], stride, lv)
        # action_embed: what the level's predictor is conditioned on and what the level above pools
        if s.kind(j + ".action_embed") == "Embedder":
            d_in, d_emb = _embedder(s, w, j + ".action_embed", f"lvl.{k}.act_embed")
            if k == 1:
                lv["action_dim"] = d_in
            elif d_in != lv["action_dim"]:
                raise SystemExit(f"{j}.action_embed reads {d_in}-d actions, the action encoder emits {lv['action_dim']}-d")
            lv["act_embed_dim"] = d_emb
        else:
            s.expect_kind(j + ".action_embed", "Identity")
            if k == 1:   # the action width is then whatever the predictor's own embedder reads
                lv["action_dim"] = int(s.st.peek(j + ".predictor.action_embedder.0.weight").shape[1])
            lv["act_embed_dim"] = lv["action_dim"]
        _predictor(s, w, j + ".predictor", k, lv["embed_dim"], lv["act_embed_dim"], lv)
        q = s.st.take_opt(j + ".latent_action_queue")
        if q is not None and q.ndim == 2 and q.shape[0] > 0:
            if q.shape[1] != lv["action_dim"]:
                raise SystemExit(f"{j}.latent_action_queue has shape {q.shape}")
            w.add_tensor(f"lvl.{k}.act_queue", q, False)   # latent actions seen in training: the planner's prior
        levels.append(lv)

    # training normaliser: z-score statistics of the action / proprio columns (std ddof 1)
    norm = (root_attr(s.obj, "normalizer_artifact") or {}).get("stats") or {}
    raw_action_dim = 0
    for col in ("action", "proprio"):
        st = norm.get(col)
        if not isinstance(st, dict) or not isinstance(st.get("mean"), np.ndarray):
            continue
        mean, std = st["mean"].reshape(-1), st["std"].reshape(-1)
        w.add_tensor(f"norm.{col}.mean", mean, False)
        w.add_tensor(f"norm.{col}.std", std, False)
        if col == "action":
            raw_action_dim = int(mean.shape[0])
        elif levels[0]["proprio_dim"] not in (0, int(mean.shape[0])):
            raise SystemExit(f"proprio statistics are {mean.shape[0]}-d, the encoder reads {levels[0]['proprio_dim']}-d")
    l1 = levels[0]
    frameskip = l1["action_dim"] // raw_action_dim if raw_action_dim and l1["action_dim"] % raw_action_dim == 0 else 0

    img = enc_hp["jepa.enc.img_size"]
    hp = dict(enc_hp)
    hp.update({
        "jepa.head.kind": "none",
        "jepa.pre.mean": IMAGENET_MEAN, "jepa.pre.std": IMAGENET_STD,
        "jepa.pre.resize_short": img, "jepa.pre.crop": img,
        "jepa.pre.resample": "bilinear", "jepa.pre.resize_mode": "squash",
        "jepa.hjepa.n_levels": n_levels,
        "jepa.hjepa.ln_eps": LN_EPS,
        "jepa.hjepa.adaln_eps": ADALN_EPS,
        "jepa.hjepa.proj_kind": l1["proj_kind"],
        "jepa.hjepa.pixel_dim": l1["pixel_dim"],
        "jepa.hjepa.proprio_dim": l1["proprio_dim"],
        "jepa.hjepa.raw_action_dim": raw_action_dim,
        "jepa.hjepa.frameskip": frameskip,
    })
    for key in ("embed_dim", "stride", "window", "target_length", "action_dim", "act_embed_dim", "act_enc_dim",
                "act_enc_n_layer", "act_enc_n_head", "act_enc_max_chunk", "pred_dim", "pred_n_layer", "pred_n_head",
                "pred_head_dim", "pred_ffn_dim", "pred_n_frames"):
        hp[f"jepa.hjepa.{key}"] = [int(lv[key]) for lv in levels]
    for key in ("enc_kind", "act_enc_kind", "pred_kind"):
        hp[f"jepa.hjepa.{key}"] = [str(lv[key]) for lv in levels]
    w.add_hparams(hp)
    for k, lv in enumerate(levels, 1):
        log(f"hjepa: level {k}: state {lv['embed_dim']}-d ({lv['enc_kind']}), stride {lv['stride']}, "
            f"action {lv['action_dim']}-d ({lv['act_enc_kind']}) -> cond {lv['act_embed_dim']}-d, "
            f"predictor {lv['pred_kind']} D={lv['pred_dim']} L={lv['pred_n_layer']} "
            f"H={lv['pred_n_head']}x{lv['pred_head_dim']} F={lv['pred_ffn_dim']} frames={lv['pred_n_frames']}")
    log(f"hjepa: encoder ViT D={hp['jepa.enc.embed_dim']} L={hp['jepa.enc.n_layer']} P={hp['jepa.enc.patch_size']} "
        f"img={img} proj={l1['proj_kind']} proprio={l1['proprio_dim']} | raw action {raw_action_dim} x frameskip {frameskip}")

    # training-only modules: the inverse-dynamics head of the DROID runs and the pixel decoder some
    # runs kept from the probing evaluation (it reads states, nothing reads it)
    skip = tuple(f"jepas.{i}.{mod}" for i in range(n_levels)
                 for mod in ("idm_loss_fn.", "decoder_level")) + ("decoder_level",)   # also registered on the root
    check_unmapped(s.st, skip, allow_unmapped)
    return w.write()
