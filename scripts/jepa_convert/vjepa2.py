#!/usr/bin/env python3
"""HF V-JEPA 2 -> jepa.cpp GGUF converter (standalone).

Handles both HuggingFace checkpoints of the V-JEPA 2 family:

  * ``VJEPA2Model``                   (facebook/vjepa2-vitl-fpc64-256, ...):  encoder + predictor
  * ``VJEPA2ForVideoClassification``  (facebook/vjepa2-vitl-fpc16-256-ssv2, ...): encoder + predictor
                                       + attentive pooler + linear classifier

Usage:
    python scripts/jepa_convert/vjepa2.py --src models/facebook/vjepa2-vitl-fpc64-256 \
        --out models/gguf/vjepa2-vitl-fpc64-256-f16.gguf --ftype f16
    python scripts/jepa_convert/vjepa2.py --info models/gguf/vjepa2-vitl-fpc64-256-f16.gguf

Tensor names / metadata follow docs/gguf-schema.md.  See VJEPA_NOTES.md (same directory) for the
exact key mapping, the 3-D RoPE convention and everything the C++ side must know.

Only numpy, safetensors and the ``gguf`` python package are imported.  ``convert(src, out, ftype)``
is exposed at module level so a package-level dispatcher can call it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import numpy as np
from safetensors import safe_open

import gguf
from gguf import GGMLQuantizationType

SCHEMA_VERSION = 1

# ggml "file type" (general.file_type) values, same numbering as llama.cpp
FTYPES = {"f32": 0, "f16": 1, "q8_0": 7}

# PIL resampling ids as stored in HF preprocessor configs
PIL_RESAMPLE = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}

# Only these weights may be quantized (docs/gguf-schema.md "Quantization rules").
QUANTIZABLE = re.compile(
    r"(\.attn_(qkv|q|k|v|out)\.weight"
    r"|\.ffn_(up|down)\.weight"
    r"|^pred\.proj(_context)?\.weight"
    r"|^head\.cls\.weight"
    r"|^head\.xattn\.(q|k|v)\.weight)$"
)


# --------------------------------------------------------------------------------------------
# small GGUF writer wrapper implementing the dtype policy
# --------------------------------------------------------------------------------------------
class JepaGGUFWriter:
    """dtype policy:
      f32  : everything F32
      f16  : ``*.weight`` tensors with ndim >= 2 -> F16, everything else (biases, norms, tokens,
             modality vectors, mask tokens, pooler query) -> F32
      q8_0 : QUANTIZABLE weights -> Q8_0 (if the row length is a multiple of 32), other 2-D
             weights -> F16, rest F32
    """

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

    # ---- metadata ----
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

    # ---- tensors ----
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
            dt = "Q8_0"
            nbytes = q.nbytes
        elif self.ftype in ("f16", "q8_0") and is_matrix_weight:
            arr = arr.astype(np.float16)
            self.gw.add_tensor(name, arr)
            dt = "F16"
            nbytes = arr.nbytes
        else:
            self.gw.add_tensor(name, arr)
            dt = "F32"
            nbytes = arr.nbytes
        self.tensor_names.append(name)
        self.dtype_counts[dt] = self.dtype_counts.get(dt, 0) + 1
        self.n_bytes += nbytes

    def close(self) -> None:
        self.gw.write_header_to_file()
        self.gw.write_kv_data_to_file()
        self.gw.write_tensors_to_file(progress=False)
        self.gw.close()


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _readme_license(src_dir: str, default: str = "mit") -> str:
    """license: field of the HF README front matter (falls back to `default`)."""
    p = os.path.join(src_dir, "README.md")
    if not os.path.exists(p):
        return default
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(4096)
    m = re.search(r"^license:\s*([A-Za-z0-9._+-]+)\s*$", head, flags=re.M)
    return m.group(1).strip().lower() if m else default


class SourceTensors:
    """All tensors of a safetensors file as numpy, with consumption tracking."""

    def __init__(self, path: str):
        self.path = path
        self.f = safe_open(path, framework="np")
        self.keys = set(self.f.keys())
        self.used: set[str] = set()

    def has(self, k: str) -> bool:
        return k in self.keys

    def get(self, k: str) -> np.ndarray:
        if k not in self.keys:
            raise KeyError(f"{k} not in {self.path}")
        self.used.add(k)
        return self.f.get_tensor(k)

    def unused(self) -> list[str]:
        return sorted(self.keys - self.used)


def _write_block(w: JepaGGUFWriter, st: SourceTensors, dst: str, src: str, names: dict[str, str]) -> None:
    """One pre-LN transformer block.

    `names` maps the abstract roles to the HF module names inside the layer:
      ln1, q, k, v, out, ln2, up, down   (q/k/v are fused into attn_qkv, rows [q;k;v]).
    """
    for role, gname in (("ln1", "ln1"), ("ln2", "ln2")):
        w.tensor(f"{dst}.{gname}.weight", st.get(f"{src}.{names[role]}.weight"))
        w.tensor(f"{dst}.{gname}.bias", st.get(f"{src}.{names[role]}.bias"))
    qw, kw, vw = (st.get(f"{src}.{names[r]}.weight") for r in ("q", "k", "v"))
    qb, kb, vb = (st.get(f"{src}.{names[r]}.bias") for r in ("q", "k", "v"))
    w.tensor(f"{dst}.attn_qkv.weight", np.concatenate([qw, kw, vw], axis=0))
    w.tensor(f"{dst}.attn_qkv.bias", np.concatenate([qb, kb, vb], axis=0))
    w.tensor(f"{dst}.attn_out.weight", st.get(f"{src}.{names['out']}.weight"))
    w.tensor(f"{dst}.attn_out.bias", st.get(f"{src}.{names['out']}.bias"))
    w.tensor(f"{dst}.ffn_up.weight", st.get(f"{src}.{names['up']}.weight"))
    w.tensor(f"{dst}.ffn_up.bias", st.get(f"{src}.{names['up']}.bias"))
    w.tensor(f"{dst}.ffn_down.weight", st.get(f"{src}.{names['down']}.weight"))
    w.tensor(f"{dst}.ffn_down.bias", st.get(f"{src}.{names['down']}.bias"))


VJEPA2_LAYER_NAMES = dict(ln1="norm1", q="attention.query", k="attention.key", v="attention.value",
                          out="attention.proj", ln2="norm2", up="mlp.fc1", down="mlp.fc2")
POOLER_LAYER_NAMES = dict(ln1="layer_norm1", q="self_attn.q_proj", k="self_attn.k_proj", v="self_attn.v_proj",
                          out="self_attn.out_proj", ln2="layer_norm2", up="mlp.fc1", down="mlp.fc2")


# --------------------------------------------------------------------------------------------
# main conversion
# --------------------------------------------------------------------------------------------
def convert(src: str, out: str, ftype: str = "f16", name: str | None = None,
            source_url: str | None = None, quiet: bool = False) -> dict[str, Any]:
    """Convert an HF V-JEPA 2 model directory `src` to GGUF file `out`.  Returns a summary dict."""
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr))

    src = os.path.abspath(src)
    cfg = _read_json(os.path.join(src, "config.json"))
    pre_path = os.path.join(src, "video_preprocessor_config.json")
    pre = _read_json(pre_path) if os.path.exists(pre_path) else {}
    st_path = os.path.join(src, "model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"{st_path} not found (sharded checkpoints are not supported)")
    st = SourceTensors(st_path)

    archs = cfg.get("architectures") or ["VJEPA2Model"]
    arch = archs[0]
    if cfg.get("model_type") != "vjepa2":
        raise ValueError(f"model_type={cfg.get('model_type')!r} is not vjepa2")
    if arch == "VJEPA2Model":
        prefix = ""
        has_head = False
    elif arch == "VJEPA2ForVideoClassification":
        prefix = "vjepa2."
        has_head = True
    else:
        raise ValueError(f"unsupported architecture {arch}")

    if cfg.get("use_SiLU", False):
        raise NotImplementedError("use_SiLU=True (SwiGLU FFN) is not covered by the schema yet")
    if cfg.get("hidden_act", "gelu") != "gelu":
        raise NotImplementedError(f"hidden_act={cfg['hidden_act']!r}")

    model_name = name or os.path.basename(src.rstrip("/"))
    source_url = source_url or f"https://huggingface.co/facebook/{model_name}"

    D = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    H = int(cfg["num_attention_heads"])
    ffn = int(D * float(cfg.get("mlp_ratio", 4.0)))
    P = int(cfg["patch_size"])
    T = int(cfg["tubelet_size"])
    C = int(cfg.get("in_chans", 3))
    crop = int(cfg.get("crop_size", cfg.get("image_size", 256)))
    n_frames = int(cfg["frames_per_clip"])
    eps = float(cfg.get("layer_norm_eps", 1e-6))

    Dp = int(cfg["pred_hidden_size"])
    Lp = int(cfg["pred_num_hidden_layers"])
    Hp = int(cfg["pred_num_attention_heads"])
    ffn_p = int(Dp * float(cfg.get("pred_mlp_ratio", 4.0)))
    n_mask = int(cfg["pred_num_mask_tokens"])

    if D % H or Dp % Hp:
        raise ValueError("hidden size not divisible by number of heads")

    log(f"[vjepa2] {arch}: enc D={D} L={L} H={H} ffn={ffn} patch={P} tubelet={T} crop={crop} frames={n_frames}"
        f" | pred D={Dp} L={Lp} H={Hp} ffn={ffn_p} mask_tokens={n_mask} | head={has_head}")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    w = JepaGGUFWriter(out, ftype)

    # ---------------- general ----------------
    w.str("general.name", model_name)
    w.str("general.license", _readme_license(src))
    w.str("general.source_url", source_url)
    w.str("general.description", f"V-JEPA 2 ({arch}) converted from HF safetensors by scripts/jepa_convert/vjepa2.py")
    w.u32("jepa.schema_version", SCHEMA_VERSION)
    w.str("jepa.family", "vjepa2")
    w.str("jepa.modality", "video")

    # ---------------- encoder ----------------
    w.u32("jepa.enc.embed_dim", D)
    w.u32("jepa.enc.n_layer", L)
    w.u32("jepa.enc.n_head", H)
    w.u32("jepa.enc.ffn_dim", ffn)
    w.u32("jepa.enc.patch_size", P)
    w.u32("jepa.enc.tubelet_size", T)
    w.u32("jepa.enc.img_size", crop)
    w.u32("jepa.enc.n_frames", n_frames)
    w.u32("jepa.enc.in_chans", C)
    w.f32("jepa.enc.ln_eps", eps)
    w.str("jepa.enc.act", "gelu_erf")
    w.str("jepa.enc.pos_type", "rope3d")
    w.f32("jepa.enc.rope_theta", 10000.0)
    w.bool("jepa.enc.rope_interpolate", False)
    # V-JEPA 2 (HF + Meta src/) tiles the per-axis cos/sin table: [f0..f_{d/2-1}, f0..f_{d/2-1}]
    # (the "subtle bug" kept for checkpoint compatibility).  See VJEPA_NOTES.md.
    w.str("jepa.enc.rope_freq_layout", "tiled")
    w.bool("jepa.enc.cls_token", False)
    w.u32("jepa.enc.n_registers", 0)
    w.bool("jepa.enc.qkv_fused", True)
    w.bool("jepa.enc.modality_embed", False)
    w.bool("jepa.enc.image_patch_embed", False)
    w.bool("jepa.enc.layer_scale", False)

    # ---------------- predictor ----------------
    w.str("jepa.pred.kind", "masked")
    w.u32("jepa.pred.embed_dim", Dp)
    w.u32("jepa.pred.n_layer", Lp)
    w.u32("jepa.pred.n_head", Hp)
    w.u32("jepa.pred.ffn_dim", ffn_p)
    w.u32("jepa.pred.n_mask_tokens", n_mask)
    w.u32("jepa.pred.out_dim", D)
    w.str("jepa.pred.rope_freq_layout", "tiled")

    # ---------------- head ----------------
    labels: list[str] = []
    if has_head:
        id2label = cfg.get("id2label") or {}
        n_classes = int(cfg.get("num_labels") or len(id2label))
        labels = [str(id2label[str(i)]) if str(i) in id2label else str(id2label.get(i, f"LABEL_{i}"))
                  for i in range(n_classes)]
        n_pool = int(cfg.get("num_pooler_layers", 3))
        w.str("jepa.head.kind", "attentive_pool")
        w.u32("jepa.head.n_classes", n_classes)
        w.u32("jepa.head.n_pool_layers", n_pool)
        w.arr("jepa.head.labels", labels)
    else:
        w.str("jepa.head.kind", "none")

    # ---------------- preprocessing ----------------
    mean = pre.get("image_mean", [0.485, 0.456, 0.406])
    std = pre.get("image_std", [0.229, 0.224, 0.225])
    size = pre.get("size", {})
    resize_short = int(size.get("shortest_edge", int(crop * 256 / 224)))
    crop_cfg = pre.get("crop_size", crop)
    crop_px = int(crop_cfg["height"] if isinstance(crop_cfg, dict) else crop_cfg)
    resample = PIL_RESAMPLE.get(int(pre.get("resample", 2)), "bilinear")
    w.arr("jepa.pre.mean", [float(x) for x in mean])
    w.arr("jepa.pre.std", [float(x) for x in std])
    w.u32("jepa.pre.resize_short", resize_short)
    w.u32("jepa.pre.crop", crop_px)
    w.str("jepa.pre.resample", resample)
    w.f32("jepa.pre.rescale", float(pre.get("rescale_factor", 1.0 / 255.0)))

    # ============================ tensors ============================
    enc = prefix + "encoder"
    pw = st.get(f"{enc}.embeddings.patch_embeddings.proj.weight")  # [D, C, T, P, P]
    if pw.shape != (D, C, T, P, P):
        raise ValueError(f"unexpected patch embed shape {pw.shape}")
    w.tensor("enc.patch_embed.weight", pw.reshape(D, C * T * P * P))  # C-T-H-W flatten order
    w.tensor("enc.patch_embed.bias", st.get(f"{enc}.embeddings.patch_embeddings.proj.bias"))
    for i in range(L):
        _write_block(w, st, f"enc.blk.{i}", f"{enc}.layer.{i}", VJEPA2_LAYER_NAMES)
    w.tensor("enc.norm.weight", st.get(f"{enc}.layernorm.weight"))
    w.tensor("enc.norm.bias", st.get(f"{enc}.layernorm.bias"))

    pred = prefix + "predictor"
    w.tensor("pred.embed.weight", st.get(f"{pred}.embeddings.predictor_embeddings.weight"))
    w.tensor("pred.embed.bias", st.get(f"{pred}.embeddings.predictor_embeddings.bias"))
    mt = st.get(f"{pred}.embeddings.mask_tokens")  # [n_mask, 1, 1, Dp]
    w.tensor("pred.mask_tokens", mt.reshape(n_mask, Dp))
    for i in range(Lp):
        _write_block(w, st, f"pred.blk.{i}", f"{pred}.layer.{i}", VJEPA2_LAYER_NAMES)
    w.tensor("pred.norm.weight", st.get(f"{pred}.layernorm.weight"))
    w.tensor("pred.norm.bias", st.get(f"{pred}.layernorm.bias"))
    w.tensor("pred.proj.weight", st.get(f"{pred}.proj.weight"))
    w.tensor("pred.proj.bias", st.get(f"{pred}.proj.bias"))

    if has_head:
        q = st.get("pooler.query_tokens")  # [1, 1, D]
        w.tensor("head.query", q.reshape(1, D))
        for i in range(n_pool):
            _write_block(w, st, f"head.blk.{i}", f"pooler.self_attention_layers.{i}", POOLER_LAYER_NAMES)
        xa = "pooler.cross_attention_layer"
        # layer_norm1 is applied to the key/value stream only (queries are NOT normalised);
        # cross_attn has no output projection; layer_norm2 + MLP act on (query + attn).
        w.tensor("head.xattn.ln_kv.weight", st.get(f"{xa}.layer_norm1.weight"))
        w.tensor("head.xattn.ln_kv.bias", st.get(f"{xa}.layer_norm1.bias"))
        for r in ("q", "k", "v"):
            w.tensor(f"head.xattn.{r}.weight", st.get(f"{xa}.cross_attn.{r}_proj.weight"))
            w.tensor(f"head.xattn.{r}.bias", st.get(f"{xa}.cross_attn.{r}_proj.bias"))
        w.tensor("head.xattn.ln2.weight", st.get(f"{xa}.layer_norm2.weight"))
        w.tensor("head.xattn.ln2.bias", st.get(f"{xa}.layer_norm2.bias"))
        w.tensor("head.xattn.ffn_up.weight", st.get(f"{xa}.mlp.fc1.weight"))
        w.tensor("head.xattn.ffn_up.bias", st.get(f"{xa}.mlp.fc1.bias"))
        w.tensor("head.xattn.ffn_down.weight", st.get(f"{xa}.mlp.fc2.weight"))
        w.tensor("head.xattn.ffn_down.bias", st.get(f"{xa}.mlp.fc2.bias"))
        cw = st.get("classifier.weight")
        if cw.shape != (n_classes, D):
            raise ValueError(f"classifier shape {cw.shape} != ({n_classes}, {D})")
        w.tensor("head.cls.weight", cw)
        w.tensor("head.cls.bias", st.get("classifier.bias"))

    unused = st.unused()
    if unused:
        raise RuntimeError(f"{len(unused)} source tensors were not mapped: {unused[:10]} ...")

    w.close()
    summary = dict(
        out=os.path.abspath(out), ftype=ftype, arch=arch, name=model_name,
        n_tensors=len(w.tensor_names), dtype_counts=w.dtype_counts, tensor_bytes=w.n_bytes,
        n_source_tensors=len(st.keys), n_classes=len(labels),
    )
    log(f"[vjepa2] wrote {out}: {summary['n_tensors']} tensors {w.dtype_counts}, {w.n_bytes / 2**20:.1f} MiB of tensor data")
    return summary


# --------------------------------------------------------------------------------------------
# read-back helper (also usable from other scripts)
# --------------------------------------------------------------------------------------------
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
    tensors = []
    for t in r.tensors:
        counts[t.tensor_type.name] = counts.get(t.tensor_type.name, 0) + 1
        tensors.append((t.name, [int(x) for x in t.shape], t.tensor_type.name))
    info = dict(path=path, n_kv=len(kv), n_tensors=len(r.tensors), dtype_counts=counts, kv=kv)
    for k, v in kv.items():
        print(f"{k:36s} = {v}")
    print(f"tensors: {len(r.tensors)}  {counts}")
    if show_tensors:
        for n, s, d in tensors:
            print(f"  {n:40s} {s} {d}")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="HF model directory (config.json + model.safetensors)")
    ap.add_argument("--out", help="output .gguf path")
    ap.add_argument("--ftype", default="f16", choices=sorted(FTYPES), help="weight storage type (default f16)")
    ap.add_argument("--name", default=None, help="general.name override (default: basename of --src)")
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
