"""Shared helpers implementing docs/gguf-schema.md for the jepa.cpp converters.

Everything here is pure numpy + `gguf`; torch is imported lazily only where a
`.pt` state dict has to be read.

Contents
  JepaWriter        thin wrapper over gguf.GGUFWriter (arch "jepa") with typed hparams
                    and the F32/F16 dtype rule
  sincos_2d/3d      bit-matching re-implementations of Meta's get_{2,3}d_sincos_pos_embed
  fuse_qkv          [q;k;v] row concatenation
  flatten_patch_conv  conv weight [D,C,(T,)P,P] -> [D, C*T*P*P] (C-T-H-W order)
  fold_batchnorm    eval-mode BatchNorm1d folded into the preceding Linear
  fold_linear_pair  Linear2(Linear1(x)) folded into one Linear
  preproc_from_hf   preprocessor_config.json -> jepa.pre.* keys
  SourceTensors     dict-like view over a checkpoint that tracks consumed keys
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import gguf
from gguf import GGUFValueType

SCHEMA_VERSION = 1
ARCH = "jepa"

# --ftype -> general.file_type (gguf.LlamaFileType); Q8_0 etc. are produced by the C++ quantizer
FTYPES = {"f32": int(gguf.LlamaFileType.ALL_F32), "f16": int(gguf.LlamaFileType.MOSTLY_F16)}

# HF `hidden_act` / torch module names -> jepa.enc.act
ACT_NAMES = {
    "gelu": "gelu_erf",
    "gelu_erf": "gelu_erf",
    "gelu_new": "gelu_tanh",
    "gelu_pytorch_tanh": "gelu_tanh",
    "gelu_fast": "gelu_tanh",
    "gelu_tanh": "gelu_tanh",
    "silu": "silu",
    "swish": "silu",
}

# PIL resampling ids used by HF image processors
PIL_RESAMPLE = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}

# ----------------------------------------------------------------------------- hparam typing
U32, F32, BOOL, STR, AU32, AF32, ASTR = "u32", "f32", "bool", "str", "[u32]", "[f32]", "[str]"

# Every `jepa.*` key in docs/gguf-schema.md with its GGUF value type.
HPARAM_TYPES: dict[str, str] = {
    "jepa.schema_version": U32,
    "jepa.family": STR,
    "jepa.modality": STR,
    # encoder
    "jepa.enc.embed_dim": U32,
    "jepa.enc.n_layer": U32,
    "jepa.enc.n_head": U32,
    "jepa.enc.ffn_dim": U32,
    "jepa.enc.patch_size": U32,
    "jepa.enc.tubelet_size": U32,
    "jepa.enc.img_size": U32,
    "jepa.enc.n_frames": U32,
    "jepa.enc.in_chans": U32,
    "jepa.enc.ln_eps": F32,
    "jepa.enc.act": STR,
    "jepa.enc.pos_type": STR,
    "jepa.enc.rope_theta": F32,
    "jepa.enc.rope_interpolate": BOOL,
    "jepa.enc.cls_token": BOOL,
    "jepa.enc.n_registers": U32,
    "jepa.enc.qkv_fused": BOOL,
    "jepa.enc.modality_embed": BOOL,
    "jepa.enc.image_patch_embed": BOOL,
    "jepa.enc.hier_layers": AU32,
    "jepa.enc.layer_scale": BOOL,
    "jepa.enc.proj_act": STR,          # lewm: activation of the enc.proj.* MLP
    # predictor
    "jepa.pred.kind": STR,
    "jepa.pred.embed_dim": U32,
    "jepa.pred.n_layer": U32,
    "jepa.pred.n_head": U32,
    "jepa.pred.head_dim": U32,         # lewm: n_head*head_dim != embed_dim
    "jepa.pred.ffn_dim": U32,
    "jepa.pred.n_mask_tokens": U32,
    "jepa.pred.action_dim": U32,
    "jepa.pred.state_dim": U32,
    "jepa.pred.frame_causal": BOOL,
    "jepa.pred.n_frames": U32,
    "jepa.pred.ln_eps": F32,           # lewm: eps of the affine LayerNorms
    "jepa.pred.adaln_eps": F32,        # lewm: eps of the non-affine adaLN LayerNorms
    "jepa.pred.act": STR,
    "jepa.pred.qkv_bias": BOOL,
    "jepa.pred.action_act": STR,       # lewm: activation inside pred.action_embed
    "jepa.pred.proj_act": STR,         # lewm: activation inside pred.proj.* MLP
    # head
    "jepa.head.kind": STR,
    "jepa.head.n_classes": U32,
    "jepa.head.n_pool_layers": U32,
    "jepa.head.labels": ASTR,
    # preprocessing
    "jepa.pre.mean": AF32,
    "jepa.pre.std": AF32,
    "jepa.pre.resize_short": U32,
    "jepa.pre.crop": U32,
    "jepa.pre.resample": STR,
    "jepa.pre.resize_mode": STR,
}


def _infer_type(val: Any) -> str:
    if isinstance(val, bool):
        return BOOL
    if isinstance(val, (int, np.integer)):
        return U32
    if isinstance(val, (float, np.floating)):
        return F32
    if isinstance(val, str):
        return STR
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            raise ValueError("cannot infer element type of an empty array")
        e = val[0]
        if isinstance(e, bool):
            raise ValueError("bool arrays are not supported")
        if isinstance(e, (int, np.integer)):
            return AU32
        if isinstance(e, (float, np.floating)):
            return AF32
        if isinstance(e, str):
            return ASTR
    raise ValueError(f"cannot infer GGUF type for {val!r}")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- writer
class JepaWriter:
    """gguf.GGUFWriter with the jepa.cpp conventions baked in.

    dtype rule (docs/gguf-schema.md "Quantization rules"):
      quantizable tensors (attn_*, ffn_*, pred.proj, head.cls, ... 2-D weights) are stored
      as F16 when --ftype f16, else F32; everything else is always F32.  Q8_0 / Q4_K are
      produced later by tools/jepa-quantize, never here.
    """

    def __init__(self, path: str | Path, ftype: str, *, name: str, family: str, modality: str,
                 license: str, source_url: str, description: str | None = None):
        if ftype not in FTYPES:
            raise ValueError(f"ftype must be one of {list(FTYPES)}, got {ftype!r}")
        self.path = Path(path)
        self.ftype = ftype
        self.family = family
        self.tensor_log: list[tuple[str, tuple[int, ...], str]] = []
        self._hparams: dict[str, Any] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.w = gguf.GGUFWriter(str(self.path), ARCH)
        self.w.add_name(name)
        self.w.add_license(license)
        self.w.add_string("general.source_url", source_url)  # schema key; gguf-py add_source_url() would write general.source.url
        if description:
            self.w.add_description(description)
        self.w.add_file_type(FTYPES[ftype])
        self.w.add_quantization_version(gguf.GGML_QUANT_VERSION)
        self.add_hparams({
            "jepa.schema_version": SCHEMA_VERSION,
            "jepa.family": family,
            "jepa.modality": modality,
        })

    # -- metadata ------------------------------------------------------------------------
    def add_hparam(self, key: str, val: Any) -> None:
        t = HPARAM_TYPES.get(key)
        if t is None:
            t = _infer_type(val)
            log(f"  [warn] {key} is not in docs/gguf-schema.md; writing it as {t}")
        if t == U32:
            if isinstance(val, bool) or int(val) != val or int(val) < 0:
                raise ValueError(f"{key}: expected non-negative int, got {val!r}")
            self.w.add_uint32(key, int(val))
        elif t == F32:
            self.w.add_float32(key, float(val))
        elif t == BOOL:
            if not isinstance(val, (bool, np.bool_)):
                raise ValueError(f"{key}: expected bool, got {val!r}")
            self.w.add_bool(key, bool(val))
        elif t == STR:
            if not isinstance(val, str) or not val:
                raise ValueError(f"{key}: expected non-empty str, got {val!r}")
            self.w.add_string(key, val)
        elif t == AU32:
            self.w.add_key_value(key, [int(v) for v in val], GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
        elif t == AF32:
            self.w.add_key_value(key, [float(v) for v in val], GGUFValueType.ARRAY, sub_type=GGUFValueType.FLOAT32)
        elif t == ASTR:
            self.w.add_key_value(key, [str(v) for v in val], GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING)
        else:  # pragma: no cover
            raise AssertionError(t)
        self._hparams[key] = val

    def add_hparams(self, hp: dict[str, Any]) -> None:
        for k, v in hp.items():
            self.add_hparam(k, v)

    # -- tensors ---------------------------------------------------------------------------
    def add_tensor(self, name: str, arr: Any, quantizable: bool) -> None:
        a = np.ascontiguousarray(np.asarray(arr))
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        if not np.isfinite(a).all():
            raise ValueError(f"{name}: tensor contains NaN/Inf")
        if a.ndim == 0:
            raise ValueError(f"{name}: scalar tensors are not allowed")
        if quantizable and self.ftype == "f16" and a.ndim >= 2:
            a = a.astype(np.float16)
        self.w.add_tensor(name, a)
        self.tensor_log.append((name, tuple(int(x) for x in a.shape), "F16" if a.dtype == np.float16 else "F32"))

    def add_linear(self, dst: str, weight: Any, bias: Any | None, quantizable: bool) -> None:
        """weight -> dst.weight (dtype rule), bias -> dst.bias (always F32)."""
        self.add_tensor(dst + ".weight", weight, quantizable)
        if bias is not None:
            self.add_tensor(dst + ".bias", bias, False)

    def add_norm(self, dst: str, weight: Any, bias: Any | None) -> None:
        self.add_tensor(dst + ".weight", weight, False)
        if bias is not None:
            self.add_tensor(dst + ".bias", bias, False)

    # -- finish ----------------------------------------------------------------------------
    def write(self) -> Path:
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file(progress=False)
        self.w.close()
        n16 = sum(1 for _, _, d in self.tensor_log if d == "F16")
        n32 = len(self.tensor_log) - n16
        size = self.path.stat().st_size
        log(f"wrote {self.path}  ({size / 1e6:.1f} MB, {len(self.tensor_log)} tensors: {n16} F16, {n32} F32, "
            f"{len(self._hparams)} jepa.* keys)")
        return self.path


# ----------------------------------------------------------------------------- sincos tables
def _sincos_1d_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Verbatim numpy math of Meta's get_1d_sincos_pos_embed_from_grid (float64)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def sincos_2d(grid_h: int, grid_w: int, dim: int, *, w_first: bool = False) -> np.ndarray:
    """Meta `get_2d_sincos_pos_embed` generalised to a rectangular grid.  Returns float32
    [grid_h*grid_w, dim], token order h-major.  Computed in float64 exactly like the
    reference, then cast to float32 (which is what `torch.from_numpy(...).float()` does).

    w_first=False -> [emb_h | emb_w]  (vjepa2/src/models/utils/pos_embs.py)
    w_first=True  -> [emb_w | emb_h]  (MAE-style `grid[0]` ordering; this is what the
                                        HF facebook/ijepa_* checkpoints contain, bit-exact)
    """
    gh = np.arange(grid_h, dtype=float)
    gw = np.arange(grid_w, dtype=float)
    gw, gh = np.meshgrid(gw, gh)  # indexing [h, w]
    emb_h = _sincos_1d_from_grid(dim // 2, gh)
    emb_w = _sincos_1d_from_grid(dim // 2, gw)
    pos = np.concatenate([emb_w, emb_h] if w_first else [emb_h, emb_w], axis=1)
    return pos.astype(np.float32)


def sincos_3d(grid_t: int, grid_h: int, grid_w: int, dim: int, uniform_power: bool = False) -> np.ndarray:
    """Meta `get_3d_sincos_pos_embed` generalised to a rectangular grid.  Returns float32
    [grid_t*grid_h*grid_w, dim], token order t-major then h then w."""
    gd = np.arange(grid_t, dtype=float)
    gh = np.arange(grid_h, dtype=float)
    gw = np.arange(grid_w, dtype=float)
    gh, gd, gw = np.meshgrid(gh, gd, gw)  # order of meshgrid is very important for indexing as [d,h,w]
    if not uniform_power:
        h_dim = dim // 4
        w_dim = dim // 4
        d_dim = dim // 2
    else:
        h_dim = w_dim = d_dim = int(np.ceil(dim / 6) * 2)
    emb_h = _sincos_1d_from_grid(h_dim, gh)
    emb_w = _sincos_1d_from_grid(w_dim, gw)
    emb_d = _sincos_1d_from_grid(d_dim, gd)
    pos = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos = pos[:, :dim]
    return pos.astype(np.float32)


# ----------------------------------------------------------------------------- tensor helpers
def fuse_qkv(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows [q; k; v] -> attn_qkv.{weight,bias}.  Works for 2-D weights and 1-D biases."""
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"q/k/v shapes differ: {q.shape} {k.shape} {v.shape}")
    return np.concatenate([q, k, v], axis=0)


def flatten_patch_conv(w: np.ndarray) -> np.ndarray:
    """Conv weight [D, C, P, P] or [D, C, T, P, P] -> [D, C*T*P*P] in C-T-H-W order.
    PyTorch stores conv weights as [out, in, (kT,) kH, kW] so a reshape is already the
    right order; this only validates the rank."""
    if w.ndim not in (4, 5):
        raise ValueError(f"patch conv weight must be 4-D or 5-D, got {w.shape}")
    return np.ascontiguousarray(w.reshape(w.shape[0], -1))


def fold_batchnorm(w: np.ndarray, b: np.ndarray | None, gamma: np.ndarray, beta: np.ndarray,
                   running_mean: np.ndarray, running_var: np.ndarray, eps: float = 1e-5):
    """Fold an eval-mode BatchNorm1d that follows Linear(w, b) into the Linear.

    y = gamma * (W x + b - mean) / sqrt(var + eps) + beta
      = (s * W) x + (s * (b - mean) + beta),   s = gamma / sqrt(var + eps)
    Computed in float64, returned as float32."""
    w64 = w.astype(np.float64)
    b64 = np.zeros(w.shape[0], dtype=np.float64) if b is None else b.astype(np.float64)
    s = gamma.astype(np.float64) / np.sqrt(running_var.astype(np.float64) + eps)
    w_f = w64 * s[:, None]
    b_f = (b64 - running_mean.astype(np.float64)) * s + beta.astype(np.float64)
    return w_f.astype(np.float32), b_f.astype(np.float32)


def fold_linear_pair(w1: np.ndarray, b1: np.ndarray | None, w2: np.ndarray, b2: np.ndarray | None):
    """Linear2(Linear1(x)) with no nonlinearity in between -> single Linear (float64 math).
    W = W2 @ W1,  b = W2 @ b1 + b2."""
    w1 = w1.astype(np.float64)
    w2 = w2.astype(np.float64)
    w = w2 @ w1
    b = np.zeros(w2.shape[0], dtype=np.float64)
    if b1 is not None:
        b = b + w2 @ b1.astype(np.float64)
    if b2 is not None:
        b = b + b2.astype(np.float64)
    return w.astype(np.float32), b.astype(np.float32)


# ----------------------------------------------------------------------------- preprocessing
def preproc_from_hf(cfg: str | Path | dict) -> dict[str, Any]:
    """Translate a HF image/video preprocessor_config.json into jepa.pre.* keys.

    jepa.pre.resize_mode:
      "shortest_edge" resize the short side to `resize_short` (aspect kept), then centre-crop `crop`
      "squash"        resize directly to crop x crop ignoring aspect (ViTImageProcessor with
                      size={height,width} and no centre crop — this is what the HF I-JEPA
                      checkpoint ships)
    """
    if not isinstance(cfg, dict):
        with open(cfg) as f:
            cfg = json.load(f)
    if not cfg.get("do_normalize", True):
        raise ValueError("preprocessor has do_normalize=false; jepa.pre.* cannot express that")
    mean = [float(x) for x in cfg["image_mean"]]
    std = [float(x) for x in cfg["image_std"]]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("expected 3-channel mean/std")

    size = cfg.get("size")
    crop_cfg = cfg.get("crop_size")
    do_crop = bool(cfg.get("do_center_crop", False))
    if isinstance(size, dict) and "shortest_edge" in size:
        resize_short = int(size["shortest_edge"])
        mode = "shortest_edge"
    elif isinstance(size, dict) and "height" in size:
        if int(size["height"]) != int(size["width"]):
            raise ValueError(f"non-square size {size} not supported")
        resize_short = int(size["height"])
        mode = "squash"
    elif isinstance(size, int):
        resize_short = int(size)
        mode = "shortest_edge"
    else:
        raise ValueError(f"unrecognised size spec {size!r}")

    if do_crop and crop_cfg is not None:
        if isinstance(crop_cfg, dict):
            if int(crop_cfg["height"]) != int(crop_cfg["width"]):
                raise ValueError(f"non-square crop {crop_cfg} not supported")
            crop = int(crop_cfg["height"])
        else:
            crop = int(crop_cfg)
    else:
        crop = resize_short
        if mode == "shortest_edge" and not do_crop:
            # short-side resize without crop yields a non-square image; the runtime will
            # centre-crop to `crop`, which for square inputs is a no-op.
            pass

    resample = cfg.get("resample", 2)
    if isinstance(resample, str):
        resample_name = resample.lower()
    else:
        resample_name = PIL_RESAMPLE[int(resample)]
    if resample_name not in ("bilinear", "bicubic"):
        raise ValueError(f"resample {resample_name} not supported by the runtime")

    return {
        "jepa.pre.mean": mean,
        "jepa.pre.std": std,
        "jepa.pre.resize_short": resize_short,
        "jepa.pre.crop": crop,
        "jepa.pre.resample": resample_name,
        "jepa.pre.resize_mode": mode,
    }


# ----------------------------------------------------------------------------- checkpoint access
class SourceTensors:
    """dict-like view over a checkpoint that records which keys were consumed, so a
    converter can prove nothing was silently dropped (`unused()`)."""

    def __init__(self, keys: Iterable[str], getter: Callable[[str], np.ndarray]):
        self._keys = list(keys)
        self._get = getter
        self.used: set[str] = set()

    @classmethod
    def from_dict(cls, d: dict[str, np.ndarray]) -> "SourceTensors":
        return cls(d.keys(), d.__getitem__)

    @classmethod
    def from_safetensors(cls, path: str | Path) -> "SourceTensors":
        from safetensors import safe_open
        f = safe_open(str(path), framework="np")
        st = cls(list(f.keys()), lambda k: np.asarray(f.get_tensor(k)))
        st._handle = f  # keep the file open
        return st

    @classmethod
    def from_torch(cls, path: str | Path) -> "SourceTensors":
        import torch
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: expected a state dict, got {type(obj)}")
        d: dict[str, np.ndarray] = {}
        for k, v in obj.items():
            if torch.is_tensor(v):
                if v.is_floating_point():
                    v = v.float()
                d[k] = v.detach().cpu().numpy()
        return cls.from_dict(d)

    def keys(self) -> list[str]:
        return list(self._keys)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def peek(self, key: str) -> np.ndarray:
        """Read without marking as consumed."""
        return self._get(key)

    def take(self, key: str) -> np.ndarray:
        if key not in self._keys:
            raise KeyError(f"checkpoint has no tensor {key!r}")
        self.used.add(key)
        return self._get(key)

    def take_opt(self, key: str) -> np.ndarray | None:
        return self.take(key) if key in self._keys else None

    def drop(self, key: str) -> None:
        """Mark a key as intentionally ignored."""
        if key in self._keys:
            self.used.add(key)

    def unused(self) -> list[str]:
        return sorted(k for k in self._keys if k not in self.used)

    def layer_count(self, fmt: str) -> int:
        """Count consecutive layers i for which fmt.format(i=i) exists."""
        n = 0
        while fmt.format(i=n) in self._keys:
            n += 1
        return n


def check_unmapped(st: SourceTensors, expected_prefixes: tuple[str, ...], allow_unmapped: bool) -> list[str]:
    """Report unconsumed source keys.  Keys under `expected_prefixes` (heads / loss modules)
    are listed but tolerated; anything else is an error unless allow_unmapped."""
    rest = st.unused()
    if rest:
        log(f"  {len(rest)} source tensors not mapped:")
        for k in rest:
            log(f"    - {k}")
    bad = [k for k in rest if not k.startswith(expected_prefixes)]
    if bad and not allow_unmapped:
        raise SystemExit(f"unexpected unmapped tensors (pass --allow-unmapped to override): {bad}")
    return rest


def read_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path(family: str, src: str | Path, ftype: str) -> Path:
    """models/gguf/<basename(src) without extension>-<ftype>.gguf, relative to the repo root."""
    p = Path(src)
    base = p.stem if p.is_file() else p.name
    base = base.rstrip("/") or family
    return repo_root() / "models" / "gguf" / f"{base}-{ftype}.gguf"
