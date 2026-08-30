#!/usr/bin/env python3
"""Dump PyTorch golden references for the jepa.cpp parity tests.

    scripts/dump_reference.py --model NAME [--n-images N] [--n-clips N] [--frames 64,16] [--root REPO]

For every model this writes   <out>/<NAME>/manifest.json   plus one .npy per tensor per sample
(`<sample>.<tensor>.npy`, float32 C-order; the only non-float tensors are `frames_u8` (uint8) and
`top5_idx` (int64)).  Everything runs in float32 eval mode on CPU, no autocast.  See
tests/fixtures/README.md for the manifest schema and the per-model tensor list.

Models (NAME):
  ijepa-vith14-1k             HF IJepaModel                      (facebook/ijepa_vith14_1k)
  lejepa-vits16               OK-AI ViTv2 via trust_remote_code   (OK-AI/lejepa-vits16-pretrain-in1k)
  lewm-pusht                  LeWorldModel (HF ViT-Ti/14 encoder + AdaLN predictor, re-implemented here)
  vjepa2-vitl-fpc64-256       HF VJEPA2Model (encoder + default predictor pass)
  vjepa2-vitl-fpc16-256-ssv2  HF VJEPA2ForVideoClassification (attentive pooler + linear head)
  vjepa2_1-vitb-384           Meta code path (torch.hub, local clone of facebookresearch/vjepa2)
  all                         every model above, in this order

Paths default to the repository this script lives in (models/, tests/fixtures/media, tests/fixtures/ref,
tmp/vjepa2-src); pass --root to point at a different checkout.  HF_HOME / TORCH_HOME default to <root>/tmp/.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_ROOT = HERE.parents[1]

MODEL_DIRS = {
    "ijepa-vith14-1k": "facebook/ijepa_vith14_1k",
    "lejepa-vits16": "OK-AI/lejepa-vits16-pretrain-in1k",
    "lewm-pusht": "quentinll/lewm-pusht",
    "vjepa2-vitl-fpc64-256": "facebook/vjepa2-vitl-fpc64-256",
    "vjepa2-vitl-fpc16-256-ssv2": "facebook/vjepa2-vitl-fpc16-256-ssv2",
    "vjepa2_1-vitb-384": "vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt",
}
DEFAULT_N_IMAGES = {"ijepa-vith14-1k": 8, "lejepa-vits16": 8, "lewm-pusht": 2, "vjepa2_1-vitb-384": 2}
DEFAULT_FRAMES = {"vjepa2-vitl-fpc64-256": [64, 16], "vjepa2-vitl-fpc16-256-ssv2": [16], "vjepa2_1-vitb-384": [16]}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VJEPA2_SRC_URL = "https://github.com/facebookresearch/vjepa2"


# --------------------------------------------------------------------------------------------------
# environment (must run before torch / transformers are imported)
# --------------------------------------------------------------------------------------------------

def _act_name(hf_act: str) -> str:
    """Map HF activation strings onto the docs/gguf-schema.md vocabulary."""
    return {"gelu": "gelu_erf", "gelu_new": "gelu_tanh", "gelu_pytorch_tanh": "gelu_tanh", "silu": "silu", "swish": "silu"}.get(hf_act, hf_act)

def setup_env(root: Path, threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("HF_HOME", str(root / "tmp" / "hf-home"))
    os.environ.setdefault("TORCH_HOME", str(root / "tmp" / "torch-home"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything is loaded from local directories
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def to_np(t):
    import numpy as np
    import torch

    if isinstance(t, torch.Tensor):
        return t.detach().cpu().contiguous().numpy()
    return np.asarray(t)


class RefWriter:
    """Collects samples for one model and writes manifest.json + .npy files."""

    def __init__(self, out_dir: Path, model: str, source: Path, **meta):
        import torch
        import transformers

        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.samples: list[dict] = []
        self.manifest = {
            "model": model,
            "source": str(source),
            "dtype": "float32",
            "npy_dtype_note": "all tensors float32 C-order except frames_u8 (uint8) and top5_idx (int64)",
            "framework": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
                "threads": torch.get_num_threads(),
                "cpu": platform.processor() or platform.machine(),
            },
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.manifest.update(meta)

    def add(self, name: str, media, tensors: dict, timing: dict, **extra) -> None:
        import numpy as np

        entry = {"name": name, "media": media, "timing_s": timing, "tensors": {}}
        for tname, (arr, layout) in tensors.items():
            arr = to_np(arr)
            if arr.dtype not in (np.uint8, np.int64):
                arr = arr.astype(np.float32, copy=False)
            arr = np.ascontiguousarray(arr)
            fn = f"{name}.{tname}.npy"
            np.save(self.dir / fn, arr)
            entry["tensors"][tname] = {"file": fn, "shape": list(arr.shape), "dtype": str(arr.dtype), "layout": layout}
        entry.update(extra)
        self.samples.append(entry)
        shapes = " ".join(f"{k}{v['shape']}" for k, v in entry["tensors"].items())
        print(f"  [{name}] forward={timing.get('forward_s', 0.0):.3f}s  {shapes}")

    def finish(self, load_s: float, **extra) -> Path:
        fwd = [s["timing_s"].get("forward_s", 0.0) for s in self.samples]
        self.manifest["timing_s"] = {
            "model_load": round(load_s, 3),
            "forward_total": round(sum(fwd), 3),
            "forward_mean": round(sum(fwd) / max(1, len(fwd)), 3),
            "wall_total": round(time.time() - self.t0, 3),
        }
        self.manifest.update(extra)
        self.manifest["samples"] = self.samples
        p = self.dir / "manifest.json"
        p.write_text(json.dumps(self.manifest, indent=1))
        print(f"  wrote {p}  ({len(self.samples)} samples, load {load_s:.1f}s, forward total {sum(fwd):.1f}s, "
              f"wall {self.manifest['timing_s']['wall_total']:.1f}s)")
        return p


def list_images(media: Path, n: int) -> list[Path]:
    imgs = sorted(media.glob("*.jpg")) + sorted(media.glob("*.png"))
    if not imgs:
        sys.exit(f"no images in {media} (run scripts/download_fixtures.sh)")
    return imgs[:n]


def list_clips(media: Path, n: int) -> list[Path]:
    pref = [media / "archery.mp4", media / "bowling.mp4"]
    clips = [p for p in pref if p.exists()] + [p for p in sorted(media.glob("*.mp4")) if p not in pref]
    if not clips:
        sys.exit(f"no clips in {media} (run scripts/download_fixtures.sh)")
    return clips[:n]


def decode_video(path: Path):
    """Decode every frame with PyAV as RGB24 -> (T, H, W, 3) uint8, plus fps."""
    import av
    import numpy as np

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
        fps = float(s.average_rate) if s.average_rate else None
    return np.stack(frames), fps


def sample_frames(frames, n: int):
    """n frames uniformly over the clip, endpoints included: idx = round(linspace(0, T-1, n))."""
    import numpy as np

    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.ascontiguousarray(frames[idx]), idx.tolist()


def describe_processor(proc) -> dict:
    """Record the processor class, its backend (PIL vs torchvision) and its effective config."""
    mro = [c.__name__ for c in type(proc).__mro__]
    backend = "torchvision" if "TorchvisionBackend" in mro else ("pil" if "PilBackend" in mro or "BaseImageProcessor" in mro else "unknown")
    cfg = proc.to_dict() if hasattr(proc, "to_dict") else {}
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("model_valid_processing_keys",)}
    return {"class": type(proc).__name__, "backend": backend, "config": cfg}


RESIZE_NOTE = ("resampling = torchvision.transforms.v2.functional.resize on the uint8 CHW tensor with antialias=True (result rounded back "
               "to uint8), exactly what transformers>=5 image/video processors (TorchvisionBackend) do; verified to 2.4e-7 against a "
               "re-implementation, whereas PIL resampling differs by up to 1.8e-2 (1-2 uint8 levels)")


def image_to_tensor_imagenet(im, size: int):
    """PIL RGB -> squash-resize to size x size (torchvision bilinear + antialias on uint8) -> /255 -> ImageNet normalize -> [1,3,H,W]."""
    import numpy as np
    import torch
    import torchvision.transforms.v2.functional as F

    t = torch.from_numpy(np.array(im.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1)
    t = F.resize(t, [size, size], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    x = t.float() / 255.0
    x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
    return x[None].contiguous()


# --------------------------------------------------------------------------------------------------
# I-JEPA ViT-H/14 (HF)
# --------------------------------------------------------------------------------------------------
def dump_ijepa(a) -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, IJepaModel

    name = "ijepa-vith14-1k"
    d = a.models_dir / MODEL_DIRS[name]
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(d)
    model = IJepaModel.from_pretrained(d, dtype=torch.float32).eval()
    load_s = time.time() - t0
    cfg = model.config
    w = RefWriter(
        a.out / name, name, d,
        hf_id="facebook/ijepa_vith14_1k",
        hparams={"embed_dim": cfg.hidden_size, "n_layer": cfg.num_hidden_layers, "n_head": cfg.num_attention_heads,
                 "ffn_dim": cfg.intermediate_size, "patch_size": cfg.patch_size, "img_size": cfg.image_size,
                 "ln_eps": cfg.layer_norm_eps, "act": _act_name(cfg.hidden_act), "qkv_bias": cfg.qkv_bias, "cls_token": False,
                 "pos_type": "sincos2d", "n_tokens": (cfg.image_size // cfg.patch_size) ** 2},
        preprocessing={
            "description": "ViTImageProcessor: RGB -> resize to exactly 224x224 (aspect ratio NOT kept; bilinear + antialias on the "
                           "uint8 tensor) -> x/255 -> (x-0.5)/0.5 -> NCHW.  " + RESIZE_NOTE,
            "resize": {"height": 224, "width": 224, "keep_aspect": False, "resample": "bilinear", "antialias": True, "on_dtype": "uint8"},
            "center_crop": None, "rescale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
            "processor": describe_processor(proc)},
        outputs={"input": "processor pixel_values, NCHW", "last_hidden_state": "IJepaModel last_hidden_state[0] "
                 "(after final LayerNorm), 256 patch tokens H-major then W, no CLS", "pooled_mean": "mean over tokens"},
    )
    for p in list_images(a.media_dir, a.n_images):
        im = Image.open(p)
        t = time.time()
        x = proc(images=im, return_tensors="pt")["pixel_values"].float()
        pre = time.time() - t
        t = time.time()
        with torch.inference_mode():
            out = model(pixel_values=x)
        fwd = time.time() - t
        lhs = out.last_hidden_state[0]
        w.add(p.stem, p.name, {"input": (x, "NCHW"), "last_hidden_state": (lhs, "[N_tokens, D]"),
                               "pooled_mean": (lhs.mean(0), "[D]")}, {"preprocess_s": round(pre, 4), "forward_s": round(fwd, 4)},
              image_size_hw=[im.height, im.width])
    w.finish(load_s)


# --------------------------------------------------------------------------------------------------
# LeJEPA ViT-S/16 (OK-AI ViTv2, custom code)
# --------------------------------------------------------------------------------------------------
def dump_lejepa(a) -> None:
    import torch
    from PIL import Image

    name = "lejepa-vits16"
    d = a.models_dir / MODEL_DIRS[name]
    if not (d / "hf_src").is_dir():
        sys.exit(f"{d}/hf_src is missing: modelling_vitv2.py imports hf_src.* from the HF repo; download the hf_src/ "
                 f"folder of OK-AI/lejepa-vits16-pretrain-in1k next to model.safetensors")
    sys.path.insert(0, str(d))  # modelling_vitv2.py does `from hf_src... import` / `from configuration_vitv2 import`
    from transformers import AutoImageProcessor, AutoModel

    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(d)
    model = AutoModel.from_pretrained(d, trust_remote_code=True, dtype=torch.float32).eval()
    load_s = time.time() - t0
    bb = model.backbone
    cfg = model.config
    w = RefWriter(
        a.out / name, name, d,
        hf_id="OK-AI/lejepa-vits16-pretrain-in1k",
        hparams={"embed_dim": cfg.embed_dim, "n_layer": cfg.depth, "n_head": cfg.num_heads, "ffn_dim": int(cfg.embed_dim * cfg.mlp_ratio),
                 "patch_size": cfg.patch_size, "img_size": cfg.img_size, "ln_eps": bb.norm.eps, "act": "gelu_erf",
                 "qkv_bias": True, "cls_token": True, "n_registers": cfg.num_register_tokens, "layer_scale": bool(cfg.init_values),
                 "pos_type": "learned", "pos_embed_shape": list(bb.pos_embed.shape), "n_tokens": bb.pos_embed.shape[1]},
        preprocessing={
            "description": "BitImageProcessor: RGB -> resize shortest edge to 256 keeping aspect (other side = int(256*W/short), i.e. floored; BICUBIC + "
                           "antialias on the uint8 tensor) -> center crop 224x224 (top=int((H-224)/2), left=int((W-224)/2)) -> x/255 -> "
                           "ImageNet mean/std -> NCHW.  " + RESIZE_NOTE,
            "resize": {"shortest_edge": 256, "resample": "bicubic", "antialias": True, "on_dtype": "uint8"}, "center_crop": 224, "rescale": 1 / 255,
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "processor": describe_processor(proc)},
        outputs={
            "note": "ViTv2.forward returns a dict {latent, patch_latent, raw_latent, last_self_attention, logits}; "
                    "latent = norm(x)[CLS], patch_latent = norm(x)[1:], raw_latent = x[CLS] before the final norm, "
                    "logits = head(latent) with head = Identity (so logits == latent).",
            "input": "processor pixel_values, NCHW",
            "last_hidden_state": "concat([latent, patch_latent]) = ALL 197 tokens (CLS first, then 196 patches H-major) "
                                 "after the final LayerNorm (backbone.norm, eps 1e-6); identical to ViTv2.forward_backbone(x)",
            "cls": "latent = CLS token after final norm", "pooled_mean": "mean over the 196 patch tokens (after final norm)",
            "cls_raw": "raw_latent = CLS token before the final norm"},
    )
    for p in list_images(a.media_dir, a.n_images):
        im = Image.open(p)
        t = time.time()
        x = proc(images=im, return_tensors="pt")["pixel_values"].float()
        pre = time.time() - t
        t = time.time()
        with torch.inference_mode():
            out = model(x)
        fwd = time.time() - t
        cls = out["latent"][0]
        patches = out["patch_latent"][0]
        lhs = torch.cat([cls[None], patches], 0)
        w.add(p.stem, p.name, {"input": (x, "NCHW"), "last_hidden_state": (lhs, "[1+N_patches, D] CLS first"),
                               "cls": (cls, "[D]"), "pooled_mean": (patches.mean(0), "[D]"), "cls_raw": (out["raw_latent"][0], "[D]")},
              {"preprocess_s": round(pre, 4), "forward_s": round(fwd, 4)}, image_size_hw=[im.height, im.width])
    w.finish(load_s)


# --------------------------------------------------------------------------------------------------
# LeWorldModel (Push-T).  Modules re-implemented from github.com/lucas-maes/le-wm (module.py / jepa.py, MIT) and
# stable_worldmodel.wm.lewm (same code); the encoder is a stock HF ViTModel built like
# stable_pretraining.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False).
# --------------------------------------------------------------------------------------------------
def build_lewm(config: dict):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import ViTConfig, ViTModel

    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim, dropout=0.0):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

        def forward(self, x):
            return self.net(x)

    class Attention(nn.Module):
        def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
            super().__init__()
            inner = dim_head * heads
            self.heads = heads
            self.norm = nn.LayerNorm(dim)
            self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
            self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

        def forward(self, x, causal=True):
            x = self.norm(x)
            b, t, _ = x.shape
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = (z.view(b, t, self.heads, -1).transpose(1, 2) for z in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            return self.to_out(out.transpose(1, 2).reshape(b, t, -1))

    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    class ConditionalBlock(nn.Module):  # AdaLN-zero (DiT style), conditioning c is per token
        def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
            super().__init__()
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        def forward(self, x, c):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x

    class Transformer(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim)
            self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
            self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
            self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
            self.layers = nn.ModuleList([ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])

        def forward(self, x, c):
            x = self.input_proj(x)
            c = self.cond_proj(c)
            for blk in self.layers:
                x = blk(x, c)
            return self.output_proj(self.norm(x))

    class Embedder(nn.Module):  # action encoder
        def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
            super().__init__()
            self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
            self.embed = nn.Sequential(nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.SiLU(), nn.Linear(mlp_scale * emb_dim, emb_dim))

        def forward(self, x):  # (B, T, A)
            x = self.patch_embed(x.float().permute(0, 2, 1)).permute(0, 2, 1)
            return self.embed(x)

    class MLP(nn.Module):  # projector / pred_proj: Linear -> BatchNorm1d -> GELU -> Linear
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

        def forward(self, x):
            return self.net(x)

    class Predictor(nn.Module):  # ARPredictor
        def __init__(self, num_frames, input_dim, hidden_dim, output_dim, depth, heads, mlp_dim, dim_head=64, dropout=0.0, emb_dropout=0.0):
            super().__init__()
            self.num_frames = num_frames
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
            self.dropout = nn.Dropout(emb_dropout)
            self.transformer = Transformer(input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim, dropout)

        def forward(self, x, c):  # x: (B, T, D) embeddings, c: (B, T, D) action embeddings
            x = x + self.pos_embedding[:, : x.size(1)]
            return self.transformer(self.dropout(x), c)

    class LeWM(nn.Module):
        def __init__(self):
            super().__init__()
            e = config["encoder"]
            size = {"tiny": (192, 12, 3), "small": (384, 12, 6), "base": (768, 12, 12), "large": (1024, 24, 16)}[e["size"]]
            vcfg = ViTConfig(hidden_size=size[0], num_hidden_layers=size[1], num_attention_heads=size[2], intermediate_size=size[0] * 4,
                             image_size=e["image_size"], patch_size=e["patch_size"])
            self.vit_config = vcfg
            self.encoder = ViTModel(vcfg, add_pooling_layer=False, use_mask_token=e.get("use_mask_token", False))
            p = config["predictor"]
            self.predictor = Predictor(num_frames=p["num_frames"], input_dim=p["input_dim"], hidden_dim=p["hidden_dim"], output_dim=p["output_dim"],
                                       depth=p["depth"], heads=p["heads"], mlp_dim=p["mlp_dim"], dim_head=p["dim_head"], dropout=p["dropout"],
                                       emb_dropout=p["emb_dropout"])
            ae = config["action_encoder"]
            self.action_encoder = Embedder(input_dim=ae["input_dim"], smoothed_dim=ae.get("smoothed_dim", ae["input_dim"]), emb_dim=ae["emb_dim"])
            pr = config["projector"]
            self.projector = MLP(pr["input_dim"], pr["hidden_dim"], pr["output_dim"])
            pp = config["pred_proj"]
            self.pred_proj = MLP(pp["input_dim"], pp["hidden_dim"], pp["output_dim"])

        def encode(self, pixels):  # (B,3,H,W) -> tokens (B,257,D), cls (B,D), emb (B,D)
            tokens = self.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state
            cls = tokens[:, 0]
            return tokens, cls, self.projector(cls)

        def predict(self, emb, act_emb):  # (B,T,D),(B,T,D) -> (B,T,D) predicted next embeddings (projected space)
            b, t, d = emb.shape
            preds = self.predictor(emb, act_emb)
            return self.pred_proj(preds.reshape(b * t, d)).reshape(b, t, -1)

    return LeWM()


def _remap_vit_keys(sd: dict) -> dict:
    """weights.pt uses transformers<5 ViTModel names; transformers>=5 renamed them (encoder.layer.N.attention.attention.query
    -> layers.N.attention.q_proj, intermediate.dense -> mlp.fc1, ...).  Pure renames, no reshapes."""
    import re

    out = {}
    for k, v in sd.items():
        k2 = k
        if k.startswith("encoder."):
            k2 = re.sub(r"^encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1.", k2)
            for old, new in ((".attention.attention.query.", ".attention.q_proj."), (".attention.attention.key.", ".attention.k_proj."),
                             (".attention.attention.value.", ".attention.v_proj."), (".attention.output.dense.", ".attention.o_proj."),
                             (".intermediate.dense.", ".mlp.fc1."), (".output.dense.", ".mlp.fc2.")):
                k2 = k2.replace(old, new)
        out[k2] = v
    return out


def dump_lewm(a) -> None:
    import torch
    from PIL import Image

    name = "lewm-pusht"
    d = a.models_dir / MODEL_DIRS[name]
    config = json.loads((d / "config.json").read_text())
    t0 = time.time()
    model = build_lewm(config)
    sd = torch.load(d / "weights.pt", map_location="cpu", weights_only=False)
    if "encoder.layers.0.attention.q_proj.weight" in model.state_dict():
        sd = _remap_vit_keys(sd)
    model.load_state_dict(sd, strict=True)  # strict: proves the re-implementation has exactly the checkpoint's modules
    model.eval()
    load_s = time.time() - t0
    enc_cfg = model.vit_config
    img_size = config["encoder"]["image_size"]
    act_dim = config["action_encoder"]["input_dim"]
    imgs = list_images(a.media_dir, max(a.n_images, 3))
    n_single = a.n_images
    n_seq = min(3, len(imgs), config["predictor"]["num_frames"])
    g = torch.Generator().manual_seed(0)
    actions = torch.randn(8, act_dim, generator=g)[: max(n_single, n_seq)]  # fixed draw (8 rows) so row i is stable across --n-images
    w = RefWriter(
        a.out / name, name, d,
        hf_id="quentinll/lewm-pusht", config=config,
        hparams={"enc": {"embed_dim": enc_cfg.hidden_size, "n_layer": enc_cfg.num_hidden_layers, "n_head": enc_cfg.num_attention_heads,
                         "ffn_dim": enc_cfg.intermediate_size, "patch_size": enc_cfg.patch_size, "img_size": enc_cfg.image_size,
                         "ln_eps": enc_cfg.layer_norm_eps, "act": _act_name(enc_cfg.hidden_act), "qkv_bias": enc_cfg.qkv_bias, "cls_token": True,
                         "pos_type": "learned", "n_tokens": model.encoder.embeddings.position_embeddings.shape[1]},
                 "pred": {"embed_dim": config["predictor"]["hidden_dim"], "n_layer": config["predictor"]["depth"], "n_head": config["predictor"]["heads"],
                          "head_dim": config["predictor"]["dim_head"], "ffn_dim": config["predictor"]["mlp_dim"], "n_frames": config["predictor"]["num_frames"],
                          "action_dim": act_dim, "block_ln_eps": 1e-6, "inner_ln_eps": 1e-5, "bn_eps": 1e-5}},
        preprocessing={
            "description": f"RGB -> squash-resize to {img_size}x{img_size} (bilinear + antialias on the uint8 tensor; aspect NOT kept) -> x/255 "
                           "-> ImageNet mean/std -> NCHW.  (LeWM was trained on 224x224 renders; upstream eval applies ToImage/ToDtype/"
                           "Normalize(ImageNet)/Resize(224) which is a no-op on 224 renders; the resize here only exists to feed COCO images "
                           "and uses the same resampler as the HF processors.)  " + RESIZE_NOTE,
            "resize": {"height": img_size, "width": img_size, "keep_aspect": False, "resample": "bilinear", "antialias": True, "on_dtype": "uint8"},
            "center_crop": None,
            "rescale": 1 / 255, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        predictor_semantics={
            "encoder": "HF ViTModel ViT-Ti/14 @224: 16x16 patches + CLS = 257 tokens, learned pos-embed [257,192] (CLS slot first), "
                       "LayerNorm eps 1e-12 (HF ViTConfig default!), GELU(erf), qkv bias, final layernorm on all tokens -> last_hidden_state.",
            "emb": "emb = projector(last_hidden_state[:,0]) = Linear(192->2048) -> BatchNorm1d(2048; eval mode = running_mean/var, eps 1e-5) "
                   "-> GELU -> Linear(2048->192).  Only the CLS token feeds the world model; patch tokens are unused.",
            "action": "raw action vector a[10] = frameskip(5) x action_dim(2) of the Push-T dataset, z-score normalised with dataset statistics "
                      "at train time; here a ~ N(0,1) (torch.Generator seed 0, randn(3,10), row i for image/step i).  "
                      "act_emb = Embedder(a) = Conv1d(10->10,k=1) (i.e. Linear 10->10 per step) -> Linear(10->768) -> SiLU -> Linear(768->192).",
            "predictor": "x = emb_seq + pos_embedding[:T] (T<=3); 6 x AdaLN-zero blocks conditioned per-step on act_emb: "
                         "[shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp] = Linear(SiLU(c)) (192->1152); "
                         "x += gate_msa * Attn(LN_noaffine(x)*(1+scale_msa)+shift_msa); x += gate_mlp * FF(LN_noaffine(x)*(1+scale_mlp)+shift_mlp).  "
                         "Attn = LayerNorm(192, affine) -> Linear(192->3072, no bias) -> 16 heads x 64 -> CAUSAL softmax attention "
                         "-> Linear(1024->192).  FF = LayerNorm(192, affine) -> Linear(192->2048) -> GELU -> Linear(2048->192).  "
                         "Then final LayerNorm(192).  input_proj/cond_proj/output_proj are Identity (dims equal).",
            "pred_next": "pred_next = pred_proj(predictor(...)[:, -1]) with pred_proj = Linear(192->2048) -> BatchNorm1d -> GELU -> Linear(2048->192).  "
                         "It predicts the *projected* embedding (projector output) of the next frame; rollout appends it to emb_seq and keeps the last 3.",
            "samples": "per-image samples run the predictor with T=1 (one frame, one action).  The 'seq' sample runs T=3 over images "
                       "[img0,img1,img2] with actions rows 0..2 and stores the prediction at every position (causal, so position t only sees <=t).",
        },
        outputs={"input": "NCHW normalised image", "last_hidden_state": "ViTModel.last_hidden_state[0] (257 tokens, CLS first, after final LN)",
                 "cls": "last_hidden_state[0]", "emb": "projector(cls)", "action": "raw action (10)", "act_emb": "Embedder(action)",
                 "pred_next": "pred_proj(predictor(emb, act_emb))  (T=1)",
                 "emb_seq/action_seq/act_emb_seq/pred_seq": "sequence sample, [T, .] with T = 3"},
    )
    xs = []
    with torch.inference_mode():
        for i, p in enumerate(imgs[: max(n_single, n_seq)]):
            im = Image.open(p)
            t = time.time()
            x = image_to_tensor_imagenet(im, img_size)
            pre = time.time() - t
            xs.append(x)
            if i >= n_single:
                continue
            t = time.time()
            tokens, cls, emb = model.encode(x)
            act = actions[i : i + 1][None]  # (1,1,10)
            act_emb = model.action_encoder(act)
            pred_next = model.predict(emb[None], act_emb)  # (1,1,192)
            fwd = time.time() - t
            w.add(p.stem, p.name,
                  {"input": (x, "NCHW"), "last_hidden_state": (tokens[0], "[1+N_patches, D] CLS first"), "cls": (cls[0], "[D]"),
                   "emb": (emb[0], "[D]"), "action": (act[0, 0], "[action_dim]"), "act_emb": (act_emb[0, 0], "[D]"),
                   "pred_next": (pred_next[0, 0], "[D]")},
                  {"preprocess_s": round(pre, 4), "forward_s": round(fwd, 4)}, image_size_hw=[im.height, im.width])
        # sequence sample
        t = time.time()
        x_seq = torch.cat(xs[:n_seq], 0)
        tokens, cls, emb = model.encode(x_seq)
        act_seq = actions[:n_seq][None]
        act_emb = model.action_encoder(act_seq)
        pred_seq = model.predict(emb[None], act_emb)
        fwd = time.time() - t
        w.add("seq", [p.name for p in imgs[:n_seq]],
              {"input": (x_seq, "TCHW (frames in order)"), "emb_seq": (emb, "[T, D]"), "action_seq": (act_seq[0], "[T, action_dim]"),
               "act_emb_seq": (act_emb[0], "[T, D]"), "pred_seq": (pred_seq[0], "[T, D]")},
              {"preprocess_s": 0.0, "forward_s": round(fwd, 4)}, frames=n_seq)
    w.finish(load_s)


# --------------------------------------------------------------------------------------------------
# V-JEPA 2 ViT-L/16 (HF) — encoder + default predictor pass ; and the SSv2 classifier
# --------------------------------------------------------------------------------------------------
def _vjepa2_preprocessing(proc, crop: int) -> dict:
    return {
        "description": f"VJEPA2VideoProcessor: uint8 RGB frames (T,H,W,3) -> per-frame resize so the SHORT side = {int(crop * 256 / 224)} "
                       f"(= int(crop*256/224)), aspect kept, torchvision.transforms.v2.functional.resize on the uint8 CHW tensor with BILINEAR + "
                       f"antialias=True (result rounded back to uint8) -> center crop {crop}x{crop} (top = int((H-{crop})/2), left = int((W-{crop})/2)) "
                       "-> x/255 -> ImageNet mean/std -> pixel_values_videos [B, T, 3, H, W]",
        "resize": {"shortest_edge": int(crop * 256 / 224), "resample": "bilinear", "antialias": True, "on_dtype": "uint8"},
        "center_crop": crop, "rescale": 1 / 255, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        "frame_sampling": "idx = round(linspace(0, T_total-1, n)) over all decoded frames (PyAV rgb24); indices stored per sample",
        "resampling_note": RESIZE_NOTE,
        "processor": describe_processor(proc)}


def _vjepa2_hparams(cfg) -> dict:
    h = {"embed_dim": cfg.hidden_size, "n_layer": cfg.num_hidden_layers, "n_head": cfg.num_attention_heads,
         "ffn_dim": int(cfg.hidden_size * cfg.mlp_ratio), "patch_size": cfg.patch_size, "tubelet_size": cfg.tubelet_size,
         "img_size": cfg.crop_size, "n_frames": cfg.frames_per_clip, "ln_eps": cfg.layer_norm_eps, "act": _act_name(cfg.hidden_act),
         "qkv_bias": cfg.qkv_bias, "cls_token": False, "pos_type": "rope3d", "rope_theta": 10000.0,
         "pred": {"embed_dim": cfg.pred_hidden_size, "n_layer": cfg.pred_num_hidden_layers, "n_head": cfg.pred_num_attention_heads,
                  "ffn_dim": int(cfg.pred_hidden_size * cfg.pred_mlp_ratio), "n_mask_tokens": cfg.pred_num_mask_tokens}}
    if getattr(cfg, "num_pooler_layers", None) and "ForVideoClassification" in cfg.architectures[0]:
        h["head"] = {"kind": "attentive_pool", "n_pool_layers": cfg.num_pooler_layers, "n_classes": cfg.num_labels}
    return h


def _clip_samples(a, frames_list):
    """Yield (clip_path, n_frames, frames_u8 THWC, frame_idx, fps, n_total, decode_s)."""
    for clip in list_clips(a.media_dir, a.n_clips):
        t = time.time()
        all_frames, fps = decode_video(clip)
        dec = time.time() - t
        for n in frames_list:
            fr, idx = sample_frames(all_frames, n)
            yield clip, n, fr, idx, fps, len(all_frames), dec


def dump_vjepa2(a) -> None:
    import torch
    from transformers import AutoVideoProcessor, VJEPA2Model

    name = "vjepa2-vitl-fpc64-256"
    d = a.models_dir / MODEL_DIRS[name]
    t0 = time.time()
    proc = AutoVideoProcessor.from_pretrained(d)
    model = VJEPA2Model.from_pretrained(d, dtype=torch.float32).eval()
    load_s = time.time() - t0
    cfg = model.config
    w = RefWriter(
        a.out / name, name, d, hf_id="facebook/vjepa2-vitl-fpc64-256", hparams=_vjepa2_hparams(cfg),
        preprocessing=_vjepa2_preprocessing(proc, cfg.crop_size),
        outputs={"frames_u8": "the sampled RGB frames fed to the processor, THWC uint8",
                 "input": "processor pixel_values_videos, layout NTCHW (batch, frames, channels, H, W); the model permutes to NCTHW internally",
                 "last_hidden_state": "VJEPA2Model last_hidden_state[0]: [N, 1024] with N = T/2 * 16 * 16 tokens, order t-major then h then w, after final LN",
                 "pooled_mean": "mean over tokens",
                 "predictor_last_hidden_state": "VJEPA2Model(...).predictor_output.last_hidden_state[0] with the DEFAULT masks "
                                                "(context_mask = target_mask = arange(N), i.e. full context, predict all tokens): "
                                                "[N, 1024] = predictor_proj(predictor_norm(...)) for the N target (mask) tokens"},
    )
    for clip, n, fr, idx, fps, n_total, dec in _clip_samples(a, a.frames):
        t = time.time()
        x = proc(videos=fr, return_tensors="pt")["pixel_values_videos"].float()
        pre = time.time() - t
        t = time.time()
        with torch.inference_mode():
            out = model(pixel_values_videos=x)
        fwd = time.time() - t
        lhs = out.last_hidden_state[0]
        pred = out.predictor_output.last_hidden_state[0]
        w.add(f"{clip.stem}_f{n}", clip.name,
              {"frames_u8": (fr, "THWC uint8"), "input": (x, "NTCHW"), "last_hidden_state": (lhs, "[N_tokens, D] t-major,h,w"),
               "pooled_mean": (lhs.mean(0), "[D]"), "predictor_last_hidden_state": (pred, "[N_tokens, D]")},
              {"decode_s": round(dec, 4), "preprocess_s": round(pre, 4), "forward_s": round(fwd, 4)},
              frames=n, frame_indices=idx, n_frames_total=n_total, fps=fps, frame_size_hw=[int(fr.shape[1]), int(fr.shape[2])])
    w.finish(load_s)


def dump_vjepa2_ssv2(a) -> None:
    import torch
    from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

    name = "vjepa2-vitl-fpc16-256-ssv2"
    d = a.models_dir / MODEL_DIRS[name]
    t0 = time.time()
    proc = AutoVideoProcessor.from_pretrained(d)
    model = VJEPA2ForVideoClassification.from_pretrained(d, dtype=torch.float32).eval()
    load_s = time.time() - t0
    cfg = model.config
    labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
    holder = {}
    model.pooler.register_forward_hook(lambda m, i, o: holder.__setitem__("pooled", o))
    # encoder output AFTER its final LayerNorm (output_hidden_states[-1] would be pre-norm)
    model.vjepa2.encoder.register_forward_hook(
        lambda m, i, o: holder.__setitem__("lhs", o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]))
    w = RefWriter(
        a.out / name, name, d, hf_id="facebook/vjepa2-vitl-fpc16-256-ssv2", hparams=_vjepa2_hparams(cfg),
        preprocessing=_vjepa2_preprocessing(proc, cfg.crop_size), labels=labels,
        outputs={"frames_u8": "sampled RGB frames, THWC uint8", "input": "processor pixel_values_videos, NTCHW",
                 "last_hidden_state": "encoder output [N, 1024] (predictor skipped)",
                 "pooled": "VJEPA2AttentivePooler output [1024] (3 self-attn layers over the tokens, then 1 cross-attn from the learned query), "
                           "captured with a forward hook = the classifier input",
                 "logits": "classifier(pooled) [n_classes]", "top5_idx": "argsort(-logits)[:5] int64"},
    )
    for clip, n, fr, idx, fps, n_total, dec in _clip_samples(a, a.frames):
        t = time.time()
        x = proc(videos=fr, return_tensors="pt")["pixel_values_videos"].float()
        pre = time.time() - t
        t = time.time()
        with torch.inference_mode():
            out = model(pixel_values_videos=x)
        fwd = time.time() - t
        logits = out.logits[0]
        top5 = torch.argsort(logits, descending=True)[:5]
        tensors = {"frames_u8": (fr, "THWC uint8"), "input": (x, "NTCHW"), "last_hidden_state": (holder["lhs"][0], "[N_tokens, D] t-major,h,w"),
                   "pooled": (holder["pooled"][0], "[D]"), "logits": (logits, "[n_classes]"), "top5_idx": (top5.to(torch.int64), "[5]")}
        w.add(f"{clip.stem}_f{n}", clip.name, tensors,
              {"decode_s": round(dec, 4), "preprocess_s": round(pre, 4), "forward_s": round(fwd, 4)},
              frames=n, frame_indices=idx, n_frames_total=n_total, fps=fps, frame_size_hw=[int(fr.shape[1]), int(fr.shape[2])],
              top5=[{"idx": int(i), "label": labels[int(i)], "logit": round(float(logits[i]), 4)} for i in top5])
    w.finish(load_s)


# --------------------------------------------------------------------------------------------------
# V-JEPA 2.1 ViT-B/16 @384 (Meta code, torch.hub local)
# --------------------------------------------------------------------------------------------------
def dump_vjepa2_1(a) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import VJEPA2VideoProcessor

    name = "vjepa2_1-vitb-384"
    ckpt = a.models_dir / MODEL_DIRS[name]
    src = a.vjepa2_src
    if not (src / "hubconf.py").exists():
        print(f"  cloning {VJEPA2_SRC_URL} -> {src}")
        subprocess.run(["git", "clone", "--depth", "1", VJEPA2_SRC_URL, str(src)], check=True)
    t0 = time.time()
    encoder, predictor = torch.hub.load(str(src), "vjepa2_1_vit_base_384", source="local", pretrained=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)

    def clean(state):
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}

    ckpt_keys = list(sd.keys())
    encoder.load_state_dict(clean(sd["ema_encoder"]), strict=True)  # hub uses checkpoint_key="ema_encoder" for 2.1 vitb
    predictor.load_state_dict(clean(sd["predictor"]), strict=True)
    encoder = encoder.float().eval()
    ckpt_epoch = sd.get("epoch") if isinstance(sd, dict) else None
    del sd
    load_s = time.time() - t0
    crop = 384
    proc = VJEPA2VideoProcessor(crop_size=crop)  # -> shortest_edge 438, bilinear, ImageNet stats
    blk = encoder.blocks[0]
    w = RefWriter(
        a.out / name, name, ckpt, source_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        code=f"{VJEPA2_SRC_URL} (hubconf vjepa2_1_vit_base_384, app/vjepa_2_1/models/vision_transformer.py), local clone at {src}",
        checkpoint={"top_level_keys": ckpt_keys, "encoder_key_used": "ema_encoder", "key_cleanup": "strip 'module.' and 'backbone.'",
                    "epoch": ckpt_epoch},
        hparams={"embed_dim": encoder.embed_dim, "n_layer": len(encoder.blocks), "n_head": encoder.num_heads, "ffn_dim": blk.mlp.fc1.out_features,
                 "patch_size": encoder.patch_size, "tubelet_size": encoder.tubelet_size, "img_size": crop, "n_frames": encoder.num_frames,
                 "ln_eps": blk.norm1.eps, "act": "gelu_erf", "qkv_bias": blk.attn.qkv.bias is not None, "cls_token": False,
                 "pos_type": "rope3d", "rope_theta": 10000.0, "rope_interpolate": blk.attn.interpolate_rope,
                 "rope_pretrained_grid": blk.attn.pretrained_grid_size, "rope_dims_per_axis": [blk.attn.d_dim, blk.attn.h_dim, blk.attn.w_dim],
                 "modality_embed": encoder.modality_embedding, "image_patch_embed": encoder.patch_embed_img is not None,
                 "hier_layers": encoder.hierarchical_layers, "out_layers_distillation": encoder.out_layers_distillation,
                 "pred": {"embed_dim": 384, "n_layer": len(predictor.predictor_blocks), "n_mask_tokens": len(predictor.mask_tokens),
                          "teacher_embed_dim": predictor.predictor_proj.out_features}},
        preprocessing={**_vjepa2_preprocessing(proc, crop),
                       "note": "No HF processor exists for 2.1; we reuse VJEPA2VideoProcessor(crop_size=384) (short side 438, bilinear+antialias "
                               "on uint8, center crop 384, ImageNet stats) which mirrors Meta's eval transform (Resize(int(size*256/224)) + "
                               "CenterCrop(size)).  Images go through the same processor as a 1-frame video.  The tensor is then permuted to "
                               "NCTHW because the Meta encoder takes (B, C, T, H, W)."},
        outputs={"input": "NCTHW float32; video: T=16 -> tubelets of 2 -> 8 x 24 x 24 = 4608 tokens; image: T=1 -> patch_embed_img (1x16x16) "
                          "+ img_mod_embed -> 24 x 24 = 576 tokens",
                 "last_hidden_state": "encoder(x)[0] in default inference mode = norms_block[-1](x after the last block) [N, 768]; "
                                      "token order t-major then h then w",
                 "pooled_mean": "mean over tokens"},
    )
    with torch.inference_mode():
        for clip, n, fr, idx, fps, n_total, dec in _clip_samples(a, a.frames):
            t = time.time()
            x = proc(videos=fr, return_tensors="pt")["pixel_values_videos"].float().permute(0, 2, 1, 3, 4).contiguous()
            pre = time.time() - t
            t = time.time()
            y = encoder(x)
            fwd = time.time() - t
            lhs = y[0]
            w.add(f"{clip.stem}_f{n}", clip.name,
                  {"frames_u8": (fr, "THWC uint8"), "input": (x, "NCTHW"), "last_hidden_state": (lhs, "[N_tokens, D] t-major,h,w"),
                   "pooled_mean": (lhs.mean(0), "[D]")},
                  {"decode_s": round(dec, 4), "preprocess_s": round(pre, 4), "forward_s": round(fwd, 4)},
                  frames=n, frame_indices=idx, n_frames_total=n_total, fps=fps, frame_size_hw=[int(fr.shape[1]), int(fr.shape[2])], path="video")
        for p in list_images(a.media_dir, a.n_images):
            im = Image.open(p).convert("RGB")
            t = time.time()
            fr = np.asarray(im, dtype=np.uint8)[None]  # (1, H, W, 3)
            x = proc(videos=fr, return_tensors="pt")["pixel_values_videos"].float().permute(0, 2, 1, 3, 4).contiguous()  # (1,3,1,384,384)
            pre = time.time() - t
            t = time.time()
            y = encoder(x)
            fwd = time.time() - t
            lhs = y[0]
            w.add(p.stem, p.name, {"input": (x, "NCTHW (T=1, image path)"), "last_hidden_state": (lhs, "[N_tokens, D] h-major"),
                                   "pooled_mean": (lhs.mean(0), "[D]")},
                  {"preprocess_s": round(pre, 4), "forward_s": round(fwd, 4)}, image_size_hw=[im.height, im.width], path="image")
    w.finish(load_s)


DUMPERS = {
    "ijepa-vith14-1k": dump_ijepa,
    "lejepa-vits16": dump_lejepa,
    "lewm-pusht": dump_lewm,
    "vjepa2-vitl-fpc64-256": dump_vjepa2,
    "vjepa2-vitl-fpc16-256-ssv2": dump_vjepa2_ssv2,
    "vjepa2_1-vitb-384": dump_vjepa2_1,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(DUMPERS) + ["all"])
    ap.add_argument("--n-images", type=int, default=None, help="images per image model (default: 8; lewm / vjepa2_1: 2)")
    ap.add_argument("--n-clips", type=int, default=2, help="clips per video model (default 2: archery, bowling)")
    ap.add_argument("--frames", type=str, default=None, help="comma list of frames-per-clip variants (default: fpc64 -> 64,16; others -> 16)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repo root (default: this checkout)")
    ap.add_argument("--models-dir", type=Path, default=None, help="default <root>/models")
    ap.add_argument("--media-dir", type=Path, default=None, help="default <root>/tests/fixtures/media")
    ap.add_argument("--out", type=Path, default=None, help="default <root>/tests/fixtures/ref")
    ap.add_argument("--vjepa2-src", type=Path, default=None, help="local clone of facebookresearch/vjepa2 (default <root>/tmp/vjepa2-src)")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args(argv)
    a.root = a.root.resolve()
    a.models_dir = (a.models_dir or a.root / "models").resolve()
    a.media_dir = (a.media_dir or a.root / "tests" / "fixtures" / "media").resolve()
    a.out = (a.out or a.root / "tests" / "fixtures" / "ref").resolve()
    a.vjepa2_src = (a.vjepa2_src or a.root / "tmp" / "vjepa2-src").resolve()
    setup_env(a.root, a.threads)

    import torch

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    names = list(DUMPERS) if a.model == "all" else [a.model]
    n_images_arg, frames_arg = a.n_images, a.frames
    report = []
    for name in names:
        a.n_images = n_images_arg if n_images_arg is not None else DEFAULT_N_IMAGES.get(name, 8)
        a.frames = [int(f) for f in frames_arg.split(",")] if frames_arg else DEFAULT_FRAMES.get(name, [16])
        print(f"== {name}  (threads={torch.get_num_threads()}, out={a.out / name})")
        t = time.time()
        DUMPERS[name](a)
        report.append((name, time.time() - t))
    print("== wall time per model:")
    for name, s in report:
        print(f"  {name:28s} {s:8.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
