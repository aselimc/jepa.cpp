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
  vjepa2-vitg-fpc64-256       HF VJEPA2Model ViT-g/16 (facebook/vjepa2-vitg-fpc64-256), same code path
  vjepa2-ac-vitg              V-JEPA 2-AC world model (Meta code, vjepa2-ac-vitg.pt + the Franka demo trajectory)
  vjepa2-vitl-fpc16-256-ssv2  HF VJEPA2ForVideoClassification (attentive pooler + linear head)
  vjepa2_1-vitb-384           Meta code path (torch.hub, local clone of facebookresearch/vjepa2)
  levjepa-vitl16              LeVJEPAModel via trust_remote_code  (galilai-group/LeVJEPA-VideoMix-Large)
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
    "vjepa2-vitg-fpc64-256": "facebook/vjepa2-vitg-fpc64-256",
    "vjepa2-ac-vitg": "vjepa2_ac/vjepa2-ac-vitg.pt",
    "vjepa2-vitl-fpc16-256-ssv2": "facebook/vjepa2-vitl-fpc16-256-ssv2",
    "vjepa2_1-vitb-384": "vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt",
    "levjepa-vitl16": "galilai-group/LeVJEPA-VideoMix-Large",
}
DEFAULT_N_IMAGES = {"ijepa-vith14-1k": 8, "lejepa-vits16": 8, "lewm-pusht": 2, "vjepa2_1-vitb-384": 2,
                    "levjepa-vitl16": 2}
DEFAULT_FRAMES = {"vjepa2-vitl-fpc64-256": [64, 16], "vjepa2-vitg-fpc64-256": [64, 16],
                  "vjepa2-vitl-fpc16-256-ssv2": [16], "vjepa2_1-vitb-384": [16], "levjepa-vitl16": [16],
                  "vjepa2-ac-vitg": [2]}
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


def dump_vjepa2(a, name: str = "vjepa2-vitl-fpc64-256") -> None:
    import torch
    from transformers import AutoVideoProcessor, VJEPA2Model

    d = a.models_dir / MODEL_DIRS[name]
    t0 = time.time()
    proc = AutoVideoProcessor.from_pretrained(d)
    model = VJEPA2Model.from_pretrained(d, dtype=torch.float32).eval()
    load_s = time.time() - t0
    cfg = model.config
    w = RefWriter(
        a.out / name, name, d, hf_id=f"facebook/{name}", hparams=_vjepa2_hparams(cfg),
        preprocessing=_vjepa2_preprocessing(proc, cfg.crop_size),
        outputs={"frames_u8": "the sampled RGB frames fed to the processor, THWC uint8",
                 "input": "processor pixel_values_videos, layout NTCHW (batch, frames, channels, H, W); the model permutes to NCTHW internally",
                 "last_hidden_state": f"VJEPA2Model last_hidden_state[0]: [N, {cfg.hidden_size}] with N = T/2 * 16 * 16 tokens, "
                                      "order t-major then h then w, after final LN",
                 "pooled_mean": "mean over tokens",
                 "predictor_last_hidden_state": "VJEPA2Model(...).predictor_output.last_hidden_state[0] with the DEFAULT masks "
                                                "(context_mask = target_mask = arange(N), i.e. full context, predict all tokens): "
                                                f"[N, {cfg.hidden_size}] = predictor_proj(predictor_norm(...)) for the N target (mask) tokens"},
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


# --------------------------------------------------------------------------------------------------
# LeVJEPA ViT-L/16 (custom architecture shipped with the weights, trust_remote_code)
# --------------------------------------------------------------------------------------------------
LEVJEPA_RESIZE_NOTE = (
    "no preprocessor_config.json ships with this checkpoint, so the pipeline is the model card's: short side -> 224 "
    "keeping aspect (long side = int(224*long/short)), centre crop 224, /255, ImageNet mean/std. The resampler used "
    "here is torchvision.transforms.v2.functional.resize(BICUBIC, antialias=True) on the uint8 CHW tensor rounded back "
    "to uint8 -- the same reference resampler the other jepa.cpp fixtures use and the one src/preprocess.cpp targets. "
    "The model card's notebook uses PIL BICUBIC instead; the two differ by at most one uint8 level on a fraction of a "
    "percent of pixels (docs/parity.md quotes the measured effect on token cosine).")


def levjepa_frames_to_tensor(frames, resize_short: int, crop: int):
    """uint8 THWC frames -> normalised NCTHW float32 [1, 3, T, crop, crop]."""
    import numpy as np
    import torch
    import torchvision.transforms.v2.functional as F

    t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)      # T C H W, uint8
    t = F.resize(t, [resize_short], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
    t = F.center_crop(t, [crop, crop])
    x = t.float() / 255.0
    x = (x - torch.tensor(IMAGENET_MEAN)[None, :, None, None]) / torch.tensor(IMAGENET_STD)[None, :, None, None]
    return x.permute(1, 0, 2, 3)[None].contiguous()                              # 1 C T H W


def dump_levjepa(a) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    name = "levjepa-vitl16"
    d = a.models_dir / MODEL_DIRS[name]
    t0 = time.time()
    model = AutoModel.from_pretrained(d, trust_remote_code=True, dtype=torch.float32).eval()
    load_s = time.time() - t0
    cfg = model.config
    enc = model.encoder
    crop = int(cfg.img_size)
    n_frames = int(cfg.num_frames)
    grid = crop // int(cfg.patch_size)
    w = RefWriter(
        a.out / name, name, d, hf_id="galilai-group/LeVJEPA-VideoMix-Large",
        code="modeling_levjepa.py + configuration_levjepa.py from the same HF repo (trust_remote_code=True)",
        hparams={"embed_dim": cfg.embed_dim, "n_layer": cfg.depth, "n_head": cfg.num_heads,
                 "ffn_dim": int(cfg.embed_dim * cfg.mlp_ratio), "patch_size": cfg.patch_size,
                 "tubelet_size": cfg.tubelet_size, "img_size": crop, "n_frames": n_frames,
                 "ln_eps": enc.blocks[0].norm1.eps, "act": "gelu_erf",
                 "qkv_bias": enc.blocks[0].attn.qkv.bias is not None, "cls_token": True, "n_registers": 0,
                 "pos_type": "rope3d", "rope_theta": 10000.0, "rope_interpolate": False,
                 "rope_freq_layout": "tiled",
                 "rope_dims_per_axis": [enc.blocks[0].attn.d_dim, enc.blocks[0].attn.h_dim, enc.blocks[0].attn.w_dim],
                 "attn_mode": cfg.attn_mode, "layer_scale": False,
                 "n_tokens": 1 + (n_frames // cfg.tubelet_size) * grid * grid},
        preprocessing={
            "description": LEVJEPA_RESIZE_NOTE,
            "resize": {"shortest_edge": crop, "resample": "bicubic", "antialias": True, "on_dtype": "uint8"},
            "center_crop": crop, "rescale": 1 / 255, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "frame_sampling": "idx = round(linspace(0, T_total-1, n)) over all decoded frames (PyAV rgb24); indices "
                              "stored per sample. A still image is turned into a clip by repeating the frame n times, "
                              "which is what the model card prescribes.",
            "resampling_note": LEVJEPA_RESIZE_NOTE},
        outputs={"frames_u8": "the sampled RGB frames fed to the preprocessor, THWC uint8 (image samples: the one frame "
                              "repeated n_frames times, so the stored input and our own preprocessing describe the same clip)",
                 "input": "normalised pixels, layout NCTHW (batch, channels, frames, H, W) — LeVJEPAModel takes NCTHW directly",
                 "last_hidden_state": "LeVJEPAModel last_hidden_state[0]: [1+N, 1024], CLS first then t-major/h/w patch "
                                      "tokens, after the final LayerNorm",
                 "cls": "pooler_output = last_hidden_state[:, 0], the feature this model is used through",
                 "pooled_mean": "mean over the patch tokens (CLS excluded)"},
    )

    def run(sample_name, media, rgb, timing, **extra):
        t = time.time()
        x = levjepa_frames_to_tensor(rgb, crop, crop)
        pre = time.time() - t
        t = time.time()
        with torch.inference_mode():
            out = model(pixel_values=x)
        fwd = time.time() - t
        lhs = out.last_hidden_state[0]
        timing.update(preprocess_s=round(pre, 4), forward_s=round(fwd, 4))
        w.add(sample_name, media,
              {"frames_u8": (rgb, "THWC uint8"), "input": (x, "NCTHW"),
               "last_hidden_state": (lhs, "[1+N_tokens, D] CLS first, then t-major,h,w"),
               "cls": (lhs[0], "[D]"), "pooled_mean": (lhs[1:].mean(0), "[D]")},
              timing, **extra)

    for clip, n, fr, idx, fps, n_total, dec in _clip_samples(a, a.frames):
        run(f"{clip.stem}_f{n}", clip.name, fr, {"decode_s": round(dec, 4)},
            frames=n, frame_indices=idx, n_frames_total=n_total, fps=fps,
            frame_size_hw=[int(fr.shape[1]), int(fr.shape[2])], path="video")
    # the still-image path of the model card: one frame repeated n_frames times
    for p in list_images(a.media_dir, a.n_images):
        im = Image.open(p).convert("RGB")
        fr = np.repeat(np.asarray(im, dtype=np.uint8)[None], n_frames, axis=0)
        run(p.stem, p.name, fr, {}, frames=n_frames, image_size_hw=[im.height, im.width], path="image")
    w.finish(load_s)


# --------------------------------------------------------------------------------------------------
# V-JEPA 2-AC (action-conditioned world model, Meta code path)
# --------------------------------------------------------------------------------------------------
# Everything here runs facebookresearch/vjepa2's own modules on its own demo trajectory:
#   encoder / predictor  torch.hub vjepa2_ac_vit_giant (src/hub/backbones.py::_make_vjepa2_ac_model),
#                        weights from vjepa2-ac-vitg.pt ("encoder" and "predictor", 'module.' stripped)
#   trajectory           notebooks/franka_example_traj.npz (observations uint8 [1,T,256,256,3], states [1,T,7])
#   transform            make_transforms(scale=(1,1), ratio=(1,1), crop=256): a centre crop of the
#                        largest square followed by a bilinear resize to 256 (both the identity on the
#                        square Franka renders), then (x - 255*mean)/(255*std).  Reimplemented here
#                        because app/vjepa_droid/transforms.py imports cv2, which the venv does not have.
#   world-model loop     notebooks/utils/world_model_wrapper.py: a frame is encoded as a 2-frame clip
#                        (the frame repeated along T) and every latent is passed through a non-affine
#                        LayerNorm before and after the predictor.
AC_CANDIDATES = 4      # action sequences scored per planning step (the batch axis of jepa_ac_rollout)
AC_HORIZON = 2         # rollout steps (the notebook's mpc_args["rollout"])
# CEM fixture (notebooks/utils/mpc_utils.py::cem). Small on purpose: the point is the ALGORITHM --
# sampling order, elite selection, the momentum update and the clamps -- not a converged plan.
AC_CEM = dict(samples=8, topk=4, cem_steps=3, rollout=2, maxnorm=0.05,
              momentum_mean=0.15, momentum_std=0.75,
              momentum_mean_gripper=0.15, momentum_std_gripper=0.15)


def _ac_normalize(t):
    import torch.nn.functional as F
    return F.layer_norm(t, (t.size(-1),))


def _ac_step(world, reps, actions, poses, tpf):
    """WorldModel.infer_next_action's `step_predictor`, lifted out so cem() can be called directly."""
    import torch.nn.functional as F
    from utils.mpc_utils import compute_new_pose
    B, T, N_T, D = reps.size()
    nxt = world.predictor(reps.flatten(1, 2), actions, poses)[:, -tpf:]
    nxt = F.layer_norm(nxt, (nxt.size(-1),))
    return nxt.view(B, 1, N_T, D), compute_new_pose(poses[:, -1:], actions[:, -1:])


def dump_vjepa2_ac(a) -> None:
    import numpy as np
    import torch

    name = "vjepa2-ac-vitg"
    ckpt = a.models_dir / MODEL_DIRS[name]
    src = a.vjepa2_src
    if not (src / "hubconf.py").exists():
        print(f"  cloning {VJEPA2_SRC_URL} -> {src}")
        subprocess.run(["git", "clone", "--depth", "1", VJEPA2_SRC_URL, str(src)], check=True)
    traj_path = src / "notebooks" / "franka_example_traj.npz"
    if not traj_path.exists():
        sys.exit(f"{traj_path} is missing (it ships with the vjepa2 repo)")
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(src / "notebooks"))

    t0 = time.time()
    encoder, predictor = torch.hub.load(str(src), "vjepa2_ac_vit_giant", source="local", pretrained=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)

    def clean(state):
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}

    ckpt_keys = list(sd.keys())
    encoder.load_state_dict(clean(sd["encoder"]), strict=False)   # the .pt has no pos_embed; RoPE model
    predictor.load_state_dict(clean(sd["predictor"]), strict=True)
    encoder = encoder.float().eval()
    predictor = predictor.float().eval()
    ckpt_epoch = sd.get("epoch") if isinstance(sd, dict) else None
    del sd
    load_s = time.time() - t0

    crop = 256
    P = encoder.patch_size
    tpf = (crop // P) ** 2                       # tokens_per_frame, the notebook's `tokens_per_frame`
    D = encoder.embed_dim
    pblk = predictor.predictor_blocks[0]

    traj = np.load(traj_path)
    obs = traj["observations"][0]                # [T, 256, 256, 3] uint8
    st_np = traj["states"]                       # [1, T, 7]
    n_traj = len(obs)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32) * 255.0
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32) * 255.0

    def transform(frames_u8):
        """make_transforms(...)(frames) for a square source: THWC uint8 -> CTHW float32."""
        x = torch.tensor(np.ascontiguousarray(frames_u8), dtype=torch.float32).permute(3, 0, 1, 2)
        C, T, H, W = x.shape
        if H != W or H != crop:   # the reference's centre-crop-to-square + bilinear resize
            side = min(H, W)
            x = x[:, :, (H - side) // 2: (H - side) // 2 + side, (W - side) // 2: (W - side) // 2 + side]
            x = torch.nn.functional.interpolate(x, size=(crop, crop), mode="bilinear", align_corners=False)
        return ((x.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2).contiguous()

    clip = transform(obs)                        # [3, T, 256, 256]

    def encode(frame_idx):
        """world_model_wrapper.WorldModel.encode: one frame -> a 2-frame clip -> tpf tokens."""
        c = clip[:, frame_idx: frame_idx + 1]                     # [3, 1, H, W]
        x = c.unsqueeze(0).permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        return x, encoder(x)[0]                                   # [1,3,2,H,W], [tpf, D]

    from utils import mpc_utils
    from utils.mpc_utils import compute_new_pose, poses_to_diff
    from utils.world_model_wrapper import WorldModel

    # The ground-truth first action of the trajectory (energy_landscape_example.ipynb cell 3), plus
    # three fixed perturbations along x / y / z: a deterministic stand-in for a CEM candidate batch.
    gt_action = poses_to_diff(st_np[0, 0], st_np[0, 1]).float()    # [7]
    deltas = torch.tensor([[0.0, 0, 0, 0, 0, 0, 0],
                           [0.05, 0, 0, 0, 0, 0, 0],
                           [0.0, 0.05, 0, 0, 0, 0, 0],
                           [0.0, 0, 0.05, 0, 0, 0, 0]], dtype=torch.float32)
    cand = (gt_action[None] + deltas)                             # [K, 7]
    actions = cand[:, None, :].repeat(1, AC_HORIZON, 1)           # [K, H, 7] (each step repeats the action)
    state0 = torch.tensor(st_np[0, 0], dtype=torch.float32)       # [7]

    w = RefWriter(
        a.out / name, name, ckpt, source_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt",
        code=f"{VJEPA2_SRC_URL} (hubconf vjepa2_ac_vit_giant, src/models/ac_predictor.py, "
             f"notebooks/utils/mpc_utils.py + world_model_wrapper.py), local clone at {src}",
        checkpoint={"top_level_keys": ckpt_keys, "encoder_key_used": "encoder",
                    "key_cleanup": "strip 'module.' and 'backbone.'", "epoch": ckpt_epoch,
                    "encoder_equals_target_encoder": True},
        trajectory={"file": "notebooks/franka_example_traj.npz", "frames": int(n_traj),
                    "frame_size_hw": [int(obs.shape[1]), int(obs.shape[2])],
                    "states": [[float(v) for v in row] for row in st_np[0]]},
        hparams={"embed_dim": D, "n_layer": len(encoder.blocks), "n_head": encoder.num_heads,
                 "ffn_dim": encoder.blocks[0].mlp.fc1.out_features, "patch_size": P,
                 "tubelet_size": encoder.tubelet_size, "img_size": crop, "n_frames": encoder.num_frames,
                 "ln_eps": encoder.blocks[0].norm1.eps, "act": "gelu_erf", "cls_token": False,
                 "pos_type": "rope3d", "rope_theta": 10000.0, "tokens_per_frame": tpf,
                 "pred": {"kind": "ac", "embed_dim": predictor.predictor_embed.out_features,
                          "n_layer": len(predictor.predictor_blocks), "n_head": pblk.attn.num_heads,
                          "head_dim": pblk.attn.head_dim, "ffn_dim": pblk.mlp.fc1.out_features,
                          "ln_eps": pblk.norm1.eps, "action_dim": predictor.action_encoder.in_features,
                          "state_dim": predictor.state_encoder.in_features,
                          "out_dim": predictor.predictor_proj.out_features,
                          "grid_size": predictor.grid_height, "n_cond_tokens": 2,
                          "cond_order": "action,state", "frame_causal": predictor.is_frame_causal,
                          "use_extrinsics": predictor.use_extrinsics,
                          "attn_mask_shape": list(predictor.attn_mask.shape),
                          "rope_dims_per_axis": [pblk.attn.d_dim, pblk.attn.h_dim, pblk.attn.w_dim],
                          "rope_grid_size": pblk.attn.grid_size, "rope_freq_layout": "tiled"}},
        preprocessing={
            "description": "app/vjepa_droid/transforms.py::make_transforms(random_horizontal_flip=False, "
                           "random_resize_scale=(1,1), random_resize_aspect_ratio=(1,1), crop_size=256): "
                           "random_resized_crop degenerates to a centre crop of the largest square followed "
                           "by torch F.interpolate(bilinear, align_corners=False, no antialias) to 256x256, "
                           "then (x - 255*mean) / (255*std) on the CTHW float tensor. On the 256x256 Franka "
                           "renders both the crop and the resize are the identity.",
            # Recorded as shortest_edge/centre-crop because that is the same square, scaled the same
            # way, as Meta's crop-then-resize: only the discarded margin is resampled differently.
            # On the square Franka renders both orders are the identity, and the own-preprocessing
            # pass below measures the residual against the stored tensor.
            "resize": {"shortest_edge": crop, "resample": "bilinear", "antialias": False,
                       "on_dtype": "float32",
                       "reference_order": "centre-crop to the largest square, THEN resize to 256"},
            "center_crop": crop, "rescale": 1 / 255, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        world_model={
            "encode": "one observation frame -> a 2-frame clip (the frame repeated along T, tubelet 2) -> "
                      f"{tpf} tokens after the encoder's final LayerNorm; then a NON-AFFINE LayerNorm over "
                      "the feature dim (torch F.layer_norm default eps 1e-5). world_model_wrapper.py:42-52.",
            "predict": "predictor(context, actions, states) returns T*tpf rows; the world model keeps the "
                       "LAST tpf (the next frame) and normalises them again before feeding them back. "
                       "world_model_wrapper.py:56-64.",
            "state": "the pose of frame t; the pose after a step is compute_new_pose(pose, action) = "
                     "(xyz + d_xyz, euler_xyz(R(d) @ R(pose)), clip(gripper + d_gripper, 0, 1)). "
                     "mpc_utils.py:166-190.",
            "energy": "l1(a, b) = mean |a - b| over the flattened [tpf * D] final state and goal. "
                      "mpc_utils.py:17-18 and the loss_fn of energy_landscape_example.ipynb.",
            "candidates": f"{AC_CANDIDATES} fixed action sequences of {AC_HORIZON} steps: the trajectory's "
                          "ground-truth first action poses_to_diff(states[0], states[1]) and that action "
                          "shifted by +0.05 along x, y and z, each repeated over the horizon.",
        },
        outputs={"frames_u8": "the uint8 observation frame, repeated to the 2-frame clip, THWC",
                 "input": "NCTHW float32 (the 2-frame clip of one observation frame)",
                 "last_hidden_state": f"encoder(input)[0]: [{tpf}, {D}] tokens after the final LN, h-major then w",
                 "context": "F.layer_norm(last_hidden_state) -- the predictor's input",
                 "goal": "context of the LAST trajectory frame",
                 "action/state": "[7] the action driving the step and the pose at the frame",
                 "pred_next": f"predictor(context, action, state)[-{tpf}:] -- the next frame's latents, un-normalised",
                 "pred_seq": "predictor over T=2 frames, ALL T*tpf rows (row block t predicts frame t+1)",
                 "rollout": f"[K, H, {tpf}, {D}] normalised latents per candidate per step",
                 "rollout_energy": "[K] l1(rollout[:, -1], goal)",
                 "actions/states_seq": "[K, H, 7] the candidate actions and the poses compute_new_pose produced",
                 "next_state_ref": "[K, 7] compute_new_pose(state0, actions[:, 0]) -- the jepa_ac_next_state check"},
    )

    with torch.inference_mode():
        # ---- 1. encoder samples (also the encoder-parity fixture for the AC bundle)
        enc_rows = {}
        for idx, tag in ((0, "frame0"), (1, "frame1"), (n_traj - 1, "goalframe")):
            t = time.time()
            x, h = encode(idx)
            fwd = time.time() - t
            hn = _ac_normalize(h)
            enc_rows[tag] = hn
            # The uint8 frames the transform consumed, repeated along T exactly as `encode` does, so
            # test-parity's own-preprocessing pass can rerun src/preprocess.cpp on the real pixels.
            fr = np.repeat(np.ascontiguousarray(obs[idx])[None], 2, axis=0)   # [2, H, W, 3] uint8
            w.add(tag, f"franka_example_traj.npz[{idx}]",
                  {"frames_u8": (fr, "THWC uint8 (the observation frame, repeated to the 2-frame clip)"),
                   "input": (x, "NCTHW (T=2, the frame repeated)"),
                   "last_hidden_state": (h, f"[{tpf}, D] h-major then w"),
                   "pooled_mean": (h.mean(0), "[D]"),
                   "context": (hn, f"[{tpf}, D] after the non-affine LayerNorm")},
                  {"preprocess_s": 0.0, "forward_s": round(fwd, 4)}, frame_index=idx, frames=2)

        z0, goal = enc_rows["frame0"], enc_rows["goalframe"]

        # ---- 2. one predictor step (T = 1) with the ground-truth action
        t = time.time()
        p1 = predictor(z0[None], gt_action[None, None], state0[None, None])[0]
        fwd = time.time() - t
        w.add("step1", "franka_example_traj.npz[0]",
              {"context": (z0, f"[{tpf}, D]"), "action": (gt_action, "[7]"), "state": (state0, "[7]"),
               "pred_next": (p1, f"[{tpf}, D] predictor output, T=1")},
              {"preprocess_s": 0.0, "forward_s": round(fwd, 4)}, frames=1)

        # ---- 3. two frames (T = 2): every row, the block-causal check
        z1 = _ac_normalize(p1)
        s1 = compute_new_pose(state0[None, None], gt_action[None, None])[0, 0].float()
        ctx2 = torch.cat([z0, z1], 0)
        act2 = torch.stack([gt_action, gt_action])
        st2 = torch.stack([state0, s1])
        t = time.time()
        p2 = predictor(ctx2[None], act2[None], st2[None])[0]
        fwd = time.time() - t
        w.add("step2", "franka_example_traj.npz[0]",
              {"context_seq": (ctx2, f"[2*{tpf}, D]"), "action_seq": (act2, "[2, 7]"), "state_seq": (st2, "[2, 7]"),
               "pred_seq": (p2, f"[2*{tpf}, D] all rows; block t predicts frame t+1")},
              {"preprocess_s": 0.0, "forward_s": round(fwd, 4)}, frames=2)

        # ---- 4. K-candidate rollout + the planning energy
        t = time.time()
        z_hat = z0[None].repeat(AC_CANDIDATES, 1, 1)              # [K, tpf, D]
        s_hat = state0[None, None].repeat(AC_CANDIDATES, 1, 1)    # [K, 1, 7]
        a_hat = actions[:, :1]                                    # [K, 1, 7]
        roll, states_seq = [], []
        for h_step in range(AC_HORIZON):
            nxt = _ac_normalize(predictor(z_hat, a_hat, s_hat)[:, -tpf:])
            s_next = compute_new_pose(s_hat[:, -1:], a_hat[:, -1:]).float()
            roll.append(nxt)
            states_seq.append(s_next[:, 0])
            z_hat = torch.cat([z_hat, nxt], 1)
            s_hat = torch.cat([s_hat, s_next], 1)
            if h_step + 1 < AC_HORIZON:
                a_hat = torch.cat([a_hat, actions[:, h_step + 1: h_step + 2]], 1)
        fwd = time.time() - t
        roll = torch.stack(roll, 1)                               # [K, H, tpf, D]
        energy = (roll[:, -1] - goal[None]).abs().mean(dim=(1, 2))
        next_state_ref = compute_new_pose(state0[None, None].repeat(AC_CANDIDATES, 1, 1),
                                          actions[:, :1])[:, 0].float()
        w.add("rollout", "franka_example_traj.npz",
              {"context": (z0, f"[{tpf}, D] shared seed"), "goal": (goal, f"[{tpf}, D]"),
               "state0": (state0, "[7]"), "actions": (actions, f"[{AC_CANDIDATES}, {AC_HORIZON}, 7]"),
               "states_seq": (torch.stack(states_seq, 1), f"[{AC_CANDIDATES}, {AC_HORIZON}, 7]"),
               "rollout": (roll, f"[{AC_CANDIDATES}, {AC_HORIZON}, {tpf}, D]"),
               "rollout_energy": (energy, f"[{AC_CANDIDATES}]"),
               "next_state_ref": (next_state_ref, f"[{AC_CANDIDATES}, 7]")},
              {"preprocess_s": 0.0, "forward_s": round(fwd, 4)},
              candidates=AC_CANDIDATES, horizon=AC_HORIZON)

        # ---- 5. the same rollout from TWO observed context frames.
        # Meta's own loop never has more than one, so nothing upstream pins the conditioning of the
        # earlier frames; this sample fixes it and runs THEIR predictor on it, which is what makes
        # jepa_ac_rollout_ex(n_seed = 2) checkable. Frame 0 carries `seed_actions[0]` (the action that
        # actually took the arm from frame 0 to frame 1) and its own pose; frame 1 carries the
        # candidate's first planned action and the observed pose at frame 1.
        t = time.time()
        z1_obs = enc_rows["frame1"]
        seed_action = poses_to_diff(st_np[0, 0], st_np[0, 1]).float()          # frame 0 -> frame 1
        state1 = torch.tensor(st_np[0, 1], dtype=torch.float32)
        z_hat = torch.cat([z0, z1_obs], 0)[None].repeat(AC_CANDIDATES, 1, 1)   # [K, 2*tpf, D]
        s_hat = torch.stack([state0, state1])[None].repeat(AC_CANDIDATES, 1, 1)  # [K, 2, 7]
        a_hat = torch.cat([seed_action[None, None].repeat(AC_CANDIDATES, 1, 1),
                           actions[:, :1]], 1)                                 # [K, 2, 7]
        roll2, states2 = [], []
        for h_step in range(AC_HORIZON):
            nxt = _ac_normalize(predictor(z_hat, a_hat, s_hat)[:, -tpf:])
            s_next = compute_new_pose(s_hat[:, -1:], a_hat[:, -1:]).float()
            roll2.append(nxt)
            states2.append(s_next[:, 0])
            z_hat = torch.cat([z_hat, nxt], 1)
            s_hat = torch.cat([s_hat, s_next], 1)
            if h_step + 1 < AC_HORIZON:
                a_hat = torch.cat([a_hat, actions[:, h_step + 1: h_step + 2]], 1)
        fwd = time.time() - t
        roll2 = torch.stack(roll2, 1)
        w.add("rollout_seed2", "franka_example_traj.npz",
              {"context": (torch.cat([z0, z1_obs], 0), f"[2*{tpf}, D] two OBSERVED frames"),
               "goal": (goal, f"[{tpf}, D]"),
               "seed_actions": (seed_action[None], "[n_seed-1, 7] the action between the observed frames"),
               "seed_states": (torch.stack([state0, state1]), "[2, 7]"),
               "actions": (actions, f"[{AC_CANDIDATES}, {AC_HORIZON}, 7]"),
               "states_seq": (torch.stack(states2, 1), f"[{AC_CANDIDATES}, {AC_HORIZON}, 7]"),
               "rollout": (roll2, f"[{AC_CANDIDATES}, {AC_HORIZON}, {tpf}, D]"),
               "rollout_energy": ((roll2[:, -1] - goal[None]).abs().mean(dim=(1, 2)), f"[{AC_CANDIDATES}]")},
              {"preprocess_s": 0.0, "forward_s": round(fwd, 4)},
              candidates=AC_CANDIDATES, horizon=AC_HORIZON, n_seed=2)

        # ---- 6. Meta's CEM planner, with every random draw recorded.
        # torch's RNG cannot be replayed from C++, so the fixture stores the draws themselves:
        # `cem_noise` is exactly what `torch.randn` returned inside their loop, in their order
        # (iteration, then horizon step), and a planner that consumes those draws has to reproduce
        # `cem_action` at the end.
        t = time.time()
        draws = []
        real_randn = torch.randn

        def recording_randn(*a, **kw):
            v = real_randn(*a, **kw)
            draws.append(v.clone())
            return v

        world = WorldModel(encoder=encoder, predictor=predictor, tokens_per_frame=tpf,
                           transform=None, mpc_args={}, normalize_reps=True, device="cpu")
        real_logger_info = mpc_utils.logger.info
        mpc_utils.logger.info = lambda *a, **kw: None
        mpc_utils.torch.randn = recording_randn
        torch.manual_seed(1234)
        try:
            cem_action = mpc_utils.cem(context_frame=z0[None], context_pose=state0[None, None],
                                       goal_frame=goal[None],
                                       world_model=lambda r, a_, p_: _ac_step(world, r, a_, p_, tpf),
                                       verbose=False, **AC_CEM)[0]
        finally:
            mpc_utils.torch.randn = real_randn
            mpc_utils.logger.info = real_logger_info
        fwd = time.time() - t
        noise = torch.stack(draws, 0)   # [cem_steps * rollout, samples, 4]
        noise = noise.view(AC_CEM["cem_steps"], AC_CEM["rollout"], AC_CEM["samples"], -1)
        w.add("cem", "franka_example_traj.npz",
              {"context": (z0, f"[{tpf}, D]"), "goal": (goal, f"[{tpf}, D]"), "state0": (state0, "[7]"),
               "cem_noise": (noise, f"[{AC_CEM['cem_steps']}, {AC_CEM['rollout']}, {AC_CEM['samples']}, 4] "
                                    "every torch.randn draw, in Meta's order"),
               "cem_action": (cem_action, f"[{AC_CEM['rollout']}, 7] the plan cem() returned")},
              {"preprocess_s": 0.0, "forward_s": round(fwd, 4)}, cem=AC_CEM)
    w.finish(load_s)


DUMPERS = {
    "ijepa-vith14-1k": dump_ijepa,
    "lejepa-vits16": dump_lejepa,
    "lewm-pusht": dump_lewm,
    "vjepa2-vitl-fpc64-256": dump_vjepa2,
    "vjepa2-vitg-fpc64-256": lambda a: dump_vjepa2(a, "vjepa2-vitg-fpc64-256"),
    "vjepa2-ac-vitg": dump_vjepa2_ac,
    "vjepa2-vitl-fpc16-256-ssv2": dump_vjepa2_ssv2,
    "vjepa2_1-vitb-384": dump_vjepa2_1,
    "levjepa-vitl16": dump_levjepa,
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
