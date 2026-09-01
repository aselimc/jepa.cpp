#!/usr/bin/env python3
"""PyTorch-on-the-same-card baseline for the V-JEPA 2-AC predictor and rollout.

    scripts/torch_ac_baseline.py [--device cuda:1] [--candidates 1,16,64] [--horizon 2]
                                 [--repeat 5] [--warmup 2] [--dtype float32|float16]
                                 [--src tmp/vjepa2-src] [--ckpt models/vjepa2_ac/vjepa2-ac-vitg.pt]
                                 [--json out.json]

Times exactly what `jepa-bench --mode ac` and `--mode ac-rollout` time, on the same GPU, so the two
numbers are comparable: Meta's own `VisionTransformerPredictorAC` under `torch.no_grad()`, with the
world model's non-affine LayerNorm between rollout steps, on synthetic latents of the right shape.
The encoder is not run — a planner encodes once and scores thousands of candidates against that one
encode, which is the shape both harnesses measure.

Protocol, mirroring scripts/bench_gpu.sh: `--warmup` unmeasured calls, then the MINIMUM of
`--repeat` measured ones, each bracketed by `torch.cuda.synchronize()`. `pose` updates are excluded
from the rollout timing on both sides (jepa_ac_rollout does them on the host between graphs; here
the same host-side loop runs, so the comparison is like for like).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--src", type=Path, default=None, help="clone of facebookresearch/vjepa2 (default <root>/tmp/vjepa2-src)")
    ap.add_argument("--ckpt", type=Path, default=None, help="default <root>/models/vjepa2_ac/vjepa2-ac-vitg.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--candidates", default="1,16,64")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--frames", type=int, default=1, help="context frames for the single-call mode")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    a.src = a.src or a.root / "tmp" / "vjepa2-src"
    a.ckpt = a.ckpt or a.root / "models" / "vjepa2_ac" / "vjepa2-ac-vitg.pt"

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(a.src))
    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        sys.exit("no CUDA device")

    t0 = time.time()
    _, predictor = torch.hub.load(str(a.src), "vjepa2_ac_vit_giant", source="local", pretrained=False)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    predictor.load_state_dict({k.replace("module.", ""): v for k, v in sd["predictor"].items()}, strict=True)
    del sd
    dtype = getattr(torch, a.dtype)
    predictor = predictor.to(dev).to(dtype).eval()
    load_s = time.time() - t0

    HW = predictor.grid_height * predictor.grid_width
    D = predictor.predictor_embed.in_features
    A = predictor.action_encoder.in_features
    S = predictor.state_encoder.in_features
    torch.manual_seed(1234)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def timed(fn):
        for _ in range(a.warmup):
            fn()
        sync()
        best = None
        for _ in range(a.repeat):
            sync()
            t = time.perf_counter()
            fn()
            sync()
            dt = (time.perf_counter() - t) * 1e3
            best = dt if best is None else min(best, dt)
        return best

    rows = []
    for K in [int(x) for x in a.candidates.split(",")]:
        z1 = F.layer_norm(torch.randn(K, a.frames * HW, D, device=dev, dtype=dtype), (D,))
        act = torch.randn(K, a.frames, A, device=dev, dtype=dtype) * 0.05
        st = torch.randn(K, a.frames, S, device=dev, dtype=dtype)
        with torch.no_grad():
            ms = timed(lambda: predictor(z1, act, st))
        rows.append(dict(mode="ac", candidates=K, frames=a.frames, ms_min=round(ms, 3),
                         ms_per_candidate=round(ms / K, 4)))
        print(f"  ac          K={K:<3d} frames={a.frames}  {ms:9.2f} ms  ({ms / K:7.3f} ms/candidate)")

        z0 = F.layer_norm(torch.randn(K, HW, D, device=dev, dtype=dtype), (D,))
        acts = torch.randn(K, a.horizon, A, device=dev, dtype=dtype) * 0.05
        st0 = torch.randn(K, 1, S, device=dev, dtype=dtype)

        def rollout():
            z, s = z0, st0
            for h in range(a.horizon):
                aa = acts[:, : h + 1]
                nxt = predictor(z, aa, s)[:, -HW:]
                nxt = F.layer_norm(nxt, (nxt.size(-1),))
                z = torch.cat([z, nxt], 1)
                s = torch.cat([s, s[:, -1:]], 1)   # the pose update is host-side on both sides

        with torch.no_grad():
            ms = timed(rollout)
        rows.append(dict(mode="ac-rollout", candidates=K, horizon=a.horizon, ms_min=round(ms, 3),
                         ms_per_step=round(ms / a.horizon, 4),
                         ms_per_candidate_step=round(ms / (a.horizon * K), 4)))
        print(f"  ac-rollout  K={K:<3d} H={a.horizon}       {ms:9.2f} ms  "
              f"({ms / a.horizon:7.3f} ms/step, {ms / (a.horizon * K):7.3f} ms/candidate-step)")

    out = dict(tool="torch_ac_baseline.py", device=str(dev),
               gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor(),
               torch=torch.__version__, dtype=a.dtype, repeat=a.repeat, warmup=a.warmup,
               tokens_per_frame=HW, embed_dim=D, load_s=round(load_s, 2), rows=rows)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
