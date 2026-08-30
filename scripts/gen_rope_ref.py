#!/usr/bin/env python3
"""Golden vectors for the V-JEPA 3-D RoPE kernel (src/rope3d.{h,cpp}, tests/test-ops.cpp).

The expected outputs are captured from the *real* attention modules (identity q/k/v projections,
the attention call stubbed out to record the rotated q), not from a re-derivation of the math:

  variant 0  tag "hf"    HF `VJEPA2RopeAttention.forward` (transformers) for square grids, cross-checked
                         against Meta V-JEPA 2 `RoPEAttention.forward` (src/models/utils/modules.py);
                         the non-square grid can only be expressed by Meta's module (H_patches/W_patches).
  variant 1  tag "v21"   Meta V-JEPA 2.1 `RoPEAttention.forward` (app/vjepa_2_1/models/utils/modules.py),
                         interpolate_rope=False
  variant 1  tag "v21i"  same, interpolate_rope=True with the "pretrained grid" set to the brief's
                         train grid 32x24x24 (Meta's constructor hard-codes 256/patch = 16; see rope3d.h)

The cos/sin tables are extracted from the same forward passes by probing with x = [1,0,1,0,...] and
x = [0,1,0,1,...] (out = x*cos + rotate90(x)*sin picks out cos/sin exactly), then stored factorised
per axis ([2, gt+gh+gw, d], cos then sin) since a token's row only depends on its (t, h, w).

Usage (main venv, from the repo root):
  .venv/bin/python scripts/gen_rope_ref.py [--out tests/vectors/rope3d] [--vjepa2-src tmp/vjepa2-src]
"""
import argparse
import contextlib
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(32)

# name, head_dim, (grid_t, grid_h, grid_w), n_subsampled tokens (0 = all tokens)
CONFIGS = [
    ("hd64_g4x4x4", 64, (4, 4, 4), 0),
    ("hd64_g8x16x16_sub", 64, (8, 16, 16), 32),
    ("hd80_g2x3x5", 80, (2, 3, 5), 0),
]
TRAIN_GRID = (32, 24, 24)
THETA = 10000.0
PATCH = 16


# --- capture the rotated q by stubbing the attention call -------------------------------------------
class Capture:
    def __init__(self):
        self.q = None
        self.k = None

    def sdpa(self, q, k, v, *args, **kwargs):
        self.q, self.k = q.detach().clone(), k.detach().clone()
        return v


@contextlib.contextmanager
def capture_attention():
    """Route every scaled_dot_product_attention call (HF sdpa path and Meta's use_sdpa path) into a recorder."""
    cap = Capture()
    F = torch.nn.functional
    orig = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = cap.sdpa
    # Meta's modules wrap the call in `torch.backends.cuda.sdp_kernel()`, deprecated/removed in new torch.
    cuda_backend = torch.backends.cuda
    had_ctx = hasattr(cuda_backend, "sdp_kernel")
    orig_ctx = getattr(cuda_backend, "sdp_kernel", None)
    cuda_backend.sdp_kernel = contextlib.nullcontext
    try:
        yield cap
    finally:
        F.scaled_dot_product_attention = orig
        if had_ctx:
            cuda_backend.sdp_kernel = orig_ctx
        else:
            del cuda_backend.sdp_kernel


def set_identity_qkv(attn, hd, fused):
    eye = torch.eye(hd, dtype=torch.float32)
    with torch.no_grad():
        if fused:
            attn.qkv.weight.copy_(torch.cat([eye, eye, eye], 0))
            if attn.qkv.bias is not None:
                attn.qkv.bias.zero_()
        else:
            for lin in (attn.query, attn.key, attn.value):
                lin.weight.copy_(eye)
                if lin.bias is not None:
                    lin.bias.zero_()


# --- the three reference forwards ---------------------------------------------------------------------
def hf_forward(hf, x, hd, grid, ids):
    """HF VJEPA2RopeAttention (square grids only). x: [1, N, hd]; ids: LongTensor [N] or None."""
    from transformers import VJEPA2Config

    gt, gh, gw = grid
    assert gh == gw, "HF hard-codes a square spatial grid"
    cfg = VJEPA2Config(hidden_size=hd, num_attention_heads=1, crop_size=gh * PATCH, patch_size=PATCH,
                       frames_per_clip=gt * 2, tubelet_size=2)
    cfg._attn_implementation = "sdpa"
    attn = hf.VJEPA2RopeAttention(cfg, hd, 1).eval()
    assert attn.grid_size == gh and attn.grid_depth == gt and attn.d_dim == 2 * ((hd // 3) // 2)
    set_identity_qkv(attn, hd, fused=False)
    with torch.no_grad(), capture_attention() as cap:
        attn(x, position_mask=None if ids is None else ids[None])
    assert cap.q is not None and torch.equal(cap.q, cap.k)
    return cap.q[0, 0]  # [N, hd]


def meta_v2_forward(m2, x, hd, grid, ids):
    """Meta V-JEPA 2 RoPEAttention (tiled cos/sin, supports H_patches != W_patches)."""
    gt, gh, gw = grid
    attn = m2.RoPEAttention(dim=hd, num_heads=1, qkv_bias=True, grid_size=gh, use_sdpa=True).eval()
    set_identity_qkv(attn, hd, fused=True)
    with torch.no_grad(), capture_attention() as cap:
        attn(x, mask=None if ids is None else ids[None], T=gt, H_patches=gh, W_patches=gw)
    assert cap.q is not None and torch.equal(cap.q, cap.k)
    return cap.q[0, 0]


def meta_v21_forward(m21, x, hd, grid, ids, interpolate, train_grid):
    """Meta V-JEPA 2.1 RoPEAttention (interleaved cos/sin, optional interpolate_rope)."""
    gt, gh, gw = grid
    attn = m21.RoPEAttention(dim=hd, num_heads=1, qkv_bias=True, grid_size=gh, use_sdpa=True,
                             interpolate_rope=interpolate, patch_size=PATCH).eval()
    assert attn.pretrained_grid_size == 256 // PATCH  # modules.py:164-167, hard-coded from the patch size
    assert train_grid[1] == train_grid[2], "Meta has a single pretrained_grid_size for h and w"
    attn.pretrained_grid_size = train_grid[1]
    set_identity_qkv(attn, hd, fused=True)
    with torch.no_grad(), capture_attention() as cap:
        attn(x, mask=None if ids is None else ids[None], T=gt, H_patches=gh, W_patches=gw)
    assert cap.q is not None and torch.equal(cap.q, cap.k)
    return cap.q[0, 0]


# --- independent glue re-implementation used as a cross-check of the capture -----------------------------
def glue_reference(rotate, x, hd, grid, ids, interleaved, interpolate, train_grid):
    """Rotate with the reference `rotate_queries_or_keys` and explicit (t,h,w) ids. x: [N, hd]."""
    gt, gh, gw = grid
    if ids is None:
        ids = torch.arange(gt * gh * gw)
    t = ids // (gh * gw)
    h = (ids - (gh * gw) * t) // gw
    w = (ids - (gh * gw) * t) - gw * h
    if interleaved:  # V-JEPA 2.1: float positions, optional rescale (modules.py:197, 227-233)
        t, h, w = 1.0 * t, 1.0 * h, 1.0 * w
        if interpolate:
            h = h * (train_grid[1] - 1) / (gh - 1)
            w = w * (train_grid[2] - 1) / (gw - 1)
    d = 2 * ((hd // 3) // 2)
    xx = x[None, None]  # [1, 1, N, hd]
    parts = []
    for a, pos in enumerate((t, h, w)):
        chunk = xx[..., a * d:(a + 1) * d]
        if interleaved:
            parts.append(rotate(chunk, pos=pos, n_registers=0, has_cls_first=False))
        else:
            parts.append(rotate(chunk, pos=pos))
    if 3 * d < hd:
        parts.append(xx[..., 3 * d:])
    return torch.cat(parts, dim=-1)[0, 0]


# --- table extraction ---------------------------------------------------------------------------------------
def extract_tables(forward, hd, n_full):
    """Probe with x=[1,0,1,0..] and x=[0,1,0,1..]: returns cos, sin of shape [n_full, hd] exactly."""
    a = torch.zeros(1, n_full, hd)
    b = torch.zeros(1, n_full, hd)
    a[..., 0::2] = 1.0
    b[..., 1::2] = 1.0
    out_a = forward(a).numpy()  # even j: cos[j];  odd j: sin[j]
    out_b = forward(b).numpy()  # even j: -sin[j]; odd j: cos[j]
    cos = out_a.copy()
    sin = out_b.copy()
    cos[:, 1::2] = out_b[:, 1::2]
    sin[:, 0::2] = -out_b[:, 0::2]
    sin[:, 1::2] = out_a[:, 1::2]
    return cos, sin


def factorise(tab, grid, d, fill):
    """[N, hd] table -> [gt+gh+gw, d] per-axis rows; asserts the table really factorises + tail == fill."""
    gt, gh, gw = grid
    n = gt * gh * gw
    ids = np.arange(n)
    t = ids // (gh * gw)
    h = (ids % (gh * gw)) // gw
    w = ids % gw
    rows = []
    for a, (coord, g) in enumerate(((t, gt), (h, gh), (w, gw))):
        blk = tab[:, a * d:(a + 1) * d]
        for v in range(g):
            sel = blk[coord == v]
            assert np.array_equal(sel, np.broadcast_to(sel[0], sel.shape)), f"axis {a} coord {v} not constant"
            rows.append(sel[0])
    assert np.all(tab[:, 3 * d:] == fill), "untouched dims must be cos=1 / sin=0"
    return np.stack(rows).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(root, "tests", "vectors", "rope3d"))
    # `git clone --depth 1 https://github.com/facebookresearch/vjepa2 tmp/vjepa2-src` (git-ignored)
    src_default = os.path.join(root, "tmp", "vjepa2-src")
    if not os.path.isdir(src_default):  # worktrees: fall back to the main checkout's shared clone
        src_default = "/home/overseer2/workdir/jepa.cpp/tmp/vjepa2-src"
    ap.add_argument("--vjepa2-src", default=src_default)
    args = ap.parse_args()

    sys.path.insert(0, args.vjepa2_src)
    import transformers
    import transformers.models.vjepa2.modeling_vjepa2 as hf
    import src.models.utils.modules as m2  # noqa: E402  (Meta V-JEPA 2)
    import app.vjepa_2_1.models.utils.modules as m21  # noqa: E402  (Meta V-JEPA 2.1)

    os.makedirs(args.out, exist_ok=True)
    manifest = ["# name tag variant head_dim grid_t grid_h grid_w n_tok interpolate train_t train_h train_w theta",
                f"# torch {torch.__version__} transformers {transformers.__version__} vjepa2-src {args.vjepa2_src}"]
    torch.manual_seed(0)
    total = 0

    def save(name, arr):
        nonlocal total
        path = os.path.join(args.out, name)
        np.save(path, arr, allow_pickle=False)
        total += os.path.getsize(path)

    for name, hd, grid, n_sub in CONFIGS:
        gt, gh, gw = grid
        n_full = gt * gh * gw
        d = 2 * ((hd // 3) // 2)
        ids = None
        if n_sub:
            ids = torch.randperm(n_full)[:n_sub].sort().values
        n_tok = n_sub or n_full
        x = torch.randn(1, n_tok, hd)
        save(f"{name}_x.npy", x[0].numpy())
        if ids is not None:
            save(f"{name}_ids.npy", ids.numpy().astype(np.int32))

        cases = []
        # variant 0: HF / Meta V-JEPA 2 (tiled)
        if gh == gw:
            fwd_hf = lambda xx: hf_forward(hf, xx, hd, grid, ids)
            fwd_v2 = lambda xx: meta_v2_forward(m2, xx, hd, grid, ids)
            out_hf, out_v2 = fwd_hf(x), fwd_v2(x)
            diff = (out_hf - out_v2).abs().max().item()
            assert diff <= 1e-6, f"HF vs Meta V-JEPA 2 mismatch {diff}"
            print(f"[{name}] hf: HF forward vs Meta V-JEPA 2 forward max|diff| = {diff:.3g}")
            fwd0, fwd0_full = fwd_hf, (lambda xx: hf_forward(hf, xx, hd, grid, None))
        else:
            print(f"[{name}] hf: non-square grid -> Meta V-JEPA 2 forward (HF class cannot express it)")
            fwd0, fwd0_full = (lambda xx: meta_v2_forward(m2, xx, hd, grid, ids)), \
                              (lambda xx: meta_v2_forward(m2, xx, hd, grid, None))
        cases.append(("hf", 0, False, fwd0, fwd0_full, hf.rotate_queries_or_keys))
        # variant 1: Meta V-JEPA 2.1 (interleaved), interpolate off / on
        for tag, interp in (("v21", False), ("v21i", True)):
            cases.append((tag, 1, interp,
                          lambda xx, i=interp: meta_v21_forward(m21, xx, hd, grid, ids, i, TRAIN_GRID),
                          lambda xx, i=interp: meta_v21_forward(m21, xx, hd, grid, None, i, TRAIN_GRID),
                          m21.rotate_queries_or_keys))

        for tag, variant, interp, fwd, fwd_full, rotate in cases:
            out = fwd(x)
            glue = glue_reference(rotate, x[0], hd, grid, ids, variant == 1, interp, TRAIN_GRID)
            diff = (out - glue).abs().max().item()
            assert diff <= 1e-6, f"{name}/{tag}: captured forward vs glue mismatch {diff}"
            cos, sin = extract_tables(fwd_full, hd, n_full)
            axes = np.stack([factorise(cos, grid, d, 1.0), factorise(sin, grid, d, 0.0)])
            save(f"{name}_{tag}_out.npy", out.numpy())
            save(f"{name}_{tag}_axes.npy", axes)
            tt, th, tw = TRAIN_GRID if interp else (0, 0, 0)
            manifest.append(f"{name} {tag} {variant} {hd} {gt} {gh} {gw} {n_tok} {int(interp)} {tt} {th} {tw} {THETA:g}")
            print(f"[{name}] {tag}: N={n_tok} d={d} forward-vs-glue max|diff|={diff:.3g} "
                  f"tables {axes.shape} |out|max={out.abs().max().item():.3f}")

    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"wrote {len(manifest) - 2} cases to {args.out}, {total / 1024:.1f} KB of .npy")


if __name__ == "__main__":
    main()
