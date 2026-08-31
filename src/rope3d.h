// V-JEPA 2 / V-JEPA 2.1 3-axis rotary position embedding (RoPE) for ggml (CPU inference).
//
// Two reference implementations exist and they are NOT numerically identical; both are supported
// via jepa_rope3d_params::variant. All line numbers below refer to
//   [HF]   transformers 5.16.1, transformers/models/vjepa2/modeling_vjepa2.py
//   [V2]   facebookresearch/vjepa2 @ 204698b, src/models/utils/modules.py
//   [V21]  facebookresearch/vjepa2 @ 204698b, app/vjepa_2_1/models/utils/modules.py
//
// Shared math (identical in all three sources)
// ------------------------------------------------------------------------------------------------
//   head_dim D, per-axis rotated width d = 2 * ((D // 3) // 2)              [HF 238-240] [V21 158-160]
//   dims [0,d) rotate with the frame index t, [d,2d) with the row index h, [2d,3d) with the column
//   index w; dims [3d, D) are passed through untouched                       [HF 279-294] [V21 235-280]
//   token i -> t = i / (gh*gw), h = (i % (gh*gw)) / gw, w = i % gw (T-major, then H, then W)
//                                                                            [HF 258-277] [V2 316-329]
//   frequencies (per axis, float32):  omega_k = 1 / theta^(k / (d/2)),  k in [0, d/2), theta = 10000
//                                                                            [HF 186-188] [V21 28-30]
//   angle:                             ang_k = pos * omega_k                 [HF 189]     [V21 31]
//   rotation on INTERLEAVED pairs (x[2k], x[2k+1]) via rotate90(y1,y2) = (-y2, y1):
//       out = x * C + rotate90(x) * S                                        [HF 199-204] [V21 39-44]
//   i.e. out[2k]   = x[2k]   * C[2k]   - x[2k+1] * S[2k]
//        out[2k+1] = x[2k+1] * C[2k+1] + x[2k]   * S[2k+1]
//   where C, S are per-dimension cos/sin vectors of length d that differ between the variants:
//
// variant 0 = JEPA_ROPE3D_VJEPA2  (HF VJEPA2RopeAttention, Meta V-JEPA 2, torch.hub vjepa2 models)
//   C[j] = cos(pos * omega[j % (d/2)]),  S[j] = sin(pos * omega[j % (d/2)])         (TILED)
//   from `emb_cos.repeat(1, 1, 1, 2)` [HF 195-196] [V2 43-44]. The two members of a pair therefore
//   use DIFFERENT frequencies (omega[2k mod d/2] vs omega[(2k+1) mod d/2]) -- Meta documents this
//   as "a subtle bug where frequencies are duplicated across the vector pair" that must be kept
//   for compatibility with the pretrained weights [V2 39-42]. It is not a proper rotation, so it
//   cannot be expressed with ggml_rope*. Positions are the raw integer grid indices.
//
// variant 1 = JEPA_ROPE3D_VJEPA2_1  (Meta V-JEPA 2.1 "corrected" RoPE)
//   C[j] = cos(pos * omega[j / 2]),      S[j] = sin(pos * omega[j / 2])             (INTERLEAVED)
//   from `emb_cos.repeat_interleave(2, dim=-1)` [V21 36-37]: a true 2-D rotation per pair.
//   interpolate_rope [V21 227-233]: positions become float and h / w are rescaled to the grid the
//   model was pretrained on (t is NOT rescaled):
//       h_pos = h * (train_grid_h - 1) / (grid_h - 1),   w_pos = w * (train_grid_w - 1) / (grid_w - 1)
//   NOTE: Meta hard-codes that "pretrained grid" from the patch size, NOT from img_size:
//       pretrained_grid_size = 252/14 = 18 (patch 14) or 256/16 = 16 (patch 16)   [V21 164-167]
//   and the released vjepa2_1_vit*_384 checkpoints are built with patch 16, img_size 384 and
//   interpolate_rope=True (src/hub/backbones.py 211-240), so at their native 24x24 grid they run
//   with h_pos = h * 15 / 23. Pass train_grid_h = train_grid_w = 16 for those checkpoints.
//   Registers / a leading CLS token (n_registers, has_cls_first) are not rotated in 2.1 [V21 20-48];
//   the released checkpoints use n_registers=0, has_cls_first=False (defaults, [V21 140-141]).
//
// Implementation
// ------------------------------------------------------------------------------------------------
//   Host side (jepa_rope3d_tables*) builds [N, D] cos/sin tables in the layout above, mirroring the
//   float32 op order of the references (cos = 1, sin = 0 for the untouched dims [3d, D)).  The sin
//   table carries the SIGN of the rotation already folded in (`signed_sin`, the default):
//       S'[j] = (j even) ? -S[j] : +S[j]
//   so that rotate90(x)*S == swap(x)*S' with swap(x)[j] = x[j ^ 1].
//   Graph side (jepa_rope3d_apply) is then out = x*C + swap(x)*S', five nodes:
//       cont (only when x is a view) -> reshape [2, ...] -> roll(1) -> 2x mul -> add
//   Rolling an axis of length 2 by 1 *is* the pair swap, with no wrap-around lane to mask away, so
//   no mask has to be built in the graph. Every node has a CUDA kernel and the whole chain is one
//   scheduler split on any backend (docs/gpu-notes.md S2); the older form (two ne0-wide rolls plus
//   in-graph arange/repeat/scale masks) had none for `roll` on the strided qkv view and fragmented
//   a 24-block encoder into 193 splits. The refactor is bit-identical for finite inputs: the only
//   difference against the old expression is that it also added an exact +0.0 per lane (the masked
//   -out half of each roll), and adding +0.0 to a finite float is exact (measured: max abs 0.0).
#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

enum jepa_rope3d_variant {
    JEPA_ROPE3D_VJEPA2   = 0, // HF VJEPA2RopeAttention / Meta V-JEPA 2 (tiled cos/sin)
    JEPA_ROPE3D_VJEPA2_1 = 1, // Meta V-JEPA 2.1 (interleaved cos/sin, optional interpolate_rope)
};

struct jepa_rope3d_params {
    int   grid_t = 1, grid_h = 1, grid_w = 1; // token grid of the input (N = grid_t*grid_h*grid_w, T-major)
    int   head_dim = 64;
    float theta = 10000.0f;
    bool  interpolate = false;                // V-JEPA 2.1 interpolate_rope: rescale h/w to the train grid
    int   train_grid_t = 0;                   // accepted for symmetry, UNUSED: Meta never rescales t
    int   train_grid_h = 0, train_grid_w = 0; // Meta's `pretrained_grid_size` (16 for patch 16, 18 for patch 14)
    int   variant = JEPA_ROPE3D_VJEPA2;       // jepa_rope3d_variant
};

// Per-axis rotated width d = 2 * ((head_dim / 3) / 2); 3*d dims are rotated, the rest untouched.
int jepa_rope3d_axis_dim(int head_dim);

// (t, h, w) position of token id `i` in the T-major grid, after interpolate_rope rescaling.
void jepa_rope3d_position(const jepa_rope3d_params & p, int64_t i, float & t, float & h, float & w);

// cos/sin tables for the full grid: row-major [N, head_dim] (N = grid_t*grid_h*grid_w), token order
// T-major/H/W. Upload as a ggml tensor of shape [head_dim, 1, N] (ne0 = head_dim) for jepa_rope3d_apply.
// `signed_sin` (the default) negates the even lanes of sin_out, which is what jepa_rope3d_apply
// consumes; pass false to get the raw sin(angle) of the reference (tests/test-ops golden vectors).
void jepa_rope3d_tables(const jepa_rope3d_params & p, std::vector<float> & cos_out, std::vector<float> & sin_out,
                        bool signed_sin = true);

// Same, but only for the `n_ids` tokens whose grid ids are given (predictor / masked-token paths,
// HF `position_mask`): row r corresponds to grid token ids[r]. Output is [n_ids, head_dim].
void jepa_rope3d_tables_ids(const jepa_rope3d_params & p, const int32_t * ids, int n_ids,
                            std::vector<float> & cos_out, std::vector<float> & sin_out,
                            bool signed_sin = true);

// Build the rotation in a ggml graph.
//   x      : F32 [head_dim, n_head, N]; rows (ne0) must be contiguous, strides over n_head / N are
//            free, so a view into a fused qkv projection is fine (one ggml_cont is inserted for it).
//            head_dim must be even.
//   cos_t  : F32 [head_dim, 1, N] (the tables above), broadcast over n_head
//   sin_t  : F32 [head_dim, 1, N] — the SIGNED table (see "Implementation" above)
// Returns a new contiguous F32 tensor [head_dim, n_head, N]. Backend-agnostic: cont (only when x is
// a view) + roll + 2 mul + add, all of which have CPU and CUDA kernels.
struct ggml_tensor * jepa_rope3d_apply(struct ggml_context * ctx, struct ggml_tensor * x,
                                       struct ggml_tensor * cos_t, struct ggml_tensor * sin_t);
