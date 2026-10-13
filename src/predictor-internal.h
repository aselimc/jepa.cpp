// Internal declarations shared by the predictors:
//   src/predictor.cpp — V-JEPA 2 / 2.1 masked predictor (jepa.pred.kind = "masked")
//   src/lewm.cpp      — LeWM AdaLN-zero world model  (jepa.pred.kind = "lewm")
//   src/hjepa.cpp     — H-JEPA: one LeWM-style predictor per level (jepa.family = "hjepa")
// Public entry points live in include/jepa.h; everything here is jepa.cpp-internal.
#pragma once
#include "jepa-internal.h"
#include "rope3d.h"

#include <vector>

// ---------------------------------------------------------------------------------------------
// masked predictor (predictor.cpp)
// ---------------------------------------------------------------------------------------------
// RoPE parameters of the predictor of `m`: grid (gt, gh, gw) on which the token ids are decoded,
// head_dim = jepa.pred.head_dim (32 for V-JEPA 2 / 2.1), layout / interpolation from jepa.pred.*.
//   gt is derived from `max_id` (the largest token id the caller passes), gh/gw from
//   jepa.pred.grid_size (2.1) or jepa.enc.img_size / patch_size (V-JEPA 2, what HF hard-codes).
jepa_rope3d_params jepa_predictor_rope_params(const jepa_model & m, int64_t max_id);

// Build the masked-predictor graph on ctx->ctx_g.
//   inp      : [enc_dim, n_ctx] F32 input tensor (the encoder rows at the context ids), caller-created
//   modality : JEPA_MODALITY_VIDEO / _IMAGE — picks pred.mod_embed_video vs pred.mod_embed_img for the
//              2.1 files (already resolved: _AUTO must not reach here)
//   cos/sin  : [head_dim, 1, n_ctx + n_tgt] F32 input tensors (jepa_rope3d_tables_ids of ctx ids then tgt ids)
//   returns  : [out_dim, n_tgt] F32 (the target rows after pred.norm and pred.proj)
ggml_tensor * jepa_build_predictor_masked(jepa_context * ctx, ggml_tensor * inp, int n_tgt, int mask_index,
                                          int modality, ggml_tensor * cos_t, ggml_tensor * sin_t);

// ---------------------------------------------------------------------------------------------
// LeWM world model (lewm.cpp)
// ---------------------------------------------------------------------------------------------
// The adaLN-zero causal transformer LeWM introduced, as a list of tensors: the LeWM predictor
// (pred.*) and every level of an H-JEPA hierarchy (lvl.{k}.pred.*) are both one of these.
// Optional tensors are nullptr.
struct jepa_adaln_spec {
    ggml_tensor * pos = nullptr;                                   // [D_state, n_frames] added to the states
    ggml_tensor * in_w = nullptr,   * in_b = nullptr;              // state -> transformer width (H-JEPA)
    ggml_tensor * cond_w = nullptr, * cond_b = nullptr;            // conditioning -> transformer width (H-JEPA)
    const std::vector<jepa_layer> * layers = nullptr;              // ln1 / ln2 may be absent (adaln_bare)
    ggml_tensor * norm_w = nullptr, * norm_b = nullptr;            // final LayerNorm
    ggml_tensor * out_w = nullptr,  * out_b = nullptr;             // transformer width -> output (H-JEPA)
    ggml_tensor * proj0_w = nullptr, * proj0_b = nullptr;          // projector MLP on the output
    ggml_tensor * proj2_w = nullptr, * proj2_b = nullptr;
    int   n_head = 0, head_dim = 0;
    float ln_eps = 1e-5f, adaln_eps = 1e-6f;
    jepa_act_id act = JEPA_ACT_GELU_ERF, proj_act = JEPA_ACT_GELU_ERF;
};

// Build that transformer on ctx->ctx_g. emb: [D_state, T] F32 states, cond: [D_cond, T] F32 the
// per-step conditioning (the embedded action), mask: [T, T] causal additive mask or nullptr for
// T == 1. Returns [D_state, T] F32, row t = the predicted next state given steps 0..t.
ggml_tensor * jepa_build_adaln(jepa_context * ctx, const jepa_adaln_spec & s, ggml_tensor * emb,
                               ggml_tensor * cond, ggml_tensor * mask);

// The [T, T] F16 causal mask those graphs take (query i sees key j iff j <= i), as host data.
void jepa_causal_mask_f16(int64_t T, std::vector<ggml_fp16_t> & out);

// Build the LeWM predictor graph on ctx->ctx_g. emb: [D, T] F32, act: [action_dim, T] F32,
// mask: [T, T] causal additive mask (F16 for flash, F32 for the naive path) or nullptr for T == 1.
// Returns [D, T] F32: row t is pred.proj(...) of frame t, i.e. the predicted *next* projected
// embedding given frames 0..t (the attention is causal over frames).
ggml_tensor * jepa_build_lewm(jepa_context * ctx, ggml_tensor * emb, ggml_tensor * act, ggml_tensor * mask);

// ---------------------------------------------------------------------------------------------
// V-JEPA 2-AC action-conditioned predictor (vjepa2_ac.cpp)
// ---------------------------------------------------------------------------------------------
// RoPE parameters of the AC predictor over `n_frames` frames: the (n_frames, grid, grid) grid the
// token ids below are decoded on, head_dim = jepa.pred.embed_dim / n_head (64), tiled layout.
jepa_rope3d_params jepa_ac_rope_params(const jepa_model & m, int n_frames);

// Build the AC predictor graph on ctx->ctx_g. The context is split in two because a planning
// rollout's observed frames are the same bytes for every candidate:
//   shared  : [enc_dim, n_shared*tokens_per_frame, 1] F32, broadcast over the batch (may be a
//             device-resident tensor a cached context handle owns), or nullptr
//   tail    : [enc_dim, n_tail*tokens_per_frame, B] F32, the per-candidate frames, or nullptr
//   act_in  : [action_dim, n_frames, B] F32, st_in: [state_dim, n_frames, B] F32 (always per-item)
//   mask    : [N, N] F16 block-causal mask over frames (nullptr for one frame), N = n_frames*(K+HW)
//   cos/sin : [head_dim, 1, N] F32 (jepa_rope3d_tables_ids on the interleaved id list)
//   n_out   : trailing frames to project (n_frames for every prefix, 1 for the next frame only)
//   returns : [jepa.pred.out_dim, n_out*tokens_per_frame, B] F32
ggml_tensor * jepa_build_ac(jepa_context * ctx, ggml_tensor * shared, ggml_tensor * tail,
                            ggml_tensor * act_in, ggml_tensor * st_in, ggml_tensor * mask,
                            ggml_tensor * cos_t, ggml_tensor * sin_t, int n_out);
