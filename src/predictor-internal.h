// Internal declarations shared by the two predictors:
//   src/predictor.cpp — V-JEPA 2 / 2.1 masked predictor (jepa.pred.kind = "masked")
//   src/lewm.cpp      — LeWM AdaLN-zero world model  (jepa.pred.kind = "lewm")
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
// Build the LeWM predictor graph on ctx->ctx_g. emb: [D, T] F32, act: [action_dim, T] F32,
// mask: [T, T] causal additive mask (F16 for flash, F32 for the naive path) or nullptr for T == 1.
// Returns [D, T] F32: row t is pred.proj(...) of frame t, i.e. the predicted *next* projected
// embedding given frames 0..t (the attention is causal over frames).
ggml_tensor * jepa_build_lewm(jepa_context * ctx, ggml_tensor * emb, ggml_tensor * act, ggml_tensor * mask);
