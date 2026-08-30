// V-JEPA 2 / V-JEPA 2.1 masked predictor (jepa.pred.kind == "masked").
//
// Semantics (HF VJEPA2Predictor / Meta vit_predictor; executable spec:
// scripts/jepa_convert/vjepa2_numpy_ref.py::predictor_forward, docs: scripts/jepa_convert/VJEPA_NOTES.md §3/§4.4):
//
//   ctx = enc[context_idx] @ pred.embed            # [n_ctx, pred_dim]   (2.1 ViT-g/G: 2-layer MLP embed.0/.2)
//   tgt = pred.mask_tokens[mask_index % n_mask]    # repeated for every target id
//   x   = concat(ctx, tgt)  [+ pred.mod_embed_video|img  (2.1, selected by `modality`)]
//   3-D RoPE from the *token ids* on the encoder grid (predictor head_dim 32 -> d = 10 per axis),
//   tiled cos/sin for V-JEPA 2, interleaved and NOT interpolated for 2.1 (grid = jepa.pred.grid_size)
//   12 pre-LN blocks with full attention over ctx+tgt, then pred.norm
//   out = x[n_ctx:] @ pred.proj                    # [n_tgt, enc_dim] (2.1: teacher dim)
//
// Attention is permutation-equivariant, so the sort-by-id that Meta's code does around the blocks is
// skipped here (the outputs are returned in the caller's target order either way).
#include "predictor-internal.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------------------------
// RoPE parameters
// ---------------------------------------------------------------------------------------------
jepa_rope3d_params jepa_predictor_rope_params(const jepa_model & m, int64_t max_id) {
    const jepa_enc_hparams  & e = m.hp.enc;
    const jepa_pred_hparams & p = m.hp.pred;
    jepa_rope3d_params rp;
    // HF hard-codes the spatial grid from the config (crop_size / patch_size); 2.1 stores it as
    // jepa.pred.grid_size (24 for the 384-px checkpoints). VJEPA_NOTES.md §4.4.
    const int grid = p.grid_size > 0 ? p.grid_size : e.grid_size();
    rp.grid_h = rp.grid_w = grid;
    const int64_t per_frame = (int64_t) grid * grid;
    rp.grid_t = (int) (max_id / per_frame) + 1;
    rp.head_dim = p.head_dim_eff();
    rp.theta = e.rope_theta;
    rp.interpolate = p.rope_interpolate;             // false for both released families
    rp.train_grid_h = rp.train_grid_w = p.rope_ref_grid;
    if (!p.rope_freq_layout.empty()) {
        rp.variant = p.rope_freq_layout == "interleaved" ? JEPA_ROPE3D_VJEPA2_1 : JEPA_ROPE3D_VJEPA2;
    } else {
        rp.variant = m.hp.family == JEPA_FAMILY_VJEPA2_1 ? JEPA_ROPE3D_VJEPA2_1 : JEPA_ROPE3D_VJEPA2;
    }
    return rp;
}

// ---------------------------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------------------------
ggml_tensor * jepa_build_predictor_masked(jepa_context * ctx, ggml_tensor * inp, int n_tgt, int mask_index,
                                          int modality, ggml_tensor * cos_t, ggml_tensor * sin_t) {
    const jepa_model * m = ctx->model;
    const jepa_pred_hparams & p = m->hp.pred;
    ggml_context * g = ctx->ctx_g;
    const int64_t n_ctx = inp ? inp->ne[1] : 0;   // inp == nullptr: no context, only mask tokens
    const int64_t Dp = p.embed_dim;

    // 1. context projection: single Linear, or the 2-layer MLP of the 2.1 ViT-g/G files
    ggml_tensor * x = nullptr;
    if (inp) {
        if (ggml_tensor * w0 = m->get("pred.embed.0.weight")) {
            x = jepa_build_mlp2(g, inp, w0, m->get("pred.embed.0.bias"),
                                m->require("pred.embed.2.weight"), m->get("pred.embed.2.bias"), p.act);
        } else {
            x = jepa_build_linear(g, inp, m->require("pred.embed.weight"), m->get("pred.embed.bias"));
        }
    }

    // 2. mask tokens at the target positions ([Dp, n_mask_tokens] -> one row, repeated)
    if (n_tgt > 0) {
        ggml_tensor * mt = m->require("pred.mask_tokens");
        const int n_mask = p.n_mask_tokens > 0 ? p.n_mask_tokens : (int) mt->ne[1];
        int idx = n_mask > 0 ? mask_index % n_mask : 0;
        if (idx < 0) idx += n_mask;
        ggml_tensor * row = ggml_cont(g, ggml_view_2d(g, mt, Dp, 1, mt->nb[1], (size_t) idx * mt->nb[1]));
        ggml_tensor * tgt = ggml_repeat_4d(g, row, Dp, n_tgt, 1, 1);
        x = n_ctx > 0 ? ggml_concat(g, x, tgt, 1) : tgt;
    }

    // 3. modality vector (2.1: added to context *and* mask tokens).  The image tokenizer path has its
    // own vector -- adding the video one instead is a silent two-digit error (mean cos 0.862, worst
    // row 0.655 on a 576-token image against the numpy spec, vs 1.0000000 with the right one).
    if (p.modality_embed) {
        const bool image = modality == JEPA_MODALITY_IMAGE;
        ggml_tensor * mod = m->get(image ? "pred.mod_embed_img" : "pred.mod_embed_video");
        if (!mod) {
            jepa_log("jepa: jepa_predict: jepa.pred.modality_embed is set but pred.mod_embed_%s is missing\n",
                     image ? "img" : "video");
        } else {
            x = ggml_add(g, x, mod);
        }
    }

    // 4. blocks with 3-D RoPE on q and k
    jepa_block_opts opts;
    // The reference (vjepa2_numpy_ref.py::predictor_forward) uses the *encoder* LN eps for the
    // predictor blocks; jepa.pred.ln_eps carries the same 1e-6 for every released file.
    opts.ln_eps = m->hp.enc.ln_eps;
    opts.act = p.act;
    opts.n_head = p.n_head;
    opts.head_dim = p.head_dim_eff();
    opts.attn.flash = ctx->params.use_flash_attn;
    opts.attn.kv_type = jepa_context_kv_type(ctx);
    opts.qk_hook = [&](ggml_context * gc, ggml_tensor * t, bool /*is_k*/) {
        return jepa_rope3d_apply(gc, t, cos_t, sin_t);
    };
    for (int i = 0; i < p.n_layer; i++) {
        x = jepa_build_block(g, x, m->pred_layers[i], opts);
    }
    x = jepa_build_layer_norm(g, x, m->require("pred.norm.weight"), m->get("pred.norm.bias"), opts.ln_eps);

    // 5. target rows -> pred.proj
    ggml_tensor * tgt_rows = x;
    if (n_ctx > 0 && n_tgt > 0) {
        tgt_rows = ggml_cont(g, ggml_view_2d(g, x, Dp, n_tgt, x->nb[1], (size_t) n_ctx * x->nb[1]));
    }
    ggml_tensor * out = jepa_build_linear(g, tgt_rows, m->require("pred.proj.weight"), m->get("pred.proj.bias"));
    ggml_set_name(out, "predictor_last_hidden_state");
    ggml_set_output(out);
    return out;
}

// ---------------------------------------------------------------------------------------------
// public entry point
// ---------------------------------------------------------------------------------------------
static int predict_masked(jepa_context * ctx, const jepa_output * enc,
                          const int32_t * context_idx, int n_context,
                          const int32_t * target_idx, int n_target,
                          int mask_index, int modality, jepa_output * out) {
    const jepa_model * m = ctx->model;
    const jepa_pred_hparams & p = m->hp.pred;
    const int64_t enc_dim = m->hp.enc.embed_dim;
    if (enc->dim != enc_dim) {
        jepa_log("jepa: jepa_predict: encoder output has dim %lld, model embed_dim is %lld\n",
                 (long long) enc->dim, (long long) enc_dim);
        return -1;
    }
    if (n_target <= 0 || n_context < 0) {
        jepa_log("jepa: jepa_predict: need n_target > 0 and n_context >= 0 (got %d / %d)\n", n_target, n_context);
        return -1;
    }

    // token ids: context first, then targets (the row order of the predictor input)
    std::vector<int32_t> ids((size_t) n_context + n_target);
    for (int i = 0; i < n_context; i++) ids[i] = context_idx ? context_idx[i] : i;
    for (int i = 0; i < n_target; i++)  ids[n_context + i] = target_idx ? target_idx[i] : i;
    int64_t max_id = 0;
    for (int32_t v : ids) {
        if (v < 0) { jepa_log("jepa: jepa_predict: negative token id %d\n", v); return -1; }
        if (v > max_id) max_id = v;
    }
    for (int i = 0; i < n_context; i++) {
        if (ids[i] >= enc->n_tokens) {
            jepa_log("jepa: jepa_predict: context id %d is outside the encoder output (%lld rows)\n",
                     ids[i], (long long) enc->n_tokens);
            return -1;
        }
    }

    // gather the context rows
    std::vector<float> rows((size_t) n_context * enc_dim);
    for (int i = 0; i < n_context; i++) {
        memcpy(rows.data() + (size_t) i * enc_dim, enc->data + (size_t) ids[i] * enc_dim, (size_t) enc_dim * sizeof(float));
    }

    // RoPE tables for ctx+tgt ids
    const jepa_rope3d_params rp = jepa_predictor_rope_params(*m, max_id);
    // AUTO: a single temporal slice on the predictor grid is the V-JEPA 2.1 image path
    // (24x24 = 576 ids for the 384-px checkpoints); anything longer is a clip.
    if (modality == JEPA_MODALITY_AUTO) {
        modality = (rp.grid_t == 1 && m->get("pred.mod_embed_img")) ? JEPA_MODALITY_IMAGE : JEPA_MODALITY_VIDEO;
    }
    std::vector<float> cosv, sinv;
    jepa_rope3d_tables_ids(rp, ids.data(), (int) ids.size(), cosv, sinv);

    const size_t nodes = (size_t) p.n_layer * 96 + 256;
    jepa_graph_begin(ctx, nodes);
    ggml_context * g = ctx->ctx_g;
    ggml_tensor * inp = nullptr;
    if (n_context > 0) {
        inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, enc_dim, n_context);
        ggml_set_name(inp, "context");
        ggml_set_input(inp);
    }
    ggml_tensor * cos_t = ggml_new_tensor_3d(g, GGML_TYPE_F32, rp.head_dim, 1, (int64_t) ids.size());
    ggml_tensor * sin_t = ggml_new_tensor_3d(g, GGML_TYPE_F32, rp.head_dim, 1, (int64_t) ids.size());
    ggml_set_input(cos_t);
    ggml_set_input(sin_t);

    ggml_tensor * y = jepa_build_predictor_masked(ctx, inp, n_target, mask_index, modality, cos_t, sin_t);
    ggml_build_forward_expand(ctx->gf, y);
    if (!jepa_graph_alloc(ctx)) return -1;
    if (inp) ggml_backend_tensor_set(inp, rows.data(), 0, rows.size() * sizeof(float));
    ggml_backend_tensor_set(cos_t, cosv.data(), 0, cosv.size() * sizeof(float));
    ggml_backend_tensor_set(sin_t, sinv.data(), 0, sinv.size() * sizeof(float));
    if (jepa_graph_compute(ctx) != 0) return -1;

    out->n_tokens = n_target;
    out->dim = y->ne[0];
    out->data = (float *) malloc((size_t) ggml_nbytes(y));
    if (!out->data) return -1;
    ggml_backend_tensor_get(y, out->data, 0, ggml_nbytes(y));
    return 0;
}

extern "C" int jepa_predict_mod(jepa_context * ctx, const jepa_output * enc,
                                const int32_t * context_idx, int n_context,
                                const int32_t * target_idx, int n_target,
                                int mask_index, int modality, jepa_output * out) {
    if (!ctx || !enc || !enc->data || !out) return -1;
    const jepa_model * m = ctx->model;
    if (!m->hp.pred.present) {
        jepa_log("jepa: jepa_predict: '%s' has no predictor\n", m->hp.name.c_str());
        return -1;
    }
    if (m->hp.pred.kind != "masked") {
        jepa_log("jepa: jepa_predict: predictor kind '%s' is not the masked predictor%s\n",
                 m->hp.pred.kind.c_str(), m->hp.pred.kind == "lewm" ? " (use jepa_lewm_predict)" : "");
        return -1;
    }
    if (modality != JEPA_MODALITY_AUTO && modality != JEPA_MODALITY_VIDEO && modality != JEPA_MODALITY_IMAGE) {
        jepa_log("jepa: jepa_predict: unknown modality %d (JEPA_MODALITY_AUTO/VIDEO/IMAGE)\n", modality);
        return -1;
    }
    return predict_masked(ctx, enc, context_idx, n_context, target_idx, n_target, mask_index, modality, out);
}

extern "C" int jepa_predict_ex(jepa_context * ctx, const jepa_output * enc,
                               const int32_t * context_idx, int n_context,
                               const int32_t * target_idx, int n_target,
                               int mask_index, jepa_output * out) {
    // video is the historical (and HF / Meta default) behaviour of this entry point
    return jepa_predict_mod(ctx, enc, context_idx, n_context, target_idx, n_target, mask_index,
                            JEPA_MODALITY_VIDEO, out);
}

extern "C" int jepa_predict(jepa_context * ctx, const jepa_output * enc,
                            const int32_t * context_idx, int n_context,
                            const int32_t * target_idx, int n_target, jepa_output * out) {
    // mask_index 1 is the HF / Meta default (VJEPA2Model.forward, vit_predictor(..., mask_index=1)).
    return jepa_predict_ex(ctx, enc, context_idx, n_context, target_idx, n_target, 1, out);
}
