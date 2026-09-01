// V-JEPA 2-AC action-conditioned predictor (jepa.pred.kind == "ac").
//
// Reference: facebookresearch/vjepa2, src/models/ac_predictor.py::VisionTransformerPredictorAC
// (forward at :136-190), src/models/utils/modules.py::ACRoPEAttention (:114-262),
// ::ACBlock (:437-503) and ::build_action_block_causal_attention_mask (:12-24).  The executable
// spec is scripts/jepa_convert/selftest.py::ac_predictor_forward; docs/gguf-schema.md "vjepa2_ac".
//
//   x    = pred.embed(context)                                  # [T*HW, Dp]
//   a_t  = pred.action_embed(action_t)                          # [T, Dp]
//   s_t  = pred.state_embed(state_t)                            # [T, Dp]
//   seq  = per frame t: [a_t, s_t, x_{t,0} .. x_{t,HW-1}]       # T*(2+HW) rows   (:146-153)
//   24 pre-LN blocks, full-width attention under an action-block-causal mask, 3-D RoPE on q/k
//   seq  = seq minus the 2 conditioning rows of every frame     # [T*HW, Dp]      (:184-185)
//   out  = pred.proj(pred.norm(seq))                            # [T*HW, enc_dim] (:187-188)
//
// Two details of the reference are easy to get wrong and are what the token-id trick below buys:
//
//  * RoPE positions.  Patch token (t, h, w) is rotated on the (T, grid, grid) grid exactly as the
//    encoder's tokens are (separate_positions, modules.py:154-173).  The action and state tokens
//    are rotated on the DEPTH axis only, with pos = t, and their h/w slots are left alone
//    (modules.py:186-198: `rotate_queries_or_keys(q[..., :d_dim], pos=arange(T))`, `q[..., d_dim:]`
//    untouched).  A grid id of `t*grid*grid` has h = w = 0, whose cos/sin rows are exactly 1 and 0 —
//    the identity rotation — so ONE id list feeds jepa_rope3d_tables_ids for every row:
//        action/state of frame t -> id  t*grid*grid
//        patch (t, h, w)         -> id  t*grid*grid + h*grid + w
//    (modules.py:180-181 also rescales h/w by grid_size/H; the released checkpoint runs at its own
//    grid, where the factor is 1, and jepa_ac_predict refuses anything else.)
//
//  * The mask is block-causal over FRAMES, not over tokens: every row of frame t sees every row of
//    frames 0..t, including the conditioning rows (build_action_block_causal_attention_mask fills
//    whole N_T x N_T blocks).  It is built on the host and handed to jepa_attn_opts::mask, exactly
//    as src/lewm.cpp does.
//
// The K action candidates of a planning step live on the graph's batch axis (ne[3] of q/k/v), so
// attention never mixes them and one call scores K sequences; on the CPU at f32 that is bit-identical
// to K sequential calls (tests/test-predictor.cpp gates it).
#include "predictor-internal.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------
static bool ac_check(const jepa_context * ctx, const char * fn) {
    const jepa_model * m = ctx->model;
    if (!m->hp.pred.present || m->hp.pred.kind != "ac") {
        jepa_log("jepa: %s: '%s' has no action-conditioned predictor (jepa.pred.kind = '%s')\n",
                 fn, m->hp.name.c_str(), m->hp.pred.kind.c_str());
        return false;
    }
    return true;
}

jepa_rope3d_params jepa_ac_rope_params(const jepa_model & m, int n_frames) {
    const jepa_pred_hparams & p = m.hp.pred;
    jepa_rope3d_params rp;
    rp.grid_h = rp.grid_w = p.grid_size;
    rp.grid_t = n_frames;
    rp.head_dim = p.head_dim_eff();
    rp.theta = m.hp.enc.rope_theta;
    rp.interpolate = false;                       // ACRoPEAttention never interpolates
    rp.variant = p.rope_freq_layout == "interleaved" ? JEPA_ROPE3D_VJEPA2_1 : JEPA_ROPE3D_VJEPA2;
    return rp;
}

// ---------------------------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------------------------
// ctx_in  : [enc_dim, T*HW, B] F32 — the encoder latents of the T context frames, per batch item
// act_in  : [action_dim, T, B] F32
// st_in   : [state_dim,  T, B] F32
// mask    : [N, N] F16 additive block-causal mask (nullptr for T == 1) with N = T*(n_cond+HW)
// cos/sin : [head_dim, 1, N] F32
// n_out   : how many trailing frames to project (T for every prefix, 1 for the next frame only)
// returns [out_dim, n_out*HW, B] F32
ggml_tensor * jepa_build_ac(jepa_context * ctx, ggml_tensor * ctx_in, ggml_tensor * act_in, ggml_tensor * st_in,
                            ggml_tensor * mask, ggml_tensor * cos_t, ggml_tensor * sin_t, int n_out) {
    const jepa_model * m = ctx->model;
    const jepa_pred_hparams & p = m->hp.pred;
    ggml_context * g = ctx->ctx_g;
    const int64_t Dp = p.embed_dim;
    const int64_t HW = (int64_t) p.grid_size * p.grid_size;
    const int64_t K = p.n_cond_tokens;
    const int64_t T = act_in->ne[1];
    const int64_t B = ctx_in->ne[2];

    // 1. context projection and the two conditioning streams
    ggml_tensor * x = jepa_build_linear(g, ctx_in, m->require("pred.embed.weight"), m->get("pred.embed.bias"));
    ggml_tensor * a = jepa_build_linear(g, act_in, m->require("pred.action_embed.weight"), m->get("pred.action_embed.bias"));
    ggml_tensor * s = jepa_build_linear(g, st_in,  m->require("pred.state_embed.weight"),  m->get("pred.state_embed.bias"));

    // 2. interleave: [Dp, K, T, B] conditioning rows in front of [Dp, HW, T, B] patch rows.
    //    ggml_concat on axis 1 of the 4-D views IS the reference's
    //    `torch.cat([a, s, x], dim=2).flatten(1, 2)` (ac_predictor.py:148-153) — same row order,
    //    one node per stream instead of one per frame.
    ggml_tensor * cond = ggml_concat(g, ggml_reshape_4d(g, a, Dp, 1, T, B),
                                        ggml_reshape_4d(g, s, Dp, 1, T, B), 1);   // [Dp, 2, T, B]
    ggml_tensor * seq = ggml_concat(g, cond, ggml_reshape_4d(g, x, Dp, HW, T, B), 1);  // [Dp, K+HW, T, B]
    const int64_t N = T * (K + HW);
    seq = ggml_reshape_3d(g, seq, Dp, N, B);

    // 3. blocks
    jepa_block_opts opts;
    opts.ln_eps = p.ln_eps;
    opts.act = p.act;
    opts.n_head = p.n_head;
    opts.head_dim = p.head_dim_eff();
    opts.attn.flash = ctx->params.use_flash_attn;
    opts.attn.kv_type = jepa_context_kv_type(ctx);
    if (opts.attn.flash && !jepa_gpu_flash_ok(ctx, opts.head_dim)) opts.attn.flash = false;
    opts.attn.mask = mask;
    opts.qk_hook = [&](ggml_context * gc, ggml_tensor * t, bool /*is_k*/) {
        return jepa_rope3d_apply(gc, t, cos_t, sin_t);
    };
    for (int i = 0; i < p.n_layer; i++) {
        seq = jepa_build_block(g, seq, m->pred_layers[i], opts);
    }

    // 4. drop the conditioning rows of every frame, keeping the last `n_out` frames.
    //    A [Dp, HW, n_out, B] view of the [Dp, K+HW, T, B] block, offset by K rows and by the
    //    (T - n_out) leading frames — the reference's `x[:, :, cond_tokens:, :]` (ac_predictor.py:185).
    ggml_tensor * s4 = ggml_reshape_4d(g, seq, Dp, K + HW, T, B);
    ggml_tensor * rows = ggml_cont(g, ggml_view_4d(g, s4, Dp, HW, n_out, B,
                                                   s4->nb[1], s4->nb[2], s4->nb[3],
                                                   (size_t) K * s4->nb[1] + (size_t) (T - n_out) * s4->nb[2]));
    rows = ggml_reshape_3d(g, rows, Dp, (int64_t) n_out * HW, B);

    // 5. final norm and projection back to the encoder width
    rows = jepa_build_layer_norm(g, rows, m->require("pred.norm.weight"), m->get("pred.norm.bias"), p.ln_eps);
    ggml_tensor * out = jepa_build_linear(g, rows, m->require("pred.proj.weight"), m->get("pred.proj.bias"));
    ggml_set_name(out, "ac_predictor_out");
    ggml_set_output(out);
    return out;
}

// ---------------------------------------------------------------------------------------------
// introspection
// ---------------------------------------------------------------------------------------------
extern "C" int jepa_ac_tokens_per_frame(const jepa_model * model) {
    if (!model || !model->hp.pred.present || model->hp.pred.kind != "ac") return 0;
    return model->hp.pred.grid_size * model->hp.pred.grid_size;
}
extern "C" int jepa_ac_action_dim(const jepa_model * model) {
    if (!model || !model->hp.pred.present || model->hp.pred.kind != "ac") return 0;
    return model->hp.pred.action_dim;
}
extern "C" int jepa_ac_state_dim(const jepa_model * model) {
    if (!model || !model->hp.pred.present || model->hp.pred.kind != "ac") return 0;
    return model->hp.pred.state_dim;
}
extern "C" int jepa_ac_max_frames(const jepa_model * model) {
    if (!model || !model->hp.pred.present || model->hp.pred.kind != "ac") return 0;
    return model->hp.pred.n_frames;
}
extern "C" bool jepa_ac_normalize_reps(const jepa_model * model) {
    if (!model || !model->hp.pred.present || model->hp.pred.kind != "ac") return false;
    return model->hp.pred.normalize_reps;
}

extern "C" void jepa_ac_normalize(const jepa_model * model, float * rows, int64_t n_rows, int64_t dim) {
    if (!rows || n_rows <= 0 || dim <= 0) return;
    const float eps = model && model->hp.pred.present ? model->hp.pred.norm_reps_eps : 1e-5f;
    for (int64_t r = 0; r < n_rows; r++) {
        float * v = rows + r * dim;
        double mu = 0;
        for (int64_t i = 0; i < dim; i++) mu += v[i];
        mu /= (double) dim;
        double var = 0;
        for (int64_t i = 0; i < dim; i++) { const double d = v[i] - mu; var += d * d; }
        var /= (double) dim;
        const float inv = 1.0f / sqrtf((float) var + eps);
        for (int64_t i = 0; i < dim; i++) v[i] = (float) ((v[i] - (float) mu) * inv);
    }
}

// Meta's pose update, notebooks/utils/mpc_utils.py::compute_new_pose (:166-190):
//   xyz      += action[0:3]
//   rotation  = R(action[3:6]) @ R(pose[3:6]), both scipy Rotation.from_euler("xyz") (LOWERCASE =
//               EXTRINSIC, i.e. R = Rz(c) Ry(b) Rx(a)), then decomposed back to extrinsic xyz
//   gripper   = clip(pose[6] + action[6], 0, 1)
// The decomposition of R = Rz(c) Ry(b) Rx(a) is b = asin(-R20), a = atan2(R21, R22),
// c = atan2(R10, R00); at |R20| = 1 (gimbal lock) scipy pins the third angle to 0, which is the
// branch below. Verified against scipy on random poses in scripts/dump_reference.py (max |d| is
// recorded in the vjepa2-ac-vitg manifest).
extern "C" void jepa_ac_next_state(const jepa_model * model, const float * state, const float * action,
                                   float * out) {
    const int dim = model ? jepa_ac_state_dim(model) : 7;
    if (!state || !action || !out || dim < 7) {
        if (out && state && dim > 0) memcpy(out, state, (size_t) dim * sizeof(float));
        return;
    }
    auto euler_to_mat = [](const float * e, double R[3][3]) {
        const double ca = cos(e[0]), sa = sin(e[0]);
        const double cb = cos(e[1]), sb = sin(e[1]);
        const double cc = cos(e[2]), sc = sin(e[2]);
        R[0][0] = cb * cc; R[0][1] = sa * sb * cc - ca * sc; R[0][2] = ca * sb * cc + sa * sc;
        R[1][0] = cb * sc; R[1][1] = sa * sb * sc + ca * cc; R[1][2] = ca * sb * sc - sa * cc;
        R[2][0] = -sb;     R[2][1] = sa * cb;                R[2][2] = ca * cb;
    };
    double Rp[3][3], Rd[3][3], R[3][3];
    euler_to_mat(state + 3, Rp);
    euler_to_mat(action + 3, Rd);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j] = Rd[i][0] * Rp[0][j] + Rd[i][1] * Rp[1][j] + Rd[i][2] * Rp[2][j];
        }
    }
    double s = -R[2][0];
    if (s > 1.0) s = 1.0;
    if (s < -1.0) s = -1.0;
    const double beta = asin(s);
    double alpha, gamma;
    if (fabs(s) < 1.0 - 1e-9) {
        alpha = atan2(R[2][1], R[2][2]);
        gamma = atan2(R[1][0], R[0][0]);
    } else {   // gimbal lock: only alpha -/+ gamma is determined; scipy reports gamma = 0
        alpha = atan2(-R[1][2], R[1][1]);
        gamma = 0.0;
    }
    for (int i = 0; i < 3; i++) out[i] = state[i] + action[i];
    out[3] = (float) alpha;
    out[4] = (float) beta;
    out[5] = (float) gamma;
    float grip = state[6] + action[6];
    if (grip < 0.0f) grip = 0.0f;
    if (grip > 1.0f) grip = 1.0f;
    out[6] = grip;
    for (int i = 7; i < dim; i++) out[i] = state[i];
}

extern "C" void jepa_ac_energy(const float * pred, const float * goal, int n_batch,
                               int64_t n_rows, int64_t dim, float * out) {
    if (!pred || !goal || !out || n_batch <= 0 || n_rows <= 0 || dim <= 0) return;
    const int64_t per = n_rows * dim;
    for (int b = 0; b < n_batch; b++) {
        const float * a = pred + (int64_t) b * per;
        double acc = 0;
        for (int64_t i = 0; i < per; i++) acc += std::fabs((double) a[i] - (double) goal[i]);
        out[b] = (float) (acc / (double) per);
    }
}

// ---------------------------------------------------------------------------------------------
// one predictor call
// ---------------------------------------------------------------------------------------------
static int ac_predict(jepa_context * ctx, const float * context, int n_frames, int n_batch,
                      const float * actions, const float * states, int n_out, jepa_output * out) {
    const jepa_model * m = ctx->model;
    const jepa_pred_hparams & p = m->hp.pred;
    const int64_t enc_dim = m->hp.enc.embed_dim;
    const int64_t Dp = p.embed_dim;
    const int64_t HW = (int64_t) p.grid_size * p.grid_size;
    const int64_t K = p.n_cond_tokens;
    const int64_t A = p.action_dim, S = p.state_dim;
    const int64_t T = n_frames, B = n_batch;

    if (T <= 0 || B <= 0) {
        jepa_log("jepa: jepa_ac_predict: need n_frames > 0 and n_batch > 0 (got %d / %d)\n", n_frames, n_batch);
        return -1;
    }
    if (p.n_frames > 0 && T > p.n_frames) {
        jepa_log("jepa: jepa_ac_predict: %d context frames, but the predictor's block-causal mask was "
                 "built for %d frame slots (jepa.pred.n_frames)\n", n_frames, p.n_frames);
        return -1;
    }
    const int64_t N = T * (K + HW);
    // Everything below is the caller's arithmetic, so it is checked against the same
    // $JEPA_MAX_GRAPH_MIB ceiling jepa_encode and jepa_predict apply. The mask is the one N^2 term.
    {
        const double per_row = 3.0 * Dp + (double) p.ffn_dim + 8.0 * Dp;
        const double need = (double) N * (double) B * per_row * sizeof(float)
                          + 2.0 * (double) N * (double) N;    // the F16 mask
        const size_t budget = ctx->max_graph_bytes ? ctx->max_graph_bytes : JEPA_DEFAULT_MAX_GRAPH_BYTES;
        if (need >= (double) budget) {
            jepa_log("jepa: jepa_ac_predict: %lld frames x %lld candidates is %lld rows and needs about "
                     "%.1f MiB of graph memory, over the %.1f MiB limit — raise $JEPA_MAX_GRAPH_MIB, "
                     "shorten the context or score fewer candidates per call\n",
                     (long long) T, (long long) B, (long long) N, need / (1024.0 * 1024.0),
                     (double) budget / (1024.0 * 1024.0));
            return -1;
        }
    }
    if (!ctx->params.use_flash_attn || !jepa_gpu_flash_ok(ctx, p.head_dim_eff())) {
        if (!jepa_gpu_naive_attn_fits(ctx, N * B, p.n_head, "the AC predictor")) return -1;
    }

    // RoPE ids: conditioning rows of frame t sit at h = w = 0, whose rotation is the identity, so
    // they get exactly the depth-only rotation the reference applies to them (file header).
    std::vector<int32_t> ids((size_t) N);
    {
        size_t r = 0;
        for (int64_t t = 0; t < T; t++) {
            const int32_t base = (int32_t) (t * HW);
            for (int64_t c = 0; c < K; c++) ids[r++] = base;
            for (int64_t i = 0; i < HW; i++) ids[r++] = base + (int32_t) i;
        }
    }
    const jepa_rope3d_params rp = jepa_ac_rope_params(*m, (int) T);
    std::vector<float> cosv, sinv;
    jepa_rope3d_tables_ids(rp, ids.data(), (int) ids.size(), cosv, sinv);

    // Host-built block-causal mask over frames: row block t sees key blocks 0..t
    // (build_action_block_causal_attention_mask, modules.py:12-24). F16 serves both
    // ggml_flash_attn_ext (which requires it) and ggml_soft_max_ext (docs/ggml-notes.md §2).
    std::vector<ggml_fp16_t> mdata;
    const bool want_mask = T > 1 && p.frame_causal;
    if (want_mask) {
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        mdata.assign((size_t) N * N, ninf);
        const int64_t NT = K + HW;
        for (int64_t t1 = 0; t1 < T; t1++) {
            for (int64_t i = 0; i < NT; i++) {
                ggml_fp16_t * row = mdata.data() + (size_t) (t1 * NT + i) * N;
                for (int64_t j = 0; j < (t1 + 1) * NT; j++) row[j] = zero;
            }
        }
    }

    const size_t nodes = (size_t) p.n_layer * 128 + 512;
    jepa_graph_begin(ctx, nodes);
    ggml_context * g = ctx->ctx_g;
    ggml_tensor * ctx_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, enc_dim, T * HW, B);
    ggml_tensor * act_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, A, T, B);
    ggml_tensor * st_in  = ggml_new_tensor_3d(g, GGML_TYPE_F32, S, T, B);
    ggml_set_name(ctx_in, "ac_context");
    ggml_set_name(act_in, "ac_action");
    ggml_set_name(st_in,  "ac_state");
    ggml_set_input(ctx_in);
    ggml_set_input(act_in);
    ggml_set_input(st_in);
    ggml_tensor * mask = nullptr;
    if (want_mask) {
        mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, N, N);
        ggml_set_input(mask);
    }
    ggml_tensor * cos_t = ggml_new_tensor_3d(g, GGML_TYPE_F32, rp.head_dim, 1, N);
    ggml_tensor * sin_t = ggml_new_tensor_3d(g, GGML_TYPE_F32, rp.head_dim, 1, N);
    ggml_set_input(cos_t);
    ggml_set_input(sin_t);

    ggml_tensor * y = jepa_build_ac(ctx, ctx_in, act_in, st_in, mask, cos_t, sin_t, n_out);
    if (!y) return -1;
    ggml_build_forward_expand(ctx->gf, y);
    if (!jepa_graph_alloc(ctx)) return -1;
    ggml_backend_tensor_set(ctx_in, context, 0, (size_t) enc_dim * T * HW * B * sizeof(float));
    ggml_backend_tensor_set(act_in, actions, 0, (size_t) A * T * B * sizeof(float));
    ggml_backend_tensor_set(st_in,  states,  0, (size_t) S * T * B * sizeof(float));
    if (mask) ggml_backend_tensor_set(mask, mdata.data(), 0, mdata.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(cos_t, cosv.data(), 0, cosv.size() * sizeof(float));
    ggml_backend_tensor_set(sin_t, sinv.data(), 0, sinv.size() * sizeof(float));
    if (jepa_graph_compute(ctx) != 0) return -1;

    out->n_tokens = (int64_t) n_out * HW * B;
    out->dim = y->ne[0];
    out->data = (float *) malloc(ggml_nbytes(y));
    if (!out->data) return -1;
    ggml_backend_tensor_get(y, out->data, 0, ggml_nbytes(y));
    return 0;
}

static int ac_entry(jepa_context * ctx, const float * context, int n_frames, int n_batch,
                    const float * actions, const float * states, bool all_frames, jepa_output * out,
                    const char * fn) {
    if (!ctx || !context || !actions || !states || !out) return -1;
    if (!ac_check(ctx, fn)) return -1;
    return ac_predict(ctx, context, n_frames, n_batch, actions, states,
                      all_frames ? n_frames : 1, out);
}

extern "C" int jepa_ac_predict(jepa_context * ctx, const float * context, int n_frames, int n_batch,
                               const float * actions, const float * states, jepa_output * out) {
    return ac_entry(ctx, context, n_frames, n_batch, actions, states, false, out, "jepa_ac_predict");
}

extern "C" int jepa_ac_predict_all(jepa_context * ctx, const float * context, int n_frames, int n_batch,
                                   const float * actions, const float * states, jepa_output * out) {
    return ac_entry(ctx, context, n_frames, n_batch, actions, states, true, out, "jepa_ac_predict_all");
}

// ---------------------------------------------------------------------------------------------
// rollout
// ---------------------------------------------------------------------------------------------
extern "C" int jepa_ac_rollout(jepa_context * ctx, const float * context, int n_seed,
                               const float * seed_states, const float * actions, const float * states,
                               int n_cand, int horizon, float * out) {
    if (!ctx || !context || !seed_states || !actions || !out) return -1;
    if (!ac_check(ctx, "jepa_ac_rollout")) return -1;
    const jepa_model * m = ctx->model;
    const jepa_pred_hparams & p = m->hp.pred;
    const int64_t enc_dim = m->hp.enc.embed_dim;
    const int64_t HW = (int64_t) p.grid_size * p.grid_size;
    const int64_t A = p.action_dim, S = p.state_dim;
    if (n_seed <= 0 || n_cand <= 0 || horizon <= 0) {
        jepa_log("jepa: jepa_ac_rollout: need n_seed > 0, n_cand > 0 and horizon > 0 (got %d / %d / %d)\n",
                 n_seed, n_cand, horizon);
        return -1;
    }
    const int64_t n_seq = (int64_t) n_seed + horizon - 1;   // frames the last step conditions on
    if (p.n_frames > 0 && n_seq > p.n_frames) {
        jepa_log("jepa: jepa_ac_rollout: %d seed frames + %d steps reaches a %lld-frame context, over the "
                 "predictor's %d frame slots (jepa.pred.n_frames)\n", n_seed, horizon,
                 (long long) n_seq, p.n_frames);
        return -1;
    }
    // The growing context is [n_cand, n_seq*HW, enc_dim]; refuse it up front rather than in malloc.
    const size_t budget = ctx->max_graph_bytes ? ctx->max_graph_bytes : JEPA_DEFAULT_MAX_GRAPH_BYTES;
    const double seq_bytes = (double) n_cand * n_seq * HW * enc_dim * sizeof(float);
    if (seq_bytes >= (double) budget) {
        jepa_log("jepa: jepa_ac_rollout: %d candidates x %lld frames x %lld tokens need %.1f MiB for the "
                 "context buffer alone, over the %.1f MiB limit ($JEPA_MAX_GRAPH_MIB)\n",
                 n_cand, (long long) n_seq, (long long) HW, seq_bytes / (1024.0 * 1024.0),
                 (double) budget / (1024.0 * 1024.0));
        return -1;
    }

    // Per-candidate context, seeded with the shared encoder latents; grows by one frame per step.
    const int64_t per_frame = HW * enc_dim;
    std::vector<float> seq((size_t) n_cand * n_seq * per_frame);
    std::vector<float> st((size_t) n_cand * n_seq * S);
    std::vector<float> act((size_t) n_cand * n_seq * A);
    for (int c = 0; c < n_cand; c++) {
        memcpy(seq.data() + (size_t) c * n_seq * per_frame, context, (size_t) n_seed * per_frame * sizeof(float));
        memcpy(st.data() + (size_t) c * n_seq * S, seed_states, (size_t) n_seed * S * sizeof(float));
    }

    double total_ms = 0;
    std::vector<float> ctx_buf, act_buf, st_buf;
    for (int step = 0; step < horizon; step++) {
        const int64_t T = (int64_t) n_seed + step;   // context frames for this step
        // Frame j of a candidate's context carries the action/state that drives the step it starts:
        // the seed frames before the last reuse row 0, exactly as the reference repeats the context
        // pose (notebooks/utils/world_model_wrapper.py::step_predictor passes the whole history).
        for (int c = 0; c < n_cand; c++) {
            float * ac = act.data() + (size_t) c * n_seq * A;
            float * sc = st.data() + (size_t) c * n_seq * S;
            for (int64_t j = 0; j < T; j++) {
                int64_t k = j - ((int64_t) n_seed - 1);
                if (k < 0) k = 0;
                if (k > (int64_t) step) k = step;
                memcpy(ac + j * A, actions + ((size_t) c * horizon + k) * A, (size_t) A * sizeof(float));
                if (j >= n_seed) {
                    // Pose of the frame this step produced: the caller's, or Meta's own update
                    // (compute_new_pose of the previous pose and the action that drove that step).
                    if (states) {
                        memcpy(sc + j * S, states + ((size_t) c * horizon + (j - n_seed)) * S,
                               (size_t) S * sizeof(float));
                    } else {
                        jepa_ac_next_state(m, sc + (j - 1) * S, ac + (j - 1) * A, sc + j * S);
                    }
                }
            }
        }
        // Pack the [n_cand, T*HW, enc_dim] context / [n_cand, T, A|S] conditioning for this step.
        ctx_buf.resize((size_t) n_cand * T * per_frame);
        act_buf.resize((size_t) n_cand * T * A);
        st_buf.resize((size_t) n_cand * T * S);
        for (int c = 0; c < n_cand; c++) {
            memcpy(ctx_buf.data() + (size_t) c * T * per_frame,
                   seq.data() + (size_t) c * n_seq * per_frame, (size_t) T * per_frame * sizeof(float));
            memcpy(act_buf.data() + (size_t) c * T * A, act.data() + (size_t) c * n_seq * A,
                   (size_t) T * A * sizeof(float));
            memcpy(st_buf.data() + (size_t) c * T * S, st.data() + (size_t) c * n_seq * S,
                   (size_t) T * S * sizeof(float));
        }
        jepa_output step_out = {};
        if (ac_predict(ctx, ctx_buf.data(), (int) T, n_cand, act_buf.data(), st_buf.data(), 1, &step_out) != 0) {
            return -1;
        }
        total_ms += ctx->last_compute_ms;
        // step_out is [n_cand * HW, enc_dim]: the next frame of every candidate.
        if (p.normalize_reps) jepa_ac_normalize(m, step_out.data, (int64_t) n_cand * HW, enc_dim);
        for (int c = 0; c < n_cand; c++) {
            const float * pred = step_out.data + (size_t) c * per_frame;
            memcpy(out + ((size_t) c * horizon + step) * per_frame, pred, (size_t) per_frame * sizeof(float));
            if (T < n_seq) {
                memcpy(seq.data() + (size_t) c * n_seq * per_frame + (size_t) T * per_frame,
                       pred, (size_t) per_frame * sizeof(float));
            }
        }
        free(step_out.data);
    }
    ctx->last_compute_ms = total_ms;
    return 0;
}
