// LeWM (quentinll/lewm-pusht) world model: AdaLN-zero predictor over projected frame embeddings.
//
// Semantics (docs/gguf-schema.md "lewm" + scripts/jepa_convert/selftest.py::lewm_predictor_forward,
// which is the executable spec; the PyTorch original is stable_worldmodel.wm.lewm.module.Predictor):
//
//   c = pred.action_embed.2( silu( pred.action_embed.0( a_t ) ) )        # [T, 192]
//   x = emb + pred.pos_embed[:T]
//   for each of the 6 blocks:
//       sh_a, sc_a, g_a, sh_m, sc_m, g_m = split6( adaln( silu(c) ) )    # [T, 192] each
//       h = LN(x, no affine, eps 1e-6) * (1 + sc_a) + sh_a
//       h = ln1(h)                                    # affine LN, eps 1e-5
//       x = x + g_a * attn_out( causal_attn( qkv(h) ) )                  # 16 heads x 64, no qkv bias
//       h = LN(x, no affine, eps 1e-6) * (1 + sc_m) + sh_m
//       x = x + g_m * ffn_down( gelu_erf( ffn_up( ln2(h) ) ) )
//   x = pred.norm(x)
//   out_t = pred.proj.2( gelu_erf( pred.proj.0( x_t ) ) )                # predicted *next* projected emb
//
// The attention is causal over frames, so row t of the output only depends on frames 0..t: one call
// with T frames yields the prediction after every prefix, and the last row is the next-frame state.
//
// jepa_build_adaln is that graph over a list of tensors rather than over pred.*, because every
// level of an H-JEPA hierarchy is the same transformer under another name (src/hjepa.cpp).
#include "predictor-internal.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------------------------
ggml_tensor * jepa_build_adaln(jepa_context * ctx, const jepa_adaln_spec & s, ggml_tensor * emb,
                               ggml_tensor * c, ggml_tensor * mask) {
    ggml_context * g = ctx->ctx_g;
    const int64_t T = emb->ne[1];
    const size_t es = sizeof(float);

    // x = emb + pos_embed[:T]
    ggml_tensor * x = emb;
    if (s.pos) {
        ggml_tensor * pv = T == s.pos->ne[1] ? s.pos : ggml_view_2d(g, s.pos, s.pos->ne[0], T, s.pos->nb[1], 0);
        x = ggml_add(g, x, pv);
    }
    // H-JEPA runs the transformer at its own width: states and conditioning are projected in
    if (s.in_w)   x = jepa_build_linear(g, x, s.in_w, s.in_b);
    if (s.cond_w) c = jepa_build_linear(g, c, s.cond_w, s.cond_b);
    const int64_t D = x->ne[0];

    ggml_tensor * cs = jepa_build_act(g, c, JEPA_ACT_SILU);   // silu(c), shared by every block's adaLN

    jepa_attn_opts attn;
    attn.flash = ctx->params.use_flash_attn;
    // T <= n_frames (3) queries hit ggml's per-row flash kernel, which rounds q and the PV
    // accumulator to F16 when K/V are F16 (docs/ggml-notes.md §1) — keep K/V in F32, it is free here.
    // That is a CPU-only statement: on CUDA every fattn kernel converts an F32 K/V to F16 into a
    // scratch buffer first, so asking for F32 there only buys an extra pass
    // (docs/architecture.md "Attention and precision").
    attn.kv_type = ctx->is_gpu ? GGML_TYPE_F16 : GGML_TYPE_F32;
    attn.mask = mask;

    for (const jepa_layer & L : *s.layers) {
        ggml_tensor * mod = jepa_build_linear(g, cs, L.adaln_w, L.adaln_b);   // [6D, T]
        ggml_tensor * six[6];
        for (int k = 0; k < 6; k++) {
            six[k] = ggml_cont(g, ggml_view_2d(g, mod, D, T, mod->nb[1], (size_t) k * D * es));
        }
        ggml_tensor * sh_a = six[0], * sc_a = six[1], * g_a = six[2];
        ggml_tensor * sh_m = six[3], * sc_m = six[4], * g_m = six[5];

        // attention half
        ggml_tensor * h = ggml_norm(g, x, s.adaln_eps);                       // LN without affine
        h = ggml_mul(g, h, ggml_scale_bias(g, sc_a, 1.0f, 1.0f));             // * (1 + scale)
        h = ggml_add(g, h, sh_a);                                             // + shift
        if (L.ln1_w) h = jepa_build_layer_norm(g, h, L.ln1_w, L.ln1_b, s.ln_eps);
        ggml_tensor * q, * k, * v;
        jepa_build_qkv(g, h, L, s.n_head, s.head_dim, &q, &k, &v);
        ggml_tensor * a = jepa_build_attention(g, q, k, v, attn);             // [n_head*head_dim, T]
        a = jepa_build_linear(g, a, L.out_w, L.out_b);                        // [D, T]
        x = ggml_add(g, x, ggml_mul(g, a, g_a));

        // mlp half
        h = ggml_norm(g, x, s.adaln_eps);
        h = ggml_mul(g, h, ggml_scale_bias(g, sc_m, 1.0f, 1.0f));
        h = ggml_add(g, h, sh_m);
        if (L.ln2_w) h = jepa_build_layer_norm(g, h, L.ln2_w, L.ln2_b, s.ln_eps);
        ggml_tensor * f = jepa_build_mlp2(g, h, L.up_w, L.up_b, L.down_w, L.down_b, s.act);
        x = ggml_add(g, x, ggml_mul(g, f, g_m));
    }

    x = jepa_build_layer_norm(g, x, s.norm_w, s.norm_b, s.ln_eps);
    if (s.out_w)   x = jepa_build_linear(g, x, s.out_w, s.out_b);
    if (s.proj0_w) x = jepa_build_mlp2(g, x, s.proj0_w, s.proj0_b, s.proj2_w, s.proj2_b, s.proj_act);
    ggml_set_name(x, "pred_next");
    ggml_set_output(x);
    return x;
}

void jepa_causal_mask_f16(int64_t T, std::vector<ggml_fp16_t> & out) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    out.resize((size_t) T * T);
    for (int64_t i = 0; i < T; i++) {
        for (int64_t j = 0; j < T; j++) out[(size_t) i * T + j] = j <= i ? zero : ninf;
    }
}

ggml_tensor * jepa_build_lewm(jepa_context * ctx, ggml_tensor * emb, ggml_tensor * act, ggml_tensor * mask) {
    const jepa_model * m = ctx->model;
    const jepa_pred_hparams & p = m->hp.pred;

    // action embedding (SiLU inside the 2-layer MLP)
    ggml_tensor * c = jepa_build_mlp2(ctx->ctx_g, act,
                                      m->require("pred.action_embed.0.weight"), m->get("pred.action_embed.0.bias"),
                                      m->require("pred.action_embed.2.weight"), m->get("pred.action_embed.2.bias"),
                                      p.action_act);
    ggml_set_name(c, "act_emb");

    jepa_adaln_spec s;
    s.pos      = m->get("pred.pos_embed");
    s.layers   = &m->pred_layers;
    s.norm_w   = m->require("pred.norm.weight");   s.norm_b  = m->get("pred.norm.bias");
    s.proj0_w  = m->require("pred.proj.0.weight"); s.proj0_b = m->get("pred.proj.0.bias");
    s.proj2_w  = m->require("pred.proj.2.weight"); s.proj2_b = m->get("pred.proj.2.bias");
    s.n_head   = p.n_head;
    s.head_dim = p.head_dim_eff();
    s.ln_eps   = p.ln_eps;
    s.adaln_eps = p.adaln_eps;
    s.act      = p.act;
    s.proj_act = p.proj_act;
    return jepa_build_adaln(ctx, s, emb, c, mask);
}

// ---------------------------------------------------------------------------------------------
// public entry points
// ---------------------------------------------------------------------------------------------
static bool lewm_check(const jepa_context * ctx) {
    const jepa_model * m = ctx->model;
    if (!m->hp.pred.present || m->hp.pred.kind != "lewm") {
        jepa_log("jepa: jepa_lewm_predict: '%s' has no LeWM predictor (jepa.pred.kind = '%s')\n",
                 m->hp.name.c_str(), m->hp.pred.kind.c_str());
        return false;
    }
    return true;
}

extern "C" int jepa_lewm_n_frames(const jepa_model * model) {
    return model && model->hp.pred.present ? model->hp.pred.n_frames : 0;
}
extern "C" int jepa_lewm_action_dim(const jepa_model * model) {
    return model && model->hp.pred.present ? model->hp.pred.action_dim : 0;
}

extern "C" int jepa_lewm_predict(jepa_context * ctx, const float * embs, const float * actions,
                                 int n_frames, jepa_output * out) {
    if (!ctx || !embs || !actions || !out || n_frames <= 0) return -1;
    if (!lewm_check(ctx)) return -1;
    const jepa_model * m = ctx->model;
    const jepa_pred_hparams & p = m->hp.pred;
    const int64_t D = p.embed_dim, A = p.action_dim, T = n_frames;
    if (p.n_frames > 0 && T > p.n_frames) {
        jepa_log("jepa: jepa_lewm_predict: %d frames requested, the predictor has %d position slots\n",
                 n_frames, p.n_frames);
        return -1;
    }
    // The causal mask is the one T^2 term here; the rest is linear. Same ceiling as everywhere else.
    {
        const size_t budget = ctx->max_graph_bytes ? ctx->max_graph_bytes : JEPA_DEFAULT_MAX_GRAPH_BYTES;
        const double need = 2.0 * 2.0 * (double) T * (double) T + (double) T * (D + A) * sizeof(float);
        if (need >= (double) budget) {
            jepa_log("jepa: jepa_lewm_predict: %d frames need about %.1f MiB of graph memory, over the "
                     "%.1f MiB limit ($JEPA_MAX_GRAPH_MIB)\n", n_frames, need / (1024.0 * 1024.0),
                     (double) budget / (1024.0 * 1024.0));
            return -1;
        }
    }

    jepa_graph_begin(ctx, (size_t) p.n_layer * 64 + 128);
    ggml_context * g = ctx->ctx_g;
    ggml_tensor * emb = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_tensor * act = ggml_new_tensor_2d(g, GGML_TYPE_F32, A, T);
    ggml_set_name(emb, "emb");
    ggml_set_name(act, "action");
    ggml_set_input(emb);
    ggml_set_input(act);

    // causal mask over frames: query i (row) may attend key j (column) iff j <= i. F16 works for
    // both ggml_flash_attn_ext (which requires F16) and ggml_soft_max_ext (docs/ggml-notes.md §2).
    ggml_tensor * mask = nullptr;
    std::vector<ggml_fp16_t> mdata;
    if (T > 1 && p.frame_causal) {
        mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, T, T);
        ggml_set_input(mask);
        jepa_causal_mask_f16(T, mdata);
    }

    ggml_tensor * y = jepa_build_lewm(ctx, emb, act, mask);
    ggml_build_forward_expand(ctx->gf, y);
    if (!jepa_graph_alloc(ctx)) return -1;
    ggml_backend_tensor_set(emb, embs, 0, (size_t) D * T * sizeof(float));
    ggml_backend_tensor_set(act, actions, 0, (size_t) A * T * sizeof(float));
    if (mask) ggml_backend_tensor_set(mask, mdata.data(), 0, mdata.size() * sizeof(ggml_fp16_t));
    if (jepa_graph_compute(ctx) != 0) return -1;

    out->n_tokens = T;
    out->dim = y->ne[0];
    out->data = (float *) malloc(ggml_nbytes(y));
    if (!out->data) return -1;
    ggml_backend_tensor_get(y, out->data, 0, ggml_nbytes(y));
    return 0;
}

extern "C" int jepa_lewm_rollout(jepa_context * ctx, const float * embs, int n_seed,
                                 const float * actions, int n_steps, float * out) {
    if (!ctx || !embs || !actions || !out || n_seed <= 0 || n_steps <= 0) return -1;
    if (!lewm_check(ctx)) return -1;
    const jepa_model * m = ctx->model;
    const jepa_pred_hparams & p = m->hp.pred;
    const int64_t D = p.embed_dim, A = p.action_dim;
    const int win = p.n_frames > 0 ? p.n_frames : 1;

    // `n_seed + n_steps` in int overflows before the cast, so the sum is taken in int64 and the
    // buffer it sizes is checked against SIZE_MAX before it is asked for.
    const int64_t n_seq = (int64_t) n_seed + n_steps;
    const size_t budget = ctx->max_graph_bytes ? ctx->max_graph_bytes : JEPA_DEFAULT_MAX_GRAPH_BYTES;
    if ((double) n_seq * (double) D * sizeof(float) >= (double) budget) {
        jepa_log("jepa: jepa_lewm_rollout: %d seed + %d step frames of %lld need %.1f MiB for the "
                 "sequence buffer alone, over the %.1f MiB limit ($JEPA_MAX_GRAPH_MIB)\n",
                 n_seed, n_steps, (long long) D,
                 (double) n_seq * (double) D * sizeof(float) / (1024.0 * 1024.0),
                 (double) budget / (1024.0 * 1024.0));
        return -1;
    }
    // Growing frame sequence: the seeds, then one predicted embedding per step.
    std::vector<float> seq((size_t) n_seq * D);
    memcpy(seq.data(), embs, (size_t) n_seed * D * sizeof(float));

    std::vector<float> win_emb((size_t) win * D), win_act((size_t) win * A);
    double total_ms = 0;
    for (int step = 0; step < n_steps; step++) {
        // int64 throughout: `n_seed + step` is a sum of two caller-supplied ints
        const int64_t have = (int64_t) n_seed + step;   // frames known so far
        const int w = have < win ? (int) have : win;    // window length
        const int64_t first = have - w;                 // global index of the first window frame
        memcpy(win_emb.data(), seq.data() + (size_t) first * D, (size_t) w * D * sizeof(float));
        for (int i = 0; i < w; i++) {
            // Frame j uses the action of the step that produced its successor; the seed frames
            // before the last one have no action of their own and reuse actions[0].
            int64_t ai = first + i - ((int64_t) n_seed - 1);
            if (ai < 0) ai = 0;
            if (ai > (int64_t) n_steps - 1) ai = n_steps - 1;
            memcpy(win_act.data() + (size_t) i * A, actions + (size_t) ai * A, (size_t) A * sizeof(float));
        }
        jepa_output step_out = {};
        if (jepa_lewm_predict(ctx, win_emb.data(), win_act.data(), w, &step_out) != 0) return -1;
        total_ms += ctx->last_compute_ms;
        const float * last = step_out.data + (size_t) (w - 1) * D;
        memcpy(seq.data() + (size_t) have * D, last, (size_t) D * sizeof(float));
        memcpy(out + (size_t) step * D, last, (size_t) D * sizeof(float));
        free(step_out.data);
    }
    ctx->last_compute_ms = total_ms;
    return 0;
}
