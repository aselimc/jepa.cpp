// H-JEPA (arXiv 2610.06805, github.com/kevinghst/H-JEPA): a stack of action-conditioned JEPA world
// models, each predicting further ahead in its own latent space. jepa.family = "hjepa".
//
// Semantics (docs/gguf-schema.md "hjepa" + scripts/jepa_convert/hjepa_numpy_ref.py, which is the
// executable spec; the PyTorch original is h_jepa/models/{jepa,hjepa}.py). Levels are 1-based and
// level k's tensors live under "lvl.{k}."; the ViT level 1 reads is the ordinary enc.*.
//
//   states    s1_t = embed( ViT(pixels_t)[CLS] [, proprio_t] )       jepa_hjepa_embed
//             sk_T = enc_k( s{k-1}_{T * stride_k} )                  jepa_hjepa_lift
//   actions   e1_t = act_embed_1( a_t )                              jepa_hjepa_action_embed
//             ak_T = act_enc_k( e{k-1}_{T*stride .. T*stride + stride - 1} )   jepa_hjepa_action_encode
//             ek_T = act_embed_k( ak_T )
//   dynamics  sk_{T+1} ~ pred_k( sk_{..T}, ek_{..T} )                jepa_hjepa_predict / _rollout
//
// The predictor of every level is the adaLN-zero causal transformer of src/lewm.cpp
// (jepa_build_adaln) at that level's own width; "adaln_bare" (the DROID checkpoints) drops the
// inner LayerNorms and the projector MLP and embeds the action inside the predictor.
//
// The released hierarchies all have window_size 1: a level's state is one state of the level
// below, re-encoded. Planning (the paper's top-down gradient solver) is src/hjepa-plan.cpp.
#include "predictor-internal.h"

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string lvl_prefix(int level) { return "lvl." + std::to_string(level) + "."; }

// ---------------------------------------------------------------------------------------------
// load-time binding and validation (called by jepa_model_load_ex)
// ---------------------------------------------------------------------------------------------
namespace {

// Every graph below reads its weights through jepa_model::get and multiplies or adds them without
// further checks, so a file whose tensors do not have the shapes its metadata promises has to be
// refused here: inside the builders it is a GGML_ASSERT, i.e. an abort on an untrusted file.
struct binder {
    const jepa_model & m;
    bool ok = true;

    void fail(const char * fmt, const std::string & name, int64_t a = 0, int64_t b = 0) {
        jepa_log(fmt, name.c_str(), a, b);
        ok = false;
    }
    // f32 vector of exactly n values. Returns whether it is there.
    bool vec(const std::string & name, int64_t n, bool required) {
        ggml_tensor * t = m.get(name);
        if (!t) {
            if (required) fail("jepa: hjepa: required tensor %s is missing\n", name);
            return false;
        }
        if (t->type != GGML_TYPE_F32) fail("jepa: hjepa: tensor %s has to be f32\n", name);
        if (ggml_nelements(t) != n) fail("jepa: hjepa: tensor %s holds %" PRId64 " values, expected %" PRId64 "\n", name, ggml_nelements(t), n);
        return true;
    }
    // Linear `name`.weight [n_in, n_out] (+ optional f32 bias of n_out); 0 = any. Returns the
    // output width, or 0 when the tensor is absent.
    int64_t lin(const std::string & name, int64_t n_in, int64_t n_out, bool required) {
        ggml_tensor * w = m.get(name + ".weight");
        if (!w) {
            if (required) fail("jepa: hjepa: required tensor %s.weight is missing\n", name);
            return 0;
        }
        if (ggml_n_dims(w) > 2 || (n_in > 0 && w->ne[0] != n_in) || (n_out > 0 && w->ne[1] != n_out)) {
            jepa_log("jepa: hjepa: tensor %s.weight has shape [%" PRId64 ", %" PRId64 "], expected [%" PRId64 ", %" PRId64 "]\n",
                     name.c_str(), w->ne[0], w->ne[1], n_in > 0 ? n_in : w->ne[0], n_out > 0 ? n_out : w->ne[1]);
            ok = false;
        }
        vec(name + ".bias", w->ne[1], false);
        return w->ne[1];
    }
    // f32 table [dim, >= min_rows]
    void table(const std::string & name, int64_t dim, int64_t min_rows, bool required) {
        ggml_tensor * t = m.get(name);
        if (!t) {
            if (required) fail("jepa: hjepa: required tensor %s is missing\n", name);
            return;
        }
        if (t->type != GGML_TYPE_F32) fail("jepa: hjepa: tensor %s has to be f32\n", name);
        if (ggml_n_dims(t) > 2 || t->ne[0] != dim || t->ne[1] < min_rows) {
            fail("jepa: hjepa: tensor %s needs width %" PRId64 " and at least %" PRId64 " rows\n", name, dim, min_rows);
        }
    }
    void want(bool cond, const char * what, int level) {
        if (cond) return;
        jepa_log("jepa: hjepa: level %d: %s\n", level, what);
        ok = false;
    }
};

}  // namespace

bool jepa_hjepa_bind(jepa_model & m) {
    const jepa_hjepa_hparams & hj = m.hp.hjepa;
    const int64_t Dv = m.hp.enc.embed_dim;
    binder b{m};
    m.hj_pred_layers.assign(hj.levels.size(), {});
    m.hj_act_enc_layers.assign(hj.levels.size(), {});
    if (!m.hp.enc.cls_token) {
        jepa_log("jepa: hjepa: the level-1 state is read off the CLS token, but jepa.enc.cls_token is false\n");
        return false;
    }

    for (size_t li = 0; li < hj.levels.size() && b.ok; li++) {
        const int k = (int) li + 1;
        const jepa_hjepa_level & lv = hj.levels[li];
        const jepa_hjepa_level * below = li ? &hj.levels[li - 1] : nullptr;
        const std::string p = lvl_prefix(k);

        // ---- state encoder
        if (k == 1) {
            b.want(lv.stride == 1, "level 1 must have stride 1", k);
            b.want(lv.act_enc_kind == "none", "level 1 has no action encoder (jepa.hjepa.act_enc_kind must be 'none')", k);
            if (hj.proj_kind == "mlp") {
                const int64_t h = b.lin(p + "enc.proj.0", Dv, 0, true);
                b.lin(p + "enc.proj.2", h, hj.pixel_dim, true);
            } else if (hj.proj_kind == "layernorm") {
                b.vec(p + "enc.proj_ln.weight", Dv, true);
                b.vec(p + "enc.proj_ln.bias", Dv, false);
                b.want(hj.pixel_dim == Dv, "a layernorm projector cannot change the width (jepa.hjepa.pixel_dim)", k);
            } else if (hj.proj_kind == "identity") {
                b.want(hj.pixel_dim == Dv, "an identity projector cannot change the width (jepa.hjepa.pixel_dim)", k);
            } else {
                b.want(false, "jepa.hjepa.proj_kind must be mlp | layernorm | identity", k);
            }
            if (lv.enc_kind == "vit_proprio") {
                b.want(hj.proprio_dim > 0, "enc_kind 'vit_proprio' needs jepa.hjepa.proprio_dim > 0", k);
                const int64_t h = b.lin(p + "enc.proprio.0", hj.proprio_dim, 0, true);
                b.vec(p + "enc.proprio.ln.weight", h, true);
                b.vec(p + "enc.proprio.ln.bias", h, false);
                const int64_t q = b.lin(p + "enc.proprio.2", h, 0, true);
                b.want(hj.pixel_dim + q == lv.embed_dim, "pixel and proprio widths do not add up to jepa.hjepa.embed_dim", k);
            } else if (lv.enc_kind == "vit") {
                b.want(hj.pixel_dim == lv.embed_dim, "jepa.hjepa.pixel_dim differs from the level's embed_dim", k);
            } else {
                b.want(false, "jepa.hjepa.enc_kind must be vit | vit_proprio at level 1", k);
            }
        } else {
            if (lv.enc_kind == "latent_mlp") {
                b.vec(p + "enc.in_norm.weight", below->embed_dim, true);
                b.vec(p + "enc.in_norm.bias", below->embed_dim, false);
                const int64_t h = b.lin(p + "enc.fc1", below->embed_dim, 0, true);
                b.lin(p + "enc.fc2", h, lv.embed_dim, true);
                if (b.vec(p + "enc.out_norm.weight", lv.embed_dim, false)) b.vec(p + "enc.out_norm.bias", lv.embed_dim, false);
            } else if (lv.enc_kind == "identity") {
                b.want(lv.embed_dim == below->embed_dim, "an identity encoder cannot change the width", k);
            } else {
                b.want(false, "jepa.hjepa.enc_kind must be latent_mlp | identity above level 1", k);
            }

            // ---- action encoder: a chunk of `stride` action embeddings of the level below -> one latent action
            const int64_t E = below->act_embed_dim;
            if (lv.act_enc_kind == "seq") {
                const int64_t d = lv.act_enc_dim;
                b.want(d > 0 && lv.act_enc_n_head > 0 && d % lv.act_enc_n_head == 0 && lv.act_enc_n_layer > 0,
                       "a 'seq' action encoder needs act_enc_dim divisible by act_enc_n_head and at least one layer", k);
                b.want(lv.stride <= lv.act_enc_max_chunk, "stride exceeds jepa.hjepa.act_enc_max_chunk", k);
                if (!b.ok) break;
                b.lin(p + "act_enc.step", E, d, true);
                b.table(p + "act_enc.pos", d, (int64_t) lv.act_enc_max_chunk + 1, true);
                for (int i = 0; i < lv.act_enc_n_layer && b.ok; i++) {
                    const std::string q = p + "act_enc.blk." + std::to_string(i) + ".";
                    m.hj_act_enc_layers[li].push_back(jepa_layer_from_model(m, q));
                    b.vec(q + "ln1.weight", d, true);  b.vec(q + "ln1.bias", d, false);
                    b.vec(q + "ln2.weight", d, true);  b.vec(q + "ln2.bias", d, false);
                    b.lin(q + "attn_qkv", d, 3 * d, true);
                    b.lin(q + "attn_out", d, d, true);
                    const int64_t f = b.lin(q + "ffn_up", d, 0, true);
                    b.lin(q + "ffn_down", f, d, true);
                }
                const int64_t h = b.lin(p + "act_enc.head.0", d, 0, true);
                b.vec(p + "act_enc.head.ln.weight", h, true);
                b.vec(p + "act_enc.head.ln.bias", h, false);
                b.lin(p + "act_enc.head.2", h, lv.action_dim, true);
                b.vec(p + "act_enc.out_norm.weight", lv.action_dim, true);
                b.vec(p + "act_enc.out_norm.bias", lv.action_dim, false);
            } else if (lv.act_enc_kind == "mlp") {
                const int64_t h = b.lin(p + "act_enc.fc1", (int64_t) lv.stride * E, 0, true);
                b.lin(p + "act_enc.fc2", h, lv.action_dim, true);
                if (b.vec(p + "act_enc.out_norm.weight", lv.action_dim, false)) b.vec(p + "act_enc.out_norm.bias", lv.action_dim, false);
            } else {
                b.want(false, "jepa.hjepa.act_enc_kind must be seq | mlp above level 1", k);
            }
        }

        // ---- action embedding (absent = identity)
        if (const int64_t h = b.lin(p + "act_embed.0", lv.action_dim, 0, false)) {
            b.lin(p + "act_embed.2", h, lv.act_embed_dim, true);
        } else {
            b.want(lv.act_embed_dim == lv.action_dim, "no act_embed tensors, yet act_embed_dim differs from action_dim", k);
        }
        b.table(p + "act_queue", lv.action_dim, 1, false);

        // ---- predictor
        const bool bare = lv.pred_kind == "adaln_bare";
        b.want(bare || lv.pred_kind == "adaln", "jepa.hjepa.pred_kind must be adaln | adaln_bare", k);
        const int64_t D = lv.pred_dim, inner = (int64_t) lv.pred_n_head * lv.pred_head_dim;
        b.table(p + "pred.pos_embed", lv.embed_dim, lv.pred_n_frames, true);
        int64_t cond = lv.act_embed_dim;
        if (bare) {   // the predictor's own action embedder
            const int64_t h = b.lin(p + "pred.act_embed.0", lv.act_embed_dim, 0, true);
            cond = b.lin(p + "pred.act_embed.2", h, 0, true);
        }
        if (!b.lin(p + "pred.in_proj", lv.embed_dim, D, false)) {
            b.want(lv.embed_dim == D, "pred.in_proj is missing but the predictor width differs from the state width", k);
        }
        if (!b.lin(p + "pred.cond_proj", cond, D, false)) {
            b.want(cond == D, "pred.cond_proj is missing but the predictor width differs from the conditioning width", k);
        }
        for (int i = 0; i < lv.pred_n_layer && b.ok; i++) {
            const std::string q = p + "pred.blk." + std::to_string(i) + ".";
            m.hj_pred_layers[li].push_back(jepa_layer_from_model(m, q));
            b.lin(q + "adaln", D, 6 * D, true);
            b.vec(q + "ln1.weight", D, !bare);  b.vec(q + "ln1.bias", D, false);
            b.vec(q + "ln2.weight", D, !bare);  b.vec(q + "ln2.bias", D, false);
            b.lin(q + "attn_qkv", D, 3 * inner, true);
            b.lin(q + "attn_out", inner, D, true);
            b.lin(q + "ffn_up", D, lv.pred_ffn_dim, true);
            b.lin(q + "ffn_down", lv.pred_ffn_dim, D, true);
        }
        b.vec(p + "pred.norm.weight", D, true);
        b.vec(p + "pred.norm.bias", D, false);
        int64_t out = D;
        if (const int64_t o = b.lin(p + "pred.out_proj", D, 0, false)) out = o;
        if (const int64_t h = b.lin(p + "pred.proj.0", out, 0, false)) out = b.lin(p + "pred.proj.2", h, 0, true);
        b.want(out == lv.embed_dim, "the predictor's output width differs from the level's state width", k);
    }

    // training normaliser (optional): mean / std of the raw action and proprio columns
    for (const char * col : { "action", "proprio" }) {
        const std::string n = std::string("norm.") + col;
        ggml_tensor * mean = m.get(n + ".mean");
        if (!mean) continue;
        b.vec(n + ".mean", ggml_nelements(mean), true);
        b.vec(n + ".std", ggml_nelements(mean), true);
    }
    return b.ok;
}

// ---------------------------------------------------------------------------------------------
// shared plumbing
// ---------------------------------------------------------------------------------------------
static const jepa_hjepa_level * level_of(const jepa_model * m, int level) {
    if (!m || !m->hp.hjepa.present || level < 1 || level > (int) m->hp.hjepa.levels.size()) return nullptr;
    return &m->hp.hjepa.levels[(size_t) level - 1];
}

// The level a public entry point was asked for, or nullptr after saying why not.
static const jepa_hjepa_level * check_level(const jepa_context * ctx, int level, const char * fn) {
    const jepa_model * m = ctx->model;
    if (!m->hp.hjepa.present) {
        jepa_log("jepa: %s: '%s' is not an H-JEPA model (jepa.family = '%s')\n", fn, m->hp.name.c_str(), m->hp.family_str.c_str());
        return nullptr;
    }
    const jepa_hjepa_level * lv = level_of(m, level);
    if (!lv) jepa_log("jepa: %s: level %d is out of range, the model has levels 1..%zu\n", fn, level, m->hp.hjepa.levels.size());
    return lv;
}

// Refuse a call whose buffers alone pass the graph budget ($JEPA_MAX_GRAPH_MIB): every graph here
// is linear in its row count, so the row buffers are the bound.
static bool rows_fit(const jepa_context * ctx, const char * fn, int64_t n_rows, int64_t width) {
    const size_t budget = ctx->max_graph_bytes ? ctx->max_graph_bytes : JEPA_DEFAULT_MAX_GRAPH_BYTES;
    const double need = (double) n_rows * (double) width * sizeof(float);
    if (need < (double) budget) return true;
    jepa_log("jepa: %s: %lld rows of %lld floats need %.1f MiB, over the %.1f MiB limit ($JEPA_MAX_GRAPH_MIB)\n",
             fn, (long long) n_rows, (long long) width, need / (1024.0 * 1024.0), (double) budget / (1024.0 * 1024.0));
    return false;
}

static int emit(jepa_output * out, const float * data, int64_t n_rows, int64_t dim) {
    out->n_tokens = n_rows;
    out->dim = dim;
    out->data = (float *) malloc((size_t) n_rows * dim * sizeof(float));
    if (!out->data) return -1;
    memcpy(out->data, data, (size_t) n_rows * dim * sizeof(float));
    return 0;
}

// Allocate and run the graph ending in `y` after uploading `n_in` inputs; copy y out as rows of y->ne[0].
struct graph_input { ggml_tensor * t; const float * data; };
static int run_graph(jepa_context * ctx, ggml_tensor * y, const graph_input * in, int n_in, std::vector<float> & res) {
    ggml_set_output(y);
    ggml_build_forward_expand(ctx->gf, y);
    if (!jepa_graph_alloc(ctx)) return -1;
    for (int i = 0; i < n_in; i++) ggml_backend_tensor_set(in[i].t, in[i].data, 0, ggml_nbytes(in[i].t));
    if (jepa_graph_compute(ctx) != 0) return -1;
    GGML_ASSERT(y->type == GGML_TYPE_F32);
    res.resize((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, res.data(), 0, ggml_nbytes(y));
    return 0;
}

static ggml_tensor * new_input(jepa_context * ctx, int64_t d0, int64_t d1, int64_t d2 = 1) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx->ctx_g, GGML_TYPE_F32, d0, d1, d2);
    ggml_set_input(t);
    return t;
}

// act_embed_k as a graph node (identity when the file has no lvl.{k}.act_embed.*).
static ggml_tensor * build_act_embed(jepa_context * ctx, int level, ggml_tensor * act) {
    const jepa_model * m = ctx->model;
    const std::string p = lvl_prefix(level) + "act_embed.";
    ggml_tensor * w0 = m->get(p + "0.weight");
    if (!w0) return act;
    return jepa_build_mlp2(ctx->ctx_g, act, w0, m->get(p + "0.bias"), m->get(p + "2.weight"), m->get(p + "2.bias"), JEPA_ACT_SILU);
}

// ---------------------------------------------------------------------------------------------
// metadata
// ---------------------------------------------------------------------------------------------
extern "C" int jepa_hjepa_n_levels(const jepa_model * model) {
    return model && model->hp.hjepa.present ? (int) model->hp.hjepa.levels.size() : 0;
}
extern "C" int jepa_hjepa_embed_dim(const jepa_model * model, int level) {
    const jepa_hjepa_level * lv = level_of(model, level);
    return lv ? lv->embed_dim : 0;
}
extern "C" int jepa_hjepa_action_dim(const jepa_model * model, int level) {
    const jepa_hjepa_level * lv = level_of(model, level);
    return lv ? lv->action_dim : 0;
}
extern "C" int jepa_hjepa_action_embed_dim(const jepa_model * model, int level) {
    const jepa_hjepa_level * lv = level_of(model, level);
    return lv ? lv->act_embed_dim : 0;
}
extern "C" int jepa_hjepa_stride(const jepa_model * model, int level) {
    const jepa_hjepa_level * lv = level_of(model, level);
    return lv ? lv->stride : 0;
}
extern "C" int jepa_hjepa_n_frames(const jepa_model * model, int level) {
    const jepa_hjepa_level * lv = level_of(model, level);
    return lv ? lv->pred_n_frames : 0;
}
extern "C" int jepa_hjepa_proprio_dim(const jepa_model * model) {
    return model && model->hp.hjepa.present ? model->hp.hjepa.proprio_dim : 0;
}
extern "C" int jepa_hjepa_raw_action_dim(const jepa_model * model) {
    return model && model->hp.hjepa.present ? model->hp.hjepa.raw_action_dim : 0;
}
extern "C" int jepa_hjepa_frameskip(const jepa_model * model) {
    return model && model->hp.hjepa.present ? model->hp.hjepa.frameskip : 0;
}

extern "C" int jepa_hjepa_norm_stats(const jepa_model * model, const char * column, float * mean, float * std, int capacity) {
    if (!model || !column || !model->hp.hjepa.present) return 0;
    if (strcmp(column, "action") != 0 && strcmp(column, "proprio") != 0) return 0;
    ggml_tensor * tm = model->get(std::string("norm.") + column + ".mean");
    ggml_tensor * ts = model->get(std::string("norm.") + column + ".std");
    if (!tm || !ts) return 0;
    const int64_t n = ggml_nelements(tm);
    if (!mean || !std || capacity < n) return (int) n;   // the size to come back with
    ggml_backend_tensor_get(tm, mean, 0, (size_t) n * sizeof(float));
    ggml_backend_tensor_get(ts, std, 0, (size_t) n * sizeof(float));
    return (int) n;
}

extern "C" int jepa_hjepa_action_prior(const jepa_model * model, int level, float * out, int capacity) {
    if (!level_of(model, level)) return 0;
    ggml_tensor * q = model->get(lvl_prefix(level) + "act_queue");
    if (!q) return 0;
    const int64_t n = q->ne[1];
    if (!out || capacity < n) return (int) n;
    ggml_backend_tensor_get(q, out, 0, ggml_nbytes(q));
    return (int) n;
}

// ---------------------------------------------------------------------------------------------
// states
// ---------------------------------------------------------------------------------------------
extern "C" int jepa_hjepa_embed(jepa_context * ctx, const float * cls_rows, const float * proprio, int n_rows, jepa_output * out) {
    if (!ctx || !cls_rows || !out || n_rows <= 0) return -1;
    const jepa_hjepa_level * lv = check_level(ctx, 1, "jepa_hjepa_embed");
    if (!lv) return -1;
    const jepa_model * m = ctx->model;
    const jepa_hjepa_hparams & hj = m->hp.hjepa;
    const bool fusion = lv->enc_kind == "vit_proprio";
    if (fusion && !proprio) {
        jepa_log("jepa: jepa_hjepa_embed: this model's state includes a %d-d proprio stream; pass it\n", hj.proprio_dim);
        return -1;
    }
    const int64_t Dv = m->hp.enc.embed_dim;
    if (!rows_fit(ctx, "jepa_hjepa_embed", n_rows, Dv + lv->embed_dim + hj.proprio_dim)) return -1;

    jepa_graph_begin(ctx, 64);
    ggml_context * g = ctx->ctx_g;
    const std::string p = "lvl.1.enc.";
    ggml_tensor * cls = new_input(ctx, Dv, n_rows);
    ggml_tensor * x = cls;
    if (hj.proj_kind == "mlp") {
        x = jepa_build_mlp2(g, x, m->get(p + "proj.0.weight"), m->get(p + "proj.0.bias"),
                            m->get(p + "proj.2.weight"), m->get(p + "proj.2.bias"), JEPA_ACT_GELU_ERF);
    } else if (hj.proj_kind == "layernorm") {
        x = jepa_build_layer_norm(g, x, m->get(p + "proj_ln.weight"), m->get(p + "proj_ln.bias"), hj.ln_eps);
    }
    graph_input in[2] = { { cls, cls_rows }, { nullptr, nullptr } };
    int n_in = 1;
    if (fusion) {
        ggml_tensor * pr = new_input(ctx, hj.proprio_dim, n_rows);
        in[n_in++] = { pr, proprio };
        ggml_tensor * h = jepa_build_linear(g, pr, m->get(p + "proprio.0.weight"), m->get(p + "proprio.0.bias"));
        h = jepa_build_layer_norm(g, h, m->get(p + "proprio.ln.weight"), m->get(p + "proprio.ln.bias"), hj.ln_eps);
        h = jepa_build_act(g, h, JEPA_ACT_GELU_ERF);
        h = jepa_build_linear(g, h, m->get(p + "proprio.2.weight"), m->get(p + "proprio.2.bias"));
        x = ggml_concat(g, x, h, 0);   // [pixel | proprio]
    }
    if (x == cls) return emit(out, cls_rows, n_rows, Dv);   // identity projector, no proprio: nothing to compute
    std::vector<float> res;
    if (run_graph(ctx, x, in, n_in, res) != 0) return -1;
    return emit(out, res.data(), n_rows, x->ne[0]);
}

extern "C" int jepa_hjepa_lift(jepa_context * ctx, int level, const float * below, int n_rows, jepa_output * out) {
    if (!ctx || !below || !out || n_rows <= 0) return -1;
    const jepa_hjepa_level * lv = check_level(ctx, level, "jepa_hjepa_lift");
    if (!lv) return -1;
    if (level < 2) {
        jepa_log("jepa: jepa_hjepa_lift: level 1 states come from jepa_hjepa_embed; lift takes level >= 2\n");
        return -1;
    }
    const jepa_model * m = ctx->model;
    const jepa_hjepa_hparams & hj = m->hp.hjepa;
    const int64_t Db = hj.levels[(size_t) level - 2].embed_dim;
    const int64_t n_out = ((int64_t) n_rows - 1) / lv->stride + 1;
    if (!rows_fit(ctx, "jepa_hjepa_lift", n_out, Db + lv->embed_dim)) return -1;

    // window 1: level k's state T is the encoding of the level below's state T * stride
    std::vector<float> sub((size_t) n_out * Db);
    for (int64_t t = 0; t < n_out; t++) {
        memcpy(sub.data() + (size_t) t * Db, below + (size_t) t * lv->stride * Db, (size_t) Db * sizeof(float));
    }
    if (lv->enc_kind == "identity") return emit(out, sub.data(), n_out, Db);

    jepa_graph_begin(ctx, 64);
    ggml_context * g = ctx->ctx_g;
    const std::string p = lvl_prefix(level) + "enc.";
    ggml_tensor * inp = new_input(ctx, Db, n_out);
    ggml_tensor * x = jepa_build_layer_norm(g, inp, m->get(p + "in_norm.weight"), m->get(p + "in_norm.bias"), hj.ln_eps);
    x = jepa_build_mlp2(g, x, m->get(p + "fc1.weight"), m->get(p + "fc1.bias"),
                        m->get(p + "fc2.weight"), m->get(p + "fc2.bias"), JEPA_ACT_GELU_ERF);
    if (ggml_tensor * w = m->get(p + "out_norm.weight")) x = jepa_build_layer_norm(g, x, w, m->get(p + "out_norm.bias"), hj.ln_eps);
    const graph_input in = { inp, sub.data() };
    std::vector<float> res;
    if (run_graph(ctx, x, &in, 1, res) != 0) return -1;
    return emit(out, res.data(), n_out, lv->embed_dim);
}

// ---------------------------------------------------------------------------------------------
// actions
// ---------------------------------------------------------------------------------------------
extern "C" int jepa_hjepa_action_embed(jepa_context * ctx, int level, const float * actions, int n_rows, jepa_output * out) {
    if (!ctx || !actions || !out || n_rows <= 0) return -1;
    const jepa_hjepa_level * lv = check_level(ctx, level, "jepa_hjepa_action_embed");
    if (!lv) return -1;
    if (!rows_fit(ctx, "jepa_hjepa_action_embed", n_rows, (int64_t) lv->action_dim + lv->act_embed_dim)) return -1;
    jepa_graph_begin(ctx, 32);
    ggml_tensor * inp = new_input(ctx, lv->action_dim, n_rows);
    ggml_tensor * y = build_act_embed(ctx, level, inp);
    if (y == inp) return emit(out, actions, n_rows, lv->action_dim);   // identity
    const graph_input in = { inp, actions };
    std::vector<float> res;
    if (run_graph(ctx, y, &in, 1, res) != 0) return -1;
    return emit(out, res.data(), n_rows, lv->act_embed_dim);
}

// The 'seq' action encoder over B chunks of L steps each: inp [E, L, B] -> [action_dim, B].
//   x   = [ pos[0] ; step(inp) + pos[1..L] ]            a CLS row, then the L steps
//   x   = pre-LN transformer layers (ReLU FFN), every row sees every row
//   out = out_norm( head.2( head.ln( gelu( head.0( x[CLS] ) ) ) ) )
// The reference pads a short trailing chunk to `stride` steps and masks the padding as attention
// keys. Only the CLS row is read, and a masked key contributes nothing to it, so running the chunk
// at its true length L is the same function — which is why there is no mask here.
static ggml_tensor * build_act_enc_seq(jepa_context * ctx, int level, const jepa_hjepa_level & lv, ggml_tensor * inp) {
    const jepa_model * m = ctx->model;
    const jepa_hjepa_hparams & hj = m->hp.hjepa;
    ggml_context * g = ctx->ctx_g;
    const std::string p = lvl_prefix(level) + "act_enc.";
    const int64_t d = lv.act_enc_dim, L = inp->ne[1], B = inp->ne[2];
    ggml_tensor * pos = m->get(p + "pos");

    ggml_tensor * x = jepa_build_linear(g, inp, m->get(p + "step.weight"), m->get(p + "step.bias"));   // [d, L, B]
    x = ggml_add(g, x, ggml_view_2d(g, pos, d, L, pos->nb[1], pos->nb[1]));
    ggml_tensor * cls = ggml_view_2d(g, pos, d, 1, pos->nb[1], 0);
    if (B > 1) cls = ggml_repeat_4d(g, cls, d, 1, B, 1);
    x = ggml_concat(g, cls, x, 1);                                                                     // [d, 1 + L, B]

    jepa_block_opts opts;
    opts.ln_eps   = hj.ln_eps;
    opts.act      = JEPA_ACT_RELU;
    opts.n_head   = lv.act_enc_n_head;
    opts.head_dim = (int) (d / lv.act_enc_n_head);
    opts.attn.flash   = ctx->params.use_flash_attn && jepa_gpu_flash_ok(ctx, opts.head_dim);
    opts.attn.kv_type = ctx->is_gpu ? GGML_TYPE_F16 : GGML_TYPE_F32;   // as jepa_build_adaln: a handful of rows
    for (const jepa_layer & layer : m->hj_act_enc_layers[(size_t) level - 1]) x = jepa_build_block(g, x, layer, opts);

    ggml_tensor * y = ggml_cont(g, ggml_view_3d(g, x, d, 1, B, x->nb[1], x->nb[2], 0));   // the CLS rows
    y = ggml_reshape_2d(g, y, d, B);
    y = jepa_build_linear(g, y, m->get(p + "head.0.weight"), m->get(p + "head.0.bias"));
    y = jepa_build_act(g, y, JEPA_ACT_GELU_ERF);
    y = jepa_build_layer_norm(g, y, m->get(p + "head.ln.weight"), m->get(p + "head.ln.bias"), hj.ln_eps);
    y = jepa_build_linear(g, y, m->get(p + "head.2.weight"), m->get(p + "head.2.bias"));
    return jepa_build_layer_norm(g, y, m->get(p + "out_norm.weight"), m->get(p + "out_norm.bias"), 1e-6f);
}

extern "C" int jepa_hjepa_action_encode(jepa_context * ctx, int level, const float * below_act_emb, int n_rows, jepa_output * out) {
    if (!ctx || !below_act_emb || !out || n_rows <= 0) return -1;
    const jepa_hjepa_level * lv = check_level(ctx, level, "jepa_hjepa_action_encode");
    if (!lv) return -1;
    if (level < 2) {
        jepa_log("jepa: jepa_hjepa_action_encode: level 1 actions are given, not encoded; this takes level >= 2\n");
        return -1;
    }
    const jepa_model * m = ctx->model;
    const jepa_hjepa_hparams & hj = m->hp.hjepa;
    const int64_t E = hj.levels[(size_t) level - 2].act_embed_dim;
    const int64_t S = lv->stride, A = lv->action_dim;
    const int64_t n_full = n_rows / S, rem = n_rows % S, n_out = n_full + (rem ? 1 : 0);
    if (!rows_fit(ctx, "jepa_hjepa_action_encode", n_out, S * E + A)) return -1;
    std::vector<float> all((size_t) n_out * A), res;

    if (lv->act_enc_kind == "mlp") {
        // flatten each chunk; a short trailing chunk is zero-padded, as the reference does
        std::vector<float> flat((size_t) n_out * S * E, 0.0f);
        memcpy(flat.data(), below_act_emb, (size_t) n_rows * E * sizeof(float));
        jepa_graph_begin(ctx, 32);
        ggml_context * g = ctx->ctx_g;
        const std::string p = lvl_prefix(level) + "act_enc.";
        ggml_tensor * inp = new_input(ctx, S * E, n_out);
        ggml_tensor * y = jepa_build_mlp2(g, inp, m->get(p + "fc1.weight"), m->get(p + "fc1.bias"),
                                          m->get(p + "fc2.weight"), m->get(p + "fc2.bias"), JEPA_ACT_RELU);
        if (ggml_tensor * w = m->get(p + "out_norm.weight")) y = jepa_build_layer_norm(g, y, w, m->get(p + "out_norm.bias"), hj.ln_eps);
        const graph_input in = { inp, flat.data() };
        if (run_graph(ctx, y, &in, 1, res) != 0) return -1;
        return emit(out, res.data(), n_out, A);
    }

    // seq: the full chunks share one graph (the batch dimension), a short trailing chunk gets its own
    double total_ms = 0;
    const size_t n_nodes = (size_t) lv->act_enc_n_layer * 48 + 64;
    if (n_full > 0) {
        jepa_graph_begin(ctx, n_nodes);
        ggml_tensor * inp = new_input(ctx, E, S, n_full);
        const graph_input in = { inp, below_act_emb };
        if (run_graph(ctx, build_act_enc_seq(ctx, level, *lv, inp), &in, 1, res) != 0) return -1;
        total_ms += ctx->last_compute_ms;
        memcpy(all.data(), res.data(), (size_t) n_full * A * sizeof(float));
    }
    if (rem > 0) {
        jepa_graph_begin(ctx, n_nodes);
        ggml_tensor * inp = new_input(ctx, E, rem, 1);
        const graph_input in = { inp, below_act_emb + (size_t) n_full * S * E };
        if (run_graph(ctx, build_act_enc_seq(ctx, level, *lv, inp), &in, 1, res) != 0) return -1;
        total_ms += ctx->last_compute_ms;
        memcpy(all.data() + (size_t) n_full * A, res.data(), (size_t) A * sizeof(float));
    }
    ctx->last_compute_ms = total_ms;
    return emit(out, all.data(), n_out, A);
}

// ---------------------------------------------------------------------------------------------
// dynamics
// ---------------------------------------------------------------------------------------------
extern "C" int jepa_hjepa_predict(jepa_context * ctx, int level, const float * states, const float * actions,
                                  int n_frames, jepa_output * out) {
    if (!ctx || !states || !actions || !out || n_frames <= 0) return -1;
    const jepa_hjepa_level * lv = check_level(ctx, level, "jepa_hjepa_predict");
    if (!lv) return -1;
    const jepa_model * m = ctx->model;
    const jepa_hjepa_hparams & hj = m->hp.hjepa;
    const int64_t D = lv->embed_dim, A = lv->action_dim, T = n_frames;
    if (T > lv->pred_n_frames) {
        jepa_log("jepa: jepa_hjepa_predict: %d frames requested, the level-%d predictor has %d position slots\n",
                 n_frames, level, lv->pred_n_frames);
        return -1;
    }
    // T <= pred_n_frames <= JEPA_LIMIT_N_FRAMES, so the T^2 mask is bounded; the rest is linear
    if (!rows_fit(ctx, "jepa_hjepa_predict", T, D + A + 4 * T)) return -1;

    jepa_graph_begin(ctx, (size_t) lv->pred_n_layer * 64 + 160);
    ggml_context * g = ctx->ctx_g;
    const std::string p = lvl_prefix(level) + "pred.";
    ggml_tensor * emb = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_tensor * act = ggml_new_tensor_2d(g, GGML_TYPE_F32, A, T);
    ggml_set_name(emb, "state");
    ggml_set_name(act, "action");
    ggml_set_input(emb);
    ggml_set_input(act);
    ggml_tensor * mask = nullptr;
    std::vector<ggml_fp16_t> mdata;
    if (T > 1) {
        mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, T, T);
        ggml_set_input(mask);
        jepa_causal_mask_f16(T, mdata);
    }

    ggml_tensor * c = build_act_embed(ctx, level, act);
    if (ggml_tensor * w0 = m->get(p + "act_embed.0.weight")) {   // adaln_bare: the predictor's own embedder
        c = jepa_build_mlp2(g, c, w0, m->get(p + "act_embed.0.bias"), m->get(p + "act_embed.2.weight"),
                            m->get(p + "act_embed.2.bias"), JEPA_ACT_SILU);
    }
    jepa_adaln_spec s;
    s.pos      = m->get(p + "pos_embed");
    s.in_w     = m->get(p + "in_proj.weight");    s.in_b    = m->get(p + "in_proj.bias");
    s.cond_w   = m->get(p + "cond_proj.weight");  s.cond_b  = m->get(p + "cond_proj.bias");
    s.layers   = &m->hj_pred_layers[(size_t) level - 1];
    s.norm_w   = m->get(p + "norm.weight");       s.norm_b  = m->get(p + "norm.bias");
    s.out_w    = m->get(p + "out_proj.weight");   s.out_b   = m->get(p + "out_proj.bias");
    s.proj0_w  = m->get(p + "proj.0.weight");     s.proj0_b = m->get(p + "proj.0.bias");
    s.proj2_w  = m->get(p + "proj.2.weight");     s.proj2_b = m->get(p + "proj.2.bias");
    s.n_head   = lv->pred_n_head;
    s.head_dim = lv->pred_head_dim;
    s.ln_eps   = hj.ln_eps;
    s.adaln_eps = hj.adaln_eps;
    ggml_tensor * y = jepa_build_adaln(ctx, s, emb, c, mask);

    ggml_build_forward_expand(ctx->gf, y);
    if (!jepa_graph_alloc(ctx)) return -1;
    ggml_backend_tensor_set(emb, states, 0, (size_t) D * T * sizeof(float));
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

extern "C" int jepa_hjepa_rollout(jepa_context * ctx, int level, const float * states, int n_seed,
                                  const float * actions, int n_steps, int history, float * out) {
    if (!ctx || !states || !actions || !out || n_seed <= 0 || n_steps <= 0) return -1;
    const jepa_hjepa_level * lv = check_level(ctx, level, "jepa_hjepa_rollout");
    if (!lv) return -1;
    if (history < 1 || history > lv->pred_n_frames) {
        jepa_log("jepa: jepa_hjepa_rollout: history %d is outside 1..%d (the level-%d predictor's position slots)\n",
                 history, lv->pred_n_frames, level);
        return -1;
    }
    const int64_t D = lv->embed_dim, A = lv->action_dim;
    const int64_t n_seq = (int64_t) n_seed + n_steps;   // int64: a sum of two caller-supplied ints
    if (!rows_fit(ctx, "jepa_hjepa_rollout", n_seq, D + A)) return -1;

    // Growing sequence: the seeds, then one predicted state per step. Row j is conditioned on
    // actions[j], so the caller supplies n_seed + n_steps - 1 of them.
    std::vector<float> seq((size_t) n_seq * D);
    memcpy(seq.data(), states, (size_t) n_seed * D * sizeof(float));
    double total_ms = 0;
    for (int step = 0; step < n_steps; step++) {
        const int64_t have = (int64_t) n_seed + step;
        const int w = have < history ? (int) have : history;
        const int64_t first = have - w;
        jepa_output one = {};
        if (jepa_hjepa_predict(ctx, level, seq.data() + (size_t) first * D, actions + (size_t) first * A, w, &one) != 0) return -1;
        total_ms += ctx->last_compute_ms;
        const float * last = one.data + (size_t) (w - 1) * D;
        memcpy(seq.data() + (size_t) have * D, last, (size_t) D * sizeof(float));
        memcpy(out + (size_t) step * D, last, (size_t) D * sizeof(float));
        free(one.data);
    }
    ctx->last_compute_ms = total_ms;
    return 0;
}
