// jepa.cpp — core: context, graph builders (LN / linear / attention / block), image encoder graph,
// pooling and the LeWM projector. GGUF loading is in jepa-gguf.cpp, preprocessing in preprocess.cpp.
#include "jepa-internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

void jepa_log(const char * fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
}
const char * jepa_version(void) { return JEPA_VERSION; }
void jepa_print_system_info(void) {
    fprintf(stderr, "jepa.cpp %s | ggml | threads available: %d\n", JEPA_VERSION, (int) std::thread::hardware_concurrency());
}
jepa_context_params jepa_context_default_params(void) {
    jepa_context_params p;
    p.n_threads = 0;
    p.use_flash_attn = true;
    p.verbose = false;
    p.flash_kv = JEPA_KV_AUTO;
    return p;
}

// ---------------------------------------------------------------------------------------------
// context
// ---------------------------------------------------------------------------------------------
jepa_context * jepa_context_new(jepa_model * model, jepa_context_params params) {
    if (!model) return nullptr;
    jepa_context * ctx = new jepa_context();
    ctx->model  = model;
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : (int) std::thread::hardware_concurrency();
    if (ctx->n_threads < 1) ctx->n_threads = 1;
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        jepa_log("jepa: ggml_backend_cpu_init() failed\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    return ctx;
}

void jepa_context_free(jepa_context * ctx) {
    if (!ctx) return;
    if (ctx->ctx_g)   ggml_free(ctx->ctx_g);
    if (ctx->galloc)  ggml_gallocr_free(ctx->galloc);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

int    jepa_context_n_threads(const jepa_context * ctx)       { return ctx ? ctx->n_threads : 0; }
double jepa_context_last_compute_ms(const jepa_context * ctx) { return ctx ? ctx->last_compute_ms : 0.0; }

// ---------------------------------------------------------------------------------------------
// graph execution
// ---------------------------------------------------------------------------------------------
void jepa_graph_begin(jepa_context * ctx, size_t max_nodes) {
    if (max_nodes < 64) max_nodes = 64;
    const size_t mem = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false) + 4096;
    if (ctx->ctx_g) { ggml_free(ctx->ctx_g); ctx->ctx_g = nullptr; }
    ctx->graph_meta.resize(mem);
    ggml_init_params ip = { mem, ctx->graph_meta.data(), true };
    ctx->ctx_g = ggml_init(ip);
    ctx->gf = ggml_new_graph_custom(ctx->ctx_g, max_nodes, false);
    ctx->graph_max_nodes = max_nodes;
}

bool jepa_graph_alloc(jepa_context * ctx) {
    if (!ggml_gallocr_alloc_graph(ctx->galloc, ctx->gf)) {
        jepa_log("jepa: ggml_gallocr_alloc_graph failed (%d nodes)\n", ggml_graph_n_nodes(ctx->gf));
        return false;
    }
    return true;
}

int jepa_graph_compute(jepa_context * ctx) {
    auto t0 = std::chrono::steady_clock::now();
    ggml_status st = ggml_backend_graph_compute(ctx->backend, ctx->gf);
    auto t1 = std::chrono::steady_clock::now();
    ctx->last_compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (st != GGML_STATUS_SUCCESS) {
        jepa_log("jepa: ggml_backend_graph_compute failed (status %d)\n", (int) st);
        return -1;
    }
    return 0;
}

int jepa_graph_run(jepa_context * ctx, ggml_tensor * inp, const float * inp_data, size_t inp_bytes,
                   ggml_tensor * out_t, std::vector<float> & out) {
    ggml_build_forward_expand(ctx->gf, out_t);
    if (!jepa_graph_alloc(ctx)) return -1;
    if (inp && inp_data) {
        if (inp_bytes != ggml_nbytes(inp)) {
            jepa_log("jepa: input size mismatch (%zu vs tensor %zu bytes)\n", inp_bytes, ggml_nbytes(inp));
            return -1;
        }
        ggml_backend_tensor_set(inp, inp_data, 0, inp_bytes);
    }
    if (jepa_graph_compute(ctx) != 0) return -1;
    out.resize((size_t) ggml_nelements(out_t));
    if (out_t->type != GGML_TYPE_F32) {
        jepa_log("jepa: output tensor is not F32\n");
        return -1;
    }
    ggml_backend_tensor_get(out_t, out.data(), 0, ggml_nbytes(out_t));
    return 0;
}

// ---------------------------------------------------------------------------------------------
// graph builders
// ---------------------------------------------------------------------------------------------
ggml_tensor * jepa_build_layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    if (w) x = ggml_mul(ctx, x, w);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor * jepa_build_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_mul_mat(ctx, w, x);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor * jepa_build_act(ggml_context * ctx, ggml_tensor * x, jepa_act_id act) {
    switch (act) {
        case JEPA_ACT_GELU_ERF:  return ggml_gelu_erf(ctx, x);
        case JEPA_ACT_GELU_TANH: return ggml_gelu(ctx, x);
        case JEPA_ACT_SILU:      return ggml_silu(ctx, x);
    }
    return x;
}

ggml_tensor * jepa_build_mlp2(ggml_context * ctx, ggml_tensor * x,
                              ggml_tensor * w1, ggml_tensor * b1, ggml_tensor * w2, ggml_tensor * b2, jepa_act_id act) {
    x = jepa_build_linear(ctx, x, w1, b1);
    x = jepa_build_act(ctx, x, act);
    return jepa_build_linear(ctx, x, w2, b2);
}

void jepa_build_qkv(ggml_context * ctx, ggml_tensor * x, const jepa_layer & L, int n_head, int head_dim,
                    ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v) {
    const int64_t N = x->ne[1];
    const int64_t inner = (int64_t) n_head * head_dim;
    if (L.qkv_w) {
        ggml_tensor * qkv = jepa_build_linear(ctx, x, L.qkv_w, L.qkv_b);   // [3*inner, N]
        GGML_ASSERT(qkv->ne[0] == 3 * inner);
        const size_t es = ggml_element_size(qkv);
        *q = ggml_view_3d(ctx, qkv, head_dim, n_head, N, head_dim * es, qkv->nb[1], 0);
        *k = ggml_view_3d(ctx, qkv, head_dim, n_head, N, head_dim * es, qkv->nb[1], inner * es);
        *v = ggml_view_3d(ctx, qkv, head_dim, n_head, N, head_dim * es, qkv->nb[1], 2 * inner * es);
    } else {
        GGML_ASSERT(L.q_w && L.k_w && L.v_w);
        *q = ggml_reshape_3d(ctx, jepa_build_linear(ctx, x, L.q_w, L.q_b), head_dim, n_head, N);
        *k = ggml_reshape_3d(ctx, jepa_build_linear(ctx, x, L.k_w, L.k_b), head_dim, n_head, N);
        *v = ggml_reshape_3d(ctx, jepa_build_linear(ctx, x, L.v_w, L.v_b), head_dim, n_head, N);
    }
}

ggml_tensor * jepa_build_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                   const jepa_attn_opts & opts) {
    const int64_t hd = q->ne[0], n_head = q->ne[1], n_q = q->ne[2];
    const float scale = opts.scale > 0.0f ? opts.scale : 1.0f / sqrtf((float) hd);
    // [hd, n_head, N] -> [hd, N, n_head]
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
    if (opts.flash) {
        if (opts.kv_type != GGML_TYPE_F32) {
            kp = ggml_cast(ctx, kp, opts.kv_type);
            vp = ggml_cast(ctx, vp, opts.kv_type);
        }
        ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, kp, vp, opts.mask, scale, 0.0f, 0.0f);  // [hd, n_head, n_q]
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        return ggml_reshape_2d(ctx, out, hd * n_head, n_q);
    }
    // naive: softmax(k^T q * scale) v
    ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                       // [n_kv, n_q, n_head]
    kq = ggml_soft_max_ext(ctx, kq, opts.mask, scale, 0.0f);
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [n_kv, hd, n_head]
    ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);                      // [hd, n_q, n_head]
    ggml_tensor * out = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [hd, n_head, n_q]
    return ggml_reshape_2d(ctx, out, hd * n_head, n_q);
}

ggml_tensor * jepa_build_block(ggml_context * ctx, ggml_tensor * x, const jepa_layer & L, const jepa_block_opts & opts) {
    const int64_t D = x->ne[0];
    const int n_head = opts.n_head;
    const int head_dim = opts.head_dim > 0 ? opts.head_dim : (int) (D / n_head);
    GGML_ASSERT(n_head > 0 && head_dim > 0);

    // attention
    ggml_tensor * h = jepa_build_layer_norm(ctx, x, L.ln1_w, L.ln1_b, opts.ln_eps);
    ggml_tensor * q, * k, * v;
    jepa_build_qkv(ctx, h, L, n_head, head_dim, &q, &k, &v);
    if (opts.qk_hook) {
        q = opts.qk_hook(ctx, q, false);
        k = opts.qk_hook(ctx, k, true);
    }
    ggml_tensor * a = jepa_build_attention(ctx, q, k, v, opts.attn);
    a = jepa_build_linear(ctx, a, L.out_w, L.out_b);
    if (L.ls1) a = ggml_mul(ctx, a, L.ls1);
    x = ggml_add(ctx, x, a);

    // ffn
    h = jepa_build_layer_norm(ctx, x, L.ln2_w, L.ln2_b, opts.ln_eps);
    ggml_tensor * f = jepa_build_linear(ctx, h, L.up_w, L.up_b);
    f = jepa_build_act(ctx, f, opts.act);
    f = jepa_build_linear(ctx, f, L.down_w, L.down_b);
    if (L.ls2) f = ggml_mul(ctx, f, L.ls2);
    x = ggml_add(ctx, x, f);
    return x;
}

// ---------------------------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------------------------
int64_t jepa_patchify(const float * cthw, int C, int T, int H, int W, int patch, int tubelet,
                      std::vector<float> & rows, int * gt, int * gh, int * gw) {
    const int GT = T / tubelet, GH = H / patch, GW = W / patch;
    const int64_t N = (int64_t) GT * GH * GW;
    const int64_t K = (int64_t) C * tubelet * patch * patch;
    rows.resize((size_t) N * K);
    for (int t = 0; t < GT; t++) {
        for (int h = 0; h < GH; h++) {
            for (int w = 0; w < GW; w++) {
                float * row = rows.data() + ((int64_t) t * GH * GW + (int64_t) h * GW + w) * K;
                int64_t o = 0;
                for (int c = 0; c < C; c++) {
                    for (int tt = 0; tt < tubelet; tt++) {
                        const int64_t plane = ((int64_t) c * T + (int64_t) t * tubelet + tt) * H * W;
                        for (int py = 0; py < patch; py++) {
                            const float * src = cthw + plane + (int64_t) (h * patch + py) * W + (int64_t) w * patch;
                            memcpy(row + o, src, patch * sizeof(float));
                            o += patch;
                        }
                    }
                }
            }
        }
    }
    if (gt) *gt = GT;
    if (gh) *gh = GH;
    if (gw) *gw = GW;
    return N;
}

// ---------------------------------------------------------------------------------------------
// image encoder graph (ijepa / hfvit / lewm)
// ---------------------------------------------------------------------------------------------
ggml_type jepa_context_kv_type(const jepa_context * ctx) {
    switch (ctx->params.flash_kv) {
        case JEPA_KV_F16: return GGML_TYPE_F16;
        case JEPA_KV_F32: return GGML_TYPE_F32;
        default:          return ctx->model->hp.file_type == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    }
}

static jepa_block_opts encoder_block_opts(const jepa_context * ctx) {
    const jepa_enc_hparams & e = ctx->model->hp.enc;
    jepa_block_opts o;
    o.ln_eps = e.ln_eps;
    o.act = e.act;
    o.n_head = e.n_head;
    o.head_dim = e.head_dim();
    o.attn.flash = ctx->params.use_flash_attn;
    o.attn.kv_type = jepa_context_kv_type(ctx);
    return o;
}

ggml_tensor * jepa_build_encoder_image(jepa_context * ctx, ggml_tensor * inp, int gh, int gw) {
    const jepa_model * m = ctx->model;
    const jepa_enc_hparams & e = m->hp.enc;
    ggml_context * g = ctx->ctx_g;
    const int64_t D = e.embed_dim;
    const int64_t n_patches = (int64_t) gh * gw;
    GGML_ASSERT(inp->ne[1] == n_patches);

    ggml_tensor * x = jepa_build_linear(g, inp, m->patch_embed_w, m->patch_embed_b);   // [D, N]
    if (e.n_registers > 0) {
        GGML_ASSERT(m->reg_tokens && m->reg_tokens->ne[1] == e.n_registers);
        x = ggml_concat(g, m->reg_tokens, x, 1);
    }
    if (e.cls_token) {
        x = ggml_concat(g, ggml_reshape_2d(g, m->cls_token, D, 1), x, 1);
    }
    if (m->pos_embed) {
        if (m->pos_embed->ne[1] != x->ne[1]) {
            jepa_log("jepa: token grid %dx%d (+%d prefix) = %lld tokens does not match enc.pos_embed (%lld rows); "
                     "position-table interpolation is not implemented yet — feed a %dx%d input\n",
                     gh, gw, (int) (x->ne[1] - n_patches), (long long) x->ne[1], (long long) m->pos_embed->ne[1],
                     e.img_size, e.img_size);
            return nullptr;
        }
        x = ggml_add(g, x, m->pos_embed);
    }
    const jepa_block_opts opts = encoder_block_opts(ctx);
    for (int i = 0; i < e.n_layer; i++) {
        x = jepa_build_block(g, x, m->enc_layers[i], opts);
    }
    x = jepa_build_layer_norm(g, x, m->norm_w, m->norm_b, e.ln_eps);
    ggml_set_name(x, "last_hidden_state");
    ggml_set_output(x);
    return x;
}

static size_t encoder_graph_nodes(const jepa_model * m) {
    return (size_t) m->hp.enc.n_layer * 40 + 64;
}

// One graph per (batch, frame) slice: the image families have no temporal dimension.
static int jepa_encode_image(jepa_context * ctx, const jepa_input * in, jepa_output * out) {
    const jepa_model * m = ctx->model;
    const jepa_enc_hparams & e = m->hp.enc;
    if (in->height % e.patch_size != 0 || in->width % e.patch_size != 0) {
        jepa_log("jepa: input %dx%d is not a multiple of the patch size %d\n", in->height, in->width, e.patch_size);
        return -1;
    }
    const int gh = in->height / e.patch_size, gw = in->width / e.patch_size;
    const int64_t n_patches = (int64_t) gh * gw;
    const int64_t n_tokens  = n_patches + (e.cls_token ? 1 : 0) + e.n_registers;
    const int64_t K = (int64_t) e.in_chans * e.patch_size * e.patch_size;
    const int64_t n_items = (int64_t) in->n_batch * in->n_frames;
    const size_t  plane = (size_t) in->height * in->width;

    out->n_tokens = n_items * n_tokens;
    out->dim = e.embed_dim;
    out->data = (float *) malloc((size_t) out->n_tokens * out->dim * sizeof(float));
    if (!out->data) return -1;

    std::vector<float> chw((size_t) e.in_chans * plane), rows, res;
    double total_ms = 0;
    for (int64_t item = 0; item < n_items; item++) {
        const int64_t b = item / in->n_frames, t = item % in->n_frames;
        // gather the (b, t) slice into CHW
        for (int c = 0; c < e.in_chans; c++) {
            const float * src = in->data + (((size_t) b * in->n_chans + c) * in->n_frames + t) * plane;
            memcpy(chw.data() + (size_t) c * plane, src, plane * sizeof(float));
        }
        jepa_patchify(chw.data(), e.in_chans, 1, in->height, in->width, e.patch_size, 1, rows, nullptr, nullptr, nullptr);

        jepa_graph_begin(ctx, encoder_graph_nodes(m));
        ggml_tensor * inp = ggml_new_tensor_2d(ctx->ctx_g, GGML_TYPE_F32, K, n_patches);
        ggml_set_name(inp, "patches");
        ggml_set_input(inp);
        ggml_tensor * y = jepa_build_encoder_image(ctx, inp, gh, gw);
        if (!y) { free(out->data); out->data = nullptr; return -1; }
        if (jepa_graph_run(ctx, inp, rows.data(), rows.size() * sizeof(float), y, res) != 0) {
            free(out->data); out->data = nullptr;
            return -1;
        }
        total_ms += ctx->last_compute_ms;
        memcpy(out->data + (size_t) item * n_tokens * e.embed_dim, res.data(), res.size() * sizeof(float));
    }
    ctx->last_compute_ms = total_ms;
    return 0;
}

// ---------------------------------------------------------------------------------------------
// video encoder (V-JEPA 2 / V-JEPA 2.1)
// ---------------------------------------------------------------------------------------------
// What the two families share: tubelet patchify -> one mul_mat, 3-D RoPE on q/k of every block
// (host-side cos/sin tables uploaded once per input shape, src/rope3d.*), full attention (no mask),
// final enc.norm.  V-JEPA 2.1 adds the 1-frame image tokenizer (enc.patch_embed_img) and the
// img/video modality vector added to every token.  Exact conventions:
// scripts/jepa_convert/VJEPA_NOTES.md S3-S4 and its numpy twin vjepa2_numpy_ref.py.
bool jepa_video_shape_for(const jepa_model * m, int n_frames, int height, int width, jepa_video_shape & vs, bool verbose) {
    const jepa_enc_hparams & e = m->hp.enc;
    const int P = e.patch_size;
    if (P <= 0 || n_frames <= 0 || height <= 0 || width <= 0) return false;
    if (height % P != 0 || width % P != 0) {
        jepa_log("jepa: input %dx%d is not a multiple of the patch size %d\n", height, width, P);
        return false;
    }
    vs.image_path = false;
    vs.tubelet = e.tubelet_size > 0 ? e.tubelet_size : 1;
    if (n_frames == 1 && e.image_patch_embed && m->patch_embed_img_w) {
        // V-JEPA 2.1 native image path: 1x16x16 patches + the image modality vector.
        vs.image_path = true;
        vs.tubelet = 1;
    }
    if (n_frames % vs.tubelet != 0) {
        jepa_log("jepa: %d frames is not a multiple of the tubelet size %d (an image has to be fed to this "
                 "model as a %d-frame clip, i.e. repeated)\n", n_frames, vs.tubelet, vs.tubelet);
        return false;
    }
    vs.gt = n_frames / vs.tubelet;
    vs.gh = height / P;
    vs.gw = width / P;
    vs.n_tokens  = (int64_t) vs.gt * vs.gh * vs.gw;
    vs.patch_dim = (int64_t) e.in_chans * vs.tubelet * P * P;
    ggml_tensor * pe = vs.image_path ? m->patch_embed_img_w : m->patch_embed_w;
    if (!pe || pe->ne[0] != vs.patch_dim) {
        jepa_log("jepa: patch embedding %s takes rows of %lld values, this shape needs %lld\n",
                 vs.image_path ? "enc.patch_embed_img.weight" : "enc.patch_embed.weight",
                 pe ? (long long) pe->ne[0] : 0LL, (long long) vs.patch_dim);
        return false;
    }
    if (verbose) {
        jepa_log("jepa: %s path: %d frames %dx%d -> grid %dx%dx%d = %lld tokens (tubelet %d, patch %d)\n",
                 vs.image_path ? "image" : "video", n_frames, height, width, vs.gt, vs.gh, vs.gw,
                 (long long) vs.n_tokens, vs.tubelet, P);
    }
    return true;
}

jepa_rope3d_params jepa_encoder_rope_params(const jepa_model * m, int gt, int gh, int gw) {
    const jepa_enc_hparams & e = m->hp.enc;
    jepa_rope3d_params rp;
    rp.grid_t = gt; rp.grid_h = gh; rp.grid_w = gw;
    rp.head_dim = e.head_dim();
    rp.theta = e.rope_theta;
    // "interleaved" = V-JEPA 2.1 (true rotation); "tiled" / unset = V-JEPA 2 (the HF/Meta layout).
    rp.variant = e.rope_freq_layout == "interleaved" ? JEPA_ROPE3D_VJEPA2_1 : JEPA_ROPE3D_VJEPA2;
    rp.interpolate = e.rope_interpolate;
    rp.train_grid_h = rp.train_grid_w = e.rope_ref_grid;
    return rp;
}

ggml_tensor * jepa_build_encoder_video(jepa_context * ctx, ggml_tensor * inp,
                                       ggml_tensor * cos_t, ggml_tensor * sin_t, bool image_path) {
    const jepa_model * m = ctx->model;
    const jepa_enc_hparams & e = m->hp.enc;
    ggml_context * g = ctx->ctx_g;

    ggml_tensor * pw = image_path ? m->patch_embed_img_w : m->patch_embed_w;
    ggml_tensor * pb = image_path ? m->patch_embed_img_b : m->patch_embed_b;
    ggml_tensor * x = jepa_build_linear(g, inp, pw, pb);                        // [D, N]
    if (e.modality_embed) {
        ggml_tensor * mod = image_path ? m->mod_embed_img : m->mod_embed_video;
        if (!mod) {
            jepa_log("jepa: jepa.enc.modality_embed is set but enc.mod_embed_%s is missing\n", image_path ? "img" : "video");
            return nullptr;
        }
        x = ggml_add(g, x, mod);
    }
    jepa_block_opts opts = encoder_block_opts(ctx);
    opts.qk_hook = [cos_t, sin_t](ggml_context * c, ggml_tensor * t, bool /*is_k*/) {
        return jepa_rope3d_apply(c, t, cos_t, sin_t);   // q and k (not v), before the 1/sqrt(d) scale
    };
    for (int i = 0; i < e.n_layer; i++) {
        x = jepa_build_block(g, x, m->enc_layers[i], opts);
    }
    // 2.1: enc.norm is norms_block[-1], which is what VisionTransformer.forward returns.
    x = jepa_build_layer_norm(g, x, m->norm_w, m->norm_b, e.ln_eps);
    ggml_set_name(x, "last_hidden_state");
    ggml_set_output(x);
    return x;
}

static size_t video_graph_nodes(const jepa_model * m) {
    // ~55 nodes per block: LN 3 + qkv 2 + 3 views + 2x13 RoPE + attention 7 + out 2 + ffn 5 + 2 adds
    return (size_t) m->hp.enc.n_layer * 96 + 256;
}

static int jepa_encode_video(jepa_context * ctx, const jepa_input * in, jepa_output * out) {
    const jepa_model * m = ctx->model;
    const jepa_enc_hparams & e = m->hp.enc;
    jepa_video_shape vs;
    if (!jepa_video_shape_for(m, in->n_frames, in->height, in->width, vs, ctx->params.verbose)) return -1;

    const int64_t D = e.embed_dim;
    const int64_t N = vs.n_tokens;
    const int64_t n_clips = in->n_batch;
    out->n_tokens = n_clips * N;
    out->dim = D;
    out->data = (float *) malloc((size_t) out->n_tokens * D * sizeof(float));
    if (!out->data) return -1;

    // RoPE tables: identical for every clip of this shape, so build them once and upload per graph.
    std::vector<float> rope_cos, rope_sin;
    jepa_rope3d_tables(jepa_encoder_rope_params(m, vs.gt, vs.gh, vs.gw), rope_cos, rope_sin);
    const int hd = e.head_dim();

    const size_t clip_floats = (size_t) in->n_chans * in->n_frames * in->height * in->width;
    std::vector<float> rows;
    double total_ms = 0;
    for (int64_t b = 0; b < n_clips; b++) {
        jepa_patchify(in->data + (size_t) b * clip_floats, in->n_chans, in->n_frames, in->height, in->width,
                      e.patch_size, vs.tubelet, rows, nullptr, nullptr, nullptr);

        jepa_graph_begin(ctx, video_graph_nodes(m));
        ggml_tensor * inp = ggml_new_tensor_2d(ctx->ctx_g, GGML_TYPE_F32, vs.patch_dim, N);
        ggml_set_name(inp, "patches");
        ggml_set_input(inp);
        ggml_tensor * cos_t = ggml_new_tensor_3d(ctx->ctx_g, GGML_TYPE_F32, hd, 1, N);
        ggml_tensor * sin_t = ggml_new_tensor_3d(ctx->ctx_g, GGML_TYPE_F32, hd, 1, N);
        ggml_set_name(cos_t, "rope_cos"); ggml_set_input(cos_t);
        ggml_set_name(sin_t, "rope_sin"); ggml_set_input(sin_t);
        ggml_tensor * y = jepa_build_encoder_video(ctx, inp, cos_t, sin_t, vs.image_path);
        if (!y) { free(out->data); out->data = nullptr; return -1; }
        ggml_build_forward_expand(ctx->gf, y);
        if (!jepa_graph_alloc(ctx)) { free(out->data); out->data = nullptr; return -1; }
        ggml_backend_tensor_set(inp, rows.data(), 0, rows.size() * sizeof(float));
        ggml_backend_tensor_set(cos_t, rope_cos.data(), 0, rope_cos.size() * sizeof(float));
        ggml_backend_tensor_set(sin_t, rope_sin.data(), 0, rope_sin.size() * sizeof(float));
        if (jepa_graph_compute(ctx) != 0) { free(out->data); out->data = nullptr; return -1; }
        total_ms += ctx->last_compute_ms;
        ggml_backend_tensor_get(y, out->data + (size_t) b * N * D, 0, (size_t) N * D * sizeof(float));
    }
    ctx->last_compute_ms = total_ms;
    return 0;
}

// ---------------------------------------------------------------------------------------------
// attentive-pool head (V-JEPA 2 video classifiers)
// ---------------------------------------------------------------------------------------------
// HF VJEPA2AttentivePooler + classifier, in this exact order (VJEPA_NOTES.md S2):
//   x = enc_out; for i in 0..n_pool_layers-1: x = block_i(x)        # plain self-attention, no RoPE
//   q0 = head.query (raw, un-normalised); kv = ln_kv(x)             # the LN is on K/V only
//   o  = softmax(q Wq^T (kv Wk^T)^T / sqrt(hd)) (kv Wv^T)           # NO output projection
//   y  = q0 + o ; y += ffn_down(gelu(ffn_up(ln2(y))))               # residual on the RAW query
//   logits = y Wcls^T + bcls                                        # no final norm
ggml_tensor * jepa_build_head(jepa_context * ctx, ggml_tensor * inp, ggml_tensor ** pooled_out) {
    const jepa_model * m = ctx->model;
    const jepa_enc_hparams & e = m->hp.enc;
    ggml_context * g = ctx->ctx_g;
    const int64_t D = e.embed_dim, N = inp->ne[1];
    const int n_head = e.n_head, hd = e.head_dim();

    ggml_tensor * x = inp;
    jepa_block_opts opts = encoder_block_opts(ctx);   // same ln_eps / act / heads, no RoPE hook
    for (size_t i = 0; i < m->head_layers.size(); i++) {
        x = jepa_build_block(g, x, m->head_layers[i], opts);
    }

    ggml_tensor * q0 = ggml_reshape_2d(g, m->require("head.query"), D, 1);
    ggml_tensor * kv = jepa_build_layer_norm(g, x, m->require("head.xattn.ln_kv.weight"),
                                             m->get("head.xattn.ln_kv.bias"), e.ln_eps);
    ggml_tensor * q = jepa_build_linear(g, q0, m->require("head.xattn.q.weight"), m->get("head.xattn.q.bias"));
    ggml_tensor * k = jepa_build_linear(g, kv, m->require("head.xattn.k.weight"), m->get("head.xattn.k.bias"));
    ggml_tensor * v = jepa_build_linear(g, kv, m->require("head.xattn.v.weight"), m->get("head.xattn.v.bias"));
    jepa_attn_opts ao;
    ao.flash = ctx->params.use_flash_attn;
    ao.kv_type = GGML_TYPE_F32;   // N_q == 1 -> ggml's per-row kernel, which would round q and the
                                  // PV accumulator to F16 with F16 K/V (docs/ggml-notes.md S1)
    ggml_tensor * o = jepa_build_attention(g, ggml_reshape_3d(g, q, hd, n_head, 1),
                                              ggml_reshape_3d(g, k, hd, n_head, N),
                                              ggml_reshape_3d(g, v, hd, n_head, N), ao);   // [D, 1]
    ggml_tensor * y = ggml_add(g, o, q0);                       // residual = the raw query token
    ggml_tensor * h = jepa_build_layer_norm(g, y, m->require("head.xattn.ln2.weight"), m->get("head.xattn.ln2.bias"), e.ln_eps);
    h = jepa_build_mlp2(g, h, m->require("head.xattn.ffn_up.weight"), m->get("head.xattn.ffn_up.bias"),
                        m->require("head.xattn.ffn_down.weight"), m->get("head.xattn.ffn_down.bias"), e.act);
    y = ggml_add(g, y, h);
    ggml_set_name(y, "pooled");
    ggml_set_output(y);
    if (pooled_out) *pooled_out = y;
    ggml_tensor * logits = jepa_build_linear(g, y, m->require("head.cls.weight"), m->get("head.cls.bias"));
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    return logits;
}

int jepa_head_ex(jepa_context * ctx, const jepa_output * enc, jepa_output * pooled, jepa_output * logits) {
    if (!ctx || !enc || !enc->data || enc->n_tokens <= 0) return -1;
    const jepa_model * m = ctx->model;
    if (!m->hp.head.present || m->hp.head.kind != "attentive_pool") {
        jepa_log("jepa: jepa_head: model '%s' has no attentive-pool head\n", m->hp.name.c_str());
        return -1;
    }
    if (enc->dim != m->hp.enc.embed_dim) {
        jepa_log("jepa: jepa_head: encoder rows are %lld wide, expected %d\n", (long long) enc->dim, m->hp.enc.embed_dim);
        return -1;
    }
    const int64_t D = enc->dim, N = enc->n_tokens;
    jepa_graph_begin(ctx, (size_t) m->hp.head.n_pool_layers * 64 + 128);
    ggml_tensor * inp = ggml_new_tensor_2d(ctx->ctx_g, GGML_TYPE_F32, D, N);
    ggml_set_name(inp, "tokens");
    ggml_set_input(inp);
    ggml_tensor * pooled_t = nullptr;
    ggml_tensor * logits_t = jepa_build_head(ctx, inp, &pooled_t);
    if (!logits_t) return -1;
    ggml_build_forward_expand(ctx->gf, pooled_t);
    ggml_build_forward_expand(ctx->gf, logits_t);
    if (!jepa_graph_alloc(ctx)) return -1;
    ggml_backend_tensor_set(inp, enc->data, 0, (size_t) D * N * sizeof(float));
    if (jepa_graph_compute(ctx) != 0) return -1;
    if (pooled) {
        pooled->n_tokens = 1; pooled->dim = D;
        pooled->data = (float *) malloc((size_t) D * sizeof(float));
        if (!pooled->data) return -1;
        ggml_backend_tensor_get(pooled_t, pooled->data, 0, (size_t) D * sizeof(float));
    }
    if (logits) {
        logits->n_tokens = 1; logits->dim = logits_t->ne[0];
        logits->data = (float *) malloc((size_t) logits_t->ne[0] * sizeof(float));
        if (!logits->data) return -1;
        ggml_backend_tensor_get(logits_t, logits->data, 0, (size_t) logits_t->ne[0] * sizeof(float));
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// small host helpers for the classification tools
// ---------------------------------------------------------------------------------------------
void jepa_softmax(const float * logits, int n, float * probs) {
    if (n <= 0 || !logits || !probs) return;
    float mx = logits[0];
    for (int i = 1; i < n; i++) mx = logits[i] > mx ? logits[i] : mx;
    double sum = 0;
    for (int i = 0; i < n; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
    const float inv = (float) (1.0 / (sum > 0 ? sum : 1.0));
    for (int i = 0; i < n; i++) probs[i] *= inv;
}

int jepa_top_k(const float * logits, int n, int k, int32_t * idx) {
    if (n <= 0 || k <= 0 || !logits || !idx) return 0;
    if (k > n) k = n;
    std::vector<int32_t> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int32_t a, int32_t b) { return logits[a] != logits[b] ? logits[a] > logits[b] : a < b; });
    for (int i = 0; i < k; i++) idx[i] = order[i];
    return k;
}

int64_t jepa_token_grid(const jepa_model * model, int n_frames, int height, int width, int * gt, int * gh, int * gw) {
    if (!model) return 0;
    switch (model->hp.family) {
        case JEPA_FAMILY_VJEPA:
        case JEPA_FAMILY_VJEPA2:
        case JEPA_FAMILY_VJEPA2_1: {
            jepa_video_shape vs;
            if (!jepa_video_shape_for(model, n_frames, height, width, vs, false)) return 0;
            if (gt) *gt = vs.gt;
            if (gh) *gh = vs.gh;
            if (gw) *gw = vs.gw;
            return vs.n_tokens;
        }
        default: {
            const jepa_enc_hparams & e = model->hp.enc;
            if (e.patch_size <= 0 || height % e.patch_size || width % e.patch_size) return 0;
            if (gt) *gt = n_frames;
            if (gh) *gh = height / e.patch_size;
            if (gw) *gw = width / e.patch_size;
            return (int64_t) n_frames * ((int64_t) (height / e.patch_size) * (width / e.patch_size)
                                         + jepa_model_n_prefix_tokens(model));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// jepa_encode: per-family dispatch
// ---------------------------------------------------------------------------------------------
int jepa_encode(jepa_context * ctx, const jepa_input * in, jepa_output * out) {
    if (!ctx || !in || !out || !in->data) return -1;
    const jepa_model * m = ctx->model;
    if (in->n_chans != m->hp.enc.in_chans) {
        jepa_log("jepa: input has %d channels, model expects %d\n", in->n_chans, m->hp.enc.in_chans);
        return -1;
    }
    if (in->n_batch < 1 || in->n_frames < 1 || in->height < 1 || in->width < 1) {
        jepa_log("jepa: bad input shape [%d, %d, %d, %d, %d]\n", in->n_batch, in->n_chans, in->n_frames, in->height, in->width);
        return -1;
    }
    out->data = nullptr; out->n_tokens = 0; out->dim = 0;
    switch (m->hp.family) {
        case JEPA_FAMILY_IJEPA:
        case JEPA_FAMILY_HFVIT:
        case JEPA_FAMILY_LEWM:
            // every (batch, frame) slice is encoded independently, rows concatenated in that order
            return jepa_encode_image(ctx, in, out);
        case JEPA_FAMILY_VJEPA2:
        case JEPA_FAMILY_VJEPA2_1:
            // one graph per clip: tubelet patchify + 3-D RoPE over the whole T x H x W token grid
            return jepa_encode_video(ctx, in, out);
        default:
            jepa_log("jepa: jepa_encode: family '%s' is not implemented\n", m->hp.family_str.c_str());
            return -1;
    }
}

// ---------------------------------------------------------------------------------------------
// pooling / projector
// ---------------------------------------------------------------------------------------------
int jepa_pool_mean(const jepa_model * model, const jepa_output * enc, jepa_output * out) {
    if (!model || !enc || !out || !enc->data) return -1;
    const int64_t skip = jepa_model_n_prefix_tokens(model);
    const int64_t D = enc->dim, n = enc->n_tokens - skip;
    if (n <= 0) return -1;
    out->n_tokens = 1;
    out->dim = D;
    out->data = (float *) calloc((size_t) D, sizeof(float));
    if (!out->data) return -1;
    std::vector<double> acc((size_t) D, 0.0);
    for (int64_t i = skip; i < enc->n_tokens; i++) {
        const float * row = enc->data + i * D;
        for (int64_t d = 0; d < D; d++) acc[d] += row[d];
    }
    for (int64_t d = 0; d < D; d++) out->data[d] = (float) (acc[d] / (double) n);
    return 0;
}

int jepa_pool_cls(const jepa_model * model, const jepa_output * enc, jepa_output * out) {
    if (!model || !enc || !out || !enc->data) return -1;
    if (!model->hp.enc.cls_token) {
        jepa_log("jepa: jepa_pool_cls: model has no CLS token\n");
        return -1;
    }
    out->n_tokens = 1;
    out->dim = enc->dim;
    out->data = (float *) malloc((size_t) enc->dim * sizeof(float));
    if (!out->data) return -1;
    memcpy(out->data, enc->data, (size_t) enc->dim * sizeof(float));
    return 0;
}

int jepa_lewm_project_rows(jepa_context * ctx, const float * cls_rows, int n_rows, jepa_output * out) {
    if (!ctx || !cls_rows || !out || n_rows <= 0) return -1;
    const jepa_model * m = ctx->model;
    if (!m->proj0_w || !m->proj2_w) {
        jepa_log("jepa: model has no enc.proj.* projector\n");
        return -1;
    }
    const int64_t D = m->hp.enc.embed_dim;
    jepa_graph_begin(ctx, 64);
    ggml_tensor * inp = ggml_new_tensor_2d(ctx->ctx_g, GGML_TYPE_F32, D, n_rows);
    ggml_set_input(inp);
    ggml_tensor * y = jepa_build_mlp2(ctx->ctx_g, inp, m->proj0_w, m->proj0_b, m->proj2_w, m->proj2_b, m->hp.enc.proj_act);
    ggml_set_output(y);
    std::vector<float> res;
    if (jepa_graph_run(ctx, inp, cls_rows, (size_t) D * n_rows * sizeof(float), y, res) != 0) return -1;
    out->n_tokens = n_rows;
    out->dim = y->ne[0];
    out->data = (float *) malloc(res.size() * sizeof(float));
    if (!out->data) return -1;
    memcpy(out->data, res.data(), res.size() * sizeof(float));
    return 0;
}

int jepa_lewm_project(jepa_context * ctx, const jepa_output * enc, jepa_output * out) {
    if (!ctx || !enc || !enc->data || enc->n_tokens < 1) return -1;
    if (!ctx->model->hp.enc.cls_token) {
        jepa_log("jepa: jepa_lewm_project: model has no CLS token\n");
        return -1;
    }
    return jepa_lewm_project_rows(ctx, enc->data, 1, out);
}

int jepa_head(jepa_context * ctx, const jepa_output * enc, jepa_output * out) {
    return jepa_head_ex(ctx, enc, nullptr, out);
}

int jepa_predict(jepa_context * ctx, const jepa_output * enc,
                 const int32_t * context_idx, int n_context,
                 const int32_t * target_idx,  int n_target, jepa_output * out) {
    (void) enc; (void) context_idx; (void) n_context; (void) target_idx; (void) n_target; (void) out;
    jepa_log("jepa: jepa_predict is not implemented for family '%s'\n", ctx ? ctx->model->hp.family_str.c_str() : "?");
    return -1;
}
