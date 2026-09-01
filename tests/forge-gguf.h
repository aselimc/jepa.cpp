// forge-gguf.h — build a tiny but genuinely loadable jepa GGUF in a temp file, with one knob per
// thing that can be wrong with it. Shared by tests/test-errors.cpp (which breaks one knob per case)
// and tests/test-threads.cpp (which needs a model on a runner that has none), so both suites run at
// full strength with no weights and can therefore run in the ASAN+UBSAN CI job.
#pragma once
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// --------------------------------------------------------------------------------------------
// a minimal but genuinely loadable hfvit GGUF, and one knob per case to break it with
// --------------------------------------------------------------------------------------------
// D=8, 2 heads, ffn 16, 1 block, patch 2 on an 8x8 image -> 16 patch tokens + CLS. Small enough to
// build in memory, complete enough that the loader accepts it: every case below is that file with
// exactly one thing wrong, so what the case proves is that one check and nothing else.
struct forge_opts {
    int  embed_dim = 8, n_head = 2, ffn_dim = 16, n_layer = 1, patch = 2, img = 8;
    const char * family = "hfvit";
    const char * arch = "jepa";
    const char * act = "gelu_erf";
    const char * attn_mode = nullptr;   // absent = "full"
    const char * pred_kind = nullptr;
    const char * head_kind = nullptr;
    // A complete, loadable V-JEPA 2-AC predictor (jepa.pred.kind = "ac", metadata AND tensors),
    // so the AC-specific loader checks and the jepa_ac_* argument guards can be exercised without
    // the 5 GB released file. Each knob below breaks exactly one of those invariants.
    bool ac_pred = false;
    int  ac_pred_dim = 8, ac_pred_heads = 2, ac_pred_layers = 1, ac_pred_ffn = 16;
    int  ac_action_dim = 4, ac_state_dim = 4, ac_cond_tokens = 2, ac_grid = 2, ac_frames = 5;
    bool ac_drop_tensors = false;      // metadata promises a predictor the file does not carry
    bool ac_bad_action_shape = false;  // pred.action_embed.weight with the wrong input width
    int  n_registers = 0;
    int  extra_patch_in = 0;            // +N on enc.patch_embed.weight ne[0]: a shape mismatch
    int  ln1_len = 0;                   // 0 = embed_dim; anything else is a norm weight that cannot broadcast
    int  qkv_bias_len = 0;              // 0 = 3*embed_dim
    bool add_layer_scale = false;       // ls1/ls2 at the wrong length
    bool drop_ln1 = false;              // an incomplete block
    bool drop_cls_tensor = false;       // cls_token = true with no enc.cls_token
    bool quantize_patch_embed = false;  // a tensor type whose block size does not divide the row
    bool f16_cls_token = false;         // an operand the graph concatenates with f32 activations
    bool i32_ffn_up = false;            // a matmul weight of an integer type: no CPU vec_dot at all
    float ln_eps = 1e-6f;
    float pre_std0 = 0.229f;
    int  pre_resize_short = 8, pre_crop = 8;
};

static ggml_tensor * add_t(ggml_context * ctx, gguf_context * gg, const char * name,
                           int64_t ne0, int64_t ne1, ggml_type type = GGML_TYPE_F32) {
    ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1) : ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_name(t, name);
    // deterministic, small and non-degenerate: a zero weight matrix would make every norm 0/0
    float * d = (float *) t->data;
    if (type == GGML_TYPE_F32) {
        const size_t n = (size_t) ggml_nelements(t);
        for (size_t i = 0; i < n; i++) d[i] = 0.05f * (float) ((i * 37 % 41) - 20);
    } else {
        memset(t->data, 0, ggml_nbytes(t));
    }
    gguf_add_tensor(gg, t);
    return t;
}

static void forge(const std::string & path, const forge_opts & o) {
    gguf_context * gg = gguf_init_empty();
    // 8 MiB of tensor arena is far more than this model needs; ggml_init with no_alloc=false
    // allocates it once and every add_t carves a tensor out of it.
    ggml_init_params ip = { 8u << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    gguf_set_val_str(gg, "general.architecture", o.arch);
    gguf_set_val_str(gg, "general.name", "forged");
    gguf_set_val_u32(gg, "general.file_type", 0);
    gguf_set_val_u32(gg, "jepa.schema_version", 1);
    gguf_set_val_str(gg, "jepa.family", o.family);
    gguf_set_val_str(gg, "jepa.modality", "image");
    gguf_set_val_i32(gg, "jepa.enc.embed_dim", o.embed_dim);
    gguf_set_val_i32(gg, "jepa.enc.n_layer", o.n_layer);
    gguf_set_val_i32(gg, "jepa.enc.n_head", o.n_head);
    gguf_set_val_i32(gg, "jepa.enc.ffn_dim", o.ffn_dim);
    gguf_set_val_i32(gg, "jepa.enc.patch_size", o.patch);
    gguf_set_val_i32(gg, "jepa.enc.img_size", o.img);
    gguf_set_val_i32(gg, "jepa.enc.in_chans", 3);
    gguf_set_val_i32(gg, "jepa.enc.tubelet_size", 1);
    gguf_set_val_i32(gg, "jepa.enc.n_frames", 1);
    gguf_set_val_i32(gg, "jepa.enc.n_registers", o.n_registers);
    gguf_set_val_bool(gg, "jepa.enc.cls_token", true);
    gguf_set_val_bool(gg, "jepa.enc.qkv_fused", true);
    gguf_set_val_str(gg, "jepa.enc.act", o.act);
    gguf_set_val_str(gg, "jepa.enc.pos_type", "learned");
    gguf_set_val_f32(gg, "jepa.enc.ln_eps", o.ln_eps);
    if (o.attn_mode) gguf_set_val_str(gg, "jepa.enc.attn_mode", o.attn_mode);
    if (o.pred_kind) gguf_set_val_str(gg, "jepa.pred.kind", o.pred_kind);
    if (o.ac_pred) {
        gguf_set_val_str(gg, "jepa.pred.kind", "ac");
        gguf_set_val_i32(gg, "jepa.pred.embed_dim", o.ac_pred_dim);
        gguf_set_val_i32(gg, "jepa.pred.n_layer", o.ac_pred_layers);
        gguf_set_val_i32(gg, "jepa.pred.n_head", o.ac_pred_heads);
        gguf_set_val_i32(gg, "jepa.pred.ffn_dim", o.ac_pred_ffn);
        gguf_set_val_i32(gg, "jepa.pred.out_dim", o.embed_dim);
        gguf_set_val_i32(gg, "jepa.pred.action_dim", o.ac_action_dim);
        gguf_set_val_i32(gg, "jepa.pred.state_dim", o.ac_state_dim);
        gguf_set_val_i32(gg, "jepa.pred.n_cond_tokens", o.ac_cond_tokens);
        gguf_set_val_str(gg, "jepa.pred.cond_order", "action,state");
        gguf_set_val_i32(gg, "jepa.pred.grid_size", o.ac_grid);
        gguf_set_val_i32(gg, "jepa.pred.n_frames", o.ac_frames);
        gguf_set_val_bool(gg, "jepa.pred.frame_causal", true);
        gguf_set_val_str(gg, "jepa.pred.rope_freq_layout", "tiled");
        gguf_set_val_f32(gg, "jepa.pred.ln_eps", 1e-6f);
        gguf_set_val_bool(gg, "jepa.pred.normalize_reps", true);
        gguf_set_val_f32(gg, "jepa.pred.norm_reps_eps", 1e-5f);
    }
    if (o.head_kind) gguf_set_val_str(gg, "jepa.head.kind", o.head_kind);

    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float stdv[3] = { o.pre_std0, 0.224f, 0.225f };
    gguf_set_arr_data(gg, "jepa.pre.mean", GGUF_TYPE_FLOAT32, mean, 3);
    gguf_set_arr_data(gg, "jepa.pre.std", GGUF_TYPE_FLOAT32, stdv, 3);
    gguf_set_val_i32(gg, "jepa.pre.resize_short", o.pre_resize_short);
    gguf_set_val_i32(gg, "jepa.pre.crop", o.pre_crop);
    gguf_set_val_str(gg, "jepa.pre.resample", "bicubic");
    gguf_set_val_str(gg, "jepa.pre.resize_mode", "shortest_edge");
    gguf_set_val_f32(gg, "jepa.pre.rescale", 1.0f / 255.0f);

    const int64_t D = o.embed_dim, F = o.ffn_dim;
    const int64_t patch_in = 3 * (int64_t) o.patch * o.patch + o.extra_patch_in;
    const int64_t grid = o.img / o.patch;
    const int64_t n_pos = grid * grid + 1 + o.n_registers;

    add_t(ctx, gg, "enc.patch_embed.weight", patch_in, D,
          o.quantize_patch_embed ? GGML_TYPE_Q4_0 : GGML_TYPE_F32);
    add_t(ctx, gg, "enc.patch_embed.bias", D, 0);
    if (!o.drop_cls_tensor) add_t(ctx, gg, "enc.cls_token", D, 0, o.f16_cls_token ? GGML_TYPE_F16 : GGML_TYPE_F32);
    add_t(ctx, gg, "enc.pos_embed", D, n_pos);
    if (o.n_registers > 0) add_t(ctx, gg, "enc.reg_tokens", D, o.n_registers);
    for (int i = 0; i < o.n_layer; i++) {
        const std::string p = "enc.blk." + std::to_string(i) + ".";
        if (!o.drop_ln1) {
            add_t(ctx, gg, (p + "ln1.weight").c_str(), o.ln1_len > 0 ? o.ln1_len : D, 0);
            add_t(ctx, gg, (p + "ln1.bias").c_str(), D, 0);
        }
        if (o.add_layer_scale) {
            add_t(ctx, gg, (p + "ls1").c_str(), D + 1, 0);
            add_t(ctx, gg, (p + "ls2").c_str(), D, 0);
        }
        add_t(ctx, gg, (p + "attn_qkv.weight").c_str(), D, 3 * D);
        add_t(ctx, gg, (p + "attn_qkv.bias").c_str(), o.qkv_bias_len > 0 ? o.qkv_bias_len : 3 * D, 0);
        add_t(ctx, gg, (p + "attn_out.weight").c_str(), D, D);
        add_t(ctx, gg, (p + "attn_out.bias").c_str(), D, 0);
        add_t(ctx, gg, (p + "ln2.weight").c_str(), D, 0);
        add_t(ctx, gg, (p + "ln2.bias").c_str(), D, 0);
        add_t(ctx, gg, (p + "ffn_up.weight").c_str(), D, F, o.i32_ffn_up ? GGML_TYPE_I32 : GGML_TYPE_F32);
        add_t(ctx, gg, (p + "ffn_up.bias").c_str(), F, 0);
        add_t(ctx, gg, (p + "ffn_down.weight").c_str(), F, D);
        add_t(ctx, gg, (p + "ffn_down.bias").c_str(), D, 0);
    }
    add_t(ctx, gg, "enc.norm.weight", D, 0);
    add_t(ctx, gg, "enc.norm.bias", D, 0);

    if (o.ac_pred && !o.ac_drop_tensors) {
        const int64_t Dp = o.ac_pred_dim, Fp = o.ac_pred_ffn;
        const int64_t inner = (int64_t) o.ac_pred_heads * (Dp / o.ac_pred_heads);
        add_t(ctx, gg, "pred.embed.weight", D, Dp);
        add_t(ctx, gg, "pred.embed.bias", Dp, 0);
        // A zero action/state width is a metadata-only case (the loader has to refuse it before any
        // tensor is looked at); ggml cannot hold a zero-row tensor, so skip those two here.
        if (o.ac_action_dim > 0) {
            add_t(ctx, gg, "pred.action_embed.weight", o.ac_bad_action_shape ? o.ac_action_dim + 1 : o.ac_action_dim, Dp);
            add_t(ctx, gg, "pred.action_embed.bias", Dp, 0);
        }
        if (o.ac_state_dim > 0) {
            add_t(ctx, gg, "pred.state_embed.weight", o.ac_state_dim, Dp);
            add_t(ctx, gg, "pred.state_embed.bias", Dp, 0);
        }
        for (int i = 0; i < o.ac_pred_layers; i++) {
            const std::string q = "pred.blk." + std::to_string(i) + ".";
            add_t(ctx, gg, (q + "ln1.weight").c_str(), Dp, 0);
            add_t(ctx, gg, (q + "ln1.bias").c_str(), Dp, 0);
            add_t(ctx, gg, (q + "attn_qkv.weight").c_str(), Dp, 3 * inner);
            add_t(ctx, gg, (q + "attn_qkv.bias").c_str(), 3 * inner, 0);
            add_t(ctx, gg, (q + "attn_out.weight").c_str(), inner, Dp);
            add_t(ctx, gg, (q + "attn_out.bias").c_str(), Dp, 0);
            add_t(ctx, gg, (q + "ln2.weight").c_str(), Dp, 0);
            add_t(ctx, gg, (q + "ln2.bias").c_str(), Dp, 0);
            add_t(ctx, gg, (q + "ffn_up.weight").c_str(), Dp, Fp);
            add_t(ctx, gg, (q + "ffn_up.bias").c_str(), Fp, 0);
            add_t(ctx, gg, (q + "ffn_down.weight").c_str(), Fp, Dp);
            add_t(ctx, gg, (q + "ffn_down.bias").c_str(), Dp, 0);
        }
        add_t(ctx, gg, "pred.norm.weight", Dp, 0);
        add_t(ctx, gg, "pred.norm.bias", Dp, 0);
        add_t(ctx, gg, "pred.proj.weight", Dp, D);
        add_t(ctx, gg, "pred.proj.bias", D, 0);
    }

    if (!gguf_write_to_file(gg, path.c_str(), false)) {
        fprintf(stderr, "cannot forge %s\n", path.c_str());
        exit(2);
    }
    gguf_free(gg);
    ggml_free(ctx);
}
