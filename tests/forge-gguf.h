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

    if (!gguf_write_to_file(gg, path.c_str(), false)) {
        fprintf(stderr, "cannot forge %s\n", path.c_str());
        exit(2);
    }
    gguf_free(gg);
    ggml_free(ctx);
}
