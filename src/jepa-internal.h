// Internal shared declarations for jepa.cpp. Public API lives in include/jepa.h.
//
// Layout of this header
//   1. hparams structs      : one-to-one with the jepa.* keys of docs/gguf-schema.md
//   2. model struct         : all GGUF tensors on the CPU backend + a name -> tensor map,
//                             per-block tensor bundles for enc.blk.* / pred.blk.* / head.blk.*
//   3. context struct       : backend, graph allocator, thread count
//   4. graph-building helpers: layer norm, linear, activation, attention (flash / naive),
//                             a full pre-LN transformer block, small MLPs — reusable by the
//                             encoder (here), the video encoder and the predictors/heads.
//   5. graph execution     : jepa_graph_begin / jepa_graph_compute
//   6. host-side helpers    : patchify, preprocessing params
#pragma once
#include "jepa.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#define JEPA_VERSION "0.1.0-dev"

// Log helper (stderr).
void jepa_log(const char * fmt, ...);

// ---------------------------------------------------------------------------------------------
// 1. hparams (docs/gguf-schema.md)
// ---------------------------------------------------------------------------------------------
enum jepa_family_id {
    JEPA_FAMILY_UNKNOWN = 0,
    JEPA_FAMILY_IJEPA,     // 2-D sincos table, no CLS
    JEPA_FAMILY_VJEPA,     // 3-D sincos table
    JEPA_FAMILY_VJEPA2,    // rope3d (tiled)
    JEPA_FAMILY_VJEPA2_1,  // rope3d (interleaved) + image tokenizer + modality vectors
    JEPA_FAMILY_HFVIT,     // DINOv2-style: CLS + learned pos (LeJEPA)
    JEPA_FAMILY_LEWM,      // hfvit encoder + projector MLP + adaLN predictor
};

enum jepa_act_id {
    JEPA_ACT_GELU_ERF = 0,
    JEPA_ACT_GELU_TANH,
    JEPA_ACT_SILU,
};

enum jepa_pos_id {
    JEPA_POS_NONE = 0,
    JEPA_POS_SINCOS2D,
    JEPA_POS_SINCOS3D,
    JEPA_POS_LEARNED,
    JEPA_POS_ROPE3D,
};

jepa_family_id jepa_family_from_string(const std::string & s);
jepa_act_id    jepa_act_from_string(const std::string & s);      // aborts on unknown
jepa_pos_id    jepa_pos_from_string(const std::string & s);
const char *   jepa_act_name(jepa_act_id a);

struct jepa_enc_hparams {
    int   embed_dim    = 0;
    int   n_layer      = 0;
    int   n_head       = 0;
    int   ffn_dim      = 0;
    int   patch_size   = 16;
    int   tubelet_size = 1;
    int   img_size     = 224;
    int   n_frames     = 1;
    int   in_chans     = 3;
    float ln_eps       = 1e-6f;
    jepa_act_id act    = JEPA_ACT_GELU_ERF;
    jepa_pos_id pos_type = JEPA_POS_NONE;
    std::string pos_type_str;
    // rope3d (V-JEPA 2 / 2.1)
    float rope_theta        = 10000.0f;
    bool  rope_interpolate  = false;
    std::string rope_freq_layout;     // "tiled" | "interleaved" | ""
    int   rope_ref_grid     = 0;
    // tokens
    bool  cls_token   = false;
    int   n_registers = 0;
    bool  qkv_fused   = true;
    bool  modality_embed    = false;
    bool  image_patch_embed = false;
    std::vector<int> hier_layers;
    bool  layer_scale = false;
    // lewm projector
    bool        has_proj = false;
    jepa_act_id proj_act = JEPA_ACT_GELU_ERF;

    int head_dim() const { return n_head > 0 ? embed_dim / n_head : 0; }
    int grid_size() const { return patch_size > 0 ? img_size / patch_size : 0; }
};

struct jepa_pred_hparams {
    bool present = false;
    std::string kind;                 // masked | ac | lewm
    int   embed_dim = 0, n_layer = 0, n_head = 0, ffn_dim = 0;
    int   n_mask_tokens = 0;
    int   out_dim = 0;
    std::string rope_freq_layout;
    bool  rope_interpolate = false;
    int   rope_ref_grid = 0;
    int   grid_size = 0;
    int   n_hier_in = 1;
    bool  modality_embed = false;
    bool  context_proj = false;
    int   action_dim = 0, state_dim = 0;
    bool  frame_causal = false;
    int   n_frames = 0;
    int   head_dim = 0;               // 0 -> embed_dim / n_head
    float ln_eps = 1e-6f;
    float adaln_eps = 1e-6f;
    jepa_act_id act = JEPA_ACT_GELU_ERF;
    bool  qkv_bias = true;
    jepa_act_id action_act = JEPA_ACT_SILU;
    jepa_act_id proj_act = JEPA_ACT_GELU_ERF;

    int head_dim_eff() const { return head_dim > 0 ? head_dim : (n_head > 0 ? embed_dim / n_head : 0); }
};

struct jepa_head_hparams {
    bool present = false;
    std::string kind;                 // attentive_pool | linear_cls | none
    int n_classes = 0;
    int n_pool_layers = 0;
    std::vector<std::string> labels;
};

// jepa.pre.* — how the reference processor turned pixels into the model input.
struct jepa_pre_hparams {
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std [3] = {0.229f, 0.224f, 0.225f};
    int   resize_short = 224;
    int   crop = 224;
    std::string resample = "bilinear";       // bilinear | bicubic
    std::string resize_mode = "shortest_edge"; // shortest_edge | squash
    float rescale = 1.0f / 255.0f;
};

struct jepa_hparams {
    std::string arch, name, license, source_url, description;
    uint32_t file_type = 0;          // general.file_type (0 f32, 1 f16, 7 q8_0, ...)
    uint32_t schema_version = 0;
    std::string family_str, modality;
    jepa_family_id family = JEPA_FAMILY_UNKNOWN;
    jepa_enc_hparams  enc;
    jepa_pred_hparams pred;
    jepa_head_hparams head;
    jepa_pre_hparams  pre;
    // every general.* / jepa.* key rendered as text, in file order (jepa-info)
    std::vector<std::pair<std::string, std::string>> raw_kv;
};

// ---------------------------------------------------------------------------------------------
// 2. model
// ---------------------------------------------------------------------------------------------
// Tensors of one transformer block (enc.blk.{i} / pred.blk.{i} / head.blk.{i}). Missing tensors
// are nullptr (e.g. no qkv bias, no layer-scale, separate q/k/v instead of fused).
struct jepa_layer {
    ggml_tensor * ln1_w = nullptr, * ln1_b = nullptr;
    ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr;                 // fused [3D, D]
    ggml_tensor * q_w = nullptr, * q_b = nullptr;                     // or separate
    ggml_tensor * k_w = nullptr, * k_b = nullptr;
    ggml_tensor * v_w = nullptr, * v_b = nullptr;
    ggml_tensor * out_w = nullptr, * out_b = nullptr;
    ggml_tensor * ln2_w = nullptr, * ln2_b = nullptr;
    ggml_tensor * up_w = nullptr, * up_b = nullptr;
    ggml_tensor * down_w = nullptr, * down_b = nullptr;
    ggml_tensor * ls1 = nullptr, * ls2 = nullptr;                     // layer-scale
    ggml_tensor * adaln_w = nullptr, * adaln_b = nullptr;             // lewm predictor
};

struct jepa_model {
    std::string   path;
    jepa_hparams  hp;

    ggml_backend_t        backend = nullptr;   // CPU backend used for weight storage
    ggml_context        * ctx_w   = nullptr;   // holds the tensor metadata (no_alloc)
    ggml_backend_buffer_t buf_w   = nullptr;   // the weight bytes
    std::map<std::string, ggml_tensor *> tensors;   // every GGUF tensor by name
    size_t n_bytes_weights = 0;

    // encoder
    ggml_tensor * patch_embed_w = nullptr, * patch_embed_b = nullptr;
    ggml_tensor * patch_embed_img_w = nullptr, * patch_embed_img_b = nullptr;  // 2.1
    ggml_tensor * pos_embed = nullptr;         // [D, n_tokens] (ggml order)
    ggml_tensor * cls_token = nullptr;         // [D]
    ggml_tensor * reg_tokens = nullptr;        // [D, n_registers]
    ggml_tensor * mod_embed_img = nullptr, * mod_embed_video = nullptr;
    ggml_tensor * norm_w = nullptr, * norm_b = nullptr;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> hier_norms;  // 2.1 (w, b)
    ggml_tensor * proj0_w = nullptr, * proj0_b = nullptr;   // lewm enc.proj.0
    ggml_tensor * proj2_w = nullptr, * proj2_b = nullptr;   // lewm enc.proj.2
    std::vector<jepa_layer> enc_layers;
    // predictor (generic bundle; the predictor agents interpret it)
    std::vector<jepa_layer> pred_layers;
    // head
    std::vector<jepa_layer> head_layers;

    ggml_tensor * get(const std::string & name) const;      // nullptr if absent
    ggml_tensor * require(const std::string & name) const;  // logs + aborts if absent
    bool has(const std::string & name) const { return tensors.count(name) != 0; }
};

// Parse every general.* / jepa.* key of an opened GGUF into hp. Returns false on a hard error
// (missing mandatory key, unknown family). Implemented in jepa-gguf.cpp.
bool jepa_hparams_from_gguf(const gguf_context * gg, jepa_hparams & hp);
// Fill the per-block tensor bundle for `prefix` (e.g. "enc.blk.3.").
jepa_layer jepa_layer_from_model(const jepa_model & m, const std::string & prefix);
// Human-readable dtype name for a file_type value (f32 / f16 / q8_0 / ...).
const char * jepa_file_type_name(uint32_t ftype);

// ---------------------------------------------------------------------------------------------
// 3. context
// ---------------------------------------------------------------------------------------------
struct jepa_context {
    jepa_model *        model   = nullptr;
    jepa_context_params params;
    int                 n_threads = 1;
    ggml_backend_t      backend = nullptr;   // CPU backend (compute)
    ggml_gallocr_t      galloc  = nullptr;   // graph allocator, reused across runs
    std::vector<uint8_t> graph_meta;         // memory for the graph ggml_context
    ggml_context *      ctx_g   = nullptr;   // current graph context (rebuilt per run)
    ggml_cgraph *       gf      = nullptr;   // current graph
    size_t              graph_max_nodes = 0;
    double              last_compute_ms = 0; // wall time of the last ggml_backend_graph_compute
};

// ---------------------------------------------------------------------------------------------
// 4. graph builders (all tensors in ggml order: ne[0] = feature dim)
// ---------------------------------------------------------------------------------------------
// Options shared by the attention / block builders.
struct jepa_attn_opts {
    bool      flash    = true;             // ggml_flash_attn_ext vs mul_mat + soft_max_ext
    ggml_type kv_type  = GGML_TYPE_F16;    // dtype K/V are cast to for flash attention (F16 or F32)
    ggml_tensor * mask = nullptr;          // optional additive mask [n_kv, n_q(_pad)] F16 (flash) / F32 (naive)
    float     scale    = 0.0f;             // 0 -> 1/sqrt(head_dim)
};

// Hook applied to the q and k views ([head_dim, n_head, N], row-contiguous) before attention.
// Used by the video encoder for 3-D RoPE. Must return a tensor of the same shape.
using jepa_qk_hook = std::function<ggml_tensor * (ggml_context * ctx, ggml_tensor * x, bool is_k)>;

struct jepa_block_opts {
    float       ln_eps  = 1e-6f;
    jepa_act_id act     = JEPA_ACT_GELU_ERF;
    int         n_head  = 0;
    int         head_dim = 0;              // 0 -> embed_dim / n_head
    jepa_attn_opts attn;
    jepa_qk_hook   qk_hook;               // optional
};

// y = norm(x, eps) * w + b   (w / b may be nullptr)
ggml_tensor * jepa_build_layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps);
// y = w x + b   with w [in, out] (PyTorch [out, in]); b may be nullptr
ggml_tensor * jepa_build_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * jepa_build_act(ggml_context * ctx, ggml_tensor * x, jepa_act_id act);
// 2-layer MLP: w2(act(w1 x + b1)) + b2
ggml_tensor * jepa_build_mlp2(ggml_context * ctx, ggml_tensor * x,
                              ggml_tensor * w1, ggml_tensor * b1, ggml_tensor * w2, ggml_tensor * b2, jepa_act_id act);
// Multi-head attention. q: [head_dim, n_head, N_q], k/v: [head_dim, n_head, N_kv] (row-contiguous
// views are fine). Returns [n_head*head_dim, N_q] contiguous F32.
ggml_tensor * jepa_build_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                   const jepa_attn_opts & opts);
// q/k/v projection of a block: returns the three [head_dim, n_head, N] views (fused or separate).
void jepa_build_qkv(ggml_context * ctx, ggml_tensor * x, const jepa_layer & L, int n_head, int head_dim,
                    ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v);
// Pre-LN transformer block: x += ls1*attn(ln1(x)); x += ls2*ffn(ln2(x)). x: [D, N] F32.
ggml_tensor * jepa_build_block(ggml_context * ctx, ggml_tensor * x, const jepa_layer & L, const jepa_block_opts & opts);

// ---------------------------------------------------------------------------------------------
// 5. graph execution
// ---------------------------------------------------------------------------------------------
// Start a new graph: (re)creates ctx->ctx_g (no_alloc) and ctx->gf sized for max_nodes nodes.
void jepa_graph_begin(jepa_context * ctx, size_t max_nodes);
// Allocate the graph's tensors with the context's allocator (call after ggml_build_forward_expand;
// input tensors can be filled with ggml_backend_tensor_set afterwards).
bool jepa_graph_alloc(jepa_context * ctx);
// Run the current graph on the CPU backend with ctx->n_threads. Returns 0 on success.
int  jepa_graph_compute(jepa_context * ctx);
// K/V dtype flash attention should use for this context (resolves JEPA_KV_AUTO by file_type).
ggml_type jepa_context_kv_type(const jepa_context * ctx);
// Convenience: alloc + set one input + compute + fetch one output.
// Returns 0 on success; out is resized to ggml_nelements(out_t) floats.
int  jepa_graph_run(jepa_context * ctx, ggml_tensor * inp, const float * inp_data, size_t inp_bytes,
                    ggml_tensor * out_t, std::vector<float> & out);

// ---------------------------------------------------------------------------------------------
// 6. host-side helpers
// ---------------------------------------------------------------------------------------------
// Rearrange one normalized CTHW image/clip (C-major) into patch rows [N, C*T_p*P*P]
// (token order t-major, then h, then w; inner order c, t, py, px). Returns N = gt*gh*gw and the grid.
// H and W must be multiples of P and T of tubelet (extra rows/cols are dropped like a conv would).
int64_t jepa_patchify(const float * cthw, int C, int T, int H, int W, int patch, int tubelet,
                      std::vector<float> & rows, int * gt, int * gh, int * gw);

// Build the encoder graph for one image/clip of `n_patches` patch rows already uploaded to `inp`
// ([C*T*P*P, n_patches] F32). Returns the [D, n_tokens] output tensor (after enc.norm).
// Only the ijepa / hfvit / lewm image families are handled here; video builders live elsewhere.
ggml_tensor * jepa_build_encoder_image(jepa_context * ctx, ggml_tensor * inp, int gh, int gw);
