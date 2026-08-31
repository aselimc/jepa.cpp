// jepa.cpp — public C API (v0). See docs/architecture.md and docs/gguf-schema.md.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jepa_model   jepa_model;    // weights + hparams (immutable after load)
typedef struct jepa_context jepa_context;  // per-thread compute state (graph allocator, scratch)

enum { JEPA_KV_AUTO = 0, JEPA_KV_F16 = 1, JEPA_KV_F32 = 2 };

typedef struct {
    int   n_threads;      // 0 = hardware concurrency
    bool  use_flash_attn; // default true: ggml_flash_attn_ext; false: mul_mat + soft_max (debug)
    bool  verbose;
    int   flash_kv;       // K/V dtype inside flash attention: JEPA_KV_AUTO (default; F32 for f32
                          // models — the F16 cast costs ~3 digits of worst-token cosine on ViT-H —
                          // and F16 for f16/quantized models), JEPA_KV_F16, or JEPA_KV_F32
} jepa_context_params;

// Preprocessed input: NCTHW float32, already normalized. For images T == 1.
typedef struct {
    const float * data;   // size = n_batch * n_chans * n_frames * height * width
    int n_batch, n_chans, n_frames, height, width;
} jepa_input;

typedef struct {
    float * data;         // row-major [n_tokens, dim] (encoder) or [n_classes] (head)
    int64_t n_tokens;
    int64_t dim;
} jepa_output;

// --- lifecycle --------------------------------------------------------------
jepa_context_params jepa_context_default_params(void);
jepa_model   * jepa_model_load(const char * gguf_path, bool verbose);
void           jepa_model_free(jepa_model * model);
jepa_context * jepa_context_new(jepa_model * model, jepa_context_params params);
void           jepa_context_free(jepa_context * ctx);
int            jepa_context_n_threads(const jepa_context * ctx);
// Wall time (ms) of the ggml graph compute of the last jepa_encode / jepa_lewm_project call.
double         jepa_context_last_compute_ms(const jepa_context * ctx);

// --- introspection ------------------------------------------------------------
const char * jepa_model_family(const jepa_model * model);   // "ijepa", "vjepa2", ...
const char * jepa_model_name(const jepa_model * model);     // general.name
int          jepa_model_embed_dim(const jepa_model * model);
int          jepa_model_patch_size(const jepa_model * model);
int          jepa_model_tubelet_size(const jepa_model * model);
int          jepa_model_img_size(const jepa_model * model);
int          jepa_model_n_frames(const jepa_model * model);
int          jepa_model_n_layer(const jepa_model * model);
int          jepa_model_n_head(const jepa_model * model);
bool         jepa_model_has_cls(const jepa_model * model);
int          jepa_model_n_registers(const jepa_model * model);
int          jepa_model_n_prefix_tokens(const jepa_model * model);  // CLS + registers
bool         jepa_model_has_predictor(const jepa_model * model);
bool         jepa_model_has_head(const jepa_model * model);
bool         jepa_model_has_projector(const jepa_model * model);    // lewm enc.proj.*
int          jepa_model_n_classes(const jepa_model * model);
const char * jepa_model_label(const jepa_model * model, int idx);
int          jepa_model_file_type(const jepa_model * model);        // general.file_type (0 f32, 1 f16, 7 q8_0, ...)
const char * jepa_model_file_type_name(const jepa_model * model);   // "f32", "f16", "q8_0", ...
size_t       jepa_model_n_bytes(const jepa_model * model);          // weight bytes resident in memory

// --- preprocessing (host side, matches the reference processors) ---------------
enum { JEPA_RESAMPLE_BILINEAR = 0, JEPA_RESAMPLE_BICUBIC = 1 };
enum { JEPA_RESIZE_SHORTEST_EDGE = 0, JEPA_RESIZE_SQUASH = 1 };

// The jepa.pre.* pipeline, editable (test-parity builds it from the reference manifest).
typedef struct {
    float mean[3];
    float std[3];
    int   resize_short;   // shortest_edge: target short side; squash: the square output size
    int   crop;           // center crop after the resize (shortest_edge mode)
    int   resample;       // JEPA_RESAMPLE_*
    int   resize_mode;    // JEPA_RESIZE_*
    float rescale;        // pixel scale (1/255)
    bool  fused_norm;     // true (HF torchvision processors): (px - mean/rescale) / (std/rescale);
                          // false: (px*rescale - mean) / std   — differs by ~1 float32 ulp only
} jepa_preprocess_params;

jepa_preprocess_params jepa_preprocess_default_params(const jepa_model * model);

// Decodes an image file, resizes/crops/normalizes per the model's jepa.pre.* metadata.
// Returns malloc'd NCTHW floats (N=1, T=1); caller frees with jepa_free. out_h/out_w receive the crop size.
float * jepa_preprocess_image_file(const jepa_model * model, const char * path, int * out_h, int * out_w);
// Same for a raw RGB8 buffer (HWC).
float * jepa_preprocess_image_rgb(const jepa_model * model, const uint8_t * rgb, int h, int w, int * out_h, int * out_w);
// Frames: array of n_frames RGB8 HWC buffers of identical size → NCTHW (N=1).
float * jepa_preprocess_frames_rgb(const jepa_model * model, const uint8_t * const * frames, int n_frames, int h, int w, int * out_h, int * out_w);
// Explicit-parameter variants of the above.
float * jepa_preprocess_image_rgb_ex(const jepa_preprocess_params * p, const uint8_t * rgb, int h, int w, int * out_h, int * out_w);
float * jepa_preprocess_frames_rgb_ex(const jepa_preprocess_params * p, const uint8_t * const * frames, int n_frames, int h, int w, int * out_h, int * out_w);
// Decode an image file to RGB8 HWC (malloc'd, free with jepa_free). Returns NULL on failure.
uint8_t * jepa_load_image_rgb(const char * path, int * h, int * w);
// torchvision/PIL-style antialiased uint8 resize (separable, int16 fixed-point weights): HWC in → HWC out.
// out must hold out_h*out_w*c bytes.
void jepa_resize_antialias_u8(const uint8_t * src, int h, int w, int c, uint8_t * dst, int out_h, int out_w, int resample);
void    jepa_free(void * p);

// --- inference -------------------------------------------------------------------
// Encoder: returns [n_tokens, embed_dim] (final-norm applied). Caller frees out->data with jepa_free.
// Image families encode every (batch, frame) slice independently; rows are concatenated in that order.
int jepa_encode(jepa_context * ctx, const jepa_input * in, jepa_output * out);
// Mean over patch tokens (excluding CLS/registers) → [embed_dim]. (enc must hold exactly one item.)
int jepa_pool_mean(const jepa_model * model, const jepa_output * enc, jepa_output * out);
// CLS token (row 0, after the final norm) → [embed_dim]. Requires jepa_model_has_cls.
int jepa_pool_cls(const jepa_model * model, const jepa_output * enc, jepa_output * out);
// LeWM world-model state: emb = enc.proj.2(act(enc.proj.0(CLS))) → [embed_dim]. Requires jepa_model_has_projector.
int jepa_lewm_project(jepa_context * ctx, const jepa_output * enc, jepa_output * out);
// Same for n explicit CLS rows [n, embed_dim] → [n, embed_dim].
int jepa_lewm_project_rows(jepa_context * ctx, const float * cls_rows, int n_rows, jepa_output * out);
// Attentive-pool head → logits [n_classes]. Requires jepa_model_has_head.
int jepa_head(jepa_context * ctx, const jepa_output * enc, jepa_output * out);
// Masked predictor: predict encoder features at target_idx given context_idx (indices into token grid).
int jepa_predict(jepa_context * ctx, const jepa_output * enc,
                 const int32_t * context_idx, int n_context,
                 const int32_t * target_idx,  int n_target, jepa_output * out);

// ==============================================================================================
// Appended for the video encoders (V-JEPA 2 / V-JEPA 2.1) and the attentive-pool head.
// Everything above is unchanged; this section is append-only.
// ==============================================================================================

// Video input: pass the clip as one item — jepa_input NCTHW with n_frames = T (a multiple of the
// tubelet size; V-JEPA 2.1 also accepts T = 1, which selects its image tokenizer). jepa_encode()
// then returns [n_batch * (T/tubelet)*(H/patch)*(W/patch), embed_dim], token order t-major, h, w.

// Token grid this model would produce for one T x H x W item: t-major grid dimensions in
// gt/gh/gw (any may be NULL). Returns the token count per item, or 0 if the shape is not encodable.
int64_t jepa_token_grid(const jepa_model * model, int n_frames, int height, int width, int * gt, int * gh, int * gw);

// Attentive-pool head with both outputs: `pooled` (the pooler output = classifier input,
// [1, embed_dim]) and `logits` ([1, n_classes]); either may be NULL. jepa_head() is this with
// pooled = NULL. `enc` must hold the tokens of exactly one item. Caller frees the .data pointers.
int jepa_head_ex(jepa_context * ctx, const jepa_output * enc, jepa_output * pooled, jepa_output * logits);

// probs[i] = softmax(logits)[i] over n classes (host side; probs may not alias logits).
void jepa_softmax(const float * logits, int n, float * probs);
// Indices of the k largest logits, descending (ties by smaller index). Returns the number written.
int  jepa_top_k(const float * logits, int n, int k, int32_t * idx);

// --- misc ----------------------------------------------------------------------------
const char * jepa_version(void);
void jepa_print_system_info(void);

// ======================================================================================
// APPEND-ONLY: predictors & world model (src/predictor.cpp, src/lewm.cpp) — owned by the
// predictor agent. Keep additions inside this block.
// ======================================================================================

// Masked predictor (V-JEPA 2 / 2.1, jepa.pred.kind == "masked") with an explicit mask-token index.
// jepa_predict() above is this call with mask_index = 1 (the HF / Meta default; note that the
// released checkpoints have mask_tokens[1..] == 0, only index 0 carries signal).
//   enc          : the encoder output of the whole clip, [n_tokens, enc_dim] (jepa_encode)
//   context_idx  : token ids of the context (index both the rows of `enc` and the position grid);
//                  NULL means 0..n_context-1
//   target_idx   : token ids to predict (positions only, no encoder rows); NULL means 0..n_target-1
//   out          : [n_target, jepa.pred.out_dim] (enc_dim for V-JEPA 2), caller frees out->data
int jepa_predict_ex(jepa_context * ctx, const jepa_output * enc,
                    const int32_t * context_idx, int n_context,
                    const int32_t * target_idx,  int n_target,
                    int mask_index, jepa_output * out);

// Modality of a V-JEPA 2.1 predictor call: which of `pred.mod_embed_video` / `pred.mod_embed_img`
// is added to the context and mask tokens (VJEPA_NOTES.md §4.4, vjepa2_numpy_ref.py
// predictor_forward(mode=...)).  Ignored by families without `jepa.pred.modality_embed`
// (V-JEPA 2), where there is only one set of tokens.
//   AUTO  : image when the token ids span a single temporal slice (grid_t == 1) and the file has
//           an image modality vector, video otherwise;
//   VIDEO : always pred.mod_embed_video — what jepa_predict / jepa_predict_ex use, so their
//           behaviour is unchanged;
//   IMAGE : always pred.mod_embed_img (the 1x16x16 image tokenizer path of V-JEPA 2.1).  Measured on
//           the 576-token COCO image against the numpy spec's mode="image" (docs/parity.md): the
//           image vector gives cosine 1.0000000 on every row at f32, the video vector 0.862 mean /
//           0.655 worst -- a silent two-digit error, not noise.
// AUTO picks IMAGE when the ids span a single temporal slice and the file carries pred.mod_embed_img.
// A single-tubelet video clip (2 frames, tubelet 2) is indistinguishable from an image on the id grid,
// so pass JEPA_MODALITY_VIDEO explicitly for 2-frame clips — the wrong vector costs ~0.14 cosine.
enum { JEPA_MODALITY_AUTO = 0, JEPA_MODALITY_VIDEO = 1, JEPA_MODALITY_IMAGE = 2 };

// jepa_predict_ex with an explicit modality (jepa_predict_ex == modality JEPA_MODALITY_VIDEO).
int jepa_predict_mod(jepa_context * ctx, const jepa_output * enc,
                     const int32_t * context_idx, int n_context,
                     const int32_t * target_idx,  int n_target,
                     int mask_index, int modality, jepa_output * out);

// --- LeWM world model (jepa.pred.kind == "lewm") ---------------------------------------
int jepa_lewm_n_frames(const jepa_model * model);    // predictor context window (3)
int jepa_lewm_action_dim(const jepa_model * model);  // 10

// One predictor call over `n_frames` (<= jepa_lewm_n_frames) consecutive frames.
//   embs    : [n_frames, embed_dim] projected embeddings (jepa_lewm_project)
//   actions : [n_frames, action_dim] — one action per frame (for n_frames == 1 simply the action)
//   out     : [n_frames, embed_dim]; the attention is causal over frames, so row t is the predicted
//             next embedding given frames 0..t and the LAST row is the next-frame prediction.
int jepa_lewm_predict(jepa_context * ctx, const float * embs, const float * actions,
                      int n_frames, jepa_output * out);

// Autoregressive rollout: predictions are fed back as frames, the window is clipped to n_frames.
//   embs    : [n_seed, embed_dim] seed embeddings (n_seed >= 1)
//   actions : [n_steps, action_dim]; action k drives step k. Frame j of the growing sequence uses
//             actions[j - (n_seed-1)] (clamped), so with n_seed == 1 frame j uses actions[j].
//   out     : [n_steps, embed_dim], caller-allocated.
int jepa_lewm_rollout(jepa_context * ctx, const float * embs, int n_seed,
                      const float * actions, int n_steps, float * out);

// ======================================================================================
// APPEND-ONLY: encoder batching (src/jepa.cpp).
// ======================================================================================

// How many image items (`n_batch * n_frames` slices of one jepa_input) the encoder folds into ONE
// ggml graph. The items stay independent — they live on the graph's batch dimension, so attention
// never mixes them — and the output rows are the same, in the same (batch, frame) order, as the
// one-graph-per-item path; f32 is bit-identical (tests/test-batch.cpp gates this).
// Only the image families (ijepa / hfvit / lewm) batch; V-JEPA 2 / 2.1 still run one clip per graph.
//   n <= 0 : restore the default (32, or $JEPA_MAX_BATCH when the context was created)
//   n == 1 : the old per-item path — the debug switch (`jepa-embed --no-batch`, `JEPA_MAX_BATCH=1`)
// Inputs larger than n are encoded in ceil(n_items / n) graphs, so this caps memory, not batch size.
void jepa_context_set_max_batch(jepa_context * ctx, int n);
int  jepa_context_max_batch(const jepa_context * ctx);
// Items in the last graph the encoder built (1 unless batching kicked in) — for tools that report
// per-item timings from jepa_context_last_compute_ms().
int  jepa_context_last_batch(const jepa_context * ctx);

#ifdef __cplusplus
}
#endif
