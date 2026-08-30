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

typedef struct {
    int   n_threads;      // 0 = hardware concurrency
    bool  use_flash_attn; // default true
    bool  verbose;
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

// --- introspection ------------------------------------------------------------
const char * jepa_model_family(const jepa_model * model);   // "ijepa", "vjepa2", ...
int          jepa_model_embed_dim(const jepa_model * model);
int          jepa_model_patch_size(const jepa_model * model);
int          jepa_model_tubelet_size(const jepa_model * model);
int          jepa_model_img_size(const jepa_model * model);
int          jepa_model_n_frames(const jepa_model * model);
bool         jepa_model_has_predictor(const jepa_model * model);
bool         jepa_model_has_head(const jepa_model * model);
int          jepa_model_n_classes(const jepa_model * model);
const char * jepa_model_label(const jepa_model * model, int idx);

// --- preprocessing (host side, matches the reference processors) ---------------
// Decodes an image file, resizes/crops/normalizes per the model's jepa.pre.* metadata.
// Returns malloc'd NCTHW floats (T=1); caller frees with jepa_free.
float * jepa_preprocess_image_file(const jepa_model * model, const char * path, int * out_h, int * out_w);
// Same for a raw RGB8 buffer (HWC).
float * jepa_preprocess_image_rgb(const jepa_model * model, const uint8_t * rgb, int h, int w, int * out_h, int * out_w);
// Frames: array of n_frames RGB8 HWC buffers of identical size → NCTHW.
float * jepa_preprocess_frames_rgb(const jepa_model * model, const uint8_t * const * frames, int n_frames, int h, int w, int * out_h, int * out_w);
void    jepa_free(void * p);

// --- inference -------------------------------------------------------------------
// Encoder: returns [n_tokens, embed_dim] (final-norm applied). Caller frees out->data with jepa_free.
int jepa_encode(jepa_context * ctx, const jepa_input * in, jepa_output * out);
// Mean over patch tokens (excluding CLS/registers) → [embed_dim].
int jepa_pool_mean(const jepa_model * model, const jepa_output * enc, jepa_output * out);
// Attentive-pool head → logits [n_classes]. Requires jepa_model_has_head.
int jepa_head(jepa_context * ctx, const jepa_output * enc, jepa_output * out);
// Masked predictor: predict encoder features at target_idx given context_idx (indices into token grid).
int jepa_predict(jepa_context * ctx, const jepa_output * enc,
                 const int32_t * context_idx, int n_context,
                 const int32_t * target_idx,  int n_target, jepa_output * out);

// --- misc ----------------------------------------------------------------------------
const char * jepa_version(void);
void jepa_print_system_info(void);

#ifdef __cplusplus
}
#endif
