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
// Video families encode in->n_frames frames as ONE clip, exactly as given: a shorter clip is a
// shorter clip, not an error, so in->n_frames = 1 through a video model is a one-frame clip and not
// the still-image path its model card may describe. LeVJEPA's card, for instance, feeds a still
// image as the frame repeated to jepa.enc.n_frames; jepa-embed does that repeat for the caller,
// this entry point does not.
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

// --- V-JEPA 2-AC action-conditioned world model (jepa.pred.kind == "ac") ---------------
// The predictor of facebookresearch/vjepa2's `vjepa2_ac_vit_giant`: it takes the encoder latents
// of T context frames plus a 7-d end-effector action and a 7-d state (pose) per frame, and
// predicts the encoder latents of the NEXT frame. Attention is block-causal over frames, so one
// call with T frames also yields the prediction after every shorter prefix.
//
// Layout, everywhere below: latents are row-major [n_frames * tokens_per_frame, enc_dim] per item
// with the frames in order and, inside a frame, the encoder's token order (h-major then w);
// actions and states are [n_frames, action_dim] / [n_frames, state_dim]. `n_batch` items are
// concatenated in that order. Row block t of a context carries the action that drives the step
// from frame t to frame t+1 and the state (pose) AT frame t.
int  jepa_ac_tokens_per_frame(const jepa_model * model);  // grid_size^2 (256 for the released ViT-g)
int  jepa_ac_action_dim(const jepa_model * model);        // 7
int  jepa_ac_state_dim(const jepa_model * model);         // 7
int  jepa_ac_max_frames(const jepa_model * model);        // jepa.pred.n_frames: the frame-slot cap
// True when the released world-model loop normalises latents between steps (jepa.pred.normalize_reps).
bool jepa_ac_normalize_reps(const jepa_model * model);

// Non-affine LayerNorm over each row, in place -- Meta's `F.layer_norm(h, (D,))`
// (notebooks/utils/world_model_wrapper.py). Apply it to the encoder latents BEFORE the first
// jepa_ac_predict and to any predicted frame you feed back by hand; jepa_ac_rollout does it for you
// when jepa_ac_normalize_reps() is true. eps comes from jepa.pred.norm_reps_eps (1e-5).
void jepa_ac_normalize(const jepa_model * model, float * rows, int64_t n_rows, int64_t dim);

// One predictor call over `n_frames` context frames for `n_batch` independent items (the K action
// candidates of a planning step live on the graph's batch axis; on the CPU at f32 a batched call is
// bit-identical to n_batch sequential ones -- tests/test-predictor.cpp gates that).
//   context : [n_batch * n_frames * tokens_per_frame, enc_dim]
//   actions : [n_batch * n_frames, action_dim]
//   states  : [n_batch * n_frames, state_dim]
//   out     : [n_batch * tokens_per_frame, enc_dim] -- the next frame of each item.
//             Caller frees out->data.
int jepa_ac_predict(jepa_context * ctx, const float * context, int n_frames, int n_batch,
                    const float * actions, const float * states, jepa_output * out);
// Same call, returning every frame's prediction: [n_batch * n_frames * tokens_per_frame, enc_dim].
// Row block t is the prediction of frame t+1 given frames 0..t -- this is what Meta's
// `predictor(x, actions, states)` returns, and what the parity fixtures hold.
int jepa_ac_predict_all(jepa_context * ctx, const float * context, int n_frames, int n_batch,
                        const float * actions, const float * states, jepa_output * out);

// Autoregressive rollout of `n_cand` candidate action sequences over `horizon` steps, all batched
// on the batch axis of ONE graph per step. The encoded context is shared by every candidate.
//   context     : [n_seed * tokens_per_frame, enc_dim] -- already normalised if
//                 jepa_ac_normalize_reps(); one encode, reused by every candidate (step 7.2 turns
//                 this argument into a cached handle without changing the rest of the signature)
//   seed_states : [n_seed, state_dim]
//   actions     : [n_cand * horizon, action_dim] -- candidate c, step h at row c*horizon + h
//   states      : [n_cand * horizon, state_dim] poses of the frames the steps produce, or NULL to
//                 have jepa_ac_next_state() generate them from `actions` (what the reference does)
//   out         : [n_cand * horizon * tokens_per_frame, enc_dim], caller-allocated: candidate c's
//                 step h at row (c*horizon + h) * tokens_per_frame
int jepa_ac_rollout(jepa_context * ctx, const float * context, int n_seed, const float * seed_states,
                    const float * actions, const float * states, int n_cand, int horizon, float * out);

// The same rollout with the observed frames' own history. With n_seed > 1 the frames before the last
// are past observations, and the actions BETWEEN them are known — `seed_actions` is [n_seed-1,
// action_dim], the action that took the arm from observed frame j to frame j+1. The LAST observed
// frame is where planning starts, so it always carries the candidate's actions[c][0].
// jepa_ac_rollout() is this with seed_actions = NULL, which makes every earlier observed frame reuse
// actions[c][0] — a placeholder that is only right for n_seed == 1, where there are no earlier
// frames. Prefer this entry point whenever n_seed > 1 (tests/test-predictor.cpp gates it against a
// two-observed-frame reference produced by Meta's own predictor).
int jepa_ac_rollout_ex(jepa_context * ctx, const float * context, int n_seed,
                       const float * seed_actions, const float * seed_states,
                       const float * actions, const float * states,
                       int n_cand, int horizon, float * out);

// --- cached planning context (plan item 7.2) ------------------------------------------------
// A planner encodes once and scores thousands of candidates against that encode, over many CEM
// iterations and many receding-horizon steps. This handle keeps the observed frames' latents ON THE
// DEVICE: the graph references them directly, so they are neither replicated across the K candidates
// on the host nor re-uploaded per step, and `pred.embed` runs on them once per graph instead of K
// times. Results are identical to the explicit-context path (tests/test-predictor.cpp gates
// bit-exactness on the CPU); the difference is only where the bytes come from.
typedef struct jepa_ac_context jepa_ac_context;

// latents : [n_frames * tokens_per_frame, enc_dim], already normalised if jepa_ac_normalize_reps()
// actions : [n_frames - 1, action_dim] history between the frames (NULL = zeros)
// states  : [n_frames, state_dim] the pose AT each frame
// Capacity is jepa.pred.n_frames (the predictor's frame slots); the device allocation is made once.
jepa_ac_context * jepa_ac_context_new(jepa_context * ctx, const float * latents, int n_frames,
                                      const float * actions, const float * states);
void jepa_ac_context_free(jepa_ac_context * handle);
int  jepa_ac_context_n_frames(const jepa_ac_context * handle);
int  jepa_ac_context_capacity(const jepa_ac_context * handle);
// Append one newly observed frame — the receding-horizon step. `action` is the action that took the
// arm from the current last frame to this one; `state` is the pose at the new frame.
int  jepa_ac_context_update(jepa_ac_context * handle, const float * latents, const float * action,
                            const float * state);
// Keep only the last `n_keep` frames (a planner's sliding window).
int  jepa_ac_context_trim(jepa_ac_context * handle, int n_keep);
// jepa_ac_rollout_ex over the handle's frames; `actions` / `states` / `out` as above.
int  jepa_ac_rollout_cached(jepa_context * ctx, jepa_ac_context * handle,
                            const float * actions, const float * states,
                            int n_cand, int horizon, float * out);

// --- CEM planner ------------------------------------------------------------------------------
// The loop V-JEPA 2-AC plans with (notebooks/utils/mpc_utils.py::cem): sample `samples` action
// trajectories from a diagonal Gaussian, roll them all out in one batched graph per horizon step,
// score the final frame against the goal with jepa_ac_energy, keep the `topk` elites and move the
// mean and standard deviation towards them with momentum.
//
// Only FOUR of the seven action dimensions are sampled — translation (0..2) and the gripper (6);
// the three rotation dimensions are hard zeros. That is the reference's action space, not a
// simplification, and the clamps and momenta below all live in it.
typedef struct {
    int   samples;                 // candidates per iteration (K, on the graph's batch axis)
    int   topk;                    // elites kept per iteration
    int   cem_steps;               // iterations
    int   horizon;                 // H, the planned action-sequence length
    float maxnorm;                 // |dx|,|dy|,|dz| clamp and the initial translation std (0.05)
    float gripper_clamp;           // |dgripper| clamp (0.75)
    float momentum_mean, momentum_std;                  // translation
    float momentum_mean_gripper, momentum_std_gripper;  // gripper
    float round_gripper;           // zero a final |gripper| below this (0.25)
    uint32_t seed;                 // RNG seed, used only when `noise` is NULL
} jepa_ac_cem_params;

// Every field this struct HAS, at the reference's default (mpc_utils.py's own signature defaults).
// Two of cem()'s parameters have no field here because the released planner never varies them:
// `axis={}` (pin a sampled dimension to a constant) and `close_gripper=None` (force the gripper shut
// from step h onwards). Both are reference defaults, not omissions from the algorithm; a caller that
// needs them can pin the corresponding lanes in `noise` or post-process the plan.
jepa_ac_cem_params jepa_ac_cem_default_params(void);

// Plan against `goal` ([tokens_per_frame, enc_dim], normalised like the handle's latents).
//   noise       : [cem_steps, horizon, samples, 4] standard-normal draws consumed in the reference's
//                 order (iteration, horizon step, then the rows of one randn(samples, 4)), or NULL
//                 to generate them from `seed`. Pass the draws to reproduce a PyTorch run: the
//                 built-in generator is xorshift+Box-Muller, not torch's.
//   out_actions : [horizon, action_dim] — the plan, rotation zeroed and small gripper commands
//                 rounded away, exactly what cem() returns.
//   out_energy  : [cem_steps] best energy per iteration, or NULL.
int jepa_ac_plan(jepa_context * ctx, jepa_ac_context * handle, const float * goal,
                 const jepa_ac_cem_params * params, const float * noise,
                 float * out_actions, float * out_energy);

// Meta's pose update (notebooks/utils/mpc_utils.py::compute_new_pose): translation added, rotation
// composed as extrinsic-xyz Euler angles, gripper added and clipped to [0, 1]. `state`, `action`
// and `out` are state_dim/action_dim long; `out` may not alias `state`.
void jepa_ac_next_state(const jepa_model * model, const float * state, const float * action, float * out);

// The planning energy the reference scores candidates with (notebooks/utils/mpc_utils.py::l1 and
// the energy-landscape notebook's loss_fn): mean |pred - goal| over ALL rows and dims of one item.
//   pred : [n_batch * n_rows, dim] (e.g. the last frame of each candidate's rollout)
//   goal : [n_rows, dim] -- the goal latents, shared by every item
//   out  : [n_batch] energies; lower is closer to the goal
void jepa_ac_energy(const float * pred, const float * goal, int n_batch, int64_t n_rows, int64_t dim,
                    float * out);

// ======================================================================================
// APPEND-ONLY: encoder batching (src/jepa.cpp).
// ======================================================================================

// How many image items (`n_batch * n_frames` slices of one jepa_input) the encoder folds into ONE
// ggml graph. The items stay independent — they live on the graph's batch dimension, so attention
// never mixes them — and the output rows are the same, in the same (batch, frame) order, as the
// one-graph-per-item path; on the CPU f32 is bit-identical (tests/test-batch.cpp gates this). On a
// CUDA device the two agree to ~1e-7 cosine but not bit-for-bit: GEMM tiling varies with the batch
// shape.
// Only the image families (ijepa / hfvit / lewm) batch; V-JEPA 2 / 2.1 still run one clip per graph.
//   n <= 0 : restore the default (32, or $JEPA_MAX_BATCH when the context was created)
//   n == 1 : the old per-item path — the debug switch (`jepa-embed --no-batch`, `JEPA_MAX_BATCH=1`)
// Inputs larger than n are encoded in ceil(n_items / n) graphs, so this caps memory, not batch size.
void jepa_context_set_max_batch(jepa_context * ctx, int n);
int  jepa_context_max_batch(const jepa_context * ctx);
// The per-graph item cap the last jepa_encode call actually used — jepa_context_max_batch(), or
// less when the $JEPA_MAX_GRAPH_MIB memory guard shrank it (the final ragged chunk may hold fewer
// items than this). Only jepa_encode updates it; head/predictor/projector calls leave it unchanged.
int  jepa_context_last_batch(const jepa_context * ctx);

// ======================================================================================
// APPEND-ONLY: backend / device selection (src/jepa.cpp, src/jepa-gguf.cpp).
// See docs/architecture.md "GPU backend". Requires a build configured with -DJEPA_CUDA=ON; without one
// jepa_device_count() is 0 and any device >= 0 is rejected with a message.
// ======================================================================================

// Devices are numbered over the GPU devices of the ggml backend registry, in registry order,
// so device 0 is the first GPU (CUDA0), 1 the second, and so on. -1 means the CPU.
typedef struct {
    bool verbose;
    int  device;   // -1 = CPU (the default), >= 0 = the n-th GPU device
} jepa_model_params;

// Defaults: verbose = false, device from $JEPA_DEVICE ("cpu" | "cuda:N" | "gpu:N" | "N"; anything
// unparseable is reported and ignored), i.e. -1 unless the environment says otherwise.
jepa_model_params jepa_model_default_params(void);
// jepa_model_load(path, verbose) == jepa_model_load_ex(path, {verbose, $JEPA_DEVICE}).
// The weights are allocated on `params->device` and every jepa_context built from this model
// computes there: a model and its contexts never straddle two backends.
jepa_model * jepa_model_load_ex(const char * gguf_path, const jepa_model_params * params);

// GPU devices the ggml backend registry can see (0 for a CPU-only build).
int          jepa_device_count(void);
const char * jepa_device_name(int device);         // ggml device name, e.g. "CUDA0"
const char * jepa_device_description(int device);  // e.g. "NVIDIA RTX 4500 Ada Generation"
void         jepa_device_memory(int device, size_t * free_bytes, size_t * total_bytes);  // either may be NULL

int          jepa_model_device(const jepa_model * model);       // -1 = CPU
const char * jepa_model_device_name(const jepa_model * model);  // "CPU" | "CUDA0" | ...
bool         jepa_model_is_gpu(const jepa_model * model);

// GPU numeric policy. On CUDA an f16 weight's mul_mat otherwise runs cuBLAS with F16 compute AND
// (on Ada) an F16 GEMM output; GGML_PREC_F32 takes the error against the CPU from 4.6e-03 to
// 2.6e-05 for -21 % throughput at 2048 tokens, +9 % at 8192, and nothing at all on quantized
// weights, which never reach cuBLAS. On by default for a GPU context (correctness first), a no-op
// on the CPU. $JEPA_GPU_PREC=f16 is the same opt-out. It cannot help f32 weights: ggml's "F32"
// CUDA path is TF32 by way of the CUBLAS_GEMM_DEFAULT_TENSOR_OP algo enum, not the compute type.
// Takes effect from the next graph built by this context.
void jepa_context_set_mul_mat_prec_f32(jepa_context * ctx, bool on);
bool jepa_context_mul_mat_prec_f32(const jepa_context * ctx);

// ======================================================================================
// APPEND-ONLY: diagnostics capture (src/jepa.cpp) — for callers with no stderr to read.
// ======================================================================================

// Every diagnostic this library emits — the reason an entry point returned -1 or NULL included —
// is written to stderr through a single internal function. A language binding that has to turn a
// failed call into an error value cannot read that stream, so the same text is also appended to a
// per-thread capture buffer: call jepa_error_reset() to empty it, make the call, and on failure
// read jepa_error_text() for the message. Both the buffer and the reset are thread-local, so a
// capture only ever sees the lines the calling thread produced, and neither changes what is
// written to stderr. The buffer holds the last few kilobytes and is truncated beyond that; it also
// collects the informational lines of a verbose model load, which is why it is worth resetting
// immediately before the call whose failure you want to describe.
void         jepa_error_reset(void);
const char * jepa_error_text(void);   // never NULL; "" when nothing was logged since the reset

// ======================================================================================
// Thread safety
//
// The contract, which tests/test-threads.cpp checks and docs/architecture.md "Robustness"
// repeats:
//
//   jepa_model    — immutable once jepa_model_load returns. Any number of threads may share one
//                   model and call every introspection entry point (jepa_model_*), the
//                   preprocessing entry points and jepa_pool_* on it concurrently. The weights are
//                   read, never written, after load. The one exception is jepa_model_free, which
//                   must not overlap with any use of that model or of a context built from it.
//   jepa_context  — one per thread. It owns the graph arena, the graph allocator and the
//                   last-call statistics, all of which every jepa_encode / jepa_predict /
//                   jepa_head / jepa_lewm_* call rewrites, so two threads must never be inside one
//                   context at the same time. Build a context per thread from the shared model:
//                   contexts of one model are independent and never touch each other's state.
//                   Concurrent encodes are bit-identical to running them one after another, at the
//                   same n_threads, on the CPU.
//   jepa_error_reset / jepa_error_text — thread-local. A capture only ever holds the lines the
//                   calling thread produced, which is what makes it usable from a worker pool.
//   preprocessing — jepa_preprocess_*, jepa_load_image_rgb, jepa_resize_antialias_u8,
//                   jepa_softmax and jepa_top_k keep no state at all and are re-entrant.
//   diagnostics   — the "said once" warnings (JEPA_DEVICE, the GPU K/V and flash-attention notes,
//                   the mask-size note) are published atomically: at most one thread prints, and
//                   no thread races.
//   devices       — the ggml backend registry is loaded on first use behind the C++ static
//                   initialisation guarantee, so jepa_device_count / jepa_device_name and a
//                   concurrent first jepa_model_load are safe together.
//
// Also: n_threads is the width of the ggml thread pool INSIDE one graph. Raising it parallelises
// one call; a context per thread parallelises calls. Doing both oversubscribes the machine.
//
// Freed handles follow C's usual rule: a jepa_model * or jepa_context * that has been passed to its
// free function must not be used again, exactly as with free(). Passing NULL to any entry point is
// defined — it returns an error or is a no-op — but a stale pointer is not.
// ======================================================================================

#ifdef __cplusplus
}
#endif
