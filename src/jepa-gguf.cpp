// GGUF loading: hparam parsing (docs/gguf-schema.md) and tensor upload to the CPU backend.
#include "jepa-internal.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <sstream>

// ---------------------------------------------------------------------------------------------
// enum helpers
// ---------------------------------------------------------------------------------------------
jepa_family_id jepa_family_from_string(const std::string & s) {
    if (s == "ijepa")    return JEPA_FAMILY_IJEPA;
    if (s == "vjepa")    return JEPA_FAMILY_VJEPA;
    if (s == "vjepa2")   return JEPA_FAMILY_VJEPA2;
    if (s == "vjepa2_1") return JEPA_FAMILY_VJEPA2_1;
    if (s == "levjepa")  return JEPA_FAMILY_LEVJEPA;
    if (s == "hfvit")    return JEPA_FAMILY_HFVIT;
    if (s == "lewm")     return JEPA_FAMILY_LEWM;
    return JEPA_FAMILY_UNKNOWN;
}

// An unknown activation used to abort(). It is metadata from an untrusted file, so it is now a
// load failure like any other: *ok is cleared and the caller returns nullptr with the message below.
jepa_act_id jepa_act_from_string(const std::string & s, bool * ok) {
    if (s == "gelu_erf" || s == "gelu")  return JEPA_ACT_GELU_ERF;
    if (s == "gelu_tanh" || s == "gelu_new" || s == "gelu_pytorch_tanh") return JEPA_ACT_GELU_TANH;
    if (s == "silu" || s == "swish")     return JEPA_ACT_SILU;
    jepa_log("jepa: unknown activation '%s' (expected gelu_erf | gelu_tanh | silu)\n", s.c_str());
    if (ok) *ok = false;
    return JEPA_ACT_GELU_ERF;
}

const char * jepa_act_name(jepa_act_id a) {
    switch (a) {
        case JEPA_ACT_GELU_ERF:  return "gelu_erf";
        case JEPA_ACT_GELU_TANH: return "gelu_tanh";
        case JEPA_ACT_SILU:      return "silu";
    }
    return "?";
}

jepa_pos_id jepa_pos_from_string(const std::string & s) {
    if (s == "sincos2d") return JEPA_POS_SINCOS2D;
    if (s == "sincos3d") return JEPA_POS_SINCOS3D;
    if (s == "learned")  return JEPA_POS_LEARNED;
    if (s == "rope3d")   return JEPA_POS_ROPE3D;
    return JEPA_POS_NONE;
}

// general.file_type carries a **GGML_FTYPE_*** value (enum ggml_ftype in ggml/include/ggml.h), the
// same value tools/jepa-quantize.cpp writes from its TYPE_SPECS table -- NOT the llama.cpp
// LLAMA_FTYPE_* numbering (whose 12 is Q3_K_M while GGML_FTYPE_MOSTLY_Q4_K is 12).  Every type the
// quantizer can produce is listed first; the rest of the enum follows so a foreign file still names
// itself correctly.
const char * jepa_file_type_name(uint32_t ftype) {
    switch ((int32_t) ftype) {
        case GGML_FTYPE_ALL_F32:              return "f32";     // 0   jepa-quantize f32
        case GGML_FTYPE_MOSTLY_F16:           return "f16";     // 1   jepa-quantize f16
        case GGML_FTYPE_MOSTLY_Q4_0:          return "q4_0";    // 2   jepa-quantize q4_0
        case GGML_FTYPE_MOSTLY_Q4_1:          return "q4_1";    // 3   jepa-quantize q4_1
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: return "q4_1_some_f16";  // 4
        case GGML_FTYPE_MOSTLY_Q8_0:          return "q8_0";    // 7   jepa-quantize q8_0
        case GGML_FTYPE_MOSTLY_Q5_0:          return "q5_0";    // 8   jepa-quantize q5_0
        case GGML_FTYPE_MOSTLY_Q5_1:          return "q5_1";    // 9   jepa-quantize q5_1
        case GGML_FTYPE_MOSTLY_Q2_K:          return "q2_k";    // 10
        case GGML_FTYPE_MOSTLY_Q3_K:          return "q3_k";    // 11
        case GGML_FTYPE_MOSTLY_Q4_K:          return "q4_k";    // 12  jepa-quantize q4_k
        case GGML_FTYPE_MOSTLY_Q5_K:          return "q5_k";    // 13  jepa-quantize q5_k
        case GGML_FTYPE_MOSTLY_Q6_K:          return "q6_k";    // 14  jepa-quantize q6_k
        case GGML_FTYPE_MOSTLY_IQ2_XXS:       return "iq2_xxs"; // 15
        case GGML_FTYPE_MOSTLY_IQ2_XS:        return "iq2_xs";  // 16
        case GGML_FTYPE_MOSTLY_IQ3_XXS:       return "iq3_xxs"; // 17
        case GGML_FTYPE_MOSTLY_IQ1_S:         return "iq1_s";   // 18
        case GGML_FTYPE_MOSTLY_IQ4_NL:        return "iq4_nl";  // 19
        case GGML_FTYPE_MOSTLY_IQ3_S:         return "iq3_s";   // 20
        case GGML_FTYPE_MOSTLY_IQ2_S:         return "iq2_s";   // 21
        case GGML_FTYPE_MOSTLY_IQ4_XS:        return "iq4_xs";  // 22
        case GGML_FTYPE_MOSTLY_IQ1_M:         return "iq1_m";   // 23
        case GGML_FTYPE_MOSTLY_BF16:          return "bf16";    // 24
        case GGML_FTYPE_MOSTLY_MXFP4:         return "mxfp4";   // 25
        case GGML_FTYPE_MOSTLY_NVFP4:         return "nvfp4";   // 26
        case GGML_FTYPE_MOSTLY_Q1_0:          return "q1_0";    // 27
        case GGML_FTYPE_MOSTLY_Q2_0:          return "q2_0";    // 28
        default: return "unknown";
    }
}

// ---------------------------------------------------------------------------------------------
// gguf key access
// ---------------------------------------------------------------------------------------------
namespace {

struct kv_reader {
    const gguf_context * gg;

    int64_t find(const char * key) const { return gguf_find_key(gg, key); }
    bool has(const char * key) const { return find(key) >= 0; }

    // integer of any width (u8..i64) -> int64
    bool get_int(const char * key, int64_t & out) const {
        int64_t id = find(key);
        if (id < 0) return false;
        switch (gguf_get_kv_type(gg, id)) {
            case GGUF_TYPE_UINT8:   out = gguf_get_val_u8(gg, id);  return true;
            case GGUF_TYPE_INT8:    out = gguf_get_val_i8(gg, id);  return true;
            case GGUF_TYPE_UINT16:  out = gguf_get_val_u16(gg, id); return true;
            case GGUF_TYPE_INT16:   out = gguf_get_val_i16(gg, id); return true;
            case GGUF_TYPE_UINT32:  out = gguf_get_val_u32(gg, id); return true;
            case GGUF_TYPE_INT32:   out = gguf_get_val_i32(gg, id); return true;
            case GGUF_TYPE_UINT64:  out = (int64_t) gguf_get_val_u64(gg, id); return true;
            case GGUF_TYPE_INT64:   out = gguf_get_val_i64(gg, id); return true;
            case GGUF_TYPE_BOOL:    out = gguf_get_val_bool(gg, id) ? 1 : 0; return true;
            default: return false;
        }
    }
    int geti(const char * key, int def) const { int64_t v; return get_int(key, v) ? (int) v : def; }
    bool getb(const char * key, bool def) const {
        int64_t id = find(key);
        if (id < 0) return def;
        if (gguf_get_kv_type(gg, id) == GGUF_TYPE_BOOL) return gguf_get_val_bool(gg, id);
        int64_t v; return get_int(key, v) ? v != 0 : def;
    }
    float getf(const char * key, float def) const {
        int64_t id = find(key);
        if (id < 0) return def;
        switch (gguf_get_kv_type(gg, id)) {
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gg, id);
            case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(gg, id);
            default: { int64_t v; return get_int(key, v) ? (float) v : def; }
        }
    }
    std::string gets(const char * key, const std::string & def) const {
        int64_t id = find(key);
        if (id < 0 || gguf_get_kv_type(gg, id) != GGUF_TYPE_STRING) return def;
        return gguf_get_val_str(gg, id);
    }
    bool get_farr(const char * key, float * out, size_t n) const {
        int64_t id = find(key);
        if (id < 0 || gguf_get_kv_type(gg, id) != GGUF_TYPE_ARRAY) return false;
        if (gguf_get_arr_n(gg, id) != n) return false;
        const void * data = gguf_get_arr_data(gg, id);
        switch (gguf_get_arr_type(gg, id)) {
            case GGUF_TYPE_FLOAT32: memcpy(out, data, n * sizeof(float)); return true;
            case GGUF_TYPE_FLOAT64: for (size_t i = 0; i < n; i++) out[i] = (float) ((const double *) data)[i]; return true;
            default: return false;
        }
    }
    std::vector<int> get_iarr(const char * key) const {
        std::vector<int> v;
        int64_t id = find(key);
        if (id < 0 || gguf_get_kv_type(gg, id) != GGUF_TYPE_ARRAY) return v;
        size_t n = gguf_get_arr_n(gg, id);
        const void * data = gguf_get_arr_data(gg, id);
        for (size_t i = 0; i < n; i++) {
            switch (gguf_get_arr_type(gg, id)) {
                case GGUF_TYPE_UINT32: v.push_back((int) ((const uint32_t *) data)[i]); break;
                case GGUF_TYPE_INT32:  v.push_back(((const int32_t *) data)[i]); break;
                case GGUF_TYPE_UINT64: v.push_back((int) ((const uint64_t *) data)[i]); break;
                case GGUF_TYPE_INT64:  v.push_back((int) ((const int64_t *) data)[i]); break;
                case GGUF_TYPE_UINT16: v.push_back(((const uint16_t *) data)[i]); break;
                case GGUF_TYPE_UINT8:  v.push_back(((const uint8_t *) data)[i]); break;
                default: return v;
            }
        }
        return v;
    }
    std::vector<std::string> get_sarr(const char * key) const {
        std::vector<std::string> v;
        int64_t id = find(key);
        if (id < 0 || gguf_get_kv_type(gg, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gg, id) != GGUF_TYPE_STRING) return v;
        size_t n = gguf_get_arr_n(gg, id);
        for (size_t i = 0; i < n; i++) v.push_back(gguf_get_arr_str(gg, id, i));
        return v;
    }

    // render any key as text (jepa-info)
    std::string render(int64_t id) const {
        std::ostringstream os;
        gguf_type t = gguf_get_kv_type(gg, id);
        switch (t) {
            case GGUF_TYPE_UINT8:   os << (int) gguf_get_val_u8(gg, id); break;
            case GGUF_TYPE_INT8:    os << (int) gguf_get_val_i8(gg, id); break;
            case GGUF_TYPE_UINT16:  os << gguf_get_val_u16(gg, id); break;
            case GGUF_TYPE_INT16:   os << gguf_get_val_i16(gg, id); break;
            case GGUF_TYPE_UINT32:  os << gguf_get_val_u32(gg, id); break;
            case GGUF_TYPE_INT32:   os << gguf_get_val_i32(gg, id); break;
            case GGUF_TYPE_UINT64:  os << gguf_get_val_u64(gg, id); break;
            case GGUF_TYPE_INT64:   os << gguf_get_val_i64(gg, id); break;
            case GGUF_TYPE_FLOAT32: os << gguf_get_val_f32(gg, id); break;
            case GGUF_TYPE_FLOAT64: os << gguf_get_val_f64(gg, id); break;
            case GGUF_TYPE_BOOL:    os << (gguf_get_val_bool(gg, id) ? "true" : "false"); break;
            case GGUF_TYPE_STRING:  os << '"' << gguf_get_val_str(gg, id) << '"'; break;
            case GGUF_TYPE_ARRAY: {
                size_t n = gguf_get_arr_n(gg, id);
                gguf_type at = gguf_get_arr_type(gg, id);
                const void * data = at == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(gg, id);
                os << '[';
                size_t shown = n > 16 ? 16 : n;
                for (size_t i = 0; i < shown; i++) {
                    if (i) os << ", ";
                    switch (at) {
                        case GGUF_TYPE_UINT32:  os << ((const uint32_t *) data)[i]; break;
                        case GGUF_TYPE_INT32:   os << ((const int32_t *) data)[i]; break;
                        case GGUF_TYPE_FLOAT32: os << ((const float *) data)[i]; break;
                        case GGUF_TYPE_FLOAT64: os << ((const double *) data)[i]; break;
                        case GGUF_TYPE_UINT64:  os << ((const uint64_t *) data)[i]; break;
                        case GGUF_TYPE_INT64:   os << ((const int64_t *) data)[i]; break;
                        case GGUF_TYPE_UINT8:   os << (int) ((const uint8_t *) data)[i]; break;
                        case GGUF_TYPE_BOOL:    os << (((const int8_t *) data)[i] ? "true" : "false"); break;
                        case GGUF_TYPE_STRING:  os << '"' << gguf_get_arr_str(gg, id, i) << '"'; break;
                        default: os << '?'; break;
                    }
                }
                if (shown < n) os << ", ... (" << n << " items)";
                os << ']';
                break;
            }
            default: os << '?';
        }
        return os.str();
    }
};

// ---------------------------------------------------------------------------------------------
// range checks (docs/architecture.md "Robustness")
// ---------------------------------------------------------------------------------------------
// Every integer below is metadata from a file the caller did not write. Left unchecked, a zero
// n_head divides by zero, a negative dimension becomes a huge size_t, and an n_layer of 2^31 asks
// for 300 GiB of per-block bundles before a single tensor is read. The bounds are two to three
// orders of magnitude above anything a real checkpoint uses (JEPA_LIMIT_* in jepa-internal.h), so
// they only ever fire on a file that is broken or hostile — and they fire with a message naming the
// key, never with a silent clamp.
static bool kv_range(const char * key, int64_t v, int64_t lo, int64_t hi) {
    if (v >= lo && v <= hi) return true;
    jepa_log("jepa: %s = %" PRId64 " is out of range [%" PRId64 ", %" PRId64 "]\n", key, v, lo, hi);
    return false;
}

static bool kv_finite(const char * key, float v, float lo, float hi) {
    if (v >= lo && v <= hi) return true;   // false for NaN, which is the point
    jepa_log("jepa: %s = %g is not a finite value in [%g, %g]\n", key, (double) v, (double) lo, (double) hi);
    return false;
}

} // namespace

bool jepa_hparams_from_gguf(const gguf_context * gg, jepa_hparams & hp) {
    kv_reader r{gg};

    // raw dump (file order)
    const int64_t n_kv = gguf_get_n_kv(gg);
    for (int64_t i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(gg, i);
        if (strncmp(key, "general.", 8) == 0 || strncmp(key, "jepa.", 5) == 0) {
            hp.raw_kv.emplace_back(key, r.render(i));
        }
    }

    hp.arch        = r.gets("general.architecture", "");
    hp.name        = r.gets("general.name", "");
    hp.license     = r.gets("general.license", "");
    hp.source_url  = r.gets("general.source_url", r.gets("general.source.url", ""));
    hp.description = r.gets("general.description", "");
    hp.file_type   = (uint32_t) r.geti("general.file_type", 0);
    hp.schema_version = (uint32_t) r.geti("jepa.schema_version", 0);
    if (hp.arch != "jepa") {
        jepa_log("jepa: general.architecture is '%s', expected 'jepa'\n", hp.arch.c_str());
        return false;
    }
    hp.family_str = r.gets("jepa.family", "");
    hp.family     = jepa_family_from_string(hp.family_str);
    hp.modality   = r.gets("jepa.modality", "image");
    if (hp.family == JEPA_FAMILY_UNKNOWN) {
        jepa_log("jepa: unknown jepa.family '%s'\n", hp.family_str.c_str());
        return false;
    }

    // --- encoder
    jepa_enc_hparams & e = hp.enc;
    const char * required[] = {"jepa.enc.embed_dim", "jepa.enc.n_layer", "jepa.enc.n_head", "jepa.enc.ffn_dim", "jepa.enc.patch_size"};
    for (const char * k : required) {
        if (!r.has(k)) { jepa_log("jepa: missing required key %s\n", k); return false; }
    }
    e.embed_dim    = r.geti("jepa.enc.embed_dim", 0);
    e.n_layer      = r.geti("jepa.enc.n_layer", 0);
    e.n_head       = r.geti("jepa.enc.n_head", 0);
    e.ffn_dim      = r.geti("jepa.enc.ffn_dim", 0);
    e.patch_size   = r.geti("jepa.enc.patch_size", 16);
    e.tubelet_size = r.geti("jepa.enc.tubelet_size", 1);
    e.img_size     = r.geti("jepa.enc.img_size", 224);
    e.n_frames     = r.geti("jepa.enc.n_frames", 1);
    e.in_chans     = r.geti("jepa.enc.in_chans", 3);
    if (!kv_range("jepa.enc.embed_dim",    e.embed_dim,    1, JEPA_LIMIT_EMBED_DIM)  ||
        !kv_range("jepa.enc.n_layer",      e.n_layer,      1, JEPA_LIMIT_N_LAYER)    ||
        !kv_range("jepa.enc.n_head",       e.n_head,       1, JEPA_LIMIT_N_HEAD)     ||
        !kv_range("jepa.enc.ffn_dim",      e.ffn_dim,      1, JEPA_LIMIT_FFN_DIM)    ||
        !kv_range("jepa.enc.patch_size",   e.patch_size,   1, JEPA_LIMIT_PATCH_SIZE) ||
        !kv_range("jepa.enc.tubelet_size", e.tubelet_size, 1, JEPA_LIMIT_PATCH_SIZE) ||
        !kv_range("jepa.enc.img_size",     e.img_size,     1, JEPA_LIMIT_IMG_SIZE)   ||
        !kv_range("jepa.enc.n_frames",     e.n_frames,     1, JEPA_LIMIT_N_FRAMES)   ||
        !kv_range("jepa.enc.in_chans",     e.in_chans,     1, JEPA_LIMIT_IN_CHANS)) {
        return false;
    }
    e.ln_eps       = r.getf("jepa.enc.ln_eps", 1e-6f);
    if (!kv_finite("jepa.enc.ln_eps", e.ln_eps, 0.0f, 1.0f)) return false;
    bool act_ok = true;
    e.act          = jepa_act_from_string(r.gets("jepa.enc.act", "gelu_erf"), &act_ok);
    if (!act_ok) return false;
    e.pos_type_str = r.gets("jepa.enc.pos_type", "");
    e.pos_type     = jepa_pos_from_string(e.pos_type_str);
    e.rope_theta   = r.getf("jepa.enc.rope_theta", 10000.0f);
    e.rope_interpolate = r.getb("jepa.enc.rope_interpolate", false);
    e.rope_freq_layout = r.gets("jepa.enc.rope_freq_layout", "");
    e.rope_ref_grid    = r.geti("jepa.enc.rope_ref_grid", 0);
    e.cls_token    = r.getb("jepa.enc.cls_token", false);
    e.n_registers  = r.geti("jepa.enc.n_registers", 0);
    e.qkv_fused    = r.getb("jepa.enc.qkv_fused", true);
    e.modality_embed    = r.getb("jepa.enc.modality_embed", false);
    e.image_patch_embed = r.getb("jepa.enc.image_patch_embed", false);
    e.hier_layers  = r.get_iarr("jepa.enc.hier_layers");
    e.layer_scale  = r.getb("jepa.enc.layer_scale", false);
    // Absent means "full", so no file written before this key existed changes meaning. An unknown
    // value is refused rather than silently downgraded to full attention: running levjepa weights
    // unmasked does not fail, it just returns worse features (docs/gguf-schema.md, levjepa note).
    e.attn_mode    = r.gets("jepa.enc.attn_mode", "full");
    if (e.attn_mode != "full" && e.attn_mode != "block_causal") {
        jepa_log("jepa: unknown jepa.enc.attn_mode '%s' (expected full | block_causal)\n", e.attn_mode.c_str());
        return false;
    }
    // Only the video encoder builds the mask, so the key is refused on a family whose graph would
    // ignore it. Loading such a file would run unmasked attention without saying so, which is the
    // one failure mode this key exists to prevent.
    if (e.block_causal() && hp.family != JEPA_FAMILY_VJEPA && hp.family != JEPA_FAMILY_VJEPA2 &&
        hp.family != JEPA_FAMILY_VJEPA2_1 && hp.family != JEPA_FAMILY_LEVJEPA) {
        jepa_log("jepa: jepa.enc.attn_mode = block_causal on family '%s', whose encoder graph has no "
                 "attention mask — it would run unmasked. Only the video families implement it.\n",
                 hp.family_str.c_str());
        return false;
    }
    e.has_proj     = r.has("jepa.enc.proj_act");
    e.proj_act     = jepa_act_from_string(r.gets("jepa.enc.proj_act", "gelu_erf"), &act_ok);
    if (!act_ok) return false;
    if (!kv_range("jepa.enc.n_registers", e.n_registers, 0, JEPA_LIMIT_N_TOKENS)) return false;
    // array-valued keys are bounded by the file's own length, which is not a bound worth having
    if (!kv_range("jepa.enc.hier_layers length", (int64_t) e.hier_layers.size(), 0, JEPA_LIMIT_N_LAYER)) return false;
    if (!kv_finite("jepa.enc.rope_theta", e.rope_theta, 1e-6f, 1e12f)) return false;
    if (!kv_range("jepa.enc.rope_ref_grid", e.rope_ref_grid, 0, JEPA_LIMIT_IMG_SIZE)) return false;
    // Interpolated RoPE rescales the grid by train_grid / grid, so it needs a training grid to
    // divide by; rope3d_build asserts on its absence rather than dividing by zero.
    if (e.rope_interpolate && e.rope_ref_grid < 1) {
        jepa_log("jepa: jepa.enc.rope_interpolate is set but jepa.enc.rope_ref_grid is %d — "
                 "interpolation has no training grid to rescale from\n", e.rope_ref_grid);
        return false;
    }
    // n_head is >= 1 by the range check above, so this modulo cannot divide by zero
    if (e.embed_dim % e.n_head != 0) {
        jepa_log("jepa: embed_dim %d not divisible by n_head %d\n", e.embed_dim, e.n_head);
        return false;
    }
    // 3-D RoPE rotates pairs, so jepa_rope3d_apply asserts head_dim % 2 == 0. Every released video
    // checkpoint has an even one; a file that does not would abort inside the graph builder.
    const bool rope3d_enc = hp.family == JEPA_FAMILY_VJEPA2 || hp.family == JEPA_FAMILY_VJEPA2_1 ||
                            hp.family == JEPA_FAMILY_LEVJEPA || e.pos_type == JEPA_POS_ROPE3D;
    if (rope3d_enc && e.head_dim() % 2 != 0) {
        jepa_log("jepa: 3-D RoPE rotates pairs, so the encoder head width has to be even; "
                 "embed_dim %d / n_head %d is %d\n", e.embed_dim, e.n_head, e.head_dim());
        return false;
    }

    // --- predictor
    jepa_pred_hparams & p = hp.pred;
    p.present = r.has("jepa.pred.kind") && r.gets("jepa.pred.kind", "none") != "none";
    if (p.present) {
        p.kind        = r.gets("jepa.pred.kind", "");
        p.embed_dim   = r.geti("jepa.pred.embed_dim", 0);
        p.n_layer     = r.geti("jepa.pred.n_layer", 0);
        p.n_head      = r.geti("jepa.pred.n_head", 0);
        p.ffn_dim     = r.geti("jepa.pred.ffn_dim", 0);
        p.n_mask_tokens = r.geti("jepa.pred.n_mask_tokens", 0);
        p.out_dim     = r.geti("jepa.pred.out_dim", e.embed_dim);
        p.rope_freq_layout = r.gets("jepa.pred.rope_freq_layout", "");
        p.rope_interpolate = r.getb("jepa.pred.rope_interpolate", false);
        p.rope_ref_grid = r.geti("jepa.pred.rope_ref_grid", 0);
        p.grid_size   = r.geti("jepa.pred.grid_size", 0);
        p.n_hier_in   = r.geti("jepa.pred.n_hier_in", 1);
        p.modality_embed = r.getb("jepa.pred.modality_embed", false);
        p.context_proj   = r.getb("jepa.pred.context_proj", false);
        p.action_dim  = r.geti("jepa.pred.action_dim", 0);
        p.state_dim   = r.geti("jepa.pred.state_dim", 0);
        p.frame_causal = r.getb("jepa.pred.frame_causal", false);
        p.n_frames    = r.geti("jepa.pred.n_frames", 0);
        p.head_dim    = r.geti("jepa.pred.head_dim", 0);
        p.ln_eps      = r.getf("jepa.pred.ln_eps", 1e-6f);
        p.adaln_eps   = r.getf("jepa.pred.adaln_eps", 1e-6f);
        p.act         = jepa_act_from_string(r.gets("jepa.pred.act", "gelu_erf"), &act_ok);
        p.qkv_bias    = r.getb("jepa.pred.qkv_bias", true);
        p.action_act  = jepa_act_from_string(r.gets("jepa.pred.action_act", "silu"), &act_ok);
        p.proj_act    = jepa_act_from_string(r.gets("jepa.pred.proj_act", "gelu_erf"), &act_ok);
        if (!act_ok) return false;
        if (!kv_range("jepa.pred.embed_dim",     p.embed_dim,     1, JEPA_LIMIT_EMBED_DIM) ||
            !kv_range("jepa.pred.n_layer",       p.n_layer,       1, JEPA_LIMIT_N_LAYER)   ||
            !kv_range("jepa.pred.n_head",        p.n_head,        1, JEPA_LIMIT_N_HEAD)    ||
            !kv_range("jepa.pred.ffn_dim",       p.ffn_dim,       1, JEPA_LIMIT_FFN_DIM)   ||
            !kv_range("jepa.pred.out_dim",       p.out_dim,       1, JEPA_LIMIT_FFN_DIM)   ||
            !kv_range("jepa.pred.head_dim",      p.head_dim,      0, JEPA_LIMIT_EMBED_DIM) ||
            !kv_range("jepa.pred.n_mask_tokens", p.n_mask_tokens, 0, JEPA_LIMIT_N_TOKENS)  ||
            !kv_range("jepa.pred.grid_size",     p.grid_size,     0, JEPA_LIMIT_IMG_SIZE)  ||
            !kv_range("jepa.pred.rope_ref_grid", p.rope_ref_grid, 0, JEPA_LIMIT_IMG_SIZE)  ||
            !kv_range("jepa.pred.n_hier_in",     p.n_hier_in,     0, JEPA_LIMIT_N_LAYER)   ||
            !kv_range("jepa.pred.action_dim",    p.action_dim,    0, JEPA_LIMIT_EMBED_DIM) ||
            !kv_range("jepa.pred.state_dim",     p.state_dim,     0, JEPA_LIMIT_EMBED_DIM) ||
            !kv_range("jepa.pred.n_frames",      p.n_frames,      0, JEPA_LIMIT_N_FRAMES)  ||
            !kv_finite("jepa.pred.ln_eps",       p.ln_eps,        0.0f, 1.0f)              ||
            !kv_finite("jepa.pred.adaln_eps",    p.adaln_eps,     0.0f, 1.0f)) {
            return false;
        }
        // head_dim_eff() divides by n_head, which the range check pinned at >= 1
        if (p.head_dim == 0 && p.embed_dim % p.n_head != 0) {
            jepa_log("jepa: jepa.pred.embed_dim %d is not divisible by jepa.pred.n_head %d and there is "
                     "no jepa.pred.head_dim to say what the head width should be\n", p.embed_dim, p.n_head);
            return false;
        }
        if (p.rope_interpolate && p.rope_ref_grid < 1) {
            jepa_log("jepa: jepa.pred.rope_interpolate is set but jepa.pred.rope_ref_grid is %d\n",
                     p.rope_ref_grid);
            return false;
        }
        if (p.kind == "masked" && p.head_dim_eff() % 2 != 0) {
            jepa_log("jepa: the masked predictor rotates pairs with 3-D RoPE, so its head width has "
                     "to be even; it is %d\n", p.head_dim_eff());
            return false;
        }
        if (p.kind == "masked" && p.grid_size <= 0 && e.grid_size() <= 0) {
            // jepa_predictor_rope_params divides the token id by grid*grid to recover the frame
            jepa_log("jepa: a masked predictor needs a position grid: jepa.pred.grid_size is %d and "
                     "jepa.enc.img_size / jepa.enc.patch_size is %d\n", p.grid_size, e.grid_size());
            return false;
        }
        if (p.kind == "lewm" && (p.action_dim <= 0 || p.n_frames <= 0)) {
            jepa_log("jepa: a lewm predictor needs jepa.pred.action_dim > 0 (is %d) and "
                     "jepa.pred.n_frames > 0 (is %d)\n", p.action_dim, p.n_frames);
            return false;
        }
    }

    // --- head
    jepa_head_hparams & h = hp.head;
    h.kind = r.gets("jepa.head.kind", "none");
    h.present = h.kind != "none" && !h.kind.empty();
    if (h.present) {
        h.n_classes     = r.geti("jepa.head.n_classes", 0);
        h.n_pool_layers = r.geti("jepa.head.n_pool_layers", 0);
        h.labels        = r.get_sarr("jepa.head.labels");
        if (!kv_range("jepa.head.n_classes",     h.n_classes,     0, JEPA_LIMIT_N_TOKENS) ||
            !kv_range("jepa.head.n_pool_layers", h.n_pool_layers, 0, JEPA_LIMIT_N_LAYER) ||
            !kv_range("jepa.head.labels length", (int64_t) h.labels.size(), 0, JEPA_LIMIT_N_TOKENS)) {
            return false;
        }
    }

    // --- preprocessing
    jepa_pre_hparams & pre = hp.pre;
    r.get_farr("jepa.pre.mean", pre.mean, 3);
    r.get_farr("jepa.pre.std", pre.std, 3);
    pre.resize_short = r.geti("jepa.pre.resize_short", e.img_size);
    pre.crop         = r.geti("jepa.pre.crop", e.img_size);
    pre.resample     = r.gets("jepa.pre.resample", "bilinear");
    pre.resize_mode  = r.gets("jepa.pre.resize_mode", "shortest_edge");
    pre.rescale      = r.getf("jepa.pre.rescale", 1.0f / 255.0f);
    // resize_short drives the intermediate buffer of src/preprocess.cpp; crop 0 means "the short
    // side of the resized image", which is why it is allowed to be zero and resize_short is not.
    if (!kv_range("jepa.pre.resize_short", pre.resize_short, 1, JEPA_LIMIT_IMG_SIZE) ||
        !kv_range("jepa.pre.crop",         pre.crop,         0, JEPA_LIMIT_IMG_SIZE) ||
        !kv_finite("jepa.pre.rescale",     pre.rescale,      1e-9f, 1e9f)) {
        return false;
    }
    for (int i = 0; i < 3; i++) {
        if (!kv_finite("jepa.pre.mean", pre.mean[i], -1e6f, 1e6f)) return false;
        // a zero std would divide every pixel by zero
        if (!kv_finite("jepa.pre.std", pre.std[i], 1e-9f, 1e6f)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// model tensors
// ---------------------------------------------------------------------------------------------
ggml_tensor * jepa_model::get(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor * jepa_model::require(const std::string & name) const {
    ggml_tensor * t = get(name);
    if (!t) {
        jepa_log("jepa: required tensor '%s' missing from %s\n", name.c_str(), path.c_str());
        abort();
    }
    return t;
}

jepa_layer jepa_layer_from_model(const jepa_model & m, const std::string & p) {
    jepa_layer L;
    L.ln1_w  = m.get(p + "ln1.weight");      L.ln1_b  = m.get(p + "ln1.bias");
    L.qkv_w  = m.get(p + "attn_qkv.weight"); L.qkv_b  = m.get(p + "attn_qkv.bias");
    L.q_w    = m.get(p + "attn_q.weight");   L.q_b    = m.get(p + "attn_q.bias");
    L.k_w    = m.get(p + "attn_k.weight");   L.k_b    = m.get(p + "attn_k.bias");
    L.v_w    = m.get(p + "attn_v.weight");   L.v_b    = m.get(p + "attn_v.bias");
    L.out_w  = m.get(p + "attn_out.weight"); L.out_b  = m.get(p + "attn_out.bias");
    L.ln2_w  = m.get(p + "ln2.weight");      L.ln2_b  = m.get(p + "ln2.bias");
    L.up_w   = m.get(p + "ffn_up.weight");   L.up_b   = m.get(p + "ffn_up.bias");
    L.down_w = m.get(p + "ffn_down.weight"); L.down_b = m.get(p + "ffn_down.bias");
    L.ls1    = m.get(p + "ls1");             L.ls2    = m.get(p + "ls2");
    L.adaln_w = m.get(p + "adaln.weight");   L.adaln_b = m.get(p + "adaln.bias");
    return L;
}

static bool check_shape(const jepa_model & m, const char * name, int64_t ne0, int64_t ne1 = -1) {
    ggml_tensor * t = m.get(name);
    if (!t) return true;  // optional tensors are checked by the builders
    bool ok = t->ne[0] == ne0 && (ne1 < 0 || t->ne[1] == ne1);
    if (!ok) {
        jepa_log("jepa: tensor %s has shape [%" PRId64 ", %" PRId64 "], expected [%" PRId64 ", %" PRId64 "]\n",
                 name, t->ne[0], t->ne[1], ne0, ne1 < 0 ? (int64_t) 1 : ne1);
    }
    return ok;
}

// Tensors that are added to, multiplied by or concatenated with the F32 activations rather than
// multiplied through mul_mat. ggml_concat and the elementwise ops require matching types, so an
// f16 CLS token in an otherwise f32 graph aborts inside the builder. Quantization never touches
// these — even a q4_k file keeps every norm, bias, token and position table in f32, because
// tools/jepa-quantize.cpp converts only the matmul weights — and docs/gguf-schema.md
// "Quantization rules" states it as a requirement, which is what this enforces.
static bool check_f32(const jepa_model & m, const std::string & name) {
    ggml_tensor * t = m.get(name);
    if (!t || t->type == GGML_TYPE_F32) return true;
    jepa_log("jepa: tensor %s is %s, but it is added to or concatenated with the f32 activations and "
             "has to be f32 (only the matmul weights may be quantized)\n", name.c_str(), ggml_type_name(t->type));
    return false;
}

// A vector: exactly `n` f32 values, whatever rank the file gave it. Every one of these reaches ggml
// as the right-hand side of a ggml_mul or ggml_add, which asserts on a length that cannot broadcast
// and on a type that does not match — so an unchecked norm weight is an abort inside the graph
// builder rather than an error. Absent is fine: the builders treat a missing weight or bias as
// "no affine term".
static bool check_vec(const jepa_model & m, const std::string & name, int64_t n) {
    ggml_tensor * t = m.get(name);
    if (!t) return true;
    bool ok = check_f32(m, name);
    if (ggml_nelements(t) != n) {
        jepa_log("jepa: tensor %s holds %" PRId64 " values, expected %" PRId64 "\n",
                 name.c_str(), ggml_nelements(t), n);
        ok = false;
    }
    return ok;
}

// Refuse a weight whose dtype this backend has no matrix-multiply kernel for. ggml's CPU mul_mat
// dispatches through type_traits_cpu[type].vec_dot, which is a null pointer for the types that are
// part of the format but not of any GEMM — the fuzzer produced an i32 ffn_up.weight and it is a
// call through address 0 inside ggml_compute_forward, not an error. ggml_backend_dev_supports_op is
// the same predicate jepa_graph_validate walks a GPU graph with; here it is asked once per distinct
// dtype in the file, on a probe mul_mat, before anything is built.
// ggml_backend_dev_supports_op is no help here: its MUL_MAT arm looks at src1 (the f32 activation)
// and says nothing about the weight's own dtype. So the rule is stated directly — every tensor in a
// jepa GGUF is a weight, a bias or a table, so every one of them is float or quantized. That
// excludes exactly the integer types, which is the class the fuzzer found.
static bool check_tensor_types(const jepa_model & m, ggml_backend_dev_t dev) {
    const bool on_cpu = !dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    bool ok = true;
    for (const auto & kv : m.tensors) {
        const ggml_type ty = kv.second->type;
        if (ty == GGML_TYPE_F32) continue;                       // always fine, and the common case
        const bool numeric = ty == GGML_TYPE_F16 || ty == GGML_TYPE_BF16 || ggml_is_quantized(ty);
        // ...and on the CPU, ask the backend itself: mul_mat dispatches through
        // type_traits_cpu[type].vec_dot, which is a null pointer for a type with no GEMM kernel and
        // is therefore a call through address 0 rather than an error.
        const bool has_kernel = !on_cpu || ggml_get_type_traits_cpu(ty)->vec_dot != nullptr;
        if (!numeric || !has_kernel) {
            jepa_log("jepa: tensor %s has dtype %s, which is not a weight dtype this engine can "
                     "compute with (f32, f16, bf16 or a quantized type)\n", kv.first.c_str(), ggml_type_name(ty));
            ok = false;
        }
    }
    return ok;
}

// Every tensor jepa_build_block may touch, for one block. `inner` is n_head * head_dim, which is D
// for the encoder and the head and is not for the LeWM predictor (16 heads x 64 over a 192-d model).
static bool check_block_shapes(const jepa_model & m, const std::string & p, int64_t D, int64_t inner, int64_t ffn) {
    bool ok = true;
    ok &= check_shape(m, (p + "attn_qkv.weight").c_str(), D, 3 * inner);
    ok &= check_vec(m, p + "attn_qkv.bias", 3 * inner);
    ok &= check_shape(m, (p + "attn_q.weight").c_str(), D, inner);
    ok &= check_shape(m, (p + "attn_k.weight").c_str(), D, inner);
    ok &= check_shape(m, (p + "attn_v.weight").c_str(), D, inner);
    ok &= check_vec(m, p + "attn_q.bias", inner);
    ok &= check_vec(m, p + "attn_k.bias", inner);
    ok &= check_vec(m, p + "attn_v.bias", inner);
    ok &= check_shape(m, (p + "attn_out.weight").c_str(), inner, D);
    ok &= check_vec(m, p + "attn_out.bias", D);
    ok &= check_vec(m, p + "ln1.weight", D);
    ok &= check_vec(m, p + "ln1.bias", D);
    ok &= check_vec(m, p + "ln2.weight", D);
    ok &= check_vec(m, p + "ln2.bias", D);
    ok &= check_shape(m, (p + "ffn_up.weight").c_str(), D, ffn);
    ok &= check_vec(m, p + "ffn_up.bias", ffn);
    ok &= check_shape(m, (p + "ffn_down.weight").c_str(), ffn, D);
    ok &= check_vec(m, p + "ffn_down.bias", D);
    ok &= check_vec(m, p + "ls1", D);
    ok &= check_vec(m, p + "ls2", D);
    return ok;
}

// 64-bit absolute seek / size. `fseek`+`ftell` take a `long`, which is 32-bit on Windows: the 2.5 GB
// I-JEPA ViT-H f32 file would fail to load there (tools/jepa-quantize.cpp carries the same pair).
static int jepa_seek64(FILE * f, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, (long long) offset, SEEK_SET);
#else
    return fseeko(f, (off_t) offset, SEEK_SET);
#endif
}

static int64_t jepa_file_size(FILE * f) {
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return -1;
    const int64_t n = _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0) return -1;
    const int64_t n = (int64_t) ftello(f);
#endif
    return n;
}

// Tensors the graph builders call m->require() for. require() aborts when a name is missing, which
// is the right behaviour for an internal invariant and the wrong one for a file the caller did not
// write — so the file is checked here instead, once, and refused with the missing name. Everything
// listed is unconditionally dereferenced by the builder named in the comment.
static bool require_tensors(const jepa_model & m, const char * what, std::initializer_list<const char *> names) {
    bool ok = true;
    for (const char * n : names) {
        if (!m.get(n)) {
            jepa_log("jepa: %s needs tensor '%s', which is not in %s\n", what, n, m.path.c_str());
            ok = false;
        }
    }
    return ok;
}

// A transformer block jepa_build_block can actually run: the two layer norms, the output and FFN
// projections and either a fused qkv or all three of q/k/v (jepa_build_qkv asserts on the rest).
static bool block_complete(const jepa_layer & L, const char * prefix, int i) {
    if (L.ln1_w && L.ln2_w && L.out_w && L.up_w && L.down_w && (L.qkv_w || (L.q_w && L.k_w && L.v_w))) return true;
    jepa_log("jepa: block %d is incomplete (%s%d.*): needs ln1/ln2/attn_out/ffn_up/ffn_down and "
             "attn_qkv or attn_q+attn_k+attn_v\n", i, prefix, i);
    return false;
}

jepa_model_params jepa_model_default_params(void) {
    jepa_model_params p;
    p.verbose = false;
    p.device  = jepa_device_from_env();
    return p;
}

jepa_model * jepa_model_load(const char * gguf_path, bool verbose) {
    jepa_model_params p = jepa_model_default_params();
    p.verbose = verbose;
    return jepa_model_load_ex(gguf_path, &p);
}

jepa_model * jepa_model_load_ex(const char * gguf_path, const jepa_model_params * params) {
    if (!gguf_path || !*gguf_path) {
        // Otherwise this reaches fopen(nullptr) and a "%s" of a null pointer, both undefined even
        // though glibc happens to print "(null)".
        jepa_log("jepa: jepa_model_load: no path given\n");
        return nullptr;
    }
    jepa_model_params defaults = jepa_model_default_params();
    const jepa_model_params & mp = params ? *params : defaults;
    const bool verbose = mp.verbose;
    // Resolve the device before touching the file: a bad --gpu N should fail immediately.
    ggml_backend_dev_t dev = jepa_device_get(mp.device);
    if (!dev) return nullptr;

    gguf_init_params ip;
    ip.no_alloc = true;
    ggml_context * ctx_meta = nullptr;
    ip.ctx = &ctx_meta;
    gguf_context * gg = gguf_init_from_file(gguf_path, ip);
    if (!gg) {
        jepa_log("jepa: cannot open GGUF %s\n", gguf_path);
        return nullptr;
    }

    jepa_model * m = new jepa_model();
    m->path = gguf_path;
    if (!jepa_hparams_from_gguf(gg, m->hp)) {
        gguf_free(gg);
        ggml_free(ctx_meta);
        delete m;
        return nullptr;
    }

    // Every tensor has to be inside the file BEFORE anything is allocated. gguf_init_from_file with
    // no_alloc = true validates the offsets against each other but never against the file's length,
    // so a 4 KB file may legitimately parse as one that promises terabytes of weights — and the
    // weight buffer below is allocated from those promises, not from the bytes on disk.
    {
        FILE * probe = fopen(gguf_path, "rb");
        if (!probe) {
            jepa_log("jepa: cannot open %s for tensor data\n", gguf_path);
            gguf_free(gg); ggml_free(ctx_meta); delete m;
            return nullptr;
        }
        const int64_t fsize = jepa_file_size(probe);
        fclose(probe);
        const size_t data_offset = gguf_get_data_offset(gg);
        const int64_t n_tensors = gguf_get_n_tensors(gg);
        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gg, i);
            ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            if (!t) {
                jepa_log("jepa: tensor %s missing from the metadata context\n", name);
                gguf_free(gg); ggml_free(ctx_meta); delete m;
                return nullptr;
            }
            const uint64_t offs   = (uint64_t) data_offset + gguf_get_tensor_offset(gg, i);
            const uint64_t nbytes = (uint64_t) ggml_nbytes(t);
            if (fsize < 0 || offs > (uint64_t) fsize || nbytes > (uint64_t) fsize - offs) {
                jepa_log("jepa: tensor %s claims %" PRIu64 " bytes at offset %" PRIu64 ", past the end of "
                         "%s (%" PRId64 " bytes) — the file is truncated or its header is not describing it\n",
                         name, nbytes, offs, gguf_path, fsize);
                gguf_free(gg); ggml_free(ctx_meta); delete m;
                return nullptr;
            }
        }
    }

    // tensors: ctx_meta already holds one (unallocated) ggml_tensor per GGUF tensor.
    // ggml_backend_alloc_ctx_tensors + ggml_backend_tensor_set below both dispatch through the
    // backend interface, so putting the weights on a GPU is exactly this one choice of backend
    // (docs/architecture.md "GPU backend"). Every context built from this model computes on it.
    m->device  = mp.device;
    m->dev     = mp.device >= 0 ? dev : nullptr;
    m->backend = mp.device >= 0 ? ggml_backend_dev_init(dev, nullptr) : ggml_backend_cpu_init();
    if (!m->backend) {
        jepa_log("jepa: could not initialise the %s backend\n", ggml_backend_dev_name(dev));
        gguf_free(gg); ggml_free(ctx_meta); delete m;
        return nullptr;
    }
    m->ctx_w = ctx_meta;
    m->buf_w = ggml_backend_alloc_ctx_tensors(m->ctx_w, m->backend);
    if (!m->buf_w) {
        jepa_log("jepa: ggml_backend_alloc_ctx_tensors() failed\n");
        gguf_free(gg); jepa_model_free(m);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(m->buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m->n_bytes_weights = ggml_backend_buffer_get_size(m->buf_w);

    FILE * f = fopen(gguf_path, "rb");
    if (!f) {
        jepa_log("jepa: cannot reopen %s for tensor data\n", gguf_path);
        gguf_free(gg); jepa_model_free(m);
        return nullptr;
    }
    const int64_t n_tensors = gguf_get_n_tensors(gg);
    const size_t data_offset = gguf_get_data_offset(gg);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gg, i);
        ggml_tensor * t = ggml_get_tensor(m->ctx_w, name);
        if (!t) {
            jepa_log("jepa: tensor %s missing from the metadata context\n", name);
            fclose(f); gguf_free(gg); jepa_model_free(m);
            return nullptr;
        }
        const size_t nbytes = ggml_nbytes(t);
        const size_t offs = data_offset + gguf_get_tensor_offset(gg, i);
        buf.resize(nbytes);
        if (jepa_seek64(f, offs) != 0 || fread(buf.data(), 1, nbytes, f) != nbytes) {
            jepa_log("jepa: short read for tensor %s (%zu bytes @ %zu)\n", name, nbytes, offs);
            fclose(f); gguf_free(gg); jepa_model_free(m);
            return nullptr;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
        m->tensors[name] = t;
    }
    fclose(f);
    gguf_free(gg);

    // wire up the encoder
    const jepa_enc_hparams & e = m->hp.enc;
    m->patch_embed_w = m->get("enc.patch_embed.weight");
    m->patch_embed_b = m->get("enc.patch_embed.bias");
    m->patch_embed_img_w = m->get("enc.patch_embed_img.weight");
    m->patch_embed_img_b = m->get("enc.patch_embed_img.bias");
    m->pos_embed  = m->get("enc.pos_embed");
    m->cls_token  = m->get("enc.cls_token");
    m->reg_tokens = m->get("enc.reg_tokens");
    m->mod_embed_img   = m->get("enc.mod_embed_img");
    m->mod_embed_video = m->get("enc.mod_embed_video");
    m->norm_w = m->get("enc.norm.weight");
    m->norm_b = m->get("enc.norm.bias");
    for (size_t k = 0; k < e.hier_layers.size(); k++) {
        std::string p = "enc.hier_norm." + std::to_string(k) + ".";
        m->hier_norms.emplace_back(m->get(p + "weight"), m->get(p + "bias"));
    }
    m->proj0_w = m->get("enc.proj.0.weight"); m->proj0_b = m->get("enc.proj.0.bias");
    m->proj2_w = m->get("enc.proj.2.weight"); m->proj2_b = m->get("enc.proj.2.bias");
    for (int i = 0; i < e.n_layer; i++) {
        m->enc_layers.push_back(jepa_layer_from_model(*m, "enc.blk." + std::to_string(i) + "."));
    }
    if (m->hp.pred.present) {
        for (int i = 0; i < m->hp.pred.n_layer; i++) {
            m->pred_layers.push_back(jepa_layer_from_model(*m, "pred.blk." + std::to_string(i) + "."));
        }
    }
    if (m->hp.head.present) {
        for (int i = 0; i < m->hp.head.n_pool_layers; i++) {
            m->head_layers.push_back(jepa_layer_from_model(*m, "head.blk." + std::to_string(i) + "."));
        }
    }

    // sanity checks on the dtypes and the shapes we rely on
    bool ok = check_tensor_types(*m, dev);
    const int64_t D = e.embed_dim;
    const int64_t patch_in = (int64_t) e.in_chans * e.tubelet_size * e.patch_size * e.patch_size;
    if (!m->patch_embed_w) { jepa_log("jepa: enc.patch_embed.weight missing\n"); ok = false; }
    ok &= check_shape(*m, "enc.patch_embed.weight", patch_in, D);
    ok &= check_vec(*m, "enc.patch_embed.bias", D);
    ok &= check_shape(*m, "enc.pos_embed", D);
    ok &= check_vec(*m, "enc.cls_token", D);
    ok &= check_vec(*m, "enc.norm.weight", D);
    ok &= check_vec(*m, "enc.norm.bias", D);
    ok &= check_vec(*m, "enc.mod_embed_img", D);
    ok &= check_vec(*m, "enc.mod_embed_video", D);
    ok &= check_shape(*m, "enc.reg_tokens", D);
    ok &= check_f32(*m, "enc.reg_tokens");
    ok &= check_f32(*m, "enc.pos_embed");
    ok &= check_shape(*m, "enc.proj.0.weight", D);
    ok &= check_vec(*m, "enc.proj.2.bias", D);
    if (m->proj0_w && m->proj2_w) {   // the LeWM projector: D -> hidden -> D
        ok &= check_vec(*m, "enc.proj.0.bias", m->proj0_w->ne[1]);
        ok &= check_shape(*m, "enc.proj.2.weight", m->proj0_w->ne[1], D);
    }
    for (int i = 0; i < e.n_layer && ok; i++) {
        std::string p = "enc.blk." + std::to_string(i) + ".";
        if (!block_complete(m->enc_layers[i], "enc.blk.", i)) { ok = false; break; }
        // the encoder's head_dim is embed_dim / n_head, so its attention inner width is D
        ok &= check_block_shapes(*m, p, D, D, e.ffn_dim);
    }
    if (e.cls_token && !m->cls_token) { jepa_log("jepa: cls_token=true but enc.cls_token missing\n"); ok = false; }
    if (e.n_registers > 0 && (!m->reg_tokens || m->reg_tokens->ne[1] != e.n_registers)) {
        // jepa_build_encoder_image asserts on this rather than returning, so it has to be a load error
        jepa_log("jepa: jepa.enc.n_registers is %d but enc.reg_tokens is %s\n", e.n_registers,
                 m->reg_tokens ? "a different shape" : "missing");
        ok = false;
    }
    if (m->pos_embed && !kv_range("enc.pos_embed rows", m->pos_embed->ne[1], 1, JEPA_LIMIT_N_TOKENS)) ok = false;

    // Predictor and head: the graph builders reach for these through m->require(), which aborts.
    // Check them here so an incomplete file is a refused load rather than a crash on first use.
    const jepa_pred_hparams & p = m->hp.pred;
    if (p.present && p.kind == "masked") {
        ok &= require_tensors(*m, "the masked predictor", { "pred.mask_tokens", "pred.norm.weight", "pred.proj.weight" });
        if (!m->get("pred.embed.weight") && !(m->get("pred.embed.0.weight") && m->get("pred.embed.2.weight"))) {
            jepa_log("jepa: the masked predictor needs pred.embed.weight, or pred.embed.0.weight and "
                     "pred.embed.2.weight for the 2-layer form\n");
            ok = false;
        }
        if (p.modality_embed && !m->get("pred.mod_embed_video")) {
            jepa_log("jepa: jepa.pred.modality_embed is set but pred.mod_embed_video is missing\n");
            ok = false;
        }
    }
    if (p.present && p.kind == "lewm") {
        ok &= require_tensors(*m, "the LeWM predictor",
                              { "pred.action_embed.0.weight", "pred.action_embed.2.weight",
                                "pred.norm.weight", "pred.proj.0.weight", "pred.proj.2.weight" });
    }
    if (p.present) {
        const int64_t Dp = p.embed_dim;
        const int64_t inner_p = (int64_t) p.n_head * p.head_dim_eff();
        for (int i = 0; i < p.n_layer && ok; i++) {
            const std::string q = "pred.blk." + std::to_string(i) + ".";
            if (!block_complete(m->pred_layers[i], "pred.blk.", i)) { ok = false; break; }
            if (p.kind == "lewm" && !m->pred_layers[i].adaln_w) {
                jepa_log("jepa: the LeWM predictor needs tensor '%sadaln.weight', which is not in %s\n",
                         q.c_str(), m->path.c_str());
                ok = false;
                break;
            }
            ok &= check_block_shapes(*m, q, Dp, inner_p, p.ffn_dim);
            ok &= check_shape(*m, (q + "adaln.weight").c_str(), Dp, 6 * Dp);
            ok &= check_vec(*m, q + "adaln.bias", 6 * Dp);
        }
        ok &= check_vec(*m, "pred.norm.weight", Dp);
        ok &= check_vec(*m, "pred.norm.bias", Dp);
        ok &= check_shape(*m, "pred.mask_tokens", Dp);
        ok &= check_f32(*m, "pred.mask_tokens");
        // jepa_build_predictor_masked views row `mask_index % n_mask_tokens` out of this tensor, so
        // a declared count larger than the tensor has is a view past its end.
        if (ggml_tensor * mt = m->get("pred.mask_tokens")) {
            if (p.n_mask_tokens > mt->ne[1]) {
                jepa_log("jepa: jepa.pred.n_mask_tokens is %d but pred.mask_tokens holds %" PRId64 " rows\n",
                         p.n_mask_tokens, mt->ne[1]);
                ok = false;
            }
        }
        ok &= check_vec(*m, "pred.mod_embed_img", Dp);
        ok &= check_vec(*m, "pred.mod_embed_video", Dp);
        ok &= check_shape(*m, "pred.pos_embed", Dp);
        ok &= check_f32(*m, "pred.pos_embed");
        // jepa_build_lewm takes the first T rows of this table, T <= jepa.pred.n_frames
        if (ggml_tensor * pe = m->get("pred.pos_embed")) {
            if (p.n_frames > pe->ne[1]) {
                jepa_log("jepa: jepa.pred.n_frames is %d but pred.pos_embed holds %" PRId64 " rows\n",
                         p.n_frames, pe->ne[1]);
                ok = false;
            }
        }
        // The action MLP has to land on the predictor width: action_dim -> hidden -> embed_dim,
        // and its second layer feeds the adaLN matmul, which asserts on a mismatch.
        if (ggml_tensor * a0 = m->get("pred.action_embed.0.weight")) {
            ok &= check_vec(*m, "pred.action_embed.0.bias", a0->ne[1]);
            ok &= check_shape(*m, "pred.action_embed.2.weight", a0->ne[1], Dp);
            ok &= check_vec(*m, "pred.action_embed.2.bias", Dp);
        }
        ok &= check_shape(*m, "pred.embed.weight", D, Dp);
        ok &= check_vec(*m, "pred.embed.bias", Dp);
        ok &= check_shape(*m, "pred.proj.weight", Dp, p.out_dim);
        ok &= check_vec(*m, "pred.proj.bias", p.out_dim);
        if (p.action_dim > 0) ok &= check_shape(*m, "pred.action_embed.0.weight", p.action_dim);
    }
    if (m->hp.head.present && m->hp.head.kind == "attentive_pool") {
        ok &= require_tensors(*m, "the attentive-pool head",
                              { "head.query", "head.xattn.ln_kv.weight", "head.xattn.q.weight",
                                "head.xattn.k.weight", "head.xattn.v.weight", "head.xattn.ln2.weight",
                                "head.xattn.ffn_up.weight", "head.xattn.ffn_down.weight", "head.cls.weight" });
        for (int i = 0; i < m->hp.head.n_pool_layers && ok; i++) {
            const std::string q = "head.blk." + std::to_string(i) + ".";
            if (!block_complete(m->head_layers[i], "head.blk.", i)) { ok = false; break; }
            ok &= check_block_shapes(*m, q, D, D, e.ffn_dim);
        }
        // the cross-attention pooler runs at the encoder width throughout
        ok &= check_vec(*m, "head.query", D);
        ok &= check_vec(*m, "head.xattn.ln_kv.weight", D);
        ok &= check_vec(*m, "head.xattn.ln_kv.bias", D);
        ok &= check_vec(*m, "head.xattn.ln2.weight", D);
        ok &= check_vec(*m, "head.xattn.ln2.bias", D);
        ok &= check_shape(*m, "head.xattn.q.weight", D, D);
        ok &= check_shape(*m, "head.xattn.k.weight", D, D);
        ok &= check_shape(*m, "head.xattn.v.weight", D, D);
        ok &= check_shape(*m, "head.xattn.ffn_up.weight", D);
        ok &= check_shape(*m, "head.cls.weight", D);
    }
    if (!ok) { jepa_model_free(m); return nullptr; }

    if (verbose) {
        if (m->device >= 0) {
            size_t dfree = 0, dtotal = 0;
            ggml_backend_dev_memory(m->dev, &dfree, &dtotal);
            jepa_log("jepa: weights on %s (%s), %.1f MiB of %.0f MiB free\n",
                     ggml_backend_dev_name(m->dev), ggml_backend_dev_description(m->dev),
                     m->n_bytes_weights / (1024.0 * 1024.0), dtotal / (1024.0 * 1024.0));
        }
        jepa_log("jepa: loaded %s: family=%s name='%s' ftype=%s D=%d L=%d H=%d ffn=%d patch=%d tubelet=%d img=%d "
                 "cls=%d pos=%s tensors=%zu weights=%.1f MiB\n",
                 gguf_path, m->hp.family_str.c_str(), m->hp.name.c_str(), jepa_file_type_name(m->hp.file_type),
                 e.embed_dim, e.n_layer, e.n_head, e.ffn_dim, e.patch_size, e.tubelet_size, e.img_size,
                 (int) e.cls_token, e.pos_type_str.c_str(), m->tensors.size(), m->n_bytes_weights / (1024.0 * 1024.0));
    }
    return m;
}

void jepa_model_free(jepa_model * m) {
    if (!m) return;
    if (m->buf_w)   ggml_backend_buffer_free(m->buf_w);
    if (m->ctx_w)   ggml_free(m->ctx_w);
    if (m->backend) ggml_backend_free(m->backend);
    delete m;
}

// ---------------------------------------------------------------------------------------------
// public introspection
// ---------------------------------------------------------------------------------------------
const char * jepa_model_family(const jepa_model * m)   { return m->hp.family_str.c_str(); }
int  jepa_model_embed_dim(const jepa_model * m)        { return m->hp.enc.embed_dim; }
int  jepa_model_patch_size(const jepa_model * m)       { return m->hp.enc.patch_size; }
int  jepa_model_tubelet_size(const jepa_model * m)     { return m->hp.enc.tubelet_size; }
int  jepa_model_img_size(const jepa_model * m)         { return m->hp.enc.img_size; }
int  jepa_model_n_frames(const jepa_model * m)         { return m->hp.enc.n_frames; }
int  jepa_model_n_layer(const jepa_model * m)          { return m->hp.enc.n_layer; }
int  jepa_model_n_head(const jepa_model * m)           { return m->hp.enc.n_head; }
bool jepa_model_has_cls(const jepa_model * m)          { return m->hp.enc.cls_token; }
int  jepa_model_n_registers(const jepa_model * m)      { return m->hp.enc.n_registers; }
int  jepa_model_n_prefix_tokens(const jepa_model * m)  { return (m->hp.enc.cls_token ? 1 : 0) + m->hp.enc.n_registers; }
bool jepa_model_has_predictor(const jepa_model * m)    { return m->hp.pred.present; }
bool jepa_model_has_head(const jepa_model * m)         { return m->hp.head.present; }
bool jepa_model_has_projector(const jepa_model * m)    { return m->proj0_w != nullptr && m->proj2_w != nullptr; }
int  jepa_model_n_classes(const jepa_model * m)        { return m->hp.head.n_classes; }
const char * jepa_model_label(const jepa_model * m, int idx) {
    if (idx < 0 || idx >= (int) m->hp.head.labels.size()) return nullptr;
    return m->hp.head.labels[idx].c_str();
}
const char * jepa_model_name(const jepa_model * m)     { return m->hp.name.c_str(); }
int  jepa_model_file_type(const jepa_model * m)        { return (int) m->hp.file_type; }
const char * jepa_model_file_type_name(const jepa_model * m) { return jepa_file_type_name(m->hp.file_type); }
size_t jepa_model_n_bytes(const jepa_model * m)        { return m->n_bytes_weights; }
int  jepa_model_device(const jepa_model * m)           { return m ? m->device : -1; }
bool jepa_model_is_gpu(const jepa_model * m)           { return m && m->device >= 0; }
const char * jepa_model_device_name(const jepa_model * m) {
    if (!m) return nullptr;
    if (m->dev) return ggml_backend_dev_name(m->dev);
    ggml_backend_dev_t cpu = jepa_device_get(-1);
    return cpu ? ggml_backend_dev_name(cpu) : "CPU";
}
