// test-parity: run a jepa.cpp GGUF against the PyTorch golden dumps of tests/fixtures/ref/<model>/.
//
//   test-parity MODEL.gguf REF_DIR [--threads N] [--no-flash] [--kv-f32] [--json out.json]
//               [--pre manifest|model] [--rgb-dir DIR] [--samples a,b,c] [--quiet]
//
// Per sample it
//   (a) feeds the stored `input` tensor (bypassing our preprocessing) and compares `last_hidden_state`
//       (per-token cosine mean/min, max|a-b|, rel_max = max|a-b| / max|b|) plus `pooled_mean` / `cls` / `emb`
//       (and `emb_seq` for the LeWM sequence sample) when present;
//   (b) decodes the media file itself (or reads <rgb-dir>/<sample>.rgb.npy, a HWC uint8 dump of the reference
//       decoder), runs our preprocessor, reports the input diff (max abs, fraction of exactly equal values, number of
//       values off by more than one uint8 level) and the same output metrics again.
// Video samples (V-JEPA 2 / 2.1): the stored `input` is 5-D and its layout comes from the manifest
// (NTCHW for the HF processor dumps, NCTHW for V-JEPA 2.1); it is transposed to the NCTHW jepa_input
// here and fed as ONE clip.  Their own-preprocessing pass runs on the stored `frames_u8` (THWC uint8,
// the exact frames the reference processor saw), so no video decoder is needed.  Classifiers also get
// `pooled` (attentive-pooler output), `logits` and top-1 / top-5 agreement against `top5_idx`.
// Thresholds are table-driven, per model FAMILY (image ViTs vs the V-JEPA 2 video encoders) and per
// file-type TIER (f32 / f16 / q8 / low-bit), see POLICY below and docs/parity.md: the image families
// keep the hard every-token bars, only the video token maps are judged on the median cosine, and
// files below 8 bits per weight are reported but only gated on the derived tensors + top-1.
// They are applied to the stored-input run; the own-preprocessing run must meet the cosine threshold.
// Exit status 1 on any failure.
#include "jepa.h"
#include "jepa-internal.h"
#include "npy.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

struct metrics {
    double cos_mean = 0, cos_med = 0, cos_min = 0, max_abs = 0, rel_max = 0, ref_max = 0;
    int64_t n_rows = 0;
    // where the worst row is, how big it is, and how far the low tail reaches
    int64_t worst_row = -1;
    double  worst_row_norm = 0, mean_row_norm = 0;
    int64_t n_lt_999 = 0, n_lt_99 = 0;
    bool valid = false;
};

// rows x dim, cosine per row
static metrics compare(const float * a, const float * b, int64_t rows, int64_t dim) {
    metrics m;
    m.n_rows = rows;
    m.cos_min = 1.0;
    double cos_sum = 0, norm_sum = 0;
    std::vector<double> cs((size_t) rows);
    for (int64_t r = 0; r < rows; r++) {
        double dot = 0, na = 0, nb = 0;
        for (int64_t d = 0; d < dim; d++) {
            const double x = a[r * dim + d], y = b[r * dim + d];
            dot += x * y; na += x * x; nb += y * y;
            const double diff = std::fabs(x - y);
            if (diff > m.max_abs) m.max_abs = diff;
            if (std::fabs(y) > m.ref_max) m.ref_max = std::fabs(y);
        }
        const double c = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        cs[(size_t) r] = c;
        cos_sum += c;
        norm_sum += std::sqrt(nb);
        if (c < 0.999) m.n_lt_999++;
        if (c < 0.99)  m.n_lt_99++;
        if (c < m.cos_min || m.worst_row < 0) { m.cos_min = c; m.worst_row = r; m.worst_row_norm = std::sqrt(nb); }
    }
    m.cos_mean = rows ? cos_sum / rows : 0;
    m.mean_row_norm = rows ? norm_sum / rows : 0;
    if (rows) {
        std::nth_element(cs.begin(), cs.begin() + rows / 2, cs.end());
        m.cos_med = cs[rows / 2];
    }
    m.rel_max = m.max_abs / (m.ref_max + 1e-30);
    m.valid = true;
    return m;
}

// ==============================================================================================
// Thresholds — one table, indexed by [model family class][file-type tier], applied to the
// stored-input pass.  Mirrored word for word in docs/parity.md ("Thresholds").
//
// Why per family: the image ViTs (I-JEPA / LeJEPA-hfvit / LeWM) reproduce the reference on EVERY
// token, so they keep the hard bars; only the V-JEPA 2 video encoders develop a long low-cosine
// tail at f16/q8_0 while everything downstream stays exact, so their token map is gated on the
// median instead.  Relaxing the video bars globally (as an earlier revision did) would have let a
// real image-side regression through: I-JEPA f16 would pass at a token map mean of 0.99 when the
// measured value is 0.999984, and LeJEPA q8_0 at 0.95 when it measures 0.999263.
//
// Two tensor classes per cell:
//   * token maps (`last_hidden_state`, policy.lhs) — for the video families judged on the MEDIAN
//     per-token cosine plus a weak mean floor, with no worst-token bound at f16/q8_0.  Two
//     independent effects produce that low tail while everything downstream stays exact:
//       - weight quantisation amplified by low-variance tokens (docs/quantization.md: I-JEPA q8_0
//         worst token 0.43, worst sample mean 0.988, pooled >= 0.9999);
//       - ggml rounding the *activations* to F16 inside every f16 mul_mat (llamafile's AVX-512 sgemm
//         only has an F16xF16 kernel).  On V-JEPA 2 ViT-L (SSv2 bowling_f16) that is mean 0.9971 /
//         worst token 0.51, while the f32 file of the same model is exact (1.000000, rel 7.5e-4) and
//         the numpy spec re-run with F16 activations reproduces the C++ numbers to four digits
//         (mean 0.99718, worst 0.5576) -- docs/parity.md.
//     The median (>= 0.999 at f16, >= 0.99 at q8_0 on every fixture) stays tight and still collapses
//     for a real graph bug: a wrong RoPE layout alone gives ~0.63/0.91 on *every* token.
//   * derived single-row tensors (`pooled_mean`, `pooled`, `cls`, `emb`, `logits`, policy.derived):
//     the tail averages out, so these keep tight bars -- they are what users consume.  The bars are
//     set just under the worst fixture value: f16 0.9995 (SSv2 pooler output 0.999897), q8_0 0.995
//     (SSv2 q8_0 pooler 0.996645, its logits 0.998501 with top-1/top-5 still exact) for video, and
//     0.999 mean / 0.98 worst row for the image families (LeWM q8_0 emb_seq 0.999895).
//
//   image f32:  tokens mean & worst >= 0.9999, rel_max <= REL (see rel_bound); derived >= 0.9999
//   image f16:  tokens mean >= 0.9999, worst >= 0.99;         derived >= 0.9995
//   image q8:   tokens mean >= 0.98;                          derived mean >= 0.999, worst >= 0.98
//   video f32:  tokens mean & median & worst >= 0.9999, rel_max <= REL; derived >= 0.9999
//   video f16:  tokens median >= 0.999, mean >= 0.99;         derived >= 0.9995
//   video q8:   tokens median >= 0.99,  mean >= 0.95;         derived >= 0.995
//   low-bit (< 8 bits/weight: q4_*, q5_*, q6_k, iq*): advisory — the token map is reported but not
//     gated, only the derived tensors (>= 0.99) and the classifier top-1 are (docs/quantization.md
//     recommends q8_0 as the lowest parity-grade quantisation).
// Classifiers additionally have to reproduce the reference top-1 exactly and 4 of its top-5 (top-1
// only in the low-bit tier).
// The own-preprocessing pass uses the same rules with no bar stricter than 0.99: it additionally
// carries JPEG-decoder differences (stb_image vs PIL) unless --rgb-dir provides the reference pixels
// (video samples run on the stored frames_u8 and come out bit-exact).
struct thresholds { double min_mean; double min_med; double min_min; double max_rel; };  // <= 0: no gate

struct policy {
    thresholds lhs;       // last_hidden_state (the token map)
    thresholds derived;   // pooled_mean / pooled / cls / emb / logits
    bool gate_top5;       // require >= 4 of the reference top-5 (top-1 is required in every tier)
    bool advisory;        // low-bit tier: print the token map, do not gate it
};

enum fam_class { FAM_IMAGE = 0, FAM_VIDEO = 1, FAM_COUNT = 2 };
enum dt_tier   { TIER_F32 = 0, TIER_F16 = 1, TIER_Q8 = 2, TIER_LOWBIT = 3, TIER_COUNT = 4 };

static const char * const TIER_NAME[TIER_COUNT] = { "f32", "f16", "q8", "low-bit" };
static const char * const FAM_NAME[FAM_COUNT]   = { "image", "video" };

static const policy POLICY[FAM_COUNT][TIER_COUNT] = {
    // ---- image families: ijepa / hfvit (LeJEPA) / lewm ---------------------------------------
    { /* f32     */ { {0.9999, -1.0, 0.9999, 1e-3}, {0.9999, -1.0, 0.9999, 1e-3}, true,  false },
      /* f16     */ { {0.9999, -1.0, 0.99,   -1.0}, {0.9995, -1.0, -1.0,   -1.0}, true,  false },
      /* q8      */ { {0.98,   -1.0, -1.0,   -1.0}, {0.999,  -1.0, 0.98,   -1.0}, true,  false },
      /* low-bit */ { {-1.0,   -1.0, -1.0,   -1.0}, {0.99,   -1.0, -1.0,   -1.0}, false, true  } },
    // ---- video families: vjepa2 / vjepa2_1 (and vjepa v1) ------------------------------------
    { /* f32     */ { {0.9999, 0.9999, 0.9999, 1e-3}, {0.9999, -1.0, 0.9999, 1e-3}, true,  false },
      /* f16     */ { {0.99,   0.999,  -1.0,   -1.0}, {0.9995, -1.0, -1.0,   -1.0}, true,  false },
      /* q8      */ { {0.95,   0.99,   -1.0,   -1.0}, {0.995,  -1.0, -1.0,   -1.0}, true,  false },
      /* low-bit */ { {-1.0,   -1.0,   -1.0,   -1.0}, {0.99,   -1.0, -1.0,   -1.0}, false, true  } },
};

static fam_class fam_class_of(const char * family) {
    if (!family) return FAM_IMAGE;
    const std::string f = family;
    return (f == "vjepa" || f == "vjepa2" || f == "vjepa2_1") ? FAM_VIDEO : FAM_IMAGE;
}

// general.file_type carries a GGML_FTYPE_* value (see src/jepa-gguf.cpp jepa_file_type_name).
// Anything below 8 bits per stored weight (q4_*, q5_*, q6_k, iq*) lands in the advisory tier.
static dt_tier tier_of(int ftype) {
    if (ftype == GGML_FTYPE_ALL_F32) return TIER_F32;
    if (ftype == GGML_FTYPE_MOSTLY_F16 || ftype == GGML_FTYPE_MOSTLY_BF16) return TIER_F16;
    const ggml_type t = ggml_ftype_to_ggml_type((ggml_ftype) ftype);
    if (t == GGML_TYPE_COUNT || ggml_blck_size(t) <= 0) return TIER_LOWBIT;   // unknown: be lenient
    const double bits = 8.0 * (double) ggml_type_size(t) / (double) ggml_blck_size(t);
    return bits >= 8.0 ? TIER_Q8 : TIER_LOWBIT;
}

// f32 rel_max bound, widened with the sequence length.  max|a-b| over a token map grows like the
// accumulated round-off of the longest reduction in the graph (~sqrt(N) for attention over N
// tokens): the 8192-token V-JEPA 2 clip reaches 1.22e-3 at cosine 1.000000 on every token, and the
// V-JEPA 2.1 predictor 1.26e-3 at 4608 context rows.  The 2048-token reference point is the
// 16-frame ViT-L clip (7.5e-4), so the bound is 1e-3 there and only loosens beyond it.
static double rel_bound(double base, int64_t rows) {
    if (base <= 0) return -1.0;
    return std::max(base, base * std::sqrt((double) rows / 2048.0));
}

static std::string bars_to_string(const thresholds & t) {
    std::string s;
    char buf[96];
    auto add = [&](const char * what, double v) {
        if (v <= 0) return;
        snprintf(buf, sizeof(buf), "%s%s >= %g", s.empty() ? "" : ", ", what, v);
        s += buf;
    };
    add("mean", t.min_mean);
    add("median", t.min_med);
    add("worst", t.min_min);
    if (t.max_rel > 0) {
        snprintf(buf, sizeof(buf), "%srel <= %g*max(1,sqrt(N/2048))", s.empty() ? "" : ", ", t.max_rel);
        s += buf;
    }
    return s.empty() ? std::string("not gated") : s;
}

static bool passes(const metrics & m, const thresholds & t, bool own) {
    if (!m.valid) return true;
    // the own-preprocessing pass carries decoder noise on top, so nothing is stricter than 0.99 there
    const double min_mean = own ? std::min(t.min_mean, 0.99) : t.min_mean;
    const double min_med  = own ? std::min(t.min_med,  0.99) : t.min_med;
    if (t.min_mean > 0 && m.cos_mean < min_mean) return false;
    if (t.min_med  > 0 && m.cos_med  < min_med)  return false;
    if (!own && t.min_min > 0 && m.cos_min < t.min_min) return false;
    if (!own && t.max_rel > 0 && m.rel_max > rel_bound(t.max_rel, m.n_rows)) return false;
    return true;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static long peak_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return atol(line.c_str() + 6);
    }
    return -1;
}

static jepa_preprocess_params params_from_manifest(const json & pre, const jepa_preprocess_params & base, bool * ok) {
    jepa_preprocess_params p = base;
    *ok = true;
    if (pre.contains("resize") && pre["resize"].is_object()) {
        const json & r = pre["resize"];
        if (r.contains("shortest_edge")) {
            p.resize_mode = JEPA_RESIZE_SHORTEST_EDGE;
            p.resize_short = r["shortest_edge"].get<int>();
        } else if (r.contains("height")) {
            p.resize_mode = JEPA_RESIZE_SQUASH;
            p.resize_short = r["height"].get<int>();
            if (r.contains("width") && r["width"].get<int>() != p.resize_short) *ok = false;
        }
        if (r.contains("resample")) p.resample = r["resample"].get<std::string>() == "bicubic" ? JEPA_RESAMPLE_BICUBIC : JEPA_RESAMPLE_BILINEAR;
    }
    if (pre.contains("center_crop") && pre["center_crop"].is_number()) p.crop = pre["center_crop"].get<int>();
    else p.crop = p.resize_short;
    if (pre.contains("mean")) for (int i = 0; i < 3; i++) p.mean[i] = pre["mean"][i].get<float>();
    if (pre.contains("std"))  for (int i = 0; i < 3; i++) p.std[i]  = pre["std"][i].get<float>();
    if (pre.contains("rescale") && pre["rescale"].is_number()) p.rescale = pre["rescale"].get<float>();
    // HF torchvision processors fuse rescale into mean/std; dump_reference's own helper divides by 255 first
    p.fused_norm = pre.contains("processor");
    return p;
}

static std::string pre_to_string(const jepa_preprocess_params & p) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s %d->crop %d %s norm=%s mean=(%.3f,%.3f,%.3f) std=(%.3f,%.3f,%.3f)",
             p.resize_mode == JEPA_RESIZE_SQUASH ? "squash" : "shortest_edge", p.resize_short, p.crop,
             p.resample == JEPA_RESAMPLE_BICUBIC ? "bicubic" : "bilinear", p.fused_norm ? "fused" : "sequential",
             p.mean[0], p.mean[1], p.mean[2], p.std[0], p.std[1], p.std[2]);
    return buf;
}

struct sample_result {
    std::string name;
    json row;
    bool ok = true;
};

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf REF_DIR [--threads N] [--no-flash] [--kv-f32] [--json out.json] "
                        "[--pre manifest|model] [--rgb-dir DIR] [--samples a,b] [--quiet]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], ref_dir = argv[2];
    jepa_context_params cp = jepa_context_default_params();
    std::string json_out, pre_mode = "manifest", rgb_dir, sample_filter;
    bool quiet = false;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--threads" || a == "-t") cp.n_threads = atoi(next());
        else if (a == "--no-flash") cp.use_flash_attn = false;
        else if (a == "--kv-f32") cp.flash_kv = JEPA_KV_F32;
        else if (a == "--kv-f16") cp.flash_kv = JEPA_KV_F16;
        else if (a == "--json") json_out = next();
        else if (a == "--pre") pre_mode = next();
        else if (a == "--rgb-dir") rgb_dir = next();
        else if (a == "--samples") sample_filter = next();
        else if (a == "--quiet") quiet = true;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }

    json manifest;
    {
        std::ifstream f(ref_dir + "/manifest.json");
        if (!f) { fprintf(stderr, "cannot open %s/manifest.json\n", ref_dir.c_str()); return 2; }
        f >> manifest;
    }
    const std::string media_dir = ref_dir + "/../../media";

    double t0 = now_ms();
    jepa_model * model = jepa_model_load(model_path.c_str(), !quiet);
    if (!model) return 2;
    const double load_ms = now_ms() - t0;
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) return 2;
    const int ftype = jepa_model_file_type(model);
    const fam_class fam = fam_class_of(jepa_model_family(model));
    const dt_tier   tier = tier_of(ftype);
    const policy &  pol = POLICY[fam][tier];
    const thresholds thr = pol.derived;
    const thresholds thr_lhs = pol.lhs;

    jepa_preprocess_params pre_model = jepa_preprocess_default_params(model);
    bool pre_ok = true;
    jepa_preprocess_params pre_ref = manifest.contains("preprocessing") ? params_from_manifest(manifest["preprocessing"], pre_model, &pre_ok) : pre_model;
    const jepa_preprocess_params & pre = pre_mode == "model" ? pre_model : pre_ref;
    const bool pre_differs = memcmp(&pre_model.mean, &pre_ref.mean, sizeof(pre_model.mean)) != 0 || memcmp(&pre_model.std, &pre_ref.std, sizeof(pre_model.std)) != 0 ||
                             pre_model.resize_short != pre_ref.resize_short || pre_model.crop != pre_ref.crop ||
                             pre_model.resample != pre_ref.resample || pre_model.resize_mode != pre_ref.resize_mode;

    const char * kv_name = cp.flash_kv == JEPA_KV_F16 ? "f16" : cp.flash_kv == JEPA_KV_F32 ? "f32" : (ftype == 0 ? "auto(f32)" : "auto(f16)");
    printf("model: %s (%s, %s, %d layers, D=%d) | ref: %s | threads %d | flash %s | kv %s\n",
           jepa_model_name(model), jepa_model_family(model), jepa_model_file_type_name(model), jepa_model_n_layer(model),
           jepa_model_embed_dim(model), manifest.value("model", "?").c_str(), jepa_context_n_threads(ctx),
           cp.use_flash_attn ? "yes" : "no", kv_name);
    printf("thresholds [%s family, %s tier]: token map %s | derived %s | top-1 exact%s\n",
           FAM_NAME[fam], TIER_NAME[tier], bars_to_string(thr_lhs).c_str(), bars_to_string(thr).c_str(),
           pol.gate_top5 ? ", top-5 >= 4/5" : "");
    if (pol.advisory) {
        printf("NOTE: %s is below the recommended quantization for parity (q8_0, docs/quantization.md): the token "
               "map is reported but not gated; only the derived tensors and the top-1 label are.\n",
               jepa_model_file_type_name(model));
    }
    printf("preprocess (%s): %s\n", pre_mode == "model" ? "jepa.pre.* of the GGUF" : "reference manifest", pre_to_string(pre).c_str());
    if (pre_differs) {
        printf("NOTE: the GGUF jepa.pre.* pipeline differs from the reference manifest's: %s\n", pre_to_string(pre_mode == "model" ? pre_ref : pre_model).c_str());
    }

    printf("\n%-20s %-6s %8s %8s %8s %9s %9s %8s %8s %8s %5s %5s %9s %7s %9s\n", "sample", "input", "cos_mean", "cos_med",
           "cos_min", "max_abs", "rel_max", "pool", "cls/emb", "logits", "top1", "top5", "in_maxd", "in_eq%", "ms");

    std::vector<sample_result> results;
    bool all_ok = true;
    double sum_ms = 0, sum_ref_s = 0;
    int n_timed = 0;
    int64_t tokens_per_item = 0;
    for (const json & s : manifest["samples"]) {
        const std::string name = s["name"].get<std::string>();
        if (!sample_filter.empty() && ("," + sample_filter + ",").find("," + name + ",") == std::string::npos) continue;
        const json & T = s["tensors"];
        if (!T.contains("input")) continue;
        sample_result res;
        res.name = name;
        res.row["sample"] = name;

        // ---- the stored input tensor: [N, C, H, W] (images) or 5-D video, NTCHW or NCTHW per the
        // manifest's layout field. jepa_input is NCTHW, so NTCHW samples are transposed here.
        npy::Array in_npy = npy::load(ref_dir + "/" + T["input"]["file"].get<std::string>());
        const std::string in_layout = T["input"].value("layout", "");
        int N = 0, C = 0, F = 1, H = 0, W = 0;   // F = frames
        bool ntchw = false;
        if (in_npy.shape.size() == 4 && in_npy.shape[1] == 3) {
            N = (int) in_npy.shape[0]; C = 3; F = 1; H = (int) in_npy.shape[2]; W = (int) in_npy.shape[3];
        } else if (in_npy.shape.size() == 5) {
            ntchw = in_layout.rfind("NTCHW", 0) == 0;
            N = (int) in_npy.shape[0];
            F = (int) in_npy.shape[ntchw ? 1 : 2];
            C = (int) in_npy.shape[ntchw ? 2 : 1];
            H = (int) in_npy.shape[3]; W = (int) in_npy.shape[4];
            if (C != 3) { printf("%-20s skipped: input has %d channels\n", name.c_str(), C); continue; }
        } else {
            printf("%-20s skipped: input shape is not [N,3,H,W] / 5-D video\n", name.c_str());
            continue;
        }
        std::vector<float> in_ref = in_npy.to_f32();
        if ((int64_t) in_ref.size() != (int64_t) N * C * F * H * W) {
            printf("%-20s ERROR: input npy element count mismatch\n", name.c_str()); all_ok = false; continue;
        }
        if (ntchw) {   // [N,T,C,H,W] -> [N,C,T,H,W]
            std::vector<float> t((size_t) N * C * F * H * W);
            const size_t plane = (size_t) H * W;
            for (int b = 0; b < N; b++) for (int f = 0; f < F; f++) for (int c = 0; c < C; c++) {
                memcpy(t.data() + (((size_t) b * C + c) * F + f) * plane,
                       in_ref.data() + (((size_t) b * F + f) * C + c) * plane, plane * sizeof(float));
            }
            in_ref.swap(t);
        }
        const int64_t D = jepa_model_embed_dim(model);
        const int n_classes = jepa_model_n_classes(model);

        // reference tensors (optional)
        auto load_opt = [&](const char * key, std::vector<float> & out, std::vector<int64_t> & shape) -> bool {
            if (!T.contains(key)) return false;
            npy::Array a = npy::load(ref_dir + "/" + T[key]["file"].get<std::string>());
            out = a.to_f32(); shape = a.shape; return true;
        };
        std::vector<float> lhs_ref, pool_ref, cls_ref, emb_ref, emb_seq_ref, hpool_ref, logits_ref, top5_ref;
        std::vector<int64_t> sh;
        const bool has_lhs = load_opt("last_hidden_state", lhs_ref, sh);
        const bool has_pool = load_opt("pooled_mean", pool_ref, sh);
        const bool has_cls = load_opt("cls", cls_ref, sh);
        const bool has_emb = load_opt("emb", emb_ref, sh);
        const bool has_emb_seq = load_opt("emb_seq", emb_seq_ref, sh);
        const bool has_hpool = load_opt("pooled", hpool_ref, sh);            // attentive-pooler output
        const bool has_logits = load_opt("logits", logits_ref, sh);
        const bool has_top5 = load_opt("top5_idx", top5_ref, sh);

        // Guard against mismatched reference dirs: never read past a smaller ref buffer.
        auto ref_size_ok = [&](const std::vector<float> & v, int64_t rows, int64_t dim, const char * what) {
            if ((int64_t) v.size() == rows * dim) return true;
            printf("%-20s ERROR: reference '%s' has %zu elements, expected %lld x %lld — wrong ref dir for this model?\n",
                   name.c_str(), what, v.size(), (long long) rows, (long long) dim);
            all_ok = false;
            return false;
        };

        // run: (a) stored input, (b) own preprocessing
        for (int pass = 0; pass < 2; pass++) {
            const bool own = pass == 1;
            std::vector<float> in_own;
            double in_maxd = 0; double in_eq = 0; int64_t in_off1 = 0;
            if (own) {
                bool fail = false;
                if (T.contains("frames_u8")) {
                    // video sample: our preprocessor runs on the same sampled frames the reference used
                    npy::Array fr = npy::load(ref_dir + "/" + T["frames_u8"]["file"].get<std::string>());
                    if (fr.shape.size() != 4 || fr.shape[0] != F || fr.shape[3] != 3 || fr.dtype != "|u1") {
                        printf("%-20s own: frames_u8 is not [%d,h,w,3] uint8, skipped\n", name.c_str(), F);
                        break;
                    }
                    const int fh = (int) fr.shape[1], fw = (int) fr.shape[2];
                    std::vector<const uint8_t *> ptr(F);
                    for (int f = 0; f < F; f++) ptr[f] = fr.bytes.data() + (size_t) f * fh * fw * 3;
                    int oh = 0, ow = 0;
                    float * x = jepa_preprocess_frames_rgb_ex(&pre, ptr.data(), F, fh, fw, &oh, &ow);
                    if (!x || oh != H || ow != W) {
                        printf("%-20s own preprocessing produced %dx%d, reference is %dx%d\n", name.c_str(), oh, ow, H, W);
                        if (x) jepa_free(x);
                        fail = true;
                    } else {
                        in_own.assign(x, x + (size_t) 3 * F * H * W);
                        jepa_free(x);
                    }
                } else {
                    std::vector<std::string> media;
                    if (s["media"].is_string()) media.push_back(s["media"].get<std::string>());
                    else for (const json & mm : s["media"]) media.push_back(mm.get<std::string>());
                    if ((int) media.size() != N) { printf("%-20s own: %zu media files for %d inputs, skipped\n", name.c_str(), media.size(), N); break; }
                    for (int b = 0; b < N && !fail; b++) {
                        int h = 0, w = 0;
                        uint8_t * rgb = nullptr;
                        std::vector<uint8_t> rgb_vec;
                        std::string stem = media[b].substr(0, media[b].rfind('.'));
                        if (!rgb_dir.empty()) {
                            npy::Array a = npy::load(rgb_dir + "/" + stem + ".rgb.npy");
                            h = (int) a.shape[0]; w = (int) a.shape[1];
                            rgb_vec = a.bytes;
                        } else {
                            rgb = jepa_load_image_rgb((media_dir + "/" + media[b]).c_str(), &h, &w);
                            if (!rgb) { fail = true; break; }
                        }
                        int oh = 0, ow = 0;
                        float * x = jepa_preprocess_image_rgb_ex(&pre, rgb ? rgb : rgb_vec.data(), h, w, &oh, &ow);
                        if (rgb) jepa_free(rgb);
                        if (!x || oh != H || ow != W) { printf("%-20s own preprocessing produced %dx%d, reference is %dx%d\n", name.c_str(), oh, ow, H, W); fail = true; if (x) jepa_free(x); break; }
                        in_own.insert(in_own.end(), x, x + (size_t) 3 * H * W);
                        jepa_free(x);
                    }
                }
                if (fail || in_own.size() != in_ref.size()) { res.ok = false; all_ok = false; break; }
                int64_t n_eq = 0;
                const size_t chan = (size_t) F * H * W;
                for (size_t i = 0; i < in_own.size(); i++) {
                    const double d = std::fabs((double) in_own[i] - in_ref[i]);
                    if (d > in_maxd) in_maxd = d;
                    if (in_own[i] == in_ref[i]) n_eq++;
                    // one uint8 level in normalised units: rescale / std (channel of this element)
                    const int c = (int) ((i / chan) % 3);
                    if (d > 1.5 * (pre.rescale / pre.std[c])) in_off1++;
                }
                in_eq = (double) n_eq / (double) in_own.size();
            }

            jepa_input in;
            in.data = own ? in_own.data() : in_ref.data();
            in.n_batch = N; in.n_chans = 3; in.n_frames = F; in.height = H; in.width = W;
            jepa_output enc = {nullptr, 0, 0};
            const double te = now_ms();
            if (jepa_encode(ctx, &in, &enc) != 0) { printf("%-20s encode failed\n", name.c_str()); res.ok = false; all_ok = false; break; }
            const double wall = now_ms() - te, ms = jepa_context_last_compute_ms(ctx);
            const int64_t n_tok = enc.n_tokens / N;
            tokens_per_item = n_tok;

            metrics m_lhs, m_pool, m_cls, m_emb, m_hpool, m_logits;
            int top1_ok = -1;
            double top5_frac = -1;
            if (has_lhs && N == 1 && ref_size_ok(lhs_ref, enc.n_tokens, D, "last_hidden_state")) m_lhs = compare(enc.data, lhs_ref.data(), enc.n_tokens, D);
            jepa_output one = {enc.data, n_tok, D};
            if (has_pool && N == 1) { jepa_output o = {nullptr, 0, 0}; if (jepa_pool_mean(model, &one, &o) == 0) { if (ref_size_ok(pool_ref, 1, D, "pooled_mean")) m_pool = compare(o.data, pool_ref.data(), 1, D); jepa_free(o.data); } }
            if (has_cls && N == 1)  { jepa_output o = {nullptr, 0, 0}; if (jepa_pool_cls(model, &one, &o) == 0)  { if (ref_size_ok(cls_ref, 1, D, "cls")) m_cls = compare(o.data, cls_ref.data(), 1, D); jepa_free(o.data); } }
            if (has_emb && N == 1 && jepa_model_has_projector(model)) {
                jepa_output o = {nullptr, 0, 0};
                if (jepa_lewm_project(ctx, &one, &o) == 0) { if (ref_size_ok(emb_ref, 1, D, "emb")) m_emb = compare(o.data, emb_ref.data(), 1, D); jepa_free(o.data); }
            }
            if (has_emb_seq && jepa_model_has_projector(model)) {
                std::vector<float> cls_rows((size_t) N * D);
                for (int b = 0; b < N; b++) memcpy(cls_rows.data() + (size_t) b * D, enc.data + (size_t) b * n_tok * D, D * sizeof(float));
                jepa_output o = {nullptr, 0, 0};
                if (jepa_lewm_project_rows(ctx, cls_rows.data(), N, &o) == 0) { if (ref_size_ok(emb_seq_ref, N, D, "emb_seq")) m_emb = compare(o.data, emb_seq_ref.data(), N, D); jepa_free(o.data); }
            }
            // attentive-pool head: pooler output + logits + top-1 / top-5 agreement
            if ((has_logits || has_hpool) && N == 1 && jepa_model_has_head(model)) {
                jepa_output hp = {nullptr, 0, 0}, lg = {nullptr, 0, 0};
                if (jepa_head_ex(ctx, &one, &hp, &lg) == 0) {
                    if (has_hpool && ref_size_ok(hpool_ref, 1, D, "pooled")) m_hpool = compare(hp.data, hpool_ref.data(), 1, D);
                    if (has_logits && ref_size_ok(logits_ref, 1, lg.dim, "logits")) m_logits = compare(lg.data, logits_ref.data(), 1, lg.dim);
                    if (has_top5 && top5_ref.size() >= 1) {
                        const int k = (int) top5_ref.size();
                        std::vector<int32_t> top(k);
                        const int got = jepa_top_k(lg.data, n_classes, k, top.data());
                        top1_ok = (got > 0 && (int) top5_ref[0] == top[0]) ? 1 : 0;
                        int hit = 0;
                        for (int i = 0; i < got; i++) {
                            for (int j = 0; j < k; j++) if ((int) top5_ref[j] == top[i]) { hit++; break; }
                        }
                        top5_frac = k ? (double) hit / (double) k : -1;
                    }
                    jepa_free(hp.data); jepa_free(lg.data);
                }
            }
            jepa_free(enc.data);

            bool ok = passes(m_lhs, thr_lhs, own) && passes(m_pool, thr, own) && passes(m_cls, thr, own) && passes(m_emb, thr, own)
                      && passes(m_hpool, thr, own) && passes(m_logits, thr, own);
            if (top1_ok == 0) ok = false;                       // top-1 must agree with the reference
            // and (outside the advisory low-bit tier) at least 4 of the reference top-5
            if (pol.gate_top5 && top5_frac >= 0 && top5_frac < 0.8) ok = false;
            if (!m_lhs.valid && !m_pool.valid && !m_cls.valid && !m_emb.valid && !m_hpool.valid && !m_logits.valid) ok = true;
            res.ok &= ok;
            all_ok &= ok;

            auto fmt = [](const metrics & m, char * buf, size_t n, const char * f) { if (m.valid) snprintf(buf, n, f, m.cos_min); else snprintf(buf, n, "-"); };
            char b_pool[16], b_cls[16], b_log[16], b_lhs1[16], b_lhsm[16], b_lhs2[16], b_abs[16], b_rel[16], b_ind[16], b_ineq[16], b_t1[16], b_t5[16];
            // "pool" shows pooled_mean or, for a classifier, the attentive-pooler output
            fmt(m_pool.valid ? m_pool : m_hpool, b_pool, 16, "%.6f");
            fmt(m_cls.valid ? m_cls : m_emb, b_cls, 16, "%.6f");
            fmt(m_logits, b_log, 16, "%.6f");
            if (m_lhs.valid) { snprintf(b_lhs1, 16, "%.6f", m_lhs.cos_mean); snprintf(b_lhsm, 16, "%.6f", m_lhs.cos_med); snprintf(b_lhs2, 16, "%.6f", m_lhs.cos_min); snprintf(b_abs, 16, "%.2e", m_lhs.max_abs); snprintf(b_rel, 16, "%.2e", m_lhs.rel_max); }
            else { snprintf(b_lhs1, 16, "-"); snprintf(b_lhsm, 16, "-"); snprintf(b_lhs2, 16, "-"); snprintf(b_abs, 16, "-"); snprintf(b_rel, 16, "-"); }
            if (top1_ok >= 0) snprintf(b_t1, 16, "%s", top1_ok ? "ok" : "MISS"); else snprintf(b_t1, 16, "-");
            if (top5_frac >= 0) snprintf(b_t5, 16, "%d/5", (int) (top5_frac * 5 + 0.5)); else snprintf(b_t5, 16, "-");
            if (own) { snprintf(b_ind, 16, "%.2e", in_maxd); snprintf(b_ineq, 16, "%.2f", 100.0 * in_eq); } else { snprintf(b_ind, 16, "-"); snprintf(b_ineq, 16, "-"); }
            printf("%-20s %-6s %8s %8s %8s %9s %9s %8s %8s %8s %5s %5s %9s %7s %9.1f%s\n", name.c_str(), own ? "own" : "stored", b_lhs1, b_lhsm, b_lhs2, b_abs, b_rel,
                   b_pool, b_cls, b_log, b_t1, b_t5, b_ind, b_ineq, ms, ok ? "" : "  FAIL");
            if (own && in_off1 > 0 && !quiet) printf("%-20s        (%lld input values differ by more than one uint8 level)\n", "", (long long) in_off1);
            if (m_lhs.valid && m_lhs.cos_min < 0.999 && !quiet) {
                printf("%-20s        (worst token %lld/%lld: cos %.6f, |ref row| %.2f vs mean %.2f; %lld tokens < 0.999, %lld < 0.99)\n", "",
                       (long long) m_lhs.worst_row, (long long) m_lhs.n_rows, m_lhs.cos_min, m_lhs.worst_row_norm, m_lhs.mean_row_norm,
                       (long long) m_lhs.n_lt_999, (long long) m_lhs.n_lt_99);
            }

            json & r = res.row[own ? "own_preprocess" : "stored_input"];
            r["ms_compute"] = ms; r["ms_wall"] = wall; r["n_items"] = N; r["n_frames"] = F; r["n_tokens"] = n_tok; r["ok"] = ok;
            auto put = [&](const char * key, const metrics & m) { if (m.valid) r[key] = {{"cos_mean", m.cos_mean}, {"cos_med", m.cos_med}, {"cos_min", m.cos_min}, {"max_abs", m.max_abs}, {"rel_max", m.rel_max},
                                                                                          {"n_rows", m.n_rows}, {"worst_row", m.worst_row}, {"worst_row_norm", m.worst_row_norm}, {"mean_row_norm", m.mean_row_norm},
                                                                                          {"n_rows_below_cos_0.999", m.n_lt_999}, {"n_rows_below_cos_0.99", m.n_lt_99}}; };
            put("last_hidden_state", m_lhs); put("pooled_mean", m_pool); put("cls", m_cls); put("emb", m_emb);
            put("pooled", m_hpool); put("logits", m_logits);
            if (top1_ok >= 0) r["top1_match"] = top1_ok == 1;
            if (top5_frac >= 0) r["top5_overlap"] = top5_frac;
            if (own) { r["input_max_abs"] = in_maxd; r["input_frac_equal"] = in_eq; r["input_n_off_by_gt1_level"] = in_off1; }
            if (!own) { sum_ms += ms / N; n_timed++; }
        }
        if (s.contains("timing_s") && s["timing_s"].contains("forward_s")) sum_ref_s += s["timing_s"]["forward_s"].get<double>();
        results.push_back(res);
    }

    if (results.empty()) {
        printf("error: no samples matched%s%s — nothing was tested\nRESULT: FAIL\n",
               sample_filter.empty() ? "" : " filter ", sample_filter.c_str());
        return 2;
    }

    const long rss = peak_rss_kb();
    printf("\nmean graph compute: %.1f ms/item over %d samples (%d threads) | PyTorch reference: %.1f ms/sample (%d threads) | "
           "model load %.0f ms | peak RSS %.0f MiB\n",
           n_timed ? sum_ms / n_timed : 0.0, n_timed, jepa_context_n_threads(ctx),
           n_timed ? 1000.0 * sum_ref_s / n_timed : 0.0, manifest.value("framework", json::object()).value("threads", 0),
           load_ms, rss / 1024.0);
    printf("RESULT: %s\n", all_ok ? "PASS" : "FAIL");

    if (!json_out.empty()) {
        json out;
        out["model"] = model_path; out["ref"] = ref_dir; out["family"] = jepa_model_family(model);
        out["file_type"] = jepa_model_file_type_name(model); out["threads"] = jepa_context_n_threads(ctx);
        out["flash"] = cp.use_flash_attn; out["flash_kv"] = kv_name; out["pass"] = all_ok;
        out["thresholds"] = {
            {"family_class", FAM_NAME[fam]}, {"tier", TIER_NAME[tier]}, {"advisory", pol.advisory},
            {"gate_top5", pol.gate_top5},
            {"token_map", {{"min_cos_mean", thr_lhs.min_mean}, {"min_cos_med", thr_lhs.min_med},
                           {"min_cos_min", thr_lhs.min_min}, {"max_rel_at_2048_tokens", thr_lhs.max_rel}}},
            {"derived",   {{"min_cos_mean", thr.min_mean}, {"min_cos_med", thr.min_med},
                           {"min_cos_min", thr.min_min}, {"max_rel_at_2048_tokens", thr.max_rel}}},
        };
        out["mean_ms_per_item"] = n_timed ? sum_ms / n_timed : 0.0;
        out["ref_mean_ms_per_sample"] = n_timed ? 1000.0 * sum_ref_s / n_timed : 0.0;
        out["peak_rss_mib"] = rss / 1024.0;
        out["load_ms"] = load_ms;
        out["preprocess"] = pre_to_string(pre);
        out["samples"] = json::array();
        for (auto & r : results) out["samples"].push_back(r.row);
        std::ofstream f(json_out);
        f << out.dump(2) << "\n";
    }
    jepa_context_free(ctx);
    jepa_model_free(model);
    return all_ok ? 0 : 1;
}
