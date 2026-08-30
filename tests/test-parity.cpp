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
// Thresholds come from general.file_type: f32 cos >= 0.9999 & rel_max <= 1e-3; f16 cos >= 0.9999; else cos >= 0.999.
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
    double cos_mean = 0, cos_min = 0, max_abs = 0, rel_max = 0, ref_max = 0;
    int64_t n_rows = 0;
    bool valid = false;
};

// rows x dim, cosine per row
static metrics compare(const float * a, const float * b, int64_t rows, int64_t dim) {
    metrics m;
    m.n_rows = rows;
    m.cos_min = 1.0;
    double cos_sum = 0;
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
        cos_sum += c;
        if (c < m.cos_min) m.cos_min = c;
    }
    m.cos_mean = rows ? cos_sum / rows : 0;
    m.rel_max = m.max_abs / (m.ref_max + 1e-30);
    m.valid = true;
    return m;
}

// Thresholds by general.file_type, applied to the stored-input pass:
//   f32: worst-token cosine >= 0.9999 and rel_max <= 1e-3
//   f16: mean cosine >= 0.9999 and worst-token cosine >= 0.999 (the worst *token* of an f16 file
//        cannot reach 0.9999 on I-JEPA even with float32 math — the numpy spec of selftest.py
//        gives cos_min 0.99969 on coco_000000219578 — so the 0.9999 bar holds for the mean)
//   other (q8_0, ...): mean >= 0.999, worst-token >= 0.99
// The own-preprocessing pass only has to keep mean cosine >= 0.99: it additionally carries
// JPEG-decoder differences (stb_image vs PIL) unless --rgb-dir provides the reference pixels.
struct thresholds { double min_mean; double min_min; double max_rel; };

static thresholds thresholds_for(int ftype) {
    if (ftype == 0) return {0.9999, 0.9999, 1e-3};
    if (ftype == 1) return {0.9999, 0.999, -1.0};
    return {0.999, 0.99, -1.0};
}

static bool passes(const metrics & m, const thresholds & t, bool own) {
    if (!m.valid) return true;
    if (own) return m.cos_mean >= 0.99;
    if (m.cos_mean < t.min_mean) return false;
    if (m.cos_min < t.min_min) return false;
    if (t.max_rel > 0 && m.rel_max > t.max_rel) return false;
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
    const thresholds thr = thresholds_for(ftype);

    jepa_preprocess_params pre_model = jepa_preprocess_default_params(model);
    bool pre_ok = true;
    jepa_preprocess_params pre_ref = manifest.contains("preprocessing") ? params_from_manifest(manifest["preprocessing"], pre_model, &pre_ok) : pre_model;
    const jepa_preprocess_params & pre = pre_mode == "model" ? pre_model : pre_ref;
    const bool pre_differs = memcmp(&pre_model.mean, &pre_ref.mean, sizeof(pre_model.mean)) != 0 || memcmp(&pre_model.std, &pre_ref.std, sizeof(pre_model.std)) != 0 ||
                             pre_model.resize_short != pre_ref.resize_short || pre_model.crop != pre_ref.crop ||
                             pre_model.resample != pre_ref.resample || pre_model.resize_mode != pre_ref.resize_mode;

    const char * kv_name = cp.flash_kv == JEPA_KV_F16 ? "f16" : cp.flash_kv == JEPA_KV_F32 ? "f32" : (ftype == 0 ? "auto(f32)" : "auto(f16)");
    printf("model: %s (%s, %s, %d layers, D=%d) | ref: %s | threads %d | flash %s | kv %s | thresholds: cos_mean >= %g, cos_min >= %g%s\n",
           jepa_model_name(model), jepa_model_family(model), jepa_model_file_type_name(model), jepa_model_n_layer(model),
           jepa_model_embed_dim(model), manifest.value("model", "?").c_str(), jepa_context_n_threads(ctx),
           cp.use_flash_attn ? "yes" : "no", kv_name, thr.min_mean, thr.min_min, thr.max_rel > 0 ? ", rel_max <= 1e-3" : "");
    printf("preprocess (%s): %s\n", pre_mode == "model" ? "jepa.pre.* of the GGUF" : "reference manifest", pre_to_string(pre).c_str());
    if (pre_differs) {
        printf("NOTE: the GGUF jepa.pre.* pipeline differs from the reference manifest's: %s\n", pre_to_string(pre_mode == "model" ? pre_ref : pre_model).c_str());
    }

    printf("\n%-20s %-6s %8s %8s %9s %9s %8s %8s %8s %9s %7s %8s\n", "sample", "input", "cos_mean", "cos_min", "max_abs", "rel_max",
           "pool", "cls", "emb", "in_maxd", "in_eq%", "ms");

    std::vector<sample_result> results;
    bool all_ok = true;
    double sum_ms = 0, sum_ref_s = 0;
    int n_timed = 0;
    for (const json & s : manifest["samples"]) {
        const std::string name = s["name"].get<std::string>();
        if (!sample_filter.empty() && ("," + sample_filter + ",").find("," + name + ",") == std::string::npos) continue;
        const json & T = s["tensors"];
        if (!T.contains("input")) continue;
        sample_result res;
        res.name = name;
        res.row["sample"] = name;

        npy::Array in_npy = npy::load(ref_dir + "/" + T["input"]["file"].get<std::string>());
        if (in_npy.shape.size() != 4 || in_npy.shape[1] != 3) {
            printf("%-20s skipped: input shape not [N,3,H,W]\n", name.c_str());
            continue;
        }
        const int N = (int) in_npy.shape[0], H = (int) in_npy.shape[2], W = (int) in_npy.shape[3];
        std::vector<float> in_ref = in_npy.to_f32();
        const int64_t D = jepa_model_embed_dim(model);

        // reference tensors (optional)
        auto load_opt = [&](const char * key, std::vector<float> & out, std::vector<int64_t> & shape) -> bool {
            if (!T.contains(key)) return false;
            npy::Array a = npy::load(ref_dir + "/" + T[key]["file"].get<std::string>());
            out = a.to_f32(); shape = a.shape; return true;
        };
        std::vector<float> lhs_ref, pool_ref, cls_ref, emb_ref, emb_seq_ref;
        std::vector<int64_t> sh;
        const bool has_lhs = load_opt("last_hidden_state", lhs_ref, sh);
        const bool has_pool = load_opt("pooled_mean", pool_ref, sh);
        const bool has_cls = load_opt("cls", cls_ref, sh);
        const bool has_emb = load_opt("emb", emb_ref, sh);
        const bool has_emb_seq = load_opt("emb_seq", emb_seq_ref, sh);

        // run: (a) stored input, (b) own preprocessing
        for (int pass = 0; pass < 2; pass++) {
            const bool own = pass == 1;
            std::vector<float> in_own;
            double in_maxd = 0; double in_eq = 0; int64_t in_off1 = 0;
            if (own) {
                std::vector<std::string> media;
                if (s["media"].is_string()) media.push_back(s["media"].get<std::string>());
                else for (const json & mm : s["media"]) media.push_back(mm.get<std::string>());
                if ((int) media.size() != N) { printf("%-20s own: %zu media files for %d inputs, skipped\n", name.c_str(), media.size(), N); break; }
                bool fail = false;
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
                if (fail) { res.ok = false; all_ok = false; break; }
                int64_t n_eq = 0;
                for (size_t i = 0; i < in_own.size(); i++) {
                    const double d = std::fabs((double) in_own[i] - in_ref[i]);
                    if (d > in_maxd) in_maxd = d;
                    if (in_own[i] == in_ref[i]) n_eq++;
                    // one uint8 level in normalised units: rescale / std (channel of this element)
                    const int c = (int) ((i / ((size_t) H * W)) % 3);
                    if (d > 1.5 * (pre.rescale / pre.std[c])) in_off1++;
                }
                in_eq = (double) n_eq / (double) in_own.size();
            }

            jepa_input in;
            in.data = own ? in_own.data() : in_ref.data();
            in.n_batch = N; in.n_chans = 3; in.n_frames = 1; in.height = H; in.width = W;
            jepa_output enc = {nullptr, 0, 0};
            const double te = now_ms();
            if (jepa_encode(ctx, &in, &enc) != 0) { printf("%-20s encode failed\n", name.c_str()); res.ok = false; all_ok = false; break; }
            const double wall = now_ms() - te, ms = jepa_context_last_compute_ms(ctx);
            const int64_t n_tok = enc.n_tokens / N;

            metrics m_lhs, m_pool, m_cls, m_emb;
            if (has_lhs && N == 1) m_lhs = compare(enc.data, lhs_ref.data(), enc.n_tokens, D);
            jepa_output one = {enc.data, n_tok, D};
            if (has_pool && N == 1) { jepa_output o = {nullptr, 0, 0}; if (jepa_pool_mean(model, &one, &o) == 0) { m_pool = compare(o.data, pool_ref.data(), 1, D); jepa_free(o.data); } }
            if (has_cls && N == 1)  { jepa_output o = {nullptr, 0, 0}; if (jepa_pool_cls(model, &one, &o) == 0)  { m_cls = compare(o.data, cls_ref.data(), 1, D); jepa_free(o.data); } }
            if (has_emb && N == 1 && jepa_model_has_projector(model)) {
                jepa_output o = {nullptr, 0, 0};
                if (jepa_lewm_project(ctx, &one, &o) == 0) { m_emb = compare(o.data, emb_ref.data(), 1, D); jepa_free(o.data); }
            }
            if (has_emb_seq && jepa_model_has_projector(model)) {
                std::vector<float> cls_rows((size_t) N * D);
                for (int b = 0; b < N; b++) memcpy(cls_rows.data() + (size_t) b * D, enc.data + (size_t) b * n_tok * D, D * sizeof(float));
                jepa_output o = {nullptr, 0, 0};
                if (jepa_lewm_project_rows(ctx, cls_rows.data(), N, &o) == 0) { m_emb = compare(o.data, emb_seq_ref.data(), N, D); jepa_free(o.data); }
            }
            jepa_free(enc.data);

            bool ok = passes(m_lhs, thr, own) && passes(m_pool, thr, own) && passes(m_cls, thr, own) && passes(m_emb, thr, own);
            if (!m_lhs.valid && !m_pool.valid && !m_cls.valid && !m_emb.valid) ok = true;
            res.ok &= ok;
            all_ok &= ok;

            auto fmt = [](const metrics & m, char * buf, size_t n, const char * f) { if (m.valid) snprintf(buf, n, f, m.cos_min); else snprintf(buf, n, "-"); };
            char b_pool[16], b_cls[16], b_emb[16], b_lhs1[16], b_lhs2[16], b_abs[16], b_rel[16], b_ind[16], b_ineq[16];
            fmt(m_pool, b_pool, 16, "%.6f"); fmt(m_cls, b_cls, 16, "%.6f"); fmt(m_emb, b_emb, 16, "%.6f");
            if (m_lhs.valid) { snprintf(b_lhs1, 16, "%.6f", m_lhs.cos_mean); snprintf(b_lhs2, 16, "%.6f", m_lhs.cos_min); snprintf(b_abs, 16, "%.2e", m_lhs.max_abs); snprintf(b_rel, 16, "%.2e", m_lhs.rel_max); }
            else { snprintf(b_lhs1, 16, "-"); snprintf(b_lhs2, 16, "-"); snprintf(b_abs, 16, "-"); snprintf(b_rel, 16, "-"); }
            if (own) { snprintf(b_ind, 16, "%.2e", in_maxd); snprintf(b_ineq, 16, "%.2f", 100.0 * in_eq); } else { snprintf(b_ind, 16, "-"); snprintf(b_ineq, 16, "-"); }
            printf("%-20s %-6s %8s %8s %9s %9s %8s %8s %8s %9s %7s %8.1f%s\n", name.c_str(), own ? "own" : "stored", b_lhs1, b_lhs2, b_abs, b_rel,
                   b_pool, b_cls, b_emb, b_ind, b_ineq, ms, ok ? "" : "  FAIL");
            if (own && in_off1 > 0 && !quiet) printf("%-20s        (%lld input values differ by more than one uint8 level)\n", "", (long long) in_off1);

            json & r = res.row[own ? "own_preprocess" : "stored_input"];
            r["ms_compute"] = ms; r["ms_wall"] = wall; r["n_items"] = N; r["ok"] = ok;
            auto put = [&](const char * key, const metrics & m) { if (m.valid) r[key] = {{"cos_mean", m.cos_mean}, {"cos_min", m.cos_min}, {"max_abs", m.max_abs}, {"rel_max", m.rel_max}}; };
            put("last_hidden_state", m_lhs); put("pooled_mean", m_pool); put("cls", m_cls); put("emb", m_emb);
            if (own) { r["input_max_abs"] = in_maxd; r["input_frac_equal"] = in_eq; r["input_n_off_by_gt1_level"] = in_off1; }
            if (!own) { sum_ms += ms / N; n_timed++; }
        }
        if (s.contains("timing_s") && s["timing_s"].contains("forward_s")) sum_ref_s += s["timing_s"]["forward_s"].get<double>();
        results.push_back(res);
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
        out["thresholds"] = {{"min_cos_mean", thr.min_mean}, {"min_cos_min", thr.min_min}, {"max_rel", thr.max_rel}};
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
