// jepa-worldmodel: LeWM world-model demo / CI check.
//
//   jepa-worldmodel -m lewm-pusht-f32.gguf --image a.jpg [--image b.jpg ...]
//                   (--actions '0.1,0.2,...;0.3,...' | --random-actions K [--seed 0])
//                   [-t N] [-o rollout.npy] [--no-flash] [--print-n 8]
//       encodes the image(s) -> CLS -> enc.proj (the world-model state), rolls the predictor out
//       over the actions and prints, per step, the L2 / cosine drift of the predicted embedding
//       against the previous one plus the predictor time.
//
//   jepa-worldmodel -m lewm-pusht-f32.gguf --ref-check tests/fixtures/ref/lewm-pusht [--min-cos X]
//       reproduces the PyTorch reference: for every image sample encode the stored `input`, project,
//       compare `emb`, run the predictor with the stored action and compare `pred_next`; then the
//       3-frame `seq` sample against `pred_seq`. Exit status 1 if any cosine is below the threshold.
#include "jepa.h"
#include "npy.h"
#include "json.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s -m lewm.gguf --image a.jpg [--image b.jpg] (--actions 'a,b,..;c,d,..' | --random-actions K) [options]\n"
        "       %s -m lewm.gguf --ref-check tests/fixtures/ref/lewm-pusht [--min-cos X]\n"
        "  -t N              threads (default: all)\n"
        "  --seed S          RNG seed for --random-actions (default 0)\n"
        "  -o FILE.npy       write the [K, D] rollout to a float32 .npy\n"
        "  --print-n N       print the first N values of each predicted state (default 0)\n"
        "  --no-flash        naive attention instead of ggml_flash_attn_ext\n"
        "  --min-cos X       --ref-check threshold (default 0.9999 for f32/f16, 0.999 otherwise)\n", a0, a0);
}

static double cosine(const float * a, const float * b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) { dot += (double) a[i] * b[i]; na += (double) a[i] * a[i]; nb += (double) b[i] * b[i]; }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}
static double l2(const float * a, const float * b, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++) { const double d = (double) a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}
static double norm2(const float * a, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++) s += (double) a[i] * a[i];
    return std::sqrt(s);
}

// "0.1,0.2;0.3,0.4" -> K rows of action_dim
static bool parse_actions(const std::string & s, int action_dim, std::vector<float> & out) {
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(';', p);
        std::string row = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (!row.empty()) {
            int n = 0;
            size_t r = 0;
            while (r <= row.size()) {
                size_t c = row.find(',', r);
                std::string tok = row.substr(r, c == std::string::npos ? std::string::npos : c - r);
                if (!tok.empty()) { out.push_back((float) atof(tok.c_str())); n++; }
                if (c == std::string::npos) break;
                r = c + 1;
            }
            if (n != action_dim) {
                fprintf(stderr, "--actions: row has %d values, the model wants %d\n", n, action_dim);
                return false;
            }
        }
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return !out.empty();
}

// encode one preprocessed NCHW image (or n images) and return the projected world-model states
static bool project_images(jepa_context * ctx, jepa_model * m, const float * nchw, int n_images, int h, int w,
                           std::vector<float> & embs, double * enc_ms) {
    jepa_input in = { nchw, n_images, 3, 1, h, w };
    jepa_output enc = {};
    const double t0 = now_ms();
    if (jepa_encode(ctx, &in, &enc) != 0) { jepa_free(enc.data); return false; }
    if (enc_ms) *enc_ms = now_ms() - t0;
    const int64_t per = enc.n_tokens / n_images;
    std::vector<float> cls((size_t) n_images * enc.dim);
    for (int i = 0; i < n_images; i++) {
        memcpy(cls.data() + (size_t) i * enc.dim, enc.data + (size_t) i * per * enc.dim, (size_t) enc.dim * sizeof(float));
    }
    jepa_free(enc.data);
    jepa_output proj = {};
    if (jepa_lewm_project_rows(ctx, cls.data(), n_images, &proj) != 0) { jepa_free(proj.data); return false; }
    embs.assign(proj.data, proj.data + (size_t) proj.n_tokens * proj.dim);
    jepa_free(proj.data);
    (void) m;
    return true;
}

// ------------------------------------------------------------------------------------------
// --ref-check
// ------------------------------------------------------------------------------------------
static int ref_check(jepa_context * ctx, jepa_model * model, const std::string & ref, double min_cos) {
    json manifest;
    {
        std::ifstream f(ref + "/manifest.json");
        if (!f) { fprintf(stderr, "cannot open %s/manifest.json\n", ref.c_str()); return 2; }
        f >> manifest;
    }
    const int D = jepa_model_embed_dim(model);
    int fail = 0;
    printf("%-20s %-14s %10s %12s %10s\n", "sample", "tensor", "cosine", "max|d|", "ms");
    for (const auto & s : manifest["samples"]) {
        const std::string name = s.value("name", "");
        const auto & tensors = s["tensors"];
        auto load = [&](const char * key, std::vector<int64_t> * shape) -> std::vector<float> {
            npy::Array a = npy::load(ref + "/" + tensors[key]["file"].get<std::string>());
            if (shape) *shape = a.shape;
            return a.to_f32();
        };
        auto report = [&](const std::string & tname, const float * got, const float * want, int64_t rows, double ms) {
            double worst_cos = 1, max_abs = 0;
            for (int64_t r = 0; r < rows; r++) {
                worst_cos = std::fmin(worst_cos, cosine(got + r * D, want + r * D, D));
                for (int64_t i = 0; i < D; i++) max_abs = std::fmax(max_abs, std::fabs(got[r * D + i] - want[r * D + i]));
            }
            const bool ok = worst_cos >= min_cos;
            if (!ok) fail++;
            printf("%-20s %-14s %10.7f %12.3e %10.2f  %s\n", name.c_str(), tname.c_str(), worst_cos, max_abs, ms,
                   ok ? "OK" : "FAIL");
        };

        if (tensors.contains("pred_next")) {
            std::vector<int64_t> shp;
            std::vector<float> input = load("input", &shp);
            std::vector<float> emb_ref = load("emb", nullptr), act = load("action", nullptr), want = load("pred_next", nullptr);
            const int h = (int) shp[shp.size() - 2], w = (int) shp[shp.size() - 1];
            std::vector<float> emb;
            double enc_ms = 0;
            if (!project_images(ctx, model, input.data(), 1, h, w, emb, &enc_ms)) { fprintf(stderr, "encode failed\n"); return 2; }
            report("emb", emb.data(), emb_ref.data(), 1, enc_ms);
            jepa_output out = {};
            // the predictor step is checked against the *reference* embedding so a graph bug in the
            // predictor cannot be masked (or caused) by the encoder
            if (jepa_lewm_predict(ctx, emb_ref.data(), act.data(), 1, &out) != 0) { jepa_free(out.data); return 2; }
            report("pred_next", out.data, want.data(), 1, jepa_context_last_compute_ms(ctx));
            jepa_free(out.data);
        } else if (tensors.contains("pred_seq")) {
            std::vector<int64_t> shp;
            std::vector<float> input = load("input", &shp);
            std::vector<float> emb_ref = load("emb_seq", nullptr), act = load("action_seq", nullptr), want = load("pred_seq", nullptr);
            const int T = (int) shp[0], h = (int) shp[shp.size() - 2], w = (int) shp[shp.size() - 1];
            std::vector<float> emb;
            double enc_ms = 0;
            if (!project_images(ctx, model, input.data(), T, h, w, emb, &enc_ms)) { fprintf(stderr, "encode failed\n"); return 2; }
            report("emb_seq", emb.data(), emb_ref.data(), T, enc_ms);
            jepa_output out = {};
            if (jepa_lewm_predict(ctx, emb_ref.data(), act.data(), T, &out) != 0) { jepa_free(out.data); return 2; }
            report("pred_seq", out.data, want.data(), T, jepa_context_last_compute_ms(ctx));
            jepa_free(out.data);
        }
    }
    printf("%s (threshold cos >= %g)\n", fail == 0 ? "PASS" : "FAIL", min_cos);
    return fail == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string model_path, actions_str, out_path, ref;
    std::vector<std::string> images;
    jepa_context_params cp = jepa_context_default_params();
    int n_random = 0, seed = 0, print_n = 0;
    double min_cos = -1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "-m") model_path = next();
        else if (a == "--image" || a == "-i") images.push_back(next());
        else if (a == "--actions") actions_str = next();
        else if (a == "--random-actions") n_random = atoi(next());
        else if (a == "--seed") seed = atoi(next());
        else if (a == "--ref-check") ref = next();
        else if (a == "--min-cos") min_cos = atof(next());
        else if (a == "-o") out_path = next();
        else if (a == "--print-n") print_n = atoi(next());
        else if (a == "-t" || a == "--threads") cp.n_threads = atoi(next());
        else if (a == "--no-flash") cp.use_flash_attn = false;
        else if (a == "--kv-f32") cp.flash_kv = JEPA_KV_F32;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || (ref.empty() && images.empty())) { usage(argv[0]); return 2; }

    jepa_model * model = jepa_model_load(model_path.c_str(), false);
    if (!model) return 2;
    if (jepa_lewm_action_dim(model) <= 0 || !jepa_model_has_projector(model)) {
        fprintf(stderr, "%s is not a LeWM world model (no predictor action embed / enc.proj)\n", model_path.c_str());
        jepa_model_free(model);
        return 2;
    }
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) { jepa_model_free(model); return 2; }
    // one cleanup path for every exit below (the weights are the big allocation here)
    auto done = [&](int rc) {
        jepa_context_free(ctx);
        jepa_model_free(model);
        return rc;
    };
    const int D = jepa_model_embed_dim(model);
    const int A = jepa_lewm_action_dim(model);
    const int ftype = jepa_model_file_type(model);

    if (!ref.empty()) {
        const double mc = min_cos > 0 ? min_cos : (ftype <= 1 ? 0.9999 : 0.999);
        printf("model: %s (%s, %s) | threads %d | ref: %s\n", jepa_model_name(model), jepa_model_family(model),
               jepa_model_file_type_name(model), jepa_context_n_threads(ctx), ref.c_str());
        return done(ref_check(ctx, model, ref, mc));
    }

    // --- demo rollout -----------------------------------------------------------------------
    std::vector<float> actions;
    if (n_random > 0) {
        std::mt19937 rng((uint32_t) seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        actions.resize((size_t) n_random * A);
        for (auto & v : actions) v = nd(rng);
    } else if (!actions_str.empty()) {
        if (!parse_actions(actions_str, A, actions)) return done(2);
    } else {
        fprintf(stderr, "need --actions or --random-actions\n");
        return done(2);
    }
    const int K = (int) (actions.size() / A);

    // encode the seed frame(s)
    std::vector<float> pixels;
    int h = 0, w = 0;
    for (const std::string & p : images) {
        int ih = 0, iw = 0;
        float * x = jepa_preprocess_image_file(model, p.c_str(), &ih, &iw);
        if (!x) { fprintf(stderr, "cannot read %s\n", p.c_str()); return done(2); }
        if (h && (ih != h || iw != w)) { fprintf(stderr, "images preprocess to different sizes\n"); jepa_free(x); return done(2); }
        h = ih; w = iw;
        pixels.insert(pixels.end(), x, x + (size_t) 3 * ih * iw);
        jepa_free(x);
    }
    std::vector<float> embs;
    double enc_ms = 0;
    if (!project_images(ctx, model, pixels.data(), (int) images.size(), h, w, embs, &enc_ms)) return done(2);
    printf("model: %s (%s, %s) | threads %d | D=%d action_dim=%d window=%d\n", jepa_model_name(model),
           jepa_model_family(model), jepa_model_file_type_name(model), jepa_context_n_threads(ctx), D, A,
           jepa_lewm_n_frames(model));
    printf("encoded %zu image(s) in %.1f ms; |emb| = %.4f\n", images.size(), enc_ms, norm2(embs.data(), D));

    std::vector<float> steps((size_t) K * D);
    std::vector<double> step_ms((size_t) K, 0);
    const int n_seed = (int) images.size();
    const int win = jepa_lewm_n_frames(model);
    // The same windowing as jepa_lewm_rollout, unrolled here so every step can be timed on its own
    // (frame j of the growing sequence uses actions[clamp(j - (n_seed-1), 0, K-1)]).
    {
        std::vector<float> seq = embs;                       // [n_frames_so_far, D]
        std::vector<float> win_emb, win_act;
        for (int k = 0; k < K; k++) {
            const int have = n_seed + k;
            const int wlen = have < win ? have : win;
            const int first = have - wlen;
            win_emb.assign(seq.begin() + (size_t) first * D, seq.begin() + (size_t) have * D);
            win_act.resize((size_t) wlen * A);
            for (int i = 0; i < wlen; i++) {
                int ai = first + i - (n_seed - 1);
                if (ai < 0) ai = 0;
                if (ai > K - 1) ai = K - 1;
                memcpy(win_act.data() + (size_t) i * A, actions.data() + (size_t) ai * A, (size_t) A * sizeof(float));
            }
            const double t0 = now_ms();
            jepa_output so = {};
            if (jepa_lewm_predict(ctx, win_emb.data(), win_act.data(), wlen, &so) != 0) { jepa_free(so.data); return done(2); }
            step_ms[k] = now_ms() - t0;
            memcpy(steps.data() + (size_t) k * D, so.data + (size_t) (wlen - 1) * D, (size_t) D * sizeof(float));
            jepa_free(so.data);
            seq.insert(seq.end(), steps.begin() + (size_t) k * D, steps.begin() + (size_t) (k + 1) * D);
        }
    }
    // cross-check against the library rollout (same math, one call)
    {
        std::vector<float> lib((size_t) K * D);
        if (jepa_lewm_rollout(ctx, embs.data(), n_seed, actions.data(), K, lib.data()) != 0) return done(2);
        double worst = 0;
        for (size_t i = 0; i < lib.size(); i++) worst = std::fmax(worst, std::fabs(lib[i] - steps[i]));
        if (worst != 0.0) printf("warning: jepa_lewm_rollout differs from the unrolled loop by %.3e\n", worst);
    }

    printf("\n%-5s %10s %10s %10s %10s %9s\n", "step", "|pred|", "L2 drift", "cos(prev)", "cos(seed)", "ms");
    const float * prev = embs.data() + (size_t) (n_seed - 1) * D;
    for (int k = 0; k < K; k++) {
        const float * cur = steps.data() + (size_t) k * D;
        printf("%-5d %10.4f %10.4f %10.6f %10.6f %9.2f\n", k, norm2(cur, D), l2(cur, prev, D), cosine(cur, prev, D),
               cosine(cur, embs.data(), D), step_ms[k]);
        if (print_n > 0) {
            printf("      ");
            for (int i = 0; i < print_n && i < D; i++) printf("%9.4f", cur[i]);
            printf("\n");
        }
        prev = cur;
    }
    double total = 0;
    for (double v : step_ms) total += v;
    printf("rollout: %d steps, %.2f ms total, %.2f ms/step\n", K, total, K ? total / K : 0.0);

    if (!out_path.empty()) {
        npy::save_f32(out_path, { K, (int64_t) D }, steps.data());
        printf("wrote %s [%d, %d]\n", out_path.c_str(), K, D);
    }
    return done(0);
}
