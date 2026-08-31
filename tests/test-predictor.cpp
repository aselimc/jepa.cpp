// test-predictor: deterministic parity checks for the two predictors.
//
//   test-predictor --lewm  MODEL.gguf --ref tests/fixtures/ref/lewm-pusht [options]
//   test-predictor --vjepa2 MODEL.gguf --ref tests/fixtures/ref/vjepa2-vitl-fpc64-256 [--samples a,b] [options]
//   test-predictor --vjepa2 MODEL.gguf --case DIR        (numpy cross-check: DIR/{enc,ctx_idx,tgt_idx,pred}.npy)
//   options: [--threads N] [--no-flash] [--kv-f32|--kv-f16] [--min-cos X] [--modality auto|video|image] [--quiet]
//
// LeWM (f32 thresholds cos >= 0.9999):
//   * pred_next of every image sample (T = 1) and pred_seq of the 3-frame sequence sample (T = 3),
//     both against the PyTorch dump;
//   * causality: row t of the T = 3 run equals the prefix-only run over frames 0..t, and perturbing
//     a *later* frame's embedding leaves the earlier output rows bit-identical;
//   * jepa_lewm_rollout: step 0 of a rollout seeded with emb_seq[0] equals jepa_lewm_predict(T = 1).
// V-JEPA 2: the masked predictor over the reference encoder tokens with context = target = every
//   token (the HF default pass) against `predictor_last_hidden_state`.
// --modality selects the V-JEPA 2.1 modality vector (pred.mod_embed_video / _img) for --case runs:
//   the 576-token image case only reaches parity with `image` (see docs/parity.md, predictor table).
// Exit status 1 on any failure.
#include "jepa.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct metrics {
    double cos_mean = 0, cos_min = 1, max_abs = 0, rel_max = 0;
    int64_t rows = 0;
};

static metrics compare(const float * a, const float * b, int64_t rows, int64_t dim) {
    metrics m;
    m.rows = rows;
    double cos_sum = 0, ref_max = 0;
    for (int64_t r = 0; r < rows; r++) {
        double dot = 0, na = 0, nb = 0;
        for (int64_t d = 0; d < dim; d++) {
            const double x = a[r * dim + d], y = b[r * dim + d];
            dot += x * y; na += x * x; nb += y * y;
            const double diff = std::fabs(x - y);
            if (diff > m.max_abs) m.max_abs = diff;
            if (std::fabs(y) > ref_max) ref_max = std::fabs(y);
        }
        const double c = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        cos_sum += c;
        if (c < m.cos_min) m.cos_min = c;
    }
    m.cos_mean = rows ? cos_sum / rows : 0;
    m.rel_max = m.max_abs / (ref_max + 1e-30);
    return m;
}

static int g_fail = 0;
static bool g_quiet = false;

// thresholds mirror the image-family rows of tests/test-parity.cpp (docs/parity.md "Thresholds"):
// f32 mean/worst cosine >= 0.9999 and rel_max <= the length-aware bound below; f16 mean >= 0.9999,
// worst >= 0.99; quantized mean >= 0.999, worst >= 0.98.  The worst-row floors are dtype noise, not
// graph error: the numpy spec on the same GGUF bottoms out at cos_min 0.99897 for q8_0 (weight
// quantisation alone) on archery_f16.
struct thresholds { double min_mean, min_min, max_rel; };

// f32 rel_max bound, widened with the number of rows: max|a-b| grows like the accumulated round-off
// of the longest reduction in the graph (~sqrt(N) for attention over N rows).  1e-3 up to the
// 2048-row reference point, 2e-3 at 8192 rows (the 64-frame V-JEPA 2 clip measures 1.22e-3 there at
// cosine 1.000000, the 2.1 predictor 1.07e-3 at 4608 rows at cosine 1.0000000).  Same formula as
// test-parity.cpp.
static double rel_bound(double base, int64_t rows) {
    if (base <= 0) return -1.0;
    return std::fmax(base, base * std::sqrt((double) rows / 2048.0));
}

// On a GPU there is no f32 tier: ggml's CUDA "F32" mul_mat is TF32 (the algo enum, which
// GGML_PREC_F32 cannot undo), the attention path is F16-accumulate, and ggml_norm uses the
// one-pass variance -- docs/gpu-notes.md §6.4, same reasoning as test-parity's POLICY. An f32 file
// is therefore judged with the f16 bars there, and every GPU tier gates rel_max as well, because a
// wrong ggml_norm variance is a per-row *scale* error that cosine is blind to by construction.
static thresholds thresholds_for(int ftype, bool gpu) {
    if (gpu) {
        if (ftype == 0 || ftype == 1) return {0.9999, 0.99, 2e-2};
        return {0.999, 0.98, 8e-2};
    }
    if (ftype == 0) return {0.9999, 0.9999, 1e-3};
    if (ftype == 1) return {0.9999, 0.99, -1.0};
    return {0.999, 0.98, -1.0};
}

// self-consistency (prefix vs full run, rollout vs predict): the same graph on the same numbers,
// so anything beyond float32 round-off is a bug
static const thresholds exact = {0.9999999, 0.9999999, 1e-5};

// The one exception: the T = 1 prefix is not the same graph as the T = 3 run -- a single query row
// and no causal mask take ggml's per-row flash kernel (docs/ggml-notes.md §1) -- so for an f16 /
// quantized file its row matches only to dtype round-off (lewm-pusht-f16 measures max|d| 2.4e-4,
// rel 2.0e-4; T >= 2 is bit-identical at every dtype).  A broken causal mask would let row 0 see the
// later frames, which moves the cosine by ~1e-1, far below this bar.
static const thresholds exact_dtype = {0.999999, 0.999999, 1e-3};

static void check(const char * what, const metrics & m, const thresholds & t, double ms = -1) {
    const double min_cos = t.min_min, max_rel = rel_bound(t.max_rel, m.rows);
    const bool ok = m.cos_mean >= t.min_mean && m.cos_min >= min_cos && (max_rel <= 0 || m.rel_max <= max_rel);
    if (!ok) g_fail++;
    if (!g_quiet || !ok) {
        printf("  %-44s rows=%-6lld cos_mean=%.7f cos_min=%.7f max|d|=%.3e rel=%.3e", what,
               (long long) m.rows, m.cos_mean, m.cos_min, m.max_abs, m.rel_max);
        if (ms >= 0) printf(" %8.2f ms", ms);
        printf("  %s\n", ok ? "OK" : "FAIL");
        if (!ok) {
            char rel[48] = "";
            if (max_rel > 0) snprintf(rel, sizeof(rel), ", rel <= %.3e (%lld rows)", max_rel, (long long) m.rows);
            printf("  %-44s bars: cos_mean >= %g, cos_min >= %g%s\n", "", t.min_mean, min_cos, rel);
        }
    }
}

static std::vector<float> load_f32(const std::string & path, int64_t * rows, int64_t * dim) {
    npy::Array a = npy::load(path);
    std::vector<float> v = a.to_f32();
    if (rows) *rows = a.shape.size() >= 2 ? a.shape[0] : 1;
    if (dim)  *dim  = a.shape.empty() ? 0 : a.shape.back();
    return v;
}

static bool exists(const std::string & p) { std::ifstream f(p); return (bool) f; }

// ------------------------------------------------------------------------------------------
// LeWM
// ------------------------------------------------------------------------------------------
static void run_lewm(jepa_context * ctx, jepa_model * model, const std::string & ref, const thresholds & thr,
                     bool lossy) {
    const int D = jepa_model_embed_dim(model);
    const int A = jepa_lewm_action_dim(model);
    printf("LeWM predictor: D=%d action_dim=%d window=%d\n", D, A, jepa_lewm_n_frames(model));

    // per-image samples: T = 1
    for (const char * s : {"coco_000000000139", "coco_000000000285"}) {
        const std::string base = ref + "/" + s;
        if (!exists(base + ".emb.npy")) continue;
        int64_t r, d;
        std::vector<float> emb = load_f32(base + ".emb.npy", &r, &d);
        std::vector<float> act = load_f32(base + ".action.npy", &r, &d);
        std::vector<float> want = load_f32(base + ".pred_next.npy", &r, &d);
        jepa_output out = {};
        if (jepa_lewm_predict(ctx, emb.data(), act.data(), 1, &out) != 0) { printf("  %s: predict failed\n", s); g_fail++; continue; }
        check((std::string(s) + " pred_next (T=1)").c_str(), compare(out.data, want.data(), 1, D), thr,
              jepa_context_last_compute_ms(ctx));
        free(out.data);
    }

    // sequence sample: T = 3, causal
    const std::string seq = ref + "/seq";
    if (!exists(seq + ".emb_seq.npy")) return;
    int64_t T = 0, d = 0;
    std::vector<float> emb = load_f32(seq + ".emb_seq.npy", &T, &d);
    std::vector<float> act = load_f32(seq + ".action_seq.npy", &T, &d);
    std::vector<float> want = load_f32(seq + ".pred_seq.npy", &T, &d);
    jepa_output full = {};
    if (jepa_lewm_predict(ctx, emb.data(), act.data(), (int) T, &full) != 0) { printf("  seq: predict failed\n"); g_fail++; return; }
    check("seq pred_seq (T=3, all rows)", compare(full.data, want.data(), T, D), thr, jepa_context_last_compute_ms(ctx));

    // causality (a): the prefix-only run reproduces row t
    for (int t = 0; t < (int) T; t++) {
        jepa_output pre = {};
        if (jepa_lewm_predict(ctx, emb.data(), act.data(), t + 1, &pre) != 0) { g_fail++; continue; }
        char name[64];
        snprintf(name, sizeof(name), "seq causal prefix T=%d -> row %d", t + 1, t);
        check(name, compare(pre.data + (size_t) t * D, full.data + (size_t) t * D, 1, D),
              (t == 0 && lossy) ? exact_dtype : exact);
        free(pre.data);
    }

    // causality (b): perturbing the LAST frame must not move the earlier output rows at all.
    // The perturbation has to be NON-UNIFORM: the adaLN path starts with a non-affine LayerNorm, so
    // adding the same constant to every channel of a row is absorbed (mean-subtracted) and the test
    // would be vacuous -- move two channels in opposite directions instead.
    {
        std::vector<float> emb2 = emb, act2 = act;
        emb2[(size_t) (T - 1) * D + 0] += 5.0f;
        emb2[(size_t) (T - 1) * D + 1] -= 3.0f;
        act2[(size_t) (T - 1) * A + 0] += 5.0f;
        if (A > 1) act2[(size_t) (T - 1) * A + 1] -= 3.0f;
        jepa_output pert = {};
        if (jepa_lewm_predict(ctx, emb2.data(), act2.data(), (int) T, &pert) != 0) { g_fail++; }
        else {
            double worst = 0;
            for (int64_t i = 0; i < (T - 1) * D; i++) worst = std::fmax(worst, std::fabs(pert.data[i] - full.data[i]));
            const bool ok = worst == 0.0;
            if (!ok) g_fail++;
            printf("  %-44s max|d| on rows 0..%d = %.3e  %s\n", "causality: perturb frame T-1", (int) T - 2, worst,
                   ok ? "OK (bit-identical)" : "FAIL");
            // and the perturbed row itself has to move by a visible amount, or the check proves nothing
            double moved = 0;
            for (int64_t i = (T - 1) * D; i < T * D; i++) moved = std::fmax(moved, std::fabs(pert.data[i] - full.data[i]));
            const bool moved_ok = moved > 1e-2;
            if (!moved_ok) g_fail++;
            printf("  %-44s max|d| on row %d = %.3e  %s\n", "causality: perturbation is visible", (int) T - 1, moved,
                   moved_ok ? "OK (> 1e-2)" : "FAIL (the perturbation is absorbed - test is vacuous)");
            free(pert.data);
        }
    }

    // rollout: step 0 with a single seed frame must equal the T = 1 call
    {
        std::vector<float> steps((size_t) T * D);
        if (jepa_lewm_rollout(ctx, emb.data(), 1, act.data(), (int) T, steps.data()) != 0) { g_fail++; }
        else {
            jepa_output one = {};
            if (jepa_lewm_predict(ctx, emb.data(), act.data(), 1, &one) == 0) {
                check("rollout step 0 == predict(T=1)", compare(steps.data(), one.data, 1, D), exact);
                free(one.data);
            }
            // and step 1 must equal predict(T=2) over [emb0, pred0] with actions [a0, a1]
            std::vector<float> e2((size_t) 2 * D), a2((size_t) 2 * A);
            memcpy(e2.data(), emb.data(), (size_t) D * sizeof(float));
            memcpy(e2.data() + D, steps.data(), (size_t) D * sizeof(float));
            memcpy(a2.data(), act.data(), (size_t) 2 * A * sizeof(float));
            jepa_output two = {};
            if (jepa_lewm_predict(ctx, e2.data(), a2.data(), 2, &two) == 0) {
                check("rollout step 1 == predict(T=2) last row",
                      compare(steps.data() + D, two.data + D, 1, D), exact);
                free(two.data);
            }
        }
    }
    free(full.data);
}

// ------------------------------------------------------------------------------------------
// V-JEPA 2 masked predictor
// ------------------------------------------------------------------------------------------
static void run_vjepa2(jepa_context * ctx, jepa_model * model, const std::string & ref,
                       const std::vector<std::string> & samples, const thresholds & thr, int modality) {
    for (const std::string & s : samples) {
        const std::string base = ref + "/" + s;
        if (!exists(base + ".predictor_last_hidden_state.npy")) { printf("  %s: no reference predictor dump\n", s.c_str()); continue; }
        int64_t n = 0, d = 0;
        std::vector<float> enc_rows = load_f32(base + ".last_hidden_state.npy", &n, &d);
        int64_t nr = 0, dr = 0;
        std::vector<float> want = load_f32(base + ".predictor_last_hidden_state.npy", &nr, &dr);
        jepa_output enc = { enc_rows.data(), n, d };
        std::vector<int32_t> ids((size_t) n);
        for (int64_t i = 0; i < n; i++) ids[i] = (int32_t) i;
        jepa_output out = {};
        if (jepa_predict_mod(ctx, &enc, ids.data(), (int) n, ids.data(), (int) n, 1, modality, &out) != 0) {
            printf("  %s: jepa_predict failed\n", s.c_str());
            g_fail++;
            continue;
        }
        check((s + " predictor_last_hidden_state").c_str(), compare(out.data, want.data(), nr, dr), thr,
              jepa_context_last_compute_ms(ctx));
        free(out.data);
    }
}

// numpy cross-check case: DIR/{enc,ctx_idx,tgt_idx,pred}.npy, written by running
// scripts/jepa_convert/vjepa2_numpy_ref.py::predictor_forward on reference encoder tokens (the
// snippet that generates one is in docs/parity.md, "Results - predictors").  The case has to be
// generated from the SAME GGUF that is tested: the spec runs that file's weights.
static void run_case(jepa_context * ctx, const std::string & dir, const thresholds & thr, int modality) {
    int64_t n = 0, d = 0;
    std::vector<float> enc_rows = load_f32(dir + "/enc.npy", &n, &d);
    npy::Array ci = npy::load(dir + "/ctx_idx.npy"), ti = npy::load(dir + "/tgt_idx.npy");
    std::vector<float> cif = ci.to_f32(), tif = ti.to_f32();
    std::vector<int32_t> cidx(cif.begin(), cif.end()), tidx(tif.begin(), tif.end());
    int64_t nr = 0, dr = 0;
    std::vector<float> want = load_f32(dir + "/pred.npy", &nr, &dr);
    jepa_output enc = { enc_rows.data(), n, d };
    jepa_output out = {};
    if (jepa_predict_mod(ctx, &enc, cidx.data(), (int) cidx.size(), tidx.data(), (int) tidx.size(), 1, modality, &out) != 0) {
        printf("  case %s: jepa_predict failed\n", dir.c_str());
        g_fail++;
        return;
    }
    char name[160];
    snprintf(name, sizeof(name), "case %s (ctx=%zu tgt=%zu)", dir.c_str(), cidx.size(), tidx.size());
    check(name, compare(out.data, want.data(), nr, dr), thr, jepa_context_last_compute_ms(ctx));
    free(out.data);
}

int main(int argc, char ** argv) {
    std::string lewm_path, vjepa2_path, ref, case_dir, samples_arg = "archery_f16", modality_arg = "video";
    jepa_context_params cp = jepa_context_default_params();
    jepa_model_params   mp = jepa_model_default_params();
    double min_cos = -1, max_rel = -1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--lewm") lewm_path = next();
        else if (a == "--vjepa2") vjepa2_path = next();
        else if (a == "--ref") ref = next();
        else if (a == "--case") case_dir = next();
        else if (a == "--samples") samples_arg = next();
        else if (a == "--threads" || a == "-t") cp.n_threads = atoi(next());
        else if (a == "--gpu") {
            mp.device = 0;
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') mp.device = atoi(argv[++i]);
        }
        else if (a == "--no-flash") cp.use_flash_attn = false;
        else if (a == "--kv-f32") cp.flash_kv = JEPA_KV_F32;
        else if (a == "--kv-f16") cp.flash_kv = JEPA_KV_F16;
        else if (a == "--min-cos") min_cos = atof(next());
        else if (a == "--modality") modality_arg = next();
        else if (a == "--quiet") g_quiet = true;
        else if (a == "-h" || a == "--help") {
            printf("usage: %s --lewm MODEL.gguf --ref REFDIR | --vjepa2 MODEL.gguf {--ref REFDIR [--samples a,b] | --case DIR}\n"
                   "       [--threads N] [--gpu [N]] [--no-flash] [--kv-f32|--kv-f16] [--min-cos X]\n"
                   "       [--modality auto|video|image]   (V-JEPA 2.1 pred.mod_embed_*, default video)\n"
                   "       [--quiet]\n", argv[0]);
            return 0;
        } else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (lewm_path.empty() == vjepa2_path.empty()) { fprintf(stderr, "need exactly one of --lewm / --vjepa2\n"); return 2; }
    const int modality = modality_arg == "auto" ? JEPA_MODALITY_AUTO : modality_arg == "image" ? JEPA_MODALITY_IMAGE
                       : modality_arg == "video" ? JEPA_MODALITY_VIDEO : -1;
    if (modality < 0) { fprintf(stderr, "--modality must be auto, video or image (got '%s')\n", modality_arg.c_str()); return 2; }

    const std::string model_path = !lewm_path.empty() ? lewm_path : vjepa2_path;
    jepa_model * model = jepa_model_load_ex(model_path.c_str(), &mp);
    if (!model) return 2;
    jepa_context * c = jepa_context_new(model, cp);
    if (!c) return 2;
    const int ftype = jepa_model_file_type(model);
    thresholds thr = thresholds_for(ftype, jepa_model_is_gpu(model));
    if (min_cos > 0) { thr.min_mean = thr.min_min = min_cos; thr.max_rel = -1; }
    printf("model: %s (%s, %s) | device %s | threads %d | flash %s | modality %s | thresholds cos_mean >= %g, cos_min >= %g%s\n",
           jepa_model_name(model), jepa_model_family(model), jepa_model_file_type_name(model),
           jepa_model_device_name(model),
           jepa_context_n_threads(c), cp.use_flash_attn ? "yes" : "no", modality_arg.c_str(), thr.min_mean, thr.min_min,
           thr.max_rel > 0 ? ", rel <= bound*max(1,sqrt(rows/2048))" : "");

    if (!lewm_path.empty()) run_lewm(c, model, ref, thr, ftype != 0);
    if (!vjepa2_path.empty()) {
        if (!case_dir.empty()) run_case(c, case_dir, thr, modality);
        if (!ref.empty()) {
            std::vector<std::string> samples;
            size_t p = 0;
            while (p <= samples_arg.size() && !samples_arg.empty()) {
                size_t q = samples_arg.find(',', p);
                if (q == std::string::npos) { samples.push_back(samples_arg.substr(p)); break; }
                samples.push_back(samples_arg.substr(p, q - p));
                p = q + 1;
            }
            run_vjepa2(c, model, ref, samples, thr, modality);
        }
    }
    jepa_context_free(c);
    jepa_model_free(model);
    printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
