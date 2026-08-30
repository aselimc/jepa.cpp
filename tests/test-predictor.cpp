// test-predictor: deterministic parity checks for the two predictors.
//
//   test-predictor --lewm  MODEL.gguf --ref tests/fixtures/ref/lewm-pusht [options]
//   test-predictor --vjepa2 MODEL.gguf --ref tests/fixtures/ref/vjepa2-vitl-fpc64-256 [--samples a,b] [options]
//   test-predictor --vjepa2 MODEL.gguf --case DIR        (numpy cross-check: DIR/{enc,ctx_idx,tgt_idx,pred}.npy)
//   options: [--threads N] [--no-flash] [--kv-f32|--kv-f16] [--min-cos X] [--quiet]
//
// LeWM (f32 thresholds cos >= 0.9999):
//   * pred_next of every image sample (T = 1) and pred_seq of the 3-frame sequence sample (T = 3),
//     both against the PyTorch dump;
//   * causality: row t of the T = 3 run equals the prefix-only run over frames 0..t, and perturbing
//     a *later* frame's embedding leaves the earlier output rows bit-identical;
//   * jepa_lewm_rollout: step 0 of a rollout seeded with emb_seq[0] equals jepa_lewm_predict(T = 1).
// V-JEPA 2: the masked predictor over the reference encoder tokens with context = target = every
//   token (the HF default pass) against `predictor_last_hidden_state`.
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

// thresholds mirror tests/test-parity.cpp (docs/architecture.md): f32 mean/worst cosine >= 0.9999
// and rel_max <= 1e-3; f16 mean >= 0.9999, worst >= 0.99; quantized mean >= 0.999, worst >= 0.98.
// The worst-token floors are dtype noise, not graph error: the numpy spec on the same GGUF bottoms
// out at cos_min 0.99897 for q8_0 (weight quantisation alone) on archery_f16.
struct thresholds { double min_mean, min_min, max_rel; };

static thresholds thresholds_for(int ftype) {
    if (ftype == 0) return {0.9999, 0.9999, 1e-3};
    if (ftype == 1) return {0.9999, 0.99, -1.0};
    return {0.999, 0.98, -1.0};
}

// self-consistency (prefix vs full run, rollout vs predict): the same graph on the same numbers,
// so anything beyond float32 round-off is a bug
static const thresholds exact = {0.9999999, 0.9999999, 1e-5};

static void check(const char * what, const metrics & m, const thresholds & t, double ms = -1) {
    const double min_cos = t.min_min, max_rel = t.max_rel;
    const bool ok = m.cos_mean >= t.min_mean && m.cos_min >= min_cos && (max_rel <= 0 || m.rel_max <= max_rel);
    if (!ok) g_fail++;
    if (!g_quiet || !ok) {
        printf("  %-44s rows=%-6lld cos_mean=%.7f cos_min=%.7f max|d|=%.3e rel=%.3e", what,
               (long long) m.rows, m.cos_mean, m.cos_min, m.max_abs, m.rel_max);
        if (ms >= 0) printf(" %8.2f ms", ms);
        printf("  %s\n", ok ? "OK" : "FAIL");
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
static void run_lewm(jepa_context * ctx, jepa_model * model, const std::string & ref, const thresholds & thr) {
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
        check(name, compare(pre.data + (size_t) t * D, full.data + (size_t) t * D, 1, D), exact);
        free(pre.data);
    }

    // causality (b): perturbing the LAST frame must not move the earlier output rows at all
    {
        std::vector<float> emb2 = emb, act2 = act;
        for (int i = 0; i < D; i++) emb2[(size_t) (T - 1) * D + i] += 3.0f;
        for (int i = 0; i < A; i++) act2[(size_t) (T - 1) * A + i] += 3.0f;
        jepa_output pert = {};
        if (jepa_lewm_predict(ctx, emb2.data(), act2.data(), (int) T, &pert) != 0) { g_fail++; }
        else {
            double worst = 0;
            for (int64_t i = 0; i < (T - 1) * D; i++) worst = std::fmax(worst, std::fabs(pert.data[i] - full.data[i]));
            const bool ok = worst == 0.0;
            if (!ok) g_fail++;
            printf("  %-44s max|d| on rows 0..%d = %.3e  %s\n", "causality: perturb frame T-1", (int) T - 2, worst,
                   ok ? "OK (bit-identical)" : "FAIL");
            double moved = 0;
            for (int64_t i = (T - 1) * D; i < T * D; i++) moved = std::fmax(moved, std::fabs(pert.data[i] - full.data[i]));
            if (moved == 0.0) { printf("  the perturbation did not change the last row either - test is vacuous\n"); g_fail++; }
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
                       const std::vector<std::string> & samples, const thresholds & thr) {
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
        if (jepa_predict(ctx, &enc, ids.data(), (int) n, ids.data(), (int) n, &out) != 0) {
            printf("  %s: jepa_predict failed\n", s.c_str());
            g_fail++;
            continue;
        }
        check((s + " predictor_last_hidden_state").c_str(), compare(out.data, want.data(), nr, dr), thr,
              jepa_context_last_compute_ms(ctx));
        free(out.data);
    }
}

// numpy cross-check case: DIR/{enc,ctx_idx,tgt_idx,pred}.npy (scripts written by the agent's
// tmp/ driver around scripts/jepa_convert/vjepa2_numpy_ref.py::predictor_forward)
static void run_case(jepa_context * ctx, const std::string & dir, const thresholds & thr) {
    int64_t n = 0, d = 0;
    std::vector<float> enc_rows = load_f32(dir + "/enc.npy", &n, &d);
    npy::Array ci = npy::load(dir + "/ctx_idx.npy"), ti = npy::load(dir + "/tgt_idx.npy");
    std::vector<float> cif = ci.to_f32(), tif = ti.to_f32();
    std::vector<int32_t> cidx(cif.begin(), cif.end()), tidx(tif.begin(), tif.end());
    int64_t nr = 0, dr = 0;
    std::vector<float> want = load_f32(dir + "/pred.npy", &nr, &dr);
    jepa_output enc = { enc_rows.data(), n, d };
    jepa_output out = {};
    if (jepa_predict(ctx, &enc, cidx.data(), (int) cidx.size(), tidx.data(), (int) tidx.size(), &out) != 0) {
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
    std::string lewm_path, vjepa2_path, ref, case_dir, samples_arg = "archery_f16";
    jepa_context_params cp = jepa_context_default_params();
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
        else if (a == "--no-flash") cp.use_flash_attn = false;
        else if (a == "--kv-f32") cp.flash_kv = JEPA_KV_F32;
        else if (a == "--kv-f16") cp.flash_kv = JEPA_KV_F16;
        else if (a == "--min-cos") min_cos = atof(next());
        else if (a == "--quiet") g_quiet = true;
        else if (a == "-h" || a == "--help") {
            printf("usage: %s --lewm MODEL.gguf --ref REFDIR | --vjepa2 MODEL.gguf {--ref REFDIR [--samples a,b] | --case DIR}\n"
                   "       [--threads N] [--no-flash] [--kv-f32|--kv-f16] [--min-cos X] [--quiet]\n", argv[0]);
            return 0;
        } else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (lewm_path.empty() == vjepa2_path.empty()) { fprintf(stderr, "need exactly one of --lewm / --vjepa2\n"); return 2; }

    const std::string model_path = !lewm_path.empty() ? lewm_path : vjepa2_path;
    jepa_model * model = jepa_model_load(model_path.c_str(), false);
    if (!model) return 2;
    jepa_context * c = jepa_context_new(model, cp);
    if (!c) return 2;
    const int ftype = jepa_model_file_type(model);
    thresholds thr = thresholds_for(ftype);
    if (min_cos > 0) { thr.min_mean = thr.min_min = min_cos; thr.max_rel = -1; }
    printf("model: %s (%s, %s) | threads %d | flash %s | thresholds cos_mean >= %g, cos_min >= %g%s\n",
           jepa_model_name(model), jepa_model_family(model), jepa_model_file_type_name(model),
           jepa_context_n_threads(c), cp.use_flash_attn ? "yes" : "no", thr.min_mean, thr.min_min,
           thr.max_rel > 0 ? ", rel <= 1e-3" : "");

    if (!lewm_path.empty()) run_lewm(c, model, ref, thr);
    if (!vjepa2_path.empty()) {
        if (!case_dir.empty()) run_case(c, case_dir, thr);
        if (!ref.empty()) {
            std::vector<std::string> samples;
            size_t p = 0;
            while (p <= samples_arg.size() && !samples_arg.empty()) {
                size_t q = samples_arg.find(',', p);
                if (q == std::string::npos) { samples.push_back(samples_arg.substr(p)); break; }
                samples.push_back(samples_arg.substr(p, q - p));
                p = q + 1;
            }
            run_vjepa2(c, model, ref, samples, thr);
        }
    }
    jepa_context_free(c);
    jepa_model_free(model);
    printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
