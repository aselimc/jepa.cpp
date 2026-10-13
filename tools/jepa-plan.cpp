// jepa-plan — one call of the H-JEPA planner: from an observation image and a goal image to the
// level-1 actions that move the first towards the second.
//
//   jepa-plan -m MODEL.gguf --obs now.png --goal goal.png [options]
//
// The hierarchy is planned top-down (jepa_hjepa_plan): the top level optimises a few of its
// macro-actions towards the goal, and each level below is asked to reach the first state the level
// above predicted. What comes out is the level-1 plan, printed both in the model's units (z-scored,
// frameskip-stacked) and, when the file carries the training statistics, as raw environment actions.
//
// This is one planner call. A controller executes the first few actions, observes again and calls
// it again with the horizons of the next call (jepa_hjepa_horizons, or --budget / --step here).
#include "jepa.h"
#include "jepa-args.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    printf("usage: %s -m MODEL.gguf --obs IMAGE --goal IMAGE [options]\n"
           "  --horizons a,b,..   configured horizon per level, level 1 first (default: 2 x the stride of the\n"
           "                      level above; 3 at the top; 4 for a 1-level model)\n"
           "  --samples a,b,..    candidates per level (default 100; one value applies to every level)\n"
           "  --steps N           gradient iterations per level at most (default 90)\n"
           "  --lr a,b,..         AdamW learning rate per level (default 0.1)\n"
           "  --cost-last-n a,..  predicted states scored per level (default 1)\n"
           "  --action-cost W     weight of the cross-level action cost at level 1 (default 0)\n"
           "  --budget N --step K --interval R\n"
           "                      plan as the call after K of N environment steps, replanning every R:\n"
           "                      horizons then follow the reference's decreasing schedule (default: use\n"
           "                      the configured horizons as they are)\n"
           "  --proprio a,b,..  --goal-proprio a,b,..\n"
           "                      raw proprioception of the observation / the goal, for models that read it\n"
           "                      (default: the training mean)\n"
           "  --threads N         CPU threads (the planner's gradient runs on the host)\n"
           JEPA_GPU_USAGE
           "  --quiet             only the actions\n", prog);
}

static std::vector<float> parse_list(const char * s) {
    std::vector<float> v;
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        const float x = strtof(p, &end);
        if (end == p) { fprintf(stderr, "cannot parse '%s' as a comma-separated list of numbers\n", s); exit(2); }
        v.push_back(x);
        p = *end == ',' ? end + 1 : end;
        if (*end != ',' && *end != '\0') { fprintf(stderr, "cannot parse '%s' as a comma-separated list of numbers\n", s); exit(2); }
    }
    return v;
}

static float pick(const std::vector<float> & v, int level, float def) {
    if (v.empty()) return def;
    return v.size() == 1 ? v[0] : v[(size_t) level < v.size() ? (size_t) level : v.size() - 1];
}

// Every level's embedding of one image: encoder CLS -> level 1 -> lifted through the hierarchy.
static bool embed_levels(jepa_context * ctx, jepa_model * model, const char * path, const std::vector<float> & proprio,
                         std::vector<std::vector<float>> & out) {
    int h = 0, w = 0;
    float * pix = jepa_preprocess_image_file(model, path, &h, &w);
    if (!pix) { fprintf(stderr, "cannot read %s: %s\n", path, jepa_error_text()); return false; }
    jepa_input in = {};
    in.data = pix; in.n_batch = 1; in.n_chans = 3; in.n_frames = 1; in.height = h; in.width = w;
    jepa_output enc = {}, st = {};
    const bool ok = jepa_encode(ctx, &in, &enc) == 0 &&
                    jepa_hjepa_embed(ctx, enc.data, proprio.empty() ? nullptr : proprio.data(), 1, &st) == 0;
    jepa_free(pix);
    jepa_free(enc.data);
    if (!ok) { fprintf(stderr, "cannot encode %s: %s\n", path, jepa_error_text()); return false; }
    out.clear();
    out.emplace_back(st.data, st.data + st.dim);
    jepa_free(st.data);
    for (int k = 2; k <= jepa_hjepa_n_levels(model); k++) {
        jepa_output up = {};
        if (jepa_hjepa_lift(ctx, k, out.back().data(), 1, &up) != 0) { fprintf(stderr, "lift to level %d failed: %s\n", k, jepa_error_text()); return false; }
        out.emplace_back(up.data, up.data + up.dim);
        jepa_free(up.data);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, obs_path, goal_path;
    std::vector<float> horizons, samples, lrs, cost_last_n, proprio_raw, goal_proprio_raw;
    int steps = 90, threads = 0, device = -1, budget = 0, step = 0, interval = 1;
    float action_cost = 0.0f;
    bool quiet = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--obs") obs_path = next();
        else if (a == "--goal") goal_path = next();
        else if (a == "--horizons") horizons = parse_list(next());
        else if (a == "--samples") samples = parse_list(next());
        else if (a == "--lr") lrs = parse_list(next());
        else if (a == "--cost-last-n") cost_last_n = parse_list(next());
        else if (a == "--steps") steps = atoi(next());
        else if (a == "--action-cost") action_cost = strtof(next(), nullptr);
        else if (a == "--budget") budget = atoi(next());
        else if (a == "--step") step = atoi(next());
        else if (a == "--interval") interval = atoi(next());
        else if (a == "--proprio") proprio_raw = parse_list(next());
        else if (a == "--goal-proprio") goal_proprio_raw = parse_list(next());
        else if (a == "--threads" || a == "-t") threads = atoi(next());
        else if (jepa_arg_gpu(argc, argv, i, device)) {}
        else if (a == "--quiet") quiet = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || obs_path.empty() || goal_path.empty()) { usage(argv[0]); return 2; }

    jepa_model_params mp = jepa_model_default_params();
    if (device >= 0) mp.device = device;
    jepa_model * model = jepa_model_load_ex(model_path.c_str(), &mp);
    if (!model) { fprintf(stderr, "cannot load %s: %s\n", model_path.c_str(), jepa_error_text()); return 1; }
    const int n_levels = jepa_hjepa_n_levels(model);
    if (n_levels < 1) { fprintf(stderr, "%s is not an H-JEPA model (family %s)\n", model_path.c_str(), jepa_model_family(model)); return 1; }
    jepa_context_params cp = jepa_context_default_params();
    if (threads > 0) cp.n_threads = threads;
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create a context: %s\n", jepa_error_text()); return 1; }

    // proprioception: z-score what was given, or use the training mean (all zeros after z-scoring)
    const int P = jepa_hjepa_proprio_dim(model);
    std::vector<float> pmean((size_t) (P > 0 ? P : 1)), pstd((size_t) (P > 0 ? P : 1), 1.0f);
    const bool have_pstats = P > 0 && jepa_hjepa_norm_stats(model, "proprio", pmean.data(), pstd.data(), P) == P;
    auto proprio_of = [&](const std::vector<float> & raw, const char * what) {
        std::vector<float> z((size_t) P, 0.0f);
        if (raw.empty()) return z;
        if ((int) raw.size() != P || !have_pstats) {
            fprintf(stderr, "%s: the model reads %d proprio values%s, %zu were given\n", what, P,
                    have_pstats ? "" : " and carries no statistics to normalise them with", raw.size());
            exit(2);
        }
        for (int i = 0; i < P; i++) z[(size_t) i] = (raw[(size_t) i] - pmean[(size_t) i]) / pstd[(size_t) i];
        return z;
    };
    std::vector<std::vector<float>> st, gl;
    if (!embed_levels(ctx, model, obs_path.c_str(), P > 0 ? proprio_of(proprio_raw, "--proprio") : std::vector<float>(), st) ||
        !embed_levels(ctx, model, goal_path.c_str(), P > 0 ? proprio_of(goal_proprio_raw, "--goal-proprio") : std::vector<float>(), gl)) {
        return 1;
    }

    std::vector<jepa_hjepa_level_plan> cfg((size_t) n_levels, jepa_hjepa_level_plan_default());
    for (int k = 0; k < n_levels; k++) {
        jepa_hjepa_level_plan & p = cfg[(size_t) k];
        const int def_h = n_levels == 1 ? 4 : k == n_levels - 1 ? 3 : 2 * jepa_hjepa_stride(model, k + 2);
        p.horizon = (int) pick(horizons, k, (float) def_h);
        p.n_samples = (int) pick(samples, k, 100.0f);
        p.n_steps = steps;
        p.lr = pick(lrs, k, 0.1f);
        p.cost_last_n = (int) pick(cost_last_n, k, 1.0f);
        p.action_cost_weight = k == 0 ? action_cost : 0.0f;
        p.horizon_one_goal_upper = k > 0;
    }
    std::vector<jepa_hjepa_level_plan> plan = cfg;
    if (budget > 0) {
        std::vector<int> h((size_t) n_levels);
        if (jepa_hjepa_horizons(model, cfg.data(), n_levels, interval, budget, step, h.data()) != 0) {
            fprintf(stderr, "horizon schedule: %s\n", jepa_error_text());
            return 1;
        }
        for (int k = 0; k < n_levels; k++) plan[(size_t) k].horizon = h[(size_t) k];
    }

    std::vector<const float *> stp, glp;
    for (int k = 0; k < n_levels; k++) { stp.push_back(st[(size_t) k].data()); glp.push_back(gl[(size_t) k].data()); }
    const int A = jepa_hjepa_action_dim(model, 1), T = plan[0].horizon;
    std::vector<float> actions((size_t) T * A);
    jepa_hjepa_plan_info info;
    const auto t0 = std::chrono::steady_clock::now();
    if (jepa_hjepa_plan(ctx, stp.data(), glp.data(), plan.data(), n_levels, nullptr, nullptr, actions.data(), &info) != 0) {
        fprintf(stderr, "planning failed: %s\n", jepa_error_text());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (!quiet) {
        printf("model: %s (%d level%s, %s) | threads %d\n", jepa_model_name(model), n_levels, n_levels == 1 ? "" : "s",
               jepa_model_file_type_name(model), jepa_context_n_threads(ctx));
        for (int k = n_levels - 1; k >= 0; k--) {
            if (!info.planned[k]) { printf("level %d: skipped (horizon 1)\n", k + 1); continue; }
            printf("level %d: horizon %d, %d samples, %d iterations, cost %.6g in level-%d space\n", k + 1, plan[(size_t) k].horizon,
                   plan[(size_t) k].n_samples, info.iters[k], info.cost[k], info.goal_level[k]);
        }
        printf("planned in %.0f ms\n", ms);
        printf("level-1 actions, model units [%d x %d]:\n", T, A);
    }
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < A; i++) printf("%s%.6f", i ? " " : "", actions[(size_t) t * A + i]);
        printf("\n");
    }
    // raw environment actions: undo the z-score, one row per environment step
    const int R = jepa_hjepa_raw_action_dim(model), F = jepa_hjepa_frameskip(model);
    if (!quiet && R > 0 && F > 0 && R * F == A) {
        std::vector<float> mean((size_t) R), sd((size_t) R);
        if (jepa_hjepa_norm_stats(model, "action", mean.data(), sd.data(), R) == R) {
            printf("raw environment actions [%d x %d]:\n", T * F, R);
            for (int t = 0; t < T * F; t++) {
                for (int i = 0; i < R; i++) printf("%s%.6f", i ? " " : "", actions[(size_t) t * R + i] * sd[(size_t) i] + mean[(size_t) i]);
                printf("\n");
            }
        }
    }
    jepa_context_free(ctx);
    jepa_model_free(model);
    return 0;
}
