// test-threads — the thread-safety contract of include/jepa.h, checked rather than asserted in prose.
//
//   test-threads [--model MODEL.gguf] [--threads N] [--rounds N] [-v]
//
// One shared jepa_model, one jepa_context per thread, N threads encoding the same input at the same
// time. What the suite proves:
//
//   1. bit-identity  — every concurrent encode returns exactly the bytes a single-threaded encode of
//                      the same input returns, at the same n_threads. Not "close": memcmp.
//   2. no shared mutable state — the per-thread pools, CLS rows and predictor outputs agree too, so
//                      a context cannot be leaking graph state into its neighbours.
//   3. jepa_error_text is thread-local — every thread makes its own call fail, with a message only
//                      it can produce, and reads back only its own.
//   4. introspection is re-entrant — jepa_model_* and the preprocessing entry points are hammered
//                      from every thread against a single-threaded reference.
//
// Run it under TSAN to make (2) a statement about the implementation and not just the outputs:
//   cmake -B build-tsan -DCMAKE_CXX_FLAGS=-fsanitize=thread -DGGML_OPENMP=OFF ...
// GGML_OPENMP=OFF matters: TSAN cannot see through libgomp's own synchronisation and reports its
// thread pool as a race in any program that uses it.
//
// Without --model the suite forges one (tests/forge-gguf.h, the same 8-dimension hfvit
// tests/test-errors.cpp breaks a knob of): every check above then runs on a runner with no weights,
// which is what lets the ASAN+UBSAN CI job run this suite at full strength. --model repeats it on a
// real checkpoint.
#include "jepa.h"
#include "forge-gguf.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  include <direct.h>
#  include <process.h>
#  define jepa_mkdir(p) _mkdir(p)
#  define jepa_getpid() _getpid()
#else
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#  define jepa_mkdir(p) mkdir((p), 0755)
#  define jepa_getpid() getpid()
#endif

static int g_fail = 0, g_pass = 0;
static bool g_verbose = false;

static void ok(const char * name, bool cond, const char * detail = "") {
    if (cond) { g_pass++; if (g_verbose) printf("ok   %-38s %s\n", name, detail); return; }
    printf("FAIL %-38s %s\n", name, detail);
    g_fail++;
}

struct owned_output {
    std::vector<float> rows;
    int64_t n_tokens = 0, dim = 0;
    bool ok = false;

    bool same(const owned_output & o) const {
        return ok && o.ok && n_tokens == o.n_tokens && dim == o.dim && rows.size() == o.rows.size() &&
               memcmp(rows.data(), o.rows.data(), rows.size() * sizeof(float)) == 0;
    }
};

static owned_output take(jepa_output * out) {
    owned_output r;
    if (out->data && out->n_tokens > 0 && out->dim > 0) {
        r.n_tokens = out->n_tokens;
        r.dim = out->dim;
        r.rows.assign(out->data, out->data + (size_t) out->n_tokens * out->dim);
        r.ok = true;
    }
    jepa_free(out->data);
    out->data = nullptr;
    return r;
}

// One worker's whole result set, so a divergence anywhere shows up as a mismatch by name.
struct work_result {
    owned_output enc, mean, cls, proj, logits, pred;
    std::string  error_text;       // whatever THIS thread's jepa_error_text() held after its own failure
    bool         made_context = false;
};

// Everything one thread does with the shared model. `ctx` is created and destroyed inside, so the
// per-thread-context half of the contract is what is being exercised, not assumed.
static work_result run_worker(jepa_model * m, const std::vector<float> & px, int n_frames,
                              int h, int w, int n_threads, int rounds, int tag) {
    work_result r;
    jepa_context_params cp = jepa_context_default_params();
    cp.n_threads = n_threads;
    jepa_context * ctx = jepa_context_new(m, cp);
    if (!ctx) return r;
    r.made_context = true;

    const int C = 3;
    jepa_input in = { px.data(), 1, C, n_frames, h, w };
    for (int round = 0; round < rounds; round++) {
        jepa_output enc = {};
        if (jepa_encode(ctx, &in, &enc) != 0) { jepa_context_free(ctx); return r; }
        owned_output e = take(&enc);
        if (round == 0) r.enc = e;
        else if (!r.enc.same(e)) { r.enc.ok = false; break; }   // not even stable within one thread

        jepa_output view = { r.enc.rows.data(), r.enc.n_tokens, r.enc.dim };
        jepa_output tmp = {};
        if (jepa_pool_mean(m, &view, &tmp) == 0) r.mean = take(&tmp);
        if (jepa_model_has_cls(m) && jepa_pool_cls(m, &view, &tmp) == 0) r.cls = take(&tmp);
        if (jepa_model_has_projector(m) && jepa_lewm_project(ctx, &view, &tmp) == 0) r.proj = take(&tmp);
        if (jepa_model_has_head(m) && jepa_head(ctx, &view, &tmp) == 0) r.logits = take(&tmp);
        if (jepa_model_has_predictor(m) && r.enc.n_tokens > 4) {
            const int32_t ids[4] = { 0, 1, 2, 3 };
            if (jepa_predict(ctx, &view, ids, 2, ids + 2, 2, &tmp) == 0) r.pred = take(&tmp);
        }
    }

    // The capture buffer is thread-local, so a failure raised here must be readable here and
    // invisible to every other thread. The marker is this worker's own token count, which no other
    // worker asks for.
    jepa_error_reset();
    jepa_input bad = { px.data(), 1, C, n_frames, h, -(tag + 1) };
    jepa_output dummy = {};
    if (jepa_encode(ctx, &bad, &dummy) == 0) jepa_free(dummy.data);
    r.error_text = jepa_error_text();

    jepa_context_free(ctx);
    return r;
}

// The models this run will use: whatever --model gave, plus a forged one when it gave none.
// `scratch` receives the forged file's path so the caller can delete it.
static std::vector<std::string> resolve_models(std::vector<std::string> models, std::string & scratch) {
    if (!models.empty()) return models;
    const char * base = getenv("TMPDIR");
    if (!base || !*base) base = getenv("TEMP");
    const std::string dir = std::string(base && *base ? base : "/tmp") + "/jepa-test-threads-" +
                            std::to_string((long) jepa_getpid());
    if (jepa_mkdir(dir.c_str()) != 0) {
        fprintf(stderr, "cannot create %s\n", dir.c_str());
        return models;
    }
    scratch = dir + "/forged.gguf";
    forge(scratch, forge_opts{});
    models.push_back(scratch);
    return models;
}

// Everything the contract says about one model, on `n_workers` threads at once.
static void run_for_model(const std::string & model, int n_workers, int rounds, int n_threads) {
    jepa_error_reset();
    jepa_model * m = jepa_model_load(model.c_str(), false);
    if (!m) {
        printf("FAIL cannot load %s: %s\n", model.c_str(), jepa_error_text());
        g_fail++;
        return;
    }

    // One input, the model's own crop, deterministic content.
    const int S = jepa_model_img_size(m);
    const int P = jepa_model_patch_size(m);
    const int T = jepa_model_n_frames(m) > 0 ? jepa_model_n_frames(m) : 1;
    if (S <= 0 || P <= 0 || S % P != 0) {
        printf("FAIL %s: img_size %d is not a multiple of patch_size %d\n", model.c_str(), S, P);
        jepa_model_free(m);
        g_fail++;
        return;
    }
    std::vector<float> px((size_t) 3 * T * S * S);
    for (size_t i = 0; i < px.size(); i++) px[i] = 0.4f * (float) ((i * 17 % 23) - 11);

    // The reference: the same work, alone, in this thread.
    const work_result ref = run_worker(m, px, T, S, S, n_threads, rounds, 0);
    ok("reference/encoded", ref.enc.ok, ref.enc.ok ? "" : "the single-threaded run did not produce tokens");
    if (!ref.enc.ok) { jepa_model_free(m); return; }

    // N workers, one context each, all on the shared model at once.
    std::vector<work_result> res((size_t) n_workers);
    std::vector<std::thread> ts;
    for (int t = 0; t < n_workers; t++) {
        ts.emplace_back([&, t] { res[(size_t) t] = run_worker(m, px, T, S, S, n_threads, rounds, t + 1); });
    }
    for (auto & th : ts) th.join();

    int made = 0, enc_same = 0, mean_same = 0, cls_same = 0, proj_same = 0, logit_same = 0, pred_same = 0;
    int own_text = 0;
    for (int t = 0; t < n_workers; t++) {
        const work_result & r = res[(size_t) t];
        made += r.made_context;
        enc_same += r.enc.same(ref.enc);
        mean_same += !ref.mean.ok || r.mean.same(ref.mean);
        cls_same += !ref.cls.ok || r.cls.same(ref.cls);
        proj_same += !ref.proj.ok || r.proj.same(ref.proj);
        logit_same += !ref.logits.ok || r.logits.same(ref.logits);
        pred_same += !ref.pred.ok || r.pred.same(ref.pred);
        // Worker t was given tag t+1 and asked to encode a width of -(tag+1); the message it read
        // back has to name that width, closed by the "]" of the shape, so "-2]" cannot match "-21]".
        own_text += r.error_text.find(std::to_string(-(t + 2)) + "]") != std::string::npos;
    }
    char buf[200];
    snprintf(buf, sizeof buf, "%d workers, %d tokens x %d dims, %d rounds each, %s",
             n_workers, (int) ref.enc.n_tokens, (int) ref.enc.dim, rounds, jepa_model_family(m));
    ok("contexts/one per thread", made == n_workers, buf);
    ok("encode/bit-identical to single-threaded", enc_same == n_workers, buf);
    ok("pool_mean/bit-identical", mean_same == n_workers);
    ok("pool_cls/bit-identical", cls_same == n_workers);
    ok("lewm_project/bit-identical", proj_same == n_workers);
    ok("head/bit-identical", logit_same == n_workers);
    ok("predict/bit-identical", pred_same == n_workers);
    ok("error-text/thread-local", own_text == n_workers, "each worker read back only its own message");

    // Introspection and preprocessing from every thread at once, against a single-threaded answer.
    {
        int oh = 0, ow = 0;
        std::vector<uint8_t> rgb((size_t) 3 * 64 * 64);
        for (size_t i = 0; i < rgb.size(); i++) rgb[i] = (uint8_t) (i * 7 + 3);
        float * ref_px = jepa_preprocess_image_rgb(m, rgb.data(), 64, 64, &oh, &ow);
        const size_t n_ref = ref_px ? (size_t) 3 * oh * ow : 0;
        std::atomic<int> bad{0};
        std::vector<std::thread> ts2;
        for (int t = 0; t < n_workers; t++) {
            ts2.emplace_back([&] {
                for (int k = 0; k < 20; k++) {
                    if (strcmp(jepa_model_family(m), jepa_model_family(m)) != 0) bad++;
                    if (jepa_model_embed_dim(m) != (int) ref.enc.dim) bad++;
                    int h2 = 0, w2 = 0;
                    float * mine = jepa_preprocess_image_rgb(m, rgb.data(), 64, 64, &h2, &w2);
                    if (!mine || !ref_px || h2 != oh || w2 != ow ||
                        memcmp(mine, ref_px, n_ref * sizeof(float)) != 0) bad++;
                    jepa_free(mine);
                }
            });
        }
        for (auto & th : ts2) th.join();
        ok("introspection+preprocess/re-entrant", bad.load() == 0);
        jepa_free(ref_px);
    }

    jepa_model_free(m);
}

int main(int argc, char ** argv) {
    std::vector<std::string> models;
    int n_workers = 4, rounds = 2, n_threads = 2;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc)        models.push_back(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) n_workers = atoi(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc)  rounds = atoi(argv[++i]);
        else if (a == "--inner" && i + 1 < argc)   n_threads = atoi(argv[++i]);
        else if (a == "-v" || a == "--verbose")    g_verbose = true;
        else {
            fprintf(stderr, "usage: %s [--model M.gguf ...] [--threads N] [--rounds N] [--inner N] [-v]\n", argv[0]);
            return 2;
        }
    }
    if (n_workers < 2) n_workers = 2;
    if (rounds < 1) rounds = 1;

    // Thread-local error capture with no model in sight: each thread asks for a load failure only
    // it can name, and must read back only its own message.
    {
        std::atomic<int> mismatches{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < n_workers; t++) {
            ts.emplace_back([t, &mismatches] {
                for (int k = 0; k < 200; k++) {
                    jepa_error_reset();
                    const std::string path = "/nonexistent/jepa-thread-" + std::to_string(t) + ".gguf";
                    jepa_model_free(jepa_model_load(path.c_str(), false));
                    if (strstr(jepa_error_text(), path.c_str()) == nullptr) mismatches++;
                    float logits[4] = { 1.0f, 2.0f, 3.0f, 0.5f };
                    float probs[4];
                    int32_t idx[2];
                    jepa_softmax(logits, 4, probs);
                    if (jepa_top_k(logits, 4, 2, idx) != 2 || idx[0] != 2 || idx[1] != 1) mismatches++;
                }
            });
        }
        for (auto & th : ts) th.join();
        ok("error-text/thread-local (no model)", mismatches.load() == 0);
        ok("helpers/re-entrant (no model)", mismatches.load() == 0);
    }

    std::string scratch;
    const std::vector<std::string> use = resolve_models(models, scratch);
    if (use.empty()) { printf("FAIL no model was given and none could be forged\n"); return 1; }
    for (const std::string & path : use) run_for_model(path, n_workers, rounds, n_threads);

    std::string names;
    for (const std::string & p : use) names += (names.empty() ? "" : ", ") + p;
    if (!scratch.empty()) {
        remove(scratch.c_str());
        const size_t slash = scratch.rfind('/');
        if (slash != std::string::npos) {
#ifdef _WIN32
            _rmdir(scratch.substr(0, slash).c_str());
#else
            rmdir(scratch.substr(0, slash).c_str());
#endif
        }
    }
    printf("test-threads: %d passed, %d failed (%d workers x %d inner threads on %s)\n",
           g_pass, g_fail, n_workers, n_threads, names.c_str());
    return g_fail == 0 ? 0 : 1;
}
