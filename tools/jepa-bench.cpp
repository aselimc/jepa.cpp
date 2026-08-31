// jepa-bench: timing harness for every graph jepa.cpp can run.
//
//   jepa-bench -m model.gguf [--mode encoder|head|predictor|lewm-step|lewm-rollout]
//              [--frames N] [--size HxW] [--batch B] [--threads 32,96]
//              [--repeat R] [--warmup W] [--kv-f16|--kv-f32] [--no-flash]
//              [--seed S] [--steps K] [--md] [--json out.json] [--ftype-label NAME] [-v]
//
// The input is synthetic but deterministic: a seeded xorshift generates uint8 "pixels" that go through
// the model's own normalisation ((px/255 - mean)/std, from jepa.pre.*), so the tensor that reaches the
// graph has the scale of a real preprocessed clip and every run of a given (model, shape, seed) sees
// the same numbers. Nothing here reads an image or a fixture, so the tool runs wherever a GGUF is.
//
// Only include/jepa.h is used — the tool doubles as a compile+link smoke test of the public API.
//
// Modes:
//   encoder      the encoder graph on one item at the model's native crop (video: --frames frames,
//                default min(jepa.enc.n_frames, 16)); --batch B puts B items through one call, and
//                for the image families that is a single graph carrying them on the batch dimension
//                (jepa_context_set_max_batch is raised to B so no chunking happens inside). The
//                reported ms is the whole call; `ms/item` next to it is that divided by B.
//   head         encoder once, then the attentive-pool head R times (requires jepa.head).
//   predictor    encoder once, then the masked predictor with context = target = every token.
//   lewm-step    one LeWM predictor call over jepa.pred.n_frames frames.
//   lewm-rollout autoregressive rollout of --steps (default 20) steps; ms is reported per step, and
//                a step emits exactly one embedding, so `tokens` is 1 per step and tokens/s = steps/s.
//
// The reported ms is the wall time of ggml_backend_graph_compute (jepa_context_last_compute_ms), i.e.
// graph build/alloc and the host-side patchify are excluded; `wall_ms` in the JSON is the full API
// call. With --threads a,b a context is created per thread count and each gets its own warmup.
//
// In --mode head/predictor the encoder is run three times (one warmup + two measured) and the
// *minimum* of the measured passes is reported as `encoder_ms`, so a burst of load on the box during
// one pass does not become the row's encoder figure. It is still a single graph, not an average.
#include "jepa.h"
#include "jepa-args.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s -m model.gguf [options]\n"
        "  --mode M          encoder (default) | head | predictor | lewm-step | lewm-rollout\n"
        "  --frames N        frames per item (video models; default min(jepa.enc.n_frames, 16))\n"
        "  --size HxW        input crop (default: the model's jepa.enc.img_size square)\n"
        "  --batch B         items per encoder call (default 1); image families put all B through ONE graph\n"
        "  --threads L       thread count, or a comma-separated list to sweep (default: all cores)\n"
        JEPA_GPU_USAGE
        "  --gpu-prec P      f32 (default on a GPU: GGML_PREC_F32 accumulation in every mul_mat)\n"
        "                    or f16 (cuBLAS' own compute type; faster, 177x wider f16 error)\n"
        "  --repeat R        measured runs (default 3)\n"
        "  --warmup W        unmeasured runs before them (default 1)\n"
        "  --steps K         lewm-rollout steps (default 20)\n"
        "  --seed S          RNG seed for the synthetic input (default 1234)\n"
        "  --kv-f16/--kv-f32 K/V dtype in flash attention (default: f32 for f32 files, f16 otherwise)\n"
        "  --no-flash        naive attention (mul_mat + soft_max) instead of ggml_flash_attn_ext\n"
        "  --md              one markdown table row per config (header printed before the first)\n"
        "  --no-md-header    suppress that header (for appending rows to an existing table)\n"
        "  --json FILE       write every run of this process as JSON\n"
        "  --label NAME      override the model name printed/emitted\n"
        "  --ftype-label N   override the file type printed/emitted (general.file_type is the most\n"
        "                    common stored tensor type, so a q4_k mix with more q4_0 fallbacks than\n"
        "                    q4_k tensors reads as q4_0; pass the type that was asked for)\n"
        "  -v                verbose model load + ggml system info\n", a0);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Peak resident set size of this process in bytes (Linux VmHWM; 0 if unavailable).
static size_t peak_rss_bytes() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) { kb = (size_t) strtoull(line + 6, nullptr, 10); break; }
    }
    fclose(f);
    return kb * 1024;
}

// Deterministic pseudo-random stream (xorshift64*), independent of the libstdc++ version.
struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1Dull; }
    uint8_t  u8()   { return (uint8_t) (next() >> 33); }
    // ~U(-1, 1), the scale of a projected LeWM embedding / a normalised action
    float    sym()  { return (float) (double) ((int64_t) (next() >> 20) - (1LL << 43)) / (float) (1LL << 43); }
};

// Synthetic NCTHW input with the scale of a real preprocessed item.
static std::vector<float> make_input(const jepa_model * model, int n_batch, int n_frames, int h, int w, uint64_t seed) {
    const jepa_preprocess_params p = jepa_preprocess_default_params(model);
    rng r(seed);
    std::vector<float> x((size_t) n_batch * 3 * n_frames * h * w);
    size_t i = 0;
    for (int b = 0; b < n_batch; b++) {
        for (int c = 0; c < 3; c++) {
            const float mean = p.mean[c];
            const float istd = 1.0f / (p.std[c] != 0.0f ? p.std[c] : 1.0f);
            for (int t = 0; t < n_frames; t++) {
                for (int k = 0; k < h * w; k++) x[i++] = ((float) r.u8() * (1.0f / 255.0f) - mean) * istd;
            }
        }
    }
    return x;
}

static const char * kv_name(int kv) { return kv == JEPA_KV_F16 ? "f16" : kv == JEPA_KV_F32 ? "f32" : "auto"; }

struct stats { double mean = 0, min = 0, max = 0; };

static stats summarize(std::vector<double> v) {
    stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    for (double d : v) s.mean += d;
    s.mean /= (double) v.size();
    return s;
}

struct run {
    std::string model_name, model_path, family, ftype, ftype_gguf, mode, shape, kv;
    int  threads = 0, batch = 1, frames = 0, height = 0, width = 0;
    int  repeat = 0, warmup = 0, steps = 0;
    bool flash = true;
    int64_t tokens = 0;        // tokens the timed graph sees (encoder: batch included)
    int64_t units = 0;         // items (encoder/head/predictor) or steps (rollout) per reported ms
    stats   ms, wall;          // per unit: graph compute / full API call
    double  enc_ms = 0;        // the encoder pass feeding head/predictor (0 where n/a)
    double  load_ms = 0;
    size_t  weight_bytes = 0, peak_rss = 0;
};

static double tokens_per_s(const run & r) {
    return r.ms.mean > 0 ? 1000.0 * (double) r.tokens / r.ms.mean : 0.0;
}

// The timed graph covers r.units items (encoder --batch B) or steps (rollout); this is its cost each.
static double ms_per_unit(const run & r) {
    return r.units > 0 ? r.ms.mean / (double) r.units : r.ms.mean;
}

static void print_human(const run & r) {
    printf("%-26s %-5s %-13s %-15s t=%-3d %9.2f ms (min %9.2f) %8.0f tok/s %6.0f MiB",
           r.model_name.c_str(), r.ftype.c_str(), r.mode.c_str(), r.shape.c_str(), r.threads,
           r.ms.mean, r.ms.min, tokens_per_s(r), (double) r.peak_rss / (1024.0 * 1024.0));
    if (r.units > 1) printf("  %8.2f ms/item", ms_per_unit(r));
    printf("\n");
}

static void print_md(const run & r, bool header) {
    if (header) {
        printf("| model | ftype | mode | shape | tokens | threads | ms mean | ms min | tokens/s | peak RSS |\n");
        printf("|---|---|---|---|---:|---:|---:|---:|---:|---:|\n");
    }
    printf("| %s | %s | %s | %s | %lld | %d | %.1f | %.1f | %.0f | %.0f MiB |\n",
           r.model_name.c_str(), r.ftype.c_str(), r.mode.c_str(), r.shape.c_str(),
           (long long) r.tokens, r.threads, r.ms.mean, r.ms.min, tokens_per_s(r),
           (double) r.peak_rss / (1024.0 * 1024.0));
}

static void json_str(FILE * f, const char * key, const std::string & v) {
    fprintf(f, "\"%s\": \"", key);
    for (char c : v) {
        if ((unsigned char) c < 0x20) { fprintf(f, "\\u%04x", c); continue; }
        if (c == '"' || c == '\\') fputc('\\', f);
        fputc(c, f);
    }
    fputs("\", ", f);
}

// -> true on success. A JSON that could not be written (or could not be flushed to disk) must not be
// mistaken for a successful run: bench_all.sh counts the exit status, and a silently missing file
// would drop the config from docs/benchmarks.md without anyone noticing.
static bool write_json(const std::string & path, const std::vector<run> & runs) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s: %s\n", path.c_str(), strerror(errno)); return false; }
    fprintf(f, "{\n  \"tool\": \"jepa-bench\",\n  \"jepa_version\": \"%s\",\n  \"runs\": [\n", jepa_version());
    for (size_t i = 0; i < runs.size(); i++) {
        const run & r = runs[i];
        fputs("    {", f);
        json_str(f, "model", r.model_name);
        json_str(f, "path",  r.model_path);
        json_str(f, "family", r.family);
        json_str(f, "ftype", r.ftype);
        json_str(f, "ftype_gguf", r.ftype_gguf);
        json_str(f, "mode",  r.mode);
        json_str(f, "shape", r.shape);
        json_str(f, "kv",    r.kv);
        fprintf(f, "\"flash\": %s, ", r.flash ? "true" : "false");
        fprintf(f, "\"threads\": %d, \"batch\": %d, \"frames\": %d, \"height\": %d, \"width\": %d, ",
                r.threads, r.batch, r.frames, r.height, r.width);
        fprintf(f, "\"repeat\": %d, \"warmup\": %d, \"steps\": %d, ", r.repeat, r.warmup, r.steps);
        fprintf(f, "\"tokens\": %lld, \"units\": %lld, ", (long long) r.tokens, (long long) r.units);
        fprintf(f, "\"ms_mean\": %.4f, \"ms_min\": %.4f, \"ms_max\": %.4f, ", r.ms.mean, r.ms.min, r.ms.max);
        fprintf(f, "\"wall_ms_mean\": %.4f, \"wall_ms_min\": %.4f, ", r.wall.mean, r.wall.min);
        fprintf(f, "\"ms_per_unit\": %.4f, ", ms_per_unit(r));
        fprintf(f, "\"encoder_ms\": %.4f, \"tokens_per_s\": %.2f, ", r.enc_ms, tokens_per_s(r));
        fprintf(f, "\"load_ms\": %.2f, \"weight_bytes\": %zu, \"peak_rss_bytes\": %zu",
                r.load_ms, r.weight_bytes, r.peak_rss);
        fprintf(f, "}%s\n", i + 1 < runs.size() ? "," : "");
    }
    fputs("  ]\n}\n", f);
    const bool ok = ferror(f) == 0;
    if (fclose(f) != 0 || !ok) {
        fprintf(stderr, "failed to write %s: %s\n", path.c_str(), strerror(errno));
        return false;
    }
    fprintf(stderr, "wrote %s (%zu run%s)\n", path.c_str(), runs.size(), runs.size() == 1 ? "" : "s");
    return true;
}

// Parse one --threads entry. Returns false for anything that is not a whole positive number, so a
// typo ("32,,96", "-t 32x", "--threads all") fails loudly instead of silently becoming atoi()'s 0,
// which jepa_context_new() would read as "use every hardware thread".
static bool parse_thread_count(const std::string & tok, int & out) {
    errno = 0;
    char * end = nullptr;
    const long v = strtol(tok.c_str(), &end, 10);
    if (end == tok.c_str() || (end && *end != '\0')) return false;   // empty or trailing junk
    if (errno == ERANGE || v <= 0 || v > 100000) return false;
    out = (int) v;
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, mode = "encoder", json_out, label, ftype_label, size_arg, threads_arg;
    jepa_context_params cp = jepa_context_default_params();
    jepa_model_params   mp = jepa_model_default_params();
    int frames = -1, batch = 1, repeat = 3, warmup = 1, steps = 20;
    int gpu_prec = -1;   // -1 = the context default (F32 on a GPU), 0 = --gpu-prec f16, 1 = f32
    uint64_t seed = 1234;
    bool md = false, md_header = true, verbose = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")   model_path  = next("-m");
        else if (a == "--mode")                 mode        = next("--mode");
        else if (a == "--frames")               frames      = atoi(next("--frames"));
        else if (a == "--size")                 size_arg    = next("--size");
        else if (a == "--batch")                batch       = atoi(next("--batch"));
        else if (a == "-t" || a == "--threads") threads_arg = next("--threads");
        else if (a == "--repeat")               repeat      = atoi(next("--repeat"));
        else if (a == "--warmup")               warmup      = atoi(next("--warmup"));
        else if (a == "--steps")                steps       = atoi(next("--steps"));
        else if (a == "--seed")                 seed        = strtoull(next("--seed"), nullptr, 10);
        else if (a == "--kv-f16")               cp.flash_kv = JEPA_KV_F16;
        else if (a == "--kv-f32")               cp.flash_kv = JEPA_KV_F32;
        else if (jepa_arg_gpu(argc, argv, i, mp.device)) {}
        else if (a == "--gpu-prec") {
            const std::string v = next("--gpu-prec");
            if (v == "f16")      gpu_prec = 0;
            else if (v == "f32") gpu_prec = 1;
            else { fprintf(stderr, "--gpu-prec wants f16 or f32, got '%s'\n", v.c_str()); return 1; }
        }
        else if (a == "--no-flash")             cp.use_flash_attn = false;
        else if (a == "--md")                   md          = true;
        else if (a == "--no-md-header")         md_header   = false;
        else if (a == "--json")                 json_out    = next("--json");
        else if (a == "--label")                label       = next("--label");
        else if (a == "--ftype-label")          ftype_label = next("--ftype-label");
        else if (a == "-v" || a == "--verbose") verbose     = true;
        else if (a == "-h" || a == "--help")  { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (model_path.empty()) { usage(argv[0]); return 1; }
    if (mode != "encoder" && mode != "head" && mode != "predictor" &&
        mode != "lewm-step" && mode != "lewm-rollout") {
        fprintf(stderr, "unknown --mode %s\n", mode.c_str());
        return 1;
    }
    if (repeat < 1) repeat = 1;
    if (warmup < 0) warmup = 0;
    if (batch  < 1) batch  = 1;
    if (steps  < 1) steps  = 1;
    if (batch > 1 && mode != "encoder") {
        fprintf(stderr, "note: --batch only applies to --mode encoder; %s times one item\n", mode.c_str());
        batch = 1;
    }

    // --threads accepts "32" or "32,96" (one context, one reported row per entry)
    std::vector<int> thread_list;
    for (size_t p = 0; !threads_arg.empty() && p <= threads_arg.size(); ) {
        const size_t c = threads_arg.find(',', p);
        const std::string tok = threads_arg.substr(p, c == std::string::npos ? std::string::npos : c - p);
        int nth = 0;
        if (!parse_thread_count(tok, nth)) {
            fprintf(stderr, "--threads wants a comma-separated list of positive integers, got '%s' in '%s'\n",
                    tok.c_str(), threads_arg.c_str());
            return 1;
        }
        thread_list.push_back(nth);
        if (c == std::string::npos) break;
        p = c + 1;
    }
    if (thread_list.empty()) thread_list.push_back(cp.n_threads);   // 0 = hardware concurrency

    if (verbose) jepa_print_system_info();

    const double t_load = now_ms();
    mp.verbose = verbose;
    jepa_model * model = jepa_model_load_ex(model_path.c_str(), &mp);
    if (!model) return 1;
    const double load_ms = now_ms() - t_load;

    const std::string family = jepa_model_family(model);
    const bool video_model = family == "vjepa" || family == "vjepa2" || family == "vjepa2_1";
    const int  img = jepa_model_img_size(model);

    int H = img, W = img;
    if (!size_arg.empty()) {
        const size_t x = size_arg.find_first_of("xX*");
        if (x == std::string::npos) { fprintf(stderr, "--size wants HxW\n"); return 1; }
        H = atoi(size_arg.substr(0, x).c_str());
        W = atoi(size_arg.substr(x + 1).c_str());
        if (H <= 0 || W <= 0) { fprintf(stderr, "--size wants positive HxW\n"); return 1; }
    }
    int T = frames > 0 ? frames : (video_model ? std::min(jepa_model_n_frames(model), 16) : 1);
    if (T < 1) T = 1;

    const bool needs_encoder = mode == "encoder" || mode == "head" || mode == "predictor";
    const bool lewm_mode     = mode == "lewm-step" || mode == "lewm-rollout";
    if (mode == "head" && !jepa_model_has_head(model)) {
        fprintf(stderr, "%s carries no jepa.head — nothing to time in --mode head\n", model_path.c_str());
        return 1;
    }
    if (mode == "predictor" && !jepa_model_has_predictor(model)) {
        fprintf(stderr, "%s carries no jepa.pred — nothing to time in --mode predictor\n", model_path.c_str());
        return 1;
    }
    if (lewm_mode && (family != "lewm" || jepa_lewm_action_dim(model) <= 0)) {
        fprintf(stderr, "%s is not a LeWM world model — --mode %s needs jepa.pred.kind = lewm\n",
                model_path.c_str(), mode.c_str());
        return 1;
    }

    if (needs_encoder && jepa_token_grid(model, T, H, W, nullptr, nullptr, nullptr) == 0) {
        fprintf(stderr, "%d frame(s) of %dx%d is not encodable by this model (tubelet %d, patch %d)\n",
                T, H, W, jepa_model_tubelet_size(model), jepa_model_patch_size(model));
        return 1;
    }

    std::string name = label.empty() ? jepa_model_name(model) : label;
    if (name.empty()) name = model_path;
    // general.file_type is the *most common* stored tensor type. A k-quant mix falls back to q4_0 for
    // every tensor whose rows are not a multiple of the 256-element super-block, so a small model can
    // end up with more q4_0 tensors than q4_k ones (lejepa-vits16 q4_k: 36 q4_0 against 12 q4_k) and
    // reads back as a q4_0 file. --ftype-label carries the type that was asked for (the filename
    // suffix); both it and the GGUF's own answer go into the JSON.
    const std::string ftype_gguf = jepa_model_file_type_name(model);
    const std::string ftype      = ftype_label.empty() ? ftype_gguf : ftype_label;
    if (!ftype_label.empty() && ftype_label != ftype_gguf) {
        fprintf(stderr, "note: labelling as '%s'; the GGUF's general.file_type reads '%s'\n",
                ftype_label.c_str(), ftype_gguf.c_str());
    }

    // ---- synthetic inputs, built once so every thread count sees identical numbers
    std::vector<float> x, embs, acts;
    if (needs_encoder) x = make_input(model, batch, T, H, W, seed);
    if (lewm_mode) {
        const int D = jepa_model_embed_dim(model);
        const int A = jepa_lewm_action_dim(model);
        const int F = std::max(1, jepa_lewm_n_frames(model));
        const int n_emb = mode == "lewm-step" ? F : 1;              // rollout seeds with one frame
        const int n_act = mode == "lewm-step" ? F : steps;
        rng r(seed);
        embs.resize((size_t) n_emb * D);
        acts.resize((size_t) n_act * A);
        for (float & v : embs) v = r.sym();
        for (float & v : acts) v = r.sym();
    }

    std::vector<run> runs;
    bool first_row = true;
    int  rc_exit = 0;
    const int smt_threads = (int) std::thread::hardware_concurrency();
    for (int nth : thread_list) {
        jepa_context_params p = cp;
        p.n_threads = nth;
        jepa_context * ctx = jepa_context_new(model, p);
        if (!ctx) { rc_exit = 1; break; }
        if (gpu_prec >= 0) jepa_context_set_mul_mat_prec_f32(ctx, gpu_prec == 1);

        run r;
        r.model_name = name; r.model_path = model_path; r.family = family; r.mode = mode;
        r.ftype = ftype; r.ftype_gguf = ftype_gguf;
        r.threads = jepa_context_n_threads(ctx);
        // The published tables are 32/96 threads (one worker per physical core, or a third of them).
        // Falling through to hardware_concurrency() puts two workers on every SMT sibling, which is
        // both slower and not what the documents quote — say so rather than let it pass unnoticed.
        if (smt_threads > 0 && r.threads == smt_threads) {
            fprintf(stderr, "note: running with all %d hardware threads (SMT siblings included); "
                            "docs/benchmarks.md quotes 32 and 96 — pass --threads 32 to match it\n",
                    smt_threads);
        }
        r.batch = batch; r.repeat = repeat; r.warmup = warmup;
        r.flash = p.use_flash_attn; r.kv = kv_name(p.flash_kv);
        r.load_ms = load_ms; r.weight_bytes = jepa_model_n_bytes(model);

        std::vector<double> ms, wall;
        bool failed = false;

        if (needs_encoder) {
            r.frames = T; r.height = H; r.width = W;
            r.shape  = (video_model || T > 1) ? std::to_string(T) + "f " : std::string();
            r.shape += std::to_string(H) + "x" + std::to_string(W);
        }

        if (mode == "encoder") {
            if (batch > 1) r.shape += " x" + std::to_string(batch);
            jepa_context_set_max_batch(ctx, batch);   // all B items in one graph, no internal chunking
            const jepa_input in = { x.data(), batch, 3, T, H, W };
            for (int i = 0; i < warmup + repeat && !failed; i++) {
                jepa_output enc = {};
                const double t0 = now_ms();
                if (jepa_encode(ctx, &in, &enc) != 0) { failed = true; break; }
                const double w_ms = now_ms() - t0;
                if (i >= warmup) { ms.push_back(jepa_context_last_compute_ms(ctx)); wall.push_back(w_ms); }
                r.tokens = enc.n_tokens;
                jepa_free(enc.data);
            }
            r.units = batch;
        } else if (mode == "head" || mode == "predictor") {
            // The head and the masked predictor both consume exactly one item's tokens. Encode
            // three times: the first pass pages the weights in, and the *minimum* of the two after
            // it is the encoder cost reported next to the head/predictor one — a single pass would
            // hand this row whatever contention that one pass happened to see.
            const jepa_input in = { x.data(), 1, 3, T, H, W };
            jepa_output enc = {};
            double enc_min = 0.0;
            for (int i = 0; i < 3 && !failed; i++) {
                jepa_free(enc.data);
                enc = jepa_output{};
                if (jepa_encode(ctx, &in, &enc) != 0) { failed = true; break; }
                const double e = jepa_context_last_compute_ms(ctx);
                if (i > 0 && (enc_min == 0.0 || e < enc_min)) enc_min = e;
            }
            if (!failed) {
                r.enc_ms = enc_min;
                r.tokens = enc.n_tokens;
                r.units  = 1;
                std::vector<int32_t> idx;
                if (mode == "predictor") {
                    idx.resize((size_t) enc.n_tokens);
                    for (int64_t i = 0; i < enc.n_tokens; i++) idx[(size_t) i] = (int32_t) i;
                }
                for (int i = 0; i < warmup + repeat && !failed; i++) {
                    jepa_output out = {};
                    const double t0 = now_ms();
                    const int rc = mode == "head"
                        ? jepa_head(ctx, &enc, &out)
                        : jepa_predict(ctx, &enc, idx.data(), (int) idx.size(),
                                       idx.data(), (int) idx.size(), &out);
                    if (rc != 0) { failed = true; break; }
                    const double w_ms = now_ms() - t0;
                    if (i >= warmup) { ms.push_back(jepa_context_last_compute_ms(ctx)); wall.push_back(w_ms); }
                    jepa_free(out.data);
                }
            }
            jepa_free(enc.data);
        } else if (mode == "lewm-step") {
            const int F = std::max(1, jepa_lewm_n_frames(model));
            r.tokens = F;
            r.units  = 1;
            r.frames = F;
            r.shape  = std::to_string(F) + "f x " + std::to_string(jepa_model_embed_dim(model)) + "d";
            for (int i = 0; i < warmup + repeat && !failed; i++) {
                jepa_output out = {};
                const double t0 = now_ms();
                if (jepa_lewm_predict(ctx, embs.data(), acts.data(), F, &out) != 0) { failed = true; break; }
                const double w_ms = now_ms() - t0;
                if (i >= warmup) { ms.push_back(jepa_context_last_compute_ms(ctx)); wall.push_back(w_ms); }
                jepa_free(out.data);
            }
        } else {  // lewm-rollout: ms is per predicted step
            const int D = jepa_model_embed_dim(model);
            r.steps  = steps;
            // One rollout step emits one embedding, and the reported ms is already per step, so the
            // rate derived from (tokens / ms) is steps/s. Setting tokens = steps here would multiply
            // it by K — the K steps are what the ms was already divided by.
            r.tokens = 1;
            r.units  = steps;
            r.shape  = "rollout K=" + std::to_string(steps);
            std::vector<float> out((size_t) steps * D);
            for (int i = 0; i < warmup + repeat && !failed; i++) {
                const double t0 = now_ms();
                if (jepa_lewm_rollout(ctx, embs.data(), 1, acts.data(), steps, out.data()) != 0) {
                    failed = true;
                    break;
                }
                const double w_ms = now_ms() - t0;
                // jepa_lewm_rollout accumulates the compute time of all K predictor graphs.
                if (i >= warmup) {
                    ms.push_back(jepa_context_last_compute_ms(ctx) / steps);
                    wall.push_back(w_ms / steps);
                }
            }
        }

        jepa_context_free(ctx);
        if (failed) { fprintf(stderr, "jepa-bench: %s failed on %s\n", mode.c_str(), model_path.c_str()); rc_exit = 1; break; }

        r.ms       = summarize(ms);
        r.wall     = summarize(wall);
        r.peak_rss = peak_rss_bytes();
        runs.push_back(r);
        if (md) print_md(r, md_header && first_row);
        else    print_human(r);
        fflush(stdout);
        first_row = false;
    }

    if (!json_out.empty() && !runs.empty() && !write_json(json_out, runs)) rc_exit = 1;
    jepa_model_free(model);
    return rc_exit;
}
