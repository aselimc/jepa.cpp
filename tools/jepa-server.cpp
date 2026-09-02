// jepa-server: an HTTP front end for one GGUF model, over the public C API.
//
//   jepa-server -m lejepa-vits16-pretrain-in1k-f16.gguf [--port 8080] [--workers 4] [--max-batch 8]
//   jepa-server -m vjepa2-vitl-fpc16-256-ssv2-f16.gguf --allow-local-files
//   jepa-server -m vjepa2-ac-vitg-f16.gguf --gpu 1 --workers 2
//
// Endpoints
//   POST /v1/embeddings   OpenAI-shaped embeddings ({"object":"list","data":[{"embedding":[...]}]})
//   POST /classify        top-k labels of a classifier-head model
//   POST /rollout         world-model rollout energies (V-JEPA 2-AC, LeWM) and the CEM plan
//   GET  /health          model, backend, device, pool settings
//   GET  /metrics         Prometheus text exposition
//   GET  /v1/models       the one loaded model, OpenAI-shaped
//
// Shape of the process, and why
//
//   One `jepa_model` is loaded once and shared: the header's thread contract makes it immutable
//   after load, so every worker reads the same weights. Each worker owns one `jepa_context`,
//   because a context owns the graph arena and two threads must never be inside one. That is the
//   whole concurrency story — `--workers` scales calls, `--threads` scales one call, and doing both
//   past the core count oversubscribes the machine. One process per GPU is the scale-out unit;
//   a model and its contexts never straddle two backends.
//
//   Requests are cut into per-item tasks and queued. A dispatcher groups tasks that share a shape
//   into one `jepa_encode` call of that many batch slices, up to `--max-batch` and waiting at most
//   `--max-wait-ms` for the group to fill. Grouping changes nothing about the numbers: on the CPU a
//   batched encode is bit-identical to the same items encoded one at a time (tests/test-batch.cpp),
//   and on a CUDA device the two agree to ~1e-7 cosine but not bit-for-bit, because GEMM tiling
//   varies with the batch shape. Video families never batch in the library (one clip per graph), so
//   for those the group size stays 1 unless `--max-batch` says otherwise, exactly as jepa-embed does.
//
//   Decoding and preprocessing run on the HTTP thread that received the request: those entry points
//   keep no state and are re-entrant, so that work parallelises for free and leaves the workers to
//   the graph. Frames become a clip through tools/jepa-frames.h, the same function jepa-embed uses,
//   which is what makes a server embedding bit-identical to `jepa-embed` on the same input.
//
//   The server binds 127.0.0.1 unless `--host` says otherwise, has no TLS and no authentication:
//   it is a component to put behind something, not an edge.
#include "jepa.h"
#include "jepa-args.h"
#include "jepa-frames.h"

// stb_image's implementation is compiled into libjepa (src/preprocess.cpp) with these two defines;
// repeating them here keeps the declarations in agreement with it. Only the from-memory entry point
// is used — a base64 image has no path for jepa_load_image_rgb() to open.
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

// The library's own per-graph item cap, which is also the largest group the dispatcher may form.
static const int JEPA_SERVER_MAX_BATCH = 32;

static double now_s() {
    return std::chrono::duration<double>(clock_type::now().time_since_epoch()).count();
}

// ==================================================================================================
// options
// ==================================================================================================

struct Options {
    std::string model_path;
    std::string model_name;                  // the id /v1/models reports and requests may name
    std::string host = "127.0.0.1";          // loopback unless told otherwise, every time
    int  port = 8080;
    int  workers = 1;
    int  threads = 0;                        // 0 = derived from the core count and --workers
    bool threads_explicit = false;
    int  max_batch = 8;
    int  max_batch_explicit = 0;             // --max-batch given: honour it even for video families
    int  max_wait_ms = 5;
    int  device = -1;
    bool allow_local_files = false;
    bool verbose = false;
    size_t max_body_mb = 32;
    int  max_items = 64;                     // items per request
    int  max_frames = 128;                   // frames per item
    jepa_context_params cp = jepa_context_default_params();
};

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf [options]\n"
        "  --host H          bind address (default 127.0.0.1 — loopback; anything else is a\n"
        "                    deliberate exposure and the server says so on startup)\n"
        "  --port N          listen port (default 8080; 0 = pick a free port and print it)\n"
        "  --workers N       worker threads, one jepa_context each (default 1)\n"
        "  -t, --threads N   ggml threads inside one graph (default: cores / workers)\n"
        "  --max-batch N     items the dispatcher folds into one encoder graph (default 8, max %d;\n"
        "                    video families stay at 1 unless this is given explicitly)\n"
        "  --max-wait-ms N   how long a partial batch waits for company (default 5; 0 = never wait)\n"
        "  --model-name S    the id /v1/models reports (default: the file's base name)\n"
        "  --allow-local-files  accept server-local paths as input, not just {\"b64\": ...}\n"
        "  --max-body-mb N   request body limit (default 32)\n"
        "  --max-items N     items per request (default 64)\n"
        "  --max-frames N    frames per item (default 128)\n"
        JEPA_GPU_USAGE
        "  --no-flash        naive attention instead of flash attention\n"
        "  --kv-f32 / --kv-f16  K/V dtype for flash attention (default: F32 for f32 models, F16 otherwise)\n"
        "  -v, --verbose     log every request\n", argv0, JEPA_SERVER_MAX_BATCH);
}

// ==================================================================================================
// base64
// ==================================================================================================

static const int8_t B64_REV_INIT = -1;

static bool b64_decode(const std::string & in, std::vector<uint8_t> & out, std::string & err) {
    static int8_t rev[256];
    static bool built = false;
    static std::once_flag once;
    std::call_once(once, [] {
        static const char * alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        memset(rev, B64_REV_INIT, sizeof(rev));
        for (int i = 0; i < 64; i++) rev[(unsigned char) alpha[i]] = (int8_t) i;
        rev[(unsigned char) '-'] = 62;   // the URL-safe alphabet, which browsers and CLIs also emit
        rev[(unsigned char) '_'] = 63;
        built = true;
    });
    (void) built;

    // A data URL ("data:image/png;base64,....") is what a browser hands out, so accept the prefix.
    size_t start = 0;
    if (in.compare(0, 5, "data:") == 0) {
        const size_t comma = in.find(',');
        if (comma == std::string::npos) { err = "data: URL has no comma"; return false; }
        start = comma + 1;
    }
    out.clear();
    out.reserve((in.size() - start) * 3 / 4 + 3);
    uint32_t acc = 0;
    int bits = 0, pad = 0;
    for (size_t i = start; i < in.size(); i++) {
        const unsigned char c = (unsigned char) in[i];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        if (c == '=') { pad++; continue; }
        if (pad) { err = "padding in the middle of the data"; return false; }
        const int8_t v = rev[c];
        if (v < 0) { err = std::string("not base64: byte 0x") + "0123456789abcdef"[c >> 4] + "0123456789abcdef"[c & 15]; return false; }
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t) ((acc >> bits) & 0xff));
        }
    }
    if (bits >= 6) { err = "truncated base64 (a stray character at the end)"; return false; }
    return true;
}

static const char * B64_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string b64_encode(const uint8_t * p, size_t n) {
    std::string s;
    s.reserve((n + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        const uint32_t v = ((uint32_t) p[i] << 16) | ((uint32_t) p[i + 1] << 8) | p[i + 2];
        s += B64_ALPHA[(v >> 18) & 63]; s += B64_ALPHA[(v >> 12) & 63];
        s += B64_ALPHA[(v >> 6) & 63];  s += B64_ALPHA[v & 63];
    }
    if (i < n) {
        uint32_t v = (uint32_t) p[i] << 16;
        if (i + 1 < n) v |= (uint32_t) p[i + 1] << 8;
        s += B64_ALPHA[(v >> 18) & 63];
        s += B64_ALPHA[(v >> 12) & 63];
        s += (i + 1 < n) ? B64_ALPHA[(v >> 6) & 63] : '=';
        s += '=';
    }
    return s;
}

// ==================================================================================================
// metrics
// ==================================================================================================

// Latency buckets in seconds: a small-model CPU encode is a few milliseconds, a ViT-g GPU rollout is
// seconds, so the ladder has to span four orders of magnitude.
static const double LAT_BUCKETS[] = {0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0};
static const int N_LAT_BUCKETS = (int) (sizeof(LAT_BUCKETS) / sizeof(LAT_BUCKETS[0]));

struct EndpointMetrics {
    std::map<int, uint64_t> status;          // response code -> count
    uint64_t items = 0;
    uint64_t lat_bucket[N_LAT_BUCKETS] = {0};
    uint64_t lat_count = 0;
    double   lat_sum = 0;
};

struct Metrics {
    std::mutex m;
    std::map<std::string, EndpointMetrics> by_endpoint;
    // batch sizes the dispatcher actually formed, as an exact histogram: the cap is 32, so a bucket
    // per size costs nothing and answers "did batching happen" without interpolation.
    uint64_t batch_hist[JEPA_SERVER_MAX_BATCH + 1] = {0};
    uint64_t batches = 0;
    uint64_t batched_items = 0;
    std::atomic<int64_t> queue_depth{0};
    std::atomic<int64_t> in_flight{0};

    void observe(const std::string & endpoint, int status_code, double seconds, uint64_t n_items) {
        std::lock_guard<std::mutex> g(m);
        EndpointMetrics & e = by_endpoint[endpoint];
        e.status[status_code]++;
        e.items += n_items;
        e.lat_count++;
        e.lat_sum += seconds;
        for (int i = 0; i < N_LAT_BUCKETS; i++) if (seconds <= LAT_BUCKETS[i]) e.lat_bucket[i]++;
    }
    void observe_batch(int n) {
        std::lock_guard<std::mutex> g(m);
        if (n >= 0 && n <= JEPA_SERVER_MAX_BATCH) batch_hist[n]++;
        batches++;
        batched_items += (uint64_t) n;
    }
};

static Metrics g_metrics;

static std::string prom_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '\\' || c == '"') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

static std::string render_metrics(const Options & opt, const jepa_model * model) {
    std::string out;
    char buf[512];
    std::lock_guard<std::mutex> g(g_metrics.m);

    out += "# HELP jepa_build_info Static facts about this server process.\n";
    out += "# TYPE jepa_build_info gauge\n";
    snprintf(buf, sizeof(buf),
             "jepa_build_info{version=\"%s\",model=\"%s\",family=\"%s\",file_type=\"%s\",device=\"%s\"} 1\n",
             prom_escape(jepa_version()).c_str(), prom_escape(opt.model_name).c_str(),
             prom_escape(jepa_model_family(model)).c_str(),
             prom_escape(jepa_model_file_type_name(model)).c_str(),
             prom_escape(jepa_model_device_name(model)).c_str());
    out += buf;

    out += "# HELP jepa_requests_total Requests completed, by endpoint and HTTP status.\n";
    out += "# TYPE jepa_requests_total counter\n";
    for (const auto & kv : g_metrics.by_endpoint) {
        for (const auto & st : kv.second.status) {
            snprintf(buf, sizeof(buf), "jepa_requests_total{endpoint=\"%s\",status=\"%d\"} %llu\n",
                     prom_escape(kv.first).c_str(), st.first, (unsigned long long) st.second);
            out += buf;
        }
    }

    out += "# HELP jepa_items_total Items (images or clips) accepted, by endpoint.\n";
    out += "# TYPE jepa_items_total counter\n";
    for (const auto & kv : g_metrics.by_endpoint) {
        snprintf(buf, sizeof(buf), "jepa_items_total{endpoint=\"%s\"} %llu\n",
                 prom_escape(kv.first).c_str(), (unsigned long long) kv.second.items);
        out += buf;
    }

    out += "# HELP jepa_request_duration_seconds End-to-end request latency, decode and queue included.\n";
    out += "# TYPE jepa_request_duration_seconds histogram\n";
    for (const auto & kv : g_metrics.by_endpoint) {
        const std::string ep = prom_escape(kv.first);
        for (int i = 0; i < N_LAT_BUCKETS; i++) {
            snprintf(buf, sizeof(buf), "jepa_request_duration_seconds_bucket{endpoint=\"%s\",le=\"%g\"} %llu\n",
                     ep.c_str(), LAT_BUCKETS[i], (unsigned long long) kv.second.lat_bucket[i]);
            out += buf;
        }
        snprintf(buf, sizeof(buf), "jepa_request_duration_seconds_bucket{endpoint=\"%s\",le=\"+Inf\"} %llu\n",
                 ep.c_str(), (unsigned long long) kv.second.lat_count);
        out += buf;
        snprintf(buf, sizeof(buf), "jepa_request_duration_seconds_sum{endpoint=\"%s\"} %.6f\n", ep.c_str(), kv.second.lat_sum);
        out += buf;
        snprintf(buf, sizeof(buf), "jepa_request_duration_seconds_count{endpoint=\"%s\"} %llu\n",
                 ep.c_str(), (unsigned long long) kv.second.lat_count);
        out += buf;
    }

    out += "# HELP jepa_batch_size Items per encoder graph, as the dispatcher formed them.\n";
    out += "# TYPE jepa_batch_size histogram\n";
    uint64_t cum = 0;
    for (int i = 1; i <= JEPA_SERVER_MAX_BATCH; i++) {
        cum += g_metrics.batch_hist[i];
        snprintf(buf, sizeof(buf), "jepa_batch_size_bucket{le=\"%d\"} %llu\n", i, (unsigned long long) cum);
        out += buf;
    }
    snprintf(buf, sizeof(buf), "jepa_batch_size_bucket{le=\"+Inf\"} %llu\n", (unsigned long long) g_metrics.batches);
    out += buf;
    snprintf(buf, sizeof(buf), "jepa_batch_size_sum %llu\n", (unsigned long long) g_metrics.batched_items);
    out += buf;
    snprintf(buf, sizeof(buf), "jepa_batch_size_count %llu\n", (unsigned long long) g_metrics.batches);
    out += buf;

    out += "# HELP jepa_queue_depth Tasks waiting for a worker right now.\n";
    out += "# TYPE jepa_queue_depth gauge\n";
    snprintf(buf, sizeof(buf), "jepa_queue_depth %lld\n", (long long) g_metrics.queue_depth.load());
    out += buf;
    out += "# HELP jepa_requests_in_flight Requests being served right now.\n";
    out += "# TYPE jepa_requests_in_flight gauge\n";
    snprintf(buf, sizeof(buf), "jepa_requests_in_flight %lld\n", (long long) g_metrics.in_flight.load());
    out += buf;
    out += "# HELP jepa_workers Worker threads, each with its own jepa_context.\n";
    out += "# TYPE jepa_workers gauge\n";
    snprintf(buf, sizeof(buf), "jepa_workers %d\n", opt.workers);
    out += buf;
    return out;
}

// ==================================================================================================
// tasks, jobs and the worker pool
// ==================================================================================================

enum PoolKind { POOL_MEAN, POOL_CLS, POOL_LEWM, POOL_NONE };
enum TaskKind { TASK_ENCODE, TASK_ROLLOUT };

struct Job;

// One rollout / plan request's payload. It runs whole on one worker (encode + predictor), so unlike
// an encode task it is never grouped with anything.
struct RolloutTask {
    std::vector<std::vector<float>> context_px;   // one preprocessed clip per context frame
    std::vector<int> ctx_T, ctx_h, ctx_w;
    std::vector<float> goal_px;                   // empty = no goal
    int goal_T = 0, goal_h = 0, goal_w = 0;
    std::vector<float> actions;                   // [n_cand * horizon * action_dim]
    int n_cand = 0, horizon = 0;
    std::vector<float> state;                     // seed pose (AC only)
    bool want_plan = false;
    jepa_ac_cem_params cem{};
    bool want_latents = false;

    // outputs
    std::vector<std::vector<float>> energies;     // [n_cand][horizon]
    std::vector<float> plan_actions;              // [horizon * action_dim]
    std::vector<float> plan_energy;               // [cem_steps]
    std::vector<float> latents;                   // optional, [n_cand * horizon * rows * dim]
    int64_t latent_rows = 0, latent_dim = 0;
    int64_t n_tokens = 0;
};

struct Task {
    TaskKind kind = TASK_ENCODE;
    Job * job = nullptr;

    // --- encode input
    std::vector<float> px;                        // [3, T, h, w]
    int T = 0, h = 0, w = 0;
    PoolKind pool = POOL_CLS;
    bool want_logits = false;

    // --- encode output
    std::vector<float> emb;
    int64_t rows = 0, dim = 0, n_tokens = 0;
    std::vector<float> logits;

    std::unique_ptr<RolloutTask> roll;
    std::string err;                              // non-empty = this task failed

    // Two tasks may share one graph only when the encoder sees the same shape.
    bool groups_with(const Task & o) const {
        return kind == TASK_ENCODE && o.kind == TASK_ENCODE && T == o.T && h == o.h && w == o.w;
    }
};

struct Job {
    std::mutex m;
    std::condition_variable cv;
    int pending = 0;
    std::vector<std::unique_ptr<Task>> tasks;

    void finish_one() {
        std::lock_guard<std::mutex> g(m);
        if (--pending == 0) cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return pending == 0; });
    }
};

struct Worker {
    std::thread th;
    jepa_context * ctx = nullptr;
    std::mutex m;
    std::condition_variable cv;
    std::vector<Task *> batch;
    bool has_work = false;
    bool stop = false;
};

struct Engine {
    jepa_model * model = nullptr;
    Options opt;
    std::string family;
    bool video_model = false;
    int tubelet = 1;
    int eff_max_batch = 1;

    std::mutex mu;
    std::condition_variable cv_work;     // a task arrived, or a worker went idle, or we are stopping
    std::deque<Task *> q;
    std::deque<Worker *> idle;
    bool stopping = false;

    std::vector<std::unique_ptr<Worker>> workers;
    std::thread dispatcher;

    void submit(std::vector<Task *> tasks) {
        {
            std::lock_guard<std::mutex> g(mu);
            for (Task * t : tasks) q.push_back(t);
            g_metrics.queue_depth.store((int64_t) q.size());
        }
        cv_work.notify_all();
    }
};

// --- the compute side ----------------------------------------------------------------------------

// Everything the library logs about a failed call ends up in this thread's capture buffer; a worker
// has no stderr the client can read, so that text is what the JSON error carries.
static std::string engine_error(const std::string & what) {
    const char * text = jepa_error_text();
    std::string s = what;
    if (text && *text) {
        std::string t = text;
        while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
        if (!t.empty()) s += ": " + t;
    }
    return s;
}

static void run_encode_batch(Engine & eng, jepa_context * ctx, std::vector<Task *> & batch) {
    const int64_t nb = (int64_t) batch.size();
    const int T = batch[0]->T, h = batch[0]->h, w = batch[0]->w;

    std::vector<float> xb;
    xb.reserve((size_t) nb * 3 * T * h * w);
    for (Task * t : batch) xb.insert(xb.end(), t->px.begin(), t->px.end());

    jepa_input in;
    in.data = xb.data(); in.n_batch = (int) nb; in.n_chans = 3; in.n_frames = T; in.height = h; in.width = w;

    jepa_output enc = {nullptr, 0, 0};
    jepa_error_reset();
    if (jepa_encode(ctx, &in, &enc) != 0) {
        const std::string why = engine_error("encode failed");
        for (Task * t : batch) t->err = why;
        jepa_free(enc.data);
        return;
    }
    g_metrics.observe_batch((int) nb);

    const int64_t n_tokens = enc.n_tokens / nb;
    for (int64_t i = 0; i < nb; i++) {
        Task * t = batch[(size_t) i];
        jepa_output one = { enc.data + (size_t) i * n_tokens * enc.dim, n_tokens, enc.dim };
        t->n_tokens = n_tokens;

        jepa_output feat = {nullptr, 0, 0};
        const float * rows = one.data;
        int rc = 0;
        jepa_error_reset();
        if      (t->pool == POOL_MEAN) rc = jepa_pool_mean(eng.model, &one, &feat);
        else if (t->pool == POOL_CLS)  rc = jepa_pool_cls(eng.model, &one, &feat);
        else if (t->pool == POOL_LEWM) rc = jepa_lewm_project(ctx, &one, &feat);
        if (rc != 0) { t->err = engine_error("pooling failed"); jepa_free(feat.data); continue; }
        if (feat.data) rows = feat.data;

        t->rows = feat.data ? feat.n_tokens : one.n_tokens;
        t->dim  = feat.data ? feat.dim      : one.dim;
        t->emb.assign(rows, rows + (size_t) t->rows * t->dim);
        jepa_free(feat.data);

        if (t->want_logits) {
            jepa_output lg = {nullptr, 0, 0};
            jepa_error_reset();
            if (jepa_head_ex(ctx, &one, nullptr, &lg) != 0) {
                t->err = engine_error("classifier head failed");
                jepa_free(lg.data);
                continue;
            }
            t->logits.assign(lg.data, lg.data + lg.dim);
            jepa_free(lg.data);
        }
    }
    jepa_free(enc.data);
}

// Encode one already-preprocessed clip and return its rows (normalised for the AC world model, which
// is what its released loop feeds the predictor).
static bool encode_latents(Engine & eng, jepa_context * ctx, const float * px, int T, int h, int w,
                           bool normalize, std::vector<float> & rows, int64_t * n_tokens,
                           int64_t * dim, std::string & err) {
    const jepa_input in = { px, 1, 3, T, h, w };
    jepa_output enc = {nullptr, 0, 0};
    jepa_error_reset();
    if (jepa_encode(ctx, &in, &enc) != 0) { err = engine_error("encode failed"); jepa_free(enc.data); return false; }
    rows.assign(enc.data, enc.data + (size_t) enc.n_tokens * enc.dim);
    if (normalize) jepa_ac_normalize(eng.model, rows.data(), enc.n_tokens, enc.dim);
    if (n_tokens) *n_tokens = enc.n_tokens;
    if (dim) *dim = enc.dim;
    jepa_free(enc.data);
    return true;
}

// V-JEPA 2-AC: encode the observed frames, roll every candidate out on the batch axis, score each
// step with the reference's L1 energy, optionally run the CEM planner over the same cached context.
static void run_rollout_ac(Engine & eng, jepa_context * ctx, Task * t) {
    RolloutTask & r = *t->roll;
    const int64_t D = jepa_model_embed_dim(eng.model);
    const int64_t HW = jepa_ac_tokens_per_frame(eng.model);
    const int A = jepa_ac_action_dim(eng.model), S = jepa_ac_state_dim(eng.model);
    const bool norm = jepa_ac_normalize_reps(eng.model);

    const int n_seed = (int) r.context_px.size();
    std::vector<float> latents;
    int64_t total_tokens = 0;
    for (int i = 0; i < n_seed; i++) {
        std::vector<float> rows;
        int64_t nt = 0, dim = 0;
        if (!encode_latents(eng, ctx, r.context_px[(size_t) i].data(), r.ctx_T[(size_t) i],
                            r.ctx_h[(size_t) i], r.ctx_w[(size_t) i], norm, rows, &nt, &dim, t->err)) return;
        if (nt != HW || dim != D) {
            char b[256];
            snprintf(b, sizeof(b), "context frame %d encoded to %lld x %lld tokens, the predictor wants %lld x %lld",
                     i, (long long) nt, (long long) dim, (long long) HW, (long long) D);
            t->err = b;
            return;
        }
        total_tokens += nt;
        latents.insert(latents.end(), rows.begin(), rows.end());
    }
    std::vector<float> goal;
    if (!r.goal_px.empty()) {
        int64_t nt = 0, dim = 0;
        if (!encode_latents(eng, ctx, r.goal_px.data(), r.goal_T, r.goal_h, r.goal_w, norm, goal, &nt, &dim, t->err)) return;
        total_tokens += nt;
    }
    r.n_tokens = total_tokens;

    std::vector<float> states((size_t) n_seed * S, 0.0f);
    for (int i = 0; i < n_seed && i < 1; i++) {
        for (int k = 0; k < S && k < (int) r.state.size(); k++) states[(size_t) i * S + k] = r.state[(size_t) k];
    }
    // With several observed frames the pose of each later frame follows from the previous one, the
    // same way the reference planner advances it.
    for (int i = 1; i < n_seed; i++) {
        jepa_ac_next_state(eng.model, states.data() + (size_t) (i - 1) * S,
                           r.actions.empty() ? nullptr : r.actions.data(), states.data() + (size_t) i * S);
    }

    if (r.n_cand > 0 && r.horizon > 0) {
        std::vector<float> roll((size_t) r.n_cand * r.horizon * HW * D);
        jepa_error_reset();
        if (jepa_ac_rollout(ctx, latents.data(), n_seed, states.data(), r.actions.data(), nullptr,
                            r.n_cand, r.horizon, roll.data()) != 0) {
            t->err = engine_error("rollout failed");
            return;
        }
        // Meta's l1(z, h): mean |pred - goal| over the frame. Without a goal the same number against
        // the last observed frame, which reads as how far the rollout has drifted.
        const float * target = goal.empty() ? latents.data() + (size_t) (n_seed - 1) * HW * D : goal.data();
        std::vector<float> step((size_t) r.n_cand * HW * D);
        r.energies.assign((size_t) r.n_cand, std::vector<float>((size_t) r.horizon, 0.0f));
        std::vector<float> col((size_t) r.n_cand);
        for (int hstep = 0; hstep < r.horizon; hstep++) {
            for (int c = 0; c < r.n_cand; c++) {
                memcpy(step.data() + (size_t) c * HW * D,
                       roll.data() + ((size_t) c * r.horizon + hstep) * HW * D,
                       (size_t) HW * D * sizeof(float));
            }
            jepa_ac_energy(step.data(), target, r.n_cand, HW, D, col.data());
            for (int c = 0; c < r.n_cand; c++) r.energies[(size_t) c][(size_t) hstep] = col[(size_t) c];
        }
        if (r.want_latents) {
            r.latents = std::move(roll);
            r.latent_rows = HW;
            r.latent_dim = D;
        }
    }

    if (r.want_plan) {
        if (goal.empty()) { t->err = "plan needs a goal frame"; return; }
        jepa_error_reset();
        jepa_ac_context * handle = jepa_ac_context_new(ctx, latents.data(), n_seed, nullptr, states.data());
        if (!handle) { t->err = engine_error("cannot build the planning context"); return; }
        r.plan_actions.assign((size_t) r.cem.horizon * A, 0.0f);
        r.plan_energy.assign((size_t) r.cem.cem_steps, 0.0f);
        jepa_error_reset();
        const int rc = jepa_ac_plan(ctx, handle, goal.data(), &r.cem, nullptr,
                                    r.plan_actions.data(), r.plan_energy.data());
        jepa_ac_context_free(handle);
        if (rc != 0) { t->err = engine_error("plan failed"); return; }
    }
}

// LeWM: encode the seed frames, project them to world-model states and roll the predictor forward.
// The per-step energy is the same L1 the AC planner minimises, here over one row.
static void run_rollout_lewm(Engine & eng, jepa_context * ctx, Task * t) {
    RolloutTask & r = *t->roll;
    const int64_t D = jepa_model_embed_dim(eng.model);
    const int A = jepa_lewm_action_dim(eng.model);
    const int n_seed = (int) r.context_px.size();

    std::vector<float> cls;
    int64_t total_tokens = 0;
    for (int i = 0; i < n_seed; i++) {
        std::vector<float> rows;
        int64_t nt = 0, dim = 0;
        if (!encode_latents(eng, ctx, r.context_px[(size_t) i].data(), r.ctx_T[(size_t) i],
                            r.ctx_h[(size_t) i], r.ctx_w[(size_t) i], false, rows, &nt, &dim, t->err)) return;
        total_tokens += nt;
        cls.insert(cls.end(), rows.begin(), rows.begin() + dim);   // row 0 is the CLS token
    }
    jepa_output proj = {nullptr, 0, 0};
    jepa_error_reset();
    if (jepa_lewm_project_rows(ctx, cls.data(), n_seed, &proj) != 0) {
        t->err = engine_error("projection failed");
        jepa_free(proj.data);
        return;
    }
    std::vector<float> embs(proj.data, proj.data + (size_t) proj.n_tokens * proj.dim);
    jepa_free(proj.data);

    std::vector<float> goal;
    if (!r.goal_px.empty()) {
        std::vector<float> rows;
        int64_t nt = 0, dim = 0;
        if (!encode_latents(eng, ctx, r.goal_px.data(), r.goal_T, r.goal_h, r.goal_w, false, rows, &nt, &dim, t->err)) return;
        total_tokens += nt;
        jepa_output gp = {nullptr, 0, 0};
        jepa_error_reset();
        if (jepa_lewm_project_rows(ctx, rows.data(), 1, &gp) != 0) {
            t->err = engine_error("goal projection failed");
            jepa_free(gp.data);
            return;
        }
        goal.assign(gp.data, gp.data + gp.dim);
        jepa_free(gp.data);
    }
    r.n_tokens = total_tokens;

    r.energies.assign((size_t) r.n_cand, std::vector<float>((size_t) r.horizon, 0.0f));
    if (r.want_latents) { r.latent_rows = 1; r.latent_dim = D; r.latents.assign((size_t) r.n_cand * r.horizon * D, 0.0f); }
    const float * target = goal.empty() ? embs.data() + (size_t) (n_seed - 1) * D : goal.data();
    std::vector<float> out((size_t) r.horizon * D);
    for (int c = 0; c < r.n_cand; c++) {
        jepa_error_reset();
        if (jepa_lewm_rollout(ctx, embs.data(), n_seed, r.actions.data() + (size_t) c * r.horizon * A,
                              r.horizon, out.data()) != 0) {
            t->err = engine_error("rollout failed");
            return;
        }
        for (int hstep = 0; hstep < r.horizon; hstep++) {
            float e = 0;
            jepa_ac_energy(out.data() + (size_t) hstep * D, target, 1, 1, D, &e);
            r.energies[(size_t) c][(size_t) hstep] = e;
        }
        if (r.want_latents) {
            memcpy(r.latents.data() + (size_t) c * r.horizon * D, out.data(), (size_t) r.horizon * D * sizeof(float));
        }
    }
}

static void worker_main(Engine & eng, Worker & wk) {
    for (;;) {
        std::vector<Task *> batch;
        {
            std::unique_lock<std::mutex> lk(wk.m);
            wk.cv.wait(lk, [&] { return wk.has_work || wk.stop; });
            if (wk.stop && !wk.has_work) return;
            batch.swap(wk.batch);
            wk.has_work = false;
        }
        if (!batch.empty()) {
            if (batch[0]->kind == TASK_ROLLOUT) {
                Task * t = batch[0];
                if (jepa_model_has_predictor(eng.model) && jepa_ac_tokens_per_frame(eng.model) > 0) {
                    run_rollout_ac(eng, wk.ctx, t);
                } else {
                    run_rollout_lewm(eng, wk.ctx, t);
                }
            } else {
                run_encode_batch(eng, wk.ctx, batch);
            }
        }
        for (Task * t : batch) t->job->finish_one();
        {
            std::lock_guard<std::mutex> g(eng.mu);
            eng.idle.push_back(&wk);
        }
        eng.cv_work.notify_all();
    }
}

// The dispatcher: one batch at a time, formed from the head of the queue and every compatible task
// behind it. A partial batch waits up to --max-wait-ms for company, but only while there is nothing
// else to run — work that cannot join this group must not be held up by it.
static void dispatcher_main(Engine & eng) {
    const auto max_wait = std::chrono::milliseconds(eng.opt.max_wait_ms);
    for (;;) {
        std::unique_lock<std::mutex> lk(eng.mu);
        eng.cv_work.wait(lk, [&] { return eng.stopping || (!eng.q.empty() && !eng.idle.empty()); });
        if (eng.stopping) return;

        std::vector<Task *> batch;
        Task * head = eng.q.front();
        eng.q.pop_front();
        batch.push_back(head);
        const int cap = head->kind == TASK_ROLLOUT ? 1 : eng.eff_max_batch;

        auto take_compatible = [&] {
            for (auto it = eng.q.begin(); it != eng.q.end() && (int) batch.size() < cap; ) {
                if ((*it)->groups_with(*head)) { batch.push_back(*it); it = eng.q.erase(it); }
                else ++it;
            }
        };
        take_compatible();

        if ((int) batch.size() < cap && eng.opt.max_wait_ms > 0) {
            const auto deadline = clock_type::now() + max_wait;
            while ((int) batch.size() < cap && eng.q.empty() && !eng.stopping) {
                if (eng.cv_work.wait_until(lk, deadline) == std::cv_status::timeout) break;
                take_compatible();
            }
            take_compatible();
        }
        g_metrics.queue_depth.store((int64_t) eng.q.size());

        Worker * wk = eng.idle.front();
        eng.idle.pop_front();
        lk.unlock();

        {
            std::lock_guard<std::mutex> g(wk->m);
            wk->batch = std::move(batch);
            wk->has_work = true;
        }
        wk->cv.notify_one();
    }
}

// ==================================================================================================
// request parsing
// ==================================================================================================

// A request that could not be served. `status` is the HTTP code; `type` follows the OpenAI error
// vocabulary so a client written against that API can branch on it.
struct ApiError {
    int status;
    std::string type;
    std::string message;
};

static json error_body(const ApiError & e) {
    return json{{"error", {{"message", e.message}, {"type", e.type}, {"code", nullptr}}}};
}

static ApiError bad_request(const std::string & msg) { return {400, "invalid_request_error", msg}; }

struct Frame {
    std::vector<uint8_t> rgb;
    int h = 0, w = 0;
};

// One input item as the request described it: a still image, or the frames of one clip.
struct InputItem {
    std::vector<Frame> frames;
};

static bool decode_image_bytes(const std::vector<uint8_t> & bytes, Frame & out, std::string & err) {
    int x = 0, y = 0, n = 0;
    // The same decoder, the same forced 3 channels, as jepa_load_image_rgb() — only the source is a
    // buffer rather than a path, which a base64 image has no way of providing.
    uint8_t * px = stbi_load_from_memory(bytes.data(), (int) bytes.size(), &x, &y, &n, 3);
    if (!px) {
        const char * why = stbi_failure_reason();
        err = std::string("cannot decode the image: ") + (why ? why : "unknown format");
        return false;
    }
    out.rgb.assign(px, px + (size_t) x * y * 3);
    out.h = y; out.w = x;
    stbi_image_free(px);
    return true;
}

static bool load_local_file(const std::string & path, Frame & out, std::string & err) {
    int h = 0, w = 0;
    jepa_error_reset();
    uint8_t * rgb = jepa_load_image_rgb(path.c_str(), &h, &w);
    if (!rgb) { err = engine_error("cannot read " + path); return false; }
    out.rgb.assign(rgb, rgb + (size_t) h * w * 3);
    out.h = h; out.w = w;
    jepa_free(rgb);
    return true;
}

// One frame from a JSON value: a bare string (a path), {"b64": ...} or {"path": ...}.
static bool parse_frame(const json & v, const Options & opt, Frame & out, ApiError & err) {
    if (v.is_string()) {
        if (!opt.allow_local_files) {
            err = bad_request("a string input is a server-local path, which this server was not started "
                              "with --allow-local-files; send {\"b64\": \"...\"} instead");
            return false;
        }
        std::string why;
        if (!load_local_file(v.get<std::string>(), out, why)) { err = bad_request(why); return false; }
        return true;
    }
    if (!v.is_object()) {
        err = bad_request("an input item must be a string path or an object with \"b64\", \"path\" or \"frames\"");
        return false;
    }
    if (v.contains("url")) {
        err = bad_request("\"url\" inputs are not supported: the server never fetches anything itself");
        return false;
    }
    if (v.contains("b64")) {
        if (!v["b64"].is_string()) { err = bad_request("\"b64\" must be a string"); return false; }
        std::vector<uint8_t> bytes;
        std::string why;
        if (!b64_decode(v["b64"].get<std::string>(), bytes, why)) { err = bad_request("\"b64\": " + why); return false; }
        if (bytes.empty()) { err = bad_request("\"b64\" decoded to nothing"); return false; }
        if (!decode_image_bytes(bytes, out, why)) { err = bad_request(why); return false; }
        return true;
    }
    if (v.contains("path")) {
        if (!v["path"].is_string()) { err = bad_request("\"path\" must be a string"); return false; }
        if (!opt.allow_local_files) {
            err = bad_request("\"path\" inputs need the server to be started with --allow-local-files");
            return false;
        }
        std::string why;
        if (!load_local_file(v["path"].get<std::string>(), out, why)) { err = bad_request(why); return false; }
        return true;
    }
    err = bad_request("an input object needs \"b64\" or \"path\"");
    return false;
}

static bool parse_item(const json & v, const Options & opt, InputItem & out, ApiError & err) {
    if (v.is_object() && v.contains("frames")) {
        const json & fr = v["frames"];
        if (!fr.is_array() || fr.empty()) { err = bad_request("\"frames\" must be a non-empty array"); return false; }
        if ((int) fr.size() > opt.max_frames) {
            err = bad_request("this item has " + std::to_string(fr.size()) + " frames, the limit is "
                              + std::to_string(opt.max_frames) + " (--max-frames)");
            return false;
        }
        for (const json & f : fr) {
            Frame frame;
            if (!parse_frame(f, opt, frame, err)) return false;
            out.frames.push_back(std::move(frame));
        }
        return true;
    }
    Frame frame;
    if (!parse_frame(v, opt, frame, err)) return false;
    out.frames.push_back(std::move(frame));
    return true;
}

// `input` is one item or an array of them. A bare array of frames is ambiguous, so it is always read
// as several items; the frames of one clip go in {"frames": [...]}.
static bool parse_input(const json & body, const char * field, const Options & opt,
                        std::vector<InputItem> & out, ApiError & err) {
    if (!body.contains(field)) { err = bad_request(std::string("missing \"") + field + "\""); return false; }
    const json & in = body[field];
    std::vector<const json *> items;
    if (in.is_array()) { for (const json & v : in) items.push_back(&v); }
    else items.push_back(&in);
    if (items.empty()) { err = bad_request(std::string("\"") + field + "\" is empty"); return false; }
    if ((int) items.size() > opt.max_items) {
        err = bad_request("this request has " + std::to_string(items.size()) + " items, the limit is "
                          + std::to_string(opt.max_items) + " (--max-items)");
        return false;
    }
    for (const json * v : items) {
        InputItem item;
        if (!parse_item(*v, opt, item, err)) return false;
        out.push_back(std::move(item));
    }
    return true;
}

// The item's frames -> the preprocessed NCTHW clip, through the same function jepa-embed uses.
static bool preprocess_item(Engine & eng, const InputItem & item, Task & task, ApiError & err) {
    std::vector<const uint8_t *> fp;
    std::vector<int> fh, fw;
    for (const Frame & f : item.frames) { fp.push_back(f.rgb.data()); fh.push_back(f.h); fw.push_back(f.w); }
    int T = 0, h = 0, w = 0;
    jepa_error_reset();
    float * px = jepa_frames::to_clip(eng.model, fp, fh, fw, eng.tubelet, eng.video_model, "input",
                                      eng.opt.verbose ? stderr : nullptr, &T, &h, &w);
    if (!px) { err = bad_request(engine_error("preprocessing failed")); return false; }
    task.px.assign(px, px + (size_t) 3 * T * h * w);
    jepa_free(px);
    task.T = T; task.h = h; task.w = w;
    return true;
}

static bool parse_pool(const json & body, const jepa_model * model, PoolKind & out, ApiError & err) {
    // The default jepa-embed picks, so that an unqualified request and an unqualified CLI run agree.
    out = jepa_model_has_cls(model) ? POOL_CLS : POOL_MEAN;
    if (!body.contains("pool")) return true;
    if (!body["pool"].is_string()) { err = bad_request("\"pool\" must be a string"); return false; }
    const std::string p = body["pool"].get<std::string>();
    if (p == "mean") out = POOL_MEAN;
    else if (p == "cls") {
        if (!jepa_model_has_cls(model)) { err = bad_request("this model has no CLS token; use \"pool\": \"mean\""); return false; }
        out = POOL_CLS;
    } else if (p == "lewm") {
        if (!jepa_model_has_projector(model)) { err = bad_request("this model has no enc.proj projector"); return false; }
        out = POOL_LEWM;
    } else if (p == "none") out = POOL_NONE;
    else { err = bad_request("unknown \"pool\" " + p + " (mean, cls, lewm, none)"); return false; }
    return true;
}

static bool check_model_name(const json & body, const Options & opt, ApiError & err) {
    if (!body.contains("model") || body["model"].is_null()) return true;   // one model per process
    if (!body["model"].is_string()) { err = bad_request("\"model\" must be a string"); return false; }
    const std::string m = body["model"].get<std::string>();
    if (m == opt.model_name || m == opt.model_path) return true;
    err = {404, "model_not_found", "this server serves \"" + opt.model_name + "\", not \"" + m + "\""};
    return false;
}

// [K][H][A] or [H][A] (one candidate) of numbers.
static bool parse_actions(const json & v, int action_dim, int max_cand, int max_horizon,
                          std::vector<float> & out, int & n_cand, int & horizon, ApiError & err) {
    if (!v.is_array() || v.empty()) { err = bad_request("\"actions\" must be a non-empty array"); return false; }
    const bool nested = v[0].is_array() && !v[0].empty() && v[0][0].is_array();
    const json wrapped = nested ? v : json::array({v});
    n_cand = (int) wrapped.size();
    if (n_cand > max_cand) {
        err = bad_request("\"actions\" has " + std::to_string(n_cand) + " candidates, the limit is "
                          + std::to_string(max_cand));
        return false;
    }
    horizon = 0;
    for (const json & cand : wrapped) {
        if (!cand.is_array() || cand.empty()) { err = bad_request("each candidate must be a non-empty array of steps"); return false; }
        if (horizon == 0) horizon = (int) cand.size();
        else if ((int) cand.size() != horizon) { err = bad_request("every candidate must have the same number of steps"); return false; }
        if (horizon > max_horizon) {
            err = bad_request("horizon " + std::to_string(horizon) + " exceeds the limit " + std::to_string(max_horizon));
            return false;
        }
        for (const json & step : cand) {
            if (!step.is_array() || (int) step.size() != action_dim) {
                err = bad_request("each step must be " + std::to_string(action_dim) + " numbers");
                return false;
            }
            for (const json & x : step) {
                if (!x.is_number()) { err = bad_request("actions must be numbers"); return false; }
                out.push_back(x.get<float>());
            }
        }
    }
    return true;
}

// ==================================================================================================
// responses
// ==================================================================================================

// Round-trip of a float32 through JSON is exact: nlohmann writes the shortest decimal that reads
// back as the same double, and the double came from the float. `encoding_format: "base64"` skips the
// question entirely by shipping the little-endian bytes.
static json embedding_value(const std::vector<float> & v, int64_t rows, int64_t dim, bool as_b64) {
    if (as_b64) return json(b64_encode((const uint8_t *) v.data(), v.size() * sizeof(float)));
    if (rows > 1) {
        json m = json::array();
        for (int64_t r = 0; r < rows; r++) {
            m.push_back(json(std::vector<float>(v.begin() + (size_t) r * dim, v.begin() + (size_t) (r + 1) * dim)));
        }
        return m;
    }
    return json(v);
}

// ==================================================================================================
// main
// ==================================================================================================

static httplib::Server * g_server = nullptr;

static void on_signal(int) {
    if (g_server) g_server->stop();
}

int main(int argc, char ** argv) {
    Options opt;
    opt.cp = jepa_context_default_params();
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model") opt.model_path = next("-m");
        else if (a == "--host") opt.host = next("--host");
        else if (a == "--port") opt.port = atoi(next("--port"));
        else if (a == "--workers") opt.workers = atoi(next("--workers"));
        else if (a == "-t" || a == "--threads") { opt.threads = atoi(next("--threads")); opt.threads_explicit = true; }
        else if (a == "--max-batch") { opt.max_batch = atoi(next("--max-batch")); opt.max_batch_explicit = 1; }
        else if (a == "--max-wait-ms") opt.max_wait_ms = atoi(next("--max-wait-ms"));
        else if (a == "--model-name") opt.model_name = next("--model-name");
        else if (a == "--allow-local-files") opt.allow_local_files = true;
        else if (a == "--max-body-mb") opt.max_body_mb = (size_t) atoi(next("--max-body-mb"));
        else if (a == "--max-items") opt.max_items = atoi(next("--max-items"));
        else if (a == "--max-frames") opt.max_frames = atoi(next("--max-frames"));
        else if (a == "--no-flash") opt.cp.use_flash_attn = false;
        else if (a == "--kv-f32") opt.cp.flash_kv = JEPA_KV_F32;
        else if (a == "--kv-f16") opt.cp.flash_kv = JEPA_KV_F16;
        else if (a == "-v" || a == "--verbose") opt.verbose = true;
        else if (jepa_arg_gpu(argc, argv, i, opt.device)) {}
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (opt.model_path.empty()) { usage(argv[0]); return 1; }
    if (opt.workers < 1) opt.workers = 1;
    if (opt.max_batch < 1) opt.max_batch = 1;
    if (opt.max_batch > JEPA_SERVER_MAX_BATCH) {
        fprintf(stderr, "note: --max-batch %d is above the engine's %d-slice graph limit; using %d\n",
                opt.max_batch, JEPA_SERVER_MAX_BATCH, JEPA_SERVER_MAX_BATCH);
        opt.max_batch = JEPA_SERVER_MAX_BATCH;
    }
    if (opt.max_wait_ms < 0) opt.max_wait_ms = 0;
    if (opt.max_items < 1) opt.max_items = 1;
    if (opt.max_frames < 1) opt.max_frames = 1;
    if (opt.model_name.empty()) {
        std::string base = opt.model_path;
        const size_t slash = base.find_last_of("/\\");
        if (slash != std::string::npos) base = base.substr(slash + 1);
        if (base.size() > 5 && base.compare(base.size() - 5, 5, ".gguf") == 0) base.resize(base.size() - 5);
        opt.model_name = base;
    }
    // --threads is the width of one graph's thread pool and --workers the number of graphs in
    // flight; the product is what lands on the machine, so the default divides rather than repeats.
    if (!opt.threads_explicit && opt.workers > 1) {
        const unsigned hw = std::thread::hardware_concurrency();
        opt.threads = (int) std::max(1u, (hw ? hw : 1u) / (unsigned) opt.workers);
    }
    opt.cp.n_threads = opt.threads;
    opt.cp.verbose = opt.verbose;

    jepa_model_params mp = jepa_model_default_params();
    mp.verbose = opt.verbose;
    if (opt.device >= 0) mp.device = opt.device;

    Engine eng;
    eng.opt = opt;
    eng.model = jepa_model_load_ex(opt.model_path.c_str(), &mp);
    if (!eng.model) return 1;
    eng.family = jepa_model_family(eng.model);
    eng.video_model = jepa_frames::is_video_family(eng.family);
    eng.tubelet = jepa_model_tubelet_size(eng.model);
    eng.eff_max_batch = opt.max_batch;
    // The library runs one clip per graph for the video families, so grouping their items only
    // inflates the working set — jepa-embed makes the same call, and for the same reason.
    if (eng.video_model && !opt.max_batch_explicit && eng.eff_max_batch > 1) eng.eff_max_batch = 1;

    for (int i = 0; i < opt.workers; i++) {
        auto wk = std::unique_ptr<Worker>(new Worker());
        wk->ctx = jepa_context_new(eng.model, opt.cp);
        if (!wk->ctx) {
            fprintf(stderr, "cannot create a context for worker %d\n", i);
            for (auto & w : eng.workers) jepa_context_free(w->ctx);
            jepa_model_free(eng.model);
            return 1;
        }
        if (jepa_context_max_batch(wk->ctx) < eng.eff_max_batch) jepa_context_set_max_batch(wk->ctx, eng.eff_max_batch);
        eng.workers.push_back(std::move(wk));
    }
    for (auto & w : eng.workers) {
        eng.idle.push_back(w.get());
        w->th = std::thread(worker_main, std::ref(eng), std::ref(*w));
    }
    eng.dispatcher = std::thread(dispatcher_main, std::ref(eng));

    const int n_threads = jepa_context_n_threads(eng.workers[0]->ctx);

    // ---- HTTP ------------------------------------------------------------------------------------
    httplib::Server server;
    g_server = &server;
    server.set_payload_max_length(opt.max_body_mb * 1024 * 1024);
    server.set_keep_alive_max_count(1000);

    auto send_error = [&](httplib::Response & res, const ApiError & e) {
        res.status = e.status;
        res.set_content(error_body(e).dump(), "application/json");
    };

    // Every handler runs inside this: the timing, the metrics, the in-flight gauge and the promise
    // that a malformed body becomes a 400 rather than a crashed process all live in one place.
    auto guard = [&](const char * endpoint, const httplib::Request & req, httplib::Response & res,
                     const std::function<uint64_t(const json &, httplib::Response &)> & fn, bool want_body) {
        const double t0 = now_s();
        g_metrics.in_flight++;
        uint64_t items = 0;
        try {
            json body = json::object();
            if (want_body) {
                if (req.body.empty()) throw ApiError(bad_request("the request body is empty"));
                body = json::parse(req.body, nullptr, false);
                if (body.is_discarded()) throw ApiError(bad_request("the request body is not valid JSON"));
                if (!body.is_object()) throw ApiError(bad_request("the request body must be a JSON object"));
            }
            items = fn(body, res);
            // httplib only fills in the default status when it writes the response, which is after
            // this; without the line the metrics would label every success "-1".
            if (res.status < 100) res.status = 200;
        } catch (const ApiError & e) {
            send_error(res, e);
        } catch (const std::bad_alloc &) {
            send_error(res, {503, "server_error", "out of memory serving this request"});
        } catch (const std::exception & e) {
            send_error(res, {500, "server_error", std::string("unhandled: ") + e.what()});
        }
        g_metrics.in_flight--;
        const double dt = now_s() - t0;
        g_metrics.observe(endpoint, res.status, dt, items);
        if (opt.verbose) {
            fprintf(stderr, "%s %s -> %d  %.1f ms  %llu item(s)\n", req.method.c_str(), endpoint,
                    res.status, dt * 1000.0, (unsigned long long) items);
        }
    };

    // --- POST /v1/embeddings ------------------------------------------------------------------
    auto handle_embeddings = [&](const json & body, httplib::Response & res) -> uint64_t {
        ApiError err{0, "", ""};
        if (!check_model_name(body, opt, err)) throw err;
        PoolKind pool = POOL_CLS;
        if (!parse_pool(body, eng.model, pool, err)) throw err;
        bool as_b64 = false;
        if (body.contains("encoding_format")) {
            if (!body["encoding_format"].is_string()) throw bad_request("\"encoding_format\" must be a string");
            const std::string f = body["encoding_format"].get<std::string>();
            if (f == "base64") as_b64 = true;
            else if (f != "float") throw bad_request("\"encoding_format\" must be \"float\" or \"base64\"");
        }
        std::vector<InputItem> items;
        if (!parse_input(body, "input", opt, items, err)) throw err;

        Job job;
        std::vector<Task *> raw;
        for (const InputItem & it : items) {
            auto t = std::unique_ptr<Task>(new Task());
            t->job = &job;
            t->pool = pool;
            if (!preprocess_item(eng, it, *t, err)) throw err;
            raw.push_back(t.get());
            job.tasks.push_back(std::move(t));
        }
        job.pending = (int) raw.size();
        eng.submit(raw);
        job.wait();

        for (const auto & t : job.tasks) if (!t->err.empty()) throw ApiError{500, "server_error", t->err};

        json data = json::array();
        int64_t total_tokens = 0;
        for (size_t i = 0; i < job.tasks.size(); i++) {
            const Task & t = *job.tasks[i];
            total_tokens += t.n_tokens;
            // `dim` is not in the OpenAI shape and does no harm there; it is what lets a
            // base64 reader reshape a --pool none vector, whose bytes are rows x dim.
            data.push_back(json{{"object", "embedding"},
                                {"index", (int64_t) i},
                                {"dim", t.dim},
                                {"embedding", embedding_value(t.emb, t.rows, t.dim, as_b64)}});
        }
        json out{{"object", "list"},
                 {"data", data},
                 {"model", opt.model_name},
                 {"usage", {{"prompt_tokens", total_tokens}, {"total_tokens", total_tokens}}}};
        res.set_content(out.dump(), "application/json");
        return (uint64_t) job.tasks.size();
    };

    // --- POST /classify -------------------------------------------------------------------------
    auto handle_classify = [&](const json & body, httplib::Response & res) -> uint64_t {
        ApiError err{0, "", ""};
        if (!check_model_name(body, opt, err)) throw err;
        if (!jepa_model_has_head(eng.model)) {
            throw ApiError{400, "invalid_request_error",
                           "\"" + opt.model_name + "\" has no classification head (jepa.head.kind = none); "
                           "use /v1/embeddings"};
        }
        int top_k = 5;
        if (body.contains("top_k")) {
            if (!body["top_k"].is_number_integer()) throw bad_request("\"top_k\" must be an integer");
            top_k = body["top_k"].get<int>();
            if (top_k < 1) throw bad_request("\"top_k\" must be at least 1");
        }
        const int n_classes = jepa_model_n_classes(eng.model);
        if (top_k > n_classes) top_k = n_classes;

        std::vector<InputItem> items;
        if (!parse_input(body, "input", opt, items, err)) throw err;

        Job job;
        std::vector<Task *> raw;
        for (const InputItem & it : items) {
            auto t = std::unique_ptr<Task>(new Task());
            t->job = &job;
            t->pool = POOL_NONE;            // the head takes the tokens, so no pooling is needed
            t->want_logits = true;
            if (!preprocess_item(eng, it, *t, err)) throw err;
            raw.push_back(t.get());
            job.tasks.push_back(std::move(t));
        }
        job.pending = (int) raw.size();
        eng.submit(raw);
        job.wait();
        for (const auto & t : job.tasks) if (!t->err.empty()) throw ApiError{500, "server_error", t->err};

        json data = json::array();
        int64_t total_tokens = 0;
        std::vector<float> probs((size_t) n_classes);
        std::vector<int32_t> idx((size_t) top_k);
        for (size_t i = 0; i < job.tasks.size(); i++) {
            const Task & t = *job.tasks[i];
            total_tokens += t.n_tokens;
            jepa_softmax(t.logits.data(), n_classes, probs.data());
            const int k = jepa_top_k(t.logits.data(), n_classes, top_k, idx.data());
            json preds = json::array();
            for (int j = 0; j < k; j++) {
                const char * label = jepa_model_label(eng.model, idx[(size_t) j]);
                preds.push_back(json{{"index", idx[(size_t) j]},
                                     {"label", label ? label : ""},
                                     {"probability", probs[(size_t) idx[(size_t) j]]},
                                     {"logit", t.logits[(size_t) idx[(size_t) j]]}});
            }
            data.push_back(json{{"object", "classification"}, {"index", (int64_t) i}, {"predictions", preds}});
        }
        json out{{"object", "list"},
                 {"data", data},
                 {"model", opt.model_name},
                 {"usage", {{"prompt_tokens", total_tokens}, {"total_tokens", total_tokens}}}};
        res.set_content(out.dump(), "application/json");
        return (uint64_t) job.tasks.size();
    };

    // --- POST /rollout --------------------------------------------------------------------------
    auto handle_rollout = [&](const json & body, httplib::Response & res) -> uint64_t {
        ApiError err{0, "", ""};
        if (!check_model_name(body, opt, err)) throw err;
        const bool is_ac = jepa_ac_tokens_per_frame(eng.model) > 0;
        const bool is_lewm = jepa_lewm_action_dim(eng.model) > 0 && jepa_model_has_projector(eng.model);
        if (!is_ac && !is_lewm) {
            throw ApiError{400, "invalid_request_error",
                           "\"" + opt.model_name + "\" is not a world model (jepa.pred.kind is neither "
                           "\"ac\" nor \"lewm\"); /rollout needs V-JEPA 2-AC or LeWM"};
        }
        const int action_dim = is_ac ? jepa_ac_action_dim(eng.model) : jepa_lewm_action_dim(eng.model);

        std::vector<InputItem> ctx_items;
        if (!parse_input(body, body.contains("context") ? "context" : "input", opt, ctx_items, err)) throw err;
        const int max_ctx = is_ac ? jepa_ac_max_frames(eng.model) : jepa_lewm_n_frames(eng.model);
        if ((int) ctx_items.size() > max_ctx) {
            throw bad_request("this predictor holds " + std::to_string(max_ctx) + " frames, the request gave "
                              + std::to_string(ctx_items.size()));
        }

        auto roll = std::unique_ptr<RolloutTask>(new RolloutTask());
        auto task = std::unique_ptr<Task>(new Task());
        task->kind = TASK_ROLLOUT;

        for (const InputItem & it : ctx_items) {
            Task tmp;
            if (!preprocess_item(eng, it, tmp, err)) throw err;
            roll->context_px.push_back(std::move(tmp.px));
            roll->ctx_T.push_back(tmp.T); roll->ctx_h.push_back(tmp.h); roll->ctx_w.push_back(tmp.w);
        }
        if (body.contains("goal") && !body["goal"].is_null()) {
            InputItem g;
            if (!parse_item(body["goal"], opt, g, err)) throw err;
            Task tmp;
            if (!preprocess_item(eng, g, tmp, err)) throw err;
            roll->goal_px = std::move(tmp.px);
            roll->goal_T = tmp.T; roll->goal_h = tmp.h; roll->goal_w = tmp.w;
        }
        if (body.contains("state")) {
            if (!body["state"].is_array()) throw bad_request("\"state\" must be an array of numbers");
            for (const json & x : body["state"]) {
                if (!x.is_number()) throw bad_request("\"state\" must be numbers");
                roll->state.push_back(x.get<float>());
            }
        }
        roll->want_latents = body.value("return_latents", false);

        if (body.contains("actions") && !body["actions"].is_null()) {
            if (!parse_actions(body["actions"], action_dim, 4096, 64, roll->actions,
                               roll->n_cand, roll->horizon, err)) throw err;
        }
        if (body.contains("plan") && !body["plan"].is_null()) {
            if (!is_ac) throw bad_request("\"plan\" is the V-JEPA 2-AC CEM loop; this model is not an AC world model");
            const json & p = body["plan"];
            if (!p.is_object()) throw bad_request("\"plan\" must be an object");
            roll->want_plan = true;
            roll->cem = jepa_ac_cem_default_params();
            auto geti = [&](const char * k, int & dst, int lo, int hi) {
                if (!p.contains(k)) return;
                if (!p[k].is_number_integer()) throw bad_request(std::string("\"plan.") + k + "\" must be an integer");
                const int v = p[k].get<int>();
                if (v < lo || v > hi) {
                    throw bad_request(std::string("\"plan.") + k + "\" must be between " + std::to_string(lo)
                                      + " and " + std::to_string(hi));
                }
                dst = v;
            };
            auto getf = [&](const char * k, float & dst) {
                if (!p.contains(k)) return;
                if (!p[k].is_number()) throw bad_request(std::string("\"plan.") + k + "\" must be a number");
                dst = p[k].get<float>();
            };
            geti("samples", roll->cem.samples, 1, 4096);
            geti("topk", roll->cem.topk, 1, 4096);
            geti("cem_steps", roll->cem.cem_steps, 1, 256);
            geti("horizon", roll->cem.horizon, 1, 64);
            getf("maxnorm", roll->cem.maxnorm);
            getf("gripper_clamp", roll->cem.gripper_clamp);
            if (p.contains("seed")) {
                if (!p["seed"].is_number_unsigned()) throw bad_request("\"plan.seed\" must be a non-negative integer");
                roll->cem.seed = p["seed"].get<uint32_t>();
            }
            if (roll->cem.topk > roll->cem.samples) throw bad_request("\"plan.topk\" cannot exceed \"plan.samples\"");
        }
        if (!roll->want_plan && roll->n_cand == 0) throw bad_request("give \"actions\", \"plan\", or both");

        Job job;
        task->roll = std::move(roll);
        Task * raw = task.get();
        task->job = &job;
        job.tasks.push_back(std::move(task));
        job.pending = 1;
        eng.submit({raw});
        job.wait();
        if (!raw->err.empty()) throw ApiError{500, "server_error", raw->err};

        const RolloutTask & r = *raw->roll;
        json out{{"object", "rollout"}, {"model", opt.model_name}};
        if (!r.energies.empty()) {
            json energies = json::array();
            for (const auto & row : r.energies) energies.push_back(json(row));
            out["energies"] = energies;
            int best = 0;
            for (int c = 1; c < r.n_cand; c++) {
                if (r.energies[(size_t) c].back() < r.energies[(size_t) best].back()) best = c;
            }
            out["best"] = json{{"index", best},
                               {"energy", r.energies[(size_t) best].back()},
                               {"actions", json(std::vector<float>(
                                    r.actions.begin() + (size_t) best * r.horizon * action_dim,
                                    r.actions.begin() + (size_t) (best + 1) * r.horizon * action_dim))}};
            out["n_candidates"] = r.n_cand;
            out["horizon"] = r.horizon;
        }
        if (r.want_plan) {
            out["plan"] = json{{"actions", json(r.plan_actions)},
                               {"action_dim", action_dim},
                               {"horizon", r.cem.horizon},
                               {"energy_per_iteration", json(r.plan_energy)}};
        }
        if (r.want_latents && !r.latents.empty()) {
            out["latents"] = json{{"shape", json::array({r.n_cand, r.horizon, r.latent_rows, r.latent_dim})},
                                  {"b64", b64_encode((const uint8_t *) r.latents.data(), r.latents.size() * sizeof(float))}};
        }
        out["usage"] = json{{"prompt_tokens", r.n_tokens}, {"total_tokens", r.n_tokens}};
        res.set_content(out.dump(), "application/json");
        return 1;
    };

    server.Post("/v1/embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        guard("/v1/embeddings", req, res, handle_embeddings, true);
    });
    server.Post("/classify", [&](const httplib::Request & req, httplib::Response & res) {
        guard("/classify", req, res, handle_classify, true);
    });
    server.Post("/rollout", [&](const httplib::Request & req, httplib::Response & res) {
        guard("/rollout", req, res, handle_rollout, true);
    });

    server.Get("/health", [&](const httplib::Request & req, httplib::Response & res) {
        guard("/health", req, res, [&](const json &, httplib::Response & r) -> uint64_t {
            json out{{"status", "ok"},
                     {"version", jepa_version()},
                     {"model", opt.model_name},
                     {"path", opt.model_path},
                     {"family", eng.family},
                     {"file_type", jepa_model_file_type_name(eng.model)},
                     {"embed_dim", jepa_model_embed_dim(eng.model)},
                     {"n_frames", jepa_model_n_frames(eng.model)},
                     {"has_cls", jepa_model_has_cls(eng.model)},
                     {"has_head", jepa_model_has_head(eng.model)},
                     {"has_predictor", jepa_model_has_predictor(eng.model)},
                     {"has_projector", jepa_model_has_projector(eng.model)},
                     {"n_classes", jepa_model_n_classes(eng.model)},
                     {"weights_mib", (double) jepa_model_n_bytes(eng.model) / (1024.0 * 1024.0)},
                     {"backend", jepa_model_device_name(eng.model)},
                     {"device", jepa_model_device(eng.model)},
                     {"is_gpu", jepa_model_is_gpu(eng.model)},
                     {"threads", n_threads},
                     {"workers", opt.workers},
                     {"max_batch", eng.eff_max_batch},
                     {"max_wait_ms", opt.max_wait_ms},
                     {"allow_local_files", opt.allow_local_files},
                     {"queue_depth", g_metrics.queue_depth.load()}};
            r.set_content(out.dump(), "application/json");
            return 0;
        }, false);
    });

    server.Get("/v1/models", [&](const httplib::Request & req, httplib::Response & res) {
        guard("/v1/models", req, res, [&](const json &, httplib::Response & r) -> uint64_t {
            json out{{"object", "list"},
                     {"data", json::array({json{{"id", opt.model_name},
                                                {"object", "model"},
                                                {"owned_by", "jepa.cpp"},
                                                {"family", eng.family},
                                                {"file_type", jepa_model_file_type_name(eng.model)},
                                                {"embed_dim", jepa_model_embed_dim(eng.model)},
                                                {"n_classes", jepa_model_n_classes(eng.model)}}})}};
            r.set_content(out.dump(), "application/json");
            return 0;
        }, false);
    });

    server.Get("/metrics", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(render_metrics(opt, eng.model), "text/plain; version=0.0.4; charset=utf-8");
    });

    server.set_error_handler([&](const httplib::Request &, httplib::Response & res) {
        if (res.body.empty()) {
            res.set_content(error_body({res.status, res.status == 404 ? "not_found" : "server_error",
                                        res.status == 404 ? "no such endpoint" : "request failed"}).dump(),
                            "application/json");
        }
    });
    server.set_exception_handler([&](const httplib::Request &, httplib::Response & res, std::exception_ptr) {
        res.status = 500;
        res.set_content(error_body({500, "server_error", "the request handler threw"}).dump(), "application/json");
    });

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    // --port 0 asks the kernel for a free one, which is how a test harness gets a port nothing else
    // can take between the choice and the bind; the number then comes back on stdout, below.
    const int port = opt.port == 0 ? server.bind_to_any_port(opt.host.c_str())
                                   : (server.bind_to_port(opt.host.c_str(), opt.port) ? opt.port : -1);
    if (port <= 0) {
        fprintf(stderr, "cannot bind %s:%d\n", opt.host.c_str(), opt.port);
        return 1;
    }
    printf("jepa-server %s: %s (%s, %s, %s) on http://%s:%d\n", jepa_version(), opt.model_name.c_str(),
           eng.family.c_str(), jepa_model_file_type_name(eng.model), jepa_model_device_name(eng.model),
           opt.host.c_str(), port);
    printf("  workers %d x %d threads | max-batch %d | max-wait %d ms | local files %s\n",
           opt.workers, n_threads, eng.eff_max_batch, opt.max_wait_ms, opt.allow_local_files ? "allowed" : "refused");
    if (opt.host != "127.0.0.1" && opt.host != "localhost" && opt.host != "::1") {
        printf("  WARNING: %s is not loopback. This server has no TLS and no authentication — put it\n"
               "           behind something that does before anything but you can reach it.\n", opt.host.c_str());
    }
    fflush(stdout);

    server.listen_after_bind();

    // ---- shutdown ---------------------------------------------------------------------------------
    {
        std::lock_guard<std::mutex> g(eng.mu);
        eng.stopping = true;
    }
    eng.cv_work.notify_all();
    if (eng.dispatcher.joinable()) eng.dispatcher.join();
    for (auto & w : eng.workers) {
        { std::lock_guard<std::mutex> g(w->m); w->stop = true; }
        w->cv.notify_all();
    }
    for (auto & w : eng.workers) {
        if (w->th.joinable()) w->th.join();
        jepa_context_free(w->ctx);
    }
    jepa_model_free(eng.model);
    return 0;
}
