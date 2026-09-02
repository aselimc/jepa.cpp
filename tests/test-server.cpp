// test-server: jepa-server end to end, against jepa-embed.
//
//   test-server --server build/jepa-server --embed build/jepa-embed \
//               --model models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf \
//               --image a.jpg --image b.jpg [--image ...]
//
// The suite starts a real server on a loopback port the server itself picks (`--port 0`, whose
// number comes back on its stdout, so no other process can take it between the choice and the
// bind), talks HTTP to it, and shuts it down with SIGTERM. What it checks:
//
//   * /health answers before anything else does, and describes the model that was loaded;
//   * /v1/embeddings on one image is BIT-IDENTICAL to `jepa-embed --pool cls` on the same file —
//     the whole point of putting the server on top of the C API rather than beside it;
//   * a four-item request, which the dispatcher folds into one graph, returns rows bit-identical to
//     four one-item requests, which it cannot: dynamic batching is a scheduling decision and never
//     a numeric one (the engine's own guarantee, tests/test-batch.cpp, seen from the outside);
//   * /classify on a model with no head is a 400 with a JSON error object, not a crash;
//   * a body that is not JSON, an unknown model name and a truncated base64 image are all 4xx, and
//     the server still serves /health afterwards;
//   * /metrics is Prometheus text and records the batch sizes that actually happened;
//   * the process exits 0 on SIGTERM.
//
// Everything runs on 127.0.0.1 and finishes in a few seconds on the small models. With no --model
// (or a missing file) the binary prints a skip line and exits 0, like the other asset-gated suites.
#include "jepa.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
int main() {
    printf("test-server: skipped (the harness needs POSIX process control)\n");
    return 0;
}
#else

#include "httplib.h"
#include "json.hpp"

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;

static int g_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) g_fail++;
}

static bool file_exists(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

static const char * B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string b64_encode(const std::vector<uint8_t> & v) {
    std::string s;
    size_t i = 0;
    for (; i + 3 <= v.size(); i += 3) {
        const uint32_t x = ((uint32_t) v[i] << 16) | ((uint32_t) v[i + 1] << 8) | v[i + 2];
        s += B64[(x >> 18) & 63]; s += B64[(x >> 12) & 63]; s += B64[(x >> 6) & 63]; s += B64[x & 63];
    }
    if (i < v.size()) {
        uint32_t x = (uint32_t) v[i] << 16;
        if (i + 1 < v.size()) x |= (uint32_t) v[i + 1] << 8;
        s += B64[(x >> 18) & 63];
        s += B64[(x >> 12) & 63];
        s += (i + 1 < v.size()) ? B64[(x >> 6) & 63] : '=';
        s += '=';
    }
    return s;
}

static std::vector<uint8_t> read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// A jepa-server child, its port and its lifetime.
struct Server {
    pid_t pid = -1;
    int   port = 0;
    std::string banner;

    bool start(const std::string & exe, const std::vector<std::string> & args) {
        int fds[2];
        if (pipe(fds) != 0) { perror("pipe"); return false; }
        pid = fork();
        if (pid < 0) { perror("fork"); return false; }
        if (pid == 0) {
            close(fds[0]);
            dup2(fds[1], STDOUT_FILENO);
            close(fds[1]);
            std::vector<char *> argv;
            argv.push_back(const_cast<char *>(exe.c_str()));
            for (const std::string & a : args) argv.push_back(const_cast<char *>(a.c_str()));
            argv.push_back(nullptr);
            execv(exe.c_str(), argv.data());
            fprintf(stderr, "execv %s: %s\n", exe.c_str(), strerror(errno));
            _exit(127);
        }
        close(fds[1]);
        // The server prints "... on http://127.0.0.1:PORT" once it is bound, which is both the
        // readiness signal and the port number.
        std::string line;
        char c = 0;
        while (read(fds[0], &c, 1) == 1) {
            if (c == '\n') break;
            line += c;
        }
        close(fds[0]);
        banner = line;
        const size_t at = line.rfind(':');
        if (at == std::string::npos) { fprintf(stderr, "no port in the banner: %s\n", line.c_str()); return false; }
        port = atoi(line.c_str() + at + 1);
        return port > 0;
    }

    int stop() {
        if (pid <= 0) return 0;
        kill(pid, SIGTERM);
        int status = 0;
        for (int i = 0; i < 300; i++) {
            const pid_t r = waitpid(pid, &status, WNOHANG);
            if (r == pid) { pid = -1; return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status); }
            usleep(100000);
        }
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        pid = -1;
        return -1;
    }
};

// jepa-embed's own answer for one image, through the binary the tools build produces.
static bool embed_reference(const std::string & exe, const std::string & model,
                            const std::vector<std::string> & images, const std::string & pool,
                            const std::string & out_npy, std::vector<float> & rows, int64_t * dim) {
    std::string cmd = "\"" + exe + "\" -m \"" + model + "\"";
    for (const std::string & i : images) cmd += " -i \"" + i + "\"";
    cmd += " --pool " + pool + " --print-n 0 --as-images -t 4 -o \"" + out_npy + "\" > /dev/null 2>&1";
    const int rc = system(cmd.c_str());
    if (rc != 0) { fprintf(stderr, "jepa-embed failed (%d): %s\n", rc, cmd.c_str()); return false; }
    npy::Array a = npy::load(out_npy);
    if (a.shape.size() != 2) { fprintf(stderr, "jepa-embed wrote %zu dims\n", a.shape.size()); return false; }
    *dim = a.shape[1];
    const float * f = (const float *) a.bytes.data();
    rows.assign(f, f + (size_t) a.shape[0] * a.shape[1]);
    return true;
}

// Bit patterns, which is what "bit-identical" has to mean for floats.
static int n_bits_differ(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return -1;
    int n = 0;
    for (size_t i = 0; i < a.size(); i++) {
        uint32_t x, y;
        memcpy(&x, &a[i], 4);
        memcpy(&y, &b[i], 4);
        if (x != y) n++;
    }
    return n;
}

static std::vector<float> to_vec(const json & arr) {
    std::vector<float> v;
    for (const json & x : arr) v.push_back(x.get<float>());
    return v;
}

int main(int argc, char ** argv) {
    std::string server_exe, embed_exe, model;
    std::vector<std::string> images;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", argv[i]); exit(2); }
            return argv[++i];
        };
        if      (a == "--server") server_exe = next();
        else if (a == "--embed")  embed_exe = next();
        else if (a == "--model")  model = next();
        else if (a == "--image")  images.push_back(next());
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (server_exe.empty() || !file_exists(server_exe)) {
        printf("test-server: skipped (no --server binary)\n");
        return 0;
    }
    if (model.empty() || !file_exists(model)) {
        printf("test-server: skipped (no model — run scripts/download_models.sh and scripts/convert.py)\n");
        return 0;
    }
    if (images.empty()) {
        printf("test-server: skipped (no --image — run scripts/download_fixtures.sh)\n");
        return 0;
    }
    for (const std::string & i : images) {
        if (!file_exists(i)) { printf("test-server: skipped (%s is not there)\n", i.c_str()); return 0; }
    }

    Server srv;
    if (!srv.start(server_exe, {"-m", model, "--host", "127.0.0.1", "--port", "0",
                                "--workers", "2", "--threads", "4", "--max-batch", "8",
                                "--max-wait-ms", "20", "--model-name", "test-model"})) {
        fprintf(stderr, "cannot start %s\n", server_exe.c_str());
        return 1;
    }
    printf("test-server: %s\n", srv.banner.c_str());

    httplib::Client cli("127.0.0.1", srv.port);
    cli.set_read_timeout(120, 0);
    cli.set_write_timeout(120, 0);

    // ---- /health -------------------------------------------------------------------------------
    {
        auto res = cli.Get("/health");
        check(res && res->status == 200, "GET /health is 200");
        if (res && res->status == 200) {
            const json h = json::parse(res->body, nullptr, false);
            check(!h.is_discarded() && h.value("status", "") == "ok", "/health says ok");
            check(h.value("model", "") == "test-model", "/health reports --model-name");
            check(h.value("workers", 0) == 2, "/health reports the worker count");
            check(h.value("max_batch", 0) == 8, "/health reports max_batch");
            check(h.value("allow_local_files", true) == false, "local files are refused by default");
        }
    }
    {
        auto res = cli.Get("/v1/models");
        check(res && res->status == 200, "GET /v1/models is 200");
        if (res && res->status == 200) {
            const json m = json::parse(res->body, nullptr, false);
            check(!m.is_discarded() && m.value("object", "") == "list" && m["data"].size() == 1,
                  "/v1/models lists the one loaded model");
        }
    }

    // ---- one embedding, against jepa-embed -----------------------------------------------------
    std::vector<std::string> b64_images;
    for (const std::string & p : images) b64_images.push_back(b64_encode(read_file(p)));

    std::vector<float> ref_rows;
    int64_t ref_dim = 0;
    const bool have_embed = !embed_exe.empty() && file_exists(embed_exe);
    if (have_embed) {
        const std::string tmp_npy = std::string("test-server-ref.npy");
        if (!embed_reference(embed_exe, model, images, "cls", tmp_npy, ref_rows, &ref_dim)) return 1;
        remove(tmp_npy.c_str());
        check((int64_t) ref_rows.size() == (int64_t) images.size() * ref_dim,
              "jepa-embed produced one row per image");
    } else {
        printf("  note: no --embed binary, skipping the jepa-embed comparison\n");
    }

    std::vector<std::vector<float>> single(images.size());
    for (size_t i = 0; i < images.size(); i++) {
        json body{{"model", "test-model"}, {"input", json{{"b64", b64_images[i]}}}, {"pool", "cls"}};
        auto res = cli.Post("/v1/embeddings", body.dump(), "application/json");
        check(res && res->status == 200, "POST /v1/embeddings (one item) is 200");
        if (!res || res->status != 200) return 1;
        const json r = json::parse(res->body, nullptr, false);
        check(!r.is_discarded() && r.value("object", "") == "list" && r["data"].size() == 1,
              "the response is an OpenAI-shaped list of one");
        check(r["data"][0].value("object", "") == "embedding" && r["data"][0].value("index", -1) == 0,
              "the datum is an embedding at index 0");
        check(r.contains("usage") && r["usage"].value("total_tokens", 0) > 0, "usage reports the token count");
        single[i] = to_vec(r["data"][0]["embedding"]);
    }
    if (have_embed) {
        int diff = 0;
        for (size_t i = 0; i < images.size(); i++) {
            std::vector<float> want(ref_rows.begin() + (size_t) i * ref_dim,
                                    ref_rows.begin() + (size_t) (i + 1) * ref_dim);
            const int d = n_bits_differ(single[i], want);
            if (d != 0) { fprintf(stderr, "    image %zu: %d of %lld floats differ\n", i, d, (long long) ref_dim); diff += d < 0 ? 1 : d; }
        }
        check(diff == 0, "every /v1/embeddings vector is bit-identical to jepa-embed --pool cls");
    }

    // ---- one request of many items: the dispatcher's batch, same numbers -----------------------
    {
        json arr = json::array();
        for (const std::string & b : b64_images) arr.push_back(json{{"b64", b}});
        json body{{"input", arr}, {"pool", "cls"}};
        auto res = cli.Post("/v1/embeddings", body.dump(), "application/json");
        check(res && res->status == 200, "POST /v1/embeddings (batched) is 200");
        if (res && res->status == 200) {
            const json r = json::parse(res->body, nullptr, false);
            check(r["data"].size() == images.size(), "one datum per input item, in order");
            int diff = 0;
            for (size_t i = 0; i < images.size(); i++) {
                check(r["data"][i].value("index", -1) == (int) i, "datum " + std::to_string(i) + " keeps its index");
                const int d = n_bits_differ(to_vec(r["data"][i]["embedding"]), single[i]);
                diff += d < 0 ? 1 : d;
            }
            check(diff == 0, "batched rows are bit-identical to the same items served one at a time");
        }
    }

    // ---- base64 encoding_format ships the same bits ---------------------------------------------
    {
        json body{{"input", json{{"b64", b64_images[0]}}}, {"pool", "cls"}, {"encoding_format", "base64"}};
        auto res = cli.Post("/v1/embeddings", body.dump(), "application/json");
        check(res && res->status == 200, "encoding_format base64 is accepted");
        if (res && res->status == 200) {
            const json r = json::parse(res->body, nullptr, false);
            const std::string s = r["data"][0]["embedding"].get<std::string>();
            std::vector<uint8_t> raw;
            {   // decode
                int8_t rev[256];
                memset(rev, -1, sizeof(rev));
                for (int i = 0; i < 64; i++) rev[(unsigned char) B64[i]] = (int8_t) i;
                uint32_t acc = 0;
                int bits = 0;
                for (char c : s) {
                    if (c == '=') break;
                    const int8_t v = rev[(unsigned char) c];
                    if (v < 0) continue;
                    acc = (acc << 6) | (uint32_t) v;
                    bits += 6;
                    if (bits >= 8) { bits -= 8; raw.push_back((uint8_t) ((acc >> bits) & 0xff)); }
                }
            }
            std::vector<float> got(raw.size() / 4);
            memcpy(got.data(), raw.data(), got.size() * 4);
            check(n_bits_differ(got, single[0]) == 0, "the base64 floats are the JSON floats, bit for bit");
        }
    }

    // ---- error paths ----------------------------------------------------------------------------
    {
        json body{{"input", json{{"b64", b64_images[0]}}}};
        auto res = cli.Post("/classify", body.dump(), "application/json");
        check(res && res->status == 400, "/classify on a model with no head is 400");
        if (res) {
            const json e = json::parse(res->body, nullptr, false);
            check(!e.is_discarded() && e.contains("error") && e["error"].contains("message"),
                  "the 400 carries a JSON error object");
            check(!e.is_discarded() && e["error"].value("message", "").find("no classification head") != std::string::npos,
                  "the message names the missing head");
        }
    }
    {
        auto res = cli.Post("/v1/embeddings", "{not json at all", "application/json");
        check(res && res->status == 400, "a body that is not JSON is 400");
    }
    {
        auto res = cli.Post("/v1/embeddings", "[1, 2, 3]", "application/json");
        check(res && res->status == 400, "a JSON array body is 400");
    }
    {
        json body{{"model", "some-other-model"}, {"input", json{{"b64", b64_images[0]}}}};
        auto res = cli.Post("/v1/embeddings", body.dump(), "application/json");
        check(res && res->status == 404, "an unknown model name is 404");
    }
    {
        json body{{"input", json{{"b64", "@@@@not base64@@@@"}}}};
        auto res = cli.Post("/v1/embeddings", body.dump(), "application/json");
        check(res && res->status == 400, "a body that is not base64 is 400");
    }
    {
        json body{{"input", json{{"b64", b64_encode({'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'm', 'g'})}}}};
        auto res = cli.Post("/v1/embeddings", body.dump(), "application/json");
        check(res && res->status == 400, "bytes that are not an image are 400");
    }
    {
        json body{{"input", "/etc/passwd"}};
        auto res = cli.Post("/v1/embeddings", body.dump(), "application/json");
        check(res && res->status == 400, "a local path without --allow-local-files is 400");
    }
    {
        json body{{"input", json{{"url", "http://example.invalid/x.png"}}}};
        auto res = cli.Post("/v1/embeddings", body.dump(), "application/json");
        check(res && res->status == 400, "a url input is 400 — the server fetches nothing");
    }
    {
        auto res = cli.Get("/nope");
        check(res && res->status == 404, "an unknown path is 404");
    }
    {
        auto res = cli.Get("/health");
        check(res && res->status == 200, "/health still answers after every bad request");
    }

    // ---- /metrics --------------------------------------------------------------------------------
    {
        auto res = cli.Get("/metrics");
        check(res && res->status == 200, "GET /metrics is 200");
        if (res && res->status == 200) {
            const std::string & b = res->body;
            check(b.find("# TYPE jepa_requests_total counter") != std::string::npos, "/metrics has the request counter");
            check(b.find("jepa_request_duration_seconds_bucket") != std::string::npos, "/metrics has the latency histogram");
            check(b.find("jepa_batch_size_count") != std::string::npos, "/metrics has the batch-size histogram");
            check(b.find("jepa_queue_depth") != std::string::npos, "/metrics has the queue gauge");
            check(b.find("endpoint=\"/v1/embeddings\",status=\"200\"") != std::string::npos,
                  "/metrics counted the successful embeddings");
            check(b.find("endpoint=\"/v1/embeddings\",status=\"400\"") != std::string::npos,
                  "/metrics counted the refused embeddings");
            // The four-item request went through one graph, so a batch larger than one was recorded.
            const size_t le1 = b.find("jepa_batch_size_bucket{le=\"1\"}");
            const size_t cnt = b.find("jepa_batch_size_count ");
            if (le1 != std::string::npos && cnt != std::string::npos && images.size() > 1) {
                const long at_most_1 = atol(b.c_str() + b.find(' ', le1 + 20));
                const long total = atol(b.c_str() + cnt + strlen("jepa_batch_size_count "));
                check(total > at_most_1, "at least one graph carried more than one item");
            }
        }
    }

    // ---- clean shutdown ---------------------------------------------------------------------------
    const int rc = srv.stop();
    check(rc == 0, "the server exits 0 on SIGTERM");

    printf("test-server: %s\n", g_fail == 0 ? "all checks passed" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}

#endif  // _WIN32
