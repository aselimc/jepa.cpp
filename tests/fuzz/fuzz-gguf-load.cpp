// fuzz-gguf-load — feed arbitrary bytes to the GGUF loader (and, when they load, to a shallow
// inference pass) and require that jepa.cpp refuses them with a message instead of crashing.
//
// Built only with -DJEPA_FUZZ=ON. Two front ends over one `run_one(bytes, size)`:
//
//   libFuzzer   (clang, -fsanitize=fuzzer): LLVMFuzzerTestOneInput below is the entry point.
//                 cmake -S . -B build-fuzz -DJEPA_FUZZ=ON -DCMAKE_C_COMPILER=clang
//                   -DCMAKE_CXX_COMPILER=clang++
//                   -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
//                 build-fuzz/fuzz-gguf-load tests/fuzz/corpus -max_len=262144
//
//   mutation loop (any compiler, the default here): a deterministic splitmix64-driven mutator over
//                 the seed corpus, so a GCC-only box — this one — still gets the same coverage of
//                 the loader's failure paths. Same corpus directory, same inputs, reproducible from
//                 (--seed, --runs).
//                 build/fuzz-gguf-load --corpus tests/fuzz/corpus --seconds 3600 --fork
//
// The corpus comes from scripts/make_fuzz_corpus.py (shrunk-but-loadable minis of lewm-pusht and
// lejepa-vits16, their truncations, and one-key mutations).
//
// `--fork` runs every input in a forked child, so a crash is recorded (as `crash-<n>-sig<k>.gguf`
// in --out) and the run continues; without it the first crash stops the process, which is what you
// want when triaging one input under a sanitizer.
#include "jepa.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef _WIN32
#  include <dirent.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

// ---------------------------------------------------------------------------------------------
// the target
// ---------------------------------------------------------------------------------------------
namespace {

std::string g_tmp_path;            // scratch file the bytes are written to (jepa_model_load takes a path)
bool        g_load_only = false;   // skip the inference pass after a successful load

// A model that loads is also *used*, briefly: a crafted file whose metadata promises a tensor the
// graph builders require is a loader bug too, and only a graph build finds it.
void exercise(jepa_model * m) {
    jepa_context_params cp = jepa_context_default_params();
    cp.n_threads = 1;
    jepa_context * ctx = jepa_context_new(m, cp);
    if (!ctx) return;

    const int C = 3;
    const int P = jepa_model_patch_size(m) > 0 ? jepa_model_patch_size(m) : 16;
    const int S = jepa_model_img_size(m) > 0 ? jepa_model_img_size(m) : P;
    const int T = jepa_model_n_frames(m) > 0 ? jepa_model_n_frames(m) : 1;
    // cap the work: a mutated file can ask for a very large grid and this is not a benchmark
    const long long px = (long long) S * S;
    if (px > 0 && px <= 1 << 16 && T <= 64) {
        std::vector<float> px_data((size_t) C * T * px, 0.25f);
        jepa_input in = { px_data.data(), 1, C, T, S, S };
        jepa_output enc = {};
        if (jepa_encode(ctx, &in, &enc) == 0 && enc.data) {
            jepa_output pooled = {};
            if (jepa_pool_mean(m, &enc, &pooled) == 0) jepa_free(pooled.data);
            if (jepa_model_has_cls(m) && jepa_pool_cls(m, &enc, &pooled) == 0) jepa_free(pooled.data);
            if (jepa_model_has_projector(m) && jepa_lewm_project(ctx, &enc, &pooled) == 0) jepa_free(pooled.data);
            if (jepa_model_has_head(m)) {
                jepa_output logits = {};
                if (jepa_head(ctx, &enc, &logits) == 0) jepa_free(logits.data);
            }
            if (jepa_model_has_predictor(m)) {
                const int32_t ids[4] = { 0, 1, 2, 3 };
                jepa_output pred = {};
                if (jepa_predict(ctx, &enc, ids, 2, ids + 2, 2, &pred) == 0) jepa_free(pred.data);
            }
            jepa_free(enc.data);
        }
    }
    const int nf = jepa_lewm_n_frames(m), ad = jepa_lewm_action_dim(m);
    if (nf > 0 && nf <= 64 && ad > 0 && ad <= 4096) {
        const int D = jepa_model_embed_dim(m);
        if (D > 0 && D <= 65536) {
            std::vector<float> embs((size_t) nf * D, 0.1f), acts((size_t) nf * ad, 0.2f);
            jepa_output out = {};
            if (jepa_lewm_predict(ctx, embs.data(), acts.data(), nf, &out) == 0) jepa_free(out.data);
        }
    }
    jepa_context_free(ctx);
}

} // namespace

// One fuzz iteration: `data` is the whole candidate GGUF file.
extern "C" int run_one(const uint8_t * data, size_t size) {
    const char * path = g_tmp_path.empty() ? "fuzz-gguf-load.tmp" : g_tmp_path.c_str();
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "fuzz-gguf-load: cannot write the scratch file %s — pass --out to a writable "
                        "directory, or every iteration is a no-op\n", path);
        abort();
    }
    if (size) fwrite(data, 1, size, f);
    fclose(f);

    jepa_error_reset();
    jepa_model * m = jepa_model_load(path, false);
    if (m) {
        if (!g_load_only) exercise(m);
        jepa_model_free(m);
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    return run_one(data, size);
}

#ifndef JEPA_FUZZ_LIBFUZZER
// ---------------------------------------------------------------------------------------------
// deterministic mutation loop (the front end used when there is no libFuzzer)
// ---------------------------------------------------------------------------------------------
namespace {

uint64_t g_state = 0;

uint64_t next_u64() {   // splitmix64
    uint64_t z = (g_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
size_t next_below(size_t n) { return n ? (size_t) (next_u64() % n) : 0; }

const uint64_t INTERESTING[] = {
    0, 1, 2, 3, 4, 7, 8, 15, 16, 63, 64, 127, 128, 255, 256, 1023, 1024, 4095, 4096,
    0x7fffu, 0xffffu, 0x7fffffffu, 0x80000000u, 0xffffffffu,
    0x7fffffffffffffffull, 0x8000000000000000ull, 0xffffffffffffffffull,
};
const char * const WORDS[] = {
    "GGUF", "jepa", "general.architecture", "jepa.family", "ijepa", "vjepa", "vjepa2", "vjepa2_1",
    "levjepa", "hfvit", "lewm", "masked", "ac", "none", "attentive_pool", "linear_cls",
    "gelu_erf", "gelu_tanh", "silu", "swish", "relu", "sincos2d", "sincos3d", "learned", "rope3d",
    "full", "block_causal", "tiled", "interleaved", "shortest_edge", "squash", "bilinear", "bicubic",
    "jepa.enc.embed_dim", "jepa.enc.n_layer", "jepa.enc.n_head", "jepa.enc.ffn_dim",
    "jepa.enc.patch_size", "jepa.enc.attn_mode", "jepa.pred.kind", "jepa.head.kind",
    "enc.patch_embed.weight", "enc.pos_embed", "enc.cls_token", "pred.mask_tokens", "head.query",
};
const size_t N_WORDS = sizeof(WORDS) / sizeof(WORDS[0]);

void put_int(std::vector<uint8_t> & b, size_t off, int width, uint64_t v) {
    for (int i = 0; i < width && off + (size_t) i < b.size(); i++) b[off + (size_t) i] = (uint8_t) (v >> (8 * i));
}

void mutate(std::vector<uint8_t> & b, const std::vector<std::vector<uint8_t>> & corpus, size_t cap) {
    if (b.empty()) b.push_back((uint8_t) next_u64());
    switch (next_u64() % 9) {
        case 0: {   // flip one bit
            const size_t o = next_below(b.size());
            b[o] ^= (uint8_t) (1u << (next_u64() % 8));
            break;
        }
        case 1: {   // set one byte
            b[next_below(b.size())] = (uint8_t) next_u64();
            break;
        }
        case 2: {   // an interesting integer at a random offset (1/2/4/8 bytes, little endian)
            static const int widths[] = { 1, 2, 4, 8 };
            const int w = widths[next_u64() % 4];
            put_int(b, next_below(b.size()), w, INTERESTING[next_u64() % (sizeof(INTERESTING) / sizeof(INTERESTING[0]))]);
            break;
        }
        case 3: {   // truncate
            b.resize(1 + next_below(b.size()));
            break;
        }
        case 4: {   // duplicate a chunk in place (grow)
            const size_t o = next_below(b.size());
            const size_t n = 1 + next_below(b.size() - o);
            if (b.size() + n <= cap) b.insert(b.begin() + (long) o, b.begin() + (long) o, b.begin() + (long) (o + n));
            break;
        }
        case 5: {   // erase a chunk
            const size_t o = next_below(b.size());
            const size_t n = 1 + next_below(b.size() - o);
            b.erase(b.begin() + (long) o, b.begin() + (long) (o + n));
            break;
        }
        case 6: {   // splice bytes in from another corpus entry
            if (corpus.empty()) break;
            const std::vector<uint8_t> & other = corpus[next_below(corpus.size())];
            if (other.empty()) break;
            const size_t so = next_below(other.size());
            const size_t n = 1 + next_below(std::min(other.size() - so, (size_t) 512));
            const size_t o = next_below(b.size());
            for (size_t i = 0; i < n && o + i < b.size(); i++) b[o + i] = other[so + i];
            break;
        }
        case 7: {   // overwrite with a dictionary word (keys, family names, enum spellings)
            const char * w = WORDS[next_u64() % N_WORDS];
            const size_t n = strlen(w);
            if (b.size() > n) {
                const size_t o = next_below(b.size() - n);
                memcpy(b.data() + o, w, n);
            }
            break;
        }
        default: {  // a short run of random bytes
            const size_t o = next_below(b.size());
            const size_t n = 1 + next_below(std::min(b.size() - o, (size_t) 32));
            for (size_t i = 0; i < n; i++) b[o + i] = (uint8_t) next_u64();
            break;
        }
    }
    if (b.size() > cap) b.resize(cap);
}

bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? (size_t) n : 0);
    const bool ok = out.empty() || fread(out.data(), 1, out.size(), f) == out.size();
    fclose(f);
    return ok;
}

void write_file(const std::string & path, const std::vector<uint8_t> & b) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return;
    if (!b.empty()) fwrite(b.data(), 1, b.size(), f);
    fclose(f);
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s [--corpus DIR] [--out DIR] [--seconds S] [--runs N] [--seed S]\n"
            "          [--max-len N] [--timeout S] [--fork] [--load-only] [--file F ...]\n"
            "  --file F     run F verbatim (no mutation) and exit — the regression-replay mode\n"
            "  --timeout S  fork mode: seconds one input may take before it counts as a hang (10)\n",
            argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string corpus_dir = "tests/fuzz/corpus";
    std::string out_dir;
    std::vector<std::string> replay;
    double seconds = 0;
    long long runs = 0;
    uint64_t seed = 0;
    size_t max_len = 1u << 20;
    bool fork_mode = false;
    unsigned timeout_s = 10;   // fork mode: a child that outlives this is a hang, and hangs are findings

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--corpus" && i + 1 < argc)       corpus_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc)     out_dir = argv[++i];
        else if (a == "--seconds" && i + 1 < argc) seconds = atof(argv[++i]);
        else if (a == "--runs" && i + 1 < argc)    runs = atoll(argv[++i]);
        else if (a == "--seed" && i + 1 < argc)    seed = (uint64_t) atoll(argv[++i]);
        else if (a == "--max-len" && i + 1 < argc) max_len = (size_t) atoll(argv[++i]);
        else if (a == "--timeout" && i + 1 < argc) timeout_s = (unsigned) atoi(argv[++i]);
        else if (a == "--fork")                    fork_mode = true;
        else if (a == "--load-only")               g_load_only = true;
        else if (a == "--file" && i + 1 < argc)    replay.push_back(argv[++i]);
        else if (a == "-h" || a == "--help")     { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument '%s'\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (out_dir.empty()) out_dir = "fuzz-findings";
    g_tmp_path = out_dir + "/current-" + std::to_string((long long) getpid()) + ".gguf";
#ifndef _WIN32
    mkdir(out_dir.c_str(), 0755);
#endif

    // --file: replay one input, no mutation. This is how a regression case runs.
    if (!replay.empty()) {
        for (const std::string & p : replay) {
            std::vector<uint8_t> b;
            if (!read_file(p, b)) { fprintf(stderr, "cannot read %s\n", p.c_str()); return 2; }
            printf("replay %s (%zu bytes)\n", p.c_str(), b.size());
            fflush(stdout);
            run_one(b.data(), b.size());
        }
        remove(g_tmp_path.c_str());
        return 0;
    }

    // load the seed corpus
    std::vector<std::vector<uint8_t>> corpus;
#ifndef _WIN32
    if (DIR * d = opendir(corpus_dir.c_str())) {
        while (struct dirent * e = readdir(d)) {
            const std::string name = e->d_name;
            if (name.size() < 5 || name.compare(name.size() - 5, 5, ".gguf") != 0) continue;
            std::vector<uint8_t> b;
            if (read_file(corpus_dir + "/" + name, b) && b.size() <= max_len) corpus.push_back(std::move(b));
        }
        closedir(d);
    }
#endif
    if (corpus.empty()) {
        fprintf(stderr, "no seed corpus in %s — run scripts/make_fuzz_corpus.py first\n", corpus_dir.c_str());
        return 2;
    }
    if (seconds <= 0 && runs <= 0) runs = 10000;
    g_state = seed ? seed : 0x6a6570612e637070ull;

    printf("fuzz-gguf-load: %zu seeds from %s, seed %llu, %s%s\n", corpus.size(), corpus_dir.c_str(),
           (unsigned long long) seed, fork_mode ? "fork mode" : "in-process",
           g_load_only ? ", load only" : "");
    fflush(stdout);

    const double t0 = now_s();
    long long execs = 0, crashes = 0;
    std::vector<uint8_t> buf;
    while ((runs <= 0 || execs < runs) && (seconds <= 0 || now_s() - t0 < seconds)) {
        buf = corpus[next_below(corpus.size())];
        const int n_mut = 1 + (int) (next_u64() % 6);
        for (int k = 0; k < n_mut; k++) mutate(buf, corpus, max_len);
        execs++;
#ifndef _WIN32
        if (fork_mode) {
            const std::string cur = out_dir + "/input-" + std::to_string((long long) getpid()) + ".gguf";
            write_file(cur, buf);
            fflush(stdout);
            const pid_t pid = fork();
            if (pid == 0) {
                // An input that never finishes is as much a finding as one that crashes: an
                // unchecked n_layer alone is an unbounded loop. SIGALRM turns it into one.
                if (timeout_s) alarm(timeout_s);
                run_one(buf.data(), buf.size());
                // exit(), not _exit(): under ASAN this is what runs LeakSanitizer, so a leaked
                // allocation on a rejected file is a non-zero exit and lands in the findings.
                exit(0);
            }
            int st = 0;
            waitpid(pid, &st, 0);
            const bool bad = !WIFEXITED(st) || WEXITSTATUS(st) != 0;
            if (bad) {
                const int sig = WIFSIGNALED(st) ? WTERMSIG(st) : 0;
                char name[256];
                snprintf(name, sizeof(name), "%s/crash-%04lld-sig%d-exit%d.gguf", out_dir.c_str(),
                         crashes, sig, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
                if (crashes < 64) write_file(name, buf);   // one class repeats; 64 samples is plenty
                printf("CRASH %s after %lld execs (signal %d%s)\n", name, execs, sig,
                       sig == SIGALRM ? " = hang" : "");
                fflush(stdout);
                crashes++;
            }
            remove(cur.c_str());
            continue;
        }
#endif
        run_one(buf.data(), buf.size());
    }
    const double dt = now_s() - t0;
    printf("done: %lld execs in %.1f s (%.0f exec/s), %lld crashes\n", execs, dt, execs / (dt > 0 ? dt : 1), crashes);
    remove(g_tmp_path.c_str());
    return crashes == 0 ? 0 : 1;
}
#endif  // JEPA_FUZZ_LIBFUZZER
