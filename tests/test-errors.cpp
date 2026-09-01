// test-errors — the failure paths of the public API. Every case here asserts three things at once:
// the call returns an error (nullptr / non-zero), jepa_error_text() says why, and the process is
// still standing afterwards. Run under ASAN and UBSAN it also asserts there is no leak and no
// undefined behaviour on the way out.
//
//   test-errors [--model MODEL.gguf ...] [-v]
//
// The first block forges its own GGUFs in a temp directory (a 2-layer, 8-dimension hfvit that the
// loader really does accept, then one field of it broken per case), so it needs no weights and runs
// in every CI job. --model adds the cases that need real weights: the graph-memory budget, the
// oversized and NULL inference inputs, the image pipeline.
//
// What is deliberately NOT here: calling jepa_encode on a context that was already freed. That is a
// use-after-free, undefined for this API exactly as it is for free(), and a test asserting anything
// about it would be asserting about undefined behaviour. The defined form of "after close" — a NULL
// handle — is case `null-handles` below, and include/jepa.h says which is which.
#include "jepa.h"
#include "forge-gguf.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

static int  g_fail = 0, g_pass = 0;
static bool g_verbose = false;
static std::string g_tmpdir;
static std::vector<std::string> g_written;   // removed on the way out

// --------------------------------------------------------------------------------------------
// harness
// --------------------------------------------------------------------------------------------
// Every case runs jepa_error_reset() first, so `jepa_error_text()` afterwards holds exactly the
// lines this call produced. A refusal with an empty message is a failure: a binding with no stderr
// has nothing else to report (include/jepa.h "diagnostics capture").
static void check(const char * name, bool refused, bool want_message = true) {
    const char * msg = jepa_error_text();
    const bool has_msg = msg && msg[0];
    if (!refused) {
        printf("FAIL %-34s the call did not report an error\n", name);
        g_fail++;
        return;
    }
    if (want_message && !has_msg) {
        printf("FAIL %-34s refused, but jepa_error_text() is empty\n", name);
        g_fail++;
        return;
    }
    g_pass++;
    if (g_verbose) {
        std::string first(msg ? msg : "");
        const size_t nl = first.find('\n');
        if (nl != std::string::npos) first.resize(nl);
        printf("ok   %-34s %s\n", name, first.c_str());
    }
}

// Same, but the message has to name `needle`. A refusal is not automatically the RIGHT refusal:
// a too-large image on a learned-position model is rejected for its token grid long before any
// budget is consulted, and a case that cannot tell those apart proves nothing.
static void check_msg(const char * name, bool refused, const char * needle) {
    if (!refused) { printf("FAIL %-34s the call did not report an error\n", name); g_fail++; return; }
    const char * msg = jepa_error_text();
    if (!msg || !strstr(msg, needle)) {
        printf("FAIL %-34s refused, but not for the expected reason (no '%s' in: %.200s)\n",
               name, needle, msg ? msg : "");
        g_fail++;
        return;
    }
    g_pass++;
    if (g_verbose) printf("ok   %-34s names '%s'\n", name, needle);
}

static int g_skip = 0;
static void skip(const char * name, const char * why) {
    g_skip++;
    if (g_verbose) printf("skip %-34s %s\n", name, why);
}

static void check_ok(const char * name, bool succeeded) {
    if (succeeded) { g_pass++; if (g_verbose) printf("ok   %-34s (accepted, as it must be)\n", name); return; }
    printf("FAIL %-34s a valid input was refused: %s\n", name, jepa_error_text());
    g_fail++;
}

static std::string tmp(const char * leaf) {
    const std::string p = g_tmpdir + "/" + leaf;
    g_written.push_back(p);
    return p;
}

static void write_bytes(const std::string & path, const void * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
    if (n) fwrite(data, 1, n, f);
    fclose(f);
}

static std::vector<uint8_t> read_bytes(const std::string & path) {
    std::vector<uint8_t> v;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n > 0) { v.resize((size_t) n); if (fread(v.data(), 1, v.size(), f) != v.size()) v.clear(); }
    fclose(f);
    return v;
}

// Overwrite one integer / float metadata value in an already-written GGUF. Some of the values worth
// testing cannot be forged — a file with embed_dim 0 has no tensors to write — and patching the
// bytes is also the more faithful shape of the attack: metadata that no longer describes the data.
// GGUF KV layout is <u64 key length><key><u32 type><value>, so the value sits 4 bytes past the key.
static bool patch_kv(std::vector<uint8_t> & b, const char * key, const void * value, size_t vsize) {
    const size_t klen = strlen(key);
    for (size_t i = 8; i + klen + 4 + vsize <= b.size(); i++) {
        if (memcmp(b.data() + i, key, klen) != 0) continue;
        uint64_t n = 0;
        memcpy(&n, b.data() + i - 8, 8);
        if (n != klen) continue;                 // a substring of another key, not the key itself
        memcpy(b.data() + i + klen + 4, value, vsize);
        return true;
    }
    return false;
}

static void patched_case(const char * name, const std::string & src, const char * leaf,
                         const char * key, int32_t value) {
    std::vector<uint8_t> b = read_bytes(src);
    if (!patch_kv(b, key, &value, sizeof value)) {
        printf("FAIL %-34s could not find %s in the forged file\n", name, key);
        g_fail++;
        return;
    }
    const std::string p = tmp(leaf);
    write_bytes(p, b.data(), b.size());
    jepa_error_reset();
    jepa_model * m = jepa_model_load(p.c_str(), false);
    check(name, m == nullptr);
    jepa_model_free(m);
}

// Load `path` and expect a refusal (or, with want_ok, a successful load that is freed again).
static void load_case(const char * name, const std::string & path, bool want_ok = false) {
    jepa_error_reset();
    jepa_model * m = jepa_model_load(path.c_str(), false);
    if (want_ok) { check_ok(name, m != nullptr); }
    else         { check(name, m == nullptr); }
    jepa_model_free(m);
}

// --------------------------------------------------------------------------------------------
// 1. loader
// --------------------------------------------------------------------------------------------
static void test_loader() {
    const std::string good = tmp("good.gguf");
    forge(good, forge_opts{});
    load_case("load/valid-forged", good, true);

    // a path that is not there, and one that is a directory
    jepa_error_reset();
    check("load/missing-file", jepa_model_load(tmp("no-such-file.gguf").c_str(), false) == nullptr);
    jepa_error_reset();
    check("load/directory", jepa_model_load(g_tmpdir.c_str(), false) == nullptr);
    jepa_error_reset();
    check("load/null-path", jepa_model_load(nullptr, false) == nullptr);

    // empty, magic-only, wrong magic, a version from the future
    write_bytes(tmp("empty.gguf"), "", 0);
    load_case("load/empty-file", tmp("empty.gguf"));
    write_bytes(tmp("magic.gguf"), "GGUF", 4);
    load_case("load/magic-only", tmp("magic.gguf"));
    {
        std::vector<uint8_t> b = read_bytes(good);
        b[0] = 'X';
        write_bytes(tmp("badmagic.gguf"), b.data(), b.size());
        load_case("load/wrong-magic", tmp("badmagic.gguf"));
        b = read_bytes(good);
        b[4] = 99;                       // version
        write_bytes(tmp("badver.gguf"), b.data(), b.size());
        load_case("load/future-version", tmp("badver.gguf"));
    }

    // Truncation. Two regions, and the split is deliberate:
    //
    //   the fixed header      — 4/8/16/24 bytes in, where the magic, version and the two counts are
    //   the data section      — every tensor the metadata promises is now past the end of the file,
    //                           which is the class jepa_model_load_ex's own length check owns
    //
    // Truncation INSIDE the key-value / tensor-info block is ggml's parser and it rejects it
    // correctly, but at this submodule pin it reads the type tag into an uninitialised `gguf_type`
    // on its own failure path (ggml/src/gguf.cpp:575), which UBSan reports as an invalid enum load.
    // That is upstream's to fix and it would make this suite fail the sanitizer job for a bug it
    // does not have, so those offsets are left to tests/fuzz/fuzz-gguf-load.cpp, which runs UBSan
    // in recoverable mode and grades reports by which source file they name.
    {
        const std::vector<uint8_t> b = read_bytes(good);
        size_t data_offset = 0;
        {
            gguf_init_params ip = { /*no_alloc*/ true, nullptr };
            gguf_context * gg = gguf_init_from_file(good.c_str(), ip);
            if (gg) { data_offset = gguf_get_data_offset(gg); gguf_free(gg); }
        }
        for (size_t n : { (size_t) 4, (size_t) 8, (size_t) 16, (size_t) 24 }) {
            const std::string p = tmp(("trunc-h" + std::to_string(n) + ".gguf").c_str());
            write_bytes(p, b.data(), n);
            load_case(("load/truncated-header-" + std::to_string(n) + "B").c_str(), p);
        }
        if (data_offset > 0 && data_offset < b.size()) {
            const size_t data_len = b.size() - data_offset;
            for (int k = 0; k < 8; k++) {
                const size_t n = data_offset + data_len * (size_t) k / 8;
                const std::string p = tmp(("trunc-d" + std::to_string(k) + ".gguf").c_str());
                write_bytes(p, b.data(), n);
                load_case(("load/truncated-data-" + std::to_string(k * 100 / 8) + "%").c_str(), p);
            }
        } else {
            printf("FAIL load/truncated-data could not find the data offset of the forged file\n");
            g_fail++;
        }
    }

    forge_opts o;
    o.family = "not_a_family";  forge(tmp("f1.gguf"), o); load_case("load/unknown-family", tmp("f1.gguf"));
    o = forge_opts{}; o.family = "";
    forge(tmp("f2.gguf"), o); load_case("load/empty-family", tmp("f2.gguf"));
    o = forge_opts{}; o.arch = "llama";
    forge(tmp("f3.gguf"), o); load_case("load/wrong-architecture", tmp("f3.gguf"));
    o = forge_opts{}; o.attn_mode = "causal";
    forge(tmp("f4.gguf"), o); load_case("load/unknown-attn-mode", tmp("f4.gguf"));
    // block_causal is a real mode, but only the video families build the mask; on an image family
    // it would silently run unmasked, which is the one thing the key exists to prevent.
    o = forge_opts{}; o.attn_mode = "block_causal";
    forge(tmp("f5.gguf"), o); load_case("load/block-causal-on-image-family", tmp("f5.gguf"));
    o = forge_opts{}; o.act = "relu6";
    forge(tmp("f6.gguf"), o); load_case("load/unknown-activation", tmp("f6.gguf"));
    // 3-D RoPE rotates pairs; an odd head width used to load and then abort inside
    // jepa_rope3d_apply's GGML_ASSERT(D % 2 == 0) on the first encode.
    o = forge_opts{}; o.family = "vjepa2"; o.embed_dim = 6; o.n_head = 2;   // head_dim 3
    forge(tmp("f7.gguf"), o); load_case("load/rope3d-odd-head-dim", tmp("f7.gguf"));
    // ...and the same shape with an even head width still loads, so the check is not just "vjepa2"
    o = forge_opts{}; o.family = "vjepa2"; o.embed_dim = 8; o.n_head = 2;   // head_dim 4
    forge(tmp("f8.gguf"), o); load_case("load/rope3d-even-head-dim-ok", tmp("f8.gguf"), true);

    // Dimensions: zero, negative, and past the loader's ceiling. All patched into the bytes of the
    // good file, so each case changes exactly one number and nothing else.
    patched_case("load/n_head-zero",       good, "d1.gguf", "jepa.enc.n_head", 0);   // used to SIGFPE
    patched_case("load/n_head-negative",   good, "d2.gguf", "jepa.enc.n_head", -2);
    patched_case("load/n_head-INT32_MAX",  good, "d3.gguf", "jepa.enc.n_head", 2147483647);
    patched_case("load/embed_dim-zero",    good, "d4.gguf", "jepa.enc.embed_dim", 0);
    patched_case("load/embed_dim-negative", good, "d5.gguf", "jepa.enc.embed_dim", -8);
    patched_case("load/embed_dim-not-divisible", good, "d6.gguf", "jepa.enc.embed_dim", 9);
    patched_case("load/n_layer-zero",      good, "d7.gguf", "jepa.enc.n_layer", 0);
    patched_case("load/n_layer-negative",  good, "d8.gguf", "jepa.enc.n_layer", -1);
    patched_case("load/n_layer-INT32_MAX", good, "d9.gguf", "jepa.enc.n_layer", 2147483647);
    patched_case("load/ffn_dim-INT32_MAX", good, "e1.gguf", "jepa.enc.ffn_dim", 2147483647);
    patched_case("load/patch_size-zero",   good, "e2.gguf", "jepa.enc.patch_size", 0);
    patched_case("load/patch_size-negative", good, "e3.gguf", "jepa.enc.patch_size", -3);
    patched_case("load/img_size-INT32_MAX", good, "e4.gguf", "jepa.enc.img_size", 2147483647);
    patched_case("load/in_chans-huge",     good, "e5.gguf", "jepa.enc.in_chans", 1 << 30);
    patched_case("load/tubelet-huge",      good, "e6.gguf", "jepa.enc.tubelet_size", 1 << 30);
    patched_case("load/n_frames-negative", good, "e7.gguf", "jepa.enc.n_frames", -1);
    patched_case("load/n_registers-huge",  good, "e8.gguf", "jepa.enc.n_registers", 2147483647);

    o = forge_opts{}; o.ln_eps = NAN;
    forge(tmp("d10.gguf"), o); load_case("load/ln_eps-nan", tmp("d10.gguf"));
    o = forge_opts{}; o.pre_std0 = 0.0f;
    forge(tmp("d11.gguf"), o); load_case("load/pre-std-zero", tmp("d11.gguf"));
    o = forge_opts{}; o.pre_resize_short = 0;
    forge(tmp("d12.gguf"), o); load_case("load/pre-resize-zero", tmp("d12.gguf"));

    // tensors: a shape the hparams do not describe, a missing tensor a block needs, a type whose
    // block size does not divide the row, and a promised tensor the file is too short to hold
    o = forge_opts{}; o.extra_patch_in = 1;
    forge(tmp("t1.gguf"), o); load_case("load/shape-mismatch", tmp("t1.gguf"));
    o = forge_opts{}; o.drop_ln1 = true;
    forge(tmp("t2.gguf"), o); load_case("load/incomplete-block", tmp("t2.gguf"));
    o = forge_opts{}; o.drop_cls_tensor = true;
    forge(tmp("t3.gguf"), o); load_case("load/cls-token-missing", tmp("t3.gguf"));
    o = forge_opts{}; o.quantize_patch_embed = true;    // Q4_0 rows of 12, block size 32
    forge(tmp("t4.gguf"), o); load_case("load/unsupported-tensor-type", tmp("t4.gguf"));
    // A norm weight, a bias or a layer-scale vector of the wrong length reaches ggml_mul / ggml_add
    // as a right-hand side that cannot broadcast — a GGML_ASSERT abort inside the graph builder,
    // found by the fuzzer, now a load error.
    o = forge_opts{}; o.ln1_len = 7;
    forge(tmp("t6.gguf"), o); load_case("load/ln-weight-wrong-length", tmp("t6.gguf"));
    o = forge_opts{}; o.qkv_bias_len = 7;
    forge(tmp("t7.gguf"), o); load_case("load/qkv-bias-wrong-length", tmp("t7.gguf"));
    o = forge_opts{}; o.add_layer_scale = true;
    forge(tmp("t8.gguf"), o); load_case("load/layer-scale-wrong-length", tmp("t8.gguf"));
    // An f16 CLS token in an f32 graph is a type mismatch inside ggml_concat; an i32 matmul weight
    // is a call through a null vec_dot pointer. Both were fuzzer finds, both are load errors now.
    o = forge_opts{}; o.f16_cls_token = true;
    forge(tmp("t9.gguf"), o); load_case("load/f16-operand-in-f32-graph", tmp("t9.gguf"));
    o = forge_opts{}; o.i32_ffn_up = true;
    forge(tmp("t10.gguf"), o); load_case("load/integer-matmul-weight", tmp("t10.gguf"));
    o = forge_opts{}; o.n_registers = 4;                // reg_tokens exist but the graph needs a match
    forge(tmp("t5.gguf"), o); load_case("load/registers-vs-pos-table", tmp("t5.gguf"), true);

    // a predictor and a head the metadata promises and the tensors do not deliver: these used to be
    // an abort() inside m->require() on the first jepa_predict / jepa_head call
    o = forge_opts{}; o.pred_kind = "masked";
    forge(tmp("p1.gguf"), o); load_case("load/masked-predictor-without-tensors", tmp("p1.gguf"));
    o = forge_opts{}; o.pred_kind = "lewm";
    forge(tmp("p2.gguf"), o); load_case("load/lewm-predictor-without-tensors", tmp("p2.gguf"));
    o = forge_opts{}; o.head_kind = "attentive_pool";
    forge(tmp("p3.gguf"), o); load_case("load/head-without-tensors", tmp("p3.gguf"));
}

// --------------------------------------------------------------------------------------------
// 2. the API's own guards — NULL handles, empty and impossible shapes
// --------------------------------------------------------------------------------------------
static void test_api_guards(jepa_model * m, jepa_context * ctx) {
    // Freeing a null handle is a no-op, twice over. (The other half of the "after close" contract —
    // reusing a handle that WAS freed — is undefined and is not tested; see the file header.)
    jepa_model_free(nullptr);
    jepa_context_free(nullptr);
    jepa_free(nullptr);

    jepa_output out = {};
    jepa_input in = { nullptr, 1, 3, 1, 8, 8 };
    jepa_error_reset();
    check("null-handles/encode-null-ctx", jepa_encode(nullptr, &in, &out) != 0, false);
    check("null-handles/encode-null-in", jepa_encode(ctx, nullptr, &out) != 0, false);
    check("null-handles/encode-null-out", jepa_encode(ctx, &in, nullptr) != 0, false);
    check("null-handles/encode-null-data", jepa_encode(ctx, &in, &out) != 0, false);
    check("null-handles/pool-mean-null", jepa_pool_mean(nullptr, &out, &out) != 0, false);
    check("null-handles/pool-cls-null", jepa_pool_cls(m, nullptr, &out) != 0, false);
    check("null-handles/head-null", jepa_head(nullptr, &out, &out) != 0, false);
    check("null-handles/predict-null", jepa_predict(nullptr, &out, nullptr, 1, nullptr, 1, &out) != 0, false);
    check("null-handles/lewm-predict-null", jepa_lewm_predict(nullptr, nullptr, nullptr, 1, &out) != 0, false);
    check("null-handles/lewm-rollout-null", jepa_lewm_rollout(nullptr, nullptr, 1, nullptr, 1, nullptr) != 0, false);
    check("null-handles/context-new-null", jepa_context_new(nullptr, jepa_context_default_params()) == nullptr, false);

    // shapes with a zero or negative extent
    std::vector<float> px(3 * 8 * 8, 0.1f);
    const int C = jepa_model_embed_dim(m) > 0 ? 3 : 3;
    struct { const char * name; jepa_input in; } bad[] = {
        { "shape/batch-0",    { px.data(), 0, C, 1, 8, 8 } },
        { "shape/batch-neg",  { px.data(), -1, C, 1, 8, 8 } },
        { "shape/frames-0",   { px.data(), 1, C, 0, 8, 8 } },
        { "shape/height-0",   { px.data(), 1, C, 1, 0, 8 } },
        { "shape/width-neg",  { px.data(), 1, C, 1, 8, -8 } },
        { "shape/chans-wrong",{ px.data(), 1, 1, 1, 8, 8 } },
        // the products of these overflow int64 / size_t, which is the whole point of the guard
        { "shape/absurd-hw",  { px.data(), 1, C, 1, 1 << 30, 1 << 30 } },
        { "shape/absurd-batch", { px.data(), 1 << 30, C, 1 << 30, 1 << 20, 1 << 20 } },
    };
    for (auto & b : bad) {
        jepa_error_reset();
        jepa_output o = {};
        check(b.name, jepa_encode(ctx, &b.in, &o) != 0);
        jepa_free(o.data);
    }

    // pooling on an output header that describes nothing
    jepa_output empty = { nullptr, 0, 0 };
    jepa_error_reset();
    check("pool/mean-empty", jepa_pool_mean(m, &empty, &out) != 0, false);
    std::vector<float> rows(16, 1.0f);
    jepa_output negdim = { rows.data(), 4, -8 };
    jepa_error_reset();
    check("pool/mean-negative-dim", jepa_pool_mean(m, &negdim, &out) != 0);
    jepa_error_reset();
    check("pool/cls-negative-dim", jepa_pool_cls(m, &negdim, &out) != 0);

    // device selection
    jepa_model_params mp = jepa_model_default_params();
    mp.device = 99999;
    jepa_error_reset();
    check("device/out-of-range", jepa_model_load_ex("models/gguf/nothing.gguf", &mp) == nullptr);
    jepa_error_reset();
    check("device/name-out-of-range", jepa_device_name(99999) == nullptr, false);
    check_ok("device/cpu-name", jepa_device_name(-1) != nullptr);

    // batch: 0 restores the default rather than building an empty graph
    const int before = jepa_context_max_batch(ctx);
    jepa_context_set_max_batch(ctx, 0);
    check_ok("batch/zero-restores-default", jepa_context_max_batch(ctx) > 0);
    jepa_context_set_max_batch(ctx, -5);
    check_ok("batch/negative-restores-default", jepa_context_max_batch(ctx) > 0);
    jepa_context_set_max_batch(ctx, before);
}

// --------------------------------------------------------------------------------------------
// 3. the graph-memory budget ($JEPA_MAX_GRAPH_MIB)
// --------------------------------------------------------------------------------------------
// The budget is read when the context is created, so each case gets its own context.
static void test_budget(jepa_model * m) {
    if (setenv("JEPA_MAX_GRAPH_MIB", "1", 1) != 0) return;
    jepa_context_params cp = jepa_context_default_params();
    cp.n_threads = 2;
    jepa_context * tight = jepa_context_new(m, cp);
    if (!tight) { printf("FAIL budget: no context\n"); g_fail++; unsetenv("JEPA_MAX_GRAPH_MIB"); return; }

    // The encoder at the model's own crop. A learned-position model accepts exactly one input size,
    // so the way to make the budget bite is a small budget, not a big image; whether 1 MiB is small
    // enough depends on the model, and a model this cannot squeeze is skipped rather than faked.
    const int S = jepa_model_img_size(m);
    const int T = jepa_model_n_frames(m) > 0 ? jepa_model_n_frames(m) : 1;
    if (S > 0 && S <= 1024) {
        std::vector<float> px((size_t) 3 * T * S * S, 0.1f);
        jepa_input in = { px.data(), 1, 3, T, S, S };
        jepa_output out = {};
        jepa_error_reset();
        const int rc = jepa_encode(tight, &in, &out);
        if (rc == 0) skip("budget/oversized-image-refused", "this model fits one item in 1 MiB");
        else         check_msg("budget/oversized-image-refused", true, "JEPA_MAX_GRAPH_MIB");
        jepa_free(out.data);
    }
    // A clip that is absurd rather than merely large. The video families accept any multiple of the
    // tubelet, so n_frames is purely the caller's number and the budget is the only thing between it
    // and the allocator. The pixel buffer here is sized for ONE frame: jepa_encode_video refuses on
    // the budget before it reads a pixel, and if that ever stops being true ASAN says so here.
    if (T > 1 && S > 0) {
        std::vector<float> one((size_t) 3 * S * S, 0.1f);
        jepa_input in = { one.data(), 1, 3, T * 64, S, S };
        jepa_output out = {};
        jepa_error_reset();
        check_msg("budget/absurd-n_frames", jepa_encode(tight, &in, &out) != 0, "JEPA_MAX_GRAPH_MIB");
        jepa_free(out.data);
    }
    // The masked predictor: n_target is the caller's and used to size the graph unchecked.
    if (jepa_model_has_predictor(m)) {
        std::vector<float> rows((size_t) jepa_model_embed_dim(m) * 8, 0.1f);
        jepa_output enc = { rows.data(), 8, jepa_model_embed_dim(m) };
        std::vector<int32_t> ids(1 << 20);
        for (size_t i = 0; i < ids.size(); i++) ids[i] = (int32_t) (i % 8);
        jepa_output out = {};
        jepa_error_reset();
        const bool refused = jepa_predict(tight, &enc, ids.data(), 4, ids.data(), (int) ids.size(), &out) != 0;
        if (strstr(jepa_error_text(), "not the masked predictor")) {
            skip("budget/absurd-target-count", "this model's predictor is not the masked one");
        } else {
            check_msg("budget/absurd-target-count", refused, "JEPA_MAX_GRAPH_MIB");
        }
        jepa_free(out.data);
    }
    // The LeWM rollout: n_seed + n_steps used to be summed in int and then allocated.
    if (jepa_lewm_n_frames(m) > 0) {
        const int D = jepa_model_embed_dim(m), A = jepa_lewm_action_dim(m);
        std::vector<float> embs((size_t) D, 0.1f), acts((size_t) (A > 0 ? A : 1), 0.2f);
        // `out` is sized for one step only. The budget refusal happens before the rollout loop
        // writes anything, so this is safe — and if that ever stops being true, ASAN says so here.
        std::vector<float> out((size_t) D, 0.0f);
        jepa_error_reset();
        check_msg("budget/absurd-rollout-length",
                  jepa_lewm_rollout(tight, embs.data(), 1, acts.data(), 1 << 24, out.data()) != 0,
                  "JEPA_MAX_GRAPH_MIB");
    }
    jepa_context_free(tight);
    unsetenv("JEPA_MAX_GRAPH_MIB");
}

// --------------------------------------------------------------------------------------------
// 4. the image pipeline on bytes nobody validated
// --------------------------------------------------------------------------------------------
static void test_images(jepa_model * m) {
    int h = 0, w = 0;
    // decode failures: a file that is not there, an empty file, a JPEG header over noise, a PNG
    // header over noise, and a PNG whose IHDR says it is 20000x20000
    jepa_error_reset();
    check("image/missing-file", jepa_load_image_rgb(tmp("no-such.jpg").c_str(), &h, &w) == nullptr);
    write_bytes(tmp("empty.jpg"), "", 0);
    jepa_error_reset();
    check("image/empty-file", jepa_load_image_rgb(tmp("empty.jpg").c_str(), &h, &w) == nullptr);

    std::vector<uint8_t> junk(4096);
    for (size_t i = 0; i < junk.size(); i++) junk[i] = (uint8_t) (i * 131 + 7);
    memcpy(junk.data(), "\xff\xd8\xff\xe0", 4);                   // JPEG SOI + APP0
    write_bytes(tmp("corrupt.jpg"), junk.data(), junk.size());
    jepa_error_reset();
    check("image/corrupt-jpeg", jepa_load_image_rgb(tmp("corrupt.jpg").c_str(), &h, &w) == nullptr);
    memcpy(junk.data(), "\x89PNG\r\n\x1a\n", 8);
    write_bytes(tmp("corrupt.png"), junk.data(), junk.size());
    jepa_error_reset();
    check("image/corrupt-png", jepa_load_image_rgb(tmp("corrupt.png").c_str(), &h, &w) == nullptr);

    // Raw-buffer entry points, where the caller states the geometry. 0x0 and a negative extent must
    // be refused; 1x1 must work (it is a legal, if silly, image); a 20000x1 buffer is the aspect
    // ratio that used to make the intermediate resize a 2.5 GiB allocation.
    std::vector<uint8_t> one(3, 200);
    int oh = 0, ow = 0;
    jepa_error_reset();
    check("preprocess/0x0", jepa_preprocess_image_rgb(m, one.data(), 0, 0, &oh, &ow) == nullptr, false);
    jepa_error_reset();
    check("preprocess/negative", jepa_preprocess_image_rgb(m, one.data(), -1, 8, &oh, &ow) == nullptr, false);
    jepa_error_reset();
    check("preprocess/null-buffer", jepa_preprocess_image_rgb(m, nullptr, 8, 8, &oh, &ow) == nullptr, false);
    float * px = jepa_preprocess_image_rgb(m, one.data(), 1, 1, &oh, &ow);
    check_ok("preprocess/1x1", px != nullptr && oh > 0 && ow > 0);
    jepa_free(px);

    {   // 20000x20000, actually allocated: the pipeline has to hold this inside its own bound
        const size_t n = (size_t) 20000 * 20000 * 3;
        std::vector<uint8_t> big;
        bool allocated = true;
        try { big.assign(n, 128); } catch (...) { allocated = false; }
        if (allocated) {
            jepa_error_reset();
            float * out = jepa_preprocess_image_rgb(m, big.data(), 20000, 20000, &oh, &ow);
            check_ok("preprocess/20000x20000", out != nullptr && oh == jepa_model_img_size(m));
            jepa_free(out);
        } else if (g_verbose) {
            printf("skip preprocess/20000x20000 (1.1 GiB would not allocate)\n");
        }
    }
    {   // the pathological aspect ratio, refused rather than resized into gigabytes
        std::vector<uint8_t> strip((size_t) 20000 * 3, 90);
        jepa_error_reset();
        float * out = jepa_preprocess_image_rgb(m, strip.data(), 1, 20000, &oh, &ow);
        // Either it is refused with a message, or the model's own resize_short keeps it small
        // enough to be legal. Both are correct; an unbounded allocation is not.
        if (out) { check_ok("preprocess/1x20000-bounded", oh > 0 && ow > 0); jepa_free(out); }
        else     { check("preprocess/1x20000-refused", true); }
    }

    // The exported resize on shapes it cannot do anything with: no crash, no write.
    std::vector<uint8_t> dst(3, 0);
    jepa_error_reset();
    jepa_resize_antialias_u8(one.data(), 0, 0, 3, dst.data(), 1, 1, JEPA_RESAMPLE_BILINEAR);
    check("resize/zero-input", true);
    jepa_resize_antialias_u8(one.data(), 1, 1, 3, dst.data(), 0, 0, JEPA_RESAMPLE_BILINEAR);
    check("resize/zero-output", true);
    jepa_resize_antialias_u8(nullptr, 1, 1, 3, dst.data(), 1, 1, JEPA_RESAMPLE_BILINEAR);
    check("resize/null-source", true);
}

// --------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::vector<std::string> models;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) models.push_back(argv[++i]);
        else if (a == "-v" || a == "--verbose") g_verbose = true;
        else { fprintf(stderr, "usage: %s [--model MODEL.gguf ...] [-v]\n", argv[0]); return 2; }
    }

    // A temp directory of our own, so a parallel ctest run cannot collide with us.
    {
        const char * base = getenv("TMPDIR");
        if (!base || !*base) base = getenv("TEMP");
        g_tmpdir = std::string(base && *base ? base : "/tmp") + "/jepa-test-errors-" +
                   std::to_string((long) jepa_getpid());
        if (jepa_mkdir(g_tmpdir.c_str()) != 0) {
            fprintf(stderr, "cannot create %s\n", g_tmpdir.c_str());
            return 2;
        }
    }

    // stderr stays useful, but these cases are supposed to print: silence the expected noise unless
    // -v, and read the text back through jepa_error_text() instead.
    if (!g_verbose) {
        if (!freopen(
#ifdef _WIN32
                "NUL",
#else
                "/dev/null",
#endif
                "w", stderr)) { /* keep going with stderr as it was */ }
    }

    test_loader();

    // Cases that need a real model. The forged one is fine for most of them and is what runs when
    // no --model was given, so this block runs everywhere; --model is repeatable because the budget
    // cases are per-family (a projector and a rollout on LeWM, a masked predictor on V-JEPA 2).
    if (models.empty()) models.push_back(tmp("good.gguf"));
    for (const std::string & use : models) {
        jepa_error_reset();
        jepa_model * m = jepa_model_load(use.c_str(), false);
        if (!m) {
            printf("FAIL could not load %s for the inference cases: %s\n", use.c_str(), jepa_error_text());
            g_fail++;
            continue;
        }
        jepa_context_params cp = jepa_context_default_params();
        cp.n_threads = 2;
        jepa_context * ctx = jepa_context_new(m, cp);
        if (!ctx) { printf("FAIL no context for %s\n", use.c_str()); g_fail++; }
        else {
            test_api_guards(m, ctx);
            test_images(m);
            jepa_context_free(ctx);
            test_budget(m);
        }
        jepa_model_free(m);
    }

    // clean up: the forged files, then the directory
    for (const std::string & p : g_written) remove(p.c_str());
#ifdef _WIN32
    _rmdir(g_tmpdir.c_str());
#else
    rmdir(g_tmpdir.c_str());
#endif
    if (!g_verbose) { /* stderr was redirected; stdout still carries the result line */ }
    std::string used;
    for (const std::string & p : models) used += (used.empty() ? "" : ", ") + p;
    printf("test-errors: %d passed, %d failed, %d skipped (models: %s)\n", g_pass, g_fail, g_skip, used.c_str());
    fflush(stdout);
    return g_fail == 0 ? 0 : 1;
}
