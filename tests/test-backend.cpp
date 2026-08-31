// test-backend: the GPU backend's two safety properties.
//
//   test-backend MODEL.gguf [--gpu N] [--threads N]
//
//   1. **Graph validation fails loudly.** docs/gpu-notes.md §5.4 established that a single CUDA
//      backend performs no supports_op check of its own: ggml_backend_cuda_graph_compute dispatches
//      every node and ggml_cuda_op_roll happily reads a *strided* view as if it were contiguous,
//      producing no error, no warning and a wrong answer whose cosine (0.99996) would pass this
//      project's own f16 parity gate. jepa_graph_validate() is what turns that into a refusal, so
//      this test builds exactly that node — ggml_roll on a view into a fused qkv tensor, the shape
//      src/rope3d.cpp used to emit — and requires jepa_graph_alloc() to return false. The positive
//      control is the same roll on a contiguous tensor, which must be accepted and must compute.
//   2. **The GPU encoder agrees with the CPU one.** Both backends run the same model on the same
//      input and the per-token cosine / rel_max are reported against the CPU as reference. This is
//      a backend-difference measurement, not a PyTorch-parity one — that is test-parity --gpu.
//
// Exits 0 with a "skipped" line when the build has no GPU backend or no GPU is present, so it can
// live in the normal ctest set.
#include "jepa.h"
#include "jepa-internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

static void report(const char * what, bool ok, const char * note = "") {
    printf("%-4s %s%s%s\n", ok ? "PASS" : "FAIL", what, *note ? " — " : "", note);
    if (!ok) g_fail++;
}

// A [D, H, N] view into a fake fused [3D, H, N] qkv projection: nb[2] = 3*D*H*4 while
// ne[0]*ne[1]*4 = D*H*4, so ggml_is_contiguous() is false. This is what jepa_build_qkv hands the
// RoPE hook, and what the CUDA roll kernel cannot read.
static bool build_roll_graph(jepa_context * ctx, bool contiguous) {
    const int64_t D = 64, H = 4, N = 32;
    jepa_graph_begin(ctx, 64);
    ggml_context * g = ctx->ctx_g;
    ggml_tensor * base = ggml_new_tensor_3d(g, GGML_TYPE_F32, contiguous ? D : 3 * D, H, N);
    ggml_set_input(base);
    ggml_tensor * x = contiguous
        ? base
        : ggml_view_3d(g, base, D, H, N, base->nb[1], base->nb[2], D * sizeof(float));
    ggml_tensor * y = ggml_roll(g, x, 1, 0, 0, 0);
    ggml_set_output(y);
    ggml_build_forward_expand(ctx->gf, y);
    return jepa_graph_alloc(ctx);
}

// One encoder forward of `model` on a deterministic synthetic image, returned as [n_tokens, D].
static bool encode_once(jepa_model * model, jepa_context_params cp, std::vector<float> & out, int64_t & rows, int64_t & dim) {
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) return false;
    const int C = 3, S = jepa_model_img_size(model);
    std::vector<float> px((size_t) C * S * S);
    uint32_t s = 7u;
    for (auto & v : px) { s = s * 1664525u + 1013904223u; v = (float) (s >> 8) / (float) (1u << 24) * 2.0f - 1.0f; }
    jepa_input in = { px.data(), 1, C, 1, S, S };
    jepa_output enc = { nullptr, 0, 0 };
    const bool ok = jepa_encode(ctx, &in, &enc) == 0 && enc.data;
    if (ok) {
        rows = enc.n_tokens; dim = enc.dim;
        out.assign(enc.data, enc.data + (size_t) rows * dim);
    }
    if (enc.data) jepa_free(enc.data);
    jepa_context_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--gpu N] [--threads N]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    int device = 0, threads = 8;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') device = atoi(argv[++i]);
        else if (a == "--gpu") device = 0;
        else if ((a == "--threads" || a == "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }

    if (jepa_device_count() <= device) {
        printf("SKIP no GPU device %d (this build has %d) — nothing to check\n", device, jepa_device_count());
        return 0;
    }
    printf("device %d: %s — %s\n", device, jepa_device_name(device), jepa_device_description(device));

    jepa_context_params cp = jepa_context_default_params();
    cp.n_threads = threads;

    jepa_model_params mp = jepa_model_default_params();
    mp.device = device;
    jepa_model * gpu_model = jepa_model_load_ex(model_path.c_str(), &mp);
    if (!gpu_model) { fprintf(stderr, "cannot load %s on device %d\n", model_path.c_str(), device); return 2; }

    // --- 1. the validation walk -----------------------------------------------------------------
    {
        jepa_context * ctx = jepa_context_new(gpu_model, cp);
        if (!ctx) return 2;
        printf("\n-- graph validation (the check docs/gpu-notes.md §5.4 made mandatory)\n");
        printf("   the next lines are the expected refusal, not a failure:\n");
        const bool strided_ok = build_roll_graph(ctx, /*contiguous =*/ false);
        report("ggml_roll on a strided qkv view is REFUSED", !strided_ok,
               strided_ok ? "it was accepted — the backend would compute a silently wrong answer" : "");
        const bool contig_ok = build_roll_graph(ctx, /*contiguous =*/ true);
        report("ggml_roll on a contiguous tensor is accepted", contig_ok,
               contig_ok ? "" : "the positive control failed, so the walk rejects too much");
        if (contig_ok) report("...and computes", jepa_graph_compute(ctx) == 0);
        jepa_context_free(ctx);
    }

    // --- 2. GPU vs CPU on the same model --------------------------------------------------------
    printf("\n-- encoder: %s on %s vs the CPU\n", jepa_model_name(gpu_model), jepa_device_name(device));
    std::vector<float> ygpu, ycpu;
    int64_t rg = 0, dg = 0, rc = 0, dc = 0;
    const bool gpu_ok = encode_once(gpu_model, cp, ygpu, rg, dg);
    report("encoder runs on the GPU", gpu_ok);

    jepa_model_params cpu_mp = jepa_model_default_params();
    cpu_mp.device = -1;
    jepa_model * cpu_model = jepa_model_load_ex(model_path.c_str(), &cpu_mp);
    const bool cpu_ok = cpu_model && encode_once(cpu_model, cp, ycpu, rc, dc);
    report("encoder runs on the CPU", cpu_ok);

    if (gpu_ok && cpu_ok) {
        if (rg != rc || dg != dc) {
            report("same output shape", false, "the two backends disagree on the token count");
        } else {
            double cos_min = 1.0, cos_sum = 0, max_abs = 0, ref_max = 0;
            for (int64_t r = 0; r < rc; r++) {
                double dot = 0, na = 0, nb = 0;
                for (int64_t d = 0; d < dc; d++) {
                    const double a = ygpu[(size_t) r * dc + d], b = ycpu[(size_t) r * dc + d];
                    dot += a * b; na += a * a; nb += b * b;
                    max_abs = std::fmax(max_abs, std::fabs(a - b));
                    ref_max = std::fmax(ref_max, std::fabs(b));
                }
                const double c = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
                cos_sum += c;
                cos_min = std::fmin(cos_min, c);
            }
            const double cos_mean = cos_sum / (double) rc, rel = max_abs / (ref_max + 1e-30);
            char note[192];
            snprintf(note, sizeof(note), "%lld rows: cos_mean %.7f, cos_min %.7f, max|d| %.3e, rel %.3e",
                     (long long) rc, cos_mean, cos_min, max_abs, rel);
            // A loose bar on purpose: this measures the *backend* difference (TF32 mul_mat, F16 PV
            // accumulation, one-pass norm), which docs/gpu-notes.md §6.4 sizes at 1e-2 - 1e-1 on a
            // token map. test-parity --gpu is what gates it per family and dtype; here the job is
            // to catch a broken graph, which collapses the cosine outright.
            report("GPU and CPU agree to backend round-off (cos_mean >= 0.999)", cos_mean >= 0.999, note);
        }
    }

    if (cpu_model) jepa_model_free(cpu_model);
    jepa_model_free(gpu_model);
    printf("\n%s\n", g_fail == 0 ? "all checks passed" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
