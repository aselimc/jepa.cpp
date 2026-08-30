// Flash-attention & ops study for the jepa.cpp video graph (CPU backend, ggml).
//
//   test-attn [attn] [-t N] [--hd 32,64,80] [--nh 12,16] [--n 256,2048,...] [--max-gb G] [--max-s S] [--no-views]
//       For every (head_dim, n_head, N) combo: seeded random q/k/v, then
//         ref      double-precision softmax(q k^T * scale) v computed on the host (multithreaded)
//         naive    ggml_mul_mat -> ggml_soft_max_ext(scale) -> ggml_mul_mat, all F32
//         flash16  ggml_flash_attn_ext with F16 K/V   (the production path: sequences reach 18k tokens)
//         flash32  ggml_flash_attn_ext with F32 K/V
//         fused16  the exact production recipe: strided views into a fused [3*D*H, N] qkv tensor,
//                  ggml_permute -> ggml_cast(F16) for K/V, flash16 (timing includes the casts)
//       and reports max |err| / relative error / per-row cosine against `ref`, wall time, and the
//       memory of each graph (ggml_gallocr buffer + CPU work buffer). Combos whose score matrix would
//       exceed --max-gb or whose predicted time exceeds --max-s are skipped (and listed).
//       A block-causal mask case checks the mask semantics (-INF = blocked, 0 = attend) on both CPU
//       kernels (non-tiled N < 64 and tiled N >= 64) and the fully-masked-row behaviour.
//       Exit status 1 if any flash run has rel err > 1e-2 or min row cosine < 0.9999 (or a mask
//       case fails).
//
//   test-attn bench [-t 32,96]
//       ggml_mul_mat W[4096x1024] (F32 / F16 / Q8_0) x X[1024xN] for N = 256 / 2048 / 8192 -> GFLOP/s
//       per thread count, plus ggml_norm and ggml_gelu_erf max error vs double on 1M elements.
//
// Findings are written up in docs/ggml-notes.md.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

// --- helpers ------------------------------------------------------------------------------------------
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double peak_rss_gb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double) ru.ru_maxrss / (1024.0 * 1024.0); // ru_maxrss is in KiB on Linux
}

// deterministic normal(0,1) generator (xorshift64* + Box-Muller)
struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull) {}
    double uniform() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return (double) ((s * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 9007199254740992.0);
    }
    float normal() {
        double u1 = uniform(), u2 = uniform();
        if (u1 < 1e-300) u1 = 1e-300;
        return (float) (std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
};

static void fill_normal(std::vector<float> & v, uint64_t seed, float stdev = 1.0f) {
    rng g(seed);
    for (auto & x : v) x = stdev * g.normal();
}

// error statistics of a candidate against the double reference; both laid out as [H][N][D]
struct err_stats {
    double max_abs = 0, max_ref = 0, min_cos = 1, mean_cos = 0, mean_abs = 0;
    double rel() const { return max_ref > 0 ? max_abs / max_ref : 0; }
};

static err_stats compare(const std::vector<float> & cand, const std::vector<double> & ref, int D, int64_t rows) {
    err_stats e;
    double cos_sum = 0, abs_sum = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const float  * a = cand.data() + r * D;
        const double * b = ref.data()  + r * D;
        double dot = 0, na = 0, nb = 0;
        for (int d = 0; d < D; ++d) {
            const double diff = std::fabs((double) a[d] - b[d]);
            e.max_abs = std::fmax(e.max_abs, diff);
            e.max_ref = std::fmax(e.max_ref, std::fabs(b[d]));
            abs_sum += diff;
            dot += (double) a[d] * b[d]; na += (double) a[d] * a[d]; nb += b[d] * b[d];
        }
        const double c = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : (na == 0 && nb == 0 ? 1.0 : 0.0);
        e.min_cos = std::fmin(e.min_cos, c);
        cos_sum += c;
    }
    e.mean_cos = cos_sum / (double) rows;
    e.mean_abs = abs_sum / (double) (rows * D);
    return e;
}

// --- double-precision reference ----------------------------------------------------------------------
// q, k, v: [H][N][D] floats (identical to ggml [D, N, H] contiguous). allowed: N_q x N_kv bytes or null.
static void ref_attention(const float * q, const float * k, const float * v, int D, int H, int N, float scale,
                          const uint8_t * allowed, std::vector<double> & out, int n_threads) {
    out.assign((size_t) H * N * D, 0.0);
    const int64_t rows = (int64_t) H * N;
    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t]() {
            std::vector<double> s(N), qd(D);
            for (int64_t r = t; r < rows; r += n_threads) {
                const int h = (int) (r / N), i = (int) (r % N);
                const float * qi = q + r * D;
                for (int d = 0; d < D; ++d) qd[d] = (double) qi[d] * (double) scale;
                const float * kh = k + (size_t) h * N * D;
                const float * vh = v + (size_t) h * N * D;
                double m = -INFINITY;
                for (int j = 0; j < N; ++j) {
                    if (allowed && !allowed[(size_t) i * N + j]) { s[j] = -INFINITY; continue; }
                    const float * kj = kh + (size_t) j * D;
                    double acc = 0;
                    for (int d = 0; d < D; ++d) acc += qd[d] * (double) kj[d];
                    s[j] = acc;
                    m = std::fmax(m, acc);
                }
                double * o = out.data() + r * D;
                if (m == -INFINITY) continue; // fully masked row -> 0 (matches ggml: S == 0 -> output 0)
                double sum = 0;
                for (int j = 0; j < N; ++j) {
                    if (s[j] == -INFINITY) continue;
                    const double p = std::exp(s[j] - m);
                    sum += p;
                    const float * vj = vh + (size_t) j * D;
                    for (int d = 0; d < D; ++d) o[d] += p * (double) vj[d];
                }
                const double inv = 1.0 / sum;
                for (int d = 0; d < D; ++d) o[d] *= inv;
            }
        });
    }
    for (auto & th : pool) th.join();
}

// --- ggml runs --------------------------------------------------------------------------------------------
struct run_result {
    bool     ok = true;
    double   ms = 0;            // best-of-n wall time of ggml_backend_graph_compute
    size_t   graph_bytes = 0;   // ggml_gallocr buffer (intermediates)
    size_t   work_bytes = 0;    // CPU backend work buffer (ggml_graph_plan)
    size_t   input_bytes = 0;   // q/k/v(/mask) buffer
    std::vector<float> out;     // [H][N][D]
};

struct graph_env {
    ggml_backend_t backend;
    int n_threads;
    int n_repeat;
};

// run a graph: builds inputs via `make_inputs`, the graph via `make_graph`, returns timing + memory + output
template <typename MakeInputs, typename MakeGraph>
static run_result run_graph(const graph_env & env, int64_t n_tensors, MakeInputs make_inputs, MakeGraph make_graph,
                            int D, int H, int N, bool out_is_flash_layout) {
    run_result r;
    struct ggml_init_params ip = { (size_t) n_tensors * ggml_tensor_overhead(), nullptr, true };
    struct ggml_context * ctx_in = ggml_init(ip);
    std::vector<struct ggml_tensor *> inputs = make_inputs(ctx_in);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_in, env.backend);
    GGML_ASSERT(buf);
    r.input_bytes = ggml_backend_buffer_get_size(buf);

    struct ggml_init_params gp = { 256 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx_g = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph(ctx_g);
    struct ggml_tensor * y = make_graph(ctx_g, inputs);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(env.backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
    r.graph_bytes = ggml_gallocr_get_buffer_size(ga, 0);
    r.work_bytes  = ggml_graph_plan(gf, env.n_threads, nullptr).work_size;

    // warm-up + timed runs (best of n)
    GGML_ASSERT(ggml_backend_graph_compute(env.backend, gf) == GGML_STATUS_SUCCESS);
    r.ms = 1e300;
    for (int i = 0; i < env.n_repeat; ++i) {
        const double t0 = now_ms();
        GGML_ASSERT(ggml_backend_graph_compute(env.backend, gf) == GGML_STATUS_SUCCESS);
        r.ms = std::fmin(r.ms, now_ms() - t0);
    }

    std::vector<float> raw((size_t) D * H * N);
    GGML_ASSERT(ggml_nbytes(y) == raw.size() * sizeof(float));
    ggml_backend_tensor_get(y, raw.data(), 0, ggml_nbytes(y));
    if (out_is_flash_layout) { // flash output is [D, H, N] -> [H][N][D]
        r.out.resize(raw.size());
        for (int64_t i = 0; i < N; ++i)
            for (int h = 0; h < H; ++h)
                memcpy(&r.out[((size_t) h * N + i) * D], &raw[((size_t) i * H + h) * D], D * sizeof(float));
    } else {
        r.out = std::move(raw); // already [D, N, H]
    }

    ggml_gallocr_free(ga);
    ggml_free(ctx_g);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_in);
    return r;
}

// mask: N_q x N_kv bytes (1 = attend) -> F16 [N_kv, N_q] with 0 / -INF
static std::vector<ggml_fp16_t> make_mask_f16(const uint8_t * allowed, int N) {
    std::vector<ggml_fp16_t> m((size_t) N * N);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (size_t i = 0; i < (size_t) N * N; ++i) m[i] = allowed[i] ? zero : ninf;
    return m;
}

// (a) naive F32 attention: kq = mul_mat(k, q); p = soft_max_ext(kq, mask, scale); out = mul_mat(v^T, p)
static run_result run_naive(const graph_env & env, const std::vector<float> & q, const std::vector<float> & k,
                            const std::vector<float> & v, int D, int H, int N, float scale, const uint8_t * allowed) {
    std::vector<ggml_fp16_t> mask;
    if (allowed) mask = make_mask_f16(allowed, N);
    return run_graph(env, 4,
        [&](struct ggml_context * c) {
            struct ggml_tensor * tq = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, N, H);
            struct ggml_tensor * tk = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, N, H);
            struct ggml_tensor * tv = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, N, H);
            struct ggml_tensor * tm = allowed ? ggml_new_tensor_2d(c, GGML_TYPE_F16, N, N) : nullptr;
            return std::vector<struct ggml_tensor *>{ tq, tk, tv, tm };
        },
        [&](struct ggml_context * c, std::vector<struct ggml_tensor *> & in) {
            ggml_backend_tensor_set(in[0], q.data(), 0, ggml_nbytes(in[0]));
            ggml_backend_tensor_set(in[1], k.data(), 0, ggml_nbytes(in[1]));
            ggml_backend_tensor_set(in[2], v.data(), 0, ggml_nbytes(in[2]));
            if (in[3]) ggml_backend_tensor_set(in[3], mask.data(), 0, ggml_nbytes(in[3]));
            struct ggml_tensor * kq = ggml_mul_mat(c, in[1], in[0]);                  // [N_kv, N_q, H]
            struct ggml_tensor * p  = ggml_soft_max_ext(c, kq, in[3], scale, 0.0f);   // rows over N_kv
            struct ggml_tensor * vt = ggml_cont(c, ggml_transpose(c, in[2]));         // [N_kv, D, H]
            return ggml_mul_mat(c, vt, p);                                            // [D, N_q, H]
        }, D, H, N, false);
}

// (b) flash attention: q F32 [D, N, H]; k, v kv_type [D, N, H]; mask F16 [N_kv, N_q] (optional)
static run_result run_flash(const graph_env & env, const std::vector<float> & q, const std::vector<float> & k,
                            const std::vector<float> & v, int D, int H, int N, float scale, const uint8_t * allowed,
                            ggml_type kv_type, ggml_prec prec) {
    std::vector<ggml_fp16_t> mask, k16, v16;
    if (allowed) mask = make_mask_f16(allowed, N);
    if (kv_type == GGML_TYPE_F16) {
        k16.resize(k.size()); v16.resize(v.size());
        ggml_fp32_to_fp16_row(k.data(), k16.data(), (int64_t) k.size());
        ggml_fp32_to_fp16_row(v.data(), v16.data(), (int64_t) v.size());
    }
    return run_graph(env, 4,
        [&](struct ggml_context * c) {
            struct ggml_tensor * tq = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, N, H);
            struct ggml_tensor * tk = ggml_new_tensor_3d(c, kv_type, D, N, H);
            struct ggml_tensor * tv = ggml_new_tensor_3d(c, kv_type, D, N, H);
            struct ggml_tensor * tm = allowed ? ggml_new_tensor_2d(c, GGML_TYPE_F16, N, N) : nullptr;
            return std::vector<struct ggml_tensor *>{ tq, tk, tv, tm };
        },
        [&](struct ggml_context * c, std::vector<struct ggml_tensor *> & in) {
            ggml_backend_tensor_set(in[0], q.data(), 0, ggml_nbytes(in[0]));
            if (kv_type == GGML_TYPE_F16) {
                ggml_backend_tensor_set(in[1], k16.data(), 0, ggml_nbytes(in[1]));
                ggml_backend_tensor_set(in[2], v16.data(), 0, ggml_nbytes(in[2]));
            } else {
                ggml_backend_tensor_set(in[1], k.data(), 0, ggml_nbytes(in[1]));
                ggml_backend_tensor_set(in[2], v.data(), 0, ggml_nbytes(in[2]));
            }
            if (in[3]) ggml_backend_tensor_set(in[3], mask.data(), 0, ggml_nbytes(in[3]));
            struct ggml_tensor * y = ggml_flash_attn_ext(c, in[0], in[1], in[2], in[3], scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(y, prec);
            return y;                                                                 // [D, H, N]
        }, D, H, N, true);
}

// (b') production recipe: fused qkv [3*D*H, N] F32 (row = [q_0..q_{H-1} | k_0.. | v_0..], head-major),
// strided views + permute -> q as a non-contiguous F32 view, K/V cast to F16 (contiguous [D, N, H]).
static run_result run_flash_fused(const graph_env & env, const std::vector<float> & q, const std::vector<float> & k,
                                  const std::vector<float> & v, int D, int H, int N, float scale, const uint8_t * allowed,
                                  ggml_type kv_type, bool cast_before_permute) {
    std::vector<ggml_fp16_t> mask;
    if (allowed) mask = make_mask_f16(allowed, N);
    // host: build the fused tensor from the [H][N][D] inputs
    std::vector<float> qkv((size_t) 3 * D * H * N);
    for (int64_t i = 0; i < N; ++i)
        for (int h = 0; h < H; ++h) {
            memcpy(&qkv[(size_t) i * 3 * D * H + (size_t) 0 * D * H + (size_t) h * D], &q[((size_t) h * N + i) * D], D * sizeof(float));
            memcpy(&qkv[(size_t) i * 3 * D * H + (size_t) 1 * D * H + (size_t) h * D], &k[((size_t) h * N + i) * D], D * sizeof(float));
            memcpy(&qkv[(size_t) i * 3 * D * H + (size_t) 2 * D * H + (size_t) h * D], &v[((size_t) h * N + i) * D], D * sizeof(float));
        }
    return run_graph(env, 2,
        [&](struct ggml_context * c) {
            struct ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, 3 * D * H, N);
            struct ggml_tensor * tm = allowed ? ggml_new_tensor_2d(c, GGML_TYPE_F16, N, N) : nullptr;
            return std::vector<struct ggml_tensor *>{ t, tm };
        },
        [&](struct ggml_context * c, std::vector<struct ggml_tensor *> & in) {
            ggml_backend_tensor_set(in[0], qkv.data(), 0, ggml_nbytes(in[0]));
            if (in[1]) ggml_backend_tensor_set(in[1], mask.data(), 0, ggml_nbytes(in[1]));
            struct ggml_tensor * qkv_t = in[0];
            const size_t es = ggml_element_size(qkv_t);
            // views [D, H, N]: nb1 = head stride (D), nb2 = token stride (3*D*H)
            struct ggml_tensor * qv = ggml_view_3d(c, qkv_t, D, H, N, D * es, qkv_t->nb[1], 0 * D * H * es);
            struct ggml_tensor * kv = ggml_view_3d(c, qkv_t, D, H, N, D * es, qkv_t->nb[1], 1 * D * H * es);
            struct ggml_tensor * vv = ggml_view_3d(c, qkv_t, D, H, N, D * es, qkv_t->nb[1], 2 * D * H * es);
            struct ggml_tensor * qp = ggml_permute(c, qv, 0, 2, 1, 3);                // [D, N, H] strided view (no cont needed)
            struct ggml_tensor * kp, * vp;
            if (kv_type == GGML_TYPE_F32) {
                kp = ggml_permute(c, kv, 0, 2, 1, 3);
                vp = ggml_permute(c, vv, 0, 2, 1, 3);
            } else if (cast_before_permute) {
                kp = ggml_permute(c, ggml_cast(c, kv, kv_type), 0, 2, 1, 3);         // cast [D,H,N] (row-contiguous), then strided view
                vp = ggml_permute(c, ggml_cast(c, vv, kv_type), 0, 2, 1, 3);
            } else {
                kp = ggml_cast(c, ggml_permute(c, kv, 0, 2, 1, 3), kv_type);         // cast of a permuted view -> contiguous [D,N,H]
                vp = ggml_cast(c, ggml_permute(c, vv, 0, 2, 1, 3), kv_type);
            }
            struct ggml_tensor * y = ggml_flash_attn_ext(c, qp, kp, vp, in[1], scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
            // production would continue with ggml_reshape_2d(y, D*H, N) -> attn_out mul_mat
            return y;
        }, D, H, N, true);
}

// --- attention study --------------------------------------------------------------------------------------
static double gb(size_t b) { return (double) b / (1024.0 * 1024.0 * 1024.0); }

struct budget {
    double max_gb = 60.0, max_s = 120.0;
    // observed throughput (updated after each run) used to predict the next case
    double ref_gflops = 20.0, naive_gflops = 200.0, flash_gflops = 500.0;
};

static bool check_flash(const char * tag, const err_stats & e) {
    const bool ok = e.rel() <= 1e-2 && e.min_cos >= 0.9999;
    if (!ok) printf("  FAIL %s: rel err %.3e (limit 1e-2), min cos %.6f (limit 0.9999)\n", tag, e.rel(), e.min_cos);
    return ok;
}

static void print_row(const char * tag, const err_stats & e, const run_result & r, double flops) {
    printf("  %-12s max|err| %.3e  rel %.3e  cos min %.7f mean %.7f  | %9.2f ms  %7.1f GFLOP/s | graph %7.3f GB work %.3f GB\n",
           tag, e.max_abs, e.rel(), e.min_cos, e.mean_cos, r.ms, flops / (r.ms * 1e6), gb(r.graph_bytes), gb(r.work_bytes));
}

static bool run_attn_case(const graph_env & env, budget & B, int D, int H, int N, bool with_fused, int & n_skipped) {
    const float scale = 1.0f / std::sqrt((float) D);
    const double flops = 4.0 * (double) N * N * D * H;                 // QK^T + PV (2 mul-adds each)
    const double score_gb = gb((size_t) N * N * H * sizeof(float));    // one F32 score matrix (softmax runs in place)
    printf("case D=%d H=%d N=%d  (score matrix %.2f GB, %.1f GFLOP)\n", D, H, N, score_gb, flops / 1e9);

    const double pred_ref_s   = flops / (B.ref_gflops * 1e9);
    const double pred_naive_s = flops / (B.naive_gflops * 1e9);
    const double pred_flash_s = flops / (B.flash_gflops * 1e9);
    if (pred_ref_s + 4 * pred_flash_s > B.max_s) {
        printf("  SKIP: predicted %.0f s (ref %.0f s + flash runs) > --max-s %.0f\n", pred_ref_s + 4 * pred_flash_s, pred_ref_s, B.max_s);
        ++n_skipped;
        return true;
    }
    std::vector<float> q((size_t) H * N * D), k(q.size()), v(q.size());
    fill_normal(q, 1000 + D * 7 + H * 3 + N, 1.0f);
    fill_normal(k, 2000 + D * 7 + H * 3 + N, 1.0f);
    fill_normal(v, 3000 + D * 7 + H * 3 + N, 1.0f);

    std::vector<double> ref;
    double t0 = now_ms();
    ref_attention(q.data(), k.data(), v.data(), D, H, N, scale, nullptr, ref, std::max(env.n_threads, 1));
    const double ref_ms = now_ms() - t0;
    B.ref_gflops = flops / (ref_ms * 1e6);
    printf("  %-12s %9.2f ms  %7.1f GFLOP/s (double, %d host threads)\n", "ref(double)", ref_ms, B.ref_gflops, env.n_threads);

    bool ok = true;
    // naive
    if (score_gb > B.max_gb) {
        printf("  %-12s SKIP: score matrix %.2f GB > --max-gb %.0f\n", "naive-f32", score_gb, B.max_gb);
        ++n_skipped;
    } else if (pred_naive_s * 3 > B.max_s) {
        printf("  %-12s SKIP: predicted %.0f s x 3 runs > --max-s %.0f\n", "naive-f32", pred_naive_s, B.max_s);
        ++n_skipped;
    } else {
        run_result r = run_naive(env, q, k, v, D, H, N, scale, nullptr);
        err_stats e = compare(r.out, ref, D, (int64_t) H * N);
        print_row("naive-f32", e, r, flops);
        B.naive_gflops = flops / (r.ms * 1e6);
    }
    // flash F16 K/V
    {
        run_result r = run_flash(env, q, k, v, D, H, N, scale, nullptr, GGML_TYPE_F16, GGML_PREC_F32);
        err_stats e = compare(r.out, ref, D, (int64_t) H * N);
        print_row("flash-f16kv", e, r, flops);
        ok &= check_flash("flash-f16kv", e);
        B.flash_gflops = flops / (r.ms * 1e6);
    }
    // flash F32 K/V
    {
        run_result r = run_flash(env, q, k, v, D, H, N, scale, nullptr, GGML_TYPE_F32, GGML_PREC_F32);
        err_stats e = compare(r.out, ref, D, (int64_t) H * N);
        print_row("flash-f32kv", e, r, flops);
        ok &= check_flash("flash-f32kv", e);
    }
    // production recipe (views + casts)
    if (with_fused) {
        run_result r = run_flash_fused(env, q, k, v, D, H, N, scale, nullptr, GGML_TYPE_F16, false);
        err_stats e = compare(r.out, ref, D, (int64_t) H * N);
        print_row("fused-f16kv", e, r, flops);
        ok &= check_flash("fused-f16kv", e);
    }
    return ok;
}

// mask semantics: block-causal over `T` frames of `B` tokens (token i attends j iff frame(j) <= frame(i)),
// plus one fully masked query row, checked against the double reference on both CPU kernels.
static bool run_mask_case(const graph_env & env, int D, int H, int T, int Bt, bool fully_masked_row) {
    const int N = T * Bt;
    const float scale = 1.0f / std::sqrt((float) D);
    std::vector<uint8_t> allowed((size_t) N * N, 0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            allowed[(size_t) i * N + j] = (j / Bt) <= (i / Bt) ? 1 : 0;
    if (fully_masked_row) for (int j = 0; j < N; ++j) allowed[(size_t) (N / 2) * N + j] = 0;

    std::vector<float> q((size_t) H * N * D), k(q.size()), v(q.size());
    fill_normal(q, 77 + N, 1.0f); fill_normal(k, 78 + N, 1.0f); fill_normal(v, 79 + N, 1.0f);
    std::vector<double> ref;
    ref_attention(q.data(), k.data(), v.data(), D, H, N, scale, allowed.data(), ref, env.n_threads);

    // sanity: the reference must differ from the unmasked one (i.e. the mask does something)
    std::vector<double> ref_nomask;
    ref_attention(q.data(), k.data(), v.data(), D, H, N, scale, nullptr, ref_nomask, env.n_threads);
    double diff = 0;
    for (size_t i = 0; i < ref.size(); ++i) diff = std::fmax(diff, std::fabs(ref[i] - ref_nomask[i]));

    bool ok = true;
    const char * kernel = N >= 64 ? "tiled (N>=64)" : "one_chunk (N<64)";
    printf("mask case D=%d H=%d T=%d B=%d N=%d [%s]%s: |masked-unmasked| %.3f\n", D, H, T, Bt, N, kernel,
           fully_masked_row ? " + fully masked row" : "", diff);
    ok &= diff > 1e-3;

    run_result rn = run_naive(env, q, k, v, D, H, N, scale, allowed.data());
    err_stats en = compare(rn.out, ref, D, (int64_t) H * N);
    printf("  %-14s max|err| %.3e rel %.3e cos min %.7f\n", "naive-f32", en.max_abs, en.rel(), en.min_cos);
    if (!fully_masked_row) ok &= en.rel() <= 1e-4; // naive softmax gives NaN on a fully -INF row (documented)

    for (ggml_type kvt : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        run_result r = run_flash(env, q, k, v, D, H, N, scale, allowed.data(), kvt, GGML_PREC_F32);
        err_stats e = compare(r.out, ref, D, (int64_t) H * N);
        const char * tag = kvt == GGML_TYPE_F16 ? "flash-f16kv" : "flash-f32kv";
        printf("  %-14s max|err| %.3e rel %.3e cos min %.7f\n", tag, e.max_abs, e.rel(), e.min_cos);
        ok &= check_flash(tag, e);
        if (fully_masked_row) { // the fully masked row must be exactly 0 (S == 0 -> 0), not NaN
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const float x = r.out[((size_t) h * N + N / 2) * D + d];
                    if (!(x == 0.0f)) { printf("  FAIL %s: fully masked row is %g, expected 0\n", tag, x); ok = false; break; }
                }
        }
    }
    {
        run_result r = run_flash_fused(env, q, k, v, D, H, N, scale, allowed.data(), GGML_TYPE_F16, false);
        err_stats e = compare(r.out, ref, D, (int64_t) H * N);
        printf("  %-14s max|err| %.3e rel %.3e cos min %.7f\n", "fused-f16kv", e.max_abs, e.rel(), e.min_cos);
        ok &= check_flash("fused-f16kv", e);
    }
    if (fully_masked_row) {
        bool nan_seen = false;
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) nan_seen |= std::isnan(rn.out[((size_t) h * N + N / 2) * D + d]);
        printf("  naive-f32 fully masked row is %s\n", nan_seen ? "NaN (soft_max_ext of an all -INF row)" : "finite");
    }
    printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// which cast order is faster for the production recipe: cast(permute(view)) vs permute(cast(view))?
static void run_cast_order_probe(const graph_env & env, int D, int H, int N) {
    const float scale = 1.0f / std::sqrt((float) D);
    std::vector<float> q((size_t) H * N * D), k(q.size()), v(q.size());
    fill_normal(q, 11, 1.0f); fill_normal(k, 12, 1.0f); fill_normal(v, 13, 1.0f);
    run_result a = run_flash_fused(env, q, k, v, D, H, N, scale, nullptr, GGML_TYPE_F16, false);
    run_result b = run_flash_fused(env, q, k, v, D, H, N, scale, nullptr, GGML_TYPE_F16, true);
    run_result c = run_flash_fused(env, q, k, v, D, H, N, scale, nullptr, GGML_TYPE_F32, false);
    run_result f = run_flash(env, q, k, v, D, H, N, scale, nullptr, GGML_TYPE_F16, GGML_PREC_F32);
    double diff = 0;
    for (size_t i = 0; i < a.out.size(); ++i) diff = std::fmax(diff, std::fabs((double) a.out[i] - b.out[i]));
    printf("cast-order probe D=%d H=%d N=%d (fused qkv -> flash, F16 K/V, timing includes the casts):\n", D, H, N);
    printf("  cast(permute(view))  %9.2f ms   graph %.3f GB\n", a.ms, gb(a.graph_bytes));
    printf("  permute(cast(view))  %9.2f ms   graph %.3f GB   (|a-b| = %.1e)\n", b.ms, gb(b.graph_bytes), diff);
    printf("  F32 K/V views, no cast %7.2f ms   graph %.3f GB\n", c.ms, gb(c.graph_bytes));
    printf("  flash kernel alone   %9.2f ms   (contiguous F16 K/V inputs)\n", f.ms);
}

// --- bench ------------------------------------------------------------------------------------------------------
static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    for (const char * p = s; *p;) {
        char * end;
        long x = strtol(p, &end, 10);
        if (end == p) break;
        v.push_back((int) x);
        p = *end == ',' ? end + 1 : end;
    }
    return v;
}

static void bench_matmul(ggml_backend_t backend, const std::vector<int> & threads, const std::vector<int> & Ns) {
    const int K = 1024, M = 4096; // W: PyTorch [M=4096 out, K=1024 in] -> ggml ne [K, M]
    std::vector<float> W((size_t) K * M);
    fill_normal(W, 42, 0.02f);
    std::vector<ggml_fp16_t> W16(W.size());
    ggml_fp32_to_fp16_row(W.data(), W16.data(), (int64_t) W.size());
    std::vector<uint8_t> W8(ggml_row_size(GGML_TYPE_Q8_0, K) * M);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, W.data(), W8.data(), 0, M, K, nullptr);

    printf("mul_mat W[%dx%d] (ggml ne [%d,%d]) x X[%dxN] F32 -> Y[%dxN]; best of >=3 runs\n", M, K, K, M, K, M);
    printf("%-6s %-6s %-8s %10s %10s %12s\n", "wtype", "N", "threads", "ms", "GFLOP/s", "max rel err");
    for (int N : Ns) {
        std::vector<float> X((size_t) K * N);
        fill_normal(X, 7 + N, 1.0f);
        // double reference for the error column (columns 0..min(N,64))
        const int n_chk = std::min(N, 64);
        std::vector<double> Yref((size_t) M * n_chk);
        for (int n = 0; n < n_chk; ++n)
            for (int m = 0; m < M; ++m) {
                double acc = 0;
                for (int kk = 0; kk < K; ++kk) acc += (double) W[(size_t) m * K + kk] * (double) X[(size_t) n * K + kk];
                Yref[(size_t) n * M + m] = acc;
            }
        double yref_max = 0;
        for (double y : Yref) yref_max = std::fmax(yref_max, std::fabs(y));

        for (ggml_type wt : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            for (int nt : threads) {
                ggml_backend_cpu_set_n_threads(backend, nt);
                struct ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
                struct ggml_context * ctx_in = ggml_init(ip);
                struct ggml_tensor * tw = ggml_new_tensor_2d(ctx_in, wt, K, M);
                struct ggml_tensor * tx = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, K, N);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
                if (wt == GGML_TYPE_F32)      ggml_backend_tensor_set(tw, W.data(), 0, ggml_nbytes(tw));
                else if (wt == GGML_TYPE_F16) ggml_backend_tensor_set(tw, W16.data(), 0, ggml_nbytes(tw));
                else                          ggml_backend_tensor_set(tw, W8.data(), 0, ggml_nbytes(tw));
                ggml_backend_tensor_set(tx, X.data(), 0, ggml_nbytes(tx));

                struct ggml_init_params gp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                struct ggml_context * ctx_g = ggml_init(gp);
                struct ggml_cgraph * gf = ggml_new_graph(ctx_g);
                struct ggml_tensor * y = ggml_mul_mat(ctx_g, tw, tx);
                ggml_build_forward_expand(gf, y);
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));

                GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS); // warm-up
                double best = 1e300, total = 0;
                int reps = 0;
                while (reps < 3 || total < 500.0) {
                    const double t0 = now_ms();
                    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
                    const double dt = now_ms() - t0;
                    best = std::fmin(best, dt); total += dt; ++reps;
                    if (reps >= 20) break;
                }
                std::vector<float> Y((size_t) M * N);
                ggml_backend_tensor_get(y, Y.data(), 0, ggml_nbytes(y));
                double max_err = 0;
                for (size_t i = 0; i < Yref.size(); ++i) max_err = std::fmax(max_err, std::fabs((double) Y[i] - Yref[i]));
                const double flops = 2.0 * M * K * N;
                printf("%-6s %-6d %-8d %10.3f %10.1f %12.3e\n", ggml_type_name(wt), N, nt, best, flops / (best * 1e6), max_err / yref_max);

                ggml_gallocr_free(ga); ggml_free(ctx_g); ggml_backend_buffer_free(buf); ggml_free(ctx_in);
            }
        }
    }
}

static bool bench_norm_gelu(ggml_backend_t backend, int n_threads) {
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    const int C = 1024, R = 1024; // 1M elements
    const float eps = 1e-6f;
    std::vector<float> x((size_t) C * R);
    fill_normal(x, 99, 1.0f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * 2.5f + 0.7f;         // non-zero mean, wider spread
    std::vector<float> g((size_t) C * R);
    { rng r(123); for (auto & t : g) t = (float) (r.uniform() * 24.0 - 12.0); } // gelu inputs in [-12, 12]

    struct ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
    struct ggml_context * ctx_in = ggml_init(ip);
    struct ggml_tensor * tx = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, C, R);
    struct ggml_tensor * tg = ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, (int64_t) C * R);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
    ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
    ggml_backend_tensor_set(tg, g.data(), 0, ggml_nbytes(tg));

    struct ggml_init_params gp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx_g = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph(ctx_g);
    struct ggml_tensor * yn = ggml_norm(ctx_g, tx, eps);
    struct ggml_tensor * yg = ggml_gelu_erf(ctx_g, tg);
    struct ggml_tensor * yq = ggml_gelu(ctx_g, tg); // tanh approximation (uses an F16 lookup table on CPU) for contrast
    ggml_build_forward_expand(gf, yn);
    ggml_build_forward_expand(gf, yg);
    ggml_build_forward_expand(gf, yq);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
    const double t0 = now_ms();
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    const double ms = now_ms() - t0;

    std::vector<float> on(x.size()), og(g.size()), oq(g.size());
    ggml_backend_tensor_get(yn, on.data(), 0, ggml_nbytes(yn));
    ggml_backend_tensor_get(yg, og.data(), 0, ggml_nbytes(yg));
    ggml_backend_tensor_get(yq, oq.data(), 0, ggml_nbytes(yq));

    double err_norm = 0;
    for (int r = 0; r < R; ++r) {
        double mean = 0, var = 0;
        for (int c = 0; c < C; ++c) mean += x[(size_t) r * C + c];
        mean /= C;
        for (int c = 0; c < C; ++c) { const double d = x[(size_t) r * C + c] - mean; var += d * d; }
        var /= C;
        const double inv = 1.0 / std::sqrt(var + (double) eps);
        for (int c = 0; c < C; ++c) err_norm = std::fmax(err_norm, std::fabs((double) on[(size_t) r * C + c] - (x[(size_t) r * C + c] - mean) * inv));
    }
    double err_gelu = 0, err_gelu4 = 0, err_tanh = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        const double xi = g[i];
        const double ref = 0.5 * xi * (1.0 + std::erf(xi / std::sqrt(2.0)));
        const double e = std::fabs((double) og[i] - ref);
        err_gelu = std::fmax(err_gelu, e);
        if (std::fabs(xi) < 4.0) err_gelu4 = std::fmax(err_gelu4, e);
        err_tanh = std::fmax(err_tanh, std::fabs((double) oq[i] - ref));
    }
    printf("ggml_norm     (1024 rows x 1024, eps 1e-6) max|err| vs double: %.3e\n", err_norm);
    printf("ggml_gelu_erf (1M in [-12,12])              max|err| vs double: %.3e   (|x|<4: %.3e)\n", err_gelu, err_gelu4);
    printf("ggml_gelu     (tanh approx, F16 table)      max|err| vs erf-GELU: %.3e   <- do NOT use for gelu_erf models\n", err_tanh);
    printf("(all three ops on 1M elements, %d threads: %.2f ms)\n", n_threads, ms);

    ggml_gallocr_free(ga); ggml_free(ctx_g); ggml_backend_buffer_free(buf); ggml_free(ctx_in);
    return err_norm < 1e-4 && err_gelu < 1e-5;
}

// --- main -------------------------------------------------------------------------------------------------------
static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [attn] [-t threads] [--hd 32,64,80] [--nh 12,16] [--n 256,2048,4608,8192,18432]\n"
        "                  [--max-gb 60] [--max-s 120] [--repeat 2] [--no-fused] [--no-mask]\n"
        "       %s bench [-t 32,96] [--n 256,2048,8192]\n", argv0, argv0);
}

int main(int argc, char ** argv) {
    bool do_bench = false;
    std::vector<int> threads = { 32 };
    std::vector<int> hds = { 32, 64, 80 }, nhs = { 12, 16 }, ns = { 256, 2048, 4608, 8192, 18432 };
    bool ns_set = false;
    budget B;
    int n_repeat = 2;
    bool with_fused = true, with_mask = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { usage(argv[0]); exit(2); } return argv[++i]; };
        if      (a == "bench")     do_bench = true;
        else if (a == "attn")      do_bench = false;
        else if (a == "-t")        threads = parse_list(next());
        else if (a == "--hd")      hds = parse_list(next());
        else if (a == "--nh")      nhs = parse_list(next());
        else if (a == "--n")       { ns = parse_list(next()); ns_set = true; }
        else if (a == "--max-gb")  B.max_gb = atof(next());
        else if (a == "--max-s")   B.max_s = atof(next());
        else if (a == "--repeat")  n_repeat = atoi(next());
        else if (a == "--no-fused") with_fused = false;
        else if (a == "--no-mask") with_mask = false;
        else { usage(argv[0]); return 2; }
    }
    if (do_bench && !ns_set) ns = { 256, 2048, 8192 };

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend);
    printf("ggml %s | avx2 %d avx512 %d avx512_bf16 %d fma %d f16c %d | GGML_F32_EPR(avx512)=16 -> flash tiled path needs head_dim %% 16 == 0\n",
           ggml_commit(), ggml_cpu_has_avx2(), ggml_cpu_has_avx512(), ggml_cpu_has_avx512_bf16(), ggml_cpu_has_fma(), ggml_cpu_has_f16c());

    if (do_bench) {
        bench_matmul(backend, threads, ns);
        const bool ok = bench_norm_gelu(backend, threads[0]);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    }

    graph_env env = { backend, threads[0], n_repeat };
    ggml_backend_cpu_set_n_threads(backend, env.n_threads);
    printf("attention study: %d threads, best of %d timed runs (after 1 warm-up), budget %.0f GB / %.0f s per case\n",
           env.n_threads, n_repeat, B.max_gb, B.max_s);

    bool ok = true;
    int n_skipped = 0, n_cases = 0;
    if (with_mask) {
        ok &= run_mask_case(env, 32, 4, 3, 16, false);   // N = 48  -> one_chunk kernel
        ok &= run_mask_case(env, 32, 4, 4, 64, false);   // N = 256 -> tiled kernel
        ok &= run_mask_case(env, 64, 2, 4, 64, true);    // tiled + one fully masked row
        ok &= run_mask_case(env, 64, 2, 2, 24, true);    // one_chunk + one fully masked row
    }
    for (int N : ns)
        for (int D : hds)
            for (int H : nhs) {
                ++n_cases;
                ok &= run_attn_case(env, B, D, H, N, with_fused, n_skipped);
            }
    if (with_fused && !ns.empty()) {
        const int n_max = *std::max_element(ns.begin(), ns.end());
        if (n_max >= 2048) run_cast_order_probe(env, 64, 16, n_max >= 4608 ? 4608 : 2048);
    }
    printf("%d attention cases, %d runs skipped by budget, peak RSS %.2f GB, result: %s\n", n_cases, n_skipped, peak_rss_gb(), ok ? "PASS" : "FAIL");
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
