// Unit test for src/rope3d.{h,cpp}: V-JEPA 2 / 2.1 3-D RoPE tables + ggml graph application,
// checked against golden vectors produced by scripts/gen_rope_ref.py (tests/vectors/rope3d/).
//
//   test-ops [vectors_dir]      exit 0 iff every case has max|err| < 1e-5
//
// Per case it checks
//   1. jepa_rope3d_tables (full grid) and jepa_rope3d_tables_ids (subsampled) against the reference
//      per-axis cos/sin rows,
//   2. jepa_rope3d_apply run on the CPU backend (ggml_backend + ggml_gallocr) with 2 heads holding
//      identical data, once on a contiguous [D, H, N] tensor and once on a strided view into a fake
//      fused [3D, H, N] qkv tensor (the layout the encoder will use),
// then prints timings for the production-sized grids.
#include "rope3d.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const double TOL = 1e-5;

// --- minimal .npy reader (v1.0 / v2.0, little-endian f32 / i32, C order) -------------------------
struct npy_array {
    std::vector<int64_t> shape;
    std::string          descr;
    std::vector<uint8_t> raw;
    size_t n() const { size_t k = 1; for (auto s : shape) k *= (size_t) s; return k; }
    const float   * f32() const { return (const float   *) raw.data(); }
    const int32_t * i32() const { return (const int32_t *) raw.data(); }
};

static bool npy_load(const std::string & path, npy_array & a) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return false; }
    uint32_t hlen = 0;
    if (magic[6] == 1) { uint16_t h = 0; if (fread(&h, 2, 1, f) != 1) { fclose(f); return false; } hlen = h; }
    else               { if (fread(&hlen, 4, 1, f) != 1) { fclose(f); return false; } }
    std::string header(hlen, '\0');
    if (fread(&header[0], 1, hlen, f) != hlen) { fclose(f); return false; }
    size_t p = header.find("'descr'");
    p = header.find('\'', p + 7);
    size_t q = header.find('\'', p + 1);
    a.descr = header.substr(p + 1, q - p - 1);
    if (header.find("'fortran_order': True") != std::string::npos) { fclose(f); return false; }
    a.shape.clear();
    p = header.find("'shape'");
    p = header.find('(', p);
    q = header.find(')', p);
    std::string dims = header.substr(p + 1, q - p - 1);
    for (size_t i = 0; i < dims.size();) {
        while (i < dims.size() && !isdigit((unsigned char) dims[i])) ++i;
        if (i >= dims.size()) break;
        a.shape.push_back(strtoll(dims.c_str() + i, nullptr, 10));
        while (i < dims.size() && isdigit((unsigned char) dims[i])) ++i;
    }
    size_t item = (a.descr == "<f4" || a.descr == "<i4") ? 4 : 0;
    if (!item) { fprintf(stderr, "%s: unsupported dtype %s\n", path.c_str(), a.descr.c_str()); fclose(f); return false; }
    a.raw.resize(a.n() * item);
    bool ok = fread(a.raw.data(), 1, a.raw.size(), f) == a.raw.size();
    fclose(f);
    return ok;
}

// --- ggml runner -------------------------------------------------------------------------------------
// x_rows: N*H rows of D floats in ggml [D, H, N] order. If via_qkv_view, the data is planted in the
// "k" slot of a [3D, H, N] tensor and the kernel is fed a strided view of it.
static std::vector<float> run_rope(ggml_backend_t backend, const std::vector<float> & x_rows, int D, int H, int N,
                                   const std::vector<float> & cos_tab, const std::vector<float> & sin_tab,
                                   bool via_qkv_view, int n_repeat = 1, double * ms_per_run = nullptr) {
    struct ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    struct ggml_context * ctx_in = ggml_init(ip);
    struct ggml_tensor * qkv = nullptr, * xt = nullptr;
    if (via_qkv_view) qkv = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, 3 * D, H, N);
    else              xt  = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, D, H, N);
    struct ggml_tensor * ct = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, D, 1, N);
    struct ggml_tensor * st = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, D, 1, N);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
    GGML_ASSERT(buf);

    if (via_qkv_view) {
        std::vector<float> big((size_t) 3 * D * H * N, -7.0f);
        for (int64_t r = 0; r < (int64_t) N * H; ++r) {
            memcpy(&big[(size_t) r * 3 * D + D], &x_rows[(size_t) r * D], D * sizeof(float));
        }
        ggml_backend_tensor_set(qkv, big.data(), 0, ggml_nbytes(qkv));
    } else {
        ggml_backend_tensor_set(xt, x_rows.data(), 0, ggml_nbytes(xt));
    }
    ggml_backend_tensor_set(ct, cos_tab.data(), 0, ggml_nbytes(ct));
    ggml_backend_tensor_set(st, sin_tab.data(), 0, ggml_nbytes(st));

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx_g = ggml_init(gp);
    struct ggml_cgraph * gf = ggml_new_graph(ctx_g);
    struct ggml_tensor * xin = via_qkv_view
        ? ggml_view_3d(ctx_g, qkv, D, H, N, qkv->nb[1], qkv->nb[2], D * sizeof(float))
        : xt;
    struct ggml_tensor * y = jepa_rope3d_apply(ctx_g, xin, ct, st);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_repeat; ++i) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    }
    auto t1 = std::chrono::steady_clock::now();
    if (ms_per_run) *ms_per_run = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_repeat;

    std::vector<float> out((size_t) D * H * N);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

    ggml_gallocr_free(ga);
    ggml_free(ctx_g);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_in);
    return out;
}

// --- test cases ----------------------------------------------------------------------------------------
struct test_case {
    std::string name, tag;
    jepa_rope3d_params p;
    int n_tok = 0;
};

// The signed table jepa_rope3d_apply consumes must be exactly the raw one with the even lanes
// negated -- bit for bit, since that is what makes the new apply bit-identical to the old one.
static bool signed_matches_raw(const std::vector<float> & sgn, const std::vector<float> & raw, int D) {
    if (sgn.size() != raw.size() || sgn.empty()) return false;
    for (size_t i = 0; i < sgn.size(); ++i) {
        const float want = (i % (size_t) D) % 2 == 0 ? -raw[i] : raw[i];
        if (memcmp(&sgn[i], &want, sizeof(float)) != 0) return false;
    }
    return true;
}

static double max_abs_diff(const float * a, const float * b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}

// Reference [n, D] tables from the factorised per-axis rows (axes: [2, gt+gh+gw, d]).
static void reference_tables(const test_case & tc, const npy_array & axes, const int32_t * ids, int n,
                             std::vector<float> & cos_ref, std::vector<float> & sin_ref) {
    const int D = tc.p.head_dim, d = jepa_rope3d_axis_dim(D);
    const int gt = tc.p.grid_t, gh = tc.p.grid_h, gw = tc.p.grid_w;
    GGML_ASSERT(axes.shape.size() == 3 && axes.shape[0] == 2 && axes.shape[1] == gt + gh + gw && axes.shape[2] == d);
    const float * A = axes.f32();
    const size_t rows = (size_t) (gt + gh + gw);
    cos_ref.assign((size_t) n * D, 1.0f);
    sin_ref.assign((size_t) n * D, 0.0f);
    for (int r = 0; r < n; ++r) {
        const int64_t i = ids ? ids[r] : r;
        const int t = (int) (i / (gh * gw)), h = (int) ((i % (gh * gw)) / gw), w = (int) (i % gw);
        const int ax[3] = { t, gt + h, gt + gh + w };
        for (int a = 0; a < 3; ++a) {
            for (int j = 0; j < d; ++j) {
                cos_ref[(size_t) r * D + a * d + j] = A[(0 * rows + ax[a]) * d + j];
                sin_ref[(size_t) r * D + a * d + j] = A[(1 * rows + ax[a]) * d + j];
            }
        }
    }
}

static bool run_case(ggml_backend_t backend, const std::string & dir, const test_case & tc) {
    const int D = tc.p.head_dim, N = tc.n_tok, H = 2;
    const int64_t n_full = (int64_t) tc.p.grid_t * tc.p.grid_h * tc.p.grid_w;

    npy_array x, out, axes, ids;
    if (!npy_load(dir + "/" + tc.name + "_x.npy", x) ||
        !npy_load(dir + "/" + tc.name + "_" + tc.tag + "_out.npy", out) ||
        !npy_load(dir + "/" + tc.name + "_" + tc.tag + "_axes.npy", axes)) return false;
    const bool subsampled = N < n_full;
    if (subsampled && !npy_load(dir + "/" + tc.name + "_ids.npy", ids)) return false;
    GGML_ASSERT(x.shape.size() == 2 && x.shape[0] == N && x.shape[1] == D);
    GGML_ASSERT(out.shape.size() == 2 && out.shape[0] == N && out.shape[1] == D);
    if (subsampled) GGML_ASSERT(ids.shape.size() == 1 && ids.shape[0] == N);
    const int32_t * id_ptr = subsampled ? ids.i32() : nullptr;

    // 1. tables: full grid always, plus the subsampled rows when applicable. The golden vectors hold
    // the raw sin(angle), so they are checked against the `signed_sin = false` form; the tables fed
    // to jepa_rope3d_apply carry the rotation's sign on the even lanes (rope3d.h), and
    // signed_matches_raw() below asserts the two differ by exactly that.
    std::vector<float> cos_full, sin_full, cos_ref, sin_ref;
    jepa_rope3d_tables(tc.p, cos_full, sin_full, /*signed_sin =*/ false);
    reference_tables(tc, axes, nullptr, (int) n_full, cos_ref, sin_ref);
    double err_tab = std::fmax(max_abs_diff(cos_full.data(), cos_ref.data(), cos_ref.size()),
                               max_abs_diff(sin_full.data(), sin_ref.data(), sin_ref.size()));
    std::vector<float> cos_tab, sin_tab, sin_raw;
    if (subsampled) {
        jepa_rope3d_tables_ids(tc.p, id_ptr, N, cos_tab, sin_raw, /*signed_sin =*/ false);
        reference_tables(tc, axes, id_ptr, N, cos_ref, sin_ref);
        err_tab = std::fmax(err_tab, std::fmax(max_abs_diff(cos_tab.data(), cos_ref.data(), cos_ref.size()),
                                               max_abs_diff(sin_raw.data(), sin_ref.data(), sin_ref.size())));
        jepa_rope3d_tables_ids(tc.p, id_ptr, N, cos_tab, sin_tab);
    } else {
        sin_raw = sin_full;
        jepa_rope3d_tables(tc.p, cos_tab, sin_tab);
    }
    if (!signed_matches_raw(sin_tab, sin_raw, D)) {
        printf("FAIL %-18s signed sin table is not the raw table with negated even lanes\n", tc.name.c_str());
        return false;
    }

    // 2. apply on ggml: H identical heads
    std::vector<float> x_rows((size_t) N * H * D), expect((size_t) N * H * D);
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            memcpy(&x_rows[((size_t) n * H + h) * D], x.f32() + (size_t) n * D, D * sizeof(float));
            memcpy(&expect[((size_t) n * H + h) * D], out.f32() + (size_t) n * D, D * sizeof(float));
        }
    }
    std::vector<float> y  = run_rope(backend, x_rows, D, H, N, cos_tab, sin_tab, false);
    std::vector<float> yv = run_rope(backend, x_rows, D, H, N, cos_tab, sin_tab, true);
    const double err_apply = max_abs_diff(y.data(), expect.data(), expect.size());
    const double err_view  = max_abs_diff(yv.data(), expect.data(), expect.size());

    const bool ok = err_tab < TOL && err_apply < TOL && err_view < TOL;
    printf("%-4s %-18s %-5s variant=%d D=%d grid=%dx%dx%d N=%d interp=%d train=%dx%dx%d | "
           "tables %.3e | apply %.3e | qkv-view %.3e\n",
           ok ? "PASS" : "FAIL", tc.name.c_str(), tc.tag.c_str(), tc.p.variant, D,
           tc.p.grid_t, tc.p.grid_h, tc.p.grid_w, N, (int) tc.p.interpolate,
           tc.p.train_grid_t, tc.p.train_grid_h, tc.p.train_grid_w, err_tab, err_apply, err_view);
    return ok;
}

static void bench(ggml_backend_t backend, int gt, int gh, int gw, int D, int H, int variant) {
    jepa_rope3d_params p;
    p.grid_t = gt; p.grid_h = gh; p.grid_w = gw; p.head_dim = D; p.variant = variant;
    const int N = gt * gh * gw;
    std::vector<float> cos_tab, sin_tab;
    auto t0 = std::chrono::steady_clock::now();
    jepa_rope3d_tables(p, cos_tab, sin_tab);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> x((size_t) N * H * D);
    uint32_t s = 12345;
    for (auto & v : x) { s = s * 1664525u + 1013904223u; v = (float) (s >> 8) / (float) (1u << 24) - 0.5f; }
    double ms = 0;
    run_rope(backend, x, D, H, N, cos_tab, sin_tab, false, 5, &ms);
    printf("bench grid=%dx%dx%d (N=%d) D=%d H=%d: tables %.2f ms, apply %.2f ms/run (avg of 5)\n",
           gt, gh, gw, N, D, H, std::chrono::duration<double, std::milli>(t1 - t0).count(), ms);
}

static bool parse_manifest(const std::string & path, std::vector<test_case> & cases) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char name[128], tag[32];
        int variant, D, gt, gh, gw, n_tok, interp, tt, th, tw;
        float theta;
        if (sscanf(line, "%127s %31s %d %d %d %d %d %d %d %d %d %d %f",
                   name, tag, &variant, &D, &gt, &gh, &gw, &n_tok, &interp, &tt, &th, &tw, &theta) != 13) {
            fprintf(stderr, "bad manifest line: %s", line);
            fclose(f);
            return false;
        }
        test_case tc;
        tc.name = name; tc.tag = tag; tc.n_tok = n_tok;
        tc.p.grid_t = gt; tc.p.grid_h = gh; tc.p.grid_w = gw; tc.p.head_dim = D; tc.p.theta = theta;
        tc.p.interpolate = interp != 0; tc.p.train_grid_t = tt; tc.p.train_grid_h = th; tc.p.train_grid_w = tw;
        tc.p.variant = variant;
        cases.push_back(tc);
    }
    fclose(f);
    return !cases.empty();
}

int main(int argc, char ** argv) {
    std::string dir;
    if (argc > 1) {
        dir = argv[1];
    } else {
        for (const char * c : { "tests/vectors/rope3d", "../tests/vectors/rope3d", "../../tests/vectors/rope3d" }) {
            FILE * f = fopen((std::string(c) + "/manifest.txt").c_str(), "r");
            if (f) { fclose(f); dir = c; break; }
        }
        if (dir.empty()) { fprintf(stderr, "usage: %s <tests/vectors/rope3d>\n", argv[0]); return 2; }
    }
    std::vector<test_case> cases;
    if (!parse_manifest(dir + "/manifest.txt", cases)) return 2;

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend);
    ggml_backend_cpu_set_n_threads(backend, 8);

    int n_fail = 0;
    for (const auto & tc : cases) {
        if (!run_case(backend, dir, tc)) ++n_fail;
    }
    printf("%d/%d cases passed (tolerance %.0e)\n", (int) cases.size() - n_fail, (int) cases.size(), TOL);

    if (n_fail == 0) {
        bench(backend, 32, 16, 16, 64, 16, JEPA_ROPE3D_VJEPA2);   // V-JEPA 2 ViT-L, 64 frames @ 256
        bench(backend, 32, 24, 24, 64, 12, JEPA_ROPE3D_VJEPA2_1); // V-JEPA 2.1 ViT-B, 64 frames @ 384
    }
    ggml_backend_free(backend);
    return n_fail == 0 ? 0 : 1;
}
