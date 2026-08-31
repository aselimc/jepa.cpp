// V-JEPA 2 / 2.1 3-axis RoPE: host-side cos/sin tables + ggml graph application.
// The exact math and the source citations are in rope3d.h.
#include "rope3d.h"

#include <cmath>
#include <cstdint>
#include <vector>

int jepa_rope3d_axis_dim(int head_dim) {
    return 2 * ((head_dim / 3) / 2);
}

void jepa_rope3d_position(const jepa_rope3d_params & p, int64_t i, float & t, float & h, float & w) {
    const int64_t per_frame = (int64_t) p.grid_h * p.grid_w;
    const int64_t ti = i / per_frame;
    const int64_t hi = (i % per_frame) / p.grid_w;
    const int64_t wi = i % p.grid_w;
    t = (float) ti;
    h = (float) hi;
    w = (float) wi;
    if (p.interpolate) {
        // V-JEPA 2.1 modules.py:232-233 -- `h_mask * (pretrained_grid_size - 1) / (H_patches - 1)`,
        // evaluated left-to-right in float32. t is never rescaled. A 1-wide grid would divide by zero
        // upstream; its only coordinate is 0 anyway, so leave it.
        if (p.grid_h > 1) h = (h * (float) (p.train_grid_h - 1)) / (float) (p.grid_h - 1);
        if (p.grid_w > 1) w = (w * (float) (p.train_grid_w - 1)) / (float) (p.grid_w - 1);
    }
}

// cos/sin of one axis for one position, written into c[0..d) / s[0..d).
// variant 0: tiled     (freq index j % (d/2))  -- HF 195-196, V2 43-44
// variant 1: interleaved (freq index j / 2)    -- V21 36-37
static void rope3d_axis_row(int variant, int d, const std::vector<float> & omega, float pos, float * c, float * s) {
    const int half = d / 2;
    for (int j = 0; j < d; ++j) {
        const int   k   = variant == JEPA_ROPE3D_VJEPA2 ? (j % half) : (j / 2);
        const float ang = pos * omega[k]; // float32 product, as in torch
        c[j] = cosf(ang);
        s[j] = sinf(ang);
    }
}

// omega_k = 1 / theta^(k / (d/2)), k in [0, d/2), in float32 like the references:
//   omega = arange(d/2, float32); omega /= d/2; omega = 1.0 / theta**omega
static std::vector<float> rope3d_omega(int d, float theta) {
    const int half = d / 2;
    std::vector<float> omega(half);
    for (int k = 0; k < half; ++k) {
        const float e = (float) k / (float) half;
        omega[k] = 1.0f / powf(theta, e);
    }
    return omega;
}

// Shared worker: `pos(r, t, h, w)` yields the (rescaled) coordinates of output row r.
template <typename PosFn>
static void rope3d_build(const jepa_rope3d_params & p, int64_t n_rows, PosFn pos,
                         std::vector<float> & cos_out, std::vector<float> & sin_out, bool signed_sin) {
    const int D = p.head_dim;
    const int d = jepa_rope3d_axis_dim(D);
    GGML_ASSERT(D > 0 && d >= 0 && 3 * d <= D);
    GGML_ASSERT(p.grid_t >= 1 && p.grid_h >= 1 && p.grid_w >= 1);
    GGML_ASSERT(p.variant == JEPA_ROPE3D_VJEPA2 || p.variant == JEPA_ROPE3D_VJEPA2_1);
    if (p.interpolate) {
        GGML_ASSERT((p.grid_h <= 1 || p.train_grid_h >= 1) && (p.grid_w <= 1 || p.train_grid_w >= 1));
    }

    cos_out.assign((size_t) n_rows * D, 1.0f);
    sin_out.assign((size_t) n_rows * D, 0.0f);
    if (d == 0) return; // head_dim < 4: nothing rotates

    const std::vector<float> omega = rope3d_omega(d, p.theta);

    // Per-axis rows are cached by grid coordinate: the (rescaled) position depends only on the index.
    std::vector<float> ct((size_t) p.grid_t * d), st((size_t) p.grid_t * d);
    std::vector<float> ch((size_t) p.grid_h * d), sh((size_t) p.grid_h * d);
    std::vector<float> cw((size_t) p.grid_w * d), sw((size_t) p.grid_w * d);
    for (int i = 0; i < p.grid_t; ++i) {
        float t, h, w;
        jepa_rope3d_position(p, (int64_t) i * p.grid_h * p.grid_w, t, h, w);
        rope3d_axis_row(p.variant, d, omega, t, &ct[(size_t) i * d], &st[(size_t) i * d]);
    }
    for (int i = 0; i < p.grid_h; ++i) {
        float t, h, w;
        jepa_rope3d_position(p, (int64_t) i * p.grid_w, t, h, w);
        rope3d_axis_row(p.variant, d, omega, h, &ch[(size_t) i * d], &sh[(size_t) i * d]);
    }
    for (int i = 0; i < p.grid_w; ++i) {
        float t, h, w;
        jepa_rope3d_position(p, i, t, h, w);
        rope3d_axis_row(p.variant, d, omega, w, &cw[(size_t) i * d], &sw[(size_t) i * d]);
    }

    for (int64_t r = 0; r < n_rows; ++r) {
        int ti, hi, wi;
        pos(r, ti, hi, wi);
        GGML_ASSERT(ti >= 0 && ti < p.grid_t && hi >= 0 && hi < p.grid_h && wi >= 0 && wi < p.grid_w);
        float * c = &cos_out[(size_t) r * D];
        float * s = &sin_out[(size_t) r * D];
        for (int j = 0; j < d; ++j) {
            c[j]         = ct[(size_t) ti * d + j];  s[j]         = st[(size_t) ti * d + j];
            c[d + j]     = ch[(size_t) hi * d + j];  s[d + j]     = sh[(size_t) hi * d + j];
            c[2 * d + j] = cw[(size_t) wi * d + j];  s[2 * d + j] = sw[(size_t) wi * d + j];
        }
        // Fold the rotation's sign into the table (rope3d.h "Implementation"): the graph then needs
        // one pair swap instead of two masked rolls. out[2k] = x[2k]*C - x[2k+1]*S is unchanged.
        if (signed_sin) {
            for (int j = 0; j < D; j += 2) s[j] = -s[j];
        }
    }
}

static inline void rope3d_split_id(const jepa_rope3d_params & p, int64_t i, int & ti, int & hi, int & wi) {
    const int64_t per_frame = (int64_t) p.grid_h * p.grid_w;
    ti = (int) (i / per_frame);
    hi = (int) ((i % per_frame) / p.grid_w);
    wi = (int) (i % p.grid_w);
}

void jepa_rope3d_tables(const jepa_rope3d_params & p, std::vector<float> & cos_out, std::vector<float> & sin_out,
                        bool signed_sin) {
    const int64_t N = (int64_t) p.grid_t * p.grid_h * p.grid_w;
    rope3d_build(p, N, [&](int64_t r, int & ti, int & hi, int & wi) { rope3d_split_id(p, r, ti, hi, wi); },
                 cos_out, sin_out, signed_sin);
}

void jepa_rope3d_tables_ids(const jepa_rope3d_params & p, const int32_t * ids, int n_ids,
                            std::vector<float> & cos_out, std::vector<float> & sin_out, bool signed_sin) {
    const int64_t N = (int64_t) p.grid_t * p.grid_h * p.grid_w;
    rope3d_build(p, n_ids, [&](int64_t r, int & ti, int & hi, int & wi) {
        GGML_ASSERT(ids[r] >= 0 && ids[r] < N);
        rope3d_split_id(p, ids[r], ti, hi, wi);
    }, cos_out, sin_out, signed_sin);
}

struct ggml_tensor * jepa_rope3d_apply(struct ggml_context * ctx, struct ggml_tensor * x,
                                       struct ggml_tensor * cos_t, struct ggml_tensor * sin_t) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && cos_t->type == GGML_TYPE_F32 && sin_t->type == GGML_TYPE_F32);
    const int64_t D = x->ne[0];
    GGML_ASSERT(D % 2 == 0);
    GGML_ASSERT(x->nb[0] == sizeof(float)); // rows must be contiguous (ggml_roll / binary ops)
    GGML_ASSERT(x->ne[3] == 1);
    GGML_ASSERT(cos_t->ne[0] == D && cos_t->ne[1] == 1 && cos_t->ne[2] == x->ne[2] && cos_t->ne[3] == 1);
    GGML_ASSERT(ggml_are_same_shape(cos_t, sin_t));
    const int64_t H = x->ne[1], N = x->ne[2];

    // ggml_roll needs a fully contiguous source (its CUDA kernel indexes as if it were: see
    // ggml/src/ggml-cuda/roll.cu and docs/gpu-notes.md S1.2), and jepa_build_qkv hands us a view
    // into the fused [3D, N] projection. The copy is not wasted work: the old two-roll form
    // materialised the same bytes twice inside `roll` itself.
    struct ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);

    // rotate90(x)[j] = -x[j^1] for even j, +x[j^1] for odd j -- i.e. the pair swap with the sign
    // already folded into sin_t. Rolling a length-2 axis by 1 IS that swap: over [2, D/2*H*N],
    // out[0] = in[1] and out[1] = in[0], with no wrap-around lane left to mask off.
    struct ggml_tensor * sw = ggml_roll(ctx, ggml_reshape_2d(ctx, xc, 2, D * H * N / 2), 1, 0, 0, 0);
    sw = ggml_reshape_3d(ctx, sw, D, H, N);

    // out = x*C + swap(x)*S'  ==  out[2k] = x[2k]*C[2k] - x[2k+1]*S[2k]
    //                             out[2k+1] = x[2k+1]*C[2k+1] + x[2k]*S[2k+1]
    return ggml_add(ctx, ggml_mul(ctx, xc, cos_t), ggml_mul(ctx, sw, sin_t));
}
