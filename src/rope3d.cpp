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
                         std::vector<float> & cos_out, std::vector<float> & sin_out) {
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
    }
}

static inline void rope3d_split_id(const jepa_rope3d_params & p, int64_t i, int & ti, int & hi, int & wi) {
    const int64_t per_frame = (int64_t) p.grid_h * p.grid_w;
    ti = (int) (i / per_frame);
    hi = (int) ((i % per_frame) / p.grid_w);
    wi = (int) (i % p.grid_w);
}

void jepa_rope3d_tables(const jepa_rope3d_params & p, std::vector<float> & cos_out, std::vector<float> & sin_out) {
    const int64_t N = (int64_t) p.grid_t * p.grid_h * p.grid_w;
    rope3d_build(p, N, [&](int64_t r, int & ti, int & hi, int & wi) { rope3d_split_id(p, r, ti, hi, wi); }, cos_out, sin_out);
}

void jepa_rope3d_tables_ids(const jepa_rope3d_params & p, const int32_t * ids, int n_ids,
                            std::vector<float> & cos_out, std::vector<float> & sin_out) {
    const int64_t N = (int64_t) p.grid_t * p.grid_h * p.grid_w;
    rope3d_build(p, n_ids, [&](int64_t r, int & ti, int & hi, int & wi) {
        GGML_ASSERT(ids[r] >= 0 && ids[r] < N);
        rope3d_split_id(p, ids[r], ti, hi, wi);
    }, cos_out, sin_out);
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

    // Pair masks over head_dim, built in-graph so callers only upload the plain tables:
    //   m_odd = [0,1,0,1,...]   m_evn = [-1,0,-1,0,...]
    struct ggml_tensor * m01   = ggml_arange(ctx, 0.0f, 2.0f, 1.0f);                       // [0, 1]
    struct ggml_tensor * m_odd = ggml_reshape_1d(ctx, ggml_repeat_4d(ctx, m01, 2, D / 2, 1, 1), D);
    struct ggml_tensor * m_evn = ggml_scale_bias(ctx, m_odd, 1.0f, -1.0f);

    // rotate90(x)*S == roll(x,-1) * (-S on even dims) + roll(x,+1) * (S on odd dims):
    //   even j=2k : roll(x,-1)[j] = x[2k+1], factor -S[2k]
    //   odd  j=2k+1: roll(x,+1)[j] = x[2k],   factor +S[2k+1]
    // The wrapped-around elements (j = D-1 and j = 0) are multiplied by exactly 0.
    struct ggml_tensor * s_e = ggml_mul(ctx, sin_t, m_evn); // [D, 1, N]
    struct ggml_tensor * s_o = ggml_mul(ctx, sin_t, m_odd); // [D, 1, N]
    struct ggml_tensor * x_next = ggml_roll(ctx, x, -1, 0, 0, 0); // x_next[j] = x[j+1]
    struct ggml_tensor * x_prev = ggml_roll(ctx, x,  1, 0, 0, 0); // x_prev[j] = x[j-1]

    struct ggml_tensor * y = ggml_mul(ctx, x, cos_t);              // x * C
    y = ggml_add(ctx, y, ggml_mul(ctx, x_next, s_e));              // + rotate90(x) * S  (even dims)
    y = ggml_add(ctx, y, ggml_mul(ctx, x_prev, s_o));              // + rotate90(x) * S  (odd dims)
    return y;
}
