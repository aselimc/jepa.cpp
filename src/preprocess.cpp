// Image / video preprocessing: stb_image decode + a faithful port of the uint8 antialiased resize that
// torchvision.transforms.v2.functional.resize(antialias=True) runs on CPU (PyTorch
// aten/src/ATen/native/cpu/UpSampleKernel.cpp, `upsample_avx_bilinear_bicubic_uint8`, itself a port of
// Pillow's ImagingResample):
//   * separable: horizontal pass first, rounded to uint8, then vertical pass;
//   * filter support scaled by the downscale factor (triangle for bilinear, Keys cubic a=-0.5 for bicubic);
//   * weights computed in double, normalised, then quantised to int16 with a per-pass precision chosen so
//     that the largest weight fits 15 bits; accumulation in int32 with a rounding offset of 1<<(prec-1);
//   * output = clamp(acc >> prec, 0, 255).
// followed by centre crop (top = (H-c)/2, left = (W-c)/2) and mean/std normalisation.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

#include "jepa-internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

void jepa_free(void * p) { free(p); }

// ---------------------------------------------------------------------------------------------
// antialiased resize (uint8, int16 fixed point)
// ---------------------------------------------------------------------------------------------
namespace {

double filter_linear(double x) {
    x = std::fabs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

double filter_cubic(double x) {
    // PyTorch cubic_convolution1 / cubic_convolution2 with A = -0.5 (same op order)
    const double A = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    return 0.0;
}

struct aa_table {
    int ksize = 0;       // weights per output position
    int precision = 0;   // fixed-point bits
    std::vector<int>     xmin, xsize;
    std::vector<int16_t> k;
};

aa_table compute_aa_table(int in_size, int out_size, int resample) {
    // out_size 0 would make `scale` infinite and `(int) ceil(support)` undefined; callers check, so
    // this only closes the door.
    if (in_size <= 0 || out_size <= 0) return aa_table{};
    const int interp_size = resample == JEPA_RESAMPLE_BICUBIC ? 4 : 2;
    double (*filter)(double) = resample == JEPA_RESAMPLE_BICUBIC ? filter_cubic : filter_linear;
    const double scale = (double) in_size / (double) out_size;
    const double support = scale >= 1.0 ? (interp_size * 0.5) * scale : interp_size * 0.5;
    const int max_interp = (int) std::ceil(support) * 2 + 1;

    aa_table t;
    t.ksize = max_interp;
    t.xmin.assign(out_size, 0);
    t.xsize.assign(out_size, 0);
    std::vector<double> w((size_t) out_size * max_interp, 0.0);
    double wt_max = 0.0;
    for (int i = 0; i < out_size; i++) {
        const double center = scale * (i + 0.5);
        const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
        int64_t xmin = std::max((int64_t) (center - support + 0.5), (int64_t) 0);
        int64_t xsize = std::min((int64_t) (center + support + 0.5), (int64_t) in_size) - xmin;
        xsize = std::min(std::max(xsize, (int64_t) 0), (int64_t) max_interp);
        double * wp = w.data() + (size_t) i * max_interp;
        double total = 0.0;
        for (int64_t j = 0; j < xsize; j++) {
            const double v = filter((j + xmin - center + 0.5) * invscale);
            wp[j] = v;
            total += v;
        }
        if (total != 0.0) {
            for (int64_t j = 0; j < xsize; j++) {
                wp[j] /= total;
                wt_max = std::max(wt_max, wp[j]);
            }
        }
        t.xmin[i] = (int) xmin;
        t.xsize[i] = (int) xsize;
    }
    int prec = 0;
    for (prec = 0; prec < 22; prec++) {
        const int next_value = (int) (0.5 + wt_max * (double) (1 << (prec + 1)));
        if (next_value >= (1 << 15)) break;
    }
    t.precision = prec;
    t.k.assign(w.size(), 0);
    for (size_t j = 0; j < w.size(); j++) {
        const double v = w[j] * (double) (1 << prec);
        t.k[j] = (int16_t) (v < 0 ? (int) (-0.5 + v) : (int) (0.5 + v));
    }
    return t;
}

inline uint8_t clip8(int32_t acc, int prec) {
    const int32_t v = acc >> prec;
    return (uint8_t) (v < 0 ? 0 : (v > 255 ? 255 : v));
}

} // namespace

void jepa_resize_antialias_u8(const uint8_t * src, int h, int w, int c, uint8_t * dst, int out_h, int out_w, int resample) {
    // Public entry point: a non-positive extent would turn into a huge size_t in the memcpy below.
    if (!src || !dst || h <= 0 || w <= 0 || c <= 0 || out_h <= 0 || out_w <= 0) {
        jepa_log("jepa: jepa_resize_antialias_u8: %dx%dx%d -> %dx%d is not a resize\n", h, w, c, out_h, out_w);
        return;
    }
    if (h == out_h && w == out_w) {
        memcpy(dst, src, (size_t) h * w * c);
        return;
    }
    std::vector<uint8_t> tmp;
    const uint8_t * cur = src;
    int cur_w = w;
    if (out_w != w) {
        const aa_table t = compute_aa_table(w, out_w, resample);
        const int32_t init = t.precision > 0 ? (1 << (t.precision - 1)) : 0;
        uint8_t * out = out_h == h ? dst : (tmp.resize((size_t) h * out_w * c), tmp.data());
        for (int y = 0; y < h; y++) {
            const uint8_t * row = src + (size_t) y * w * c;
            uint8_t * orow = out + (size_t) y * out_w * c;
            for (int x = 0; x < out_w; x++) {
                const int xmin = t.xmin[x], xsize = t.xsize[x];
                const int16_t * kk = t.k.data() + (size_t) x * t.ksize;
                for (int ch = 0; ch < c; ch++) {
                    int32_t acc = init;
                    const uint8_t * p = row + (size_t) xmin * c + ch;
                    for (int j = 0; j < xsize; j++) acc += (int32_t) p[(size_t) j * c] * kk[j];
                    orow[(size_t) x * c + ch] = clip8(acc, t.precision);
                }
            }
        }
        if (out_h == h) return;
        cur = tmp.data();
        cur_w = out_w;
    }
    // vertical pass
    const aa_table t = compute_aa_table(h, out_h, resample);
    const int32_t init = t.precision > 0 ? (1 << (t.precision - 1)) : 0;
    std::vector<int32_t> acc((size_t) cur_w * c);
    for (int y = 0; y < out_h; y++) {
        const int ymin = t.xmin[y], ysize = t.xsize[y];
        const int16_t * kk = t.k.data() + (size_t) y * t.ksize;
        std::fill(acc.begin(), acc.end(), init);
        for (int j = 0; j < ysize; j++) {
            const uint8_t * row = cur + (size_t) (ymin + j) * cur_w * c;
            const int32_t kj = kk[j];
            for (size_t i = 0; i < acc.size(); i++) acc[i] += (int32_t) row[i] * kj;
        }
        uint8_t * orow = dst + (size_t) y * cur_w * c;
        for (size_t i = 0; i < acc.size(); i++) orow[i] = clip8(acc[i], t.precision);
    }
}

// ---------------------------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------------------------
jepa_preprocess_params jepa_preprocess_default_params(const jepa_model * model) {
    jepa_preprocess_params p;
    const jepa_pre_hparams & h = model->hp.pre;
    for (int i = 0; i < 3; i++) { p.mean[i] = h.mean[i]; p.std[i] = h.std[i]; }
    p.resize_short = h.resize_short;
    p.crop = h.crop;
    p.resample = h.resample == "bicubic" ? JEPA_RESAMPLE_BICUBIC : JEPA_RESAMPLE_BILINEAR;
    p.resize_mode = h.resize_mode == "squash" ? JEPA_RESIZE_SQUASH : JEPA_RESIZE_SHORTEST_EDGE;
    p.rescale = h.rescale;
    p.fused_norm = true;
    return p;
}

uint8_t * jepa_load_image_rgb(const char * path, int * h, int * w) {
    int x = 0, y = 0, n = 0;
    uint8_t * data = stbi_load(path, &x, &y, &n, 3);
    if (!data) {
        jepa_log("jepa: cannot decode image %s: %s\n", path, stbi_failure_reason());
        return nullptr;
    }
    // stbi uses malloc, so the caller can jepa_free() it
    *h = y; *w = x;
    return data;
}

// The intermediate the shortest-edge resize goes through is `resize_short` on the short side and
// resize_short * long/short on the other, so a 16384x1 image (stb decodes up to 2^24 per axis) asks
// for a 224 x 3.6 M buffer — 2.5 GiB — from a 48 KB file. That is the whole aspect-ratio class, and
// the cap below is what turns it into a refusal. 64 megapixels is 25x the largest sane input
// (8000x8000) and 260x a 224-crop model's own working set.
#define JEPA_MAX_RESIZE_PIXELS (64ll * 1024 * 1024)

// Resize + centre crop one HWC RGB frame to crop x crop (uint8), following the transformers torchvision backend.
// Returns false when the geometry is not something this pipeline can carry out.
static bool resize_crop_u8(const jepa_preprocess_params * p, const uint8_t * rgb, int h, int w, std::vector<uint8_t> & out, int * out_size) {
    if (p->resize_short <= 0) {
        jepa_log("jepa: preprocess: resize_short is %d — nothing to resize to\n", p->resize_short);
        return false;
    }
    int64_t rh64, rw64;
    if (p->resize_mode == JEPA_RESIZE_SQUASH) {
        rh64 = rw64 = p->resize_short;
    } else if (w <= h) {
        rw64 = p->resize_short;
        rh64 = (int64_t) p->resize_short * h / w;
    } else {
        rh64 = p->resize_short;
        rw64 = (int64_t) p->resize_short * w / h;
    }
    if (rh64 < 1) rh64 = 1;
    if (rw64 < 1) rw64 = 1;
    if (rh64 * rw64 > JEPA_MAX_RESIZE_PIXELS) {
        jepa_log("jepa: preprocess: %dx%d at shortest edge %d resizes to %lldx%lld = %.1f megapixels, "
                 "over the %lld-megapixel limit — the aspect ratio, not the input size, is what makes "
                 "this buffer large\n", h, w, p->resize_short, (long long) rh64, (long long) rw64,
                 (double) (rh64 * rw64) / (1024.0 * 1024.0), (long long) (JEPA_MAX_RESIZE_PIXELS >> 20));
        return false;
    }
    const int rh = (int) rh64, rw = (int) rw64;
    std::vector<uint8_t> resized((size_t) rh * rw * 3);
    jepa_resize_antialias_u8(rgb, h, w, 3, resized.data(), rh, rw, p->resample);

    const int crop = p->crop > 0 ? p->crop : std::min(rh, rw);
    const uint8_t * cur = resized.data();
    int ch = rh, cw = rw;
    std::vector<uint8_t> padded;
    if (crop > ch || crop > cw) {
        // torchvision center_crop pads with zeros when the image is smaller than the crop
        const int pl = crop > cw ? (crop - cw) / 2 : 0, pt = crop > ch ? (crop - ch) / 2 : 0;
        const int nw = std::max(crop, cw), nh = std::max(crop, ch);
        padded.assign((size_t) nh * nw * 3, 0);
        for (int y = 0; y < ch; y++) memcpy(padded.data() + ((size_t) (y + pt) * nw + pl) * 3, cur + (size_t) y * cw * 3, (size_t) cw * 3);
        cur = padded.data(); ch = nh; cw = nw;
    }
    const int top = (ch - crop) / 2, left = (cw - crop) / 2;
    out.resize((size_t) crop * crop * 3);
    for (int y = 0; y < crop; y++) {
        memcpy(out.data() + (size_t) y * crop * 3, cur + ((size_t) (top + y) * cw + left) * 3, (size_t) crop * 3);
    }
    *out_size = crop;
    return true;
}

float * jepa_preprocess_frames_rgb_ex(const jepa_preprocess_params * p, const uint8_t * const * frames, int n_frames, int h, int w, int * out_h, int * out_w) {
    if (!p || !frames || n_frames <= 0 || h <= 0 || w <= 0) return nullptr;
    // The explicit-parameter entry points take this struct from the caller, so it gets the same
    // checks jepa_hparams_from_gguf applies to jepa.pre.*: a zero std divides every pixel by zero.
    if ((double) h * w * 3 >= (double) SIZE_MAX) {
        jepa_log("jepa: preprocess: a %dx%d RGB frame cannot be a buffer\n", h, w);
        return nullptr;
    }
    for (int c = 0; c < 3; c++) {
        if (!(p->std[c] > 1e-9f) || !(p->std[c] < 1e9f) || !(p->rescale > 0.0f)) {
            jepa_log("jepa: preprocess: std[%d] = %g and rescale = %g must be finite and non-zero\n",
                     c, (double) p->std[c], (double) p->rescale);
            return nullptr;
        }
    }
    // Invert the (float32) rescale factor the way the HF processors do in double: 1/0.00392156862745098
    // = 255.00000000000003 -> 255.0f. Snap to the nearest integer so that float32(1/255) round-trips.
    double invd = 1.0 / (double) p->rescale;
    if (std::fabs(invd - std::nearbyint(invd)) < 1e-3 * std::fabs(invd)) invd = std::nearbyint(invd);
    const float inv = (float) invd;   // 255
    float ms[3], ss[3];
    for (int c = 0; c < 3; c++) {
        if (p->fused_norm) { ms[c] = p->mean[c] * inv; ss[c] = p->std[c] * inv; }
        else               { ms[c] = p->mean[c];       ss[c] = p->std[c]; }
    }
    float * out = nullptr;
    int crop = 0;
    std::vector<uint8_t> u8;
    for (int t = 0; t < n_frames; t++) {
        int sz = 0;
        if (!frames[t] || !resize_crop_u8(p, frames[t], h, w, u8, &sz) || sz <= 0) { free(out); return nullptr; }
        if (!out) {
            crop = sz;
            if ((double) 3 * n_frames * crop * crop * sizeof(float) >= (double) SIZE_MAX) {
                jepa_log("jepa: preprocess: %d frames of %dx%d floats do not fit in memory\n", n_frames, crop, crop);
                return nullptr;
            }
            out = (float *) malloc((size_t) 3 * n_frames * crop * crop * sizeof(float));
            if (!out) return nullptr;
        }
        const size_t plane = (size_t) crop * crop;
        for (int c = 0; c < 3; c++) {
            float * dst = out + ((size_t) c * n_frames + t) * plane;
            const uint8_t * src = u8.data() + c;
            if (p->fused_norm) {
                for (size_t i = 0; i < plane; i++) dst[i] = ((float) src[i * 3] - ms[c]) / ss[c];
            } else {
                for (size_t i = 0; i < plane; i++) dst[i] = ((float) src[i * 3] / inv - ms[c]) / ss[c];
            }
        }
    }
    if (out_h) *out_h = crop;
    if (out_w) *out_w = crop;
    return out;
}

float * jepa_preprocess_image_rgb_ex(const jepa_preprocess_params * p, const uint8_t * rgb, int h, int w, int * out_h, int * out_w) {
    const uint8_t * frames[1] = { rgb };
    return jepa_preprocess_frames_rgb_ex(p, frames, 1, h, w, out_h, out_w);
}

float * jepa_preprocess_frames_rgb(const jepa_model * model, const uint8_t * const * frames, int n_frames, int h, int w, int * out_h, int * out_w) {
    const jepa_preprocess_params p = jepa_preprocess_default_params(model);
    return jepa_preprocess_frames_rgb_ex(&p, frames, n_frames, h, w, out_h, out_w);
}

float * jepa_preprocess_image_rgb(const jepa_model * model, const uint8_t * rgb, int h, int w, int * out_h, int * out_w) {
    const jepa_preprocess_params p = jepa_preprocess_default_params(model);
    return jepa_preprocess_image_rgb_ex(&p, rgb, h, w, out_h, out_w);
}

float * jepa_preprocess_image_file(const jepa_model * model, const char * path, int * out_h, int * out_w) {
    int h = 0, w = 0;
    uint8_t * rgb = jepa_load_image_rgb(path, &h, &w);
    if (!rgb) return nullptr;
    float * out = jepa_preprocess_image_rgb(model, rgb, h, w, out_h, out_w);
    free(rgb);
    return out;
}
