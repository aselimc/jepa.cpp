// test-batch: the batched encoder graph must return exactly what the one-item-per-graph path returns.
//
//   test-batch --media DIR [--threads N] [--images N] [--chunk N] [--verbose]
//              --image MODEL.gguf [--images N] [--image MODEL2.gguf ...]
//              [--video MODEL.gguf --clip A.frames_u8.npy --clip B.frames_u8.npy [--clip-frames N]]
//
// `--images N` applies to every `--image` after it, so a big model can be given fewer of them;
// `--clip-frames N` truncates the clips to their first N frames (a multiple of the tubelet size).
//
// For every `--image` model the same N fixture JPEGs are encoded
//   (a) one per jepa_encode call, with jepa_context_set_max_batch(ctx, 1) — the pre-batching path;
//   (b) as ONE jepa_input with n_batch = N and max_batch = N — a single ggml graph that carries the
//       items on the batch dimension;
//   (c) as one jepa_input with n_batch = N but max_batch = `--chunk` (3 by default) — the chunk loop.
// (b) and (c) must equal (a) *bit for bit* at f32 (rows and both pooled vectors); other file types
// are reported with their measured max|delta| / worst-row cosine and gated at cosine >= 1 - 1e-6.
//
// `--video` does the same for the V-JEPA 2 / 2.1 clip path: two clips as n_batch = 2 against two
// n_batch = 1 calls. That path still builds one graph per clip (docs/results.md explains why), so
// the check is a regression guard on the shared block builders, not on a batched video graph.
//
// Exit status 1 on any mismatch. Registered with ctest as "batch" (assets permitting).
#include "jepa.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool g_verbose = false;
static int  g_fail = 0;

struct delta {
    double max_abs = 0;       // max |a - b| over every value
    double cos_min = 1.0;     // worst per-row cosine
    int64_t n_diff = 0;       // values whose bit patterns differ
    int64_t n_rows = 0;
};

// Row-wise comparison of two [rows, dim] blocks. n_diff counts *bit* differences, so 0 means the
// two runs produced the identical float32 image.
static delta compare(const float * a, const float * b, int64_t rows, int64_t dim) {
    delta d;
    d.n_rows = rows;
    for (int64_t r = 0; r < rows; r++) {
        double dot = 0, na = 0, nb = 0;
        for (int64_t i = 0; i < dim; i++) {
            const float x = a[r * dim + i], y = b[r * dim + i];
            const double e = fabs((double) x - (double) y);
            if (e > d.max_abs) d.max_abs = e;
            if (memcmp(&x, &y, sizeof(float)) != 0) d.n_diff++;
            dot += (double) x * y; na += (double) x * x; nb += (double) y * y;
        }
        const double den = sqrt(na) * sqrt(nb);
        const double c = den > 0 ? dot / den : 1.0;
        if (c < d.cos_min) d.cos_min = c;
    }
    return d;
}

// `exact` models must match bit for bit; the rest are gated on the cosine only.
static void report(const char * what, const delta & d, bool exact) {
    const bool ok = exact ? (d.n_diff == 0) : (d.cos_min >= 1.0 - 1e-6);
    printf("    %-28s %6lld rows  bit-diff %lld  max|d| %.3g  cos_min %.9f  %s\n",
           what, (long long) d.n_rows, (long long) d.n_diff, d.max_abs, d.cos_min,
           ok ? "OK" : "FAIL");
    if (!ok) g_fail++;
}

// ------------------------------------------------------------------------------------------------
// image models
// ------------------------------------------------------------------------------------------------
static int run_image(const std::string & gguf, const std::vector<std::string> & images, int threads, int chunk) {
    jepa_model * model = jepa_model_load(gguf.c_str(), g_verbose);
    if (!model) { printf("  cannot load %s\n", gguf.c_str()); return 1; }
    jepa_context_params cp = jepa_context_default_params();
    cp.n_threads = threads;
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) { jepa_model_free(model); return 1; }
    const bool exact = jepa_model_file_type(model) == 0;   // f32: bit-exactness is required
    const int64_t B = (int64_t) images.size();
    printf("  %s (%s, %s, %lld images, %d threads)\n", jepa_model_name(model), jepa_model_family(model),
           jepa_model_file_type_name(model), (long long) B, jepa_context_n_threads(ctx));

    // preprocess once; the batched input is the per-item inputs concatenated on the batch axis
    std::vector<float> x;
    int h = 0, w = 0;
    for (int64_t i = 0; i < B; i++) {
        int ih = 0, iw = 0;
        float * p = jepa_preprocess_image_file(model, images[i].c_str(), &ih, &iw);
        if (!p) { printf("    cannot preprocess %s\n", images[i].c_str()); g_fail++; break; }
        if (i == 0) { h = ih; w = iw; x.resize((size_t) B * 3 * h * w); }
        if (ih != h || iw != w) { printf("    %s preprocesses to %dx%d, expected %dx%d\n", images[i].c_str(), iw, ih, w, h); g_fail++; }
        else memcpy(x.data() + (size_t) i * 3 * h * w, p, (size_t) 3 * h * w * sizeof(float));
        jepa_free(p);
    }

    auto encode = [&](int max_batch, int64_t n_batch, const float * data, std::vector<float> & rows,
                      std::vector<float> & pooled_mean, std::vector<float> & pooled_cls) -> bool {
        jepa_context_set_max_batch(ctx, max_batch);
        jepa_input in = { data, (int) n_batch, 3, 1, h, w };
        jepa_output enc = { nullptr, 0, 0 };
        if (jepa_encode(ctx, &in, &enc) != 0) return false;
        rows.assign(enc.data, enc.data + (size_t) enc.n_tokens * enc.dim);
        // pooling wants one item at a time: slice the concatenated rows back apart
        const int64_t per = enc.n_tokens / n_batch;
        pooled_mean.clear();
        pooled_cls.clear();
        for (int64_t i = 0; i < n_batch; i++) {
            jepa_output one = { enc.data + (size_t) i * per * enc.dim, per, enc.dim }, p = { nullptr, 0, 0 };
            if (jepa_pool_mean(model, &one, &p) != 0) { jepa_free(enc.data); return false; }
            pooled_mean.insert(pooled_mean.end(), p.data, p.data + p.dim);
            jepa_free(p.data);
            if (jepa_model_has_cls(model)) {
                if (jepa_pool_cls(model, &one, &p) != 0) { jepa_free(enc.data); return false; }
                pooled_cls.insert(pooled_cls.end(), p.data, p.data + p.dim);
                jepa_free(p.data);
            }
        }
        jepa_free(enc.data);
        return true;
    };

    // (a) reference: one item per call, one graph per item
    std::vector<float> ref, ref_mean, ref_cls, one, one_mean, one_cls;
    const int64_t D = jepa_model_embed_dim(model);
    for (int64_t i = 0; i < B; i++) {
        if (!encode(1, 1, x.data() + (size_t) i * 3 * h * w, one, one_mean, one_cls)) {
            printf("    encode failed on item %lld\n", (long long) i); g_fail++;
            jepa_context_free(ctx); jepa_model_free(model); return 1;
        }
        ref.insert(ref.end(), one.begin(), one.end());
        ref_mean.insert(ref_mean.end(), one_mean.begin(), one_mean.end());
        ref_cls.insert(ref_cls.end(), one_cls.begin(), one_cls.end());
    }
    const int64_t n_tokens = (int64_t) ref.size() / D / B;

    struct { const char * name; int max_batch; } cases[] = {
        { "batched rows",   (int) B },
        { "chunked rows",   chunk   },
    };
    for (const auto & c : cases) {
        std::vector<float> got, got_mean, got_cls;
        if (!encode(c.max_batch, B, x.data(), got, got_mean, got_cls)) {
            printf("    %s: encode failed\n", c.name); g_fail++; continue;
        }
        if (got.size() != ref.size()) {
            printf("    %s: %zu values, expected %zu\n", c.name, got.size(), ref.size()); g_fail++; continue;
        }
        char label[64];
        snprintf(label, sizeof(label), "%s (max_batch %d)", c.name, c.max_batch);
        report(label, compare(ref.data(), got.data(), B * n_tokens, D), exact);
        report("  pooled mean", compare(ref_mean.data(), got_mean.data(), B, D), exact);
        if (!ref_cls.empty()) report("  pooled cls", compare(ref_cls.data(), got_cls.data(), B, D), exact);
    }

    jepa_context_free(ctx);
    jepa_model_free(model);
    return 0;
}

// ------------------------------------------------------------------------------------------------
// video models: n_batch = 2 against two n_batch = 1 calls
// ------------------------------------------------------------------------------------------------
static int run_video(const std::string & gguf, const std::vector<std::string> & clips, int threads, int clip_frames) {
    jepa_model * model = jepa_model_load(gguf.c_str(), g_verbose);
    if (!model) { printf("  cannot load %s\n", gguf.c_str()); return 1; }
    jepa_context_params cp = jepa_context_default_params();
    cp.n_threads = threads;
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) { jepa_model_free(model); return 1; }

    // every clip is preprocessed on its own, then stacked on the batch axis
    std::vector<float> x;
    int T = 0, h = 0, w = 0;
    const int tubelet = jepa_model_tubelet_size(model);
    for (size_t i = 0; i < clips.size(); i++) {
        npy::Array a = npy::load(clips[i]);
        if (a.shape.size() != 4 || a.shape[3] != 3 || a.dtype != "|u1") {
            printf("    %s: expected THWC uint8\n", clips[i].c_str()); g_fail++;
            jepa_context_free(ctx); jepa_model_free(model); return 1;
        }
        int nf = (int) a.shape[0];
        if (clip_frames > 0 && clip_frames < nf) nf = clip_frames;
        const int fh = (int) a.shape[1], fw = (int) a.shape[2];
        std::vector<const uint8_t *> fp;
        for (int t = 0; t < nf; t++) fp.push_back(a.u8() + (size_t) t * fh * fw * 3);
        while (tubelet > 1 && nf % tubelet != 0) { fp.push_back(fp.back()); nf++; }
        int oh = 0, ow = 0;
        float * p = jepa_preprocess_frames_rgb(model, fp.data(), nf, fh, fw, &oh, &ow);
        if (!p) { printf("    %s: preprocessing failed\n", clips[i].c_str()); g_fail++; continue; }
        if (i == 0) { T = nf; h = oh; w = ow; x.resize(clips.size() * (size_t) 3 * T * h * w); }
        if (nf != T || oh != h || ow != w) { printf("    %s: %d frames %dx%d, expected %d %dx%d\n", clips[i].c_str(), nf, ow, oh, T, w, h); g_fail++; }
        else memcpy(x.data() + i * (size_t) 3 * T * h * w, p, (size_t) 3 * T * h * w * sizeof(float));
        jepa_free(p);
    }
    printf("  %s (%s, %s, %zu clips of %d frames %dx%d, %d threads)\n", jepa_model_name(model),
           jepa_model_family(model), jepa_model_file_type_name(model), clips.size(), T, w, h,
           jepa_context_n_threads(ctx));

    auto encode = [&](int64_t n_batch, const float * data, std::vector<float> & rows) -> bool {
        jepa_input in = { data, (int) n_batch, 3, T, h, w };
        jepa_output enc = { nullptr, 0, 0 };
        if (jepa_encode(ctx, &in, &enc) != 0) return false;
        rows.assign(enc.data, enc.data + (size_t) enc.n_tokens * enc.dim);
        jepa_free(enc.data);
        return true;
    };

    std::vector<float> ref, one, got;
    for (size_t i = 0; i < clips.size(); i++) {
        if (!encode(1, x.data() + i * (size_t) 3 * T * h * w, one)) {
            printf("    encode failed on clip %zu\n", i); g_fail++;
            jepa_context_free(ctx); jepa_model_free(model); return 1;
        }
        ref.insert(ref.end(), one.begin(), one.end());
    }
    if (!encode((int64_t) clips.size(), x.data(), got) || got.size() != ref.size()) {
        printf("    n_batch=%zu encode failed or wrong size\n", clips.size()); g_fail++;
    } else {
        const int64_t D = jepa_model_embed_dim(model);
        report("n_batch=2 rows", compare(ref.data(), got.data(), (int64_t) ref.size() / D, D), true);
    }

    jepa_context_free(ctx);
    jepa_model_free(model);
    return 0;
}

// ------------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::string media, video;
    std::vector<std::pair<std::string, int>> image_models;   // model, how many images it gets
    std::vector<std::string> clips;
    int threads = 8, n_images = 8, chunk = 3, clip_frames = 0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(1); }
            return argv[++i];
        };
        if      (a == "--media")   media = next("--media");
        else if (a == "--image")   image_models.emplace_back(next("--image"), n_images);
        else if (a == "--video")   video = next("--video");
        else if (a == "--clip")    clips.push_back(next("--clip"));
        else if (a == "--clip-frames") clip_frames = atoi(next("--clip-frames"));
        else if (a == "--threads") threads = atoi(next("--threads"));
        else if (a == "--images")  n_images = atoi(next("--images"));
        else if (a == "--chunk")   chunk = atoi(next("--chunk"));
        else if (a == "--verbose") g_verbose = true;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
    }
    if (image_models.empty() && video.empty()) {
        fprintf(stderr, "usage: test-batch --media DIR --image M.gguf [--image M2.gguf] "
                        "[--video V.gguf --clip a.npy --clip b.npy] [--threads N] [--images N]\n");
        return 1;
    }

    // the fixture COCO JPEGs, in name order — the same set test-parity uses
    static const char * names[] = {
        "coco_000000000139.jpg", "coco_000000000285.jpg", "coco_000000000632.jpg", "coco_000000000724.jpg",
        "coco_000000000776.jpg", "coco_000000000785.jpg", "coco_000000039769.jpg", "coco_000000219578.jpg",
    };
    if (!image_models.empty() && media.empty()) { fprintf(stderr, "--media DIR is required for --image\n"); return 1; }
    for (const auto & m : image_models) {
        std::vector<std::string> images;
        for (int i = 0; i < m.second && i < (int) (sizeof(names) / sizeof(names[0])); i++) {
            images.push_back(media + "/" + names[i]);
        }
        if (images.empty()) { fprintf(stderr, "--images %d leaves nothing to encode\n", m.second); return 1; }
        run_image(m.first, images, threads, chunk);
    }
    if (!video.empty()) {
        if (clips.size() < 2) { fprintf(stderr, "--video needs at least two --clip files\n"); return 1; }
        run_video(video, clips, threads, clip_frames);
    }

    printf("%s\n", g_fail ? "FAILED" : "all batched outputs match the per-item path");
    return g_fail ? 1 : 0;
}
