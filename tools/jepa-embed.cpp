// jepa-embed: image(s) or a video clip -> feature vector(s).
//   jepa-embed -m model.gguf -i img.jpg [-i img2.jpg ...] [-o out.npy] [--pool mean|cls|lewm|none]
//              [-t threads] [--time] [--no-flash] [--kv-f32] [--repeat N] [--print-n N]
//   jepa-embed -m vjepa2.gguf --frames-npy clip.npy            # THWC uint8 frames = one clip
//   jepa-embed -m vjepa2.gguf --as-video -i f0.jpg -i f1.jpg   # the images are frames of one clip
//
// Images are encoded one by one (each is a 1-frame item); with --frames-npy / --as-video — and by
// default when several images are given to a video model — the frames form ONE clip that goes
// through the tubelet tokenizer and 3-D RoPE in a single graph.  Clip frames may have DIFFERENT
// source sizes: every frame is preprocessed on its own (resize + centre crop land them all on the
// model's crop x crop) and the CHW planes are then concatenated into the NCTHW clip.
#include "jepa.h"
#include "npy.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf (-i image.jpg [-i image2.jpg ...] | --frames-npy clip.npy) [options]\n"
        "  --frames-npy F    THWC uint8 .npy of frames (e.g. tests/fixtures/ref/<m>/<sample>.frames_u8.npy) = one clip\n"
        "  --as-video        treat the -i images as the frames of one clip (default for video models with >1 image)\n"
        "  --as-images       encode every -i image separately, even for a video model\n"
        "  -o out.npy        save the features as float32 .npy ([n_items, D] or [n_items*n_tokens, D] for --pool none)\n"
        "  --pool MODE       mean (patch tokens, default for models without CLS) | cls (default for CLS models)\n"
        "                    | lewm (enc.proj(CLS) world-model state) | none (all tokens)\n"
        "  -t N              threads (default: all)\n"
        "  --time            print preprocessing / encode timings\n"
        "  --repeat N        encode each item N times (timing)\n"
        "  --no-flash        naive attention (mul_mat + soft_max) instead of flash attention\n"
        "  --kv-f32 / --kv-f16  K/V dtype for flash attention (default: F32 for f32 models, F16 otherwise)\n"
        "  --print-n N       print the first N values of each vector (default 8; 0 = none)\n"
        "  --dump-input F    save the preprocessed NCTHW input of the (last) item as float32 .npy\n", argv0);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One item to encode: n RGB8 HWC frames, each with its own size (an image is n = 1).
struct item {
    std::string name;
    std::vector<std::vector<uint8_t>> frames;   // per frame: HWC RGB8
    std::vector<int> h, w;                      // per frame source size
    int n() const { return (int) frames.size(); }
    void add(const uint8_t * rgb, int ih, int iw) {
        frames.emplace_back(rgb, rgb + (size_t) ih * iw * 3);
        h.push_back(ih);
        w.push_back(iw);
    }
};

// Preprocess every frame of an item on its own and concatenate the results into one NCTHW clip.
// jepa_preprocess_frames_rgb() resizes/crops/normalises each frame independently as well, so for
// equal-sized frames this is bit-identical to one call over the whole stack; doing it per frame is
// what lets a clip mix differently-sized source images (`--as-video -i a.jpg -i b.jpg`).
// Returns a malloc'd [1, 3, T, crop, crop] buffer (free with jepa_free), or nullptr.
static float * preprocess_item(const jepa_model * model, const item & it, int * out_h, int * out_w) {
    const int T = it.n();
    if (T <= 0) return nullptr;
    float * out = nullptr;
    int crop_h = 0, crop_w = 0;
    for (int t = 0; t < T; t++) {
        int oh = 0, ow = 0;
        float * f = jepa_preprocess_image_rgb(model, it.frames[t].data(), it.h[t], it.w[t], &oh, &ow);
        if (!f) { jepa_free(out); return nullptr; }
        if (!out) {
            crop_h = oh; crop_w = ow;
            out = (float *) malloc((size_t) 3 * T * crop_h * crop_w * sizeof(float));
            if (!out) { jepa_free(f); return nullptr; }
        } else if (oh != crop_h || ow != crop_w) {
            fprintf(stderr, "frame %d preprocesses to %dx%d but frame 0 to %dx%d\n", t, ow, oh, crop_w, crop_h);
            jepa_free(f); jepa_free(out);
            return nullptr;
        }
        const size_t plane = (size_t) crop_h * crop_w;
        for (int c = 0; c < 3; c++) {
            memcpy(out + ((size_t) c * T + t) * plane, f + (size_t) c * plane, plane * sizeof(float));
        }
        jepa_free(f);
    }
    if (out_h) *out_h = crop_h;
    if (out_w) *out_w = crop_w;
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path, out_path, pool, dump_input, frames_npy;
    std::vector<std::string> images;
    jepa_context_params cp = jepa_context_default_params();
    bool timing = false, as_video = false, as_images = false;
    int repeat = 1, print_n = 8;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model_path = next("-m");
        else if (a == "-i") images.push_back(next("-i"));
        else if (a == "--frames-npy") frames_npy = next("--frames-npy");
        else if (a == "--as-video") as_video = true;
        else if (a == "--as-images") as_images = true;
        else if (a == "-o") out_path = next("-o");
        else if (a == "--pool") pool = next("--pool");
        else if (a == "-t") cp.n_threads = atoi(next("-t"));
        else if (a == "--time") timing = true;
        else if (a == "--repeat") repeat = atoi(next("--repeat"));
        else if (a == "--no-flash") cp.use_flash_attn = false;
        else if (a == "--kv-f32") cp.flash_kv = JEPA_KV_F32;
        else if (a == "--kv-f16") cp.flash_kv = JEPA_KV_F16;
        else if (a == "--print-n") print_n = atoi(next("--print-n"));
        else if (a == "--dump-input") dump_input = next("--dump-input");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (model_path.empty() || (images.empty() && frames_npy.empty())) { usage(argv[0]); return 1; }
    if (repeat < 1) repeat = 1;

    double t0 = now_ms();
    jepa_model * model = jepa_model_load(model_path.c_str(), true);
    if (!model) return 1;
    double load_ms = now_ms() - t0;
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) { jepa_model_free(model); return 1; }

    // every exit below goes through this: the model and context (hundreds of MiB of weights) and
    // the per-item buffers are released on the error paths too
    float * x = nullptr;
    jepa_output enc = {nullptr, 0, 0}, feat = {nullptr, 0, 0};
    auto done = [&](int rc) {
        if (feat.data && feat.data != enc.data) jepa_free(feat.data);
        if (enc.data) jepa_free(enc.data);
        if (x) jepa_free(x);
        jepa_context_free(ctx);
        jepa_model_free(model);
        return rc;
    };

    if (pool.empty()) pool = jepa_model_has_cls(model) ? "cls" : "mean";
    if (pool != "mean" && pool != "cls" && pool != "lewm" && pool != "none") {
        fprintf(stderr, "unknown --pool %s\n", pool.c_str());
        return done(1);
    }
    if (pool == "cls" && !jepa_model_has_cls(model)) { fprintf(stderr, "model has no CLS token; use --pool mean\n"); return done(1); }
    if (pool == "lewm" && !jepa_model_has_projector(model)) { fprintf(stderr, "model has no enc.proj projector\n"); return done(1); }

    const std::string family = jepa_model_family(model);
    const bool video_model = family == "vjepa" || family == "vjepa2" || family == "vjepa2_1";
    const int tubelet = jepa_model_tubelet_size(model);
    const bool clip_mode = !frames_npy.empty() || as_video || (video_model && images.size() > 1 && !as_images);

    // ---- collect the items (an image is a 1-frame clip)
    std::vector<item> items;
    if (!frames_npy.empty()) {
        npy::Array a = npy::load(frames_npy);
        if (a.shape.size() != 4 || a.shape[3] != 3 || a.dtype != "|u1") {
            fprintf(stderr, "%s: expected a THWC uint8 array, got %zu dims dtype %s\n", frames_npy.c_str(), a.shape.size(), a.dtype.c_str());
            return done(1);
        }
        item it;
        it.name = frames_npy;
        const int nf = (int) a.shape[0], fh = (int) a.shape[1], fw = (int) a.shape[2];
        for (int t = 0; t < nf; t++) it.add(a.bytes.data() + (size_t) t * fh * fw * 3, fh, fw);
        items.push_back(std::move(it));
    }
    if (!images.empty()) {
        item clip;
        clip.name = images[0] + (images.size() > 1 ? " (+" + std::to_string(images.size() - 1) + " frames)" : "");
        for (size_t i = 0; i < images.size(); i++) {
            int h = 0, w = 0;
            uint8_t * rgb = jepa_load_image_rgb(images[i].c_str(), &h, &w);
            if (!rgb) return done(1);
            if (clip_mode) {
                // frames of a clip may differ in size: every frame is preprocessed on its own below
                clip.add(rgb, h, w);
            } else {
                item it;
                it.name = images[i];
                it.add(rgb, h, w);
                items.push_back(std::move(it));
            }
            jepa_free(rgb);
        }
        if (clip_mode) items.push_back(std::move(clip));
    }

    // A video model with tubelet t needs a multiple of t frames (the HF processor repeats frames).
    for (item & it : items) {
        if (video_model && tubelet > 1 && it.n() % tubelet != 0 && jepa_token_grid(model, it.n(), jepa_model_img_size(model), jepa_model_img_size(model), nullptr, nullptr, nullptr) == 0) {
            const int pad = tubelet - it.n() % tubelet;
            for (int i = 0; i < pad; i++) {
                const std::vector<uint8_t> last = it.frames.back();   // copy: add() may reallocate
                it.add(last.data(), it.h.back(), it.w.back());
            }
            fprintf(stderr, "note: %s: repeated the last frame %d time(s) to reach a multiple of the tubelet size %d\n", it.name.c_str(), pad, tubelet);
        }
    }

    if (timing) fprintf(stderr, "model load: %.1f ms | threads: %d | flash: %s | kv: %s\n", load_ms, jepa_context_n_threads(ctx),
                        cp.use_flash_attn ? "yes" : "no",
                        cp.flash_kv == JEPA_KV_F16 ? "f16" : cp.flash_kv == JEPA_KV_F32 ? "f32" : "auto");

    std::vector<float> all;
    int64_t dim = 0, rows_per_item = 0;
    for (const item & it : items) {
        double tp = now_ms();
        int h = 0, w = 0;
        x = preprocess_item(model, it, &h, &w);
        if (!x) return done(1);
        double pre_ms = now_ms() - tp;
        if (!dump_input.empty()) {
            npy::save_f32(dump_input, {1, 3, it.n(), h, w}, x);
            fprintf(stderr, "saved preprocessed input %s [1, 3, %d, %d, %d]\n", dump_input.c_str(), it.n(), h, w);
        }

        jepa_input in;
        in.data = x; in.n_batch = 1; in.n_chans = 3; in.n_frames = it.n(); in.height = h; in.width = w;
        double enc_ms = 0, wall_ms = 0;
        for (int r = 0; r < repeat; r++) {
            if (enc.data) { jepa_free(enc.data); enc.data = nullptr; }
            double te = now_ms();
            if (jepa_encode(ctx, &in, &enc) != 0) return done(1);
            wall_ms += now_ms() - te;
            enc_ms += jepa_context_last_compute_ms(ctx);
        }
        enc_ms /= repeat; wall_ms /= repeat;
        const int64_t n_tokens = enc.n_tokens;

        if (pool == "mean")      { if (jepa_pool_mean(model, &enc, &feat) != 0) return done(1); }
        else if (pool == "cls")  { if (jepa_pool_cls(model, &enc, &feat) != 0) return done(1); }
        else if (pool == "lewm") { if (jepa_lewm_project(ctx, &enc, &feat) != 0) return done(1); }
        else                     { feat = enc; enc.data = nullptr; }

        dim = feat.dim;
        rows_per_item = feat.n_tokens;
        const size_t n = (size_t) feat.n_tokens * feat.dim;
        all.insert(all.end(), feat.data, feat.data + n);

        double norm = 0;
        for (size_t i = 0; i < n; i++) norm += (double) feat.data[i] * feat.data[i];
        printf("%s: %s [%lld x %lld] |x|=%.4f", it.name.c_str(), pool.c_str(), (long long) feat.n_tokens, (long long) feat.dim, sqrt(norm));
        if (print_n > 0) {
            printf(" [");
            for (int i = 0; i < print_n && i < (int) n; i++) printf("%s%.5f", i ? ", " : "", feat.data[i]);
            printf("%s]", (int) n > print_n ? ", ..." : "");
        }
        printf("\n");
        if (timing) {
            char src[64];
            bool same = true;
            for (int t = 1; t < it.n(); t++) same &= it.h[t] == it.h[0] && it.w[t] == it.w[0];
            if (same) snprintf(src, sizeof(src), "%dx%d", it.w[0], it.h[0]);
            else      snprintf(src, sizeof(src), "%dx%d..(mixed)", it.w[0], it.h[0]);
            fprintf(stderr, "  %d frame(s) %s -> %dx%d, %lld tokens | preprocess %.1f ms | encode %.1f ms "
                            "(graph compute %.1f ms, %.0f tokens/s, %d threads%s)\n",
                    it.n(), src, w, h, (long long) n_tokens, pre_ms, wall_ms, enc_ms,
                    enc_ms > 0 ? 1000.0 * (double) n_tokens / enc_ms : 0.0,
                    jepa_context_n_threads(ctx), repeat > 1 ? ", mean of repeats" : "");
        }
        if (feat.data && feat.data != enc.data) jepa_free(feat.data);
        feat.data = nullptr;
        if (enc.data) jepa_free(enc.data);
        enc.data = nullptr;
        jepa_free(x);
        x = nullptr;
    }

    if (!out_path.empty()) {
        std::vector<int64_t> shape;
        if (rows_per_item == 1) shape = { (int64_t) items.size(), dim };
        else                    shape = { (int64_t) items.size() * rows_per_item, dim };
        npy::save_f32(out_path, shape, all.data());
        fprintf(stderr, "saved %s [%lld x %lld]\n", out_path.c_str(), (long long) shape[0], (long long) shape[1]);
    }
    return done(0);
}
