// jepa-embed: image(s) -> feature vector(s).
//   jepa-embed -m model.gguf -i img.jpg [-i img2.jpg ...] [-o out.npy] [--pool mean|cls|lewm|none]
//              [-t threads] [--time] [--no-flash] [--kv-f32] [--repeat N] [--print-n N]
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
        "usage: %s -m model.gguf -i image.jpg [-i image2.jpg ...] [options]\n"
        "  -o out.npy        save the features as float32 .npy ([n_images, D] or [n_images*n_tokens, D] for --pool none)\n"
        "  --pool MODE       mean (patch tokens, default for models without CLS) | cls (default for CLS models)\n"
        "                    | lewm (enc.proj(CLS) world-model state) | none (all tokens)\n"
        "  -t N              threads (default: all)\n"
        "  --time            print preprocessing / encode timings\n"
        "  --repeat N        encode each image N times (timing)\n"
        "  --no-flash        naive attention (mul_mat + soft_max) instead of flash attention\n"
        "  --kv-f32 / --kv-f16  K/V dtype for flash attention (default: F32 for f32 models, F16 otherwise)\n"
        "  --print-n N       print the first N values of each vector (default 8; 0 = none)\n"
        "  --dump-input F    save the preprocessed NCHW input of the (last) image as float32 .npy\n", argv0);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    std::string model_path, out_path, pool, dump_input;
    std::vector<std::string> images;
    jepa_context_params cp = jepa_context_default_params();
    bool timing = false;
    int repeat = 1, print_n = 8;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model_path = next("-m");
        else if (a == "-i") images.push_back(next("-i"));
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
    if (model_path.empty() || images.empty()) { usage(argv[0]); return 1; }
    if (repeat < 1) repeat = 1;

    double t0 = now_ms();
    jepa_model * model = jepa_model_load(model_path.c_str(), true);
    if (!model) return 1;
    double load_ms = now_ms() - t0;
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) return 1;

    if (pool.empty()) pool = jepa_model_has_cls(model) ? "cls" : "mean";
    if (pool != "mean" && pool != "cls" && pool != "lewm" && pool != "none") {
        fprintf(stderr, "unknown --pool %s\n", pool.c_str());
        return 1;
    }
    if (pool == "cls" && !jepa_model_has_cls(model)) { fprintf(stderr, "model has no CLS token; use --pool mean\n"); return 1; }
    if (pool == "lewm" && !jepa_model_has_projector(model)) { fprintf(stderr, "model has no enc.proj projector\n"); return 1; }

    if (timing) fprintf(stderr, "model load: %.1f ms | threads: %d | flash: %s | kv: %s\n", load_ms, jepa_context_n_threads(ctx),
                        cp.use_flash_attn ? "yes" : "no",
                        cp.flash_kv == JEPA_KV_F16 ? "f16" : cp.flash_kv == JEPA_KV_F32 ? "f32" : "auto");

    std::vector<float> all;
    int64_t dim = 0, rows_per_image = 0;
    for (const std::string & img : images) {
        double tp = now_ms();
        int h = 0, w = 0;
        float * x = jepa_preprocess_image_file(model, img.c_str(), &h, &w);
        if (!x) return 1;
        double pre_ms = now_ms() - tp;
        if (!dump_input.empty()) {
            npy::save_f32(dump_input, {1, 3, h, w}, x);
            fprintf(stderr, "saved preprocessed input %s [1, 3, %d, %d]\n", dump_input.c_str(), h, w);
        }

        jepa_input in;
        in.data = x; in.n_batch = 1; in.n_chans = 3; in.n_frames = 1; in.height = h; in.width = w;
        jepa_output enc = {nullptr, 0, 0};
        double enc_ms = 0, wall_ms = 0;
        for (int r = 0; r < repeat; r++) {
            if (enc.data) jepa_free(enc.data);
            double te = now_ms();
            if (jepa_encode(ctx, &in, &enc) != 0) return 1;
            wall_ms += now_ms() - te;
            enc_ms += jepa_context_last_compute_ms(ctx);
        }
        enc_ms /= repeat; wall_ms /= repeat;

        jepa_output feat = {nullptr, 0, 0};
        if (pool == "mean")      { if (jepa_pool_mean(model, &enc, &feat) != 0) return 1; }
        else if (pool == "cls")  { if (jepa_pool_cls(model, &enc, &feat) != 0) return 1; }
        else if (pool == "lewm") { if (jepa_lewm_project(ctx, &enc, &feat) != 0) return 1; }
        else                     { feat = enc; enc.data = nullptr; }

        dim = feat.dim;
        rows_per_image = feat.n_tokens;
        const size_t n = (size_t) feat.n_tokens * feat.dim;
        all.insert(all.end(), feat.data, feat.data + n);

        double norm = 0;
        for (size_t i = 0; i < n; i++) norm += (double) feat.data[i] * feat.data[i];
        printf("%s: %s [%lld x %lld] |x|=%.4f", img.c_str(), pool.c_str(), (long long) feat.n_tokens, (long long) feat.dim, sqrt(norm));
        if (print_n > 0) {
            printf(" [");
            for (int i = 0; i < print_n && i < (int) n; i++) printf("%s%.5f", i ? ", " : "", feat.data[i]);
            printf("%s]", (int) n > print_n ? ", ..." : "");
        }
        printf("\n");
        if (timing) {
            fprintf(stderr, "  %dx%d -> %dx%d preprocess %.1f ms | encode %.1f ms (graph compute %.1f ms, %d threads%s)\n",
                    w, h, w, h, pre_ms, wall_ms, enc_ms, jepa_context_n_threads(ctx), repeat > 1 ? ", mean of repeats" : "");
        }
        if (feat.data) jepa_free(feat.data);
        if (enc.data) jepa_free(enc.data);
        jepa_free(x);
    }

    if (!out_path.empty()) {
        std::vector<int64_t> shape;
        if (rows_per_image == 1) shape = { (int64_t) images.size(), dim };
        else                     shape = { (int64_t) images.size() * rows_per_image, dim };
        npy::save_f32(out_path, shape, all.data());
        fprintf(stderr, "saved %s [%lld x %lld]\n", out_path.c_str(), (long long) shape[0], (long long) shape[1]);
    }
    jepa_context_free(ctx);
    jepa_model_free(model);
    return 0;
}
