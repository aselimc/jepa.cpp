// jepa-classify: video clip -> top-k labels of an attentive-pool classifier (V-JEPA 2 SSv2 ...).
//
//   jepa-classify -m vjepa2-vitl-fpc16-256-ssv2-f16.gguf --frames-npy clip.npy [-k 5]
//   jepa-classify -m ssv2.gguf -i f0.jpg -i f1.jpg ... [-k 5] [--json out.json]
//
// `--frames-npy` takes a THWC uint8 .npy (exactly what tests/fixtures/ref/<model>/<sample>.frames_u8.npy
// holds); `-i` images are used as the frames of one clip, in the order given. The frames go through the
// model's jepa.pre.* pipeline (per-frame shortest-edge resize + centre crop + ImageNet normalisation),
// then the encoder and the attentive pooler + classifier (jepa_head_ex), and the logits are softmaxed.
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
        "usage: %s -m model.gguf (--frames-npy clip.npy | -i frame.jpg [-i ...]) [options]\n"
        "  -k N              show the top N classes (default 5)\n"
        "  -t N              threads (default: all)\n"
        "  --time            print preprocessing / encode / head timings\n"
        "  --repeat N        run the encoder N times (timing)\n"
        "  --no-flash        naive attention instead of flash attention\n"
        "  --kv-f32 / --kv-f16  K/V dtype for flash attention (default: F32 for f32 models, F16 otherwise)\n"
        "  --logits out.npy  save the raw logits as float32 .npy\n"
        "  --dump-input F    save the preprocessed NCTHW clip as float32 .npy\n", argv0);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    std::string model_path, frames_npy, logits_out, dump_input;
    std::vector<std::string> images;
    jepa_context_params cp = jepa_context_default_params();
    int topk = 5, repeat = 1;
    bool timing = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model_path = next("-m");
        else if (a == "--frames-npy") frames_npy = next("--frames-npy");
        else if (a == "-i") images.push_back(next("-i"));
        else if (a == "-k") topk = atoi(next("-k"));
        else if (a == "-t") cp.n_threads = atoi(next("-t"));
        else if (a == "--time") timing = true;
        else if (a == "--repeat") repeat = atoi(next("--repeat"));
        else if (a == "--no-flash") cp.use_flash_attn = false;
        else if (a == "--kv-f32") cp.flash_kv = JEPA_KV_F32;
        else if (a == "--kv-f16") cp.flash_kv = JEPA_KV_F16;
        else if (a == "--logits") logits_out = next("--logits");
        else if (a == "--dump-input") dump_input = next("--dump-input");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (model_path.empty() || (frames_npy.empty() && images.empty())) { usage(argv[0]); return 1; }
    if (repeat < 1) repeat = 1;

    jepa_model * model = jepa_model_load(model_path.c_str(), timing);
    if (!model) return 1;
    if (!jepa_model_has_head(model)) {
        fprintf(stderr, "%s has no classification head (jepa.head.kind = none) — use jepa-embed\n", model_path.c_str());
        return 1;
    }
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) return 1;

    // ---- frames -> normalized NCTHW clip
    double t0 = now_ms();
    int n_frames = 0, fh = 0, fw = 0, oh = 0, ow = 0;
    float * clip = nullptr;
    std::vector<uint8_t> pixels;                 // THWC uint8
    std::vector<const uint8_t *> ptr;
    if (!frames_npy.empty()) {
        npy::Array a = npy::load(frames_npy);
        if (a.shape.size() != 4 || a.shape[3] != 3 || a.dtype != "|u1") {
            fprintf(stderr, "%s: expected a THWC uint8 array, got %zu dims dtype %s\n", frames_npy.c_str(), a.shape.size(), a.dtype.c_str());
            return 1;
        }
        n_frames = (int) a.shape[0]; fh = (int) a.shape[1]; fw = (int) a.shape[2];
        pixels.swap(a.bytes);
    } else {
        for (size_t i = 0; i < images.size(); i++) {
            int h = 0, w = 0;
            uint8_t * rgb = jepa_load_image_rgb(images[i].c_str(), &h, &w);
            if (!rgb) return 1;
            if (i == 0) { fh = h; fw = w; }
            else if (h != fh || w != fw) { fprintf(stderr, "frame %zu is %dx%d, the first frame is %dx%d — all frames must match\n", i, w, h, fw, fh); return 1; }
            pixels.insert(pixels.end(), rgb, rgb + (size_t) h * w * 3);
            jepa_free(rgb);
        }
        n_frames = (int) images.size();
    }
    // Models with tubelet t need a multiple of t frames; the HF processor repeats frames for that.
    const int tub = jepa_model_tubelet_size(model);
    if (tub > 1 && n_frames % tub != 0) {
        const int pad = tub - n_frames % tub;
        const size_t frame_bytes = (size_t) fh * fw * 3;
        const std::vector<uint8_t> last(pixels.end() - frame_bytes, pixels.end());   // copy: insert may reallocate
        for (int i = 0; i < pad; i++) pixels.insert(pixels.end(), last.begin(), last.end());
        fprintf(stderr, "note: repeated the last frame %d time(s) to reach a multiple of the tubelet size %d\n", pad, tub);
        n_frames += pad;
    }
    ptr.resize(n_frames);
    for (int t = 0; t < n_frames; t++) ptr[t] = pixels.data() + (size_t) t * fh * fw * 3;
    clip = jepa_preprocess_frames_rgb(model, ptr.data(), n_frames, fh, fw, &oh, &ow);
    if (!clip) { fprintf(stderr, "preprocessing failed\n"); return 1; }
    const double pre_ms = now_ms() - t0;
    if (!dump_input.empty()) {
        npy::save_f32(dump_input, {1, 3, n_frames, oh, ow}, clip);
        fprintf(stderr, "saved preprocessed clip %s [1, 3, %d, %d, %d]\n", dump_input.c_str(), n_frames, oh, ow);
    }

    jepa_input in;
    in.data = clip; in.n_batch = 1; in.n_chans = 3; in.n_frames = n_frames; in.height = oh; in.width = ow;
    jepa_output enc = {nullptr, 0, 0};
    double enc_ms = 0;
    for (int r = 0; r < repeat; r++) {
        if (enc.data) jepa_free(enc.data);
        if (jepa_encode(ctx, &in, &enc) != 0) return 1;
        enc_ms += jepa_context_last_compute_ms(ctx);
    }
    enc_ms /= repeat;

    jepa_output pooled = {nullptr, 0, 0}, logits = {nullptr, 0, 0};
    const double th = now_ms();
    if (jepa_head_ex(ctx, &enc, &pooled, &logits) != 0) return 1;
    const double head_ms = now_ms() - th;

    const int n_classes = (int) logits.dim;
    std::vector<float> probs((size_t) n_classes);
    jepa_softmax(logits.data, n_classes, probs.data());
    if (topk < 1) topk = 1;
    std::vector<int32_t> idx((size_t) topk);
    const int k = jepa_top_k(logits.data, n_classes, topk, idx.data());

    printf("%s: %d frames %dx%d -> %lld tokens, %d classes\n", model_path.c_str(), n_frames, fw, fh,
           (long long) enc.n_tokens, n_classes);
    for (int i = 0; i < k; i++) {
        const char * label = jepa_model_label(model, idx[i]);
        printf("  %2d. %6.2f%%  [%3d] %s\n", i + 1, 100.0 * probs[idx[i]], idx[i], label ? label : "(no label)");
    }
    if (timing) {
        fprintf(stderr, "preprocess %.1f ms | encoder %.1f ms (%d threads%s) | head %.1f ms | %.0f tokens/s\n",
                pre_ms, enc_ms, jepa_context_n_threads(ctx), repeat > 1 ? ", mean of repeats" : "", head_ms,
                1000.0 * (double) enc.n_tokens / enc_ms);
    }
    if (!logits_out.empty()) {
        npy::save_f32(logits_out, {(int64_t) n_classes}, logits.data);
        fprintf(stderr, "saved %s [%d]\n", logits_out.c_str(), n_classes);
    }

    jepa_free(pooled.data);
    jepa_free(logits.data);
    jepa_free(enc.data);
    jepa_free(clip);
    jepa_context_free(ctx);
    jepa_model_free(model);
    return 0;
}
