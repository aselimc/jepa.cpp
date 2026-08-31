// jepa-embed-clips — batch driver for the video accuracy benchmark (scripts/bench_accuracy_video.py).
//
//   jepa-embed-clips -m model.gguf -l list.txt -o feats.npy [--pool mean|cls] [-t 32]
//                    [--logits logits.npy] [--json stats.json] [--progress N]
//
// `list.txt` holds one THWC-uint8 .npy clip path per line (scripts/video_frames.py writes them).
// Every clip is encoded with ONE model load and ONE jepa_context, and the pooled vectors are
// written as a single [n_clips, D] float32 .npy in list order; `--logits` additionally runs the
// attentive-pool head and writes [n_clips, n_classes] raw logits.
//
// Why this exists: tools/jepa-embed takes a single `--frames-npy` per process, so a 405-clip sweep
// would re-load and re-mmap the weights 405 times (~0.3 s each for V-JEPA 2 ViT-L f16, ~1 s for
// f32) and emit one .npy per clip.  tools/ is owned by another agent, so rather than adding a
// `--frames-list` mode there this driver links libjepa directly (include/jepa.h) and is built
// out-of-tree; see docs/accuracy-video.md.  The per-clip numerics are identical to
// `jepa-embed --frames-npy ... --pool mean` — same preprocessing call, same jepa_encode, same
// jepa_pool_mean — which docs/accuracy-video.md verifies on a sample clip.
#include "jepa.h"
#include "npy.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    std::string model_path, list_path, out_path, logits_path, json_path, pool = "mean";
    jepa_context_params cp = jepa_context_default_params();
    int progress = 25;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")         model_path  = next("-m");
        else if (a == "-l")         list_path   = next("-l");
        else if (a == "-o")         out_path    = next("-o");
        else if (a == "--logits")   logits_path = next("--logits");
        else if (a == "--json")     json_path   = next("--json");
        else if (a == "--pool")     pool        = next("--pool");
        else if (a == "-t")         cp.n_threads = atoi(next("-t"));
        else if (a == "--progress") progress    = atoi(next("--progress"));
        else if (a == "--no-flash") cp.use_flash_attn = false;
        else if (a == "--kv-f32")   cp.flash_kv = JEPA_KV_F32;
        else if (a == "--kv-f16")   cp.flash_kv = JEPA_KV_F16;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
    }
    if (model_path.empty() || list_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -l list.txt -o feats.npy [--pool mean|cls] [-t N] "
                        "[--logits l.npy] [--json stats.json]\n", argv[0]);
        return 1;
    }

    std::vector<std::string> clips;
    { std::ifstream f(list_path);
      if (!f) { fprintf(stderr, "cannot open %s\n", list_path.c_str()); return 1; }
      std::string line;
      while (std::getline(f, line)) {
          while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
          if (!line.empty()) clips.push_back(line);
      } }
    if (clips.empty()) { fprintf(stderr, "%s is empty\n", list_path.c_str()); return 1; }

    const double t_load = now_ms();
    jepa_model * model = jepa_model_load(model_path.c_str(), true);
    if (!model) return 1;
    const double load_ms = now_ms() - t_load;
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) { jepa_model_free(model); return 1; }
    const bool want_logits = !logits_path.empty();
    if (want_logits && !jepa_model_has_head(model)) {
        fprintf(stderr, "%s has no classification head\n", model_path.c_str());
        jepa_context_free(ctx); jepa_model_free(model); return 1;
    }
    const int tubelet = jepa_model_tubelet_size(model);

    std::vector<float> feats, logit_rows;
    int64_t dim = 0, n_classes = 0;
    double pre_ms = 0, enc_ms = 0, head_ms = 0;
    const double t0 = now_ms();
    for (size_t c = 0; c < clips.size(); c++) {
        npy::Array a = npy::load(clips[c]);
        if (a.shape.size() != 4 || a.shape[3] != 3 || a.dtype != "|u1") {
            fprintf(stderr, "%s: expected THWC uint8, got %zu dims dtype %s\n", clips[c].c_str(), a.shape.size(), a.dtype.c_str());
            return 1;
        }
        int T = (int) a.shape[0];
        const int h = (int) a.shape[1], w = (int) a.shape[2];
        std::vector<const uint8_t *> fp;
        for (int t = 0; t < T; t++) fp.push_back(a.bytes.data() + (size_t) t * h * w * 3);
        // a video model with tubelet t needs a multiple of t frames (the HF processor repeats the last)
        while (tubelet > 1 && T % tubelet != 0) { fp.push_back(fp.back()); T++; }

        double t = now_ms();
        int oh = 0, ow = 0;
        float * x = jepa_preprocess_frames_rgb(model, fp.data(), T, h, w, &oh, &ow);
        if (!x) { fprintf(stderr, "%s: preprocessing failed\n", clips[c].c_str()); return 1; }
        pre_ms += now_ms() - t;

        jepa_input in;
        in.data = x; in.n_batch = 1; in.n_chans = 3; in.n_frames = T; in.height = oh; in.width = ow;
        jepa_output enc = {nullptr, 0, 0}, feat = {nullptr, 0, 0}, lg = {nullptr, 0, 0};
        t = now_ms();
        if (jepa_encode(ctx, &in, &enc) != 0) { fprintf(stderr, "%s: encode failed\n", clips[c].c_str()); return 1; }
        enc_ms += now_ms() - t;
        if (pool == "cls") { if (jepa_pool_cls(model, &enc, &feat) != 0) return 1; }
        else               { if (jepa_pool_mean(model, &enc, &feat) != 0) return 1; }
        dim = feat.dim;
        feats.insert(feats.end(), feat.data, feat.data + feat.dim);
        if (want_logits) {
            t = now_ms();
            if (jepa_head_ex(ctx, &enc, nullptr, &lg) != 0) { fprintf(stderr, "%s: head failed\n", clips[c].c_str()); return 1; }
            head_ms += now_ms() - t;
            n_classes = lg.dim;
            logit_rows.insert(logit_rows.end(), lg.data, lg.data + lg.dim);
            jepa_free(lg.data);
        }
        jepa_free(feat.data);
        jepa_free(enc.data);
        jepa_free(x);
        if (progress > 0 && ((c + 1) % (size_t) progress == 0 || c + 1 == clips.size())) {
            fprintf(stderr, "  %zu/%zu  %.1fs\n", c + 1, clips.size(), (now_ms() - t0) / 1000.0);
        }
    }
    const double wall_s = (now_ms() - t0) / 1000.0;

    npy::save_f32(out_path, {(int64_t) clips.size(), dim}, feats.data());
    if (want_logits) npy::save_f32(logits_path, {(int64_t) clips.size(), n_classes}, logit_rows.data());
    fprintf(stderr, "%zu clips -> %s [%zu x %lld] in %.1fs (%.3f s/clip, %.2f clips/s)\n",
            clips.size(), out_path.c_str(), clips.size(), (long long) dim, wall_s,
            wall_s / clips.size(), clips.size() / wall_s);

    if (!json_path.empty()) {
        std::ofstream j(json_path);
        j << "{\n  \"model\": \"" << model_path << "\",\n"
          << "  \"file_type\": \"" << jepa_model_file_type_name(model) << "\",\n"
          << "  \"family\": \"" << jepa_model_family(model) << "\",\n"
          << "  \"weights_mib\": " << (double) jepa_model_n_bytes(model) / (1024.0 * 1024.0) << ",\n"
          << "  \"n_clips\": " << clips.size() << ",\n  \"dim\": " << dim << ",\n"
          << "  \"threads\": " << jepa_context_n_threads(ctx) << ",\n"
          << "  \"model_load_s\": " << load_ms / 1000.0 << ",\n"
          << "  \"preprocess_s\": " << pre_ms / 1000.0 << ",\n"
          << "  \"encode_s\": " << enc_ms / 1000.0 << ",\n"
          << "  \"head_s\": " << head_ms / 1000.0 << ",\n"
          << "  \"wall_s\": " << wall_s << ",\n"
          << "  \"clips_per_s\": " << clips.size() / wall_s << "\n}\n";
    }
    jepa_context_free(ctx);
    jepa_model_free(model);
    return 0;
}
