// jepa-embed: image(s) or video clip(s) -> feature vector(s).
//   jepa-embed -m model.gguf -i img.jpg [-i img2.jpg ...] [-o out.npy] [--pool mean|cls|lewm|none]
//              [--batch B | --no-batch] [-t threads] [--time] [--no-flash] [--kv-f32] [--repeat N]
//              [--print-n N]
//   jepa-embed -m vjepa2.gguf --frames-npy clip.npy            # THWC uint8 frames = one clip
//   jepa-embed -m vjepa2.gguf --frames-list clips.txt -o feats.npy [--logits l.npy] [--json s.json]
//   jepa-embed -m vjepa2.gguf --as-video -i f0.jpg -i f1.jpg   # the images are frames of one clip
//   jepa-embed -m levjepa-vitl16.gguf -i cat.jpg --pool cls    # a still image -> repeated to 16 frames
//
// Images are encoded --batch at a time: the items go through ONE ggml graph on the batch dimension,
// which is bit-identical to encoding them one by one (tests/test-batch.cpp) and 1.5-2.6x faster on
// the small models.  With --frames-npy / --frames-list / --as-video — and by default when several
// images are given to a video model — the frames form ONE clip that goes through the tubelet
// tokenizer and 3-D RoPE in a single graph (that path is one clip per graph regardless of --batch).
// Clip frames may have DIFFERENT source sizes: every frame is preprocessed on its own (resize +
// centre crop land them all on the model's crop x crop) and the CHW planes are then concatenated
// into the NCTHW clip.
//
// --frames-list reads one THWC-uint8 .npy clip path per line (scripts/video_frames.py writes them),
// encodes every clip with one model load and one context, and writes the pooled vectors as a single
// [n_clips, D] float32 .npy in list order — what scripts/bench_accuracy_video.py consumes.
#include "jepa.h"
#include "jepa-args.h"
#include "npy.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf (-i image.jpg [-i image2.jpg ...] | --frames-npy clip.npy | --frames-list list.txt) [options]\n"
        "  --frames-npy F    THWC uint8 .npy of frames (e.g. tests/fixtures/ref/<m>/<sample>.frames_u8.npy) = one clip\n"
        "  --frames-list F   text file with one such .npy path per line; each line is one clip\n"
        "  --as-video        treat the -i images as the frames of one clip (default for video models with >1 image)\n"
        "  --as-images       encode every -i image separately, even for a video model\n"
        "  -o out.npy        save the features as float32 .npy ([n_items, D] or [n_items*n_tokens, D] for --pool none)\n"
        "  --pool MODE       mean (patch tokens, default for models without CLS) | cls (default for CLS models)\n"
        "                    | lewm (enc.proj(CLS) world-model state) | none (all tokens)\n"
        "  --batch B         image items per encoder graph (default 32, or $JEPA_MAX_BATCH; video models default to 1 —\n"
        "                    clips never batch in the library; $JEPA_MAX_GRAPH_MIB caps one graph's activations, default 8192)\n"
        "  --no-batch        same as --batch 1: one graph per item, the pre-batching path\n"
        "  --logits F        also run the attentive-pool head and save [n_items, n_classes] raw logits\n"
        "  --json F          write a stats JSON (timings, throughput) for benchmark drivers\n"
        "  --progress N      print progress every N items to stderr (0 = off; default 0, 25 with --frames-list,\n"
        "                    which also replaces the per-item stdout line)\n"
        "  -t N              threads (default: all)\n"
        JEPA_GPU_USAGE
        "  --time            print preprocessing / encode timings\n"
        "  --repeat N        encode each batch N times (timing)\n"
        "  --no-flash        naive attention (mul_mat + soft_max) instead of flash attention\n"
        "  --kv-f32 / --kv-f16  K/V dtype for flash attention (default: F32 for f32 models, F16 otherwise)\n"
        "  --print-n N       print the first N values of each vector (default 8; 0 = none)\n"
        "  --dump-input F    save the preprocessed NCTHW input of the last batch as float32 .npy\n", argv0);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One item to encode: n RGB8 HWC frames, each with its own size (an image is n = 1). Items that came
// from a .npy keep only the path — the frames are read when the item is preprocessed and dropped
// again, so a 400-clip --frames-list holds one clip in memory, not four hundred.
struct item {
    std::string name;
    std::string npy;                            // THWC uint8 .npy, read on demand
    std::vector<std::vector<uint8_t>> frames;   // or: per frame, HWC RGB8
    std::vector<int> h, w;                      // per frame source size
    int n() const { return (int) frames.size(); }
    void add(const uint8_t * rgb, int ih, int iw) {
        frames.emplace_back(rgb, rgb + (size_t) ih * iw * 3);
        h.push_back(ih);
        w.push_back(iw);
    }
};

// The item's frames as (pointer, h, w). For a .npy item `hold` owns the bytes the pointers refer to.
static bool item_frames(const item & it, npy::Array & hold, std::vector<const uint8_t *> & fp,
                        std::vector<int> & fh, std::vector<int> & fw) {
    fp.clear(); fh.clear(); fw.clear();
    if (!it.npy.empty()) {
        try {
            hold = npy::load(it.npy);
        } catch (const std::exception & e) {
            fprintf(stderr, "error: cannot load %s: %s\n", it.npy.c_str(), e.what());
            return false;
        }
        if (hold.shape.size() != 4 || hold.shape[3] != 3 || hold.dtype != "|u1") {
            fprintf(stderr, "%s: expected a THWC uint8 array, got %zu dims dtype %s\n",
                    it.npy.c_str(), hold.shape.size(), hold.dtype.c_str());
            return false;
        }
        const int nf = (int) hold.shape[0], h = (int) hold.shape[1], w = (int) hold.shape[2];
        for (int t = 0; t < nf; t++) {
            fp.push_back(hold.u8() + (size_t) t * h * w * 3);
            fh.push_back(h);
            fw.push_back(w);
        }
    } else {
        for (size_t t = 0; t < it.frames.size(); t++) {
            fp.push_back(it.frames[t].data());
            fh.push_back(it.h[t]);
            fw.push_back(it.w[t]);
        }
    }
    return !fp.empty();
}

// Preprocess every frame of an item on its own and concatenate the results into one NCTHW clip.
// jepa_preprocess_frames_rgb() resizes/crops/normalises each frame independently as well (it is the
// same per-frame loop, see src/preprocess.cpp), so for equal-sized frames this is bit-identical to
// one call over the whole stack; doing it per frame is what lets a clip mix differently-sized source
// images (`--as-video -i a.jpg -i b.jpg`).  A video model with tubelet t needs a multiple of t
// frames, so the last frame is repeated here exactly as the HF processor does.
// Returns a malloc'd [1, 3, T, crop, crop] buffer (free with jepa_free), or nullptr.
//
// A still image fed to a video model is a 1-frame clip, and what that should mean is per family.
// V-JEPA 2 / 2.1 decode RoPE from the ACTUAL grid and are evaluated at 16 and 64 frames alike, so a
// short clip is run as given (only the tubelet has to divide it), and V-JEPA 2.1 additionally has a
// native 1-frame tokenizer. LeVJEPA has neither: its model card prescribes repeating the frame
// along T ("video = image.unsqueeze(2).repeat(1, 1, 16, 1, 1)"), and its block-causal mask over a
// single temporal slot would make a 1-frame clip a different model rather than a cheaper one. For
// that family a still image is therefore repeated to jepa.enc.n_frames before anything else.
static bool repeats_still_image(const std::string & family) { return family == "levjepa"; }

static float * preprocess_item(const jepa_model * model, const item & it, int tubelet, bool video_model,
                               int * out_T, int * out_h, int * out_w, bool warn_pad) {
    npy::Array hold;
    std::vector<const uint8_t *> fp;
    std::vector<int> fh, fw;
    if (!item_frames(it, hold, fp, fh, fw)) return nullptr;
    int T = (int) fp.size();
    const int want = jepa_model_n_frames(model);
    const bool fixed_len = video_model && want > 1 && repeats_still_image(jepa_model_family(model));
    // A still image on a fixed-clip-length family becomes `want` copies of one frame. The copies are
    // identical, so only the first is preprocessed and the result is memcpy'd into the other slots
    // (`repeat_from_first` below); resizing the same JPEG 16 times cost 41.8 ms against 15.0.
    int repeat_from_first = 0;
    if (fixed_len && T == 1) {
        repeat_from_first = want;
        T = want;
        if (warn_pad) {
            fprintf(stderr, "note: %s: still image repeated %d times along T, which is how '%s' takes "
                            "an image\n", it.name.c_str(), want, jepa_model_family(model));
        }
    } else if (fixed_len && T != want && warn_pad) {
        // Not an error: the graph takes any length. But this family's mask and its training both
        // assume `want` slots, so a different count is a different model, quietly.
        fprintf(stderr, "note: %s: %d frames through '%s', which was trained on %d — the clip is "
                        "encoded as given, but its block-causal mask then spans %d temporal slots "
                        "instead of %d\n", it.name.c_str(), T, jepa_model_family(model), want, T, want);
    }
    if (video_model && tubelet > 1 && T % tubelet != 0 &&
        jepa_token_grid(model, T, jepa_model_img_size(model), jepa_model_img_size(model), nullptr, nullptr, nullptr) == 0) {
        const int pad = tubelet - T % tubelet;
        for (int i = 0; i < pad; i++) { fp.push_back(fp.back()); fh.push_back(fh.back()); fw.push_back(fw.back()); }
        T += pad;
        if (warn_pad) fprintf(stderr, "note: %s: repeated the last frame %d time(s) to reach a multiple of "
                                      "the tubelet size %d\n", it.name.c_str(), pad, tubelet);
    }

    float * out = nullptr;
    int crop_h = 0, crop_w = 0;
    const int n_pre = repeat_from_first ? 1 : T;    // distinct frames that actually need preprocessing
    for (int t = 0; t < n_pre; t++) {
        int oh = 0, ow = 0;
        float * f = jepa_preprocess_image_rgb(model, fp[t], fh[t], fw[t], &oh, &ow);
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
    if (repeat_from_first && out) {
        const size_t plane = (size_t) crop_h * crop_w;
        for (int c = 0; c < 3; c++) {
            const float * src = out + (size_t) c * T * plane;
            for (int t = 1; t < T; t++) memcpy(out + ((size_t) c * T + t) * plane, src, plane * sizeof(float));
        }
    }
    if (out_T) *out_T = T;
    if (out_h) *out_h = crop_h;
    if (out_w) *out_w = crop_w;
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path, out_path, pool, dump_input, frames_npy, frames_list, logits_path, json_path;
    std::vector<std::string> images;
    jepa_context_params cp = jepa_context_default_params();
    jepa_model_params   mp = jepa_model_default_params();
    mp.verbose = true;
    bool timing = false, as_video = false, as_images = false;
    int repeat = 1, print_n = 8, batch = 0, progress = -1;   // batch 0 = whatever the context defaults to
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model_path = next("-m");
        else if (a == "-i") images.push_back(next("-i"));
        else if (a == "--frames-npy") frames_npy = next("--frames-npy");
        else if (a == "--frames-list" || a == "-l") frames_list = next("--frames-list");
        else if (a == "--as-video") as_video = true;
        else if (a == "--as-images") as_images = true;
        else if (a == "-o") out_path = next("-o");
        else if (a == "--pool") pool = next("--pool");
        else if (a == "--batch") batch = atoi(next("--batch"));
        else if (a == "--no-batch") batch = 1;
        else if (a == "--logits") logits_path = next("--logits");
        else if (a == "--json") json_path = next("--json");
        else if (a == "--progress") progress = atoi(next("--progress"));
        else if (a == "-t") cp.n_threads = atoi(next("-t"));
        else if (a == "--time") timing = true;
        else if (a == "--repeat") repeat = atoi(next("--repeat"));
        else if (jepa_arg_gpu(argc, argv, i, mp.device)) {}
        else if (a == "--no-flash") cp.use_flash_attn = false;
        else if (a == "--kv-f32") cp.flash_kv = JEPA_KV_F32;
        else if (a == "--kv-f16") cp.flash_kv = JEPA_KV_F16;
        else if (a == "--print-n") print_n = atoi(next("--print-n"));
        else if (a == "--dump-input") dump_input = next("--dump-input");
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (model_path.empty() || (images.empty() && frames_npy.empty() && frames_list.empty())) { usage(argv[0]); return 1; }
    if (repeat < 1) repeat = 1;
    if (progress < 0) progress = frames_list.empty() ? 0 : 25;

    const double t_load = now_ms();
    jepa_model * model = jepa_model_load_ex(model_path.c_str(), &mp);
    if (!model) return 1;
    const double load_ms = now_ms() - t_load;
    jepa_context * ctx = jepa_context_new(model, cp);
    if (!ctx) { jepa_model_free(model); return 1; }
    // an explicit --batch / --no-batch overrides the library default (32) and $JEPA_MAX_BATCH;
    // without one the tool follows whatever the context resolved to
    const bool batch_explicit = batch > 0;   // --batch / --no-batch on the command line
    if (batch > 0) jepa_context_set_max_batch(ctx, batch);
    batch = jepa_context_max_batch(ctx);

    // every exit below goes through this: the model and context (hundreds of MiB of weights) and
    // the per-batch buffers are released on the error paths too
    jepa_output enc = {nullptr, 0, 0}, feat = {nullptr, 0, 0};   // feat never aliases enc
    auto done = [&](int rc) {
        if (feat.data) jepa_free(feat.data);
        if (enc.data) jepa_free(enc.data);
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
    const bool want_logits = !logits_path.empty();
    if (want_logits && !jepa_model_has_head(model)) {
        fprintf(stderr, "%s has no classification head\n", model_path.c_str());
        return done(1);
    }

    // a --frames-list sweep prints progress instead of one line per clip (what the old
    // scripts/jepa_embed_clips driver did); -i / --frames-npy keep the per-item line
    const bool quiet_items = !frames_list.empty();

    const std::string family = jepa_model_family(model);
    const bool video_model = family == "vjepa" || family == "vjepa2" || family == "vjepa2_1" || family == "levjepa";
    const int tubelet = jepa_model_tubelet_size(model);
    const bool clip_mode = !frames_npy.empty() || as_video || (video_model && images.size() > 1 && !as_images);
    // The library never batches video clips (one clip per graph), so grouping clip items only
    // inflates the tool's working set — measured 3.9x peak RSS for +16 % wall on 16-frame clips.
    // Default to one clip per group unless the user explicitly asked for a batch.
    if (video_model && !batch_explicit && batch > 1) { jepa_context_set_max_batch(ctx, 1); batch = 1; }

    // ---- collect the items (an image is a 1-frame clip)
    std::vector<item> items;
    if (!frames_npy.empty()) {
        item it;
        it.name = frames_npy;
        it.npy  = frames_npy;
        items.push_back(std::move(it));
    }
    if (!frames_list.empty()) {
        std::ifstream f(frames_list);
        if (!f) { fprintf(stderr, "cannot open %s\n", frames_list.c_str()); return done(1); }
        std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (line.empty()) continue;
            item it;
            it.name = line;
            it.npy  = line;
            items.push_back(std::move(it));
        }
        if (items.empty()) { fprintf(stderr, "%s is empty\n", frames_list.c_str()); return done(1); }
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

    if (timing) fprintf(stderr, "model load: %.1f ms | threads: %d | flash: %s | kv: %s | batch: %d\n", load_ms,
                        jepa_context_n_threads(ctx), cp.use_flash_attn ? "yes" : "no",
                        cp.flash_kv == JEPA_KV_F16 ? "f16" : cp.flash_kv == JEPA_KV_F32 ? "f32" : "auto", batch);

    std::vector<float> all, logit_rows;
    int64_t dim = 0, rows_per_item = 0, n_classes = 0;
    double pre_ms_total = 0, enc_ms_total = 0, head_ms_total = 0;
    const double t_all = now_ms();

    // Items are encoded in groups of at most `batch` consecutive items that share a shape (frame
    // count and preprocessed crop). One jepa_encode call per group: n_batch = group size, and for
    // the image families that is a single ggml graph.
    size_t g0 = 0;
    while (g0 < items.size()) {
        std::vector<float> xb;
        int gT = 0, gh = 0, gw = 0;
        size_t g1 = g0;
        double pre_ms = 0;
        while (g1 < items.size() && (int) (g1 - g0) < batch) {
            const double tp = now_ms();
            int T = 0, h = 0, w = 0;
            float * p = preprocess_item(model, items[g1], tubelet, video_model, &T, &h, &w, true);
            if (!p) return done(1);
            pre_ms += now_ms() - tp;
            if (g1 == g0) { gT = T; gh = h; gw = w; }
            else if (T != gT || h != gh || w != gw) {
                // a differently shaped item cannot share the graph: close this group, redo that item
                jepa_free(p);
                break;
            }
            xb.insert(xb.end(), p, p + (size_t) 3 * gT * gh * gw);
            jepa_free(p);
            g1++;
        }
        const int64_t nb = (int64_t) (g1 - g0);
        pre_ms_total += pre_ms;
        if (!dump_input.empty()) {
            npy::save_f32(dump_input, {nb, 3, gT, gh, gw}, xb.data());
            fprintf(stderr, "saved preprocessed input %s [%lld, 3, %d, %d, %d]\n", dump_input.c_str(),
                    (long long) nb, gT, gh, gw);
        }

        jepa_input in;
        in.data = xb.data(); in.n_batch = (int) nb; in.n_chans = 3; in.n_frames = gT; in.height = gh; in.width = gw;
        double enc_ms = 0, wall_ms = 0;
        for (int r = 0; r < repeat; r++) {
            if (enc.data) { jepa_free(enc.data); enc.data = nullptr; }
            const double te = now_ms();
            if (jepa_encode(ctx, &in, &enc) != 0) return done(1);
            wall_ms += now_ms() - te;
            enc_ms += jepa_context_last_compute_ms(ctx);
        }
        enc_ms /= repeat; wall_ms /= repeat;
        enc_ms_total += wall_ms;
        const int64_t n_tokens = enc.n_tokens / nb;      // per item

        for (int64_t i = 0; i < nb; i++) {
            const item & it = items[g0 + i];
            jepa_output one = { enc.data + (size_t) i * n_tokens * enc.dim, n_tokens, enc.dim };
            const float * frow = one.data;               // --pool none: the item's rows, not owned
            if (pool == "mean")      { if (jepa_pool_mean(model, &one, &feat) != 0) return done(1); frow = feat.data; }
            else if (pool == "cls")  { if (jepa_pool_cls(model, &one, &feat) != 0) return done(1); frow = feat.data; }
            else if (pool == "lewm") { if (jepa_lewm_project(ctx, &one, &feat) != 0) return done(1); frow = feat.data; }

            dim           = feat.data ? feat.dim      : one.dim;
            rows_per_item = feat.data ? feat.n_tokens : one.n_tokens;
            const size_t n = (size_t) rows_per_item * dim;
            all.insert(all.end(), frow, frow + n);

            if (want_logits) {
                const double th = now_ms();
                jepa_output lg = { nullptr, 0, 0 };
                if (jepa_head_ex(ctx, &one, nullptr, &lg) != 0) { fprintf(stderr, "%s: head failed\n", it.name.c_str()); return done(1); }
                head_ms_total += now_ms() - th;
                n_classes = lg.dim;
                logit_rows.insert(logit_rows.end(), lg.data, lg.data + lg.dim);
                jepa_free(lg.data);
            }

            if (!quiet_items) {
                double norm = 0;
                for (size_t k = 0; k < n; k++) norm += (double) frow[k] * frow[k];
                printf("%s: %s [%lld x %lld] |x|=%.4f", it.name.c_str(), pool.c_str(),
                       (long long) rows_per_item, (long long) dim, sqrt(norm));
                if (print_n > 0) {
                    printf(" [");
                    for (int k = 0; k < print_n && k < (int) n; k++) printf("%s%.5f", k ? ", " : "", frow[k]);
                    printf("%s]", (int) n > print_n ? ", ..." : "");
                }
                printf("\n");
            }
            if (feat.data) { jepa_free(feat.data); feat.data = nullptr; }
        }

        if (timing) {
            const item & it = items[g0];
            char src[64];
            bool same = true;
            for (size_t t = 1; t < it.h.size(); t++) same &= it.h[t] == it.h[0] && it.w[t] == it.w[0];
            if (!it.h.empty() && same) snprintf(src, sizeof(src), "%dx%d", it.w[0], it.h[0]);
            else if (!it.h.empty())    snprintf(src, sizeof(src), "%dx%d..(mixed)", it.w[0], it.h[0]);
            else                       snprintf(src, sizeof(src), "npy");
            if (nb == 1) {
                fprintf(stderr, "  %d frame(s) %s -> %dx%d, %lld tokens | preprocess %.1f ms | encode %.1f ms "
                                "(graph compute %.1f ms, %.0f tokens/s, %d threads%s)\n",
                        gT, src, gw, gh, (long long) n_tokens, pre_ms, wall_ms, enc_ms,
                        enc_ms > 0 ? 1000.0 * (double) n_tokens / enc_ms : 0.0,
                        jepa_context_n_threads(ctx), repeat > 1 ? ", mean of repeats" : "");
            } else {
                // last_batch < nb means the call did not fit one graph and was split (memory cap)
                const int per_graph = jepa_context_last_batch(ctx);
                char split[48] = "";
                if (per_graph > 0 && per_graph < nb) snprintf(split, sizeof(split), ", %d per graph", per_graph);
                fprintf(stderr, "  batch of %lld x %d frame(s) %s -> %dx%d, %lld tokens each | preprocess %.1f ms | "
                                "encode %.1f ms (graph compute %.1f ms = %.1f ms/item, %.0f tokens/s, %d threads%s%s)\n",
                        (long long) nb, gT, src, gw, gh, (long long) n_tokens, pre_ms, wall_ms, enc_ms,
                        enc_ms / (double) nb, enc_ms > 0 ? 1000.0 * (double) (nb * n_tokens) / enc_ms : 0.0,
                        jepa_context_n_threads(ctx), split, repeat > 1 ? ", mean of repeats" : "");
            }
        }
        if (enc.data) { jepa_free(enc.data); enc.data = nullptr; }
        g0 = g1;
        if (progress > 0 && (g0 % (size_t) progress < (size_t) batch || g0 == items.size())) {
            fprintf(stderr, "  %zu/%zu  %.1fs\n", g0, items.size(), (now_ms() - t_all) / 1000.0);
        }
    }
    const double wall_s = (now_ms() - t_all) / 1000.0;

    if (!out_path.empty()) {
        std::vector<int64_t> shape;
        if (rows_per_item == 1) shape = { (int64_t) items.size(), dim };
        else                    shape = { (int64_t) items.size() * rows_per_item, dim };
        npy::save_f32(out_path, shape, all.data());
        fprintf(stderr, "saved %s [%lld x %lld]\n", out_path.c_str(), (long long) shape[0], (long long) shape[1]);
    }
    if (want_logits) {
        npy::save_f32(logits_path, {(int64_t) items.size(), n_classes}, logit_rows.data());
        fprintf(stderr, "saved %s [%zu x %lld]\n", logits_path.c_str(), items.size(), (long long) n_classes);
    }
    if (progress > 0 || !json_path.empty()) {
        fprintf(stderr, "%zu items -> %s in %.1fs (%.3f s/item, %.2f items/s)\n", items.size(),
                out_path.empty() ? "(no output)" : out_path.c_str(), wall_s,
                wall_s / (double) items.size(), (double) items.size() / wall_s);
    }
    if (!json_path.empty()) {
        // field names kept as scripts/bench_accuracy_video.py reads them (n_clips == n_items here)
        std::ofstream j(json_path);
        j << "{\n  \"model\": \"" << model_path << "\",\n"
          << "  \"file_type\": \"" << jepa_model_file_type_name(model) << "\",\n"
          << "  \"family\": \"" << jepa_model_family(model) << "\",\n"
          << "  \"weights_mib\": " << (double) jepa_model_n_bytes(model) / (1024.0 * 1024.0) << ",\n"
          << "  \"n_clips\": " << items.size() << ",\n  \"dim\": " << dim << ",\n"
          << "  \"threads\": " << jepa_context_n_threads(ctx) << ",\n"
          << "  \"batch\": " << batch << ",\n"
          << "  \"model_load_s\": " << load_ms / 1000.0 << ",\n"
          << "  \"preprocess_s\": " << pre_ms_total / 1000.0 << ",\n"
          << "  \"encode_s\": " << enc_ms_total / 1000.0 << ",\n"
          << "  \"head_s\": " << head_ms_total / 1000.0 << ",\n"
          << "  \"wall_s\": " << wall_s << ",\n"
          << "  \"clips_per_s\": " << (double) items.size() / wall_s << "\n}\n";
    }
    return done(0);
}
