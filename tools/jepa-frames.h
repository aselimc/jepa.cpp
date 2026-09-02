// Frames -> one preprocessed NCTHW clip, shared by jepa-embed and jepa-server.
//
// Those two take the same route from decoded RGB8 frames to the tensor jepa_encode() consumes, and
// "the server preprocesses exactly like the CLI" is a property worth having structurally rather than
// by inspection — so the route lives here, in one function, and each tool only supplies the frames.
//
// jepa-classify does NOT go through here: it keeps its own per-frame loop and its own tubelet pad
// (tools/jepa-classify.cpp). That loop agrees with this one on everything it handles, because both
// call jepa_preprocess_image_rgb per frame and concatenate the CHW planes the same way, but it is a
// separate copy — it has no still-image repeat, which only the fixed-clip-length families need and
// which a classifier is never handed. Unifying it is a change to a parity-gated path and wants its
// own before/after byte comparison.
#pragma once

#include "jepa.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace jepa_frames {

// A still image fed to a video model is a 1-frame clip, and what that should mean is per family.
// V-JEPA 2 / 2.1 decode RoPE from the ACTUAL grid and are evaluated at 16 and 64 frames alike, so a
// short clip is run as given (only the tubelet has to divide it), and V-JEPA 2.1 additionally has a
// native 1-frame tokenizer. LeVJEPA has neither: its model card prescribes repeating the frame
// along T ("video = image.unsqueeze(2).repeat(1, 1, 16, 1, 1)"), and its block-causal mask over a
// single temporal slot would make a 1-frame clip a different model rather than a cheaper one. For
// that family a still image is therefore repeated to jepa.enc.n_frames before anything else.
inline bool repeats_still_image(const std::string & family) { return family == "levjepa"; }

// Preprocess every frame on its own and concatenate the results into one NCTHW clip.
// jepa_preprocess_frames_rgb() resizes/crops/normalises each frame independently as well (it is the
// same per-frame loop, see src/preprocess.cpp), so for equal-sized frames this is bit-identical to
// one call over the whole stack; doing it per frame is what lets a clip mix differently-sized source
// images.  A video model with tubelet t needs a multiple of t frames, so the last frame is repeated
// here exactly as the HF processor does.
//
// `fp` / `fh` / `fw` are the frames and their source sizes; they are copied on, not kept.
// `name` labels the item in the notes written to `log` (pass nullptr for no notes).
// Returns a malloc'd [1, 3, T, crop, crop] buffer (free with jepa_free), or nullptr.
inline float * to_clip(const jepa_model * model, std::vector<const uint8_t *> fp,
                       std::vector<int> fh, std::vector<int> fw,
                       int tubelet, bool video_model, const char * name, FILE * log,
                       int * out_T, int * out_h, int * out_w) {
    if (fp.empty()) return nullptr;
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
        if (log) {
            fprintf(log, "note: %s: still image repeated %d times along T, which is how '%s' takes "
                         "an image\n", name, want, jepa_model_family(model));
        }
    } else if (fixed_len && T != want && log) {
        // Not an error: the graph takes any length. But this family's mask and its training both
        // assume `want` slots, so a different count is a different model, quietly.
        fprintf(log, "note: %s: %d frames through '%s', which was trained on %d — the clip is "
                     "encoded as given, but its block-causal mask then spans %d temporal slots "
                     "instead of %d\n", name, T, jepa_model_family(model), want, T, want);
    }
    if (video_model && tubelet > 1 && T % tubelet != 0 &&
        jepa_token_grid(model, T, jepa_model_img_size(model), jepa_model_img_size(model), nullptr, nullptr, nullptr) == 0) {
        const int pad = tubelet - T % tubelet;
        for (int i = 0; i < pad; i++) { fp.push_back(fp.back()); fh.push_back(fh.back()); fw.push_back(fw.back()); }
        T += pad;
        if (log) fprintf(log, "note: %s: repeated the last frame %d time(s) to reach a multiple of "
                              "the tubelet size %d\n", name, pad, tubelet);
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
            if (log) fprintf(log, "frame %d preprocesses to %dx%d but frame 0 to %dx%d\n", t, ow, oh, crop_w, crop_h);
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

// The families whose encoder takes a clip rather than independent images.
inline bool is_video_family(const std::string & f) {
    return f == "vjepa" || f == "vjepa2" || f == "vjepa2_1" || f == "levjepa";
}

}  // namespace jepa_frames
