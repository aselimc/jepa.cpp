// test-video: the tools' native video ingest against the PyTorch-side frame sampler.
//
//   test-video --media DIR --ref REF_DIR [--ref REF_DIR ...] [--quiet]
//
// Every reference dump written by scripts/dump_reference.py stores, per video sample, the exact THWC
// uint8 frames PyAV decoded and scripts/video_frames.py's sampler picked, together with the clip's
// total frame count, the sampled indices and the source file name. This test decodes the same source
// file through tools/video-decode.cpp — the `ffmpeg` subprocess `jepa-embed --video` uses — and
// requires the result to be **identical**: same frame count, same indices, same pixels, byte for
// byte. That identity is what lets `--video clip.mp4` stand in for `--frames-npy` anywhere, and it
// is the only place where a libswscale or frame-rate-handling difference between ffmpeg and PyAV
// could hide.
//
// Exits 0 with a "skipped" line when `ffmpeg` is not installed or no video sample has its source
// file, so it can live in the normal ctest set.
#include "npy.h"
#include "json.hpp"
#include "video-decode.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static int g_fail = 0, g_pass = 0;

static bool file_exists(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return (bool) f;
}

static std::string join(const std::vector<int> & v, size_t max_n = 6) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < max_n; i++) s += (i ? "," : "") + std::to_string(v[i]);
    if (v.size() > max_n) s += ",...";
    return s;
}

// One video sample: decode its source and compare everything the manifest recorded about it.
static void check_sample(const std::string & media_dir, const std::string & ref_dir, const json & s, bool quiet) {
    const std::string name = s.value("name", "?");
    const std::string clip = media_dir + "/" + s.value("media", "");
    const int want = s.value("frames", 0);
    const std::string label = ref_dir.substr(ref_dir.find_last_of('/') + 1) + "/" + name;

    jepa_video::clip got;
    std::string err;
    if (!jepa_video::decode(clip, want, got, err)) {
        printf("FAIL %-38s %s\n", label.c_str(), err.c_str());
        g_fail++;
        return;
    }

    std::vector<std::string> bad;
    const int total = s.value("n_frames_total", 0);
    if (got.n_frames_total != total) {
        bad.push_back("decoded " + std::to_string(got.n_frames_total) + " frames, PyAV decoded " +
                      std::to_string(total));
    }
    const std::vector<int> idx = s.value("frame_indices", std::vector<int>{});
    if (got.frame_indices != idx) {
        bad.push_back("sampled [" + join(got.frame_indices) + "] against [" + join(idx) + "]");
    }
    const std::vector<int> hw = s.value("frame_size_hw", std::vector<int>{});
    if (hw.size() == 2 && (got.height != hw[0] || got.width != hw[1])) {
        bad.push_back("frame size " + std::to_string(got.width) + "x" + std::to_string(got.height) +
                      " against " + std::to_string(hw[1]) + "x" + std::to_string(hw[0]));
    }
    const double fps = s.value("fps", 0.0);
    if (fps > 0 && std::fabs(got.fps - fps) > 1e-6 * fps) {
        bad.push_back("fps " + std::to_string(got.fps) + " against " + std::to_string(fps));
    }

    // the pixels themselves
    size_t n_diff = 0, first_diff = 0;
    int max_diff = 0;
    const std::string npy_path = ref_dir + "/" + s["tensors"]["frames_u8"].value("file", "");
    try {
        const npy::Array ref = npy::load(npy_path);
        if (ref.shape.size() != 4 || ref.dtype != "|u1") {
            bad.push_back(npy_path + " is not a THWC uint8 array");
        } else if (ref.bytes.size() != got.data.size()) {
            bad.push_back("decoded " + std::to_string(got.data.size()) + " bytes against " +
                          std::to_string(ref.bytes.size()));
        } else {
            const uint8_t * a = ref.u8();
            for (size_t i = 0; i < got.data.size(); i++) {
                const int d = std::abs((int) a[i] - (int) got.data[i]);
                if (d) {
                    if (!n_diff) first_diff = i;
                    n_diff++;
                    if (d > max_diff) max_diff = d;
                }
            }
            if (n_diff) {
                char buf[192];
                snprintf(buf, sizeof(buf), "%zu of %zu bytes differ (max %d, first at %zu)",
                         n_diff, got.data.size(), max_diff, first_diff);
                bad.push_back(buf);
            }
        }
    } catch (const std::exception & e) {
        bad.push_back(std::string("cannot read ") + npy_path + ": " + e.what());
    }

    if (bad.empty()) {
        g_pass++;
        if (!quiet) {
            printf("PASS %-38s %d/%d frames %dx%d, %zu bytes identical\n", label.c_str(), want,
                   got.n_frames_total, got.width, got.height, got.data.size());
        }
    } else {
        g_fail++;
        printf("FAIL %-38s %s\n", label.c_str(), bad[0].c_str());
        for (size_t i = 1; i < bad.size(); i++) printf("     %-38s %s\n", "", bad[i].c_str());
    }
}

int main(int argc, char ** argv) {
    std::string media_dir;
    std::vector<std::string> ref_dirs;
    bool quiet = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--media") media_dir = next("--media");
        else if (a == "--ref") ref_dirs.push_back(next("--ref"));
        else if (a == "--quiet") quiet = true;
        else { fprintf(stderr, "usage: %s --media DIR --ref REF_DIR [--ref ...] [--quiet]\n", argv[0]); return 2; }
    }
    if (media_dir.empty() || ref_dirs.empty()) {
        fprintf(stderr, "usage: %s --media DIR --ref REF_DIR [--ref ...] [--quiet]\n", argv[0]);
        return 2;
    }

    std::string why;
    if (!jepa_video::have_ffmpeg(&why)) {
        printf("skipped: %s\n", why.c_str());
        return 0;
    }

    // The sampler on its own, against the numbers numpy produces for the awkward shapes: a clip
    // shorter than the request (indices repeat), a single frame, an exact fit, and an even divisor
    // where np.round's ties-to-even differs from C's round().
    struct { int total, n; std::vector<int> want; } cases[] = {
        {  5, 16, {0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4} },
        {  1, 16, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} },
        { 16, 16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} },
        {300, 16, {0, 20, 40, 60, 80, 100, 120, 140, 159, 179, 199, 219, 239, 259, 279, 299} },
        {  9, 17, {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8} },   // 0.5 ties -> even
        {  7,  1, {0} },
    };
    for (const auto & c : cases) {
        const std::vector<int> got = jepa_video::sample_indices(c.total, c.n);
        const bool ok = got == c.want;
        if (!ok || !quiet) {
            const std::string want = ok ? std::string() : " != [" + join(c.want, 20) + "]";
            printf("%-4s sample_indices(%3d, %2d) -> [%s]%s\n", ok ? "PASS" : "FAIL", c.total, c.n,
                   join(got, 20).c_str(), want.c_str());
        }
        ok ? g_pass++ : g_fail++;
    }

    int n_video = 0;
    for (const std::string & ref_dir : ref_dirs) {
        std::ifstream f(ref_dir + "/manifest.json");
        if (!f) { printf("skipped %s: no manifest.json\n", ref_dir.c_str()); continue; }
        json manifest;
        f >> manifest;
        for (const json & s : manifest["samples"]) {
            // A video sample is one the reference dump sampled frames out of. Still-image samples
            // also carry frames_u8 (one frame repeated), but no frame_indices / n_frames_total, and
            // the older dumps have no "path" field at all — hence the shape test rather than a
            // `path == "video"` one.
            if (s.value("path", "") == "image") continue;
            if (!s.contains("tensors") || !s["tensors"].contains("frames_u8")) continue;
            if (!s.contains("media") || !s.contains("frame_indices") || !s.contains("n_frames_total")) continue;
            if (!file_exists(media_dir + "/" + s.value("media", ""))) continue;   // clip not in the checkout
            n_video++;
            check_sample(media_dir, ref_dir, s, quiet);
        }
    }
    if (n_video == 0) {
        printf("skipped: no video sample in %zu reference dump(s) has its source clip under %s\n",
               ref_dirs.size(), media_dir.c_str());
        return 0;
    }
    printf("%d passed, %d failed (%d video sample(s))\n", g_pass, g_fail, n_video);
    return g_fail ? 1 : 0;
}
