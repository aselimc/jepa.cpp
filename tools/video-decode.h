// Native video ingest for the jepa.cpp tools: a clip file -> THWC uint8 frames, sampled exactly the
// way scripts/video_frames.py samples them, without Python.
//
// The decode runs `ffmpeg` as a subprocess and reads rgb24 frames off its stdout, so the tools need
// no build-time dependency and a checkout without ffmpeg loses nothing but this one flag (the
// `.npy` routes keep working).  The frames ffmpeg writes are bit-identical to the ones PyAV hands
// scripts/video_frames.py — same libswscale conversion, no scaling on either side — which is what
// lets `--video` and `--frames-npy` produce the same tensor for the same clip; see
// docs/architecture.md "Preprocessing" and tests/test-video.cpp.
//
// $JEPA_FFMPEG / $JEPA_FFPROBE override the binaries.
//
// The engine itself never sees a container: the C API stays frames-in (include/jepa.h), and this
// unit lives in tools/ for that reason.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jepa_video {

// One decoded and sampled clip.
struct clip {
    int n_frames = 0;                    // frames in `data` (== the requested count)
    int height = 0, width = 0;           // every frame has the same size (no scaling is done)
    std::vector<uint8_t> data;           // THWC RGB8, n_frames * height * width * 3 bytes
    int n_frames_total = 0;              // frames the decoder produced for the whole clip
    std::vector<int> frame_indices;      // the sampled indices into [0, n_frames_total)
    double fps = 0;                      // container frame rate, 0 when unknown
    double decode_s = 0;                 // wall time of the probe + decode

    const uint8_t * frame(int t) const { return data.data() + (size_t) t * height * width * 3; }
};

// `n` frames uniformly over `total`, endpoints included — the sampler of
// scripts/video_frames.py and scripts/dump_reference.py:
//     idx = np.linspace(0, total - 1, n).round().astype(int)
// numpy rounds halves to even and pins the last sample to `total - 1`; both are reproduced here, so
// the indices match for every (total, n), including a clip shorter than `n` (indices then repeat).
std::vector<int> sample_indices(int total, int n);

// Is an `ffmpeg` (and `ffprobe`) usable? On false, `why` gets a message naming the binaries and how
// to install them, ready to print.
bool have_ffmpeg(std::string * why);

// Decode `path` and keep `n_frames` frames sampled with sample_indices(). Returns false and fills
// `err` on any failure (no ffmpeg, unreadable file, no video stream, short/garbled output).
bool decode(const std::string & path, int n_frames, clip & out, std::string & err);

// The exact ffmpeg command line decode() runs, for --help text and error messages.
std::string decode_command(const std::string & path);

} // namespace jepa_video
