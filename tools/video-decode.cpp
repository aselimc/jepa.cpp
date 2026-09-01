#include "video-decode.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#  include <io.h>
#  define JEPA_POPEN  _popen
#  define JEPA_PCLOSE _pclose
#  define JEPA_QUIET  " > NUL 2>&1"
#else
#  include <sys/wait.h>
#  define JEPA_POPEN  popen
#  define JEPA_PCLOSE pclose
#  define JEPA_QUIET  " > /dev/null 2>&1"
#endif

namespace jepa_video {
namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char * env_or(const char * name, const char * fallback) {
    const char * v = getenv(name);
    return v && *v ? v : fallback;
}

std::string ffmpeg_bin()  { return env_or("JEPA_FFMPEG",  "ffmpeg"); }

// $JEPA_FFPROBE, else the ffprobe sitting next to whatever $JEPA_FFMPEG points at (so a single
// override of a non-PATH install picks up both), else "ffprobe".
std::string ffprobe_bin() {
    if (const char * v = getenv("JEPA_FFPROBE")) { if (*v) return v; }
    const std::string ff = ffmpeg_bin();
    const std::string suffix = "ffmpeg";
    if (ff.size() >= suffix.size() && ff.compare(ff.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return ff.substr(0, ff.size() - suffix.size()) + "ffprobe";
    }
    return "ffprobe";
}

// Quote one argument for the shell popen() hands the command to. POSIX sh: wrap in single quotes and
// close/escape/reopen for every embedded quote, which is safe for every byte a path can hold.
std::string shq(const std::string & s) {
#ifdef _WIN32
    return "\"" + s + "\"";     // cmd.exe: no single quotes, and " is not a legal path character
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    return out + "'";
#endif
}

// Run `cmd`, capture stdout into `out` (stderr is left attached to ours: ffmpeg's -v error messages
// are the useful half of a failure). Returns the exit status, or -1 if the child could not run.
int run_capture(const std::string & cmd, std::string * out) {
    FILE * p = JEPA_POPEN(cmd.c_str(), "r");
    if (!p) return -1;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p)) > 0) { if (out) out->append(buf, n); }
    const int st = JEPA_PCLOSE(p);
#ifdef _WIN32
    return st;
#else
    if (st == -1) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128;
#endif
}

// ffmpeg >= 5.1 spells "give me every decoded frame, do not duplicate or drop any" as
// `-fps_mode passthrough`; before that it was `-vsync 0`. Without one of the two, ffmpeg's default
// constant-frame-rate conversion silently duplicates frames of a variable-rate clip (measured: a
// 5-frame SSv2 .webm comes out as 59 frames), which would sample entirely different pixels than
// PyAV. Decided once per process with a 16x16 null source, which costs ~20 ms and asks the actual
// binary rather than parsing its version string.
const char * passthrough_flag() {
    static const char * const flag = [] {
        const std::string probe = ffmpeg_bin() +
            " -nostdin -v quiet -f lavfi -i nullsrc=s=16x16:d=0.1 -fps_mode passthrough -f null -" JEPA_QUIET;
        return run_capture(probe, nullptr) == 0 ? "-fps_mode passthrough" : "-vsync 0";
    }();
    return flag;
}

// "30000/1001" -> 29.97; "0/0" and anything unparsable -> 0.
double parse_rate(const std::string & s) {
    const size_t slash = s.find('/');
    if (slash == std::string::npos) return atof(s.c_str());
    const double num = atof(s.substr(0, slash).c_str());
    const double den = atof(s.substr(slash + 1).c_str());
    return den > 0 ? num / den : 0.0;
}

// value of `key=value` in ffprobe's default output, or "" when absent
std::string kv(const std::string & text, const char * key) {
    const std::string k = std::string(key) + "=";
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        if (eol == std::string::npos) eol = text.size();
        if (text.compare(pos, k.size(), k) == 0) {
            std::string v = text.substr(pos + k.size(), eol - pos - k.size());
            while (!v.empty() && (v.back() == '\r' || v.back() == ' ')) v.pop_back();
            return v;
        }
        pos = eol + 1;
    }
    return "";
}

// numpy's np.round: halves go to the even neighbour (C's round() sends them away from zero).
int round_half_even(double v) {
    const double f = std::floor(v);
    const double d = v - f;
    const long long i = (long long) f;
    if (d > 0.5) return (int) (i + 1);
    if (d < 0.5) return (int) i;
    return (int) (i % 2 == 0 ? i : i + 1);
}

struct probe_result {
    int width = 0, height = 0;
    int n_frames = 0;       // 0 = the container did not say
    double fps = 0;
};

bool probe(const std::string & path, probe_result & pr, std::string & err) {
    // (ffprobe has no -nostdin: it never reads stdin, and rejects the option outright)
    const std::string base = ffprobe_bin() + " -v error -select_streams v:0 -show_entries ";
    std::string text;
    const std::string cmd = base + "stream=width,height,nb_frames,avg_frame_rate -of default=nw=1 " + shq(path);
    const int st = run_capture(cmd, &text);
    if (st != 0) {
        err = "ffprobe failed on " + path + " (exit " + std::to_string(st) + ")";
        return false;
    }
    pr.width  = atoi(kv(text, "width").c_str());
    pr.height = atoi(kv(text, "height").c_str());
    pr.fps    = parse_rate(kv(text, "avg_frame_rate"));
    const std::string nb = kv(text, "nb_frames");
    pr.n_frames = nb == "N/A" ? 0 : atoi(nb.c_str());
    if (pr.width <= 0 || pr.height <= 0) {
        err = path + ": no video stream (ffprobe reported no frame size)";
        return false;
    }
    // ffprobe reads these out of the container, i.e. out of a file nobody here wrote. decode_pass
    // sizes its buffers as width*height*3 per kept frame, so an 8K x 8K claim is a 1.5 GiB
    // allocation from a header field. 16384 is twice the largest real format (8K UHD).
    const int MAX_EDGE = 16384;
    if (pr.width > MAX_EDGE || pr.height > MAX_EDGE) {
        err = path + ": the container reports " + std::to_string(pr.width) + "x" + std::to_string(pr.height) +
              " frames, past the " + std::to_string(MAX_EDGE) + "-pixel edge this decoder accepts";
        return false;
    }
    if (pr.n_frames <= 0) {
        // Matroska/WebM and any stream-copied container carry no frame count: count them the only
        // exact way there is, by decoding. Cheap for the clip sizes these models take (40-60 ms for
        // a 111-frame SSv2 .webm) and the count then agrees with PyAV's `len(list(c.decode(s)))`.
        std::string cnt;
        if (run_capture(base + "stream=nb_read_frames -count_frames -of default=nw=1:nk=1 " + shq(path), &cnt) != 0) {
            err = "ffprobe -count_frames failed on " + path;
            return false;
        }
        pr.n_frames = atoi(cnt.c_str());
    }
    return true;
}

// One rawvideo pass: decode every frame, keep the ones `idx` asks for (which is non-decreasing, so a
// single walk suffices and a repeated index copies the same frame twice). `*seen` gets the number of
// frames the decoder actually produced.
bool decode_pass(const std::string & path, const probe_result & pr, const std::vector<int> & idx,
                 std::vector<uint8_t> & out, int * seen, std::string & err) {
    const size_t fsz = (size_t) pr.width * pr.height * 3;
    if (fsz == 0 || idx.size() > (size_t) -1 / fsz) {
        err = path + ": " + std::to_string(idx.size()) + " frames of " + std::to_string(fsz) +
              " bytes do not fit in memory";
        return false;
    }
    out.assign(idx.size() * fsz, 0);

    FILE * p = JEPA_POPEN(decode_command(path).c_str(), "r");
    if (!p) { err = "cannot start " + ffmpeg_bin(); return false; }

    std::vector<uint8_t> buf(fsz);
    size_t next = 0;
    int t = 0;
    size_t tail = 0;
    for (;;) {
        const size_t got = fread(buf.data(), 1, fsz, p);
        if (got != fsz) { tail = got; break; }
        while (next < idx.size() && idx[next] == t) {
            memcpy(out.data() + next * fsz, buf.data(), fsz);
            next++;
        }
        t++;
    }
    const int st = JEPA_PCLOSE(p);
#ifndef _WIN32
    const int rc = st == -1 ? -1 : (WIFEXITED(st) ? WEXITSTATUS(st) : 128);
#else
    const int rc = st;
#endif
    *seen = t;
    if (rc != 0) {
        err = ffmpeg_bin() + " failed on " + path + " (exit " + std::to_string(rc) + "); command was:\n  " +
              decode_command(path);
        return false;
    }
    if (tail != 0) {
        err = path + ": ffmpeg wrote " + std::to_string(tail) + " bytes past the last whole " +
              std::to_string(pr.width) + "x" + std::to_string(pr.height) +
              " rgb24 frame — the stream is not the size ffprobe reported";
        return false;
    }
    return true;
}

} // namespace

std::vector<int> sample_indices(int total, int n) {
    std::vector<int> idx;
    if (n <= 0 || total <= 0) return idx;
    idx.resize((size_t) n);
    // np.linspace(0, total-1, n): y = arange(n) * step, then y[-1] = stop exactly (endpoint=True).
    // A single sample has an undefined step in numpy too and comes out as [start] = [0].
    const double step = n > 1 ? (double) (total - 1) / (double) (n - 1) : 0.0;
    for (int k = 0; k < n; k++) idx[(size_t) k] = round_half_even((double) k * step);
    if (n > 1) idx[(size_t) n - 1] = total - 1;
    return idx;
}

std::string decode_command(const std::string & path) {
    // -noautorotate: PyAV's frame.to_ndarray() ignores the display matrix, ffmpeg would honour it
    //   and hand back a rotated (and, for 90 degrees, transposed) frame.
    // -map 0:v:0: exactly the stream PyAV's `container.streams.video[0]` decodes, and nothing else —
    //   an explicit -map replaces ffmpeg's default stream selection, so no audio, subtitle or data
    //   stream can reach the rawvideo muxer (-an is kept only as a readable belt).
    // no -sws_flags: libswscale's default converts yuv420p -> rgb24 bit-identically to PyAV's
    //   reformatter here (nothing is scaled). Forcing full_chroma_int+accurate_rnd does NOT: it
    //   changes 86 % of pixels by up to 44 levels. Measured, docs/architecture.md "Preprocessing".
    return ffmpeg_bin() + " -nostdin -v error -noautorotate -i " + shq(path) +
           " -map 0:v:0 -an " + passthrough_flag() + " -f rawvideo -pix_fmt rgb24 -";
}

bool have_ffmpeg(std::string * why) {
    // asked once: a --video-list walk would otherwise spawn two extra processes per clip
    static const int cached = [] {
        const bool ff = run_capture(ffmpeg_bin()  + " -version" JEPA_QUIET, nullptr) == 0;
        const bool fp = run_capture(ffprobe_bin() + " -version" JEPA_QUIET, nullptr) == 0;
        return ff ? (fp ? 2 : 1) : 0;       // 2 = both, 1 = ffmpeg only, 0 = no ffmpeg
    }();
    if (cached == 2) return true;
    if (why) {
        *why = "cannot run '" + (cached == 1 ? ffprobe_bin() : ffmpeg_bin()) + "'. --video decodes clips with "
               "ffmpeg; install it (apt install ffmpeg / brew install ffmpeg / conda install -c "
               "conda-forge ffmpeg), or point $JEPA_FFMPEG and $JEPA_FFPROBE at one. Without ffmpeg, "
               "decode the clip with scripts/video_frames.py and pass --frames-npy.";
    }
    return false;
}

bool decode(const std::string & path, int n_frames, clip & out, std::string & err) {
    const double t0 = now_s();
    if (n_frames <= 0) { err = "--frames must be >= 1"; return false; }
    if (!have_ffmpeg(&err)) return false;

    probe_result pr;
    if (!probe(path, pr, err)) return false;
    if (pr.n_frames <= 0) { err = path + ": ffprobe found no frames"; return false; }

    std::vector<int> idx = sample_indices(pr.n_frames, n_frames);
    int seen = 0;
    if (!decode_pass(path, pr, idx, out.data, &seen, err)) return false;
    if (seen <= 0) { err = path + ": ffmpeg decoded no frames"; return false; }
    if (seen != pr.n_frames) {
        // The container lied (a truncated file, or an nb_frames that counts samples ffmpeg drops).
        // We now know the true length, so redo the pass with the right indices rather than sample a
        // clip that does not exist. One extra decode, only on the files that need it.
        fprintf(stderr, "note: %s: ffprobe said %d frames, the decoder produced %d — resampling\n",
                path.c_str(), pr.n_frames, seen);
        pr.n_frames = seen;
        idx = sample_indices(pr.n_frames, n_frames);
        int seen2 = 0;
        if (!decode_pass(path, pr, idx, out.data, &seen2, err)) return false;
        if (seen2 != seen) {
            err = path + ": the decoder is not reproducible (" + std::to_string(seen) + " frames, then " +
                  std::to_string(seen2) + ")";
            return false;
        }
    }
    out.n_frames       = n_frames;
    out.height         = pr.height;
    out.width          = pr.width;
    out.n_frames_total = pr.n_frames;
    out.frame_indices  = idx;
    out.fps            = pr.fps;
    out.decode_s       = now_s() - t0;
    return true;
}

} // namespace jepa_video
