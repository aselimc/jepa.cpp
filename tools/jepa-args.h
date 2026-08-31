// Shared command-line bits for the jepa.cpp tools.
#pragma once

#include "jepa.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// `--gpu` / `--gpu N`: run the model on the n-th GPU device of the ggml backend registry
// (docs/architecture.md "GPU backend"). Consumes the optional index only when the next argument
// is a bare number, so `--gpu -i img.jpg` still works. Returns true when the argument was ours.
//   i      : index of the current argument, advanced past a consumed index
//   device : set to the requested device (>= 0); left alone otherwise
static inline bool jepa_arg_gpu(int argc, char ** argv, int & i, int & device) {
    if (strcmp(argv[i], "--gpu") != 0) return false;
    device = 0;
    if (i + 1 < argc) {
        const char * n = argv[i + 1];
        char * end = nullptr;
        const long v = strtol(n, &end, 10);
        if (end && *end == '\0' && end != n && v >= 0) {
            device = (int) v;
            i++;
        }
    }
    return true;
}

// A string literal, so it concatenates into a tool's usage() literal list.
#define JEPA_GPU_USAGE \
    "  --gpu [N]         run on GPU device N (default 0; $JEPA_DEVICE=cuda:N does the same).\n" \
    "                    Needs a build configured with -DJEPA_CUDA=ON; --threads is then unused.\n"

// One line naming the devices the registry can see, for `--devices` style listings.
static inline void jepa_print_devices(FILE * out) {
    const int n = jepa_device_count();
    if (n == 0) {
        fprintf(out, "GPU devices: none (build with -DJEPA_CUDA=ON, or no GPU was detected)\n");
        return;
    }
    for (int i = 0; i < n; i++) {
        size_t f = 0, t = 0;
        jepa_device_memory(i, &f, &t);
        fprintf(out, "GPU %d: %s — %s (%.0f MiB free of %.0f MiB)\n", i, jepa_device_name(i),
                jepa_device_description(i), f / (1024.0 * 1024.0), t / (1024.0 * 1024.0));
    }
}
