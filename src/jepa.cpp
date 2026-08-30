// jepa.cpp — core (scaffold; see docs/architecture.md). Implementation lands in phase 1.
#include "jepa-internal.h"
#include <thread>
#include <cstdarg>
#include <cstdio>

void jepa_log(const char * fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
}
const char * jepa_version(void) { return JEPA_VERSION; }
void jepa_print_system_info(void) {
    fprintf(stderr, "jepa.cpp %s | ggml | threads available: %d\n", JEPA_VERSION, (int) std::thread::hardware_concurrency());
}
jepa_context_params jepa_context_default_params(void) {
    jepa_context_params p; p.n_threads = 0; p.use_flash_attn = true; p.verbose = false; return p;
}
