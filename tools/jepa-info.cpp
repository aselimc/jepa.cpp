#include "jepa.h"
#include <cstdio>
int main(int argc, char ** argv) {
    jepa_print_system_info();
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    fprintf(stderr, "jepa-info: model loading not implemented yet (scaffold)\n");
    return 0;
}
