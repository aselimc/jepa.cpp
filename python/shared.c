/* The translation unit that makes libjepa.so exist. Nothing in the Python package calls into it:
 * the library's content is the static `jepa` target linked with WHOLE_ARCHIVE (python/CMakeLists.txt),
 * and CMake needs a shared target to own at least one source.
 *
 * jepa_cpp_shared_abi() gives the loader in jepa_cpp/_lib.py something cheap to look up, so it can
 * tell a jepa.cpp library apart from some other libjepa that happens to be on the search path. */
#include "jepa.h"

const char * jepa_cpp_shared_abi(void) { return jepa_version(); }
