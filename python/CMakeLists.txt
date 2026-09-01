# Builds the one artefact the Python package needs: a shared library that carries all of libjepa
# and all of ggml, so `ctypes.CDLL` has a single file to open and the wheel has a single file to
# ship. The C sources are not listed here — the repository's own CMakeLists.txt is added as a
# subproject, so the bindings are always compiled from the same target definition, the same ggml
# submodule and the same options (GGML_LLAMAFILE=ON, ...) as `build/`.
cmake_minimum_required(VERSION 3.24)   # for $<LINK_LIBRARY:WHOLE_ARCHIVE,...>
project(jepa_cpp_lib LANGUAGES C CXX)

set(JEPA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
    "The jepa.cpp checkout to build against (defaults to the parent of python/)")
get_filename_component(JEPA_SOURCE_DIR "${JEPA_SOURCE_DIR}" ABSOLUTE)
if(NOT EXISTS "${JEPA_SOURCE_DIR}/include/jepa.h")
  message(FATAL_ERROR
    "no jepa.cpp checkout at ${JEPA_SOURCE_DIR} (looked for include/jepa.h). The Python package "
    "builds from the repository, not from python/ alone: install python/ from inside a recursive "
    "clone, or point -DJEPA_SOURCE_DIR=... at the checkout.")
endif()
if(NOT EXISTS "${JEPA_SOURCE_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "the ggml submodule at ${JEPA_SOURCE_DIR}/ggml is empty — "
                      "initialise the submodule in the checkout first")
endif()

# Everything static, then re-linked whole into one shared object: a wheel that dlopens a single
# self-contained file needs no RPATH and no repair pass over sibling libggml*.so files.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(JEPA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(JEPA_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# -march=native is right for a wheel built on the machine that runs it and wrong for one that is
# distributed: it is the repository default, so a local `pip install ./python` keeps the ISA — and
# therefore the arithmetic — of `build/`, and the cibuildwheel configuration in pyproject.toml
# turns it off for the wheels that get published.
option(JEPA_PY_NATIVE "Optimize the bundled library for the host CPU (-march=native)" ON)
set(JEPA_NATIVE ${JEPA_PY_NATIVE} CACHE BOOL "" FORCE)

add_subdirectory("${JEPA_SOURCE_DIR}" _jepa)

add_library(jepa_shared SHARED shared.c)
set_target_properties(jepa_shared PROPERTIES OUTPUT_NAME jepa C_VISIBILITY_PRESET default)
target_link_libraries(jepa_shared PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,jepa>")

install(TARGETS jepa_shared
        LIBRARY DESTINATION jepa_cpp/lib
        RUNTIME DESTINATION jepa_cpp/lib
        ARCHIVE DESTINATION jepa_cpp/lib)
