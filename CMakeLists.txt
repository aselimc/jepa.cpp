cmake_minimum_required(VERSION 3.16)
project(jepa.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(JEPA_BUILD_TOOLS "Build command-line tools" ON)
option(JEPA_BUILD_TESTS "Build tests" ON)
option(JEPA_NATIVE      "Optimize for the host CPU (-march=native)" ON)
# OFF by default and it stays that way: libggml-cuda.so is ~45 MB for a *single* GPU architecture,
# 30x the CPU backend, which is not something a "CPU inference in plain C/C++ on ggml" project
# should ship by default (docs/getting-started.md "CUDA build"). Configure a CUDA build in its own dir:
#   cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DJEPA_CUDA=ON
option(JEPA_CUDA        "Build with the ggml CUDA backend (adds --gpu to the tools)" OFF)

# --- ggml (submodule) --------------------------------------------------------
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ${JEPA_NATIVE} CACHE BOOL "" FORCE)
if(JEPA_CUDA)
  # GGML_NATIVE=ON resolves CMAKE_CUDA_ARCHITECTURES to `native` (89 on this box), which is right
  # for a from-source build and wrong for a distributed binary — pass an explicit
  # -DCMAKE_CUDA_ARCHITECTURES=89-real;80-virtual for one of those.
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
# 1.3-3.2x faster mul_mat on AVX-512 with identical f16/q8_0 numerics (F32 differs at round-off only);
# measured in docs/ggml-notes.md §5. flash_attn_ext is unaffected by this flag.
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
add_subdirectory(ggml EXCLUDE_FROM_ALL)

# --- compiler flags ----------------------------------------------------------
# MSVC does not speak the GCC/Clang warning spellings, needs /utf-8 for these UTF-8 sources and
# /bigobj for the translation units that include third_party/json.hpp (tools/jepa-worldmodel.cpp,
# tests/test-parity.cpp, tests/test-video.cpp), which otherwise overflow the object file's section
# limit (C1128).
if(MSVC)
  add_compile_options(/utf-8 /bigobj)
  set(JEPA_WARN_FLAGS /W3)
else()
  set(JEPA_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter -Wno-unused-function)
endif()

# --- core library ------------------------------------------------------------
add_library(jepa STATIC
  src/jepa.cpp
  src/jepa-gguf.cpp
  src/preprocess.cpp
  src/rope3d.cpp
  src/predictor.cpp
  src/lewm.cpp
)
target_include_directories(jepa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(jepa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(jepa PUBLIC ggml)
target_compile_options(jepa PRIVATE ${JEPA_WARN_FLAGS})
# jepa_version() reports the project version above, so a release binary cannot disagree with the tag.
# src/jepa-internal.h keeps a fallback for a build that does not come through this file.
target_compile_definitions(jepa PUBLIC JEPA_VERSION="${PROJECT_VERSION}")

# --- tools -------------------------------------------------------------------
if(JEPA_BUILD_TOOLS)
  foreach(tool jepa-embed jepa-classify jepa-quantize jepa-bench jepa-info jepa-worldmodel)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/${tool}.cpp)
      set(_tool_src tools/${tool}.cpp)
      # --video: ffmpeg is a *run-time* dependency of these two tools, never a build-time one, so
      # the decoder is plain C++ compiled into them and a checkout without ffmpeg still builds
      # (and still runs, minus that one flag).
      if(tool STREQUAL "jepa-embed" OR tool STREQUAL "jepa-classify")
        list(APPEND _tool_src tools/video-decode.cpp)
      endif()
      add_executable(${tool} ${_tool_src})
      target_link_libraries(${tool} PRIVATE jepa)
      target_include_directories(${tool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party ${CMAKE_CURRENT_SOURCE_DIR}/src)
    endif()
  endforeach()
endif()

# --- tests -------------------------------------------------------------------
if(JEPA_BUILD_TESTS)
  enable_testing()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-parity.cpp)
    add_executable(test-parity tests/test-parity.cpp)
    target_link_libraries(test-parity PRIVATE jepa)
    target_include_directories(test-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # fast parity checks on the small f32 models; registered only when the (git-ignored) GGUFs and
    # reference dumps are present (scripts/convert.py + scripts/dump_reference.py generate them)
    foreach(pair
        "lejepa-vits16-pretrain-in1k-f32.gguf;lejepa-vits16"
        "lewm-pusht-f32.gguf;lewm-pusht")
      list(GET pair 0 _gguf)
      list(GET pair 1 _ref)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/${_gguf} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/${_ref}/manifest.json)
        add_test(NAME parity-${_ref}
                 COMMAND test-parity ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/${_gguf}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/${_ref} --threads 8 --quiet)
      endif()
    endforeach()
    # one fast video-family check: the two V-JEPA 2.1 image samples (576 tokens each, ~1 s at 8
    # threads) cover the tubelet-1 image tokenizer, the modality vector and interpolated 3-D RoPE.
    # The clip samples (2048-8192 tokens, ~1-7 s each) are run by hand, see docs/parity.md.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/vjepa2_1-vitb-384-f16.gguf AND
       EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/vjepa2_1-vitb-384/manifest.json)
      add_test(NAME parity-vjepa2_1-vitb-384-images
               COMMAND test-parity ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/vjepa2_1-vitb-384-f16.gguf
                       ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/vjepa2_1-vitb-384 --threads 8 --quiet
                       --samples coco_000000000139,coco_000000000285)
    endif()
    # LeVJEPA: one 3137-token clip covers everything this family adds — the CLS row, its identity
    # RoPE entry and the block-causal mask on the flash path. ~26 s at 8 threads for the two passes,
    # the rest of the fixture set (the second clip and the two still-image samples) is run by hand,
    # see docs/parity.md.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/levjepa-vitl16-f16.gguf AND
       EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/levjepa-vitl16/manifest.json)
      add_test(NAME parity-levjepa-vitl16-clip
               COMMAND test-parity ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/levjepa-vitl16-f16.gguf
                       ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/levjepa-vitl16 --threads 8 --quiet
                       --samples archery_f16)
    endif()
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-predictor.cpp)
    add_executable(test-predictor tests/test-predictor.cpp)
    target_link_libraries(test-predictor PRIVATE jepa)
    target_include_directories(test-predictor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # LeWM predictor parity (f32) — registered only when the git-ignored GGUF and reference dump exist
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/lewm-pusht-f32.gguf AND
       EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/lewm-pusht/manifest.json)
      add_test(NAME predictor-lewm
               COMMAND test-predictor --lewm ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/lewm-pusht-f32.gguf
                       --ref ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/lewm-pusht --threads 8)
    endif()
    # V-JEPA 2 masked predictor parity (f16) on the 16-frame archery sample
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/vjepa2-vitl-fpc64-256-f16.gguf AND
       EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/vjepa2-vitl-fpc64-256/manifest.json)
      add_test(NAME predictor-vjepa2
               COMMAND test-predictor --vjepa2 ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/vjepa2-vitl-fpc64-256-f16.gguf
                       --ref ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/vjepa2-vitl-fpc64-256
                       --samples archery_f16 --threads 8)
    endif()
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-batch.cpp)
    add_executable(test-batch tests/test-batch.cpp)
    target_link_libraries(test-batch PRIVATE jepa)
    target_include_directories(test-batch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # batched encoder graph == per-item path, bit for bit at f32. Registered only with the
    # (git-ignored) small GGUFs present; the fixture JPEGs are in the repo. ~11 s at 8 threads.
    set(_batch_args "")
    foreach(_g lejepa-vits16-pretrain-in1k-f32.gguf lewm-pusht-f32.gguf)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/${_g})
        list(APPEND _batch_args --image ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/${_g})
      endif()
    endforeach()
    # ViT-H at f16 covers the quantized-weight mul_mat path; 4 images keep it inside the time budget
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/ijepa_vith14_1k-f16.gguf)
      list(APPEND _batch_args --images 4 --image ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/ijepa_vith14_1k-f16.gguf)
    endif()
    # V-JEPA 2.1 clip path: two clips as n_batch=2 against two single calls (4 frames = 1152 tokens)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/vjepa2_1-vitb-384-f16.gguf AND
       EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/vjepa2_1-vitb-384/bowling_f16.frames_u8.npy)
      list(APPEND _batch_args --video ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/vjepa2_1-vitb-384-f16.gguf
                              --clip-frames 4
                              --clip ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/vjepa2_1-vitb-384/archery_f16.frames_u8.npy
                              --clip ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/vjepa2_1-vitb-384/bowling_f16.frames_u8.npy)
    endif()
    if(_batch_args)
      add_test(NAME batch COMMAND test-batch --media ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/media
                                  --threads 8 --images 8 ${_batch_args})
    endif()
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-ops.cpp)
    add_executable(test-ops tests/test-ops.cpp)
    target_link_libraries(test-ops PRIVATE jepa)
    target_include_directories(test-ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    # --budget-model needs a block-causal video file to check the video path's $JEPA_MAX_GRAPH_MIB
    # guard on; without one the rope3d and mask cases still run.
    set(_ops_args "")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/levjepa-vitl16-f16.gguf)
      set(_ops_args --budget-model ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/levjepa-vitl16-f16.gguf)
    endif()
    add_test(NAME ops COMMAND test-ops ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/rope3d ${_ops_args})
  endif()
  # GPU backend checks: the supports_op graph walk (which must REFUSE a strided ggml_roll) and a
  # GPU-vs-CPU encoder comparison. The binary always builds; the test registers like the other
  # asset-gated ones (it needs a GGUF) and exits 0 with a SKIP line when the build has no GPU
  # backend or no GPU is present, so it costs a CPU-only checkout nothing.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-backend.cpp)
    add_executable(test-backend tests/test-backend.cpp)
    target_link_libraries(test-backend PRIVATE jepa)
    target_include_directories(test-backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf)
      add_test(NAME backend COMMAND test-backend
               ${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/lejepa-vits16-pretrain-in1k-f16.gguf --threads 8)
    endif()
  endif()
  # The tools' ffmpeg ingest against the frame tensors PyAV produced for the reference dumps: same
  # count, same indices, same pixels. Registered when the (git-ignored) reference dumps and their
  # source clips are present; the binary itself exits 0 with a "skipped" line when ffmpeg is absent,
  # so a checkout without it costs nothing. ~2 s.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-video.cpp)
    add_executable(test-video tests/test-video.cpp tools/video-decode.cpp)
    target_include_directories(test-video PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    set(_video_refs "")
    foreach(_ref vjepa2_1-vitb-384 vjepa2-vitl-fpc64-256 vjepa2-vitl-fpc16-256-ssv2 levjepa-vitl16)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/${_ref}/manifest.json)
        list(APPEND _video_refs --ref ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ref/${_ref})
      endif()
    endforeach()
    if(_video_refs AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/media/archery.mp4)
      add_test(NAME video COMMAND test-video --media ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/media
                                  ${_video_refs})
    endif()
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-attn.cpp)
    add_executable(test-attn tests/test-attn.cpp)
    target_link_libraries(test-attn PRIVATE ggml)
    add_test(NAME attn COMMAND test-attn --n 256,2048 --max-s 60)
  endif()
endif()
