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

# --- ggml (submodule) --------------------------------------------------------
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ${JEPA_NATIVE} CACHE BOOL "" FORCE)
add_subdirectory(ggml EXCLUDE_FROM_ALL)

# --- core library ------------------------------------------------------------
add_library(jepa STATIC
  src/jepa.cpp
  src/jepa-gguf.cpp
  src/preprocess.cpp
  src/rope3d.cpp
)
target_include_directories(jepa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(jepa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(jepa PUBLIC ggml)
target_compile_options(jepa PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-function)

# --- tools -------------------------------------------------------------------
if(JEPA_BUILD_TOOLS)
  foreach(tool jepa-embed jepa-classify jepa-quantize jepa-bench jepa-info)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/${tool}.cpp)
      add_executable(${tool} tools/${tool}.cpp)
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
    add_test(NAME parity COMMAND test-parity ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-ops.cpp)
    add_executable(test-ops tests/test-ops.cpp)
    target_link_libraries(test-ops PRIVATE jepa)
    target_include_directories(test-ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ops COMMAND test-ops)
  endif()
endif()
