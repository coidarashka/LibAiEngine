cmake_minimum_required(VERSION 3.10)
project(AiEngine)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE OMP_LIB "${NDK_HOME}/**/libomp.a")
list(GET OMP_LIB 0 OMP_PATH)

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI  ON CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

file(GLOB_RECURSE MTMD_H "llama.cpp/**/mtmd.h")
get_filename_component(MTD_DIR ${MTMD_H} DIRECTORY)

file(GLOB_RECURSE M_SRCS "llama.cpp/tools/mtmd/*.cpp")
list(FILTER M_SRCS EXCLUDE REGEX "mtmd-cli\\.cpp$|deprecation-warning\\.cpp$")
file(GLOB_RECURSE M_MOD  "llama.cpp/tools/mtmd/models/*.cpp")
file(GLOB_RECURSE M_HLP  "llama.cpp/tools/mtmd-helper.cpp")

add_library(AiEngine SHARED bridge.cpp ${M_SRCS} ${M_MOD} ${M_HLP})

target_link_libraries(AiEngine PRIVATE llama common log
    -Wl,--whole-archive "${OMP_PATH}" -Wl,--no-whole-archive)

target_include_directories(AiEngine PRIVATE
    llama.cpp/include llama.cpp/common
    llama.cpp/src llama.cpp/ggml/include
    json/include ${MTD_DIR} .)

target_link_options(AiEngine PRIVATE "-static-openmp" "-Wl,--exclude-libs,ALL")

target_compile_options(AiEngine PRIVATE
    -flto -funroll-loops -O3 -ffast-math -fopenmp -DGGML_FLASH_ATTN=ON)

target_link_options(AiEngine PRIVATE -flto)
