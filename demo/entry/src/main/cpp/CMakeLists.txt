# the minimum version of CMake.
cmake_minimum_required(VERSION 3.5.0)
project(demo)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

add_library(entry SHARED
    napi_init.cpp
    validation/rawfile_loader.cpp
    validation/tensor_metrics.cpp
    validation/nn_runtime_validator.cpp
    validation/napi_result.cpp
)

set(ENTRY_LINK_LIBS
    libace_napi.z.so
    librawfile.z.so
    libhilog_ndk.z.so
    libneural_network_core.so
)

find_library(HIAI_FOUNDATION_LIBRARY
    NAMES hiai_foundation libhiai_foundation.so
)

if(HIAI_FOUNDATION_LIBRARY)
    message(STATUS "Linking optional HiAI foundation library: ${HIAI_FOUNDATION_LIBRARY}")
    list(APPEND ENTRY_LINK_LIBS ${HIAI_FOUNDATION_LIBRARY})
    target_compile_definitions(entry PRIVATE INTERNVL_HAS_HIAI_FOUNDATION=1)
else()
    message(STATUS "Optional HiAI foundation library not found; relying on libneural_network_core.so")
endif()

target_link_libraries(entry PUBLIC
    ${ENTRY_LINK_LIBS}
)
