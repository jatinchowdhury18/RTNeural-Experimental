cmake_minimum_required(VERSION 3.1)
project(RTNeural-experimental VERSION 1.0.0)

get_directory_property(parent_dir PARENT_DIRECTORY)
if("${parent_dir}" STREQUAL "")
    set(is_toplevel ON)
    include(cmake/CPM.cmake)
    option(RTNEURAL_EIGEN "Use Eigen library for vector operations" OFF)
    option(RTNEURAL_XSIMD "Use xsimd library for vector operations" OFF)
    option(RTNEURAL_STL "Use STL for all operations" OFF)
    CPMAddPackage(
        NAME RTNeural
        GITHUB_REPOSITORY "jatinchowdhury18/RTNeural"
        GIT_TAG main
        OPTIONS "RTNEURAL_EIGEN ${RTNEURAL_EIGEN}" "RTNEURAL_XSIMD ${RTNEURAL_XSIMD}" "RTNEURAL_STL ${RTNEURAL_STL}"
    )
else()
    set(is_toplevel OFF)
endif()

if(APPLE)
    message(STATUS "RTNeural-experimental -- Configuring Accelerate backend target")
    add_library(RTNeural-accelerate INTERFACE)
    target_link_libraries(RTNeural-accelerate INTERFACE "-framework Accelerate")
    target_link_libraries(RTNeural-accelerate INTERFACE RTNeural)
    target_include_directories(RTNeural-accelerate INTERFACE include/RTNeural-accelerate)
endif()

add_library(RTNeural-experimental INTERFACE)
target_link_libraries(RTNeural-experimental INTERFACE RTNeural)
target_include_directories(RTNeural-experimental INTERFACE include/RTNeural-experiments)

if(${is_toplevel})
    add_subdirectory(examples)
endif()
