cmake_minimum_required(VERSION 3.10)
project(clipper)

set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
    # Options: Debug, Release, MinSizeRel, RelWithDebInfo
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type." FORCE)
endif()

# set(CMAKE_CXX_FLAGS "-Wall -march=native -mavx -mfma")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

###############################################################################
# Options
###############################################################################

option(BUILD_BINDINGS_MATLAB "Build MATLAB bindings" ON)
option(BUILD_TESTS "Build testsuite" ON)

###############################################################################
# Dependencies
###############################################################################

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen Version: ${EIGEN3_VERSION_STRING} ${EIGEN3_VERSION}")

find_package(OpenMP)

###############################################################################
# Targets
###############################################################################

add_library(clipper SHARED
  src/find_dense_cluster.cpp
  src/utils.cpp
  src/invariants/known_scale_pointcloud.cpp
  src/invariants/planecloud.cpp)
target_link_libraries(clipper PUBLIC Eigen3::Eigen)
target_link_libraries(clipper PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(clipper PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

###############################################################################
# Extras
###############################################################################

if(BUILD_BINDINGS_MATLAB)
    message(STATUS "Attempting to build MATLAB bindings.")
    add_subdirectory(bindings/matlab)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()