cmake_minimum_required(VERSION 3.10)
project(clipper VERSION 0.2)

set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
    # Options: Debug, Release, MinSizeRel, RelWithDebInfo
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type." FORCE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

###############################################################################
# Options
###############################################################################

option(CLIPPER_BUILD_BINDINGS_PYTHON "Build Python bindings" ON)
option(CLIPPER_BUILD_BINDINGS_MATLAB "Build MATLAB bindings" OFF)
option(CLIPPER_BUILD_TESTS "Build testsuite" OFF)
option(CLIPPER_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CLIPPER_ENABLE_MKL "Use MKL with Eigen" OFF)
option(CLIPPER_ENABLE_BLAS "Use BLAS with Eigen" OFF) # apt install libopenblas-dev

if(CLIPPER_ENABLE_MKL AND CLIPPER_ENABLE_BLAS)
  message(FATAL_ERROR "Cannot enable both MKL and BLAS (prefer MKL)")
endif()

###############################################################################
# Dependencies
###############################################################################

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen Version: ${EIGEN3_VERSION_STRING} ${EIGEN3_VERSION}")

find_package(OpenMP)

# Give preference to MKL
if(CLIPPER_ENABLE_MKL)
    find_package(MKL)
    if(MKL_FOUND)
        message(STATUS "MKL found at: ${MKL_LIBRARIES}")
    else()
        message(STATUS "MKL not found.")
    endif()
endif()

if(CLIPPER_ENABLE_BLAS)
    # OpenBLAS is generally better than ATLAS
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "BLAS found at: ${BLAS_LIBRARIES}")
    else()
        message(STATUS "BLAS not found.")
    endif()
endif()

if(CLIPPER_BUILD_BINDINGS_PYTHON)
  set(PYBIND11_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11-download" CACHE INTERNAL "pybind11 build dir" FORCE)
  configure_file(${CMAKE_MODULE_PATH}/pybind11.cmake.in ${PYBIND11_DIR}/CMakeLists.txt IMMEDIATE @ONLY)
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY ${PYBIND11_DIR})
  execute_process(COMMAND "${CMAKE_COMMAND}" --build . WORKING_DIRECTORY ${PYBIND11_DIR})
  add_subdirectory(${PYBIND11_DIR}/src)
endif()

###############################################################################
# Targets
###############################################################################

add_library(clipper SHARED)
target_sources(clipper PRIVATE
  src/clipper.cpp
  src/utils.cpp
  src/invariants/euclidean_distance.cpp
  src/invariants/pointnormal_distance.cpp)
target_link_libraries(clipper PUBLIC Eigen3::Eigen)
target_link_libraries(clipper PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(clipper PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_options(clipper PUBLIC -march=native -mavx -mfma)
set_target_properties(clipper PROPERTIES
  VERSION ${PROJECT_VERSION})
target_compile_definitions(clipper PUBLIC PROJECT_VERSION="${PROJECT_VERSION}")

if(MKL_FOUND)
    target_include_directories(clipper PRIVATE ${MKL_INCLUDE_DIR})
    target_compile_definitions(clipper PRIVATE -DEIGEN_USE_MKL_ALL)
    target_link_libraries(clipper PRIVATE ${MKL_LIBRARIES})
elseif(BLAS_FOUND)
    target_compile_definitions(clipper PRIVATE -DEIGEN_USE_BLAS=1)
    target_link_libraries(clipper PRIVATE BLAS::BLAS)
endif()

###############################################################################
# Extras
###############################################################################

if(CLIPPER_BUILD_BINDINGS_PYTHON)
    message(STATUS "Building Python bindings.")
    add_subdirectory(bindings/python)
endif()

if(CLIPPER_BUILD_BINDINGS_MATLAB)
    message(STATUS "Attempting to build MATLAB bindings.")
    add_subdirectory(bindings/matlab)
endif()

if(CLIPPER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(CLIPPER_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

###############################################################################
# Install
###############################################################################

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/clipper)

install(TARGETS clipper
    EXPORT clipper-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT clipper-targets
  FILE clipper-targets.cmake
  DESTINATION ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/clipper-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/clipper-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/clipper-config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/clipper-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/clipper-config-version.cmake
  DESTINATION ${INSTALL_CONFIGDIR})
