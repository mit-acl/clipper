include(FetchContent)
FetchContent_Declare(tinyply
    GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply
    GIT_TAG 2.3.4)
# set(BUILD_TESTS OFF)
# FetchContent_MakeAvailable(tinyply)

FetchContent_GetProperties(tinyply)
if(NOT tinyply_POPULATED)
    FetchContent_Populate(tinyply)

    set(BUILD_TESTS OFF)
    add_subdirectory(${tinyply_SOURCE_DIR} ${tinyply_BINARY_DIR})

    # fix for tinyply because it does not use target_include_directories
    target_include_directories(tinyply PUBLIC
        $<BUILD_INTERFACE:${tinyply_SOURCE_DIR}/source>)
endif()

include(FetchContent)
FetchContent_Declare(nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann
    GIT_TAG v1.4.2)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_MakeAvailable(nanoflann)

include(FetchContent)
FetchContent_Declare(libfort
    GIT_REPOSITORY https://github.com/seleznevae/libfort
    GIT_TAG v0.4.2)
set(FORT_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(libfort)

include(FetchContent)
FetchContent_Declare(indicators
    GIT_REPOSITORY https://github.com/p-ranav/indicators
    GIT_TAG v2.2)
set(INDICATORS_BUILD_TESTS OFF)
set(INDICATORS_SAMPLES OFF)
set(INDICATORS_DEMO OFF)
FetchContent_MakeAvailable(indicators)


add_executable(benchmark)
target_sources(benchmark PRIVATE main.cpp bm_utils.cpp)
target_include_directories(benchmark PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(benchmark clipper Eigen3::Eigen tinyply nanoflann fort indicators)

# trying to make life easier when running ./benchmarks/benchmark
add_custom_command(TARGET benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/examples/data/*.ply
                                     ${CMAKE_BINARY_DIR}/)