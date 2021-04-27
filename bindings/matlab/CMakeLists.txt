find_package(Matlab COMPONENTS MX_LIBRARY)

if(Matlab_FOUND)
    message(STATUS "MATLAB root directory found: ${Matlab_ROOT_DIR}")

    matlab_add_mex(NAME finddensecluster_mex SRC finddensecluster_mex.cpp LINK_TO Eigen3::Eigen clipper)
    set_target_properties(finddensecluster_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")
    set_target_properties(finddensecluster_mex PROPERTIES OUTPUT_NAME clipper_finddensecluster)

    matlab_add_mex(NAME knownscalepointcloud_mex SRC knownscalepointcloud_mex.cpp LINK_TO Eigen3::Eigen clipper)
    set_target_properties(knownscalepointcloud_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")
    set_target_properties(knownscalepointcloud_mex PROPERTIES OUTPUT_NAME clipper_knownscalepointcloud)

    matlab_add_mex(NAME planecloud_mex SRC planecloud_mex.cpp LINK_TO Eigen3::Eigen clipper)
    set_target_properties(planecloud_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")
    set_target_properties(planecloud_mex PROPERTIES OUTPUT_NAME clipper_planecloud)
else()
    message(WARNING "MATLAB root directory not found. Will not build MATLAB bindings.")
    set(BUILD_MATLAB_BINDINGS OFF)
endif()
