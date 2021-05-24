find_package(Matlab COMPONENTS MX_LIBRARY)

if(Matlab_FOUND)
    message(STATUS "MATLAB root directory found: ${Matlab_ROOT_DIR}")

    matlab_add_mex(NAME finddensecluster_mex SRC finddensecluster_mex.cpp LINK_TO Eigen3::Eigen clipper)
    set_target_properties(finddensecluster_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")
    set_target_properties(finddensecluster_mex PROPERTIES OUTPUT_NAME clipper_finddensecluster)

    matlab_add_mex(NAME euclideandistance_mex SRC euclideandistance_mex.cpp LINK_TO Eigen3::Eigen clipper)
    set_target_properties(euclideandistance_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")
    set_target_properties(euclideandistance_mex PROPERTIES OUTPUT_NAME clipper_euclideandistance)

    matlab_add_mex(NAME pointnormaldistance_mex SRC pointnormaldistance_mex.cpp LINK_TO Eigen3::Eigen clipper)
    set_target_properties(pointnormaldistance_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")
    set_target_properties(pointnormaldistance_mex PROPERTIES OUTPUT_NAME clipper_pointnormaldistance)
else()
    message(WARNING "MATLAB root directory not found. Will not build MATLAB bindings.")
    set(BUILD_MATLAB_BINDINGS OFF)
endif()
