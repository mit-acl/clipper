set(PYPKG_DIR "${CMAKE_CURRENT_BINARY_DIR}/clipperpy")

pybind11_add_module(py_clipper py_clipper.cpp)
target_link_libraries(py_clipper PUBLIC clipper)
set_target_properties(py_clipper PROPERTIES OUTPUT_NAME "clipperpy")
set_target_properties(py_clipper PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYPKG_DIR}")

# copy setup.py file binary dir for install with: pip install .
configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# Create the Python package -- Note that "." is used to conform to PEP 328
file(WRITE "${PYPKG_DIR}/__init__.py"
    "from .clipperpy import *\n"
    "from .clipperpy import __version__\n"
    "from .clipperpy import __doc__")

set(DIST "none")
if(UNIX AND NOT APPLE)
    execute_process(COMMAND bash -c "lsb_release -cs" OUTPUT_VARIABLE UBUNTU_DIST)
    string(STRIP "${UBUNTU_DIST}" UBUNTU_DIST)
    set(DIST "${UBUNTU_DIST}")
elseif(APPLE)
    # TODO: not very specific...
    set(DIST "macos")
elseif(WIN32)
    # TODO: not very specific...
    set(DIST "win10")
endif()

set(PKGSTR clipperpy-py3-${DIST}-${PROJECT_VERSION})
add_custom_target(pypkg
    DEPENDS py_clipper
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PKGSTR}
    COMMAND ${CMAKE_COMMAND} -E copy setup.py ${PKGSTR}/
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PYPKG_DIR} ${PKGSTR}/clipperpy
    COMMAND ${CMAKE_COMMAND} -E tar zcvf ${PKGSTR}.tar.gz ${PKGSTR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(pip-install
    DEPENDS pypkg
    COMMAND ${PYTHON_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PKGSTR}
)