set(PYPKG_DIR "${CMAKE_CURRENT_BINARY_DIR}/clipper")

pybind11_add_module(py_clipper py_clipper.cpp)
target_link_libraries(py_clipper PUBLIC clipper)
set_target_properties(py_clipper PROPERTIES OUTPUT_NAME "clipper")
set_target_properties(py_clipper PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYPKG_DIR}")

# copy setup.py file binary dir for install with: pip install .
configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# Create the Python package -- Note that "." is used to conform to PEP 328
file(WRITE "${PYPKG_DIR}/__init__.py"
    "from .clipper import *\n"
    "from .clipper import __version__\n"
    "from .clipper import __doc__")