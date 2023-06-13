include(GoogleTest)

add_executable(tests main.cpp affinity_test.cpp clipper_test.cpp sdp_test.cpp dsd_test.cpp)
target_link_libraries(tests clipper gtest pthread Eigen3::Eigen)
gtest_add_tests(TARGET tests TEST_LIST all_tests)
set_tests_properties(${all_tests} PROPERTIES TIMEOUT 10)