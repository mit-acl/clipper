/**
 * @file main.cpp
 * @brief Entry point for all CLIPPER test suites
 * @author Parker Lusk <plusk@mit.edu>
 * @date 19 July 2020
 */

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}