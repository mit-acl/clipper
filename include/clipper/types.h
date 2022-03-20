/**
 * @file types.h
 * @brief Types used in CLIPPER framework
 * @author Parker Lusk <plusk@mit.edu>
 * @date 19 March 2022
 */

#pragma once

#include <Eigen/Dense>

namespace clipper {

  using Association = Eigen::Matrix<int, Eigen::Dynamic, 2>;
  using Affinity = Eigen::MatrixXd;
  using Constraint = Eigen::MatrixXd;

  using SpAffinity = Eigen::MatrixXd;
  using SpConstraint = Eigen::MatrixXd;

} // ns clipper