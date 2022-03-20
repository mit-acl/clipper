/**
 * @file types.h
 * @brief Types used in CLIPPER framework
 * @author Parker Lusk <plusk@mit.edu>
 * @date 19 March 2022
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace clipper {

  using SpMat = Eigen::SparseMatrix<double>;
  using SpTriplet = Eigen::Triplet<double>;

  using Association = Eigen::Matrix<int, Eigen::Dynamic, 2>;
  using Affinity = Eigen::MatrixXd;
  using Constraint = Eigen::MatrixXd;

  using SpAffinity = SpMat;
  using SpConstraint = SpMat;

} // ns clipper