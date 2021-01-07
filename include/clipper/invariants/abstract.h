/**
 * @file abstract.h
 * @brief Abstract base class for geometric invariants
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

namespace clipper {

  using Association = Eigen::Matrix<int, Eigen::Dynamic, 2>;
  using PairMC = std::pair<Eigen::MatrixXd, Eigen::MatrixXd>; ///< return type

namespace invariants {

  class Invariant
  {
  public:
    struct Params {
      double sigma = 0.01; ///< spread of exponential scoring ("variance")
      double epsilon = 0.06; ///< bound on consistency score, determines if inlier/outlier
    };
    using Data = Eigen::MatrixXd;
  public:
    Invariant(const Params& params)
    : params_(params) {}
    ~Invariant() = default;

    // virtual Eigen::MatrixXd createAffinityMatrix(const Data& D1, const Data& D2) = 0;

  protected:
    Params params_;

    Association createAllToAll(size_t n1, size_t n2)
    {
      Association A = Association(n1*n2, 2);
      for (size_t i=0; i<n1; ++i) {
        for (size_t j=0; j<n2; ++j) {
          A(j + i*n2, 0) = i;
          A(j + i*n2, 1) = j;
        }
      }
      return A;
    }
  };
  
  using InvariantPtr = std::shared_ptr<Invariant>;

} // ns invariants
} // ns clipper