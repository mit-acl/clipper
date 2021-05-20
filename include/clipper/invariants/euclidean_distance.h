/**
 * @file euclidean_distance.h
 * @brief Pairwise Euclidean distance geometric invariant
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

  class EuclideanDistance : public PairwiseInvariant
  {
  public:
    struct Params
    {
      double sigma = 0.01; ///< spread / "variance" of exponential kernel
      double epsilon = 0.06; ///< bound on consistency score, determines if inlier/outlier
    };
  public:
    EuclideanDistance(const Params& params)
    : params_(params) {}
    ~EuclideanDistance() = default;

    double operator()(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) override;

  private:
    Params params_;
  };

} // ns invariants
} // ns clipper