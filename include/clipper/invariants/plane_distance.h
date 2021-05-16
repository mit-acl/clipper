/**
 * @file plane_distance.h
 * @brief Pairwise geometric invariant between planes
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

  class PlaneDistance : public PairwiseInvariant
  {
  public:
    struct Params
    {
      double sigma = 0.01; ///< spread / "variance" of exponential kernel
      double epsilon = 0.06; ///< bound on consistency score, determines if inlier/outlier
    };
  public:
    PlaneDistance(const Params& params)
    : params_(params) {}
    ~PlaneDistance() = default;

    double operator()(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) override;

  private:
    Params params_;    
  };

} // ns invariants
} // ns clipper