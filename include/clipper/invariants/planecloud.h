/**
 * @file planecloud.h
 * @brief Scores geometric invariants of a set of planes
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 */

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

  class PlaneCloud : public Invariant
  {
  public:
    using Data = Eigen::Matrix<double, 4, Eigen::Dynamic>;
    using Datum = Eigen::Matrix<double, 4, 1>;
  public:
    PlaneCloud(const Params& params)
    : Invariant(params) {}
    ~PlaneCloud() = default;

    PairMC createAffinityMatrix(const Data& D1, const Data& D2, Association& A);

  private:

    double scoreInvariantConsistency(
      const Datum& d1i, const Datum& d1j, const Datum& d2i, const Datum& d2j);
    
  };

} // ns invariants
} // ns clipper