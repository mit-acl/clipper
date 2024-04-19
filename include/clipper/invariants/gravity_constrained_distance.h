/**
 * @file GravityConstrained_distance.h
 * @brief Pairwise GravityConstrained distance geometric invariant
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

  /**
   * @brief      Specialization of PairwiseInvariant to GravityConstrained distance in
   *             the real numbers using the 2-norm as the invariant.
   */
  class GravityConstrainedDistance : public PairwiseInvariant
  {
  public:
    struct Params
    {
      double sigma = 0.01; ///< spread / "variance" of exponential kernel
      double epsilon = 0.06; ///< bound on consistency score, determines if inlier/outlier
      double mindist = 0; ///< minimum allowable distance between inlier points in the same dataset
    };
  public:
    GravityConstrainedDistance(const Params& params)
    : params_(params) {}
    ~GravityConstrainedDistance() = default;

    /**
     * @brief      Functor for pairwise invariant scoring function
     *
     * @param[in]  ai    Element i from dataset 1
     * @param[in]  aj    Element j from dataset 1
     * @param[in]  bi    Element i from dataset 2
     * @param[in]  bj    Element j from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi) and (aj,bj)
     */
    double operator()(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) override;

  private:
    Params params_;
  };

  using GravityConstrainedDistancePtr = std::shared_ptr<GravityConstrainedDistance>;

} // ns invariants
} // ns clipper