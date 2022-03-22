/**
 * @file pointnormal_distance.h
 * @brief Pairwise geometric invariant between point-normals (e.g., planes)
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

  /**
   * @brief      Specialization of PairwiseInvariant to be applied to sets
   *             of planes, patches, or equivalently, points with normals.
   *             
   *             Data: 6xn matrix
   *             Datum: 6x1 vector --> top 3x1 is point, bottom 3x1 is normal.
   */
  class PointNormalDistance : public PairwiseInvariant
  {
  public:
    struct Params
    {
      double sigp = 0.5; ///< point - spread of exp kernel
      double epsp = 0.5; ///< point - bound on consistency score
      double sign = 0.10; ///< normal - spread of exp kernel
      double epsn = 0.35; ///< normal - bound on consistency score
    };
  public:
    PointNormalDistance(const Params& params)
    : params_(params) {}
    ~PointNormalDistance() = default;

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

  using PointNormalDistancePtr = std::shared_ptr<PointNormalDistance>;

} // ns invariants
} // ns clipper