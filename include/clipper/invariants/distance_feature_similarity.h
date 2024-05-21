/**
 * @file distance_feature_similarity.h
 * @brief Pairwise distance and feature similarity geometric invariant
 * @author Mason Peterson <masonbp@mit.edu>
 * @date 21 May 2024
 */

#pragma once

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

  /**
   * @brief      Specialization of PairwiseInvariant to GravityConstrained distance in
   *             the real numbers using the 2-norm as the invariant.
   */
  class DistanceFeatureSimilarity : public PairwiseInvariant
  {
  public:
    struct Params
    {
      uint8_t point_dim = 3; ///< dimension of points (2 or 3)
      uint8_t feature_dim = 0; ///< number of features to consider
      double sigma = 0.01; ///< spread / "variance" of exponential kernel
      double epsilon = 0.06; ///< bound on consistency score, determines if inlier/outlier
      double mindist = 0; ///< minimum allowable distance between inlier points in the same dataset
      Eigen::VectorXd feature_epsilon =  Eigen::VectorXd::Zero(feature_dim); ///< bound on feature consistency score, determines if inlier/outlier
      bool gravity_guided = false; ///< whether to use gravity-guided prior
    };
  public:
    DistanceFeatureSimilarity(const Params& params)
    : params_(params) {}
    ~DistanceFeatureSimilarity() = default;

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

  using DistanceFeatureSimilarityPtr = std::shared_ptr<DistanceFeatureSimilarity>;

} // ns invariants
} // ns clipper