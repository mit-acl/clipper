/**
 * @file distance_l2_similarity.h
 * @brief Pairwise/single invariant using pairwise Euclidean distance and min/max single feature similarity
 * @author Lucas Jia <yixuany@mit.edu>
 * @date 5 July 2024
 */

#pragma once

#include "clipper/invariants/distance_pairwise_and_single.h"

namespace clipper {
namespace invariants {

  /**
   * @brief      Specialization of PairwiseInvariant to GravityConstrained distance in
   *             the real numbers using the 2-norm as the invariant.
   */
  class DistanceCosScaleSimilarity : public DistancePairwiseAndSingle
  {
  public:
    DistanceCosScaleSimilarity(const Params& params)
    : DistancePairwiseAndSingle(params), params_(params)
    {}
    ~DistanceCosScaleSimilarity() = default;

    /**
     * @brief      Functor for the scoring of a single association
     *
     * @param[in]  ai    Element i from dataset 1
     * @param[in]  bi    Element i from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi)
     */
    virtual double single_similarity(const Datum& ai, const Datum& bi) override;

  private:
    Params params_;
  };

  using DistanceCosScaleSimilarityPtr = std::shared_ptr<DistanceCosScaleSimilarity>;

} // ns invariants
} // ns clipper