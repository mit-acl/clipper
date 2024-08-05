/**
 * @file distance_semantic_similarity.h
 * @brief Pairwise/single invariant using pairwise Euclidean distance and min/max single feature similarity
 * @author Lucas Jia <yixuany@mit.edu>
 * @date 4 August 2024
 */

#pragma once

#include "clipper/invariants/distance_pairwise_and_single.h"
#include <iostream>

namespace clipper {
namespace invariants {

  /**
   * @brief      Specialization of PairwiseInvariant to GravityConstrained distance in
   *             the real numbers using the 2-norm as the invariant.
   */
  class DistanceSemanticSimilarity : public DistancePairwiseAndSingle
  {
  public:
    DistanceSemanticSimilarity(const Params& params)
    : DistancePairwiseAndSingle(params), params_(params)
    {}
    ~DistanceSemanticSimilarity() = default;

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

  using DistanceSemanticSimilarityPtr = std::shared_ptr<DistanceSemanticSimilarity>;

} // ns invariants
} // ns clipper