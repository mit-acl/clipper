/**
 * @file distance_cos_similarity.cpp
 * @brief Pairwise/single invariant using pairwise Euclidean distance and cosine similarity
 * @author Mason Peterson <masonbp@mit.edu>
 * @date 5 July 2024
 */

#include "clipper/invariants/distance_cos_similarity.h"

namespace clipper {
namespace invariants {

double DistanceCosSimilarity::single_similarity(const Datum& ai, const Datum& bi)
{
  if (params_.feature_dim == 0) {
    return 1.0;
  }

  const Datum ai_feat = ai.segment(params_.point_dim, params_.feature_dim);
  const Datum bi_feat = bi.segment(params_.point_dim, params_.feature_dim);
    
  return (ai_feat.transpose() * bi_feat)(0) / 
          (ai_feat.norm() * bi_feat.norm());
}

} // ns invariants
} // ns clipper
