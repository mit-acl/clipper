/**
 * @file distance_l2_similarity.cpp
 * @brief Pairwise/single invariant using pairwise Euclidean distance and l2 similarity
 * @author Mason Peterson <masonbp@mit.edu>
 * @date 5 July 2024
 */

#include "clipper/invariants/distance_scale_similarity.h"

namespace clipper {
namespace invariants {

double DistanceScaleSimilarity::single_similarity(const Datum& ai, const Datum& bi)
{
  if (params_.feature_dim == 0) {
    return 1.0;
  }

  const Datum ai_feat = ai.segment(params_.point_dim, params_.feature_dim);
  const Datum bi_feat = bi.segment(params_.point_dim, params_.feature_dim);
    

  const float scale_ratio_1 = ai_feat.norm() / bi_feat.norm();
  const float scale_ratio_2 = bi_feat.norm() / ai_feat.norm();
  if (scale_ratio_1 < scale_ratio_2) {
    return scale_ratio_1;
  }

  return scale_ratio_2;
}

} // ns invariants
} // ns clipper
