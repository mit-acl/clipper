/**
 * @file distance_contrastive_similarity.cpp
 * @brief Pairwise/single invariant using pairwise Euclidean distance and cosine similarity
 * @author Lucas Jia <yixuany@mit.edu>
 * @date 5 July 2024
 */

#include "clipper/invariants/distance_contrastive_similarity.h"

namespace clipper {
namespace invariants {

double DistanceContrastiveSimilarity::single_similarity(const Datum& ai, const Datum& bi)
{
  if (params_.feature_dim == 0) {
    return 1.0;
  }

  const Datum ai_feat = ai.segment(params_.point_dim, params_.feature_dim);
  const Datum bi_feat = bi.segment(params_.point_dim, params_.feature_dim);
  double cosine_similarity = (ai_feat.transpose() * bi_feat)(0) / (ai_feat.norm() * bi_feat.norm());
  return std::exp( cosine_similarity / 0.07 );

} // ns invariants
} // ns clipper
}