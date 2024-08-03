/**
 * @file distance_l2_similarity.cpp
 * @brief Pairwise/single invariant using pairwise Euclidean distance and l2 similarity
 * @author Lucas Jia <yixuany@mit.edu>
 * @date 5 July 2024
 */

#include "clipper/invariants/distance_contrast_scale_similarity.h"

namespace clipper {
namespace invariants {

double DistanceContrastScaleSimilarity::single_similarity(const Datum& ai, const Datum& bi)
{
  if (params_.feature_dim == 0) {
    return 1.0;
  }

  const Datum ai_scale_feat = ai.segment(params_.point_dim, params_.feature_dim);
  const Datum bi_scale_feat = bi.segment(params_.point_dim, params_.feature_dim);

  const Datum ai_feat = ai.segment(params_.point_dim + params_.feature_dim, params_.cos_feature_dim);
  const Datum bi_feat = bi.segment(params_.point_dim + params_.feature_dim, params_.cos_feature_dim);

  const float scale_ratio_1 = ai_scale_feat.norm() / bi_scale_feat.norm();
  const float scale_ratio_2 = bi_scale_feat.norm() / ai_scale_feat.norm();

  const float cosine_score = ((ai_feat.transpose() * bi_feat)(0) / (ai_feat.norm() * bi_feat.norm()) + 1.0) / 2.0;
  const float cosine_score_clipped = cosine_score >= 0.5 ? 1.0 : cosine_score;
  if (scale_ratio_1 < scale_ratio_2) {
    return std::pow(cosine_score_clipped * std::pow(scale_ratio_1, 4.0), 1.0/5.0);
  }

  return std::pow(cosine_score_clipped * std::pow(scale_ratio_2, 4.0), 1.0/5.0);
}

} // ns invariants
} // ns clipper
