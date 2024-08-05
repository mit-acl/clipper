/**
 * @file distance_l2_similarity.cpp
 * @brief Pairwise/single invariant using pairwise Euclidean distance and l2 similarity
 * @author Mason Peterson <masonbp@mit.edu>
 * @date 5 July 2024
 */

#include "clipper/invariants/distance_cos_vol_similarity.h"

namespace clipper {
namespace invariants {

double DistanceCosVolSimilarity::single_similarity(const Datum& ai, const Datum& bi)
{
  if (params_.feature_dim == 0) {
    return 1.0;
  }

  const Datum ai_vol = ai.segment(params_.point_dim, 1);
  const Datum bi_vol = bi.segment(params_.point_dim, 1);

  const Datum ai_feat = ai.segment(params_.point_dim + 1, params_.feature_dim);
  const Datum bi_feat = bi.segment(params_.point_dim + 1, params_.feature_dim);

  const float vol_ratio_1 = ai_vol.norm() / bi_vol.norm();
  const float vol_ratio_2 = bi_vol.norm() / ai_vol.norm();

  const float cosine_score = (ai_feat.transpose() * bi_feat)(0) / (ai_feat.norm() * bi_feat.norm());
  
  if (vol_ratio_1 < vol_ratio_2) {
    // return std::pow(cosine_score * std::pow(vol_ratio_1, 2.0), 1.0 / 3.0);
    return std::pow(cosine_score * vol_ratio_1, 1.0 / 2.0);
    // return std::max(vol_ratio_1, cosine_score);
  }

//   return std::pow(cosine_score * std::pow(vol_ratio_2, 2.0), 1.0 / 3.0);
return std::pow(cosine_score * vol_ratio_2, 1.0 / 2.0);
// return std::max(vol_ratio_2, cosine_score);
}

} // ns invariants
} // ns clipper
