/**
 * @file distance_semantic_similarity.cpp
 * @brief Pairwise/single invariant using pairwise Euclidean distance and cosine similarity
 * @author Lucas Jia <yixuany@mit.edu>
 * @date 4 August 2024
 */

#include "clipper/invariants/distance_semantic_similarity.h"

namespace clipper {
namespace invariants {

double DistanceSemanticSimilarity::single_similarity(const Datum& ai, const Datum& bi)
{
  if (params_.feature_dim == 0) {
    return 1.0;
  }
    
  // compute feature similarity scores
  Eigen::VectorXd score = Eigen::VectorXd::Zero(params_.feature_dim);

  // for each feature score, similarity score is the ratio of the smaller to the larger
  for (int i=0; i<(int) params_.feature_dim; i++) {
    score(i) = ai(params_.point_dim + i) < bi(params_.point_dim + i) ? 
      ai(params_.point_dim + i) / bi(params_.point_dim + i) : 
      bi(params_.point_dim + i) / ai(params_.point_dim + i);
  }

  if ((score.array() < params_.feature_epsilon.array()).any()) {
    return 0.0;
  }

  const Datum ai_feat = ai.segment(params_.point_dim + params_.feature_dim, params_.cos_feature_dim);
  const Datum bi_feat = bi.segment(params_.point_dim + params_.feature_dim, params_.cos_feature_dim);

    const float cosine_score  = (ai_feat.transpose() * bi_feat)(0) / (ai_feat.norm() * bi_feat.norm());

    float cosine_score_scaled = 0.0;

    if (cosine_score >= params_.cosine_max)
    {
        cosine_score_scaled = 1.0;
    }
    else {
        cosine_score_scaled = ( 1.0 / (params_.cosine_max - params_.cosine_min) )*(cosine_score - params_.cosine_min);
    }
    

  return std::pow(score.head(params_.feature_dim).prod() * std::pow(cosine_score_scaled, params_.cosine_weight), 1.0/(params_.feature_dim + params_.cosine_weight));
}

} // ns invariants
} // ns clipper
