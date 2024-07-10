/**
 * @file distance_min_max_similarity.cpp
 * @brief Pairwise/single invariant using pairwise Euclidean distance and min/max single feature similarity
 * @author Mason Peterson <masonbp@mit.edu>
 * @date 5 July 2024
 */

#include "clipper/invariants/distance_min_max_similarity.h"

namespace clipper {
namespace invariants {

double DistanceMinMaxSimilarity::single_similarity(const Datum& ai, const Datum& bi)
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

  return std::pow(score.prod(), 1.0/params_.feature_dim);
}

} // ns invariants
} // ns clipper
