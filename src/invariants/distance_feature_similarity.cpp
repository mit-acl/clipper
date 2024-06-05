/**
 * @file gravity_constrained_distance.cpp
 * @brief Pairwise Volume gravity constrained distance geometric invariant
 * @author Mason Peterson <masonbp@mit.edu>
 * @date 11 Apr 2024
 */

#include "clipper/invariants/distance_feature_similarity.h"

#define SQRT_TWO_THIRDS 0.81649658092
#define SQRT_ONE_THIRD 0.57735026919 

namespace clipper {
namespace invariants {

double DistanceFeatureSimilarity::operator()(const Datum& ai, const Datum& aj,
                                     const Datum& bi, const Datum& bj)
{

  // distance between two points in the same cloud
  const double l1 = (ai.head(params_.point_dim) - aj.head(params_.point_dim)).norm();
  const double l2 = (bi.head(params_.point_dim) - bj.head(params_.point_dim)).norm();
  // enforce minimum distance criterion -- if points in the same dataset
  // are too close, then this pair of associations cannot be selected
  if (params_.mindist > 0 && (l1 < params_.mindist || l2 < params_.mindist)) {
    return 0.0;
  }

  // compute feature similarity scores
  Eigen::VectorXd feature_score_i = Eigen::VectorXd::Zero(params_.feature_dim);
  Eigen::VectorXd feature_score_j = Eigen::VectorXd::Zero(params_.feature_dim);
  // for each feature score, similarity score is the ratio of the smaller to the larger
  for (int i=0; i<(int) params_.feature_dim; i++) {
    feature_score_i(i) = ai(params_.point_dim + i) < bi(params_.point_dim + i) ? 
      ai(params_.point_dim + i) / bi(params_.point_dim + i) : 
      bi(params_.point_dim + i) / ai(params_.point_dim + i);
    feature_score_j(i) = aj(params_.point_dim + i) < bj(params_.point_dim + i) ? 
      aj(params_.point_dim + i) / bj(params_.point_dim + i) : 
      bj(params_.point_dim + i) / aj(params_.point_dim + i);
  }
  // if any feature score is below the epsilon threshold, return 0
  if ((feature_score_i.array() < params_.feature_epsilon.array()).any() || 
      (feature_score_j.array() < params_.feature_epsilon.array()).any()) {
    return 0.0;
  }

  // distance similarity score (including gravity-guidance)

  double distance_score = 0.0;
  if (params_.gravity_guided) {
    // gravity-guided distance similarity
    const double xy_dist1 = (ai.head(2) - aj.head(2)).norm();
    const double xy_dist2 = (bi.head(2) - bj.head(2)).norm();
    const double z_diff1 = ai(2) - aj(2);
    const double z_diff2 = bi(2) - bj(2);

    // consistency score
    const double c_xy = std::abs(xy_dist1 - xy_dist2);
    const double c_z = std::abs(z_diff1 - z_diff2);

    if (c_xy > SQRT_TWO_THIRDS*params_.epsilon || c_z > SQRT_ONE_THIRD*params_.epsilon) {
      return 0.0;
    }

    distance_score = std::exp(-0.5*(c_xy*c_xy/(2.0/3.0*params_.sigma*params_.sigma) + 
        c_z*c_z/(params_.sigma*params_.sigma/3.0)));

  } else {
    // standard distance similarity
    const double c = std::abs(l1 - l2);
    if (c > params_.epsilon) {
      return 0.0;
    }

    distance_score = std::exp(-0.5*c*c/(params_.sigma*params_.sigma));
  }

  double fused_score = 0;
  if (params_.feature_dim > 0) {
    switch (params_.similarity_fusion_method) {
      case SimilarityFusionMethod::GEOMETRIC_MEAN: {
        double dist_score_pow = std::pow(distance_score, params_.distance_fusion_weight);
        fused_score = std::pow(dist_score_pow * feature_score_i.prod() * feature_score_j.prod(), 1.0/(params_.distance_fusion_weight + 2.0*params_.feature_dim));
        break;
      }
      case SimilarityFusionMethod::ARITHMETIC_MEAN: {
        fused_score = (params_.distance_fusion_weight * distance_score + feature_score_i.sum() + feature_score_j.sum()) / (params_.distance_fusion_weight + 2.0*params_.feature_dim);
        break;
      }
      case SimilarityFusionMethod::PRODUCT: {
        fused_score = distance_score * feature_score_i.prod() * feature_score_j.prod();
        break;
      }
      default: {
        // Should not reach here!
        break;
      }
    }
  }

  return params_.feature_dim > 0 ? fused_score : distance_score;
}

} // ns invariants
} // ns clipper
