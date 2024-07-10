/**
 * @file distance_min_max_similarity.cpp
 * @brief Pairwise/single invariant using pairwise Euclidean distance and min/max single feature similarity
 * @author Mason Peterson <masonbp@mit.edu>
 * @date 5 July 2024
 */

#include "clipper/invariants/distance_min_max_similarity.h"
#include <iostream>

#define SQRT_TWO_THIRDS 0.81649658092
#define SQRT_ONE_THIRD 0.57735026919 

namespace clipper {
namespace invariants {

double DistancePairwiseAndSingle::pairwise_similarity(const Datum& ai, const Datum& aj,
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
    } else {
      return std::exp(-0.5*(c_xy*c_xy/(2.0/3.0*params_.sigma*params_.sigma) + 
          c_z*c_z/(params_.sigma*params_.sigma/3.0)));
    }

  } else {
    // standard distance similarity
    const double c = std::abs(l1 - l2);
    if (c > params_.epsilon) {
      return 0.0;
    } else {
      return std::exp(-0.5*c*c/(params_.sigma*params_.sigma));
    }

  }

}

double DistancePairwiseAndSingle::pairwise_single_fusion(
    const double& pair_ij, const double& single_i, const double& single_j)
{
  if (params_.feature_dim > 0) {
    switch (params_.similarity_fusion_method) {
      case SimilarityFusionMethod::GEOMETRIC_MEAN: {
        double dist_score_pow = std::pow(pair_ij, params_.distance_fusion_weight);
        return std::pow(dist_score_pow * single_i * single_j, 1.0/(params_.distance_fusion_weight + 2.0));
        break;
      }
      case SimilarityFusionMethod::ARITHMETIC_MEAN: {
        return (params_.distance_fusion_weight * pair_ij + single_i + single_j) / (params_.distance_fusion_weight + 2.0);
        break;
      }
      case SimilarityFusionMethod::PRODUCT: {
        return pair_ij * single_i * single_j;
        break;
      }
      default: {
        // Should not reach here!
        return 0.0;
        break;
      }
    }
  } else {
    return pair_ij;
  }
}

} // ns invariants
} // ns clipper
