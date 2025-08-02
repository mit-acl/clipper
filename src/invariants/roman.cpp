/**
 * @file roman.cpp
 * @brief ROMAN optionally incorporating gravity + cosine/ratio feature similarity
 * @author Mason Peterson, <masonbp@mit.edu>, Lucas Jia <yixuany@mit.edu>, Yulun Tian <yulun@mit.edu>
 * @date 3 October 2024
 */

#include "clipper/invariants/roman.h"
#include <iostream>

#define SQRT_TWO_THIRDS 0.81649658092
#define SQRT_ONE_THIRD 0.57735026919 

namespace clipper {
namespace invariants {

double ROMAN::pairwise_similarity(const Datum& ai, const Datum& aj,
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

  // handle drift aware epsilon
  const double epsilon = params_.drift_aware ? 
    std::max(params_.epsilon, params_.epsilon*params_.drift_scale*0.5*(l1 + l2)) :
    params_.epsilon;
  const double sigma = params_.drift_aware && params_.drift_scale_sigma ? 
    std::max(params_.sigma, params_.sigma*params_.drift_scale*0.5*(l1 + l2)) :
    params_.sigma;

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

    double sigma_xy = sigma;
    double sigma_z = sigma;
    double epsilon_xy = epsilon;
    double epsilon_z = epsilon;
    
    if (params_.gravity_unc_ang_rad > 0.0) {
      const double xy_dist_mean = 0.5*(xy_dist1 + xy_dist2);
      const double z_dist_mean = 0.5*(std::abs(z_diff1) + std::abs(z_diff2));

      // adjust sigma and epsilon based on gravity uncertainty
      sigma_xy += std::abs(xy_dist_mean * gravity_unc_ang_cos_ - xy_dist_mean);
      sigma_z += std::abs(z_dist_mean * gravity_unc_ang_sin_);
      epsilon_xy += std::abs(xy_dist_mean * gravity_unc_ang_cos_ - xy_dist_mean);
      epsilon_z += std::abs(z_dist_mean * gravity_unc_ang_sin_);
    }

    if (c_xy > SQRT_TWO_THIRDS*epsilon_xy || c_z > SQRT_ONE_THIRD*epsilon_z) {
      return 0.0;
    } else {
      return std::exp(-0.5*(c_xy*c_xy/(2.0/3.0*sigma_xy*sigma_xy) + 
          c_z*c_z/(sigma_z*sigma_z/3.0)));
    }

  } else {
    // standard distance similarity
    const double c = std::abs(l1 - l2);
    if (c > epsilon) {
      return 0.0;
    } else {
      return std::exp(-0.5*c*c/(sigma*sigma));
    }

  }

}

double ROMAN::pairwise_single_fusion(
    const double& pair_ij, const double& single_i, const double& single_j)
{
  if (params_.ratio_feature_dim > 0 || params_.cos_feature_dim > 0) {
    switch (params_.fusion_method) {
      case SimilarityFusionMethod::GEOMETRIC_MEAN: {
        double dist_score_pow = std::pow(pair_ij, params_.distance_weight);
        return std::pow(dist_score_pow * single_i * single_j, 1.0/(params_.distance_weight + 2.0));
        break;
      }
      case SimilarityFusionMethod::ARITHMETIC_MEAN: {
        return (params_.distance_weight * pair_ij + single_i + single_j) / (params_.distance_weight + 2.0);
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

double ROMAN::single_similarity(const Datum& ai, const Datum& bi)
{
  double cosine_score_scaled = 0.0;
  double ratio_score = 0.0;

  if (params_.cos_feature_dim == 0 && params_.ratio_feature_dim == 0) {
    return 1.0; // no features, so return 1.0
  }

  if (params_.cos_feature_dim > 0) {
    const Datum ai_feat = ai.segment(params_.point_dim + params_.ratio_feature_dim, params_.cos_feature_dim);
    const Datum bi_feat = bi.segment(params_.point_dim + params_.ratio_feature_dim, params_.cos_feature_dim);

    const double cosine_score  = (ai_feat.transpose() * bi_feat)(0) / (ai_feat.norm() * bi_feat.norm());

    if (cosine_score >= params_.cosine_max)
      cosine_score_scaled = 1.0;
    else if (cosine_score <= params_.cosine_min)
      return 0.0; // geometric mean fusion means the fused score will be 0
    else
      cosine_score_scaled = (cosine_score - params_.cosine_min) / (params_.cosine_max - params_.cosine_min);
    if (params_.ratio_feature_dim == 0) {
      return cosine_score_scaled;
    }
  }

  if (params_.ratio_feature_dim > 0) {
    // compute ratio feature similarity scores
    Eigen::VectorXd ratio_scores = Eigen::VectorXd::Zero(params_.ratio_feature_dim);

    // for each feature score, similarity score is the ratio of the smaller to the larger
    for (int i=0; i<(int) params_.ratio_feature_dim; i++) {
      ratio_scores(i) = ai(params_.point_dim + i) < bi(params_.point_dim + i) ? 
        ai(params_.point_dim + i) / bi(params_.point_dim + i) : 
        bi(params_.point_dim + i) / ai(params_.point_dim + i);
    }

    if ((ratio_scores.array() < params_.ratio_epsilon.array()).any()) {
      return 0.0;
    }

    ratio_score = std::pow(ratio_scores.prod(), 1.0 / params_.ratio_feature_dim);
    if (params_.cos_feature_dim == 0) {
      return ratio_score;
    }
  }  

  return std::pow(std::pow(ratio_score, params_.ratio_weight) * std::pow(cosine_score_scaled, params_.cosine_weight), 1.0/(params_.ratio_weight + params_.cosine_weight));
}


} // ns invariants
} // ns clipper
