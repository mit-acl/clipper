/**
 * @file roman.h
 * @brief ROMAN optionally incorporating gravity + cosine/ratio feature similarity
 * @author Mason Peterson, <masonbp@mit.edu>, Lucas Jia <yixuany@mit.edu>, Yulun Tian <yulun@mit.edu>
 * @date 3 October 2024
 */

#pragma once

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

  /**
   * @brief      Specialization of PairwiseAndSingleInvariant to GravityConstrained distance in
   *             the real numbers using the 2-norm as the invariant.
   */
  class ROMAN : public PairwiseAndSingleInvariant
  {
  public:
    enum SimilarityFusionMethod {
      GEOMETRIC_MEAN,
      ARITHMETIC_MEAN,
      PRODUCT
    };
    struct Params
    {
      uint32_t point_dim = 3; ///< dimension of points (2 or 3)
      uint32_t ratio_feature_dim = 0; ///< number of ratio features (e.g., volume)
      uint32_t cos_feature_dim = 0; ///< number of features used for cosine similarity
      SimilarityFusionMethod fusion_method = SimilarityFusionMethod::GEOMETRIC_MEAN; ///< which method to use to fuse distance and feature similarities

      double sigma = 0.01; ///< spread / "variance" of exponential kernel
      double epsilon = 0.06; ///< bound on consistency score, determines if inlier/outlier
      double mindist = 0.0; ///< minimum allowable distance between inlier points in the same dataset

      double distance_weight = 1.0; ///< weight of pairwise similarity in single/pairwise fusion
      double ratio_weight = 1.0; ///< weight of cosine similarity in single similarity fusion
      double cosine_weight = 1.0; ///< weight of cosine similarity in single similarity fusion
      
      Eigen::VectorXd ratio_epsilon =  Eigen::VectorXd::Zero(ratio_feature_dim); ///< bound on feature ratio score, determines if inlier/outlier
      double cosine_min = 0.85; ///< cosine similarity scaled so that cosine_min maps to 0.0 similarity score
      double cosine_max = 0.95; ///< cosine similarity scaled so that cosine_max maps to 1.0 similarity score
      // bool cosine_normalized = false; ///< option to speed up by sending in pre-normalized features
      
      bool gravity_guided = false; ///< whether to use gravity-guided prior
      double gravity_unc_ang_rad = 0.0; ///< uncertainty adjustment for gravity direction in radians
      
      bool drift_aware = false; ///< experimental drift aware
      bool drift_scale_sigma = false;
      double drift_scale = 0.1;
    };
  public:
    ROMAN(const Params& params)
    : params_(params) 
    {
      gravity_unc_ang_cos_ = std::cos(params_.gravity_unc_ang_rad);
      gravity_unc_ang_sin_ = std::sin(params_.gravity_unc_ang_rad);
    }
    ~ROMAN() = default;

    /**
     * @brief      Functor for pairwise invariant scoring function
     *
     * @param[in]  ai    Element i from dataset 1
     * @param[in]  aj    Element j from dataset 1
     * @param[in]  bi    Element i from dataset 2
     * @param[in]  bj    Element j from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi) and (aj,bj)
     */
    double pairwise_similarity(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) override;

    /**
     * @brief      Functor for fusing the pairwise and single scores
     *
     * @param[in]  pair_ij    Score for pair of associations
     * @param[in]  single_i   Single-association score for i
     * @param[in]  single_j   Single-association score for j
     *
     * @return     The consistency score for the fused pairwise and single scores
     */
    virtual double pairwise_single_fusion(const double& pair_ij, const double& single_i, const double& single_j) override;

    /**
     * @brief      Functor for the scoring of a single association
     *
     * @param[in]  ai    Element i from dataset 1
     * @param[in]  bi    Element i from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi)
     */
    virtual double single_similarity(const Datum& ai, const Datum& bi) override;

  private:
    Params params_;
    double gravity_unc_ang_cos_;
    double gravity_unc_ang_sin_;
  };

  using ROMANPtr = std::shared_ptr<ROMAN>;

} // ns invariants
} // ns clipper