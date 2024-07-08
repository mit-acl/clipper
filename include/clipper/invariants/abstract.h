/**
 * @file abstract.h
 * @brief Abstract base class for geometric invariants
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

namespace clipper {
namespace invariants {

  using Data = Eigen::MatrixXd;
  using Datum = Eigen::VectorXd;

  /**
   * @brief      Abstract base clase for all invariant types.
   *             An invariant is a function which yields the same output for
   *             the same subset of elements in a dataset under transformation.
   *
   *             For example, in R^n under SE(3), the most natural invariant is
   *             the norm since ||q|| = ||T p||. So for two points in a dataset,
   *             if ||qi - qj|| = ||Tpi - Tpj|| then there is a high degreee of
   *             consistency between (qi, pi) and (qj, pj).
   *
   *             The Invariant datatype implements an invariant for a specific
   *             domain and expected transformation. Further, this datatype is
   *             used to score consistency between data using the implemented
   *             invariant.
   */
  class Invariant {
  public:
    virtual ~Invariant() = default;
  };

  using InvariantPtr = std::shared_ptr<Invariant>;

  /**
   * @brief      A pairwise invariant uses pairs of points to score association
   *             consistency. This class implements a real-valued invariant
   *             scoring function that takes the form
   *
   *                f : A x A x A x A -> R
   *
   *             where A represents the specific domain of interest (e.g., R^n)
   *             Note that f is not the invariant itself, but the invariant
   *             _scoring_ function, which uses the invariant to assses
   *             consistency.
   */
  class PairwiseInvariant : public Invariant
  {
  public:
    virtual ~PairwiseInvariant() = default;

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
    virtual double operator()(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) = 0;
  };

  using PairwiseInvariantPtr = std::shared_ptr<PairwiseInvariant>;

  /**
   * @brief      An extension of the pairwise invariant that also includes scoring the similarity 
   *             between each individual association.
   */
  class PairwiseAndSingleInvariant : public PairwiseInvariant
  {
  public:
    virtual ~PairwiseAndSingleInvariant() = default;

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
    double operator()(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) 
    {
      return pairwise_similarity(ai, aj, bi, bj);
    }

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
    virtual double pairwise_similarity(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) = 0;

    /**
     * @brief      Functor for the scoring of a single association
     *
     * @param[in]  ai    Element i from dataset 1
     * @param[in]  bi    Element i from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi)
     */
    virtual double single_similarity(const Datum& ai, const Datum& bi) = 0;

    /**
     * @brief      Functor for fusing the pairwise and single scores
     *
     * @param[in]  pair_ij    Score for pair of associations
     * @param[in]  single_i   Single-association score for i
     * @param[in]  single_j   Single-association score for j
     *
     * @return     The consistency score for the fused pairwise and single scores
     */
    virtual double pairwise_single_fusion(const double& pair_ij, 
      const double& single_i, const double& single_j) = 0;
  };

  using PairwiseAndSingleInvariantPtr = std::shared_ptr<PairwiseAndSingleInvariant>;

} // ns invariants
} // ns clipper