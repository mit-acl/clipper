/**
 * @file affinity.h
 * @brief Create an affinity matrix by scoring association consistency
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include <memory>

#include <Eigen/Dense>

#include "clipper/invariants/abstract.h"
#include "clipper/association.h"

namespace clipper {

  using PairMC = std::pair<Eigen::MatrixXd, Eigen::MatrixXd>; ///< return type

  /**
   * @brief      Creates an affinity matrix containing consistency scores for
   *             each of the m pairwise associations listed in matrix A.
   *
   * @param      invariant    The geometric invariant to use for scoring
   * @param[in]  D1           Dataset 1 of n1 d-dim elements (dxn1)
   * @param[in]  D2           Dataset 2 of n2 d-dim elements (dxn2)
   * @param      A            Associations to score (mx2)
   * @param[in]  parallelize  Should parallelization be used (almost always true)
   *
   * @return     Affinity matrix M and constraint matrix C
   */
  PairMC scorePairwiseConsistency(invariants::PairwiseInvariant& invariant,
                        const invariants::Data& D1, const invariants::Data& D2,
                        Association& A, bool parallelize = true);

  PairMC scorePairwiseConsistency(const invariants::PairwiseInvariantPtr& invariant,
                        const invariants::Data& D1, const invariants::Data& D2,
                        Association& A, bool parallelize = true)
  {
    return scorePairwiseConsistency(*invariant, D1, D2, A, parallelize);
  }

} // ns clipper