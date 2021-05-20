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