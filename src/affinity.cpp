/**
 * @file affinity.h
 * @brief Create an affinity matrix by scoring association consistency
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#include "clipper/affinity.h"

namespace clipper {

/**
 * @brief      Maps a flat index to coordinate of a square symmetric matrix
 *
 * @param[in]  k     The flat index to find the corresponding r,c of
 * @param[in]  n     Dimension of the square, symmetric matrix
 *
 * @return     row, col of a matrix corresponding to flat index k
 */
static std::tuple<size_t,size_t> k2ij(size_t k, size_t n)
{
  k += 1;

  const size_t l = n*(n-1)/2 - k;
  const size_t o = std::floor( (std::sqrt(1 + 8*l) - 1) / 2. );
  const size_t p = l - o*(o+1)/2;
  const size_t i = n - (o + 1);
  const size_t j = n - p;
  return {i-1, j-1};
}

// ----------------------------------------------------------------------------

PairMC scorePairwiseConsistency(invariants::PairwiseInvariant& invariant,
                      const invariants::Data& D1, const invariants::Data& D2,
                      Association& A, bool parallelize)
{
  if (A.size() == 0) A = createAllToAll(D1.cols(), D2.cols());

  const size_t m = A.rows();

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(m,m);
  Eigen::MatrixXd C = Eigen::MatrixXd::Ones(m,m);

#pragma omp parallel for shared(A, D1, D2, M, C) if(parallelize)
  for (size_t k=0; k<m*(m-1)/2; ++k) {
    size_t i, j; std::tie(i, j) = k2ij(k, m);

    if (A(i,0) == A(j,0) || A(i,1) == A(j,1)) {
      C(i,j) = C(j,i) = 0; // distinctness constraint
      continue;
    }

    //
    // Evaluate the consistency of geometric invariants associated with ei, ej
    //

    // points to extract invariant from in D1
    const auto& d1i = D1.col(A(i,0));
    const auto& d1j = D1.col(A(j,0));

    // points to extract invariant from in D2
    const auto& d2i = D2.col(A(i,1));
    const auto& d2j = D2.col(A(j,1));

    const double scr = invariant(d1i, d1j, d2i, d2j);
    if (scr > 0) M(i,j) = M(j,i) = scr;
    else C(i,j) = C(j,i) = 0; // inconsistency constraint
  }

  // make diagonals one
  M += Eigen::MatrixXd::Identity(m,m);

  return {M, C};
}

// ----------------------------------------------------------------------------

PairMC scorePairwiseConsistency(const invariants::PairwiseInvariantPtr& invariant,
                      const invariants::Data& D1, const invariants::Data& D2,
                      Association& A, bool parallelize)
{
  return scorePairwiseConsistency(*invariant, D1, D2, A, parallelize);
}

} // ns clipper