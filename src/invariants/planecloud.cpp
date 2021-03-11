/**
 * @file planecloud.cpp
 * @brief Scores geometric invariants of a set of planes
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#include <iostream>

#include "clipper/utils.h"
#include "clipper/invariants/planecloud.h"

namespace clipper {
namespace invariants {

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

PairMC PlaneCloud::createAffinityMatrix(const Data& D1, const Data& D2, Association& A)
{
  if (A.size() == 0) A = utils::createAllToAll(D1.cols(), D2.cols());

  const size_t m = A.rows();

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(m,m);
  Eigen::MatrixXd C = Eigen::MatrixXd::Ones(m,m);

#pragma omp parallel for shared(A, D1, D2, M, C)
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

    const double scr = scoreInvariantConsistency(d1i, d1j, d2i, d2j);
    if (scr > 0) M(i,j) = M(j,i) = scr;
    else C(i,j) = C(j,i) = 0; // inconsistency constraint
  }

  // make diagonals one
  M += Eigen::MatrixXd::Identity(m,m);

  return {M, C};
}

// ----------------------------------------------------------------------------

double PlaneCloud::scoreInvariantConsistency(
  const Datum& d1i, const Datum& d1j, const Datum& d2i, const Datum& d2j)
{

  // assumption: normals are unit length
  const double alpha1 = d1i.head<3>().transpose() * d1j.head<3>();
  const double alpha2 = d2i.head<3>().transpose() * d2j.head<3>();

  const double c = std::abs(alpha1 - alpha2);

  return (c<params_.epsilon) ? std::exp(-0.5*c*c/(params_.sigma*params_.sigma)) : 0;
}

} // ns invariants
} // ns clipper
