/**
 * @file known_scale_pointcloud.cpp
 * @brief Scores geometric invariants of a point cloud with known scale
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 */

#include <iostream>
#include <tuple>

#include "clipper/invariants/known_scale_pointcloud.h"

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

PairMC KnownScalePointCloud::createAffinityMatrix(const Data& D1, const Data& D2, Association& A)
{
  if (A.size() == 0) A = createAllToAll(D1.cols(), D2.cols());

  const size_t m = A.rows();

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(m,m);
  Eigen::MatrixXd C = Eigen::MatrixXd::Ones(m,m);

#pragma omp parallel for default(none) shared(A, D1, D2, M, C)
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

double KnownScalePointCloud::scoreInvariantConsistency(
  const Datum& d1i, const Datum& d1j, const Datum& d2i, const Datum& d2j)
{

  // distance between two points in the same cloud
  const double l1 = (d1i - d1j).norm();
  const double l2 = (d2i - d2j).norm();

  // normalized consistency score
  const double mean = (l1 + l2) / 2;
  const double c = std::abs(l1 - l2);// / mean;

  return (c<params_.epsilon) ? std::exp(-0.5*c*c/(params_.sigma*params_.sigma)) : 0;
}

} // ns invariants
} // ns clipper