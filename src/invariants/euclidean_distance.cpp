/**
 * @file euclidean_distance.cpp
 * @brief Pairwise Euclidean distance geometric invariant
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#include "clipper/invariants/euclidean_distance.h"

#include <boost/math/distributions/non_central_chi_squared.hpp>

namespace clipper {
namespace invariants {

double EuclideanDistance::operator()(const Datum& ai, const Datum& aj,
                                     const Datum& bi, const Datum& bj)
{

  // distance between two points in the same cloud
  const double l1 = (ai - aj).norm();
  const double l2 = (bi - bj).norm();

  // enforce minimum distance criterion -- if points in the same dataset
  // are too close, then this pair of associations cannot be selected
  if (params_.mindist > 0 && (l1 < params_.mindist || l2 < params_.mindist)) {
    return 0.0;
  }

  // distance between points should be nearly the same
  const double c = std::abs(l1 - l2);

  // if not, then bail
  if (c > params_.epsilon) return 0;

  if (params_.use_ncx2) {
    const double sigmay2 = 2 * params_.sigma * params_.sigma;

    static constexpr int k = 3;
    const double lambda = (l1 * l1) / sigmay2;
    boost::math::non_central_chi_squared ncx2(k, lambda);

    const double smax = boost::math::pdf(ncx2, lambda) / sigmay2;

    const double x = (l2 * l2) / sigmay2;
    double s = boost::math::pdf(ncx2, x) / sigmay2 / smax;

    if (s>1) s = 1;

    return s;
  }

  // consistency score
  return std::exp(-0.5*c*c/(params_.sigma*params_.sigma));
}

} // ns invariants
} // ns clipper
