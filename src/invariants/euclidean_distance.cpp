/**
 * @file euclidean_distance.cpp
 * @brief Pairwise Euclidean distance geometric invariant
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#include "clipper/invariants/euclidean_distance.h"

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

  // consistency score
  const double c = std::abs(l1 - l2);

  return (c<params_.epsilon) ? std::exp(-0.5*c*c/(params_.sigma*params_.sigma)) : 0;
}

} // ns invariants
} // ns clipper
