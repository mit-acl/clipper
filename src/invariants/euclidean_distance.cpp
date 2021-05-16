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

  // normalized consistency score
  const double mean = (l1 + l2) / 2.0;
  const double c = std::abs(l1 - l2);// / mean;

  return (c<params_.epsilon) ? std::exp(-0.5*c*c/(params_.sigma*params_.sigma)) : 0;
}

} // ns invariants
} // ns clipper
