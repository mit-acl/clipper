/**
 * @file gravity_constrained_distance.cpp
 * @brief Pairwise GravityConstrained distance geometric invariant
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#include "clipper/invariants/gravity_constrained_distance.h"

namespace clipper {
namespace invariants {

double GravityConstrainedDistance::operator()(const Datum& ai, const Datum& aj,
                                     const Datum& bi, const Datum& bj)
{

  // distance between two points in the same cloud
  const double l1 = (ai.head(2) - aj.head(2)).norm();
  const double l2 = (bi.head(2) - bj.head(2)).norm();
  const double height_diff1 = ai(2) - aj(2);
  const double height_diff2 = bi(2) - bj(2);

  // enforce minimum distance criterion -- if points in the same dataset
  // are too close, then this pair of associations cannot be selected
  if (params_.mindist > 0 && (l1 < params_.mindist || l2 < params_.mindist)) {
    return 0.0;
  }

  // consistency score
  const double c_xy = std::abs(l1 - l2);
  const double c_z = std::abs(height_diff1 - height_diff2);

  return (c_xy<params_.epsilon && c_z<params_.epsilon) ? std::exp(-0.5*c_xy*c_xy/(params_.sigma*params_.sigma) + -0.5*c_z*c_z/(params_.sigma*params_.sigma)) : 0;
}

} // ns invariants
} // ns clipper
