/**
 * @file pointnormal_distance.cpp
 * @brief Pairwise geometric invariant between point-normals (e.g., planes)
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#include "clipper/invariants/pointnormal_distance.h"

namespace clipper {
namespace invariants {

double PointNormalDistance::operator()(const Datum& ai, const Datum& aj,
                                       const Datum& bi, const Datum& bj)
{
  // point distance
  const double l1 = (ai.head<3>() - aj.head<3>()).norm();
  const double l2 = (bi.head<3>() - bj.head<3>()).norm();

  // normal distance
  const double alpha1 = std::acos(ai.tail<3>().transpose() * aj.tail<3>());
  const double alpha2 = std::acos(bi.tail<3>().transpose() * bj.tail<3>());

  // check consistency
  const double dp = std::abs(l1 - l2);
  const double dn = std::abs(alpha1 - alpha2);

  if (dp < params_.epsp && dn < params_.epsn) {
    const double sp = std::exp(-0.5*dp*dp/(params_.sigp*params_.sigp));
    const double sn = std::exp(-0.5*dn*dn/(params_.sign*params_.sign));
    return sp * sn;
  } else {
    return 0.0;
  }
}

} // ns invariants
} // ns clipper
