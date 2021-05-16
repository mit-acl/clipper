/**
 * @file plane_distance.cpp
 * @brief Pairwise cosine similarity geometric invariant
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#include "clipper/invariants/plane_distance.h"

namespace clipper {
namespace invariants {

double PlaneDistance::operator()(const Datum& ai, const Datum& aj,
                                     const Datum& bi, const Datum& bj)
{
  const double alpha1 = ai.head<3>().transpose() * aj.head<3>();
  const double alpha2 = bi.head<3>().transpose() * bj.head<3>();

  // TODO need to also add distance to plane

  const double c = std::abs(alpha1 - alpha2);

  return (c<params_.epsilon) ? std::exp(-0.5*c*c/(params_.sigma*params_.sigma)) : 0;
}

} // ns invariants
} // ns clipper
