/**
 * @file gravity_constrained_distance.cpp
 * @brief Pairwise VolumeEuclidean distance geometric invariant
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#include "clipper/invariants/volume_euclidean_distance.h"

namespace clipper {
namespace invariants {

double VolumeEuclideanDistance::operator()(const Datum& ai, const Datum& aj,
                                     const Datum& bi, const Datum& bj)
{

  // distance between two points in the same cloud
  const int n = ai.size() - 1;
  const double l1 = (ai.head(n) - aj.head(n)).norm();
  const double l2 = (bi.head(n) - bj.head(n)).norm();
  const double volume_score_i = (ai(n) < bi(n) ? ai(n) / bi(n) : bi(n) / ai(n));
  const double volume_score_j = (aj(n) < bj(n) ? aj(n) / bj(n) : bj(n) / aj(n));
  // const double volume_score_i = sqrt(ai(n) < bi(n) ? ai(n) / bi(n) : bi(n) / ai(n));
  // const double volume_score_j = sqrt(aj(n) < bj(n) ? aj(n) / bj(n) : bj(n) / aj(n));

  // enforce minimum distance criterion -- if points in the same dataset
  // are too close, then this pair of associations cannot be selected
  if (params_.mindist > 0 && (l1 < params_.mindist || l2 < params_.mindist)) {
    return 0.0;
  }

  // consistency score
  const double c = std::abs(l1 - l2);

  return (c<params_.epsilon && volume_score_i>params_.epsilon_volume && volume_score_j>params_.epsilon_volume) ? 
    std::pow(std::exp(-0.5*c*c/(params_.sigma*params_.sigma)) * volume_score_i * volume_score_j, 1.0/3.0) : 0;
}

} // ns invariants
} // ns clipper
