/**
 * @file abstract.h
 * @brief Abstract base class for geometric invariants
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

namespace clipper {
namespace invariants {

  using Data = Eigen::MatrixXd;
  using Datum = Eigen::VectorXd;

  class Invariant {
  public:
    virtual ~Invariant() {}
  };

  class PairwiseInvariant : public Invariant
  {
  public:
    virtual ~PairwiseInvariant() {}

    virtual double operator()(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) = 0;
  };

  using PairwiseInvariantConstPtr = std::shared_ptr<const PairwiseInvariant>;

} // ns invariants
} // ns clipper