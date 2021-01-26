/**
 * @file clipper.h
 * @brief CLIPPER data association framework
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#pragma once

#include <tuple>

#include <Eigen/Dense>

#include "clipper/invariants/abstract.h"
#include "clipper/find_dense_cluster.h"

namespace clipper {

  /**
   * @brief      Convenience function to select inlier associations
   *
   * @param[in]  soln  The solution of the dense cluster
   * @param[in]  A     The initial set of associations
   *
   * @return     The subset of associations deemed as inliers via solution
   */
  Association selectInlierAssociations(const Solution& soln, const Association& A)
  {
    Association Ainliers = Association::Zero(soln.nodes.size(), 2);
    for (size_t i=0; i<soln.nodes.size(); ++i) {
      Ainliers.row(i) = A.row(soln.nodes[i]);
    }
    return Ainliers;
  }

  /**
   * @brief      Convenience class to use CLIPPER for data association.
   *
   * @tparam     Ti    The invariant to extract from geometric input data.
   */
  template<class Ti>
  class CLIPPER
  {
  public:
    CLIPPER(const Params& params, const typename Ti::Params& iparams)
    : params_(params), invariant_(iparams) {}
    ~CLIPPER() = default;

    Association findCorrespondences(const typename Ti::Data& D1,
      const typename Ti::Data& D2, Association A = Association())
    {
      // Create consistency graph
      std::tie(M_, C_) = invariant_.createAffinityMatrix(D1, D2, A);

      // Find a dense cluster of consistent links
      soln_ = findDenseCluster(M_, C_, params_);

      return selectInlierAssociations(soln_, A);
    }

    const Solution& getSolution() const { return soln_; }
    Eigen::MatrixXd getAffinityMatrix() const { return M_; }
    Eigen::MatrixXd getConstraintMatrix() const { return C_; }

  private:
    Params params_;
    Ti invariant_;

    // \brief Problem data from latest instance of data association
    Solution soln_; ///< solution information from CLIPPER dense cluster solver
    Eigen::MatrixXd M_; ///< affinity matrix (i.e., weighted consistency graph)
    Eigen::MatrixXd C_; ///< constraint matrix (i.e., prevents forming links)
  };

} // ns clipper