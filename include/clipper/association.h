/**
 * @file association.h
 * @brief Functions related to creation and managing the association matrix
 * @author Parker Lusk <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include <Eigen/Dense>

#include "clipper/find_dense_cluster.h"

namespace clipper {

  using Association = Eigen::Matrix<int, Eigen::Dynamic, 2>;

  /**
   * @brief      Creates an all-to-all association hypothesis
   *
   * @param[in]  n1    Number of items in view 1
   * @param[in]  n2    Number of items in view 2
   *
   * @return     an (n1*n2)x2 association matrix
   */
  inline Association createAllToAll(size_t n1, size_t n2)
  {
    Association A = Association(n1*n2, 2);
    for (size_t i=0; i<n1; ++i) {
      for (size_t j=0; j<n2; ++j) {
        A(j + i*n2, 0) = i;
        A(j + i*n2, 1) = j;
      }
    }
    return A;
  }

  /**
   * @brief      Convenience function to select inlier associations
   *
   * @param[in]  soln  The solution of the dense cluster
   * @param[in]  A     The initial set of associations
   *
   * @return     The subset of associations deemed as inliers via solution
   */
  inline Association selectInlierAssociations(const Solution& soln, const Association& A)
  {
    Association Ainliers = Association::Zero(soln.nodes.size(), 2);
    for (size_t i=0; i<soln.nodes.size(); ++i) {
      Ainliers.row(i) = A.row(soln.nodes[i]);
    }
    return Ainliers;
  }

} // ns clipper