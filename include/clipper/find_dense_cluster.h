/**
 * @file clipper.h
 * @brief CLIPPER dense cluster finding
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 */

#pragma once

#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "clipper/utils.h"

namespace clipper {

  using SpMat = Eigen::SparseMatrix<double>;

  /**
   * @brief      CLIPPER parameters
   */
  struct Params {

    // \brief Basic gradient descent stopping criteria
    double tol_u = 1e-8; ///< stop when change in u < tol
    double tol_F = 1e-9; ///< stop when change in F < tol
    double tol_Fop = 1e-10; ///< stop when ||dFop|| < tol
    int maxiniters = 200; ///< max num of gradient ascent steps for each d
    int maxoliters = 1000; ///< max num of outer loop iterations to find d

    // \brief Line search parameters
    double beta = 0.1; ///< backtracking step size reduction, in (0, 1)
    int maxlsiters = 99; ///< maximum number of line search iters per grad step

    double eps = 1e-9; ///< numerical threshold around 0
  };

  /**
   * @brief      Data associated with a CLIPPER dense cluster solution
   */
  struct Solution
  {
    double t; ///< duration spent solving [s]
    int ifinal; ///< number of outer iterations before convergence
    std::vector<int> nodes; ///< indices of graph vertices in dense cluster
    Eigen::VectorXd u; ///< characteristic vector associated with graph
    double score; ///< value of objective function / largest eigenvalue
  };

  /**
   * @brief      SFINAE helper to allow only Eigen types - supports both
   *             dense and sparse Eigen matrices.
   */
  template <typename T>
  using IsEigenBase = std::enable_if_t<std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>;

  /**
   * @brief      Identifies a dense cluster of an undirected graph G from its
   *             weighted affinity matrix M while satisfying any active
   *             constraints in C (indicated with zeros).
   *
   *             If M is binary and C==M then CLIPPER returns a maximal clique.
   *
   *             This algorithm employs a projected gradient descent method to
   *             solve a symmetric rank-one nonnegative matrix approximation.
   *
   * @param[in]  M        Symmetric, non-negative nxn affinity matrix where
   *                      each element is in [0,1]. Nodes can also be weighted
   *                      between [0,1] (e.g., if there is a prior indication
   *                      that a node belongs to the desired cluster). In the
   *                      case that all nodes are equally likely to be in the
   *                      densest cluster/node weights should not be considered
   *                      set the diagonal of M to identity.
   * @param[in]  C        nxn binary constraint matrix. Active const. are 0.
   * @param[in]  params   Parameters of the algorithm run
   *
   * @tparam     T        Can handle dense or sparse Eigen matrices
   *
   * @return     Solutions structure containing dense cluster
   */
  template <typename T, IsEigenBase<T> = true>
  Solution findDenseCluster(const T& M,
    const T& C, const Params& params = Params());

  /**
   * @brief      Identifies a dense cluster of an undirected graph G.
   *
   * @param[in]  M       Weighted affinity matrix
   * @param[in]  C       Binary constraint matrix
   * @param[in]  u0      Initial value of the optimization variable.
   * @param[in]  params  Parameters of the algorithm run
   *
   * @tparam     T        Can handle dense or sparse Eigen matrices
   *
   * @return     Solutions structure containing dense cluster
   */
  template <typename T, IsEigenBase<T> = true>
  Solution findDenseCluster(const T& M,
    const T& C, const Eigen::VectorXd& u0,
    const Params& params = Params());


  /**
  * @brief      Identifies a dense cluster of an undirected graph G.
  *             This is an optimized version of `findDenseCluster`
  *             that exploits the sparsity pattern of G
  *             and should be used with M and C matrices coming from
  *             the `scoreSparsePairwiseConsistency`.
  *
  *
  * @param[in]  M       Sparse Weighted affinity matrix (zeros along the diagonal)
  * @param[in]  C       Sparse Binary constraint matrix (zeros along the diagonal)
  * @param[in]  u0      Initial value of the optimization variable.
  * @param[in]  params  Parameters of the algorithm run
  *
  *
  * @return     Solutions structure containing dense cluster
  */
  Solution findDenseClusterOfSparseGraph(const SpMat &M, const SpMat &C,
                                        const Eigen::VectorXd &u0,
                                        const Params &params);

  /**
  * @brief      Encapsulates `findDenseClusterOfSparseGraph`
  *             and passes its arguments along with the
  *             the intial value of the optimization variable u0
  *
  *
  * @param[in]  M       Sparse Weighted affinity matrix (zeros along the diagonal)
  * @param[in]  C       Sparse Binary constraint matrix (zeros along the diagonal)
  * @param[in]  params  Parameters of the algorithm run
  *
  *
  * @return     Solutions structure containing dense cluster
  */
  Solution findDenseClusterOfSparseGraph(const SpMat &M, const SpMat &C,
                                        const Params &params);

} // ns clipper
