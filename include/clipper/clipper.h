/**
 * @file clipper.h
 * @brief CLIPPER data association framework
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 */

#pragma once

#include <tuple>

#include <Eigen/Dense>

#include "clipper/invariants/abstract.h"
#include "clipper/invariants/builtins.h"
#include "clipper/types.h"

namespace clipper {

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
    double beta = 0.25; ///< backtracking step size reduction, in (0, 1)
    int maxlsiters = 99; ///< maximum number of line search iters per grad step

    double eps = 1e-9; ///< numerical threshold around 0
  };

  /**
   * @brief      Data associated with a CLIPPER dense clique solution
   */
  struct Solution
  {
    double t; ///< duration spent solving [s]
    int ifinal; ///< number of outer iterations before convergence
    std::vector<int> nodes; ///< indices of graph vertices in dense clique
    Eigen::VectorXd u; ///< characteristic vector associated with graph
    double score; ///< value of objective function / largest eigenvalue
  };

  /**
   * @brief      Convenience class to use CLIPPER for data association.
   */
  class CLIPPER
  {
  public:
    CLIPPER(const invariants::PairwiseInvariantPtr& invariant, const Params& params);
    ~CLIPPER() = default;

    /**
   * @brief      Creates an affinity matrix containing consistency scores for
   *             each of the m pairwise associations listed in matrix A.
   *
   * @param[in]  D1           Dataset 1 of n1 d-dim elements (dxn1)
   * @param[in]  D2           Dataset 2 of n2 d-dim elements (dxn2)
   * @param[in]  A            Associations to score (mx2)
   */
    void scorePairwiseConsistency(const invariants::Data& D1,
                                  const invariants::Data& D2,
                                  const Association& A = Association());

    void solve();

    const Solution& getSolution() const { return soln_; }
    Affinity getAffinityMatrix(); // const { return M_; }
    Constraint getConstraintMatrix(); // const { return C_; }

    void setAffinityMatrix(const Affinity& M);
    void setConstraintMatrix(const Constraint& C);

    Association getInitialAssociations(); // const { return A_; }
    Association getSelectedAssociations(); // const { return A_; }

    void setParallelize(bool parallelize) { parallelize_ = parallelize; };

  private:
    Params params_;
    invariants::PairwiseInvariantPtr invariant_;

    bool parallelize_ = true; ///< should affinity calculation be parallelized

    // SpAffinity M_;
    // SpAffinity C_;

    Association A_; ///< initial (putative) set of associations

    // \brief Problem data from latest instance of data association
    Solution soln_; ///< solution information from CLIPPER dense clique solver
    Affinity M_; ///< affinity matrix (i.e., weighted consistency graph)
    Constraint C_; ///< constraint matrix (i.e., prevents forming links)

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
    void findDenseClique(const Affinity& _M, const Constraint& C,
                          const Eigen::VectorXd& u0);

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
  };

} // ns clipper