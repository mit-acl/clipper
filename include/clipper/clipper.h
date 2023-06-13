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

#include "clipper/dsd.h"
#include "clipper/sdp.h"
#include "clipper/maxclique.h"

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

    double affinityeps = 1e-4; ///< sparsity-promoting threshold for affinities

    bool rescale_u0 = true; ///< Rescale u0 using one power iteration. This
                            ///< removes some randomness of the initial guess;
                            ///< i.e., after one step of power method, random
                            ///< u0's look similar.

    // \brief Rounding procedure
    enum Rounding { NONZERO, DSD, DSD_HEU };
    // NONZERO - any nonzero elements of u are selected as nodes
    // DSD - select the densest edge-weighted subgraph of the
    //       subgraph induced by NONZERO rounding
    // DSD_HEU - A heuristic for selecting the top best nodes
    //       of the subgraph induced by NONZERO rounding
    //       DSD_HEU tends to pick smaller subgraphs than DSD,
    //       sometimes leading to higher precision at the cost
    //       of lower recall
    Rounding rounding = Rounding::DSD_HEU;
  };

  /**
   * @brief      Data associated with a CLIPPER dense clique solution
   */
  struct Solution
  {
    double t; ///< duration spent solving [s]
    int ifinal; ///< number of outer iterations before convergence
    std::vector<int> nodes; ///< indices of graph vertices in dense clique
    Eigen::VectorXd u0; ///< initial vector used for local solver
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

    /**
     * @brief      Solves the MSRC problem using
     * graduated projected gradient ascent
     *
     * @param[in]  u0    Initial condition, if none provided random vec is used
     */
    void solve(const Eigen::VectorXd& u0 = Eigen::VectorXd());

    /**
     * @brief      Solves the maximum clique problem
     *
     * @param[in]  params Clique solver parameters
     */
    void solveAsMaximumClique(const maxclique::Params& params = {});

    /**
     * @brief      Solves the maximum spectral radius clique problem using
     *             a semidefinite relaxation.
     *
     * @param[in]  params  The parameters
     */
    void solveAsMSRCSDR(const sdp::Params& params = {});

    const Solution& getSolution() const { return soln_; }
    Affinity getAffinityMatrix();
    Constraint getConstraintMatrix();

    /**
     * @brief      Skip using scorePairwiseConsistency and directly set the
     *             affinity and constraint matrices. Note that this function
     *             accepts dense matrices. Use the sparse version for better
     *             performance if you already have sparse matrices available.
     *
     * @param[in]  M     Affinity matrix
     * @param[in]  C     Constraint matrix
     */
    void setMatrixData(const Affinity& M, const Constraint& C);

    /**
     * @brief      Skip using scorePairwiseConsistency and directly set the
     *             affinity and constraint matrices. Note that this function
     *             accepts sparse matrices. These matrices should be upper
     *             triangular and should not have diagonal values set.
     *
     * @param[in]  M     Affinity matrix
     * @param[in]  C     Constraint matrix
     */
    void setSparseMatrixData(const SpAffinity& M, const SpConstraint& C);

    Association getInitialAssociations();
    Association getSelectedAssociations();

    void setParallelize(bool parallelize) { parallelize_ = parallelize; };

  private:
    Params params_;
    invariants::PairwiseInvariantPtr invariant_;

    bool parallelize_ = true; ///< should affinity calculation be parallelized

    Association A_; ///< initial (putative) set of associations

    // \brief Problem data from latest instance of data association
    Solution soln_; ///< solution information from CLIPPER dense clique solver
    SpAffinity M_; ///< affinity matrix (i.e., weighted consistency graph)
    SpConstraint C_; ///< constraint matrix (i.e., prevents forming links)

    /**
     * @brief      Identifies a dense clique of an undirected graph G from its
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
     */
    void findDenseClique(const Eigen::VectorXd& u0);
  };

} // ns clipper