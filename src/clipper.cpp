/**
 * @file clipper.cpp
 * @brief CLIPPER data association framework
 * @author Parker Lusk <plusk@mit.edu>
 * @date 19 March 2022
 */

#include <iostream>

#include "clipper/clipper.h"
#include "clipper/utils.h"

namespace clipper {

CLIPPER::CLIPPER(const invariants::PairwiseInvariantPtr& invariant, const Params& params)
: invariant_(invariant), params_(params)
{}

// ----------------------------------------------------------------------------

void CLIPPER::scorePairwiseConsistency(const invariants::Data& D1,
                              const invariants::Data& D2, const Association& A)
{
  if (A.size() == 0) A_ = utils::createAllToAll(D1.cols(), D2.cols());
  else A_ = A;

  const size_t m = A_.rows();

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(m, m);

#pragma omp parallel for shared(A_, D1, D2, M_, C_) if(parallelize_)
  for (size_t k=0; k<m*(m-1)/2; ++k) {
    size_t i, j; std::tie(i, j) = utils::k2ij(k, m);

    if (A_(i,0) == A_(j,0) || A_(i,1) == A_(j,1)) {
      // violates distinctness constraint
      continue;
    }

    //
    // Evaluate the consistency of geometric invariants associated with ei, ej
    //

    // points to extract invariant from in D1
    const auto& d1i = D1.col(A_(i,0));
    const auto& d1j = D1.col(A_(j,0));

    // points to extract invariant from in D2
    const auto& d2i = D2.col(A_(i,1));
    const auto& d2j = D2.col(A_(j,1));

    const double scr = (*invariant_)(d1i, d1j, d2i, d2j);
    if (scr > params_.affinityeps) { // does not violate inconsistency constraint
      M(i,j) = scr;
    }
  }

  // Identity on diagonal is taken care of implicitly in findDenseClique()
  // M += Eigen::MatrixXd::Identity(m, m);

  M_ = M.sparseView();

  C_ = M_;
  C_.coeffs() = 1;
}

// ----------------------------------------------------------------------------

void CLIPPER::solve(const Eigen::VectorXd& _v0)
{
  Eigen::VectorXd v0;
  if (_v0.size() == 0) {
    v0 = utils::randvec(M_.cols());
  } else {
    v0 = _v0;
  }
  findDenseClique(v0);
}

// ----------------------------------------------------------------------------

void CLIPPER::solveAsMaximumClique(const maxclique::Params& params)
{
  Eigen::MatrixXd C = getConstraintMatrix();
  C = C - Eigen::MatrixXd::Identity(C.rows(), C.cols());

  //
  // Solve the Maximum Clique problem
  //

  std::vector<int> nodes;
  Eigen::VectorXd u, v;
  double Fmsrc = 0, t_solve = 0, t_round = 0;

  utils::Timer tim;
  tim.start();
  nodes = maxclique::solve(C, params);
  tim.stop();
  t_solve = tim.getElapsedSeconds();

  u = utils::createIndicator(nodes, C.rows());
  v = utils::projectIndicatorOntoMSRC(M_, u);
  Fmsrc = utils::evalMSRCObj(M_, v);

  //
  // Solve the densest subgraph problem
  //

  if (params.usedsd) {
    utils::Timer tim2;
    tim2.start();
    nodes = selectNodesByRounding(v, Fmsrc);
    tim2.stop();
    t_round = tim2.getElapsedSeconds();

    u = utils::createIndicator(nodes, C.rows());
    v = utils::projectIndicatorOntoMSRC(M_, u);
    Fmsrc = utils::evalMSRCObj(M_, v);
  }


  soln_.t_solve = t_solve;
  soln_.t_round = t_round;
  soln_.t = soln_.t_solve + soln_.t_round;
  soln_.ifinal = 0;
  soln_.Fmsrc = Fmsrc;
  soln_.Fdewc = utils::evalDEWCObj(M_, u);
  std::swap(soln_.nodes, nodes);
  soln_.v0 = Eigen::VectorXd::Zero(C.rows());
  soln_.v.swap(v);
  soln_.u.swap(u);
}

// ----------------------------------------------------------------------------

sdp::Solution CLIPPER::solveAsMSRCSDR(const sdp::Params& params)
{
  Eigen::MatrixXd M = getAffinityMatrix();
  Eigen::MatrixXd C = getConstraintMatrix();

  //
  // Solve MSRC to global optimality
  //

  std::vector<int> nodes;
  Eigen::VectorXd u, v;
  double Fmsrc = 0, t_solve = 0, t_round = 0;

  sdp::Solution soln = sdp::solve(M, C, params);
  nodes = soln.nodes;
  t_solve = soln.t;

  u = utils::createIndicator(nodes, C.rows());
  v = utils::projectIndicatorOntoMSRC(M_, u);
  Fmsrc = utils::evalMSRCObj(M_, v);

  //
  // Solve the densest subgraph problem
  //

  if (params.usedsd) {
    utils::Timer tim2;
    tim2.start();
    nodes = selectNodesByRounding(v, Fmsrc);
    tim2.stop();
    t_round = tim2.getElapsedSeconds();

    u = utils::createIndicator(nodes, C.rows());
    v = utils::projectIndicatorOntoMSRC(M_, u);
    Fmsrc = utils::evalMSRCObj(M_, v);
  }


  soln_.t_solve = t_solve;
  soln_.t_round = t_round;
  soln_.t = soln_.t_solve + soln_.t_round;
  soln_.ifinal = 0;
  soln_.Fmsrc = Fmsrc;
  soln_.Fdewc = utils::evalDEWCObj(M_, u);
  std::swap(soln_.nodes, nodes);
  soln_.v0 = Eigen::VectorXd::Zero(C.rows());
  soln_.v.swap(v);
  soln_.u.swap(u);

  return soln;
}

// ----------------------------------------------------------------------------

Association CLIPPER::getInitialAssociations()
{
  return A_;
}

// ----------------------------------------------------------------------------

Association CLIPPER::getSelectedAssociations()
{
  return utils::selectInlierAssociations(soln_, A_);
}

// ----------------------------------------------------------------------------

Affinity CLIPPER::getAffinityMatrix()
{
  Affinity M = SpAffinity(M_.selfadjointView<Eigen::Upper>())
                + Affinity::Identity(M_.rows(), M_.cols());
  return M;
}

// ----------------------------------------------------------------------------

Constraint CLIPPER::getConstraintMatrix()
{
  Constraint C = SpConstraint(C_.selfadjointView<Eigen::Upper>())
                  + Constraint::Identity(C_.rows(), C_.cols());
  return C;
}

// ----------------------------------------------------------------------------

void CLIPPER::setMatrixData(const Affinity& M, const Constraint& C)
{
  Eigen::MatrixXd MM = M.triangularView<Eigen::Upper>();
  MM.diagonal().setZero();
  M_ = MM.sparseView();

  Eigen::MatrixXd CC = C.triangularView<Eigen::Upper>();
  CC.diagonal().setZero();
  C_ = CC.sparseView();
}

// ----------------------------------------------------------------------------

void CLIPPER::setSparseMatrixData(const SpAffinity& M, const SpConstraint& C)
{
  M_ = M;
  C_ = C;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void CLIPPER::findDenseClique(const Eigen::VectorXd& v0)
{
  const auto t1 = std::chrono::high_resolution_clock::now();

  //
  // Initialization
  //

  const size_t n = M_.cols();
  const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);

  // initialize memory
  Eigen::VectorXd gradF(n);
  Eigen::VectorXd gradFnew(n);
  Eigen::VectorXd v(n);
  Eigen::VectorXd vnew(n);
  Eigen::VectorXd Mv(n);
  Eigen::VectorXd num(n);
  Eigen::VectorXd den(n);

  // one step of power method to have a good scaling of v
  if (params_.rescale_v0) {
    v = M_.selfadjointView<Eigen::Upper>() * v0 + v0;
  } else {
    v = v0;
  }
  v /= v.norm();

  // initial value of d
  double d = 0; // zero if there are no active constraints
  Eigen::VectorXd Cbv = ones * v.sum() - C_.selfadjointView<Eigen::Upper>() * v - v;
  const Eigen::VectorXi idxD = ((Cbv.array() > params_.eps) && (v.array() > params_.eps)).cast<int>();
  if (idxD.sum() > 0) {
    Mv = M_.selfadjointView<Eigen::Upper>() * v + v;
    num = utils::selectFromIndicator(Mv, idxD);
    den = utils::selectFromIndicator(Cbv, idxD);
    d = (num.array() / den.array()).mean();
  }

  //
  // Orthogonal projected gradient ascent with homotopy
  //

  double F = 0; // objective value

  size_t i, j, k; // iteration counters
  for (i=0; i<params_.maxoliters; ++i) {
    gradF = (1 + d) * v - d * ones * v.sum() + M_.selfadjointView<Eigen::Upper>() * v + C_.selfadjointView<Eigen::Upper>() * v * d;
    F = v.dot(gradF); // current objective value

    //
    // Orthogonal projected gradient ascent
    //

    for (j=0; j<params_.maxiniters; ++j) {
      double alpha = 1;

      //
      // Backtracking line search on gradient ascent
      //

      double Fnew = 0, deltaF = 0;
      for (k=0; k<params_.maxlsiters; ++k) {
        vnew = v + alpha * gradF;                     // gradient step
        vnew = vnew.cwiseMax(0);                      // project onto positive orthant
        vnew.normalize();                             // project onto S^n
        gradFnew = (1 + d) * vnew // because M/C is missing identity on diagonal
                    - d * ones * vnew.sum()
                    + M_.selfadjointView<Eigen::Upper>() * vnew
                    + C_.selfadjointView<Eigen::Upper>() * vnew * d;
        Fnew = vnew.dot(gradFnew);                    // new objective value after step

        deltaF = Fnew - F;                            // change in objective value

        if (deltaF < -params_.eps) {
          // objective value decreased---we need to backtrack, so reduce step size
          alpha = alpha * params_.beta;
        } else {
          break; // obj value increased, stop line search
        }
      }
      const double deltau = (vnew - v).norm();

      // update values
      F = Fnew;
      v = vnew;
      gradF = gradFnew;

      // check if desired accuracy has been reached by gradient ascent 
      if (deltau < params_.tol_v || std::abs(deltaF) < params_.tol_F) break;
    }

    //
    // Increase d
    //

    Cbv = ones * v.sum() - C_.selfadjointView<Eigen::Upper>() * v - v;
    const Eigen::VectorXi idxD = ((Cbv.array() > params_.eps) && (v.array() > params_.eps)).cast<int>();
    if (idxD.sum() > 0) {
      Mv = M_.selfadjointView<Eigen::Upper>() * v + v;
      num = utils::selectFromIndicator(Mv, idxD);
      den = utils::selectFromIndicator(Cbv, idxD);
      const double deltad = (num.array() / den.array()).abs().mean();

      d += deltad;

    } else {
      break;
    }
  }

  const auto t2 = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
  const double t_solve = static_cast<double>(duration.count()) / 1e9;

  //
  // Round v to binary vector
  //

  // select nodes based on rounding the v vector to binary u
  const auto t3 = std::chrono::high_resolution_clock::now();
  std::vector<int> nodes = selectNodesByRounding(v, F);
  const auto t4 = std::chrono::high_resolution_clock::now();
  const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3);
  const double t_round = static_cast<double>(duration2.count()) / 1e9;

  // create indicator vector u from selected node indices
  Eigen::VectorXd u = utils::createIndicator(nodes, v.rows());

  //
  // Bookkeeping
  //

  soln_.t_solve = t_solve;
  soln_.t_round = t_round;
  soln_.t = soln_.t_solve + soln_.t_round;
  soln_.ifinal = i;
  soln_.Fmsrc = utils::evalMSRCObj(M_, v); // should be same as F
  soln_.Fdewc = utils::evalDEWCObj(M_, u);
  std::swap(soln_.nodes, nodes);
  soln_.v0 = v0;
  soln_.v.swap(v);
  soln_.u.swap(u);
}

// ----------------------------------------------------------------------------

std::vector<int> CLIPPER::selectNodesByRounding(const Eigen::VectorXd& v, double F)
{
  // node indices of rounded v vector
  std::vector<int> nodes;

  if (params_.rounding == Params::Rounding::NONZERO) {

    nodes = utils::findIndicesWhereAboveThreshold(v, 0.0);

  } else if (params_.rounding == Params::Rounding::DSD) {

    // subgraph induced by non-zero elements of v
    const std::vector<int> S = utils::findIndicesWhereAboveThreshold(v, 0.0);

    // TODO(plusk): make this faster by leveraging matrix sparsity
    nodes = dsd::solve(M_, S);

  } else if (params_.rounding == Params::Rounding::DSD_HEU) {

    // estimate cluster size using largest eigenvalue
    const int omega = std::round(F);

    // extract indices of nodes in identified dense cluster
    nodes = utils::findIndicesOfkLargest(v, omega);

  }

  return nodes;
}

} // ns clipper
