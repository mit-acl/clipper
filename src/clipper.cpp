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

void CLIPPER::solve()
{
  findDenseClique(utils::randvec(M_.cols()));
}

// ----------------------------------------------------------------------------

Association CLIPPER::getInitialAssociations()
{
  return A_;
}

// ----------------------------------------------------------------------------

Association CLIPPER::getSelectedAssociations()
{
  return selectInlierAssociations(soln_, A_);
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

void CLIPPER::setAffinityMatrix(const Affinity& M)
{
  M_ = M.sparseView();
  M_.diagonal().setZero();
}

// ----------------------------------------------------------------------------

void CLIPPER::setConstraintMatrix(const Constraint& C)
{
  C_ = C.sparseView();
  C_.diagonal().setZero();
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void CLIPPER::findDenseClique(const Eigen::VectorXd& u0)
{
  const auto t1 = std::chrono::high_resolution_clock::now();

  //
  // Initialization
  //

  const size_t n = M_.cols();
  const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);

  // one step of power method to have a good scaling of u
  Eigen::VectorXd u = M_.selfadjointView<Eigen::Upper>() * u0 + u0;
  u /= u.norm();

  // initial value of d
  double d = 0; // zero if there are no active constraints
  Eigen::VectorXd Cbu = ones * u.sum() - C_.selfadjointView<Eigen::Upper>() * u - u;
  const auto idxD = ((Cbu.array()>params_.eps) && (u.array()>params_.eps));
  if (idxD.sum() > 0) {
    Eigen::VectorXd Mu = M_.selfadjointView<Eigen::Upper>() * u + u;
    const Eigen::VectorXd num = idxD.select(Mu, std::numeric_limits<double>::infinity());
    const Eigen::VectorXd den = idxD.select(Cbu, 1);
    d = (num.array() / den.array()).minCoeff();
  }

  // initialize memory
  Eigen::VectorXd gradF = Eigen::VectorXd(n);
  Eigen::VectorXd gradFnew = Eigen::VectorXd(n);
  Eigen::VectorXd unew = Eigen::VectorXd(n);
  Eigen::VectorXd Mu = Eigen::VectorXd(n);
  Eigen::VectorXd num = Eigen::VectorXd(n);
  Eigen::VectorXd den = Eigen::VectorXd(n);

  //
  // Orthogonal projected gradient ascent with homotopy
  //

  double F = 0; // objective value

  size_t i, j, k; // iteration counters
  for (i=0; i<params_.maxoliters; ++i) {
    gradF = (1 + d) * u - d * ones * u.sum() + M_.selfadjointView<Eigen::Upper>() * u + C_.selfadjointView<Eigen::Upper>() * u * d;
    F = u.dot(gradF); // current objective value

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
        unew = u + alpha * gradF;                     // gradient step
        unew = unew.cwiseMax(0);                      // project onto positive orthant
        unew.normalize();                             // project onto S^n
        gradFnew = (1 + d) * unew // because M/C is missing identity on diagonal
                    - d * ones * unew.sum()
                    + M_.selfadjointView<Eigen::Upper>() * unew
                    + C_.selfadjointView<Eigen::Upper>() * unew * d;
        Fnew = unew.dot(gradFnew);                    // new objective value after step

        deltaF = Fnew - F;                            // change in objective value

        if (deltaF < -params_.eps) {
          // objective value decreased---we need to backtrack, so reduce step size
          alpha = alpha * params_.beta;
        } else {
          break; // obj value increased, stop line search
        }
      }
      const double deltau = (unew - u).norm();

      // update values
      F = Fnew;
      u = unew;

      // check if desired accuracy has been reached by gradient ascent 
      if (deltau < params_.tol_u || std::abs(deltaF) < params_.tol_F) break;
    }

    //
    // Increase d
    //

    Cbu = ones * u.sum() - C_.selfadjointView<Eigen::Upper>() * u - u;
    const auto idxD = ((Cbu.array() > params_.eps) && (u.array() > params_.eps));
    if (idxD.sum() > 0) {
      Mu = M_.selfadjointView<Eigen::Upper>() * u + u;
      num = idxD.select(Mu, std::numeric_limits<double>::infinity());
      den = idxD.select(Cbu, 1);
      const double deltad = (num.array() / den.array()).abs().minCoeff();

      d += deltad;

    } else {
      break;
    }
  }

  //
  // Generate output
  //

  // estimate cluster size using largest eigenvalue
  const int omega = std::round(F);

  // extract indices of nodes in identified dense cluster
  std::vector<int> I = utils::findIndicesOfkLargest(u, omega);

  const auto t2 = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
  const double elapsed = static_cast<double>(duration.count()) / 1e9;

  // set solution
  soln_.t = elapsed;
  soln_.ifinal = i;
  std::swap(soln_.nodes, I);
  soln_.u.swap(u);
  soln_.score = F;
}

} // ns clipper