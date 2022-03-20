/**
 * @file clipper.cpp
 * @brief CLIPPER data association framework
 * @author Parker Lusk <plusk@mit.edu>
 * @date 19 March 2022
 */

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

  // allocate memory for calculations
  M_ = Affinity::Zero(m,m);
  C_ = Constraint::Ones(m,m);

#pragma omp parallel for shared(A_, D1, D2, M_, C_) if(parallelize_)
  for (size_t k=0; k<m*(m-1)/2; ++k) {
    size_t i, j; std::tie(i, j) = utils::k2ij(k, m);

    if (A_(i,0) == A_(j,0) || A_(i,1) == A_(j,1)) {
      C_(i,j) = C_(j,i) = 0; // distinctness constraint
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
    if (scr > 0) M_(i,j) = M_(j,i) = scr;
    else C_(i,j) = C_(j,i) = 0; // inconsistency constraint
  }

  // make diagonals one
  M_ += Eigen::MatrixXd::Identity(m,m);
}

// ----------------------------------------------------------------------------

void CLIPPER::solve()
{
  findDenseClique(M_, C_, utils::randvec(M_.cols()));
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
  return M_;
}

// ----------------------------------------------------------------------------

Constraint CLIPPER::getConstraintMatrix()
{
  return C_;
}

// ----------------------------------------------------------------------------

void CLIPPER::setAffinityMatrix(const Affinity& M)
{
  M_ = M;
}

// ----------------------------------------------------------------------------

void CLIPPER::setConstraintMatrix(const Constraint& C)
{
  C_ = C;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void CLIPPER::findDenseClique(const Affinity& _M, const Constraint& C,
                          const Eigen::VectorXd& u0)
{
  const auto t1 = std::chrono::high_resolution_clock::now();
  //
  // Initialization
  //

  const size_t n = _M.cols();

  // Zero out any entry corresponding to an active constraint
  const Eigen::MatrixXd M = _M.cwiseProduct(C);

  // Binary complement of constraint matrix
  const Eigen::MatrixXd Cb = Eigen::MatrixXd::Ones(n,n) - C;

  // one step of power method to have a good scaling of u
  Eigen::VectorXd u = M * u0;
  u /= u.norm();

  // initial value of d
  double d = 0; // zero if there are no active constraints
  Eigen::MatrixXd Cbu = Cb * u;
  const auto idxD = ((Cbu.array()>params_.eps) && (u.array()>params_.eps));
  if (idxD.sum() > 0) {
    Eigen::MatrixXd Mu = M * u;
    const Eigen::VectorXd num = idxD.select(Mu, std::numeric_limits<double>::infinity());
    const Eigen::VectorXd den = idxD.select(Cbu, 1);
    d = (num.array() / den.array()).minCoeff();
  }

  Eigen::MatrixXd Md = Eigen::MatrixXd(M.rows(), M.cols());
  // homotopy(Md, M, Cb, d);
  Md = M - d*Cb;

  // initialize memory
  Eigen::VectorXd gradF = Eigen::VectorXd(n);
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
    F = u.transpose() * Md * u; // current objective value

    //
    // Orthogonal projected gradient ascent
    //

    for (j=0; j<params_.maxiniters; ++j) {
      gradF = Md * u;

      // if (params_.orthogonal) {
      //   // orthogonal projection of gradient onto tangent plane to S^n at u
      //   gradF = gradF - (gradF.transpose() * u) * u;

      //   if (gradF.norm() < params_.tol_Fop) break;
      // }

      // double alpha = params_.alpha;
      // if (alpha <= 0) {
      //   const auto idxA = ((gradF.array()<-params_.eps) && (u.array()>params_.eps));
      //   if (idxA.sum()) {
      //     const Eigen::VectorXd num = idxA.select(u, std::numeric_limits<double>::infinity());
      //     const Eigen::VectorXd den = idxA.select(gradF, 1);
      //     alpha = (num.array() / den.array()).abs().minCoeff();
      //   } else {
      //     alpha = std::pow(1.0/params_.beta, 3) / gradF.norm();
      //   }
      // }

      double alpha = 1;

      //
      // Backtracking line search on gradient ascent
      //

      double Fnew = 0, deltaF = 0;
      for (k=0; k<params_.maxlsiters; ++k) {
        unew = u + alpha * gradF;                     // gradient step
        unew = unew.cwiseMax(0);                      // project onto positive orthant
        unew.normalize();                             // project onto S^n
        Fnew = unew.transpose() * Md * unew;          // new objective value after step
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

    Cbu = Cb * u;
    const auto idxD = ((Cbu.array() > params_.eps) && (u.array() > params_.eps));
    if (idxD.sum() > 0) {
      Mu = M * u;
      num = idxD.select(Mu, std::numeric_limits<double>::infinity());
      den = idxD.select(Cbu, 1);
      const double deltad = (num.array() / den.array()).abs().minCoeff();

      d += deltad;
      // homotopy(Md, M, Cb, d);
      Md = M - d*Cb;

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