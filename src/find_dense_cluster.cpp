/**
 * @file clipper.cpp
 * @brief Core CLIPPER algorithm: find dense clusters w.r.t constraints
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#include <chrono>
#include <iostream>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

#include "clipper/find_dense_cluster.h"
#include "clipper/utils.h"

namespace clipper {

using SpMat = Eigen::SparseMatrix<double>;


inline void homotopy(Eigen::MatrixXd& Md, const Eigen::MatrixXd& M, const Eigen::MatrixXd& Cb, double d)
{
  // if (M.cols() < 2600) {
    Md = M - d*Cb;
  //   return;
  // }

// #pragma omp parallel for default(none) shared(Md, M, d, Cb)
//   for (size_t c=0; c<M.cols(); ++c) {
//     Md.col(c) = M.col(c) - d*Cb.col(c);
//   }
}


template <typename T, typename std::enable_if_t<std::is_base_of<Eigen::EigenBase<T>, T>::value, int> = 0>
Solution findDenseCluster(const T& _M, const T& C,
                          const Eigen::VectorXd& u0, const Params& params)
{
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
  const auto idxD = ((Cbu.array()>params.eps) && (u.array()>params.eps));
  if (idxD.sum() > 0) {
    Eigen::MatrixXd Mu = M * u;
    const Eigen::VectorXd num = idxD.select(Mu, std::numeric_limits<double>::infinity());
    const Eigen::VectorXd den = idxD.select(Cbu, 1);
    d = (num.array() / den.array()).minCoeff();
  }

  Eigen::MatrixXd Md = Eigen::MatrixXd(M.rows(), M.cols());
  homotopy(Md, M, Cb, d);

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
  for (i=0; i<params.maxoliters; ++i) {
    F = u.transpose() * Md * u; // current objective value

    //
    // Orthogonal projected gradient ascent
    //

    for (j=0; j<params.maxiniters; ++j) {
      gradF = Md * u;

      // if (params.orthogonal) {
      //   // orthogonal projection of gradient onto tangent plane to S^n at u
      //   gradF = gradF - (gradF.transpose() * u) * u;

      //   if (gradF.norm() < params.tol_Fop) break;
      // }

      // double alpha = params.alpha;
      // if (alpha <= 0) {
      //   const auto idxA = ((gradF.array()<-params.eps) && (u.array()>params.eps));
      //   if (idxA.sum()) {
      //     const Eigen::VectorXd num = idxA.select(u, std::numeric_limits<double>::infinity());
      //     const Eigen::VectorXd den = idxA.select(gradF, 1);
      //     alpha = (num.array() / den.array()).abs().minCoeff();
      //   } else {
      //     alpha = std::pow(1.0/params.beta, 3) / gradF.norm();
      //   }
      // }

      double alpha = 1;

      //
      // Backtracking line search on gradient ascent
      //

      double Fnew = 0, deltaF = 0;
      for (k=0; k<params.maxlsiters; ++k) {
        unew = u + alpha * gradF;                     // gradient step
        unew = unew.cwiseMax(0);                      // project onto positive orthant
        unew.normalize();                             // project onto S^n
        Fnew = unew.transpose() * Md * unew;          // new objective value after step
        deltaF = Fnew - F;                            // change in objective value

        if (deltaF < -params.eps) {
          // objective value decreased---we need to backtrack, so reduce step size
          alpha = alpha * params.beta;
        } else {
          break; // obj value increased, stop line search
        }
      }
      const double deltau = (unew - u).norm();

      // update values
      F = Fnew;
      u = unew;

      // check if desired accuracy has been reached by gradient ascent 
      if (deltau < params.tol_u || std::abs(deltaF) < params.tol_F) break;
    }

    //
    // Increase d
    //

    Cbu = Cb * u;
    const auto idxD = ((Cbu.array() > params.eps) && (u.array() > params.eps));
    if (idxD.sum() > 0) {
      Mu = M * u;
      num = idxD.select(Mu, std::numeric_limits<double>::infinity());
      den = idxD.select(Cbu, 1);
      const double deltad = (num.array() / den.array()).abs().minCoeff();

      d += deltad;
      homotopy(Md, M, Cb, d);

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

  Solution soln;
  soln.ifinal = i;
  std::swap(soln.nodes, I);
  soln.u.swap(u);
  soln.score = F;

  return soln;
}

template Solution findDenseCluster<Eigen::MatrixXd>(const Eigen::MatrixXd&,
            const Eigen::MatrixXd&, const Eigen::VectorXd&, const Params&);
template Solution findDenseCluster<SpMat>(const SpMat&,
            const SpMat&, const Eigen::VectorXd&, const Params&);

} // ns clipper
