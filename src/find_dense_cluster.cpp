/**
 * @file clipper.cpp
 * @brief Core CLIPPER algorithm: find dense clusters w.r.t constraints
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
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


template <typename T, IsEigenBase<T>>
Solution findDenseCluster(const T& _M, const T& C,
                          const Eigen::VectorXd& u0, const Params& params)
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

  const auto t2 = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
  const double elapsed = static_cast<double>(duration.count()) / 1e9;

  Solution soln;
  soln.t = elapsed;
  soln.ifinal = i;
  std::swap(soln.nodes, I);
  soln.u.swap(u);
  soln.score = F;

  return soln;
}

template <typename T, IsEigenBase<T>>
Solution findDenseCluster(const T& M, const T& C,
                          const Params& params)
{
  return findDenseCluster(M, C, utils::randvec(M.cols()), params);
}

// template specializations for dense and sparse Eigen matrices
template Solution findDenseCluster<Eigen::MatrixXd>(const Eigen::MatrixXd&,
            const Eigen::MatrixXd&, const Eigen::VectorXd&, const Params&);
template Solution findDenseCluster<SpMat>(const SpMat&,
            const SpMat&, const Eigen::VectorXd&, const Params&);
template Solution findDenseCluster<Eigen::MatrixXd>(const Eigen::MatrixXd&,
            const Eigen::MatrixXd&, const Params&);
template Solution findDenseCluster<SpMat>(const SpMat&,
            const SpMat&, const Params&);




// methods for sparse matrices from scoreSparseConsistency


// sparse method similar to the above with the difference that M,C have zero
// elements along the diagonal
Solution findDenseClusterOfSparseGraph(const SpMat &M, const SpMat &C,
                                       const Eigen::VectorXd &u0,
                                       const Params &params) {
  // std::cout << "number of threads : " << Eigen::nbThreads() << std::endl;
  const auto t1 = std::chrono::high_resolution_clock::now();
  //
  // Initialization
  //

  const size_t n = M.cols();

  // un-necessary compute
  // Zero out any entry corresponding to an active constraint
  // const Eigen::MatrixXd M = _M.cwiseProduct(C);

  // this needs to be replaced
  // Binary complement of constraint matrix
  // const Eigen::MatrixXd Cb = Eigen::MatrixXd::Ones(n, n) - C;
  const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);

  // one step of power method to have a good scaling of u
  Eigen::VectorXd u = M * u0 + u0; // since M here is not diagonal
  // Eigen::VectorXd u = u0;
  u /= u.norm();

  // initial value of d
  double d = 0; // zero if there are no active constraints
  // Eigen::MatrixXd Cbu = Cb * u;
  Eigen::MatrixXd Cbu = ones * u.sum() - C * u - u;
  const auto idxD = ((Cbu.array() > params.eps) && (u.array() > params.eps));
  if (idxD.sum() > 0) {
    Eigen::MatrixXd Mu = M * u + u;
    const Eigen::VectorXd num =
        idxD.select(Mu, std::numeric_limits<double>::infinity());
    const Eigen::VectorXd den = idxD.select(Cbu, 1);
    d = (num.array() / den.array()).minCoeff();
  }

  // this should be replaced with an efficient representation
  // Md = M - d * Cb;
  // Eigen::MatrixXd Md = Eigen::MatrixXd(M.rows(), M.cols());
  // homotopy(Md, M, Cb, d);

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
  for (i = 0; i < params.maxoliters; ++i) {
    // F = u.transpose() * Md * u; // current objective value
    // gradF = (1 + d) * (M * u + u) - d * ones * u.sum();
    gradF = (1 + d) * u - d * ones * u.sum() + M * u + d * C * u;
    F = u.dot(gradF);
    // std::cout << std::endl;
    // std::cout << "obj F: " << F << " - dval: " << d << std::endl;
    //
    // Orthogonal projected gradient ascent
    //

    for (j = 0; j < params.maxiniters; ++j) {
      // gradF = Md * u;

      // if (params.orthogonal) {
      //   // orthogonal projection of gradient onto tangent plane to S^n at u
      //   gradF = gradF - (gradF.transpose() * u) * u;

      //   if (gradF.norm() < params.tol_Fop) break;
      // }

      // double alpha = params.alpha;
      // if (alpha <= 0) {
      //   const auto idxA = ((gradF.array()<-params.eps) &&
      //   (u.array()>params.eps)); if (idxA.sum()) {
      //     const Eigen::VectorXd num = idxA.select(u,
      //     std::numeric_limits<double>::infinity()); const Eigen::VectorXd den
      //     = idxA.select(gradF, 1); alpha = (num.array() /
      //     den.array()).abs().minCoeff();
      //   } else {
      //     alpha = std::pow(1.0/params.beta, 3) / gradF.norm();
      //   }
      // }

      double alpha = 1;

      //
      // Backtracking line search on gradient ascent
      //

      double Fnew = 0, deltaF = 0;
      for (k = 0; k < params.maxlsiters; ++k) {
        unew = u + alpha * gradF; // gradient step
        unew = unew.cwiseMax(0);  // project onto positive orthant
        unew.normalize();         // project onto S^n
        // Fnew = unew.transpose() * Md * unew; // new objective value after
        // step
        // gradFnew = (1 + d) * (M * unew + unew) - d * ones * unew.sum();
        gradFnew =
            (1 + d) * unew - d * ones * unew.sum() + M * unew + d * C * unew;
        Fnew = unew.dot(gradFnew);

        deltaF = Fnew - F; // change in objective value

        if (deltaF < -params.eps) {
          // objective value decreased---we need to backtrack, so reduce step
          // size
          alpha = alpha * params.beta;
        } else {
          // std::cout << "breaking at " << k << " out of ls " <<
          // params.maxlsiters
          //           << std::endl;
          break; // obj value increased, stop line search
        }
      }
      const double deltau = (unew - u).norm();

      // std::cout << "Fnew: " << Fnew << " Unew: " << unew.sum() << std::endl;
      // update values
      F = Fnew;
      u = unew;
      // gradF = gradFnew;

      // check if desired accuracy has been reached by gradient ascent
      if (deltau < params.tol_u || std::abs(deltaF) < params.tol_F) {
        // std::cout << "breaking at " << j << " out of in " <<
        // params.maxiniters
        //           << std::endl;
        break;
      }
    }

    //
    // Increase d
    //

    // Cbu = Cb * u;
    Cbu = ones * u.sum() - C * u - u;
    const auto idxD = ((Cbu.array() > params.eps) && (u.array() > params.eps));
    if (idxD.sum() > 0) {
      Mu = M * u + u;
      num = idxD.select(Mu, std::numeric_limits<double>::infinity());
      den = idxD.select(Cbu, 1);
      const double deltad = (num.array() / den.array()).abs().minCoeff();

      // std::cout << "delta d:" << deltad << std::endl;
      d += deltad;
      // homotopy(Md, M, Cb, d);

    } else {
      // std::cout << "breaking at " << i << " out of oo " << params.maxoliters
      //           << std::endl;
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
  const auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
  const double elapsed = static_cast<double>(duration.count()) / 1e9;

  Solution soln;
  soln.t = elapsed;
  soln.ifinal = i;
  std::swap(soln.nodes, I);
  soln.u.swap(u);
  soln.score = F;

  return soln;
}


Solution findDenseClusterOfSparseGraph(const SpMat &M, const SpMat &C,
                                       const Params &params) {
  return findDenseClusterOfSparseGraph(M, C,
                                       C * Eigen::VectorXd::Ones(C.cols()) +
                                           Eigen::VectorXd::Ones(C.cols()),
                                       params);
}

} // ns clipper
