/**
 * @file sdp.cpp
 * @brief Problem parser for CLIPPER SDR optimization
 * @author Parker Lusk <plusk@mit.edu>
 * @date 23 Nov 2021
 */

#include "clipper/sdp.h"

#include <chrono>
#include <iostream>

#ifdef CLIPPER_HAS_SCS
#include <glbopts.h>
#include <scs.h>
#include <util.h>
#endif

namespace clipper {
namespace sdp {

namespace utils {

/**
 * @brief      Half-vectorization of a symmetric matrix. Returns the column
 *             stack of the lower triangular part of the matrix.
 *
 * @param[in]  S     A nxn symmetric matrix
 *
 * @return     s = vech(S), a n(n+1)/2 length vector
 */
Eigen::VectorXd tril_colstack_from_mat(const Eigen::MatrixXd& S)
{
  assert(S.rows() == S.cols());

  const size_t n = S.rows();
  const size_t k = n * (n + 1) / 2;

  Eigen::VectorXd s(k);

  size_t i = 0;
  for (size_t c=0; c<n; ++c) {
    for (size_t r=c; r<n; ++r) {
      s[i++] = S(r,c);
    }
  }

  return s;
}

// ----------------------------------------------------------------------------

/**
 * @brief      Given a half-vectorization based on column stacking of the lower
 *             triangular part of a matrix, produce a symmetric matrix.
 *
 * @param[in]  s     A kx1 vector produced from vech(S)
 *
 * @return     An nxn symmetric matrix S, where k = n(n+1)/2
 */
Eigen::MatrixXd sym_mat_from_tril_colstack(const Eigen::VectorXd& s)
{
  const size_t k = s.size();
  const size_t n = static_cast<size_t>(std::sqrt(0.25 + 2 * k) - 0.5);

  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n, n);

  size_t r = 0, c = 0, C = 0;
  for (size_t i=0; i<k; ++i) {

    // populate each row in the current column until we reach the bottom
    S(r,c) = S(c,r) = s[i];
    if (++r == n) {
      C++; // move to the next column
      r = c = C; // start on the diagonal

      if (C == n) break;
    }
  }

  return S;
}

} // ns utils

// ----------------------------------------------------------------------------

Solution solve(const Eigen::MatrixXd& M, const Eigen::MatrixXd& C,
                const Params& params)
{
#ifdef CLIPPER_HAS_SCS
  const auto t0 = std::chrono::high_resolution_clock::now();

  const scs_int N = M.rows();

  const Eigen::VectorXd vechC = utils::tril_colstack_from_mat(C);
  const scs_int nzC = (vechC.array() == 0).count();

  const scs_int n = N * (N + 1) / 2; // num decision variables (PSD Cone)
  const scs_int z = nzC;             // num zeros in lower triangle of C
  const scs_int l = n - z + 1;       // every nz var is >0; plus trace equality
  scs_int s[] = {N};                 // PSD Cone --> size of PSD, not num vars
  const scs_int m = z + l + n;       // num constraints

  //
  // Create data matrix A in CSC format
  //

  const scs_int nnzA = n + N + n; // n for z or l cone, N for trace, n for PSD Cone
  scs_float *Ax;  // CSC data
  scs_int *Ai;    // CSC (row) indices
  scs_int *Ap;    // CSC (col) ptr
  Ax = new scs_float[nnzA];
  Ai = new scs_int[nnzA];
  Ap = new scs_int[n+1];


  const Eigen::VectorXd vechI = utils::tril_colstack_from_mat(Eigen::MatrixXd::Identity(N, N));

  // row counters for each cone
  size_t zr = 0;
  size_t lr = 0;
  size_t sr = 0;

  // index of entry in Ax, Ai
  size_t idx = 0;
  // index of entry in Ap
  size_t p = 0;

  for (size_t col=0; col<n; ++col) {

    // each column will have at least one nnz and either the zero or pos cone
    // will be the first entry of this column
    Ap[p++] = idx;

    if (vechC[col] == 0) { // this decision var has a zero cone constraint (X_ij == 0 == ui uj)
      Ax[idx] = 1;
      Ai[idx++] = zr++;
    } else { // this decision var has a positive orthant constraint (X_ij >= 0)
      Ax[idx] = -1;
      Ai[idx++] = z + lr++; // offset row idx by num rows required by zero cone
    }

    // check if this decision var is a diagonal element
    if (vechI[col] == 1) {

      // handle trace constraint
      Ax[idx] = 1;
      Ai[idx++] = z + l - 1; // last row of pos cone, offset by num zero cone rows

      // add the right PSD cone constraint
      Ax[idx] = -1;
      Ai[idx++] = z + l + sr++;

    } else {
      // add the right PSD cone constraint
      Ax[idx] = -std::sqrt(2);
      Ai[idx++] = z + l + sr++;
    }
    
  }

  // last entry indicates where the first entry of column n+1 would go
  Ap[p++] = idx;

  //
  // Create data matrix b
  //

  scs_float *b = new scs_float[m]{};
  b[n] = 1; // trace == 1 constraint

  //
  // Create data matrix c
  //

  const Eigen::VectorXd vechM = utils::tril_colstack_from_mat(M);

  scs_float *c = new scs_float[m];
  for (size_t col=0; col<n; ++col) {
    if (vechI[col] == 1) {
      c[col] = -vechM[col];
    } else {
      c[col] = 2. * -vechM[col];
    }
  }

  //
  // SCS Data setup
  //

  ScsData d{};

  d.m = m;
  d.n = n;
  d.b = b;
  d.c = c;

  d.A = new ScsMatrix;
  d.A->m = m;
  d.A->n = n;
  d.A->x = Ax;
  d.A->i = Ai;
  d.A->p = Ap;

  //
  // SCS Cone setup
  // 

  ScsCone k{};
  k.z = z;
  k.l = l;
  k.s = s;
  k.ssize = 1;

  const auto t1 = std::chrono::high_resolution_clock::now();

  //
  // SCS Settings
  //

  ScsSettings stgs{};
  scs_set_default_settings(&stgs);
  stgs.verbose = (params.verbose) ? 1 : 0;
  stgs.eps_abs = static_cast<scs_float>(params.eps_abs);
  stgs.eps_rel = static_cast<scs_float>(params.eps_rel);
  stgs.eps_infeas = static_cast<scs_float>(params.eps_infeas);
  stgs.max_iters = static_cast<scs_int>(params.max_iters);
  stgs.time_limit_secs = static_cast<scs_float>(params.time_limit_secs);

  //
  // Run SCS
  //

  ScsSolution sol{};
  ScsInfo info{};
  scs_int exitflag = scs(&d, &k, &stgs, &sol, &info);

  const auto t2 = std::chrono::high_resolution_clock::now();

  //
  // Unpack solution
  //

  Solution soln;

  // symmetric matrix from tril-colstack
  Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(sol.x, d.n);
  soln.X = utils::sym_mat_from_tril_colstack(x);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(soln.X);
  soln.lambdas = eigensolver.eigenvalues();
  soln.evec1 = eigensolver.eigenvectors().col(soln.X.cols()-1);

  soln.thr = soln.evec1.array().abs().maxCoeff() / 2.;

  soln.nodes.reserve(N);
  for (size_t i=0; i<soln.evec1.size(); ++i) {
    if (std::fabs(soln.evec1[i]) > soln.thr) {
      soln.nodes.push_back(i);
    }
  }

  // solution info
  soln.iters = info.iter;
  soln.pobj = info.pobj;
  soln.dobj = info.dobj;

  const auto t3 = std::chrono::high_resolution_clock::now();
  soln.t = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count()) * 1e-9;
  soln.t_parse = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) * 1e-9;
  soln.t_scs = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) * 1e-9;
  soln.t_scs_setup = info.setup_time * 1e-3;
  soln.t_scs_solve = info.solve_time * 1e-3;
  soln.t_scs_linsys = info.lin_sys_time * 1e-3;
  soln.t_scs_cone = info.cone_time * 1e-3;
  soln.t_scs_accel = info.accel_time * 1e-3;
  soln.t_extract = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()) * 1e-9;

  //
  // Clean up our mess
  //

  delete Ax;
  delete Ai;
  delete Ap;

  delete b;
  delete c;

  delete d.A;

  delete sol.x;
  delete sol.y;
  delete sol.s;

  return soln;
#else
  std::cout << "Warning: SCS was not built with CLIPPER, "
               "semidefinite relaxation will not be solved" << std::endl;
  return {};
#endif
}

} // ns sdp
} // ns clipper
