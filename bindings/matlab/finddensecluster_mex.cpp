/**
 * @file finddensecluster_mex.cpp
 * @brief MATLAB/MEX binding for running CLIPPER's core findDenseCluster algo
 * @author Parker Lusk <plusk@mit.edu>
 * @date 5 October 2020
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>

#include <mex.h>

#include <Eigen/Dense>

#include <clipper/clipper.h>

#include "mexutils.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // nlhs   number of expected outputs
  // plhs   array to be populated by outputs (data passed back to matlab)
  // nrhs   number of inputs
  // prhs   array poplulated by inputs (data passed from matlab)

  bool useSparse = false;   ///< as determined by input (M, C)
  Eigen::MatrixXd M;        ///< consistency graph / affinity matrix
  SpMat Ms;                 ///< sparse consistency graph / affinity matrix
  Eigen::MatrixXd C;        ///< matrix of hard constraints
  SpMat Cs;                 ///< sparse matrix of hard constraints
  Eigen::VectorXd u0;       ///< initial guess for gradient ascent

  if (nrhs >= 2) {
    if (mxIsSparse(prhs[0]) && mxIsSparse(prhs[1])) {
      Ms = mexMatrixToEigenSparse(prhs[0]);
      Cs = mexMatrixToEigenSparse(prhs[1]);
      useSparse = true;
    } else if (!mxIsSparse(prhs[0]) && !mxIsSparse(prhs[1])) {
      mexMatrixToEigen(prhs[0], &M);
      mexMatrixToEigen(prhs[1], &C);
    } else {
      mexErrMsgIdAndTxt("findcorrespondences:nargin", "M and C must be both sparse or both full.");
    }
  }

  // capture number of associations, i.e., size of graph
  const int m = (useSparse) ? Ms.rows() : M.rows();

  if (nrhs >= 3) {
    mexMatrixToEigen(prhs[2], &u0);
    if (u0.rows() != m) {
      char BUF[100];
      std::snprintf(BUF, 100, "Expected u0 size (%d x 1) but got (%d x 1)", m, static_cast<int>(u0.rows()));
      mexErrMsgIdAndTxt("findcorrespondences:nargin", BUF);
    }
  }

  if (nrhs > 3) {
    mexErrMsgIdAndTxt("findcorrespondences:nargin", "Two or three arguments (M, C, u0) required.");
  }

  // if no initial condition supplied, generate random vector
  if (u0.size() == 0) u0 = clipper::utils::randvec(m);

  const auto t1 = std::chrono::high_resolution_clock::now();

  clipper::Params params;

  // Find the largest weighted cluster of consistent links
  clipper::Solution soln;
  if (useSparse) {
    soln = clipper::findDenseCluster(Ms, Cs, u0, params);
  } else {
    soln = clipper::findDenseCluster(M, C, u0, params);
  }

  const auto t2 = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  const double elapsed = static_cast<double>(duration.count()) / 1e6;

  if (nlhs >= 1) {
    std::vector<double> idx; idx.reserve(soln.nodes.size());
    plhs[0] = mxCreateDoubleMatrix(soln.nodes.size(), 1, mxREAL);
    std::transform(soln.nodes.begin(), soln.nodes.end(),
          std::back_inserter(idx), [](size_t x){ return x+1; }); // for matlab 1-based indexing
    memcpy(mxGetPr(plhs[0]), &idx[0], idx.size()*sizeof(double));
  }

  if (nlhs >= 2) {
    plhs[1] = mxCreateDoubleMatrix(soln.u.rows(), soln.u.cols(), mxREAL);
    Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), soln.u.rows(), soln.u.cols()) = soln.u;
  }

  if (nlhs >= 3) {
    plhs[2] = mxCreateDoubleScalar(elapsed);
  }

  if (nlhs > 3) {
    mexErrMsgIdAndTxt("findcorrespondences:nargout", "Only 1, 2, or 3 output args supported (idx, u, t)");
  }
}
