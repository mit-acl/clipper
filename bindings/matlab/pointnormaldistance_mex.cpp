/**
 * @file pointnormaldistance_mex.cpp
 * @brief MATLAB/MEX for scoring the point-normal pair invariant
 * @author Parker Lusk <plusk@mit.edu>
 * @date 5 October 2020
 */

#include <iostream>

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <tuple>

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

  clipper::Association A;
  clipper::invariants::Data D1, D2;

  const mxArray * pStruct = nullptr;

  if (nrhs >= 3) {
    mexMatrixToEigen(prhs[0], &D1);
    mexMatrixToEigen(prhs[1], &D2);
    mexMatrixToEigen(prhs[2], &A, false);
  }

  if (nrhs >= 4) {
    pStruct = prhs[3];
  }

  if (nrhs > 4) {
    mexErrMsgIdAndTxt("pointnormal:nargin", "Only 3 or 4 input args supported (D1, D2, A, params)");
  }

  // shift indexing from MATLAB to C++
  if (A.size()) A -= clipper::Association::Ones(A.rows(), A.cols());

  // extract parameters from optionally provided parameters
  ParamsMap<clipper::invariants::PointNormalDistance::Params> map;
  map.add_field<double>("sigp", &clipper::invariants::PointNormalDistance::Params::sigp);
  map.add_field<double>("epsp", &clipper::invariants::PointNormalDistance::Params::epsp);
  map.add_field<double>("sign", &clipper::invariants::PointNormalDistance::Params::sign);
  map.add_field<double>("epsn", &clipper::invariants::PointNormalDistance::Params::epsn);
  clipper::invariants::PointNormalDistance::Params params = map.extract(pStruct);

  clipper::invariants::PointNormalDistance invariant(params);
  Eigen::MatrixXd M, C;
  std::tie(M, C) = clipper::scorePairwiseConsistency(invariant, D1, D2, A);

  if (nlhs >= 1) {
    plhs[0] = mxCreateDoubleMatrix(M.rows(), M.cols(), mxREAL);
    Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M.rows(), M.cols()) = M;
  }

  if (nlhs >= 2) {
    plhs[1] = mxCreateDoubleMatrix(C.rows(), C.cols(), mxREAL);
    Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), C.rows(), C.cols()) = C;
  }

  if (nlhs >= 3) {
    plhs[2] = mxCreateDoubleMatrix(A.rows(), A.cols(), mxREAL);
    Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[2]), A.rows(), A.cols()) =
        A.cast<double>() + Eigen::MatrixXd::Ones(A.rows(), A.cols()); // for matlab 1-based indexing
  }

  if (nlhs > 3) {
    mexErrMsgIdAndTxt("pointnormal:nargout", "Only 1, 2, or 3 output args supported (M, C, A)");
  }
}