/**
 * @file knownscalepointcloud_mex.cpp
 * @brief MATLAB/MEX for scoring invariants for known scale point cloud reg
 * @author Parker Lusk <plusk@mit.edu>
 * @date 5 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#include <iostream>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <tuple>

#include <mex.h>

#include <Eigen/Dense>

#include <clipper/invariants/builtins.h>

#include "mexutils.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // nlhs   number of expected outputs
  // plhs   array to be populated by outputs (data passed back to matlab)
  // nrhs   number of inputs
  // prhs   array poplulated by inputs (data passed from matlab)

  clipper::Association A;
  clipper::invariants::KnownScalePointCloud::Data D1, D2;
  clipper::invariants::KnownScalePointCloud::Params params;

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
    mexErrMsgIdAndTxt("knownscalepointcloud:nargin", "Only 3 or 4 input args supported (D1, D2, A, params)");
  }

  // shift indexing from MATLAB to C++
  if (A.size()) A -= clipper::Association::Ones(A.rows(), A.cols());

  std::map<std::string, std::function<void(clipper::invariants::KnownScalePointCloud::Params&,const void*)>> map;
  map = {
    {"sigma", [](auto& p, const void* v){ p.sigma = *(double*)v; }},
    {"epsilon", [](auto& p, const void* v){ p.epsilon = *(double*)v; }},
  };

  if (pStruct) {
    const int nfields = mxGetNumberOfFields(pStruct);

    for (size_t i=0; i<nfields; ++i) {
      std::string field{mxGetFieldNameByNumber(pStruct, i)};

      std::transform(field.begin(), field.end(), field.begin(),
        [](unsigned char c){ return std::tolower(c); });

      const double * value = mxGetPr(mxGetFieldByNumber(pStruct, 0, i));

      auto it = map.find(field);
      if (it != map.end()) it->second(params, value);
    }
  }

  clipper::invariants::KnownScalePointCloud invariant(params);
  Eigen::MatrixXd M, C;
  std::tie(M, C) = invariant.createAffinityMatrix(D1, D2, A);

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
    mexErrMsgIdAndTxt("knownscalepointcloud:nargout", "Only 1, 2, or 3 output args supported (M, C, A)");
  }
}