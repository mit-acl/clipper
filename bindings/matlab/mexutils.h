/**
 * @file mexutils.h
 * @brief MATLAB/MEX utils for CLIPPER bindings
 * @author Parker Lusk <plusk@mit.edu>
 * @date 5 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#pragma once

#include <cstdio>

#include <mex.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

using MatlabSparse = Eigen::SparseMatrix<double,Eigen::ColMajor,std::make_signed<mwIndex>::type>;
using SpMat = Eigen::SparseMatrix<double>;

template<typename T, int r, int c>
void mapDblBufToEigen(Eigen::Matrix<T, r, c>* pmat, double* buf)
{
  Eigen::Matrix<double, r, c> tmp = Eigen::Map<Eigen::Matrix<double, r, c>>(buf, pmat->rows(), pmat->cols());
  *pmat = tmp.template cast<T>();
}

// ----------------------------------------------------------------------------

template<int r, int c>
void mapDblBufToEigen(Eigen::Matrix<double, r, c>* pmat, double* buf)
{
  *pmat = Eigen::Map<Eigen::Matrix<double, r, c>>(buf, pmat->rows(), pmat->cols());
}

// ----------------------------------------------------------------------------

template<typename T, int r, int c>
void mexMatrixToEigen(const mxArray* pa, Eigen::Matrix<T, r, c>* pmat, bool checkSize = true)
{
  int rows = mxGetM(pa);
  int cols = mxGetN(pa);

  if (checkSize && ((r != Eigen::Dynamic && rows != r) || (c != Eigen::Dynamic && cols != c))) {
    char BUF[100];
    std::snprintf(BUF, 100, "Expected size (%d x %d) but got (%d x %d)", r, c, rows, cols);
    mexErrMsgIdAndTxt("mexutils:mexMatrixToEigen", BUF);
  }

  pmat->resize(rows, cols);

  double* buf = (double *)mxGetData(pa);
  mapDblBufToEigen(pmat, buf);
}

// ----------------------------------------------------------------------------

Eigen::Map<MatlabSparse> mexMatrixToEigenSparse(const mxArray* pa)
{
    mxAssert(mxGetClassID(pa) == mxDOUBLE_CLASS, "Type of the input matrix isn't double");
    const int m = mxGetM(pa);
    const int n = mxGetN(pa);
    const int nz = mxGetNzmax(pa);

    MatlabSparse::StorageIndex* ir = reinterpret_cast<MatlabSparse::StorageIndex*>(mxGetIr(pa));
    MatlabSparse::StorageIndex* jc = reinterpret_cast<MatlabSparse::StorageIndex*>(mxGetJc(pa));
    return Eigen::Map<MatlabSparse>(m, n, nz, jc, ir, mxGetPr(pa));
}
