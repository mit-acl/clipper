/**
 * @file affinity_test.cpp
 * @brief CLIPPER affinity scoring tests
 * @author Parker Lusk <plusk@mit.edu>
 * @date 19 March 2022
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <clipper/clipper.h>

TEST(Affinity, EuclideanDistance) {

  //
  // Algorithm setup
  //

  // instantiate the invariant function that will be used to score associations
  clipper::invariants::EuclideanDistance::Params iparams;
  clipper::invariants::EuclideanDistancePtr invariant =
            std::make_shared<clipper::invariants::EuclideanDistance>(iparams);

  clipper::Params params;
  clipper::CLIPPER clipper(invariant, params);

  //
  // Data setup
  //

  // create a target/model point cloud of data
  Eigen::Matrix3Xd model(3, 4);
  model.col(0) << 0, 0, 0;
  model.col(1) << 2, 0, 0;
  model.col(2) << 0, 3, 0;
  model.col(3) << 2, 2, 0;

  // transform of data w.r.t model
  Eigen::Affine3d T_MD;
  T_MD = Eigen::AngleAxisd(M_PI/8, Eigen::Vector3d::UnitZ());
  T_MD.translation() << 5, 3, 0;

  // create source/data point cloud
  Eigen::Matrix3Xd data = T_MD.inverse() * model;

  // remove one point from the tgt (model) cloud---simulates a partial view
  data.conservativeResize(3, 3);

  //
  // Identify data association
  //

  // unspecified association set assumed to be all-to-all (this case: 12 total)
  clipper.scorePairwiseConsistency(model, data);

  // access the association set used to score pairwise consistency
  clipper::Association A = clipper.getInitialAssociations();

  // A should be an all-to-all hypothesis
  const int n = model.cols() * data.cols();
  EXPECT_EQ(A.rows(), n);
  EXPECT_EQ(A.cols(), 2); // CLIPPER is a pair-wise data association algo

  // Ensure that an all-to-all hypothesis was correctly created
  for (size_t i=0; i<model.cols(); i++) {
    for (size_t j=0; j<data.cols(); j++) {
      const size_t k = i * data.cols() + j;
      EXPECT_EQ(A(k,0), i);
      EXPECT_EQ(A(k,1), j);
    }
  }

  // access the calculated affinity and constraint matrix
  clipper::Affinity M = clipper.getAffinityMatrix();
  clipper::Constraint C = clipper.getConstraintMatrix();

  // affinity matrix made up of all-to-all hypothesis between 4 and 3 items
  EXPECT_EQ(M.rows(), A.rows());
  EXPECT_EQ(M.cols(), A.rows());

  // diagonal of the affinity matrix should be all ones
  EXPECT_EQ(M.diagonal(), Eigen::VectorXd::Ones(M.rows()));

  // matrices should be symmetric
  EXPECT_EQ(M, M.transpose());
  EXPECT_EQ(C, C.transpose());

  // in this case with perfect data, affinity matrix is binary and so
  // affinity matrix == constraint matrix
  EXPECT_EQ(M, C);

  // expected affinity matrix, from MATLAB
  Eigen::MatrixXd Mtrue = Eigen::MatrixXd(M.rows(), M.cols());
  Mtrue << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
           0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  EXPECT_EQ(M, Mtrue);
}