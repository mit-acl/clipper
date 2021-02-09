/**
 * @file pointcloud_test.cpp
 * @brief Point Cloud CLIPPER tests
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <clipper/clipper.h>
#include <clipper/find_dense_cluster.h>
#include <clipper/invariants/builtins.h>

TEST(PointCloud, KnownScaleInvariant) {

  //
  // Algorithm setup
  //

  // instantiate the invariant function that will be used to score associations
  clipper::invariants::KnownScalePointCloud::Params iparams;
  clipper::invariants::KnownScalePointCloud invariant(iparams);

  //
  // Data setup
  //

  // create a target/model point cloud of data
  clipper::invariants::KnownScalePointCloud::Data model;
  model.resize(3, 4);
  model.col(0) << 0, 0, 0;
  model.col(1) << 2, 0, 0;
  model.col(2) << 0, 3, 0;
  model.col(3) << 2, 2, 0;

  // transform of data w.r.t model
  Eigen::Affine3d T_MD;
  T_MD = Eigen::AngleAxisd(M_PI/8, Eigen::Vector3d::UnitZ());
  T_MD.translation() << 5, 3, 0;

  // create source/data point cloud
  clipper::invariants::KnownScalePointCloud::Data data = T_MD.inverse() * model;

  // remove one point from the tgt (model) cloud---simulates a partial view
  data.conservativeResize(3,3);

  //
  // Identify data association
  //

  // an empty association set will be assumed to be all-to-all (12 total)
  clipper::Association A;

  Eigen::MatrixXd M, C;
  std::tie(M, C) = invariant.createAffinityMatrix(model, data, A);

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

// ----------------------------------------------------------------------------

TEST(PointCloud, KnownScale) {

  //
  // Algorithm setup
  //

  // instantiate the invariant function that will be used to score associations
  clipper::invariants::KnownScalePointCloud::Params iparams;
  iparams.sigma = 0.01;
  iparams.epsilon = 0.06;
  clipper::invariants::KnownScalePointCloud invariant(iparams);

  //
  // Data setup
  //

  // create a target/model point cloud of data
  clipper::invariants::KnownScalePointCloud::Data model;
  model.resize(3, 4);
  model.col(0) << 0, 0, 0;
  model.col(1) << 2, 0, 0;
  model.col(2) << 0, 3, 0;
  model.col(3) << 2, 2, 0;

  // transform of data w.r.t model
  Eigen::Affine3d T_MD;
  T_MD = Eigen::AngleAxisd(M_PI/8, Eigen::Vector3d::UnitZ());
  T_MD.translation() << 5, 3, 0;

  // create source/data point cloud
  clipper::invariants::KnownScalePointCloud::Data data = T_MD.inverse() * model;

  // remove one point from the tgt (model) cloud---simulates a partial view
  data.conservativeResize(3,3);

  //
  // Identify data association
  //

  // an empty association set will be assumed to be all-to-all
  clipper::Association A;

  Eigen::MatrixXd M, C;
  std::tie(M, C) = invariant.createAffinityMatrix(model, data, A);

  clipper::Params params;
  clipper::Solution soln = clipper::findDenseCluster(M, C, params);

  clipper::Association Ainliers = clipper::selectInlierAssociations(soln, A);

  ASSERT_EQ(Ainliers.rows(), 3);
  for (size_t i=0; i<Ainliers.rows(); ++i) {
    EXPECT_EQ(Ainliers(i, 0), Ainliers(i, 1));
  }

}

// ----------------------------------------------------------------------------

TEST(PointCloud, KnownScaleConvenience) {

  //
  // Algorithm setup
  //

  clipper::Params params;
  clipper::invariants::KnownScalePointCloud::Params iparams;

  // instantiate the clipper object that will process incoming pairs of sensor
  // data to determine the outlier-free set of pairwise associations
  clipper::CLIPPER<clipper::invariants::KnownScalePointCloud> clipper(params, iparams);

  //
  // Data setup
  //

  // create a target/model point cloud of data
  clipper::invariants::KnownScalePointCloud::Data model;
  model.resize(3, 4);
  model.col(0) << 0, 0, 0;
  model.col(1) << 2, 0, 0;
  model.col(2) << 0, 3, 0;
  model.col(3) << 2, 2, 0;

  // transform of data w.r.t model
  Eigen::Affine3d T_MD;
  T_MD = Eigen::AngleAxisd(M_PI/8, Eigen::Vector3d::UnitZ());
  T_MD.translation() << 5, 3, 0;

  // create source/data point cloud
  clipper::invariants::KnownScalePointCloud::Data data = T_MD.inverse() * model;

  // remove one point from the tgt (model) cloud---simulates a partial view
  data.conservativeResize(3,3);

  //
  // Identify data association
  //

  // note that data types are specified by invariant class
  clipper::Association Ainliers = clipper.findCorrespondences(model, data);

  ASSERT_EQ(Ainliers.rows(), 3);
  for (size_t i=0; i<Ainliers.rows(); ++i) {
    EXPECT_EQ(Ainliers(i, 0), Ainliers(i, 1));
  }

}

// ----------------------------------------------------------------------------

TEST(PointCloud, LargePointCloud) {

  //
  // Algorithm setup
  //

  clipper::Params params;
  clipper::invariants::KnownScalePointCloud::Params iparams;
  iparams.sigma = 0.015;
  iparams.epsilon = 0.02;
  clipper::CLIPPER<clipper::invariants::KnownScalePointCloud> clipper(params, iparams);

  //
  // Data setup
  //

  // create a target/model point cloud of data
  static constexpr int N = 32;
  clipper::invariants::KnownScalePointCloud::Data model;
  model = 5*clipper::invariants::KnownScalePointCloud::Data::Random(3, N);

  // transform of data w.r.t model
  Eigen::Affine3d T_MD;
  T_MD = Eigen::AngleAxisd(M_PI/8, Eigen::Vector3d::UnitZ());
  T_MD.translation() << 5, 3, 0;

  // create source/data point cloud
  clipper::invariants::KnownScalePointCloud::Data data = T_MD.inverse() * model;


  //
  // Identify data association
  //

  // note that data types are specified by invariant class
  clipper::Association Ainliers = clipper.findCorrespondences(model, data);

  ASSERT_EQ(Ainliers.rows(), N);
  for (size_t i=0; i<Ainliers.rows(); ++i) {
    EXPECT_EQ(Ainliers(i, 0), Ainliers(i, 1));
  }

}
