/**
 * @file bm_utils.h
 * @brief Benchmark utilities
 * @author Parker Lusk <plusk@mit.edu>
 * @date 21 March 2022
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <tinyply.h>
#include <nanoflann.hpp>

#include <clipper/clipper.h>
#include <clipper/utils.h>

namespace utils {

/**
 * @brief      Reads x-y-z vertices from a PLY file (binary or ascii).
 *
 * @param[in]  plyfile  The path of the PLY file to read
 * @param      pts      The x-y-z points as an nx3 matrix
 * @param[in]  silent   Keep output (error) messages quiet
 *
 * @return     true if success
 */
bool read_ply(const std::string& plyfile, Eigen::MatrixXd& pts, bool silent = true);

/**
 * @brief      Writes x-y-z vertices to a PLY file.
 *
 * @param[in]  plyfile  The path of the PLY file to create
 * @param[in]  pts      The x-y-z points as an nx3 matrix
 * @param[in]  bin      Write as binary if true, ascii if false
 *
 * @return     true if success
 */
bool write_ply(const std::string& plyfile, const Eigen::MatrixXd& pts, bool bin = true);

/**
 * @brief      Rescale a point cloud so that it fits inside of a cube
 *
 * @param      pts   Point cloud points
 * @param[in]  s     Cube side dimension to fit point cloud in
 */
void scale_to_cube(Eigen::MatrixXd& pts, double s);

/**
 * @brief      Generates noise according to a truncated normal distribution.
 *
 * @param[in]  n      The number of 3-vectors to generate
 * @param[in]  sigma  Standard deviation of the normal distribution
 * @param[in]  beta   All noise 3-vectors will have norm < beta
 *
 * @return     An nx3 matrix of noise
 */
Eigen::MatrixXd generate_bounded_normal_noise(size_t n, double sigma, double beta);

/**
 * @brief      Identifies correspondences between two point clouds based on
 *             nearest neighbor radius search.
 *
 * @param[in]  pcd0          Model point cloud
 * @param[in]  pcd1          Data point cloud
 * @param[in]  knn           Num of nearest neighbors to evaluate distance of
 * @param[in]  radius        Corresponding points cannot be further than this
 * @param[in]  enforce_1to1  Only accept correspondences which mutually agree
 *
 * @return     The association set
 */
clipper::Association distance_based_correspondences(const Eigen::MatrixXd& pcd0,
                const Eigen::MatrixXd& pcd1, size_t knn, double radius, bool enforce_1to1);

/**
 * @brief      Generates synthetic correspondencs for experiment.
 *
 * @param[in]  pcd0   Model point clound
 * @param[in]  pcd1   Data point cloud
 * @param[in]  Agood  Set of good associations
 * @param[in]  m      Number of final associations to generate
 * @param[in]  rho    Outlier ratio, [0,1]
 *
 * @return     Pair of A (for use in expt) and Agt (for checking result)
 */
std::pair<clipper::Association, clipper::Association>
generate_synthetic_correspondences(const Eigen::MatrixXd& pcd0,
    const Eigen::MatrixXd& pcd1, const clipper::Association& Agood,
    size_t m, double rho);

/**
 * @brief      Calculates the precision and recall of a given association set A
 *
 * @param[in]  A     Selected associations
 * @param[in]  Agt   Ground truth associations
 *
 * @return     Pair of precision, recall -- in [0,1]
 */
std::pair<double,double>
get_precision_recall(const clipper::Association& A, const clipper::Association& Agt);

} // ns utils