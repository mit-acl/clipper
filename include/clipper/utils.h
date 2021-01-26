 /**
 * @file utils.h
 * @brief Usefult utilities
 * @author Parker Lusk <plusk@mit.edu>
 * @date 12 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#pragma once

#include <vector>

#include <Eigen/Dense>

namespace clipper {
namespace utils {

  /**
   * @brief      Produce an nx1 vector where each element is drawn from U[0, 1).
   *
   * @param[in]  n     Dimension of produced vector
   *
   * @return     Uniform random vector
   */
  Eigen::VectorXd randvec(size_t n);

  /**
   * @brief      Find indices of k largest elements of vector (similar to MATLAB find)
   *
   * @param[in]  x     Vector to find large elements in
   * @param[in]  k     How many of the largest elements to find
   *
   * @return     Indices of the largest elements in vector x
   */
  std::vector<int> findIndicesOfkLargest(const Eigen::VectorXd& x, int k);

} // ns utils
} // ns clipper