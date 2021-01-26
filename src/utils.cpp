/**
 * @file utils.cpp
 * @brief Usefult utilities
 * @author Parker Lusk <plusk@mit.edu>
 * @date 12 October 2020
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#include <functional>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace clipper {
namespace utils {

Eigen::VectorXd randvec(size_t n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0, 1);

  return Eigen::VectorXd::NullaryExpr(n, 1, [&](){ return dis(gen); });
}

// ----------------------------------------------------------------------------

std::vector<int> findIndicesOfkLargest(const Eigen::VectorXd& x, int k)
{
  using T = std::pair<double, int>; // pair value to be compared and index
  // n.b., the top of this queue is smallest element
  std::priority_queue<T, std::vector<T>, std::greater<T>> q;
  for (size_t i=0; i<x.rows(); ++i) {
    if (q.size() < k) {
      q.push({x(i), i});
    } else if (q.top().first < x(i)) {
      q.pop();
      q.push({x(i), i});
    }
  }

  std::vector<int> indices(k);
  for (size_t i=0; i<k; ++i) {
    indices[k - i - 1] = q.top().second;
    q.pop();
  }

  return indices;
}

} // ns utils
} // ns clipper