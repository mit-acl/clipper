/**
 * @file utils.cpp
 * @brief Usefult utilities
 * @author Parker Lusk <plusk@mit.edu>
 * @date 12 October 2020
 */

#include <functional>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "clipper/clipper.h"
#include "clipper/utils.h"

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
  if (k < 1) return {}; // invalid input
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

// ----------------------------------------------------------------------------

std::vector<int> findIndicesWhereAboveThreshold(const Eigen::VectorXd& x,
                                                double thr)
{
  std::vector<int> indices;
  indices.reserve(x.size());
  for (size_t i=0; i<x.rows(); ++i) {
    if (x(i) > thr) indices.push_back(i);
  }
  return indices;
}

// ----------------------------------------------------------------------------

Eigen::VectorXd selectFromIndicator(const Eigen::VectorXd& x,
                                    const Eigen::VectorXi& ind)
{
  Eigen::VectorXd y(ind.sum());
  size_t idx = 0;
  for (size_t i=0; i<x.size(); ++i) {
    if (ind[i]) {
      y[idx++] = x[i];
    }
  }
  return y;
}

// ----------------------------------------------------------------------------

std::tuple<size_t,size_t> k2ij(size_t k, size_t n)
{
  k += 1;

  const size_t l = n*(n-1)/2 - k;
  const size_t o = std::floor( (std::sqrt(1 + 8*l) - 1) / 2. );
  const size_t p = l - o*(o+1)/2;
  const size_t i = n - (o + 1);
  const size_t j = n - p;
  return {i-1, j-1};
}

// ----------------------------------------------------------------------------

Association selectInlierAssociations(const Solution& soln, const Association& A)
  {
    Association Ainliers = Association::Zero(soln.nodes.size(), 2);
    for (size_t i=0; i<soln.nodes.size(); ++i) {
      Ainliers.row(i) = A.row(soln.nodes[i]);
    }
    return Ainliers;
  }

} // ns utils
} // ns clipper