 /**
 * @file utils.h
 * @brief Usefult utilities
 * @author Parker Lusk <plusk@mit.edu>
 * @date 12 October 2020
 */

#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "clipper/invariants/abstract.h"
#include "clipper/types.h"

namespace clipper {
  class Solution;
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
   * @param[in]  k     How many of the largest elements to find (k > 0)
   *
   * @return     Indices of the largest elements in vector x
   */
  std::vector<int> findIndicesOfkLargest(const Eigen::VectorXd& x, int k);

  /**
   * @brief      Find indices of vector that are greater than a threshold.
   *
   * @param[in]  x     Vector to find elements of
   * @param[in]  thr   The threshold
   *
   * @return     Indices such that x[i] > thr for all i \in indices
   */
  std::vector<int> findIndicesWhereAboveThreshold(const Eigen::VectorXd& x,
                                                  double thr);

  /**
   * @brief      Creates an all-to-all association hypothesis
   *
   * @param[in]  n1    Number of items in view 1
   * @param[in]  n2    Number of items in view 2
   *
   * @return     an (n1*n2)x2 association matrix
   */
  inline Association createAllToAll(size_t n1, size_t n2)
  {
    Association A = Association(n1*n2, 2);
    for (size_t i=0; i<n1; ++i) {
      for (size_t j=0; j<n2; ++j) {
        A(j + i*n2, 0) = i;
        A(j + i*n2, 1) = j;
      }
    }
    return A;
  }

  /**
   * @brief      Select the elements of a vector x given an indicator vector.
   *
   * @param[in]  x     Vector to select elements of
   * @param[in]  ind   The indicator vector
   *
   * @return     Vector of selected elements, with size <= x.size
   */
  Eigen::VectorXd selectFromIndicator(const Eigen::VectorXd& x,
                                      const Eigen::VectorXi& ind);

  /**
   * @brief      Convenience function to select inlier associations
   *
   * @param[in]  soln  The solution of the dense cluster
   * @param[in]  A     The initial set of associations
   *
   * @return     The subset of associations deemed as inliers via solution
   */
  Association selectInlierAssociations(const Solution& soln, const Association& A);

  /**
 * @brief      Maps a flat index to coordinate of a square symmetric matrix
 *
 * @param[in]  k     The flat index to find the corresponding r,c of
 * @param[in]  n     Dimension of the square, symmetric matrix
 *
 * @return     row, col of a matrix corresponding to flat index k
 */
  std::tuple<size_t,size_t> k2ij(size_t k, size_t n);

  /**
   * @brief      Simple named profiling timer
   */
  class Timer
  {
  public:
    Timer() = default;
    Timer(const std::string& name) : name_(name) {}
    ~Timer() = default;

    void start()
    {
      t1_ = std::chrono::high_resolution_clock::now();
      running_ = true;
    }

    void stop()
    {
      t2_ = std::chrono::high_resolution_clock::now();

      if (running_) {
        const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_ - t1_);
        const double elapsed = static_cast<double>(duration.count()) / 1e9;
        total_ += elapsed;
        running_ = false;
        count_++;
      }
    }

    void reset()
    {
      total_ = 0;
    }

    double getElapsedSeconds() const
    {
      return total_;
    }

  private:
    double total_ = 0;
    std::string name_;
    int count_ = 0;
    bool running_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> t1_, t2_;

  public:
    friend std::ostream& operator<<(std::ostream& os, const Timer& t)
    {
      if (!t.name_.empty()) os << t.name_ << ": ";
      os << t.total_ << " s (" << t.count_ << "x)";
      return os;
    }

    friend Timer operator+(const Timer& lhs, const Timer& rhs) {
      Timer t;
      t.total_ = lhs.total_ + rhs.total_;
      return t;
    }
  };

} // ns utils
} // ns clipper