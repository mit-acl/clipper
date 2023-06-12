/**
 * @file maxclique.h
 * @brief Library for solving maximum clique
 * @author Parker Lusk <plusk@mit.edu>
 * @date 20 December 2021
 */

#include <vector>

#include <Eigen/Dense>

namespace clipper {
namespace maxclique {

enum class Method { EXACT, HEU, KCORE };

struct Params
{
  Method method = Method::EXACT; ///< EXACT is ROBIN*, KCORE is ROBIN
  size_t threads = 24; ///< num threads for OpenMP to use
  int time_limit = 3600; ///< [s], maximum time allotted for MCP solving
  bool verbose = false;
};

std::vector<int> solve(const Eigen::MatrixXd& A, const Params& params = {});

} // ns maxclique
} // ns clipper
