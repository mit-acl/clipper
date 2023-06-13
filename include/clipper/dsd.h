/**
 * @file dsd.h
 * @brief Dense subgraph discovery using Goldberg's algorithm
 * @brief See https://github.com/MengLiuPurdue/find_densest_subgraph
 * @author Parker Lusk <plusk@mit.edu>
 * @date 13 June 2023
 */

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "clipper/types.h"

namespace clipper {
namespace dsd {

/**
 * @brief      Find the densest subgraph in an unweighted edge-weighted graph
 *             using Goldberg's flow-based polynomial-time algorithm. This
 *             solves the problem exactly.
 *             
 *             The optimization formulation is as follows
 *             
 *                max u' * A * u / (u' * u)
 *                s.t. u \in {0, 1}^n
 *                
 *             In the graph networks community, the objective is frequently
 *             written as f(S) = w(S) / |S|, where S is the subgraph, w(S)
 *             is the sum of the edges in the subgraph S, and |S| is the
 *             number of vertices in the subgraph. For example, see second
 *             paragraph of https://arxiv.org/pdf/1809.04802.pdf.
 *
 * @param[in]  A     Weighted adjacency matrix of graph. Symmetric upper tri.
 * @param[in]  S     (Optional) restrict search to subgraph of A
 *
 * @return     Nodes of densest subgraph
 */
std::vector<int> solve(const SpAffinity& A, const std::vector<int>& S = {});

/**
 * @brief      Dense specialization of dsd::solve
 *
 * @param[in]  A     Dense weighted adjacency matrix of graph
 * @param[in]  S     (Optional) restrict search to subgraph of A
 *
 * @return     Nodes of densest subgraph
 */
std::vector<int> solve(const Eigen::MatrixXd& A, const std::vector<int>& S = {});

} // ns dsd
} // ns clipper