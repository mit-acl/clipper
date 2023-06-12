/**
 * @file maxclique.cpp
 * @brief Library for solving maximum clique
 * @author Parker Lusk <plusk@mit.edu>
 * @date 20 December 2021
 */

#include <iostream>

#include "clipper/maxclique.h"

#ifdef CLIPPER_HAS_PMC
#include <pmc/pmc.h>
#endif

namespace clipper {
namespace maxclique {

#ifdef CLIPPER_HAS_PMC
pmc::pmc_graph pmc_graph_from_adjmat(const Eigen::MatrixXd& A)
{
  // Create a PMC graph from the adjacency matrix
  std::vector<int> edges;
  std::vector<long long> vertices;
  vertices.push_back(0);

  const size_t n = A.rows();
  size_t total_num_edges = 0;
  for (size_t i=0; i<n; i++) {
    for (size_t j=0; j<n; j++) {
      if (A(i,j)) {
        edges.push_back(j);
        total_num_edges++;
      }
    }
    vertices.push_back(total_num_edges);
  }

  // Use PMC to calculate
  pmc::pmc_graph G(vertices, edges);
  return G;
}
#endif

// ----------------------------------------------------------------------------

std::vector<int> solve(const Eigen::MatrixXd& A, const Params& params)
{
  // vector to represent max clique
  std::vector<int> C;

#ifdef CLIPPER_HAS_PMC
  pmc::pmc_graph G = pmc_graph_from_adjmat(A);

  if (params.verbose) {
    G.basic_stats(0);
  }

  // used to silence cout from pmc library
  std::streambuf * oldbuf;

  // Prepare PMC input
  pmc::input in;
  in.algorithm = 0;
  in.threads = params.threads;
  in.experiment = 0;
  in.lb = 0;
  in.ub = 0;
  in.param_ub = 0;
  in.adj_limit = 20000;
  in.time_limit = params.time_limit;
  in.remove_time = 4;
  in.graph_stats = false;
  in.verbose = params.verbose;
  in.help = false;
  in.MCE = false;
  in.decreasing_order = false;
  in.heu_strat = "kcore";
  in.vertex_search_order = "deg";

  // upper-bound of max clique
  G.compute_cores();
  const int max_core = G.get_max_core();
  if (in.ub == 0) in.ub = max_core + 1;


  // check for k-core heuristic threshold
  // check whether threshold equals 1 to short circuit the comparison
  if (params.method == Method::KCORE) {
    // remove all nodes with core number less than max core number
    // k_cores is a vector saving the core number of each vertex
    std::vector<int>* k_cores = G.get_kcores();
    for (int i=1; i<k_cores->size(); ++i) {
      // Note: k_core has size equals to num vertices + 1
      if ((*k_cores)[i] >= max_core) {
        C.push_back(i-1);
      }
    }
    return C;
  }

  // lower-bound of max clique - skip if given as input
  if (in.lb == 0 && in.heu_strat != "0") {
    if (!params.verbose) oldbuf = std::cout.rdbuf(nullptr);
    pmc::pmc_heu maxclique(G, in);
    in.lb = maxclique.search(G, C);
    if (!params.verbose) std::cout.rdbuf(oldbuf);
  }

  assert(in.lb != 0);
  if (in.lb == 0) {
    // This means that max clique has a size of one
    return C;
  }

  if (params.method == Method::HEU || in.lb == in.ub) {
    return C;
  }

  // Optional exact max clique finding
  if (params.method == Method::EXACT) {
    // The following methods are used:
    // 1. k-core pruning
    // 2. neigh-core pruning/ordering
    // 3. dynamic coloring bounds/sort
    // see the original PMC paper and implementation for details:
    // R. A. Rossi, D. F. Gleich, and A. H. Gebremedhin, “Parallel Maximum Clique Algorithms with
    // Applications to Network Analysis,” SIAM J. Sci. Comput., vol. 37, no. 5, pp. C589–C616, Jan.
    // 2015.
    if (!params.verbose) oldbuf = std::cout.rdbuf(nullptr);
    if (G.num_vertices() < in.adj_limit) {
      G.create_adj();
      pmc::pmcx_maxclique finder(G, in);
      finder.search_dense(G, C);
    } else {
      pmc::pmcx_maxclique finder(G, in);
      finder.search(G, C);
    }
    if (!params.verbose) std::cout.rdbuf(oldbuf);
  }
#else
  std::cout << "Warning: PMC was not built with CLIPPER, "
               "maximum clique will not be found" << std::endl;
#endif

  return C;
}

} // ns maxclique
} // ns clipper
