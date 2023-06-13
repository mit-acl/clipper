/**
 * @file sdp.h
 * @brief Problem parser for CLIPPER SDR optimization
 * @author Parker Lusk <plusk@mit.edu>
 * @date 23 Nov 2021
 */

#include <vector>

#include <Eigen/Dense>

namespace clipper {
namespace sdp {

  struct Solution
  {
    Eigen::MatrixXd X;
    Eigen::VectorXd lambdas;
    Eigen::VectorXd evec1;

    double thr; ///< threshold for selecting nodes
    std::vector<int> nodes; ///< indices of selected nodes

    int iters; ///< number of iterations
    float pobj; ///< primal objective value
    float dobj; ///< dual objective value

    double t; ///< total tile: parsing, solving, extraction
    double t_parse; ///< time spent setting up the problem data
    double t_scs; ///< total SCS time
    double t_scs_setup; ///< SCS setup time
    double t_scs_solve; ///< SCS solve time
    double t_scs_linsys; ///< SCS total time spent in the linear system solver
    double t_scs_cone; ///< total  time spent on cone projections
    double t_scs_accel; ///< total time spent in the accel routine
    double t_extract; ///< time spent extracting which nodes to select
  };

  struct Params
  {
    bool verbose = false;

    int max_iters = 2000;
    int acceleration_interval = 10;
    int acceleration_lookback = 10;

    float eps_abs = 1e-3;
    float eps_rel = 1e-3;
    float eps_infeas = 1e-7;

    float time_limit_secs = 0;
  };

  Solution solve(const Eigen::MatrixXd& M, const Eigen::MatrixXd& C,
                  const Params& params = Params{});

} // ns sdp
} // ns clipper