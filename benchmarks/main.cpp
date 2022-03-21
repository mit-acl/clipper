/**
 * @file main.cpp
 * @brief CLIPPER benchmarks
 * @author Parker Lusk <plusk@mit.edu>
 * @date 21 March 2022
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <fort.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#include <clipper/clipper.h>

#include "bm_utils.h"

struct BMParams
{
  int m; ///< number of associations
  double rho; ///< outlier ratio, [0,1]
  double noise_sigma = 0.01; ///< std dev for bounded normal noise
  double noise_beta = 5.54 * 0.01; ///< bound on bounded normal
  bool parallelize = true; ///< should use parallelization in affinity creation
};

struct Trial
{
  double t_affinity = 0; ///< time [s] for affinity creation
  double t_solver = 0; ///< time [s] for dense clique solver
  double p = 0; ///< precision
  double r = 0; ///< recall
};

using trials_t = std::vector<Trial>;

// ----------------------------------------------------------------------------

Eigen::MatrixXd open_ply()
{
  // trying to automate discovery of the bun1k.ply file
  // this path will be found if running ./benchmarks/benchmark or ./benchmark
  constexpr char const * PLY = "bun10k.ply";
  std::vector<std::string> potential_paths = {PLY, "../" + std::string(PLY)};

  Eigen::MatrixXd pts;
  for (const auto& path : potential_paths) {
    if (utils::read_ply(path, pts)) {
      // std::cout << "Found " << PLY << " at " << path << std::endl;
      break;
    }
  }

  if (pts.size() == 0) {
    std::cerr << "Could not find " << PLY << std::endl;
  }

  utils::scale_to_cube(pts, 1);

  return pts;
}

// ----------------------------------------------------------------------------

Eigen::MatrixXd make_noisy(const Eigen::MatrixXd& pcd0, double sigma, double beta)
{

  Eigen::MatrixXd eta =
        utils::generate_bounded_normal_noise(pcd0.rows(), sigma, beta);

  return pcd0 + eta;
}

// ----------------------------------------------------------------------------

clipper::Association get_ground_truth_associations(const Eigen::MatrixXd& pcd0,
    const Eigen::MatrixXd& pcd1, double radius)
{
  constexpr bool enforce_1to1 = true;
  constexpr int knn = 1;
  return utils::distance_based_correspondences(pcd0, pcd1, knn, radius, enforce_1to1);
}

// ----------------------------------------------------------------------------

clipper::invariants::PairwiseInvariantPtr
build_euclidean_distance_invariant(double sigma, double epsilon)
{
  // instantiate the invariant function that will be used to score associations
  clipper::invariants::EuclideanDistance::Params iparams;
  iparams.sigma = sigma;
  iparams.epsilon = epsilon;
  clipper::invariants::EuclideanDistancePtr invariant =
            std::make_shared<clipper::invariants::EuclideanDistance>(iparams);
  return invariant;
}

// ----------------------------------------------------------------------------

Trial average_trial(const trials_t& trials)
{
  Trial avg;
  for (const auto& trial : trials) {
    avg.t_affinity += trial.t_affinity;
    avg.t_solver += trial.t_solver;
    avg.p += trial.p;
    avg.r += trial.r;
  }

  avg.t_affinity /= trials.size();
  avg.t_solver /= trials.size();
  avg.p /= trials.size();
  avg.r /= trials.size();

  return avg;
}

// ----------------------------------------------------------------------------

Trial bm_euclidean_distance(
    const clipper::invariants::PairwiseInvariantPtr& invariant,
    const Eigen::MatrixXd& pcd0, const BMParams& P)
{
  Trial trial;
  const Eigen::MatrixXd pcd1_aligned = make_noisy(pcd0, P.noise_sigma, P.noise_beta);
  const clipper::Association Agt0 = get_ground_truth_associations(pcd0, pcd1_aligned, P.noise_beta);

  clipper::Association A, Agt;
  std::tie(A, Agt) = utils::generate_synthetic_correspondences(pcd0,
    pcd1_aligned, Agt0, P.m, P.rho);

  clipper::Params params;
  clipper::CLIPPER clipper(invariant, params);
  clipper.setParallelize(P.parallelize);


  const Eigen::Matrix3Xd model = pcd0.transpose();
  const Eigen::Matrix3Xd data = pcd1_aligned.transpose();

  // time affinity matrix creation
  auto t1 = std::chrono::high_resolution_clock::now();
  clipper.scorePairwiseConsistency(model, data, A);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
  trial.t_affinity = static_cast<double>(duration.count()) * 1e-9;

  // time dense clique solver
  t1 = std::chrono::high_resolution_clock::now();
  clipper.solve();
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
  trial.t_solver = static_cast<double>(duration.count()) * 1e-9;

  // get p / r of clipper solution
  const clipper::Association Ain = clipper.getSelectedAssociations();
  std::tie(trial.p, trial.r) = utils::get_precision_recall(Ain, Agt);

  return trial;
}

// ----------------------------------------------------------------------------

int main(int argc, char const *argv[])
{

  bool full = false;

  std::vector<double> outrats;
  std::vector<size_t> num_assocs;
  size_t M;

  if (full) {
    num_assocs = { 50, 100, 500, 1000, 4000, 10000, 16000 };
    outrats = {0, 0.5, 0.8, 0.9};
    M = 10;
  } else {
    num_assocs = { 50, 100, 500, 1000, /*4000, 10000, 16000*/ };
    outrats = {0, 0.5, 0.8, 0.9};

    M = 1;
  }

  std::cout << std::endl;
  std::cout << "Benchmarking over " << M << " trials" << std::endl;
  std::cout << std::endl;

  std::map<double, std::map<size_t, trials_t>> trials_at_rho_at_m;

  //
  // Benchmark using Euclidean Distance invariant
  //

  const Eigen::MatrixXd pcd0 = open_ply();
  const auto invariant = build_euclidean_distance_invariant(0.015, 0.05);

  // table configuration
  fort::utf8_table table;
  table << fort::header
        << "ρ [%]" << "# assoc"
        << "affinity [ms]" << "dense clique [ms]"
        << "precision [%]" << "recall [%]"
        << fort::endr;
  table.column(2).set_cell_text_align(fort::text_align::right);
  table.column(3).set_cell_text_align(fort::text_align::right);

  // progress bar configuration
  std::cout << std::endl;
  indicators::show_console_cursor(false);
  indicators::ProgressBar bar{
    indicators::option::BarWidth{50},
    indicators::option::Start{"["},
    indicators::option::Fill{"="},
    indicators::option::Lead{">"},
    indicators::option::Remainder{" "},
    indicators::option::End{"]"},
    indicators::option::PostfixText{"Running Benchmarks"},
    indicators::option::ForegroundColor{indicators::Color::yellow},
    indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
  };

  // configure progress increments
  const size_t ticks = outrats.size() * num_assocs.size() * M;
  size_t tick = 0;

  for (const double rho : outrats) {
    bar.set_option(indicators::option::PostfixText{"Benchmarking ρ = " + std::to_string(static_cast<int>(rho*100)) + "%"});
    for (const int m : num_assocs) {

      BMParams bm_params;
      bm_params.m = m;
      bm_params.rho = rho;
      bm_params.parallelize = true;

      // execute monte carlo trials
      for (size_t i=0; i<M; ++i) {
        const Trial trial = bm_euclidean_distance(invariant, pcd0, bm_params);
        trials_at_rho_at_m[rho][m].push_back(trial);

        bar.set_progress(static_cast<double>(++tick)*100 / ticks);
      }

      // average the trials
      const Trial trial = average_trial(trials_at_rho_at_m[rho][m]);

      table << std::fixed << std::setprecision(3)
            << static_cast<int>(rho*100) << m
            << trial.t_affinity*1e3 << trial.t_solver*1e3
            << static_cast<int>(trial.p*1e2)
            << static_cast<int>(trial.r*1e2) << fort::endr;

    }
    table << fort::separator;
  }

  indicators::show_console_cursor(true);
  std::cout << std::endl;

  std::cout << table.to_string() << std::endl;
  return 0;
}