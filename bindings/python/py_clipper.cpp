/**
 * @file py_clipper.cpp
 * @brief Python bindings for CLIPPER
 * @author Parker Lusk <plusk@mit.edu>
 * @date 28 January 2021
 */

#include <cstdint>
#include <sstream>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clipper/clipper.h"
#include "clipper/utils.h"

#include "trampolines.h"

namespace py = pybind11;
using namespace pybind11::literals;

void pybind_invariants(py::module& m)
{
  m.doc() = "Invariants are quantities that do not change under the"
            "transformation between two sets of objects. They are used to"
            "build a consistency graph. Some built-in invariants are provided.";

  using namespace clipper::invariants;

  //
  // Base Invariants
  //

  py::class_<Invariant, PyInvariant<>, std::shared_ptr<Invariant>>(m, "Invariant")
    .def(py::init<>());
  py::class_<PairwiseInvariant, Invariant, PyPairwiseInvariant<>, std::shared_ptr<PairwiseInvariant>>(m, "PairwiseInvariant")
    .def(py::init<>())
    .def("__call__", &clipper::invariants::PairwiseInvariant::operator());
  py::class_<PairwiseAndSingleInvariant, PairwiseInvariant, Invariant, PyPairwiseAndSingleInvariant<>, std::shared_ptr<PairwiseAndSingleInvariant>>(m, "PairwiseAndSingleInvariant")
    .def(py::init<>())
    .def("pairwise_similarity", &clipper::invariants::PairwiseAndSingleInvariant::pairwise_similarity)
    .def("single_similarity", &clipper::invariants::PairwiseAndSingleInvariant::single_similarity)
    .def("pairwise_single_fusion", &clipper::invariants::PairwiseAndSingleInvariant::pairwise_single_fusion);

  //
  // Gravity Constrained Distance
  //

  py::class_<GravityConstrainedDistance::Params>(m, "GravityConstrainedDistanceParams")
    .def(py::init<>())
    .def("__repr__", [](const GravityConstrainedDistance::Params &params) {
      std::ostringstream repr;
      repr << "<GravityConstrainedDistanceParams : sigma=" << params.sigma;
      repr << " epsilon=" << params.epsilon;
      repr << " mindist=" << params.mindist << ">";
      return repr.str();
    })
    .def_readwrite("sigma", &clipper::invariants::GravityConstrainedDistance::Params::sigma)
    .def_readwrite("epsilon", &clipper::invariants::GravityConstrainedDistance::Params::epsilon)
    .def_readwrite("mindist", &clipper::invariants::GravityConstrainedDistance::Params::mindist);

  py::class_<GravityConstrainedDistance, PairwiseInvariant, PyPairwiseInvariant<GravityConstrainedDistance>, std::shared_ptr<GravityConstrainedDistance>>(m, "GravityConstrainedDistance")
    .def(py::init<const GravityConstrainedDistance::Params&>());
    
  //
  // Volume Euclidean Distance
  //

  py::class_<VolumeEuclideanDistance::Params>(m, "VolumeEuclideanDistanceParams")
    .def(py::init<>())
    .def("__repr__", [](const VolumeEuclideanDistance::Params &params) {
      std::ostringstream repr;
      repr << "<VolumeEuclideanDistanceParams : sigma=" << params.sigma;
      repr << " epsilon=" << params.epsilon;
      repr << " mindist=" << params.mindist;
      repr << " epsilon_volume=" << params.epsilon_volume << ">";
      return repr.str();
    })
    .def_readwrite("sigma", &clipper::invariants::VolumeEuclideanDistance::Params::sigma)
    .def_readwrite("epsilon", &clipper::invariants::VolumeEuclideanDistance::Params::epsilon)
    .def_readwrite("mindist", &clipper::invariants::VolumeEuclideanDistance::Params::mindist)
    .def_readwrite("epsilon_volume", &clipper::invariants::VolumeEuclideanDistance::Params::epsilon_volume);

  py::class_<VolumeEuclideanDistance, PairwiseInvariant, PyPairwiseInvariant<VolumeEuclideanDistance>, std::shared_ptr<VolumeEuclideanDistance>>(m, "VolumeEuclideanDistance")
    .def(py::init<const VolumeEuclideanDistance::Params&>());

  //
  // Volume Gravity Constrained Distance
  //

  py::class_<VolumeGravityConstrainedDistance::Params>(m, "VolumeGravityConstrainedDistanceParams")
    .def(py::init<>())
    .def("__repr__", [](const VolumeGravityConstrainedDistance::Params &params) {
      std::ostringstream repr;
      repr << "<VolumeGravityConstrainedDistanceParams : sigma=" << params.sigma;
      repr << " epsilon=" << params.epsilon;
      repr << " mindist=" << params.mindist;
      repr << " epsilon_volume=" << params.epsilon_volume << ">";
      return repr.str();
    })
    .def_readwrite("sigma", &clipper::invariants::VolumeGravityConstrainedDistance::Params::sigma)
    .def_readwrite("epsilon", &clipper::invariants::VolumeGravityConstrainedDistance::Params::epsilon)
    .def_readwrite("mindist", &clipper::invariants::VolumeGravityConstrainedDistance::Params::mindist)
    .def_readwrite("epsilon_volume", &clipper::invariants::VolumeGravityConstrainedDistance::Params::epsilon_volume);

  py::class_<VolumeGravityConstrainedDistance, PairwiseInvariant, PyPairwiseInvariant<VolumeGravityConstrainedDistance>, std::shared_ptr<VolumeGravityConstrainedDistance>>(m, "VolumeGravityConstrainedDistance")
    .def(py::init<const VolumeGravityConstrainedDistance::Params&>());

  //
  // Distance Feature Similarity
  //
  py::class_<DistanceFeatureSimilarity, PairwiseInvariant, PyPairwiseInvariant<DistanceFeatureSimilarity>, std::shared_ptr<DistanceFeatureSimilarity>> distfeatsim(m, "DistanceFeatureSimilarity");
  distfeatsim.def(py::init<const DistanceFeatureSimilarity::Params&>());

  py::enum_<DistanceFeatureSimilarity::SimilarityFusionMethod>(distfeatsim, "SimilarityFusionMethod")
    .value("GEOMETRIC_MEAN", DistanceFeatureSimilarity::SimilarityFusionMethod::GEOMETRIC_MEAN)
    .value("ARITHMETIC_MEAN", DistanceFeatureSimilarity::SimilarityFusionMethod::ARITHMETIC_MEAN)
    .value("PRODUCT", DistanceFeatureSimilarity::SimilarityFusionMethod::PRODUCT)
    .export_values();

  py::class_<DistanceFeatureSimilarity::Params>(m, "DistanceFeatureSimilarityParams")
    .def(py::init<>())
    .def("__repr__", [](const DistanceFeatureSimilarity::Params &params) {
      std::ostringstream repr;
      repr << "<DistanceFeatureSimilarityParams : point_dim=" << params.point_dim;
      repr << " feature_dim=" << params.feature_dim;
      repr << " sigma=" << params.sigma;
      repr << " epsilon=" << params.epsilon;
      repr << " mindist=" << params.mindist;
      repr << " feature_epsilon=" << params.feature_epsilon;
      repr << " gravity_guided=" << params.gravity_guided;
      repr << " similarity_fusion_method=" << params.similarity_fusion_method;
      repr << " distance_fusion_weight=" << params.distance_fusion_weight << ">";
      return repr.str();
    })
    .def_readwrite("point_dim", &clipper::invariants::DistanceFeatureSimilarity::Params::point_dim)
    .def_readwrite("feature_dim", &clipper::invariants::DistanceFeatureSimilarity::Params::feature_dim)
    .def_readwrite("sigma", &clipper::invariants::DistanceFeatureSimilarity::Params::sigma)
    .def_readwrite("epsilon", &clipper::invariants::DistanceFeatureSimilarity::Params::epsilon)
    .def_readwrite("mindist", &clipper::invariants::DistanceFeatureSimilarity::Params::mindist)
    .def_readwrite("feature_epsilon", &clipper::invariants::DistanceFeatureSimilarity::Params::feature_epsilon)
    .def_readwrite("gravity_guided", &clipper::invariants::DistanceFeatureSimilarity::Params::gravity_guided)
    .def_readwrite("similarity_fusion_method", &clipper::invariants::DistanceFeatureSimilarity::Params::similarity_fusion_method)
    .def_readwrite("distance_fusion_weight", &clipper::invariants::DistanceFeatureSimilarity::Params::distance_fusion_weight);

  //
  // Distance Pairwise and Single
  //
  py::class_<DistancePairwiseAndSingle, PairwiseAndSingleInvariant, PyPairwiseAndSingleInvariant<DistancePairwiseAndSingle>, std::shared_ptr<DistancePairwiseAndSingle>> distpairwiseandsingle(m, "DistancePairwiseAndSingle");
  distpairwiseandsingle.def(py::init<const DistancePairwiseAndSingle::Params&>());

  py::enum_<DistancePairwiseAndSingle::SimilarityFusionMethod>(distpairwiseandsingle, "SimilarityFusionMethod")
    .value("GEOMETRIC_MEAN", DistancePairwiseAndSingle::SimilarityFusionMethod::GEOMETRIC_MEAN)
    .value("ARITHMETIC_MEAN", DistancePairwiseAndSingle::SimilarityFusionMethod::ARITHMETIC_MEAN)
    .value("PRODUCT", DistancePairwiseAndSingle::SimilarityFusionMethod::PRODUCT)
    .export_values();

  py::class_<DistancePairwiseAndSingle::Params>(m, "DistancePairwiseAndSingleParams")
    .def(py::init<>())
    .def("__repr__", [](const DistancePairwiseAndSingle::Params &params) {
      std::ostringstream repr;
      repr << "<DistancePairwiseAndSingleParams : point_dim=" << params.point_dim;
      repr << " feature_dim=" << params.feature_dim;
      repr << " sigma=" << params.sigma;
      repr << " epsilon=" << params.epsilon;
      repr << " mindist=" << params.mindist;
      repr << " feature_epsilon=" << params.feature_epsilon;
      repr << " gravity_guided=" << params.gravity_guided;
      repr << " similarity_fusion_method=" << params.similarity_fusion_method;
      repr << " distance_fusion_weight=" << params.distance_fusion_weight << ">";
      return repr.str();
    })
    .def_readwrite("point_dim", &clipper::invariants::DistancePairwiseAndSingle::Params::point_dim)
    .def_readwrite("feature_dim", &clipper::invariants::DistancePairwiseAndSingle::Params::feature_dim)
    .def_readwrite("sigma", &clipper::invariants::DistancePairwiseAndSingle::Params::sigma)
    .def_readwrite("epsilon", &clipper::invariants::DistancePairwiseAndSingle::Params::epsilon)
    .def_readwrite("mindist", &clipper::invariants::DistancePairwiseAndSingle::Params::mindist)
    .def_readwrite("feature_epsilon", &clipper::invariants::DistancePairwiseAndSingle::Params::feature_epsilon)
    .def_readwrite("gravity_guided", &clipper::invariants::DistancePairwiseAndSingle::Params::gravity_guided)
    .def_readwrite("similarity_fusion_method", &clipper::invariants::DistancePairwiseAndSingle::Params::similarity_fusion_method)
    .def_readwrite("distance_fusion_weight", &clipper::invariants::DistancePairwiseAndSingle::Params::distance_fusion_weight);

  //
  // Distance Min Max Similarity
  //
  py::class_<DistanceMinMaxSimilarity, DistancePairwiseAndSingle, PairwiseAndSingleInvariant, PyPairwiseAndSingleInvariant<DistanceMinMaxSimilarity>, std::shared_ptr<DistanceMinMaxSimilarity>> distminmaxsimilarity(m, "DistanceMinMaxSimilarity");
  distminmaxsimilarity.def(py::init<const DistancePairwiseAndSingle::Params&>());
  
  //
  // Distance Cosine Similarity
  //
  py::class_<DistanceCosSimilarity, DistancePairwiseAndSingle, PairwiseAndSingleInvariant, PyPairwiseAndSingleInvariant<DistanceCosSimilarity>, std::shared_ptr<DistanceCosSimilarity>> distcossimilarity(m, "DistanceCosSimilarity");
  distcossimilarity.def(py::init<const DistancePairwiseAndSingle::Params&>());

  //
  // Distance Cosine and Scale Similarity
  //
  py::class_<DistanceCosScaleSimilarity, DistancePairwiseAndSingle, PairwiseAndSingleInvariant, PyPairwiseAndSingleInvariant<DistanceCosScaleSimilarity>, std::shared_ptr<DistanceCosScaleSimilarity>> distcosscalesimilarity(m, "DistanceCosScaleSimilarity");
  distcosscalesimilarity.def(py::init<const DistancePairwiseAndSingle::Params&>());
  
  //
  // Distance Contrastive Similarity
  //
  py::class_<DistanceContrastiveSimilarity, DistancePairwiseAndSingle, PairwiseAndSingleInvariant, PyPairwiseAndSingleInvariant<DistanceContrastiveSimilarity>, std::shared_ptr<DistanceContrastiveSimilarity>> distcontrastivesimilarity(m, "DistanceContrastiveSimilarity");
  distcontrastivesimilarity.def(py::init<const DistancePairwiseAndSingle::Params&>());


  //
  // Euclidean Distance
  //

  py::class_<EuclideanDistance::Params>(m, "EuclideanDistanceParams")
    .def(py::init<>())
    .def("__repr__", [](const EuclideanDistance::Params &params) {
      std::ostringstream repr;
      repr << "<EuclideanDistanceParams : sigma=" << params.sigma;
      repr << " epsilon=" << params.epsilon;
      repr << " mindist=" << params.mindist << ">";
      return repr.str();
    })
    .def_readwrite("sigma", &clipper::invariants::EuclideanDistance::Params::sigma)
    .def_readwrite("epsilon", &clipper::invariants::EuclideanDistance::Params::epsilon)
    .def_readwrite("mindist", &clipper::invariants::EuclideanDistance::Params::mindist);

  py::class_<EuclideanDistance, PairwiseInvariant, PyPairwiseInvariant<EuclideanDistance>, std::shared_ptr<EuclideanDistance>>(m, "EuclideanDistance")
    .def(py::init<const EuclideanDistance::Params&>());

  //
  // Point-Normal Distance
  //

  py::class_<PointNormalDistance::Params>(m, "PointNormalDistanceParams")
    .def(py::init<>())
    .def("__repr__", [](const PointNormalDistance::Params &params) {
      std::ostringstream repr;
      repr << "<PointNormalDistanceParams : sigp=" << params.sigp;
      repr << " epsp=" << params.epsp << " sign=" << params.sign;
      repr << " epsn=" << params.epsn << ">";
      return repr.str();
    })
    .def_readwrite("sigp", &clipper::invariants::PointNormalDistance::Params::sigp)
    .def_readwrite("epsp", &clipper::invariants::PointNormalDistance::Params::epsp)
    .def_readwrite("sign", &clipper::invariants::PointNormalDistance::Params::sign)
    .def_readwrite("epsn", &clipper::invariants::PointNormalDistance::Params::epsn);

  py::class_<PointNormalDistance, PairwiseInvariant, PyPairwiseInvariant<PointNormalDistance>, std::shared_ptr<PointNormalDistance>>(m, "PointNormalDistance")
    .def(py::init<const PointNormalDistance::Params&>());
}

// ----------------------------------------------------------------------------

void pybind_utils(py::module& m)
{
  m.doc() = "Various convenience utilities for working with CLIPPER";

  m.def("create_all_to_all", clipper::utils::createAllToAll,
    "n1"_a, "n2"_a,
    "Create an all-to-all hypothesis for association. Useful for the case of"
    " no prior information or putative associations.");

  m.def("k2ij", clipper::utils::k2ij,
    "k"_a, "n"_a,
    "Maps a flat index k to coordinate of a square nxn symmetric matrix");
}

// ----------------------------------------------------------------------------

void pybind_dsd(py::module& m)
{
  m.doc() = "Exact dense edge-weighted subgraph discovery using Goldberg";

  // TODO(plusk): Support sparse matrices from python
  m.def("solve", py::overload_cast<const Eigen::MatrixXd&,
                        const std::vector<int>&>(clipper::dsd::solve),
    "A"_a.noconvert(), "S"_a=std::vector<int>{},
    "Find densest edge-weighted subgraph of weighted adj mat A.");
}

// ----------------------------------------------------------------------------

PYBIND11_MODULE(clipperpy, m)
{
  m.doc() = "A graph-theoretic framework for robust data association";
  m.attr("__version__") = CLIPPER_VERSION;

  py::module m_invariants = m.def_submodule("invariants");
  pybind_invariants(m_invariants);

  py::module m_utils = m.def_submodule("utils");
  pybind_utils(m_utils);

  py::module m_dsd = m.def_submodule("dsd");
  pybind_utils(m_dsd);

  py::class_<clipper::maxclique::Params>(m, "MCParams")
    .def(py::init<>())
    .def("__repr__", [](const clipper::maxclique::Params &params) {
      std::ostringstream repr;
      repr << "<CLIPPER Maximum Clique Parameters>";
      return repr.str();
    })
    .def_readwrite("method", &clipper::maxclique::Params::method)
    .def_readwrite("threads", &clipper::maxclique::Params::threads)
    .def_readwrite("time_limit", &clipper::maxclique::Params::time_limit)
    .def_readwrite("verbose", &clipper::maxclique::Params::verbose);

  py::class_<clipper::sdp::Params>(m, "SDPParams")
    .def(py::init<>())
    .def("__repr__", [](const clipper::sdp::Params &params) {
      std::ostringstream repr;
      repr << "<CLIPPER SDP Parameters>";
      return repr.str();
    })
    .def_readwrite("verbose", &clipper::sdp::Params::verbose)
    .def_readwrite("max_iters", &clipper::sdp::Params::max_iters)
    .def_readwrite("acceleration_interval", &clipper::sdp::Params::acceleration_interval)
    .def_readwrite("acceleration_lookback", &clipper::sdp::Params::acceleration_lookback)
    .def_readwrite("eps_abs", &clipper::sdp::Params::eps_abs)
    .def_readwrite("eps_rel", &clipper::sdp::Params::eps_rel)
    .def_readwrite("eps_infeas", &clipper::sdp::Params::eps_infeas)
    .def_readwrite("time_limit_secs", &clipper::sdp::Params::time_limit_secs);

  py::enum_<clipper::Params::Rounding>(m, "Rounding")
      .value("NONZERO", clipper::Params::Rounding::NONZERO)
      .value("DSD", clipper::Params::Rounding::DSD)
      .value("DSD_HEU", clipper::Params::Rounding::DSD_HEU)
      .export_values();

  py::class_<clipper::Params>(m, "Params")
    .def(py::init<>())
    .def("__repr__", [](const clipper::Params &params) {
      std::ostringstream repr;
      repr << "<CLIPPER Parameters>";
      return repr.str();
    })
    .def_readwrite("tol_u", &clipper::Params::tol_u)
    .def_readwrite("tol_F", &clipper::Params::tol_F)
    .def_readwrite("tol_Fop", &clipper::Params::tol_Fop)
    .def_readwrite("maxiniters", &clipper::Params::maxiniters)
    .def_readwrite("maxoliters", &clipper::Params::maxoliters)
    .def_readwrite("beta", &clipper::Params::beta)
    .def_readwrite("maxlsiters", &clipper::Params::maxlsiters)
    .def_readwrite("eps", &clipper::Params::eps)
    .def_readwrite("affinityeps", &clipper::Params::affinityeps)
    .def_readwrite("rescale_u0", &clipper::Params::rescale_u0)
    .def_readwrite("rounding", &clipper::Params::rounding);

  py::class_<clipper::Solution>(m, "Solution")
    .def(py::init<>())
    .def("__repr__", [](const clipper::Solution &soln) {
      std::ostringstream repr;
      repr << "<CLIPPER Solution>";
      return repr.str();
    })
    .def_readwrite("t", &clipper::Solution::t)
    .def_readwrite("ifinal", &clipper::Solution::ifinal)
    .def_readwrite("nodes", &clipper::Solution::nodes)
    .def_readwrite("u0", &clipper::Solution::u0)
    .def_readwrite("u", &clipper::Solution::u)
    .def_readwrite("score", &clipper::Solution::score);

  py::class_<clipper::CLIPPER>(m, "CLIPPER")
    .def(py::init(
      [](const clipper::invariants::PairwiseInvariantPtr& invariant,
          const clipper::Params& params)
      {
        clipper::CLIPPER *clipper = new clipper::CLIPPER(invariant, params);
        // Python extended c++ classes cannot use parallelization due to
        // GIL-related resoure deadlocking issues for derived classes.
        // See also https://github.com/pybind/pybind11/issues/813.
        // Python extended c++ classes will inherit from PyPairwiseInvariant.
        bool parallelize = (std::dynamic_pointer_cast<PyPairwiseInvariant<>>(invariant)) ? false : true;
        clipper->setParallelize(parallelize);
        return clipper;
      }))
    .def("__repr__", [](const clipper::CLIPPER &clipper) {
      std::ostringstream repr;
      repr << "<CLIPPER>";
      return repr.str();
    })
    .def("score_pairwise_consistency", &clipper::CLIPPER::scorePairwiseConsistency,
          // py::call_guard<py::gil_scoped_release>(),
          "D1"_a.noconvert(), "D2"_a.noconvert(), "A"_a.noconvert())
    .def("solve", &clipper::CLIPPER::solve,
          "u0"_a.noconvert()=Eigen::VectorXd())
    .def("solve_as_maximum_clique", &clipper::CLIPPER::solveAsMaximumClique,
          "params"_a=clipper::maxclique::Params{})
    .def("solve_as_msrc_sdr", &clipper::CLIPPER::solveAsMSRCSDR,
          "params"_a=clipper::sdp::Params{})
    .def("get_initial_associations", &clipper::CLIPPER::getInitialAssociations)
    .def("get_selected_associations", &clipper::CLIPPER::getSelectedAssociations)
    .def("get_solution", &clipper::CLIPPER::getSolution)
    .def("get_affinity_matrix", &clipper::CLIPPER::getAffinityMatrix)
    .def("get_constraint_matrix", &clipper::CLIPPER::getConstraintMatrix)
    .def("set_matrix_data", &clipper::CLIPPER::setMatrixData,
          "M"_a.noconvert(), "C"_a.noconvert())
    .def("set_parallelize", &clipper::CLIPPER::setParallelize);

  py::class_<clipper::CLIPPERPairwiseAndSingle>(m, "CLIPPERPairwiseAndSingle")
    .def(py::init(
      [](const clipper::invariants::PairwiseAndSingleInvariantPtr& invariant,
          const clipper::Params& params)
      {
        clipper::CLIPPERPairwiseAndSingle *clipper = new clipper::CLIPPERPairwiseAndSingle(invariant, params);
        // Python extended c++ classes cannot use parallelization due to
        // GIL-related resoure deadlocking issues for derived classes.
        // See also https://github.com/pybind/pybind11/issues/813.
        // Python extended c++ classes will inherit from PyPairwiseInvariant.
        // bool parallelize = (std::dynamic_pointer_cast<PyPairwiseAndSingleInvariant<>>(invariant)) ? false : true;
        // clipper->setParallelize(parallelize);
        bool parallelize = (std::dynamic_pointer_cast<PyPairwiseAndSingleInvariant<>>(invariant)) ? false : true;
        clipper->setParallelize(parallelize);
        return clipper;
      }))
    .def("__repr__", [](const clipper::CLIPPERPairwiseAndSingle &clipper) {
      std::ostringstream repr;
      repr << "<CLIPPERPairwiseAndSingle>";
      return repr.str();
    })
    .def("score_pairwise_and_single_consistency", &clipper::CLIPPERPairwiseAndSingle::scorePairwiseAndSingleConsistency,
          // py::call_guard<py::gil_scoped_release>(),
          "D1"_a.noconvert(), "D2"_a.noconvert(), "A"_a.noconvert())
    .def("solve", &clipper::CLIPPERPairwiseAndSingle::solve,
          "u0"_a.noconvert()=Eigen::VectorXd())
    .def("solve_as_maximum_clique", &clipper::CLIPPERPairwiseAndSingle::solveAsMaximumClique,
          "params"_a=clipper::maxclique::Params{})
    .def("solve_as_msrc_sdr", &clipper::CLIPPERPairwiseAndSingle::solveAsMSRCSDR,
          "params"_a=clipper::sdp::Params{})
    .def("get_initial_associations", &clipper::CLIPPERPairwiseAndSingle::getInitialAssociations)
    .def("get_selected_associations", &clipper::CLIPPERPairwiseAndSingle::getSelectedAssociations)
    .def("get_solution", &clipper::CLIPPERPairwiseAndSingle::getSolution)
    .def("get_affinity_matrix", &clipper::CLIPPERPairwiseAndSingle::getAffinityMatrix)
    .def("get_constraint_matrix", &clipper::CLIPPERPairwiseAndSingle::getConstraintMatrix)
    .def("set_matrix_data", &clipper::CLIPPERPairwiseAndSingle::setMatrixData,
          "M"_a.noconvert(), "C"_a.noconvert())
    .def("set_parallelize", &clipper::CLIPPERPairwiseAndSingle::setParallelize);
}