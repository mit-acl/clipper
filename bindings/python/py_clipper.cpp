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
}