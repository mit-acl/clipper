/**
 * @file py_clipper.cpp
 * @brief Python bindings for CLIPPER
 * @author Parker Lusk <plusk@mit.edu>
 * @date 28 January 2021
 * @copyright Copyright MIT, Ford Motor Company (c) 2020-2021
 */

#include <cstdint>
#include <sstream>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clipper/clipper.h"
#include "clipper/utils.h"
#include "clipper/find_dense_cluster.h"
#include "clipper/invariants/builtins.h"

namespace py = pybind11;
using namespace pybind11::literals;

void pybind_invariants(py::module& m)
{
  m.doc() = "Invariants are quantities that do not change under the"
            "transformation between two sets of objects. They are used to"
            "build a consistency graph. Some built-in invariants are provided.";

  using namespace clipper::invariants;

  m.def("create_all_to_all", clipper::utils::createAllToAll,
    "n1"_a, "n2"_a,
    "Create an all-to-all hypothesis for association. Useful for the case of"
    " no prior information or putative associations.");

  //
  // Known Scale Point Cloud
  //

  py::class_<KnownScalePointCloud::Params>(m, "KnownScalePointCloudParams")
    .def(py::init<>())
    .def("__repr__", [](const KnownScalePointCloud::Params &params) {
       std::ostringstream repr;
       repr << "<KnownScalePointCloudParams : sigma=" << params.sigma;
       repr << " epsilon=" << params.epsilon << ">";
       return repr.str();
    })
    .def_readwrite("sigma", &clipper::invariants::KnownScalePointCloud::Params::sigma)
    .def_readwrite("epsilon", &clipper::invariants::KnownScalePointCloud::Params::epsilon);

  py::class_<KnownScalePointCloud>(m, "KnownScalePointCloud")
    .def(py::init<const KnownScalePointCloud::Params&>())
    .def("create_affinity_matrix", &KnownScalePointCloud::createAffinityMatrix,
      "D1"_a.noconvert(), "D2"_a.noconvert(), "A"_a);

  //
  // Plane Cloud
  //

  // since these params are actually the same abstract Invariant::Params,
  // create an alias on the Python side.
  m.attr("PlaneCloudParams") = m.attr("KnownScalePointCloudParams");

  py::class_<PlaneCloud>(m, "PlaneCloud")
    .def(py::init<const PlaneCloud::Params&>())
    .def("create_affinity_matrix", &PlaneCloud::createAffinityMatrix,
      "D1"_a.noconvert(), "D2"_a.noconvert(), "A"_a);
}

PYBIND11_MODULE(clipper, m)
{
  m.doc() = "CLIPPER is a graph-theoretic framework for robust data association";
  m.attr("__version__") = "0.1";

  py::module m_invariants = m.def_submodule("invariants");
  pybind_invariants(m_invariants);

  py::class_<clipper::Params>(m, "Params")
    .def(py::init<>())
    .def("__repr__", [](const clipper::Params &params) {
       std::ostringstream repr;
       repr << "<Parameters for CLIPPER dense cluster solver>";
       return repr.str();
    })
    .def_readwrite("tol_u", &clipper::Params::tol_u)
    .def_readwrite("tol_F", &clipper::Params::tol_F)
    .def_readwrite("tol_Fop", &clipper::Params::tol_Fop)
    .def_readwrite("maxiniters", &clipper::Params::maxiniters)
    .def_readwrite("maxoliters", &clipper::Params::maxoliters)
    .def_readwrite("beta", &clipper::Params::beta)
    .def_readwrite("maxlsiters", &clipper::Params::maxlsiters)
    .def_readwrite("eps", &clipper::Params::eps);

  py::class_<clipper::Solution>(m, "Solution")
    .def(py::init<>())
    .def("__repr__", [](const clipper::Solution &soln) {
       std::ostringstream repr;
       repr << "<CLIPPER dense cluster solution>";
       return repr.str();
    })
    .def_readwrite("t", &clipper::Solution::t)
    .def_readwrite("ifinal", &clipper::Solution::ifinal)
    .def_readwrite("nodes", &clipper::Solution::nodes)
    .def_readwrite("u", &clipper::Solution::u)
    .def_readwrite("score", &clipper::Solution::score);

  m.def("find_dense_cluster",
    py::overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
          const clipper::Params&>(clipper::findDenseCluster<Eigen::MatrixXd>),
    "M"_a.noconvert(), "C"_a.noconvert(), "params"_a);

  m.def("find_dense_cluster",
    py::overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&,
          const clipper::Params&>(clipper::findDenseCluster<Eigen::MatrixXd>),
    "M"_a.noconvert(), "C"_a.noconvert(), "u0"_a, "params"_a);

  m.def("select_inlier_associations", &clipper::selectInlierAssociations,
    "soln"_a, "A"_a);
}