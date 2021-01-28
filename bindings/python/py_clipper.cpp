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

#include "clipper/clipper.h"
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
    .def("create_affinity_matrix", &KnownScalePointCloud::createAffinityMatrix);
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
    .def_readwrite("ifinal", &clipper::Solution::ifinal)
    .def_readwrite("nodes", &clipper::Solution::nodes)
    .def_readwrite("u", &clipper::Solution::u)
    .def_readwrite("score", &clipper::Solution::score);

  m.def("find_dense_cluster",
    py::overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
          const clipper::Params&>(clipper::findDenseCluster<Eigen::MatrixXd>),
    "M"_a, "C"_a, "params"_a);

  m.def("find_dense_cluster",
    py::overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&,
          const clipper::Params&>(clipper::findDenseCluster<Eigen::MatrixXd>),
    "M"_a, "C"_a, "u0"_a, "params"_a);
}