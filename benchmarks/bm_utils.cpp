/**
 * @file bm_utils.cpp
 * @brief Benchmark utilities
 * @author Parker Lusk <plusk@mit.edu>
 * @date 21 March 2022
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <unordered_set>

#include "bm_utils.h"

namespace utils {

struct float3 { float x, y, z; };
struct double3 { double x, y, z; };

bool read_ply(const std::string& plyfile, Eigen::MatrixXd& pts, bool silent)
{
  std::unique_ptr<std::istream> file_stream;

  try {
    file_stream.reset(new std::ifstream(plyfile, std::ios::binary));

    if (!file_stream || file_stream->fail()) {
      if (!silent)
        std::cerr << "Failed to open ply file " << plyfile << std::endl;
      return false;
    }

    tinyply::PlyFile ply;
    ply.parse_header(*file_stream);

    std::shared_ptr<tinyply::PlyData> vertices;
    try {
      vertices = ply.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception& e) {
      if (!silent)
        std::cerr << "tinply exception: " << e.what() << std::endl;
      return false;
    }

    ply.read(*file_stream);

    if (vertices) {
      pts = Eigen::MatrixXd(vertices->count, 3);
      if (vertices->t == tinyply::Type::FLOAT32) {
        std::vector<float3> v(vertices->count);
        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        std::memcpy(v.data(), vertices->buffer.get(), numVerticesBytes);
        for (size_t i=0; i<vertices->count; ++i) {
          pts.row(i) = Eigen::Vector3d(static_cast<double>(v[i].x),
                                        static_cast<double>(v[i].y),
                                        static_cast<double>(v[i].z));
        }
      } else if (vertices->t == tinyply::Type::FLOAT64) {
        std::vector<double3> v(vertices->count);
        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        std::memcpy(v.data(), vertices->buffer.get(), numVerticesBytes);
        for (size_t i=0; i<vertices->count; ++i) {
          pts.row(i) = Eigen::Vector3d(v[i].x, v[i].y, v[i].z);
        }
      }
    }

  } catch (const std::exception& e) {
    if (!silent)
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------

bool write_ply(const std::string& plyfile, const Eigen::MatrixXd& pts, bool bin)
{
  std::filebuf fb;
  if (bin) fb.open(plyfile, std::ios::out | std::ios::binary);
  else fb.open(plyfile, std::ios::out);

  std::ostream os(&fb);
  if (os.fail()) {
    std::cerr << "Failed to open " << plyfile << " for writing." << std::endl;
    return false;
  }

  tinyply::PlyFile ply;
  std::vector<double3> vertices;
  for (size_t i=0; i<pts.rows(); ++i) {
    vertices.push_back({pts(i,0), pts(i,1), pts(i,2)});
  }

  ply.add_properties_to_element("vertex", {"x", "y", "z"},
      tinyply::Type::FLOAT64, vertices.size(),
      reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
  ply.write(os, bin);

  return true;
}

// ----------------------------------------------------------------------------

void scale_to_cube(Eigen::MatrixXd& pts, double s)
{
  const Eigen::Vector3d d = pts.colwise().maxCoeff() - pts.colwise().minCoeff();
  const double sf = d.array().maxCoeff();
  pts *= s / sf;
}

// ----------------------------------------------------------------------------

Eigen::Vector3d vec3randn(double sigma)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dis(0, sigma);

  return Eigen::Vector3d::NullaryExpr([&](){ return dis(gen); });
}

// ----------------------------------------------------------------------------

Eigen::MatrixXd generate_bounded_normal_noise(size_t n, double sigma, double beta)
{
  Eigen::MatrixXd eta = Eigen::MatrixXd::Zero(n, 3);
  Eigen::Vector3d v;
  for (size_t i=0; i<eta.rows(); ++i) {
    do {
      v = vec3randn(sigma);
    } while (v.norm() > beta);
    eta.row(i) = v;
  }

  return eta;
}

// ----------------------------------------------------------------------------

clipper::Association distance_based_correspondences(const Eigen::MatrixXd& pcd0,
                const Eigen::MatrixXd& pcd1, size_t knn, double radius, bool enforce_1to1)
{
  constexpr int DIM = 3;

  // construct a kd-tree index
  using kdtree_t = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd>;

  // build an Eigen matrix index
  kdtree_t index(DIM, std::cref(pcd1), 10 /*max leaves*/);
  index.index->buildIndex();

  // pre-allocate space for results
  std::vector<size_t> ret_indices(pcd0.rows() * knn);
  std::vector<double> ret_sqdists(pcd0.rows() * knn);

  // for each point in pcd0, find the closest pt of pts1
  nanoflann::SearchParams params;
  for (size_t i=0; i<pcd0.rows(); ++i) {

    // copy current row as query point
    std::vector<double> query_pt(DIM);
    Eigen::Map<Eigen::RowVector3d>(&query_pt[0], 1, DIM) = pcd0.row(i);

    // map the segment of the pre-allocated space to this particular result set
    nanoflann::KNNResultSet<double> resultSet(knn);
    resultSet.init(&ret_indices[i * knn], &ret_sqdists[i * knn]);

    // find the closest point in pcd1 to this query point of pcd0
    index.index->findNeighbors(resultSet, &query_pt[0], params);
  }

  // map results to Eigen matrices for manipulation convenience
  using idxmat_t = Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using sqdmat_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<idxmat_t> idx_0_to_1(ret_indices.data(), pcd0.rows(), knn);
  Eigen::Map<sqdmat_t> sqd_0_to_1(ret_sqdists.data(), pcd0.rows(), knn);

  //
  // Convert to association matrix
  //

  std::map<size_t, std::vector<size_t>> corres;
  std::map<size_t, std::vector<double>> sqdist;

  const double radius_sq = radius * radius;

  clipper::Association A;
  for (size_t i=0; i<idx_0_to_1.rows(); ++i) {
    const size_t c0 = i;
    for (size_t j=0; j<idx_0_to_1.cols(); ++j) {
      const size_t c1 = idx_0_to_1(i,j);
      const double sd = sqd_0_to_1(i,j);
      // only capture association if distance within requested radius
      if (sd <= radius_sq) {
        A.conservativeResize(A.rows() + 1, Eigen::NoChange);
        A.bottomRows(1) << c0, c1;

        if (enforce_1to1) {
          // note: a single point in pcd1 may have multiple pcd0 points
          corres[c1].push_back(c0);
          sqdist[c1].push_back(sd);
        }
      }
    }
  }

  if (enforce_1to1) {
    A = clipper::Association::Zero(corres.size(), 2);

    size_t i = 0;
    for (const auto& it : corres) {
      const size_t c1 = it.first;
      size_t k = 0;
      if (it.second.size() > 1) {
        // find the index of the closest point
        k = std::distance(sqdist[c1].begin(),
                std::min_element(sqdist[c1].begin(), sqdist[c1].end()));
      }

      A.row(i++) << it.second[k], c1;
    }
  }

  return A;
}

// ----------------------------------------------------------------------------

bool is_row_contained_in(const Eigen::RowVectorXi& row, const Eigen::MatrixXi& mat)
{
  std::vector<std::vector<size_t>> colindices(row.size());
  std::vector<size_t> intersection;

  for (size_t c=0; c<row.size(); ++c) {
    for (size_t i=0; i<mat.rows(); ++i) {
      if (row(c) == mat(i, c)) {
        colindices[c].push_back(i);
      }
    }

    // already sorted
    // std::sort(colindices[c].begin(), colindices[c].end());

    if (c == 0) {
      intersection = colindices[0];
    } else {
      std::vector<size_t> tmp;
      std::set_intersection(intersection.begin(), intersection.end(),
                            colindices[c].begin(), colindices[c].end(),
                            std::back_inserter(tmp));
      intersection = tmp;
    }
  }

  return intersection.size() > 0;
}

// ----------------------------------------------------------------------------

Eigen::Vector2i k2ij_full(size_t k, size_t ncols)
{
  // this function does not assume symmetry like clipper::utils::k2ij
  const size_t r = k / ncols;
  const size_t c = k % ncols;
  return Eigen::Vector2i(r, c);
}

// ----------------------------------------------------------------------------

std::pair<clipper::Association, clipper::Association>
generate_synthetic_correspondences(const Eigen::MatrixXd& pcd0,
    const Eigen::MatrixXd& pcd1, const clipper::Association& Agood,
    size_t m, double rho)
{
  assert((rho >= 0 && rho <= 1) && "outlier ratio must be in [0, 1]");

  // number of inliers amongst final putative associations, and in final gt
  const size_t ni = static_cast<size_t>(std::round(m * (1 - rho)));

  // number of outliers amongst final putative associations
  const size_t no = m - ni;

  // number of good associations to choose from
  const size_t p = Agood.rows();

  if (ni > p) {
    std::cerr << "Not enough initial inliers (" << p << ") for the requested "
              << "outlier ratio (" << rho << ", " << ni << ")." << std::endl;
    return {};
  }

  std::random_device rd;
  std::mt19937 rng(rd());


  clipper::Association A = clipper::Association(m, 2);
  clipper::Association Agt = clipper::Association(ni, 2);

  // select inliers to use (choose ni elements from 1:p without replacement)
  std::vector<size_t> I(p);
  std::iota(I.begin(), I.end(), 0);
  std::shuffle(I.begin(), I.end(), rng);

  // add the good inliers
  for (size_t i=0; i<ni; ++i) {
    Agt.row(i) = Agood.row(I[i]);
    A.row(no+i) = Agood.row(I[i]);
  }

  // Sample from all possible associations
  // const clipper::Association Aall =
  //      clipper::utils::createAllToAll(pcd0.rows(), pcd1.rows()); // too slow
  const size_t mall = pcd0.rows() * pcd1.rows();
  std::uniform_int_distribution<size_t> dis(0, mall);
  std::vector<bool> mask(mall, false);
  // std::unordered_set<size_t> tried;

  // select the outliers to use
  size_t nele = 0;
  while (nele < no) {

    // select an association to sample
    const size_t k = dis(rng);

    // don't sample this row again
    if (mask[k]) continue;
    mask[k] = true;

    // create the appropriate row of Aall on the fly
    Eigen::Vector2i row = k2ij_full(k, pcd1.rows());

    // make sure this association is not good
    if (is_row_contained_in(row, Agood)) {
      continue;
    }

    A.row(nele) = row;
    ++nele;
  }

  return std::make_pair(A, Agt);
}

// ----------------------------------------------------------------------------

std::pair<double,double>
get_precision_recall(const clipper::Association& A, const clipper::Association& Agt)
{

  // precision == TP / (TP + FP)
  // recall == TP / (TP + FN)

  if (Agt.size() == 0 || A.size() == 0) return {0, 0};

  size_t TP = 0;
  for (size_t i=0; i<A.rows(); ++i) {
    TP += (is_row_contained_in(A.row(i), Agt)) ? 1 : 0;
  }

  double p = static_cast<double>(TP) / A.rows();
  double r = static_cast<double>(TP) / Agt.rows();

  return {p, r};
}

} // ns utils