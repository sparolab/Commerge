// Python bindings of the C++ core: `import commerge_cpp`.
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "commerge/commerge.hpp"

namespace py = pybind11;
using namespace commerge;

namespace {
Poses to_poses(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& a) {
  // (N, 16) row-major 4x4 poses, or (N, 12) KITTI rows
  if (a.cols() != 16 && a.cols() != 12) throw std::invalid_argument("poses must be (N, 4, 4) or (N, 12)");
  Poses out(a.rows(), Pose::Identity());
  for (Eigen::Index k = 0; k < a.rows(); ++k)
    for (int i = 0; i < a.cols(); ++i) out[k](i / 4, i % 4) = a(k, i);
  return out;
}
}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Commerge C++ core: scan selection (steps A, B, C), solvers, registration";
  m.attr("__version__") = kVersion;

  m.def("cosine_affinity", &cosine_affinity, py::arg("desc_a"), py::arg("desc_b"));
  m.def("enhance_affinity", &enhance_affinity, py::arg("affinity"), py::arg("length"));
  m.def("extract_regions", [](const Matrix& aff, double thr, int min_area) {
    std::vector<std::tuple<int, int, int, int>> out;
    for (auto& b : extract_regions(aff, thr, min_area)) out.emplace_back(b.a0, b.a1, b.b0, b.b1);
    return out;
  }, py::arg("affinity"), py::arg("threshold"), py::arg("min_area"));
  m.def("gurobi_available", &gurobi_available);
  m.def("registration_available", &registration_available);
  m.def("min_vertex_cover", &min_vertex_cover, py::arg("na"), py::arg("nb"), py::arg("edges"), py::arg("weights"),
        py::arg("solver") = "auto", py::arg("time_limit") = 0.0, py::arg("verbose") = false);
  m.def("max_edge_weight_clique", &max_edge_weight_clique, py::arg("M"), py::arg("solver") = "auto",
        py::arg("time_limit") = 0.0, py::arg("verbose") = false);

  py::class_<Region>(m, "Region")
      .def_property_readonly("box", [](const Region& r) { return std::make_tuple(r.box.a0, r.box.a1, r.box.b0, r.box.b1); })
      .def_readonly("candidates", &Region::candidates)
      .def_readonly("cover_a", &Region::cover_a)
      .def_readonly("cover_b", &Region::cover_b)
      .def_readonly("pairs", &Region::pairs)
      .def_readonly("seconds", &Region::seconds)
      .def_property_readonly("keyframes_a", &Region::keyframes_a)
      .def_property_readonly("keyframes_b", &Region::keyframes_b);
  py::class_<Selection>(m, "Selection")
      .def_readonly("regions", &Selection::regions)
      .def_readonly("steps", &Selection::steps)
      .def_readonly("solver", &Selection::solver)
      .def_property_readonly("keyframes_a", &Selection::keyframes_a)
      .def_property_readonly("keyframes_b", &Selection::keyframes_b)
      .def_property_readonly("pairs", [](const Selection& s) {
        std::vector<Pair> p;
        for (auto& r : s.regions) p.insert(p.end(), r.pairs.begin(), r.pairs.end());
        return p;
      });

  m.def("select",
        [](const Matrix& aff, const Positions& pos_a, const Positions& pos_b, double threshold, int min_area,
           double consistency, const std::string& steps, const std::string& solver, double time_limit, bool verbose,
           bool tie_break, py::object log) {
          SelectOptions o{threshold, min_area, consistency, steps, solver, time_limit, verbose, tie_break};
          Logger lg = nullptr;
          if (!log.is_none()) lg = [log](const std::string& s) { py::gil_scoped_acquire g; log(s); };
          return select(aff, pos_a, pos_b, o, lg);
        },
        py::arg("affinity"), py::arg("pos_a"), py::arg("pos_b"), py::arg("threshold"), py::arg("min_area"),
        py::arg("consistency") = 3.0, py::arg("steps") = "ABC", py::arg("solver") = "auto",
        py::arg("time_limit") = 0.0, py::arg("verbose") = false, py::arg("tie_break") = true,
        py::arg("log") = py::none(),
        "Steps (A), (B), (C) on an (Na, Nb) affinity; pos_a/pos_b are keyframe positions (N, 3).");
  m.def("choose_regions",
        [](const Selection& s, const Matrix& aff, const Positions& pa, const Positions& pb, const std::string& policy,
           double consistency) { return choose_regions(s, aff, pa, pb, policy, consistency); },
        py::arg("selection"), py::arg("affinity"), py::arg("pos_a"), py::arg("pos_b"), py::arg("policy") = "score",
        py::arg("consistency") = 3.0);

  py::class_<Registration>(m, "Registration")
      .def_readonly("T_a_b", &Registration::T_a_b)
      .def_readonly("inliers", &Registration::inliers)
      .def_readonly("fitness", &Registration::fitness)
      .def_readonly("seconds", &Registration::seconds);
  m.def("register", [](const Eigen::Ref<const Matrix>& a, const Eigen::Ref<const Matrix>& b, double voxel,
                       double max_corr) {
    auto cloud = [](const Eigen::Ref<const Matrix>& p) {
      Cloud c = Cloud::Zero(p.rows(), 4);
      c.leftCols(std::min<Eigen::Index>(p.cols(), 4)) = p.leftCols(std::min<Eigen::Index>(p.cols(), 4)).cast<float>();
      return c;
    };
    py::gil_scoped_release r;
    return register_submaps(cloud(a), cloud(b), voxel, max_corr);
  }, py::arg("submap_a"), py::arg("submap_b"), py::arg("voxel") = 0.5, py::arg("max_corr") = 2.0,
        "T_a_b aligning B's submap (N, 3+) to A's; None if the global registration fails.");
  m.def("register_pairs", [](const Eigen::Ref<const Matrix>& pa, const Eigen::Ref<const Matrix>& pb,
                             const std::vector<Pair>& pairs, int min_pairs, double min_spread) {
    return register_pairs(to_poses(pa), to_poses(pb), pairs, min_pairs, min_spread);
  }, py::arg("poses_a"), py::arg("poses_b"), py::arg("pairs"), py::arg("min_pairs") = 3, py::arg("min_spread") = 2.0,
        "Rigid fit (yaw + translation) of the pairs' keyframe positions; poses as (N, 16) or (N, 12) arrays.");
  m.def("alignment_error", [](const Pose& T, const Eigen::Ref<const Matrix>& pa, const Eigen::Ref<const Matrix>& pb,
                              const Eigen::Ref<const Matrix>& ga, const Eigen::Ref<const Matrix>& gb) {
    AlignmentError e = alignment_error(T, to_poses(pa), to_poses(pb), to_poses(ga), to_poses(gb));
    return std::make_tuple(e.trans_m, e.rot_deg, e.merged_mean_m);
  }, py::arg("T_a_b"), py::arg("poses_a"), py::arg("poses_b"), py::arg("gt_a"), py::arg("gt_b"),
        "(trans_m, rot_deg, merged_mean_m); poses as (N, 16) or (N, 12) arrays.");
}
