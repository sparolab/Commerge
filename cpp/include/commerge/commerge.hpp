// Commerge (C++): communication-efficient multi-robot LiDAR map merging.
//
// Same pipeline as the Python package `commerge`:
//   (A) sequential matching: connected regions of the affinity above a threshold,
//   (B) balanced minimum vertex cover of each region's candidate pairs,
//   (C) maximum edge-weighted clique of the mutually consistent pairs,
// then the exchanged submaps are registered (KISS-Matcher + GICP) and B's map is merged into A's frame.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace commerge {

inline constexpr const char* kVersion = "2.1.0";

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Pose = Eigen::Matrix4d;
using Poses = std::vector<Pose, Eigen::aligned_allocator<Pose>>;
using Positions = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Cloud = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;  // x y z intensity
using Pair = std::pair<int, int>;                                        // (keyframe of A, keyframe of B)
using Logger = std::function<void(const std::string&)>;

// ------------------------------------------------------------------------------------------------
// Selection (steps A, B, C)
// ------------------------------------------------------------------------------------------------
Matrix cosine_affinity(const Matrix& desc_a, const Matrix& desc_b);
Matrix enhance_affinity(const Matrix& affinity, int length);

struct Box {
  int a0, a1, b0, b1;  // half-open; rows index robot A, columns robot B
};
std::vector<Box> extract_regions(const Matrix& affinity, double threshold, int min_area);

// Pair-score matrix M (strict upper triangle): w_p * exp(-e_pq) * w_q where e_pq <= max_error.
Matrix consistency_scores(const std::vector<Pair>& pairs, const std::vector<double>& weights,
                          const Positions& pos_a, const Positions& pos_b, double max_error);

struct Region {
  Box box{};
  int candidates = 0;
  std::vector<int> cover_a, cover_b;
  std::vector<Pair> pairs;
  std::map<std::string, double> seconds;
  std::vector<int> keyframes_a() const;
  std::vector<int> keyframes_b() const;
};

struct Selection {
  std::vector<Region> regions;
  std::string steps, solver;
  std::vector<int> keyframes_a() const;
  std::vector<int> keyframes_b() const;
};

struct SelectOptions {
  double threshold = 0.96;
  int min_area = 2000;
  double consistency = 3.0;
  std::string steps = "ABC";
  std::string solver = "auto";  // auto | gurobi | highs
  double time_limit = 0.0;
  bool verbose = false;
  bool tie_break = true;
};

Selection select(const Matrix& affinity, const Positions& pos_a, const Positions& pos_b,
                 const SelectOptions& opt, const Logger& log = nullptr);

// ------------------------------------------------------------------------------------------------
// Solvers
// ------------------------------------------------------------------------------------------------
bool gurobi_available();  // compiled in and licensed
std::string resolve_solver(const std::string& solver);

// Minimum-weight cover of the bipartite graph (A nodes 0..na-1, B nodes na..na+nb-1), with at least
// min(na, nb) / 2 nodes on each side. Returns the chosen node indices.
std::vector<int> min_vertex_cover(int na, int nb, const std::vector<std::pair<int, int>>& edges,
                                  const std::vector<double>& weights, const std::string& solver,
                                  double time_limit = 0.0, bool verbose = false);

// maximize sum_{i<j} M_ij x_i x_j  s.t.  x_i + x_j <= 1 wherever M_ij = 0 (strict upper triangle of M).
std::vector<int> max_edge_weight_clique(const Matrix& M, const std::string& solver, double time_limit = 0.0,
                                        bool verbose = false);

// ------------------------------------------------------------------------------------------------
// Sessions (SOLiD / LT-mapper format)
// ------------------------------------------------------------------------------------------------
Poses read_kitti_poses(const std::string& path);
void write_kitti_poses(const std::string& path, const Poses& poses);
Cloud read_pcd(const std::string& path);
void write_pcd(const std::string& path, const Cloud& cloud);
Matrix read_npy(const std::string& path);
void write_npy(const std::string& path, const Matrix& m);  // float32

struct Session {
  std::string name, root;
  Poses poses;
  std::vector<std::string> scans;
  Matrix rsolid, asolid;  // (N, D); empty without SOLiDs/
  static Session load(const std::string& root, const std::string& poses = "auto");
  int size() const { return static_cast<int>(poses.size()); }
  Positions positions() const;
  Cloud cloud_in_world(int k, double min_range = 0.0) const;
};

Cloud build_submap(const Session& s, const std::vector<int>& keyframes, double min_range = 3.0, double voxel = 0.0);
Cloud voxel_downsample(const Cloud& cloud, double leaf);

// ------------------------------------------------------------------------------------------------
// Exchange and merging
// ------------------------------------------------------------------------------------------------
double region_score(const Region& r, const Matrix& affinity, const Positions& pos_a, const Positions& pos_b,
                    double consistency = 3.0);
// policy: score | area | candidates | all
std::vector<int> choose_regions(const Selection& sel, const Matrix& affinity, const Positions& pos_a,
                                const Positions& pos_b, const std::string& policy = "score",
                                double consistency = 3.0);

struct Registration {
  Pose T_a_b = Pose::Identity();  // B's frame in A's frame
  int inliers = 0;                // KISS-Matcher final inliers
  double fitness = 0.0;           // GICP error per inlier
  double seconds = 0.0;
};
bool registration_available();  // built with KISS-Matcher and small_gicp
std::optional<Registration> register_submaps(const Cloud& submap_a, const Cloud& submap_b, double voxel = 0.5,
                                             double max_corr = 2.0);

// Registration without point clouds, for any sensor: B's keyframe positions of the selected pairs fitted onto
// A's (yaw + translation, gravity up), dropping pairs more than twice the median residual off (at least 1 m).
// inliers: pairs kept; fitness: their RMS residual (m). nullopt with fewer than min_pairs pairs or when B's
// keyframes span less than min_spread m.
std::optional<Registration> register_pairs(const Poses& poses_a, const Poses& poses_b, const std::vector<Pair>& pairs,
                                           int min_pairs = 3, double min_spread = 2.0);

struct AlignmentError {
  double trans_m, rot_deg, merged_mean_m;
};
AlignmentError alignment_error(const Pose& T_a_b, const Poses& poses_a, const Poses& poses_b, const Poses& gt_a,
                               const Poses& gt_b);

}  // namespace commerge
