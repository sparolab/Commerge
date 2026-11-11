// Exchange policy, registration of the exchanged submaps (KISS-Matcher + GICP) and the merge error.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "commerge/commerge.hpp"

#ifdef COMMERGE_WITH_REGISTRATION
#include <kiss_matcher/KISSMatcher.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#endif

namespace commerge {

double region_score(const Region& r, const Matrix& affinity, const Positions& pos_a, const Positions& pos_b,
                    double consistency) {
  if (r.pairs.size() < 2) return static_cast<double>(r.pairs.size());
  std::vector<double> w;
  for (auto& p : r.pairs) w.push_back(affinity(p.first, p.second));
  return consistency_scores(r.pairs, w, pos_a, pos_b, consistency).sum();
}

std::vector<int> choose_regions(const Selection& sel, const Matrix& affinity, const Positions& pos_a,
                                const Positions& pos_b, const std::string& policy, double consistency) {
  if (policy != "score" && policy != "area" && policy != "candidates" && policy != "all")
    throw std::invalid_argument("policy must be score, area, candidates or all");
  std::vector<int> live;
  for (size_t k = 0; k < sel.regions.size(); ++k)
    if (!sel.regions[k].pairs.empty()) live.push_back(static_cast<int>(k));
  if (policy == "all" || live.empty()) return live;
  auto key = [&](int k) -> double {
    const Region& r = sel.regions[k];
    if (policy == "score") return region_score(r, affinity, pos_a, pos_b, consistency);
    if (policy == "area") return static_cast<double>(r.box.a1 - r.box.a0) * (r.box.b1 - r.box.b0);
    return r.candidates;
  };
  int best = live[0];
  double bv = key(best);
  for (size_t i = 1; i < live.size(); ++i) {
    double v = key(live[i]);
    if (v > bv) best = live[i], bv = v;  // first maximum, as Python's max()
  }
  return {best};
}

bool registration_available() {
#ifdef COMMERGE_WITH_REGISTRATION
  return true;
#else
  return false;
#endif
}

std::optional<Registration> register_submaps(const Cloud& submap_a, const Cloud& submap_b, double voxel,
                                             double max_corr) {
#ifdef COMMERGE_WITH_REGISTRATION
  auto t0 = std::chrono::steady_clock::now();
  std::vector<Eigen::Vector3f> src, tgt;
  for (long i = 0; i < submap_b.rows(); ++i) src.emplace_back(submap_b(i, 0), submap_b(i, 1), submap_b(i, 2));
  for (long i = 0; i < submap_a.rows(); ++i) tgt.emplace_back(submap_a(i, 0), submap_a(i, 1), submap_a(i, 2));
  kiss_matcher::KISSMatcher matcher(kiss_matcher::KISSMatcherConfig(static_cast<float>(voxel)));
  kiss_matcher::RegistrationSolution sol = matcher.estimate(src, tgt);
  const int inliers = static_cast<int>(matcher.getNumFinalInliers());
  if (!sol.valid || inliers < 5) return std::nullopt;
  Eigen::Isometry3d T0 = Eigen::Isometry3d::Identity();
  T0.linear() = sol.rotation;
  T0.translation() = sol.translation;

  std::vector<Eigen::Vector3d> s3, t3;
  for (auto& p : src) s3.push_back(p.cast<double>());
  for (auto& p : tgt) t3.push_back(p.cast<double>());
  small_gicp::RegistrationSetting setting;
  setting.type = small_gicp::RegistrationSetting::GICP;
  setting.downsampling_resolution = voxel;
  setting.max_correspondence_distance = max_corr;
  setting.num_threads = 4;
  small_gicp::RegistrationResult res = small_gicp::align(t3, s3, T0, setting);
  Registration reg;
  reg.T_a_b = (res.converged ? res.T_target_source : T0).matrix();
  reg.inliers = inliers;
  reg.fitness = res.converged ? res.error / std::max<size_t>(res.num_inliers, 1) : std::nan("");
  reg.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return reg;
#else
  (void)submap_a, (void)submap_b, (void)voxel, (void)max_corr;
  throw std::runtime_error("this build has no registration (needs KISS-Matcher and small_gicp)");
#endif
}

std::optional<Registration> register_pairs(const Poses& poses_a, const Poses& poses_b, const std::vector<Pair>& pairs,
                                           int min_pairs, double min_spread) {
  auto t0 = std::chrono::steady_clock::now();
  const int n = static_cast<int>(pairs.size());
  if (n < min_pairs) return std::nullopt;
  Eigen::MatrixX3d P(n, 3), Q(n, 3);
  for (int k = 0; k < n; ++k) {
    Q.row(k) = poses_a.at(pairs[k].first).block<3, 1>(0, 3).transpose();
    P.row(k) = poses_b.at(pairs[k].second).block<3, 1>(0, 3).transpose();
  }
  Eigen::Vector2d span = P.leftCols<2>().colwise().maxCoeff() - P.leftCols<2>().colwise().minCoeff();
  if (span.maxCoeff() < min_spread) return std::nullopt;
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);  // numpy's median
  };
  std::vector<bool> keep(n, true);
  Pose T = Pose::Identity();
  Eigen::VectorXd r(n);
  for (int it = 0; it < 5; ++it) {
    Eigen::Vector2d mp = Eigen::Vector2d::Zero(), mq = Eigen::Vector2d::Zero();
    int m = 0;
    for (int k = 0; k < n; ++k)
      if (keep[k]) mp += P.row(k).head<2>().transpose(), mq += Q.row(k).head<2>().transpose(), ++m;
    mp /= m, mq /= m;
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    std::vector<double> dz;
    for (int k = 0; k < n; ++k)
      if (keep[k]) {
        H += (P.row(k).head<2>().transpose() - mp) * (Q.row(k).head<2>().transpose() - mq).transpose();
        dz.push_back(Q(k, 2) - P(k, 2));
      }
    const double yaw = std::atan2(H(0, 1) - H(1, 0), H(0, 0) + H(1, 1));
    T = Pose::Identity();
    T.block<2, 2>(0, 0) << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
    T.block<2, 1>(0, 3) = mq - T.block<2, 2>(0, 0) * mp;
    T(2, 3) = median(dz);
    std::vector<double> rk;
    for (int k = 0; k < n; ++k) {
      r(k) = (T.block<3, 3>(0, 0) * P.row(k).transpose() + T.block<3, 1>(0, 3) - Q.row(k).transpose()).norm();
      if (keep[k]) rk.push_back(r(k));
    }
    const double gate = std::max(2.0 * median(rk), 1.0);
    std::vector<bool> next(n);
    int count = 0;
    for (int k = 0; k < n; ++k) next[k] = r(k) <= gate, count += next[k];
    if (count < min_pairs || next == keep) break;
    keep = next;
  }
  double ss = 0;
  int m = 0;
  for (int k = 0; k < n; ++k)
    if (keep[k]) ss += r(k) * r(k), ++m;
  Registration reg;
  reg.T_a_b = T;
  reg.inliers = m;
  reg.fitness = std::sqrt(ss / m);
  reg.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return reg;
}

namespace {
Pose rigid(const Poses& from, const Poses& to) {
  Eigen::Matrix3Xd s(3, from.size()), d(3, to.size());
  for (size_t k = 0; k < from.size(); ++k) s.col(k) = from[k].block<3, 1>(0, 3), d.col(k) = to[k].block<3, 1>(0, 3);
  return Eigen::umeyama(s, d, false);
}
}  // namespace

AlignmentError alignment_error(const Pose& T_a_b, const Poses& poses_a, const Poses& poses_b, const Poses& gt_a,
                               const Poses& gt_b) {
  if (poses_a.size() != gt_a.size() || poses_b.size() != gt_b.size())
    throw std::invalid_argument("ground truth and keyframe poses differ in length");
  Pose G_a = rigid(poses_a, gt_a), G_b = rigid(poses_b, gt_b);
  Pose T_true = G_a.inverse() * G_b;
  Pose E = T_true.inverse() * T_a_b;
  double c = std::clamp((E.block<3, 3>(0, 0).trace() - 1.0) / 2.0, -1.0, 1.0);
  double merged = 0.0;
  for (size_t k = 0; k < poses_b.size(); ++k)
    merged += ((G_a * T_a_b * poses_b[k]).block<3, 1>(0, 3) - gt_b[k].block<3, 1>(0, 3)).norm();
  return {E.block<3, 1>(0, 3).norm(), std::acos(c) * 180.0 / M_PI, merged / std::max<size_t>(poses_b.size(), 1)};
}

}  // namespace commerge
