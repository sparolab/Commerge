// commerge_cpp: the `commerge` command in C++ (same options and outputs, no --plot).
//
//   commerge_cpp <session A> <session B> --out merged/ --threshold 0.96 --min-area 2000 [options]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include "commerge/commerge.hpp"

namespace fs = std::filesystem;
using namespace commerge;

namespace {

const char* kUsage = R"(usage: commerge_cpp SESSION_A SESSION_B --out DIR --threshold T --min-area N [options]

Select the scans two robots exchange to merge their maps (Commerge, C++).
Sessions are SOLiD outputs (Scans/, SOLiDs/, *_poses.txt).

affinity:
  --affinity FILE.npy     (Na, Nb) affinity of any descriptor, instead of SOLiD
  --descriptor D          rsolid (default) | asolid | both
  --poses P               auto (default) | optimized | odom | a KITTI pose file
selection:
  --threshold T           affinity threshold of a candidate pair (required)
  --min-area N            smallest region, cells of its bounding box (required)
  --enhance L             sequence-enhance the affinity over L keyframes (z-scores; thresholds ~3)
  --consistency M         largest distance mismatch (m) of two consistent pairs (default 3)
  --steps S               any subset of ABC (default ABC)
  --solver S              auto (default) | gurobi | highs
  --time-limit S          per-problem solver time limit (0: none)
exchange and merging:
  --policy P              score (default) | area | candidates | all
  --no-register           do not register the exchanged submaps
  --registration R        kiss (default): KISS-Matcher + GICP on the exchanged scans;
                          pairs: fit the selected pairs' keyframe positions (no scans: any sensor)
  --reg-voxel V           registration voxel (m, default 0.5)
  --gt-a FILE --gt-b FILE ground-truth keyframe poses (KITTI, common frame): report the error
output:
  --no-submaps            only write the selection
  --min-range R           drop points closer than R m (default 3)
  --voxel V               submap voxel size (m, 0: none)
  --verbose               solver logs
  --version
)";

double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

std::string json_ints(const std::vector<int>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "]";
}

std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

std::string num(double v) {
  if (std::isnan(v)) return "NaN";
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

Matrix descriptor(const Session& s, const std::string& which) {
  if (s.rsolid.size() == 0) throw std::runtime_error(s.name + ": no SOLiDs/ folder; give the affinity with --affinity");
  if (which == "rsolid") return s.rsolid;
  if (s.asolid.size() == 0) throw std::runtime_error(s.name + ": SOLiDs/ has no A-SOLiD line");
  if (which == "asolid") return s.asolid;
  Matrix both(s.rsolid.rows(), s.rsolid.cols() + s.asolid.cols());
  both << s.rsolid, s.asolid;
  return both;
}

int run(int argc, char** argv) {
  std::vector<std::string> pos;
  std::string out, affinity_file, desc = "rsolid", poses = "auto", policy = "score", gt_a, gt_b, method = "kiss";
  SelectOptions opt;
  bool have_thr = false, have_area = false, no_register = false, no_submaps = false;
  int enhance = 0;
  double reg_voxel = 0.5, min_range = 3.0, voxel = 0.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      std::cout << kUsage;
      return 0;
    } else if (a == "--version") {
      std::cout << "commerge " << kVersion << " (C++)\n";
      return 0;
    } else if (a == "--out") out = val();
    else if (a == "--affinity") affinity_file = val();
    else if (a == "--descriptor") desc = val();
    else if (a == "--poses") poses = val();
    else if (a == "--threshold") opt.threshold = std::stod(val()), have_thr = true;
    else if (a == "--min-area") opt.min_area = std::stoi(val()), have_area = true;
    else if (a == "--enhance") enhance = std::stoi(val());
    else if (a == "--consistency") opt.consistency = std::stod(val());
    else if (a == "--steps") opt.steps = val();
    else if (a == "--solver") opt.solver = val();
    else if (a == "--time-limit") opt.time_limit = std::stod(val());
    else if (a == "--policy") policy = val();
    else if (a == "--no-register") no_register = true;
    else if (a == "--reg-voxel") reg_voxel = std::stod(val());
    else if (a == "--registration") method = val();
    else if (a == "--gt-a") gt_a = val();
    else if (a == "--gt-b") gt_b = val();
    else if (a == "--no-submaps") no_submaps = true;
    else if (a == "--min-range") min_range = std::stod(val());
    else if (a == "--voxel") voxel = std::stod(val());
    else if (a == "--verbose") opt.verbose = true;
    else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a + " (see --help)");
    else pos.push_back(a);
  }
  if (method != "kiss" && method != "pairs") throw std::runtime_error("--registration must be kiss or pairs");
  if (pos.size() != 2 || out.empty() || !have_thr || !have_area) {
    std::cerr << kUsage;
    return 2;
  }
  fs::create_directories(out);
  auto log = [](const std::string& s) { std::cout << s << std::endl; };

  Session A = Session::load(pos[0], poses), B = Session::load(pos[1], poses);
  std::printf("[commerge] %s: %d keyframes | %s: %d keyframes\n", A.name.c_str(), A.size(), B.name.c_str(), B.size());
  Matrix aff;
  if (!affinity_file.empty()) {
    aff = read_npy(affinity_file);
  } else {
    double t = now();
    aff = cosine_affinity(descriptor(A, desc), descriptor(B, desc));
    std::printf("[commerge] affinity (%ld, %ld) from %s in %.2f s\n", static_cast<long>(aff.rows()),
                static_cast<long>(aff.cols()), desc.c_str(), now() - t);
    write_npy((fs::path(out) / "affinity.npy").string(), aff);
  }
  if (enhance) {
    aff = enhance_affinity(aff, enhance);
    std::printf("[commerge] affinity sequence-enhanced over %d keyframes\n", enhance);
  }
  Positions pa = A.positions(), pb = B.positions();
  Selection sel = select(aff, pa, pb, opt, log);

  {
    std::ofstream f(fs::path(out) / "pairs.csv");
    f << "region,keyframe_a,keyframe_b,affinity\r\n";
    char buf[96];
    for (size_t k = 0; k < sel.regions.size(); ++k)
      for (auto& [a, b] : sel.regions[k].pairs) {
        std::snprintf(buf, sizeof(buf), "%zu,%d,%d,%.6f\r\n", k, a, b, aff(a, b));
        f << buf;
      }
  }

  std::vector<int> chosen = choose_regions(sel, aff, pa, pb, policy, opt.consistency);
  std::set<int> sa, sb;
  for (int k : chosen) {
    for (int a : sel.regions[k].keyframes_a()) sa.insert(a);
    for (int b : sel.regions[k].keyframes_b()) sb.insert(b);
  }
  std::vector<int> ka(sa.begin(), sa.end()), kb(sb.begin(), sb.end());

  std::ostringstream js;
  js << "{\n \"session_a\": " << json_str(pos[0]) << ",\n \"session_b\": " << json_str(pos[1])
     << ",\n \"keyframes\": [" << A.size() << ", " << B.size() << "],\n \"threshold\": " << num(opt.threshold)
     << ",\n \"min_area\": " << opt.min_area << ",\n \"consistency\": " << num(opt.consistency)
     << ",\n \"steps\": " << json_str(sel.steps) << ",\n \"solver\": " << json_str(sel.solver)
     << ",\n \"selected_scans\": [" << sel.keyframes_a().size() << ", " << sel.keyframes_b().size()
     << "],\n \"regions\": [";
  for (size_t k = 0; k < sel.regions.size(); ++k) {
    const Region& r = sel.regions[k];
    js << (k ? "," : "") << "\n  {\"box_a\": [" << r.box.a0 << ", " << r.box.a1 << "], \"box_b\": [" << r.box.b0
       << ", " << r.box.b1 << "], \"A_candidates\": " << r.candidates << ", \"B_cover\": [" << r.cover_a.size()
       << ", " << r.cover_b.size() << "], \"C_pairs\": " << r.pairs.size()
       << ", \"keyframes_a\": " << json_ints(r.keyframes_a()) << ", \"keyframes_b\": " << json_ints(r.keyframes_b())
       << ", \"seconds\": {";
    bool first = true;
    for (auto& [s, v] : r.seconds) js << (first ? "" : ", ") << json_str(s) << ": " << num(v), first = false;
    js << "}}";
  }
  js << "\n ],\n \"exchange\": {\"policy\": " << json_str(policy) << ", \"regions\": " << json_ints(chosen)
     << ", \"keyframes_a\": " << json_ints(ka) << ", \"keyframes_b\": " << json_ints(kb) << "}";

  if (!ka.empty() && !kb.empty() && !(no_submaps && no_register)) {
    const bool scans = !A.scans.empty() && !B.scans.empty();
    if (!scans && method != "pairs") throw std::runtime_error("no Scans/ in the sessions: use --registration pairs");
    Cloud sub_a, sub_b;
    if (scans) {
      sub_a = build_submap(A, ka, min_range, voxel);
      sub_b = build_submap(B, kb, min_range, voxel);
      if (!no_submaps) {
        write_pcd((fs::path(out) / "submap_A.pcd").string(), sub_a);
        write_pcd((fs::path(out) / "submap_B.pcd").string(), sub_b);
      }
    }
    if (!no_register) {
      std::optional<Registration> reg;
      bool skipped = false;
      if (method == "pairs") {
        std::vector<Pair> pairs;
        for (int k : chosen) pairs.insert(pairs.end(), sel.regions[k].pairs.begin(), sel.regions[k].pairs.end());
        reg = register_pairs(A.poses, B.poses, pairs);
      } else if (!registration_available()) {
        std::printf("[commerge] registration skipped: this build has no KISS-Matcher / small_gicp\n");
        skipped = true;
      } else {
        reg = register_submaps(sub_a, sub_b, reg_voxel);
      }
      if (reg) {
        {
          std::ofstream f(fs::path(out) / "T_a_b.txt");
          char buf[32];
          for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) {
              std::snprintf(buf, sizeof(buf), "%.9f", reg->T_a_b(r, c));
              f << buf << (c == 3 ? "\n" : " ");
            }
        }
        Poses merged;
        for (auto& T : B.poses) merged.push_back(reg->T_a_b * T);
        write_kitti_poses((fs::path(out) / "poses_b_in_a.txt").string(), merged);
        js << ",\n \"registration\": {\"inliers\": " << reg->inliers << ", \"gicp_error\": " << num(reg->fitness)
           << ", \"seconds\": " << num(reg->seconds);
        char msg[512];
        if (method == "pairs")
          std::snprintf(msg, sizeof(msg), "[commerge] registered B to A from %d selected pairs: RMS %.2f m", reg->inliers,
                        reg->fitness);
        else
          std::snprintf(msg, sizeof(msg), "[commerge] registered B to A: %d KISS-Matcher inliers, GICP in %.1f s",
                        reg->inliers, reg->seconds);
        std::string m = msg;
        if (!gt_a.empty() && !gt_b.empty()) {
          AlignmentError e = alignment_error(reg->T_a_b, A.poses, B.poses, read_kitti_poses(gt_a), read_kitti_poses(gt_b));
          js << ", \"error\": {\"trans_m\": " << num(e.trans_m) << ", \"rot_deg\": " << num(e.rot_deg)
             << ", \"merged_mean_m\": " << num(e.merged_mean_m) << "}";
          std::snprintf(msg, sizeof(msg), "; vs ground truth: %.2f m, %.2f deg (B keyframes %.2f m off on average)",
                        e.trans_m, e.rot_deg, e.merged_mean_m);
          m += msg;
        }
        js << "}";
        std::cout << m << std::endl;
      } else if (!skipped) {
        std::printf("[commerge] registration failed (too few inliers); try another --policy\n");
      }
    }
  }
  js << "\n}\n";
  std::ofstream(fs::path(out) / "selection.json") << js.str();

  const int total = A.size() + B.size();
  const size_t sent = ka.size() + kb.size();
  char size[64];
  const char* unit = "scans";
  if (method == "pairs" && !no_register) {
    std::snprintf(size, sizeof(size), "%.1f KB of poses", sent * 12 * 8 / 1e3);
    unit = "keyframes";
  } else if (!A.scans.empty() && !B.scans.empty()) {
    double mb = 0.0;
    for (int k : ka) mb += fs::file_size(A.scans[k]) / 1e6;
    for (int k : kb) mb += fs::file_size(B.scans[k]) / 1e6;
    std::snprintf(size, sizeof(size), "%.1f MB", mb);
  } else {
    std::snprintf(size, sizeof(size), "no Scans/");
    unit = "keyframes";
  }
  std::printf("[commerge] exchanged %zu A + %zu B %s (%.1f %% of %d, %s) from region(s) %s of %zu -> %s\n", ka.size(),
              kb.size(), unit, 100.0 * sent / std::max(total, 1), total, size, json_ints(chosen).c_str(),
              sel.regions.size(), out.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "commerge_cpp: " << e.what() << "\n";
    return 1;
  }
}
