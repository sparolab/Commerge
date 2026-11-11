// Steps (A), (B), (C): which keyframes two robots exchange.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>

#include "commerge/commerge.hpp"

namespace commerge {
namespace {

double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

std::string fmt(const char* f, double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), f, v);
  return buf;
}

}  // namespace

std::vector<int> Region::keyframes_a() const {
  std::set<int> s;
  for (auto& p : pairs) s.insert(p.first);
  return {s.begin(), s.end()};
}

std::vector<int> Region::keyframes_b() const {
  std::set<int> s;
  for (auto& p : pairs) s.insert(p.second);
  return {s.begin(), s.end()};
}

std::vector<int> Selection::keyframes_a() const {
  std::set<int> s;
  for (auto& r : regions)
    for (int a : r.keyframes_a()) s.insert(a);
  return {s.begin(), s.end()};
}

std::vector<int> Selection::keyframes_b() const {
  std::set<int> s;
  for (auto& r : regions)
    for (int b : r.keyframes_b()) s.insert(b);
  return {s.begin(), s.end()};
}

Matrix cosine_affinity(const Matrix& desc_a, const Matrix& desc_b) {
  auto normalize = [](Matrix d) {
    for (Eigen::Index i = 0; i < d.rows(); ++i) {
      for (Eigen::Index j = 0; j < d.cols(); ++j)
        if (!std::isfinite(d(i, j))) d(i, j) = 0.0;
      double n = d.row(i).norm();
      if (n > 0) d.row(i) /= n;
    }
    return d;
  };
  return normalize(desc_a) * normalize(desc_b).transpose();
}

Matrix enhance_affinity(const Matrix& S, int length) {
  const Eigen::Index na = S.rows(), nb = S.cols();
  Matrix z(na, nb);
  Eigen::VectorXd rm = S.rowwise().mean(), cm = S.colwise().mean().transpose();
  Eigen::VectorXd rs(na), cs(nb);
  for (Eigen::Index i = 0; i < na; ++i) rs(i) = std::sqrt((S.row(i).array() - rm(i)).square().mean());
  for (Eigen::Index j = 0; j < nb; ++j) cs(j) = std::sqrt((S.col(j).array() - cm(j)).square().mean());
  for (Eigen::Index i = 0; i < na; ++i)
    for (Eigen::Index j = 0; j < nb; ++j)
      z(i, j) = 0.5 * ((S(i, j) - rm(i)) / (rs(i) + 1e-12) + (S(i, j) - cm(j)) / (cs(j) + 1e-12));
  if (length <= 1) return z;
  // scipy.ndimage.convolve(z, eye(L) / L, mode="constant") and its anti-diagonal twin, centred like scipy
  const int c = length / 2;
  Matrix out(na, nb);
  for (Eigen::Index i = 0; i < na; ++i)
    for (Eigen::Index j = 0; j < nb; ++j) {
      double d = 0, ad = 0;
      for (int k = 0; k < length; ++k) {
        // convolution flips the kernel: out[i,j] = sum_k K[k,l] z[i + c - k, j + c - l]
        Eigen::Index r = i + c - k;
        Eigen::Index cd = j + c - k;                 // K = eye: l = k
        Eigen::Index ca = j + c - (length - 1 - k);  // K = eye[::-1]: l = L-1-k
        if (r < 0 || r >= na) continue;
        if (cd >= 0 && cd < nb) d += z(r, cd);
        if (ca >= 0 && ca < nb) ad += z(r, ca);
      }
      out(i, j) = std::max(d, ad) / length;
    }
  return out;
}

std::vector<Box> extract_regions(const Matrix& affinity, double threshold, int min_area) {
  // 4-connected components in raster order (scipy.ndimage.label default), bounding boxes.
  const int na = static_cast<int>(affinity.rows()), nb = static_cast<int>(affinity.cols());
  std::vector<int> label(static_cast<size_t>(na) * nb, 0);
  std::vector<Box> boxes;
  std::vector<int> stack;
  int next = 0;
  for (int i = 0; i < na; ++i)
    for (int j = 0; j < nb; ++j) {
      size_t id = static_cast<size_t>(i) * nb + j;
      if (label[id] || !(affinity(i, j) > threshold)) continue;
      ++next;
      Box b{i, i + 1, j, j + 1};
      label[id] = next;
      stack.assign(1, static_cast<int>(id));
      while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        int r = cur / nb, c = cur % nb;
        b.a0 = std::min(b.a0, r), b.a1 = std::max(b.a1, r + 1);
        b.b0 = std::min(b.b0, c), b.b1 = std::max(b.b1, c + 1);
        const int nr[4] = {r - 1, r + 1, r, r}, nc[4] = {c, c, c - 1, c + 1};
        for (int k = 0; k < 4; ++k) {
          if (nr[k] < 0 || nr[k] >= na || nc[k] < 0 || nc[k] >= nb) continue;
          size_t nid = static_cast<size_t>(nr[k]) * nb + nc[k];
          if (!label[nid] && affinity(nr[k], nc[k]) > threshold) {
            label[nid] = next;
            stack.push_back(static_cast<int>(nid));
          }
        }
      }
      if (static_cast<long>(b.a1 - b.a0) * (b.b1 - b.b0) >= min_area) boxes.push_back(b);
    }
  return boxes;
}

Matrix consistency_scores(const std::vector<Pair>& pairs, const std::vector<double>& w, const Positions& pos_a,
                          const Positions& pos_b, double max_error) {
  const int n = static_cast<int>(pairs.size());
  Matrix M = Matrix::Zero(n, n);
  for (int p = 0; p < n; ++p)
    for (int q = p + 1; q < n; ++q) {
      double da = (pos_a.row(pairs[p].first) - pos_a.row(pairs[q].first)).norm();
      double db = (pos_b.row(pairs[p].second) - pos_b.row(pairs[q].second)).norm();
      double e = std::abs(da - db);
      if (e <= max_error) M(p, q) = w[p] * std::exp(-e) * w[q];
    }
  return M;
}

Selection select(const Matrix& affinity, const Positions& pos_a, const Positions& pos_b, const SelectOptions& opt,
                 const Logger& log) {
  std::string steps;
  for (char ch : opt.steps) {
    char u = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (u == ',' || u == ' ') continue;
    if (u != 'A' && u != 'B' && u != 'C') throw std::invalid_argument(std::string("unknown step ") + ch);
  }
  for (char s : std::string("ABC"))
    for (char ch : opt.steps)
      if (std::toupper(static_cast<unsigned char>(ch)) == s) {
        steps += s;
        break;
      }
  const bool A = steps.find('A') != std::string::npos, B = steps.find('B') != std::string::npos,
             C = steps.find('C') != std::string::npos;
  std::string solver = (B || C) ? resolve_solver(opt.solver) : opt.solver;
  const int na = static_cast<int>(affinity.rows()), nb = static_cast<int>(affinity.cols());
  if (pos_a.rows() < na || pos_b.rows() < nb) throw std::invalid_argument("fewer poses than affinity rows/columns");

  double t0 = now();
  std::vector<Box> boxes = A ? extract_regions(affinity, opt.threshold, opt.min_area) : std::vector<Box>{{0, na, 0, nb}};
  if (log) {
    std::string st;
    for (char s : steps) st += (st.empty() ? "(" : "+(") + std::string(1, s) + ")";
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "[commerge] steps %s, solver %s: (A) %zu region(s) (threshold %g, min area %d) in %.3f s",
                  st.empty() ? "none" : st.c_str(), solver.c_str(), boxes.size(), opt.threshold, opt.min_area,
                  now() - t0);
    log(buf);
  }

  Selection sel;
  sel.steps = steps;
  sel.solver = solver;
  for (size_t k = 0; k < boxes.size(); ++k) {
    const Box& bx = boxes[k];
    Region r;
    r.box = bx;
    std::vector<Pair> pairs;
    for (int i = bx.a0; i < bx.a1; ++i)
      for (int j = bx.b0; j < bx.b1; ++j)
        if (affinity(i, j) >= opt.threshold) pairs.emplace_back(i, j);
    r.candidates = static_cast<int>(pairs.size());
    if (pairs.empty()) {
      sel.regions.push_back(r);
      continue;
    }

    if (B) {
      double t = now();
      std::set<int> sa, sb;
      for (auto& p : pairs) sa.insert(p.first), sb.insert(p.second);
      std::vector<int> an(sa.begin(), sa.end()), bn(sb.begin(), sb.end());
      std::map<int, int> ia, ib;
      for (size_t u = 0; u < an.size(); ++u) ia[an[u]] = static_cast<int>(u);
      for (size_t u = 0; u < bn.size(); ++u) ib[bn[u]] = static_cast<int>(an.size() + u);
      const int nn = static_cast<int>(an.size() + bn.size());
      std::vector<std::pair<int, int>> edges;
      std::vector<double> score(nn, 0.0);
      for (auto& p : pairs) {
        int u = ia[p.first], v = ib[p.second];
        edges.emplace_back(u, v);
        score[u] += affinity(p.first, p.second);
        score[v] += affinity(p.first, p.second);
      }
      // Minimum-cardinality cover; a tie-break far below 1 prefers the keyframes with the strongest pairs.
      std::vector<double> weights(nn, 1.0);
      if (opt.tie_break) {
        double top = *std::max_element(score.begin(), score.end());
        double eps = 0.5 / (nn + 1);
        for (int u = 0; u < nn; ++u) weights[u] = 1.0 - eps * score[u] / top;
      }
      std::vector<int> cover = min_vertex_cover(static_cast<int>(an.size()), static_cast<int>(bn.size()), edges,
                                                weights, solver, opt.time_limit, opt.verbose);
      for (int u : cover) {
        if (u < static_cast<int>(an.size()))
          r.cover_a.push_back(an[u]);
        else
          r.cover_b.push_back(bn[u - an.size()]);
      }
      std::sort(r.cover_a.begin(), r.cover_a.end());
      std::sort(r.cover_b.begin(), r.cover_b.end());
      std::set<int> ca(r.cover_a.begin(), r.cover_a.end()), cb(r.cover_b.begin(), r.cover_b.end());
      std::vector<Pair> kept;
      for (auto& p : pairs)
        if (ca.count(p.first) && cb.count(p.second)) kept.push_back(p);
      pairs.swap(kept);
      r.seconds["B"] = now() - t;
    } else {
      std::set<int> sa, sb;
      for (auto& p : pairs) sa.insert(p.first), sb.insert(p.second);
      r.cover_a.assign(sa.begin(), sa.end());
      r.cover_b.assign(sb.begin(), sb.end());
    }

    if (C && pairs.size() > 1) {
      double t = now();
      std::vector<double> w;
      for (auto& p : pairs) w.push_back(affinity(p.first, p.second));
      Matrix M = consistency_scores(pairs, w, pos_a, pos_b, opt.consistency);
      std::vector<int> chosen = max_edge_weight_clique(M, solver, opt.time_limit, opt.verbose);
      std::vector<Pair> kept;
      for (int c : chosen) kept.push_back(pairs[c]);
      pairs.swap(kept);
      r.seconds["C"] = now() - t;
    }
    r.pairs = pairs;
    if (log) {
      char buf[512];
      std::snprintf(buf, sizeof(buf),
                    "[commerge] region %zu: A %d-%d x B %d-%d: (A) %d candidate pairs -> (B) %zu+%zu keyframes -> "
                    "(C) %zu pairs = %zu A + %zu B scans",
                    k, bx.a0, bx.a1 - 1, bx.b0, bx.b1 - 1, r.candidates, r.cover_a.size(), r.cover_b.size(),
                    r.pairs.size(), r.keyframes_a().size(), r.keyframes_b().size());
      std::string msg = buf;
      for (auto& [s, v] : r.seconds) msg += ", (" + s + ") " + fmt("%.2f s", v);
      log(msg);
    }
    sel.regions.push_back(r);
  }
  return sel;
}

}  // namespace commerge
