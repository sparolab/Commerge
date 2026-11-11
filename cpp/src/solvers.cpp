// The two optimization problems, solved with HiGHS (always built) or Gurobi (C API, when found at build time).
#include <cmath>
#include <stdexcept>

#include "Highs.h"
#include "commerge/commerge.hpp"

#ifdef COMMERGE_WITH_GUROBI
extern "C" {
#include "gurobi_c.h"
}
#endif

namespace commerge {
namespace {

// Rows of a sparse constraint matrix, for HiGHS.
struct Rows {
  std::vector<HighsInt> start{0}, index;
  std::vector<double> value, lower, upper;
  void add(const std::vector<std::pair<int, double>>& row, double lo, double hi) {
    for (auto& [c, v] : row) index.push_back(c), value.push_back(v);
    start.push_back(static_cast<HighsInt>(index.size()));
    lower.push_back(lo), upper.push_back(hi);
  }
};

std::vector<double> solve_highs(const std::vector<double>& cost, const std::vector<bool>& integer, const Rows& rows,
                                double time_limit, bool verbose) {
  HighsLp lp;
  lp.num_col_ = static_cast<HighsInt>(cost.size());
  lp.num_row_ = static_cast<HighsInt>(rows.lower.size());
  lp.col_cost_ = cost;
  lp.col_lower_.assign(cost.size(), 0.0);
  lp.col_upper_.assign(cost.size(), 1.0);
  lp.row_lower_ = rows.lower;
  lp.row_upper_ = rows.upper;
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.num_col_ = lp.num_col_;
  lp.a_matrix_.num_row_ = lp.num_row_;
  lp.a_matrix_.start_ = rows.start;
  lp.a_matrix_.index_ = rows.index;
  lp.a_matrix_.value_ = rows.value;
  lp.sense_ = ObjSense::kMinimize;
  for (bool b : integer) lp.integrality_.push_back(b ? HighsVarType::kInteger : HighsVarType::kContinuous);
  Highs h;
  h.setOptionValue("output_flag", verbose);
  h.setOptionValue("mip_rel_gap", 0.0);
  if (time_limit > 0) h.setOptionValue("time_limit", time_limit);
  if (h.passModel(lp) != HighsStatus::kOk) throw std::runtime_error("HiGHS: invalid model");
  h.run();
  if (h.getInfo().primal_solution_status != kSolutionStatusFeasible) return {};
  return h.getSolution().col_value;
}

#ifdef COMMERGE_WITH_GUROBI
struct GurobiModel {
  GRBenv* env = nullptr;
  GRBmodel* model = nullptr;
  explicit GurobiModel(bool verbose) {
    if (GRBemptyenv(&env)) throw std::runtime_error("Gurobi: no environment");
    GRBsetintparam(env, "OutputFlag", verbose ? 1 : 0);
    if (GRBstartenv(env)) {
      std::string msg = GRBgeterrormsg(env);
      GRBfreeenv(env);
      env = nullptr;
      throw std::runtime_error("Gurobi: " + msg);
    }
    if (GRBnewmodel(env, &model, "commerge", 0, nullptr, nullptr, nullptr, nullptr, nullptr))
      throw std::runtime_error("Gurobi: no model");
  }
  ~GurobiModel() {
    if (model) GRBfreemodel(model);
    if (env) GRBfreeenv(env);
  }
  void check(int err) const {
    if (err) throw std::runtime_error(std::string("Gurobi: ") + GRBgeterrormsg(env));
  }
  std::vector<double> run(int n, double time_limit) {
    GRBenv* menv = GRBgetenv(model);
    check(GRBsetdblparam(menv, "MIPGap", 0.0));
    if (time_limit > 0) check(GRBsetdblparam(menv, "TimeLimit", time_limit));
    check(GRBoptimize(model));
    int count = 0;
    check(GRBgetintattr(model, "SolCount", &count));
    if (count == 0) return {};
    std::vector<double> x(n);
    check(GRBgetdblattrarray(model, "X", 0, n, x.data()));
    return x;
  }
};
#endif

}  // namespace

bool gurobi_available() {
#ifdef COMMERGE_WITH_GUROBI
  static int cached = -1;
  if (cached < 0) {
    try {
      GurobiModel m(false);
      cached = 1;
    } catch (const std::exception&) {
      cached = 0;
    }
  }
  return cached == 1;
#else
  return false;
#endif
}

std::string resolve_solver(const std::string& solver) {
  if (solver == "auto") return gurobi_available() ? "gurobi" : "highs";
  if (solver == "highs") return solver;
  if (solver == "gurobi") {
#ifndef COMMERGE_WITH_GUROBI
    throw std::runtime_error("this build has no Gurobi (configure with Gurobi installed, or use --solver highs)");
#endif
    return solver;
  }
  throw std::invalid_argument("solver must be auto, gurobi or highs");
}

std::vector<int> min_vertex_cover(int na, int nb, const std::vector<std::pair<int, int>>& edges,
                                  const std::vector<double>& w, const std::string& solver, double time_limit,
                                  bool verbose) {
  const int n = na + nb;
  if (n == 0) return {};
  const double half = std::min(na, nb) / 2.0;
  std::vector<double> x;
  if (resolve_solver(solver) == "gurobi") {
#ifdef COMMERGE_WITH_GUROBI
    GurobiModel m(verbose);
    std::vector<char> vtype(n, GRB_BINARY);
    m.check(GRBaddvars(m.model, n, 0, nullptr, nullptr, nullptr, const_cast<double*>(w.data()), nullptr, nullptr,
                       vtype.data(), nullptr));
    for (auto& [u, v] : edges) {
      int ind[2] = {u, v};
      double val[2] = {1.0, 1.0};
      m.check(GRBaddconstr(m.model, 2, ind, val, GRB_GREATER_EQUAL, 1.0, nullptr));
    }
    for (int side = 0; side < 2; ++side) {
      std::vector<int> ind;
      for (int u = side ? na : 0; u < (side ? n : na); ++u) ind.push_back(u);
      std::vector<double> val(ind.size(), 1.0);
      m.check(GRBaddconstr(m.model, static_cast<int>(ind.size()), ind.data(), val.data(), GRB_GREATER_EQUAL, half,
                           nullptr));
    }
    x = m.run(n, time_limit);
#endif
  } else {
    Rows rows;
    for (auto& [u, v] : edges) rows.add({{u, 1.0}, {v, 1.0}}, 1.0, kHighsInf);
    std::vector<std::pair<int, double>> ra, rb;
    for (int u = 0; u < na; ++u) ra.emplace_back(u, 1.0);
    for (int u = na; u < n; ++u) rb.emplace_back(u, 1.0);
    rows.add(ra, half, kHighsInf);
    rows.add(rb, half, kHighsInf);
    x = solve_highs(w, std::vector<bool>(n, true), rows, time_limit, verbose);
    if (x.empty()) throw std::runtime_error("vertex cover: HiGHS found no solution");
  }
  std::vector<int> cover;
  for (int u = 0; u < n && u < static_cast<int>(x.size()); ++u)
    if (x[u] > 0.5) cover.push_back(u);
  return cover;
}

std::vector<int> max_edge_weight_clique(const Matrix& M, const std::string& solver, double time_limit, bool verbose) {
  const int n = static_cast<int>(M.rows());
  if (n <= 1) {
    std::vector<int> all;
    for (int i = 0; i < n; ++i) all.push_back(i);
    return all;
  }
  std::vector<std::pair<int, int>> E, Z;  // weighted pairs, inconsistent pairs
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) (M(i, j) > 0 ? E : Z).emplace_back(i, j);
  std::vector<double> x;
  if (resolve_solver(solver) == "gurobi") {
#ifdef COMMERGE_WITH_GUROBI
    GurobiModel m(verbose);
    std::vector<char> vtype(n, GRB_BINARY);
    m.check(GRBaddvars(m.model, n, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, vtype.data(), nullptr));
    std::vector<int> qr, qc;
    std::vector<double> qv;
    for (auto& [i, j] : E) qr.push_back(i), qc.push_back(j), qv.push_back(M(i, j));
    if (!qr.empty())
      m.check(GRBaddqpterms(m.model, static_cast<int>(qr.size()), qr.data(), qc.data(), qv.data()));
    m.check(GRBsetintattr(m.model, "ModelSense", GRB_MAXIMIZE));
    for (auto& [i, j] : Z) {
      int ind[2] = {i, j};
      double val[2] = {1.0, 1.0};
      m.check(GRBaddconstr(m.model, 2, ind, val, GRB_LESS_EQUAL, 1.0, nullptr));
    }
    x = m.run(n, time_limit);
#endif
  } else {
    // Linearized: y_ij <= x_i, y_ij <= x_j per weighted pair; x_i + x_j <= 1 per inconsistent pair.
    const int m = static_cast<int>(E.size());
    std::vector<double> cost(n + m, 0.0);
    std::vector<bool> integer(n + m, false);
    for (int i = 0; i < n; ++i) integer[i] = true;
    Rows rows;
    for (int k = 0; k < m; ++k) {
      cost[n + k] = -M(E[k].first, E[k].second);
      rows.add({{E[k].first, -1.0}, {n + k, 1.0}}, -kHighsInf, 0.0);
      rows.add({{E[k].second, -1.0}, {n + k, 1.0}}, -kHighsInf, 0.0);
    }
    for (auto& [i, j] : Z) rows.add({{i, 1.0}, {j, 1.0}}, -kHighsInf, 1.0);
    x = solve_highs(cost, integer, rows, time_limit, verbose);
  }
  std::vector<int> chosen;
  for (int i = 0; i < n && i < static_cast<int>(x.size()); ++i)
    if (x[i] > 0.5) chosen.push_back(i);
  return chosen;
}

}  // namespace commerge
