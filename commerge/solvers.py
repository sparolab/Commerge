"""The two optimization problems of Commerge, each with two back-ends:

* ``gurobi`` - the formulation of the paper (needs ``gurobipy`` and a Gurobi license);
* ``highs``  - the same problems as mixed-integer linear programs for SciPy's HiGHS (no license).

``auto`` uses Gurobi when it is installed and licensed, HiGHS otherwise.
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Sequence

import numpy as np

SOLVERS = ("auto", "gurobi", "highs")


def _env(verbose: bool = False):
    """A Gurobi environment that prints nothing (not even the license banner) unless verbose."""
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", int(verbose))
    env.start()
    return env


def _gurobi_available() -> bool:
    try:
        import gurobipy as gp

        with _env() as env, gp.Model(env=env) as m:
            m.addVar()
            m.optimize()
        return True
    except Exception:
        return False


def resolve(solver: str) -> str:
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if solver == "auto":
        return "gurobi" if _gurobi_available() else "highs"
    return solver


# ---------------------------------------------------------------------------------------------
# Minimum vertex cover of the bipartite correspondence graph
# ---------------------------------------------------------------------------------------------
def min_vertex_cover(nodes: Sequence[Hashable], edges: Sequence[tuple], a_nodes: Sequence[Hashable],
                     b_nodes: Sequence[Hashable], weights: Dict[Hashable, float] | None = None,
                     solver: str = "auto", time_limit: float = 0.0, verbose: bool = False) -> set:
    """Smallest (weighted) set of keyframes that touches every candidate pair.

    Both robots keep at least half of the smaller side, so the cover cannot collapse onto one
    robot: sum(x_A) >= min(|A|, |B|) / 2 and sum(x_B) >= min(|A|, |B|) / 2.
    """
    nodes = list(nodes)
    if not nodes:
        return set()
    idx = {v: k for k, v in enumerate(nodes)}
    w = np.array([(weights or {}).get(v, 1.0) for v in nodes], dtype=float)
    half = min(len(a_nodes), len(b_nodes)) / 2.0
    solver = resolve(solver)

    if solver == "gurobi":
        import gurobipy as gp
        from gurobipy import GRB

        with _env(verbose) as env, gp.Model("MVC", env=env) as m:
            m.Params.MIPGap = 0.0  # exact optimum: the tie-break is smaller than the default gap
            if time_limit > 0:
                m.Params.TimeLimit = time_limit
            x = m.addVars(len(nodes), vtype=GRB.BINARY)
            m.setObjective(gp.quicksum(w[k] * x[k] for k in range(len(nodes))), GRB.MINIMIZE)
            for u, v in edges:
                m.addConstr(x[idx[u]] + x[idx[v]] >= 1)
            m.addConstr(gp.quicksum(x[idx[v]] for v in a_nodes) >= half)
            m.addConstr(gp.quicksum(x[idx[v]] for v in b_nodes) >= half)
            m.optimize()
            return {nodes[k] for k in range(len(nodes)) if x[k].X > 0.5}

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    rows, cols = [], []
    for r, (u, v) in enumerate(edges):  # x_u + x_v >= 1
        rows += [r, r]
        cols += [idx[u], idx[v]]
    ne = len(edges)
    for r, group in ((ne, a_nodes), (ne + 1, b_nodes)):  # balance between the robots
        rows += [r] * len(group)
        cols += [idx[v] for v in group]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(ne + 2, len(nodes))).tocsr()
    lb = np.r_[np.ones(ne), half, half]
    opts: dict = {"disp": verbose, "mip_rel_gap": 0.0}
    if time_limit > 0:
        opts["time_limit"] = time_limit
    res = milp(w, constraints=LinearConstraint(A, lb, np.inf), integrality=np.ones(len(nodes)),
               bounds=Bounds(0, 1), options=opts)
    if res.x is None:
        raise RuntimeError(f"vertex cover: HiGHS found no solution ({res.message})")
    return {nodes[k] for k in range(len(nodes)) if res.x[k] > 0.5}


# ---------------------------------------------------------------------------------------------
# Maximum edge-weight clique of the pairwise-consistency graph
# ---------------------------------------------------------------------------------------------
def max_edge_weight_clique(M: np.ndarray, solver: str = "auto", time_limit: float = 0.0,
                           verbose: bool = False) -> List[int]:
    """Largest-weight set of mutually consistent candidate pairs.

    ``M`` is the pair-score matrix (only its strict upper triangle is used; 0 = inconsistent).
    maximize sum_{i<j} M_ij x_i x_j  s.t.  x_i + x_j <= 1 wherever M_ij = 0.
    """
    n = len(M)
    if n <= 1:
        return list(range(n))
    U = np.triu(np.asarray(M, dtype=float), 1)
    solver = resolve(solver)

    if solver == "gurobi":
        import gurobipy as gp
        from gurobipy import GRB

        E = np.argwhere(U > 0)
        Z = np.argwhere(np.triu(U == 0, 1))
        with _env(verbose) as env, gp.Model("MEWC", env=env) as m:
            m.Params.MIPGap = 0.0
            if time_limit > 0:
                m.Params.TimeLimit = time_limit
            x = m.addVars(n, vtype=GRB.BINARY)
            m.setObjective(gp.quicksum(U[i, j] * x[i] * x[j] for i, j in E), GRB.MAXIMIZE)
            for i, j in Z:
                m.addConstr(x[i] + x[j] <= 1)
            m.optimize()
            if m.SolCount == 0:
                return []
            return [i for i in range(n) if x[i].X > 0.5]

    # Linearized for HiGHS: y_ij <= x_i, y_ij <= x_j for each weighted pair (y_ij = x_i x_j at
    # the optimum since M_ij > 0), x_i + x_j <= 1 for each inconsistent pair.
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    E = np.argwhere(U > 0)
    Z = np.argwhere(np.triu(U == 0, 1))
    m = len(E)
    c = np.r_[np.zeros(n), -U[E[:, 0], E[:, 1]]]  # variables: x (n), then y (m); minimize -score
    k = np.arange(m)
    rows = np.r_[2 * k, 2 * k, 2 * k + 1, 2 * k + 1, 2 * m + np.repeat(np.arange(len(Z)), 2)]
    cols = np.r_[n + k, E[:, 0], n + k, E[:, 1], Z.ravel()]
    vals = np.r_[np.ones(m), -np.ones(m), np.ones(m), -np.ones(m), np.ones(2 * len(Z))]
    A = coo_matrix((vals, (rows, cols)), shape=(2 * m + len(Z), n + m)).tocsr()
    ub = np.r_[np.zeros(2 * m), np.ones(len(Z))]
    opts: dict = {"disp": verbose, "mip_rel_gap": 0.0}
    if time_limit > 0:
        opts["time_limit"] = time_limit
    res = milp(c, constraints=LinearConstraint(A, -np.inf, ub),
               integrality=np.r_[np.ones(n), np.zeros(m)], bounds=Bounds(0, 1), options=opts)
    if res.x is None:
        return []
    return [i for i in range(n) if res.x[i] > 0.5]
