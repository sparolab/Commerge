"""Synthetic check: robot B drives part of robot A's route again; Commerge must find that overlap,
send few scans and keep only consistent pairs. Runs with HiGHS (no license); with Gurobi when present."""
import numpy as np
import pytest

from commerge import select
from commerge.baselines.automerge import AutoMergeParams, automerge_select
from commerge.selection import consistency_scores
from commerge.solvers import _gurobi_available


def scene(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(400) * 1.0
    pos_a = np.c_[t, 5 * np.sin(t / 40), np.zeros_like(t)]               # robot A: 400 m route
    idx_b = np.r_[np.arange(120, 220), rng.uniform(0, 400, 150)]          # B re-drives 120-220, then elsewhere
    pos_b = np.c_[idx_b + 0.3 * rng.standard_normal(len(idx_b)), 5 * np.sin(idx_b / 40) + 60 * (np.arange(len(idx_b)) >= 100), np.zeros(len(idx_b))]
    d = np.linalg.norm(pos_a[:, None, :2] - pos_b[None, :, :2], axis=2)
    aff = 0.85 + 0.14 * np.exp(-d / 3.0) + 0.01 * rng.random(d.shape)     # similar where they are close
    aff[50:56, 200:206] = 0.995       # isolated perceptual aliasing: sequential matching must drop it
    return aff, pos_a, pos_b


@pytest.mark.parametrize("solver", ["highs"] + (["gurobi"] if _gurobi_available() else []))
def test_pipeline(solver):
    aff, pa, pb = scene()
    sel = select(aff, pa, pb, threshold=0.95, min_area=60, solver=solver, log=lambda *a: None)
    assert sel.regions, "the overlap must be found"
    assert not any(r.box[0] < 60 and r.box[2] >= 200 for r in sel.regions), "aliasing patch kept"
    sent = len(sel.keyframes_a) + len(sel.keyframes_b)
    assert 0 < sent < 0.2 * (len(pa) + len(pb))
    # every selected pair is a true revisit (B's first 100 keyframes re-drive A's 120-220)
    for a, b in sel.pairs:
        assert np.linalg.norm(pa[a, :2] - pb[b, :2]) < 3.0
    # and the selected pairs are mutually consistent
    p = np.array(sel.pairs)
    M = consistency_scores(p, aff[p[:, 0], p[:, 1]], pa, pb, 3.0)
    assert np.all(M[np.triu_indices(len(p), 1)] > 0)


def test_steps_shrink():
    aff, pa, pb = scene()
    n = {}
    for steps in ("A", "AB", "ABC"):
        sel = select(aff, pa, pb, 0.95, 60, steps=steps, solver="highs", log=lambda *a: None)
        n[steps] = len(sel.pairs)
    a, b, c = n.values()
    assert a >= b >= c > 0


@pytest.mark.skipif(not _gurobi_available(), reason="no Gurobi license")
def test_gurobi_equals_highs():
    aff, pa, pb = scene(1)
    g = select(aff, pa, pb, 0.95, 60, solver="gurobi", log=lambda *a: None)
    h = select(aff, pa, pb, 0.95, 60, solver="highs", log=lambda *a: None)
    assert sorted(g.pairs) == sorted(h.pairs)


def test_automerge_runs():
    aff, _, _ = scene()
    sel = automerge_select(aff, AutoMergeParams(min_area=60, ds=20, contrast_r=20), log=lambda *a: None)
    assert all(r.scans_a is not None for r in sel.regions)


def test_exchange_policy():
    from commerge import choose_regions
    from commerge.merge import region_score

    aff, pa, pb = scene()
    sel = select(aff, pa, pb, 0.95, 60, solver="highs", log=lambda *a: None)
    best = choose_regions(sel, aff, pa, pb, "score")
    assert len(best) == 1
    scores = [region_score(r, aff, pa, pb) for r in sel.regions]
    assert scores[best[0]] == max(scores)
    assert len(choose_regions(sel, aff, pa, pb, "all")) == sum(1 for r in sel.regions if r.pairs)
