"""The C++ core (commerge_cpp) must give the same selection as the Python package. Skipped when not installed."""
import numpy as np
import pytest

from commerge import select
from commerge.selection import cosine_affinity, enhance_affinity, extract_regions

cc = pytest.importorskip("commerge_cpp")
from test_selection import scene  # noqa: E402


def test_affinity_and_regions():
    rng = np.random.default_rng(1)
    da, db = rng.random((60, 30)), rng.random((80, 30))
    da[3] = 0  # empty descriptor
    np.testing.assert_allclose(cc.cosine_affinity(da, db), cosine_affinity(da, db), atol=1e-12)
    S = rng.random((50, 70))
    for L in (1, 4, 5):
        np.testing.assert_allclose(cc.enhance_affinity(S, L), enhance_affinity(S, L), atol=1e-9)
    aff, _, _ = scene()
    assert cc.extract_regions(aff, 0.95, 60) == extract_regions(aff, 0.95, 60)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_selection(seed):
    aff, pa, pb = scene(seed)
    py = select(aff, pa, pb, threshold=0.95, min_area=60, solver="highs", log=lambda *a: None)
    cp = cc.select(aff, pa, pb, threshold=0.95, min_area=60, solver="highs")
    assert [tuple(r.box) for r in py.regions] == [tuple(r.box) for r in cp.regions]
    for r, q in zip(py.regions, cp.regions):
        assert r.candidates == q.candidates
        assert (r.cover_a, r.cover_b) == (list(q.cover_a), list(q.cover_b))
        assert r.pairs == [tuple(p) for p in q.pairs]


def test_same_pairs_registration():
    from test_pairs import move, trajectory
    from commerge.merge import register_pairs

    rng = np.random.default_rng(2)
    A = trajectory()
    B = np.linalg.inv(move(60.0, [30.0, -20.0, 0.0])) @ A
    B[:, :2, 3] += rng.normal(0, 1.0, (len(B), 2))
    pairs = [(i, i) for i in range(0, 120, 5)] + [(3, 90)]
    py = register_pairs(A, B, pairs)
    cp = cc.register_pairs(A.reshape(len(A), 16), B.reshape(len(B), 16), pairs)
    np.testing.assert_allclose(cp.T_a_b, py.T_a_b, atol=1e-9)
    assert cp.inliers == py.inliers
