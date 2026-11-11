"""--registration pairs: recover a robot's frame from the selected pairs' keyframe positions alone."""
import numpy as np
import pytest

from commerge.merge import register_pairs


def trajectory(n=120, seed=0):
    t = np.linspace(0, 2 * np.pi, n)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3], T[:, 1, 3] = 60 * np.cos(t), 40 * np.sin(2 * t)
    return T


def move(yaw_deg, shift):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    M = np.eye(4)
    M[:2, :2] = [[c, -s], [s, c]]
    M[:3, 3] = shift
    return M


@pytest.mark.parametrize("yaw", [60.0, -150.0])
def test_recovers_frame(yaw):
    rng = np.random.default_rng(1)
    A = trajectory()
    M = move(yaw, [30.0, -20.0, 1.0])
    B = np.linalg.inv(M) @ A                       # B's poses in its own frame; T_a_b = M
    B[:, :2, 3] += rng.normal(0, 1.0, (len(B), 2))  # place recognition is a few metres loose
    pairs = [(i, i) for i in range(0, 120, 7)] + [(3, 90), (50, 10)]  # two wrong pairs
    reg = register_pairs(A, B, pairs)
    E = np.linalg.inv(M) @ reg.T_a_b
    assert np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))) < 1.0
    assert np.linalg.norm(E[:3, 3]) < 1.5
    assert reg.inliers <= len(pairs) - 2


def test_degenerate():
    A = trajectory()
    assert register_pairs(A, A, [(0, 0), (1, 1)]) is None          # too few pairs
    assert register_pairs(A, A, [(0, 0), (0, 0), (0, 0)]) is None  # no spread: yaw unobservable
