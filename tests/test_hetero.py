"""--enhance and --registration grid (the experimental options for heterogeneous LiDARs)."""
import numpy as np

from commerge import enhance_affinity, register_grid


def test_enhance_finds_an_anti_diagonal_overlap():
    rng = np.random.default_rng(0)
    S = rng.normal(0.5, 0.05, (200, 160))
    S += rng.uniform(0, 0.3, (200, 1)) + rng.uniform(0, 0.3, (1, 160))   # keyframes scoring high everywhere
    for k in range(30):                       # overlap driven in opposite directions
        S[100 + k, 130 - k] += 0.08
    i, j = np.unravel_index(S.argmax(), S.shape)
    assert not (100 <= i < 130 and 100 < j <= 130)   # raw: a biased keyframe wins
    E = enhance_affinity(S, 10)
    i, j = np.unravel_index(E.argmax(), E.shape)
    assert 100 <= i < 130 and 100 < j <= 130 and abs((i - 100) - (130 - j)) <= 2


def _scene(rng):
    walls = []
    for x0, y0, dx, dy in [(-30, -20, 60, 0), (-30, 25, 45, 0), (20, -20, 0, 30), (-25, -20, 0, 40), (5, 5, 6, 0)]:
        t = rng.uniform(0, 1, 4000)
        z = rng.uniform(0.5, 6, 4000)
        walls.append(np.c_[x0 + t * dx + rng.normal(0, 0.03, 4000), y0 + t * dy + rng.normal(0, 0.03, 4000), z])
    ground = np.c_[rng.uniform(-35, 35, 6000), rng.uniform(-25, 30, 6000), rng.normal(0, 0.03, 6000)]
    poles = np.c_[np.repeat(rng.uniform(-25, 15, (12, 2)), 200, 0) + rng.normal(0, 0.1, (2400, 2)), rng.uniform(0, 4, 2400)]
    return np.vstack(walls + [ground, poles])


def test_grid_recovers_yaw_and_shift():
    rng = np.random.default_rng(1)
    world = _scene(rng)
    yaw, t = np.deg2rad(150.0), np.array([7.0, -9.0, 0.2])     # robot B's frame in robot A's frame
    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:3, 3] = t
    sub_a = np.c_[world, np.zeros(len(world))]
    keep = world[:, 0] < 10                                   # B sees only part of the scene
    pb = (world[keep] - t) @ T[:3, :3]                        # inverse transform: in B's frame
    sub_b = np.c_[pb, np.zeros(len(pb))]
    reg = register_grid(sub_a, sub_b, np.eye(4), np.eye(4))
    assert reg is not None
    E = np.linalg.inv(T) @ reg.T_a_b
    assert np.linalg.norm(E[:3, 3]) < 0.3
    assert np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))) < 1.0
