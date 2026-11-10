"""Exchange policy and submap merging: one region's selected scans are exchanged, the two submaps
are registered, and the transform merges the whole maps.

Registration: KISS-Matcher (global, no initial guess; thirdparty/KISS-Matcher, pip kiss-matcher)
then GICP (small_gicp) to refine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .selection import Region, Selection, consistency_scores

POLICIES = ("score", "area", "candidates", "all")


def region_score(r: Region, affinity: np.ndarray, pos_a: np.ndarray, pos_b: np.ndarray,
                 consistency: float = 3.0) -> float:
    """Total weight of the region's step-(C) clique: consistent, similar pairs."""
    if len(r.pairs) < 2:
        return float(len(r.pairs))
    p = np.array(r.pairs)
    return float(consistency_scores(p, affinity[p[:, 0], p[:, 1]], pos_a, pos_b, consistency).sum())


def choose_regions(sel: Selection, affinity: np.ndarray, pos_a: np.ndarray, pos_b: np.ndarray,
                   policy: str = "score", consistency: float = 3.0) -> List[int]:
    """Indices of the region(s) whose scans are exchanged.

    score: largest step-(C) clique weight (default); area: largest bounding box;
    candidates: most step-(A) candidate pairs; all: every region."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    live = [k for k, r in enumerate(sel.regions) if r.pairs or r.scans_a is not None]
    if policy == "all" or not live:
        return live
    key = {
        "score": lambda k: region_score(sel.regions[k], affinity, pos_a, pos_b, consistency),
        "area": lambda k: (sel.regions[k].box[1] - sel.regions[k].box[0]) * (sel.regions[k].box[3] - sel.regions[k].box[2]),
        "candidates": lambda k: sel.regions[k].candidates,
    }[policy]
    return [max(live, key=key)]


@dataclass
class Registration:
    T_a_b: np.ndarray          # B's frame in A's frame
    inliers: int               # KISS-Matcher final inliers (grid: sampled points within 1 m)
    fitness: float             # GICP mean squared error (m^2) of the inliers (grid: share within 0.5 m)
    seconds: float


def register(submap_a: np.ndarray, submap_b: np.ndarray, voxel: float = 0.5,
             max_corr: float = 2.0) -> Optional[Registration]:
    """T_a_b aligning robot B's submap (B's frame) to robot A's submap (A's frame); None if the
    global registration fails."""
    import time

    try:
        import kiss_matcher as km
        import small_gicp
    except ImportError as e:  # kiss-matcher has wheels for Python 3.8-3.12 only
        raise ImportError("registration needs kiss-matcher and small-gicp (pip install kiss-matcher small-gicp; "
                          "kiss-matcher wheels exist for Python 3.8-3.12)") from e
    t = time.perf_counter()
    src = np.ascontiguousarray(submap_b[:, :3], dtype=np.float32)
    tgt = np.ascontiguousarray(submap_a[:, :3], dtype=np.float32)
    matcher = km.KISSMatcher(km.KISSMatcherConfig(voxel))
    sol = matcher.estimate(src, tgt)
    inliers = int(matcher.get_num_final_inliers())
    if not sol.valid or inliers < 5:
        return None
    T0 = np.eye(4)
    T0[:3, :3] = np.asarray(sol.rotation)
    T0[:3, 3] = np.asarray(sol.translation).ravel()
    res = small_gicp.align(tgt.astype(np.float64), src.astype(np.float64), T0, registration_type="GICP",
                           downsampling_resolution=voxel, max_correspondence_distance=max_corr, num_threads=4)
    T = np.asarray(res.T_target_source) if res.converged else T0
    fitness = float(res.error / max(res.num_inliers, 1)) if res.converged else float("nan")
    return Registration(T, inliers, fitness, time.perf_counter() - t)


def _yaw_frame(T: np.ndarray) -> np.ndarray:
    """T with its roll and pitch dropped: yaw and translation only (gravity stays z)."""
    yaw = np.arctan2(T[1, 0], T[0, 0])
    F = np.eye(4)
    F[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    F[:3, 3] = T[:3, 3]
    return F


def _drop_ground(p: np.ndarray, cell: float = 4.0, height: float = 0.4) -> np.ndarray:
    """Points more than ``height`` above the lowest point of their ``cell`` x ``cell`` column: the ground
    matches under any shift, so it would dominate the search score."""
    key = np.floor(p[:, :2] / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    low = np.full(inv.max() + 1, np.inf)
    np.minimum.at(low, inv, p[:, 2])
    return p[p[:, 2] > low[inv] + height]


def register_grid(submap_a: np.ndarray, submap_b: np.ndarray, pivot_a: np.ndarray, pivot_b: np.ndarray,
                  search: float = 40.0, step: float = 2.0, yaw_step: float = 5.0, max_range: float = 60.0,
                  samples: int = 256, seed: int = 0, hypotheses: int = 12) -> Optional[Registration]:
    """Registration for heterogeneous LiDARs (different fields of view and densities), where feature
    matching finds too few correspondences: both submaps are moved into the frame of one selected pair of
    keyframes (yaw and position only, gravity up), the yaw and the x-y shift between the two keyframes are
    searched exhaustively (+-``search`` m every ``step`` m, refined by ICP, every ``yaw_step`` degrees; score: sampled B
    points within 1 m of A, ground removed), and point-to-point ICP refines the best hypotheses with shrinking gates.
    ``pivot_a`` / ``pivot_b``: poses of the pair's keyframes in A's / B's frame. ``inliers`` is the number
    of sampled points within 1 m; ``fitness`` the share of B points within 0.5 m of A after ICP."""
    import time

    from scipy.spatial import cKDTree

    from .submap import voxel_downsample

    t = time.perf_counter()
    Fa, Fb = _yaw_frame(pivot_a), _yaw_frame(pivot_b)

    def local(sub, F):
        Fi = np.linalg.inv(F)
        p = sub[:, :3] @ Fi[:3, :3].T + Fi[:3, 3]
        return p[np.linalg.norm(p[:, :2], axis=1) < max_range]

    qa = _drop_ground(voxel_downsample(local(submap_a, Fa), 0.5)[:, :3])
    pb = _drop_ground(voxel_downsample(local(submap_b, Fb), 1.0)[:, :3])
    if len(qa) < 100 or len(pb) < 100:
        return None
    tree = cKDTree(qa)
    rng = np.random.default_rng(seed)
    P = pb[rng.choice(len(pb), min(samples, len(pb)), replace=False)]
    g = np.arange(-search, search + 1e-9, step)
    shifts = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    hyps = []
    for yaw in np.deg2rad(np.arange(0.0, 360.0, yaw_step)):
        c, s = np.cos(yaw), np.sin(yaw)
        Pr = P @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        X = Pr[None, :, :].repeat(len(shifts), 0)
        X[:, :, :2] += shifts[:, None, :]
        d, _ = tree.query(X.reshape(-1, 3), distance_upper_bound=1.0)
        score = (d < 1.0).reshape(len(shifts), -1).sum(1)
        for k in np.argsort(-score)[:3]:  # a few shifts per yaw
            hyps.append((int(score[k]), yaw, shifts[k]))
    hyps.sort(key=lambda h: -h[0])
    best = None
    for score, yaw, sh in hyps[:hypotheses]:
        T = np.eye(4)
        T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        T[:2, 3] = sh
        for gate in (3.0, 2.0, 1.0, 0.5):  # point-to-point ICP, shrinking gates
            for _ in range(15):
                X = pb @ T[:3, :3].T + T[:3, 3]
                d, i = tree.query(X, distance_upper_bound=gate)
                m = d < gate
                if m.sum() < 10:
                    break
                T = _rigid(X[m], qa[i[m]]) @ T
        d, _ = tree.query(pb @ T[:3, :3].T + T[:3, 3])
        fit = float((d < 0.5).mean())
        if best is None or fit > best[0]:
            best = (fit, score, T)
    if best is None:
        return None
    fit, score, T = best
    return Registration(Fa @ T @ np.linalg.inv(Fb), score, fit, time.perf_counter() - t)


def register_exchanged(submap_a: np.ndarray, submap_b: np.ndarray, poses_a: np.ndarray, poses_b: np.ndarray,
                       pairs, affinity: np.ndarray, method: str = "kiss", voxel: float = 0.5,
                       pivots: int = 5) -> Optional[Registration]:
    """Register the exchanged submaps: ``kiss`` (KISS-Matcher + GICP) or ``grid`` (``register_grid`` around
    up to ``pivots`` exchanged pairs, the highest affinity first; the best fit wins)."""
    if method == "kiss":
        return register(submap_a, submap_b, voxel)
    if method != "grid":
        raise ValueError("method must be kiss or grid")
    pairs = sorted(pairs, key=lambda p: -affinity[p[0], p[1]])[:pivots]
    regs = [register_grid(submap_a, submap_b, poses_a[a], poses_b[b]) for a, b in pairs]
    regs = [r for r in regs if r is not None]
    return max(regs, key=lambda r: r.fitness) if regs else None


def _rigid(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best rigid transform (Umeyama, no scale) mapping points src (N, 3) onto dst (N, 3)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    T = np.eye(4)
    T[:3, :3] = U @ D @ Vt
    T[:3, 3] = md - T[:3, :3] @ ms
    return T


def alignment_error(T_a_b: np.ndarray, poses_a: np.ndarray, poses_b: np.ndarray,
                    gt_a: np.ndarray, gt_b: np.ndarray) -> tuple:
    """Translation (m) and rotation (deg) error of T_a_b against ground truth, and the mean
    position error of B's keyframes once merged into A's frame. ``gt_*`` are (N, 4, 4) keyframe
    poses in a common frame; each robot's frame is located in it from the positions alone."""
    G_a = _rigid(poses_a[:, :3, 3], gt_a[:, :3, 3])   # A's frame in the common frame
    G_b = _rigid(poses_b[:, :3, 3], gt_b[:, :3, 3])
    T_true = np.linalg.inv(G_a) @ G_b
    E = np.linalg.inv(T_true) @ T_a_b
    rot = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
    merged = np.einsum("ij,njk->nik", G_a @ T_a_b, poses_b)[:, :3, 3]
    return float(np.linalg.norm(E[:3, 3])), float(rot), float(np.linalg.norm(merged - gt_b[:, :3, 3], axis=1).mean())
