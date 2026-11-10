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
    inliers: int               # KISS-Matcher final inliers
    fitness: float             # GICP mean squared error (m^2) of the inliers
    seconds: float


def register(submap_a: np.ndarray, submap_b: np.ndarray, voxel: float = 0.5,
             max_corr: float = 2.0) -> Optional[Registration]:
    """T_a_b aligning robot B's submap (B's frame) to robot A's submap (A's frame); None if the
    global registration fails."""
    import time

    try:
        import kiss_matcher as km
        import small_gicp
    except ImportError as e:  # pragma: no cover
        raise SystemExit("registration needs kiss-matcher and small-gicp: pip install kiss-matcher small-gicp") from e
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
