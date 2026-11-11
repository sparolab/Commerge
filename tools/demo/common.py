"""Shared helpers of the sensor demos: two robots from ground truth, one of them moved into its own frame.

Each demo computes one descriptor per keyframe with its sensor's place-recognition method, turns the descriptor
distances into an (Na, Nb) affinity, and writes two sessions Commerge reads:

    <out>/robot_a/odom_poses.txt   keyframe poses of A (ground truth)
    <out>/robot_b/odom_poses.txt   keyframe poses of B, moved by a rigid transform (default: yaw 60 deg, 30 m, -20 m)
    <out>/robot_*/gt_poses.txt     keyframe poses in the common frame (for the check only)
    <out>/robot_*/times.txt, frames.txt   keyframe times and source files
    <out>/affinity.npy             (Na, Nb) affinity, higher = more similar
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate_poses(times: np.ndarray, gt_t: np.ndarray, gt_T: np.ndarray) -> np.ndarray:
    """(N, 4, 4) poses at ``times`` from a ground-truth trajectory (gt_t sorted, gt_T (M, 4, 4))."""
    tc = np.clip(times, gt_t[0], gt_t[-1])
    T = np.tile(np.eye(4), (len(times), 1, 1))
    for k in range(3):
        T[:, k, 3] = np.interp(tc, gt_t, gt_T[:, k, 3])
    T[:, :3, :3] = Slerp(gt_t, Rotation.from_matrix(gt_T[:, :3, :3]))(tc).as_matrix()
    return T


def keyframes_by_distance(T: np.ndarray, step: float) -> np.ndarray:
    """Indices of keyframes at least ``step`` m apart along the trajectory."""
    keep = [0]
    for i in range(1, len(T)):
        if np.linalg.norm(T[i, :3, 3] - T[keep[-1], :3, 3]) >= step:
            keep.append(i)
    return np.array(keep)


def rigid(yaw_deg: float, shift) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    M[:3, 3] = shift
    return M


def write_session(root: Path, gt: np.ndarray, times=None, frames=None, move: np.ndarray | None = None) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    flat = lambda P: P[:, :3, :4].reshape(len(P), 12)
    np.savetxt(root / "gt_poses.txt", flat(gt), fmt="%.6f")
    np.savetxt(root / "odom_poses.txt", flat(gt if move is None else move @ gt), fmt="%.6f")
    if times is not None:
        np.savetxt(root / "times.txt", np.asarray(times, dtype=float), fmt="%.6f")
    if frames is not None:
        (root / "frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    return root


def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    """Descriptor distances (lower = closer) to an affinity in [0, 1] (higher = more similar): 1 - D / max(D)."""
    D = np.asarray(D, dtype=float)
    return 1.0 - D / max(float(D.max()), 1e-12)


def report_affinity(S: np.ndarray, gt_a: np.ndarray, gt_b: np.ndarray, near: float = 5.0) -> None:
    """How well the affinity separates true revisits (within ``near`` m) from the rest, per threshold."""
    d = np.linalg.norm(gt_a[:, None, :2, 3] - gt_b[None, :, :2, 3], axis=2)
    print(f"affinity {S.shape}: revisits (< {near} m) {S[d < near].mean():.3f}, others {S[d >= near].mean():.3f}")
    for q in (99.0, 99.5, 99.9):
        thr = np.percentile(S, q)
        m = S >= thr
        print(f"  top {100 - q:.1f} % (>= {thr:.4f}): {m.sum()} pairs, {(d[m] < near).mean():.0%} within {near} m")
