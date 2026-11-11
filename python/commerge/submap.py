"""Submaps from the selected keyframes: what each robot sends for map merging."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .session import Session, write_pcd


def voxel_downsample(xyzi: np.ndarray, leaf: float) -> np.ndarray:
    """One point (the first) per ``leaf``-sized voxel; ``leaf <= 0`` keeps everything."""
    if leaf <= 0 or len(xyzi) == 0:
        return xyzi
    keys = np.floor(xyzi[:, :3] / leaf).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return xyzi[np.sort(first)]


def build_submap(session: Session, keyframes: Iterable[int], min_range: float = 3.0,
                 voxel: float = 0.0) -> np.ndarray:
    """The selected keyframes in the session frame, (N, 4) x y z intensity.

    ``min_range`` drops points near the sensor (the robot itself), as in the paper (3 m)."""
    clouds = [session.cloud_in_world(k, min_range) for k in keyframes]
    cloud = np.concatenate(clouds) if clouds else np.zeros((0, 4), np.float32)
    return voxel_downsample(cloud, voxel)


def save_submap(path: Path, cloud: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pcd(path, cloud)
    return path
