"""AutoMerge-style region selection (our re-implementation, for comparison).

AutoMerge (P. Yin et al., "AutoMerge: A Framework for Map Assembling and Smoothing in City-Scale
Environments", IEEE T-RO 2023) finds overlapping sequences with SeqSLAM. This module follows that
idea on the same affinity matrix Commerge uses:

1. local contrast enhancement of the difference matrix D = 1 - affinity (SeqSLAM);
2. sequence matching along constant-velocity trajectories, with a ratio test;
3. K-means on the good matches, a line RANSAC per cluster;
4. each cluster's padded bounding box is a region; the best region is exchanged whole.

It is not the authors' code. All scans of the chosen region are sent, which is what Commerge's
selection is compared against.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..selection import Region, Selection


@dataclass
class AutoMergeParams:
    min_area: int = 20000
    contrast_r: int = 50          # rows of the local contrast window
    ds: int = 50                  # sequence length
    vmin: float = 0.8             # trajectory slopes searched
    vmax: float = 1.2
    r_window: int = 10            # window excluded around the best match for the ratio test
    ratio: float = 0.8            # keep matches with best / second-best below this
    n_clusters: int = 5
    ransac_iters: int = 100
    padding: int = 20             # region padding (keyframes)
    best_only: bool = True        # exchange only the best region (as AutoMerge does)
    seed: int = 0


def enhance_contrast(D: np.ndarray, r: int) -> np.ndarray:
    """SeqSLAM local contrast enhancement along the rows (reference keyframes)."""
    DD = np.empty_like(D, dtype=float)
    n = D.shape[0]
    for i in range(n):
        a, b = max(0, i - r // 2), min(n, i + r // 2 + 1)
        v = D[a:b]
        std = v.std(axis=0, ddof=1) if b - a > 1 else np.ones(D.shape[1])
        std[std == 0] = 1.0
        DD[i] = (D[i] - v.mean(axis=0)) / std
    return DD - DD.min()


def sequence_matches(DD: np.ndarray, p: AutoMergeParams) -> np.ndarray:
    """(Nb, 2): for each query column, the best reference row and the ratio to the second best."""
    ds = p.ds + (p.ds % 2)
    rows, cols = DD.shape
    matches = np.full((cols, 2), np.nan)
    move = np.arange(int(p.vmin * ds), int(p.vmax * ds) + 1)
    v = move / ds
    steps = np.floor(np.arange(ds + 1)[None, :] * v[:, None]).astype(int)  # (V, ds+1)
    padded = np.vstack([DD, np.full((1, cols), np.inf)])                     # out-of-bounds row
    starts = np.arange(rows)
    for n in range(ds // 2, cols - ds // 2):
        x = np.arange(n - ds // 2, n + ds // 2 + 1)                          # ds+1 query columns
        y = np.minimum(starts[:, None, None] + steps[None], rows)            # (S, V, ds+1)
        scores = padded[y, x[None, None, :]].sum(axis=2).min(axis=1)         # (S,)
        if not np.isfinite(scores).any():
            continue
        k = int(np.argmin(scores))
        mask = np.ones(rows, bool)
        mask[max(0, k - p.r_window // 2): min(rows, k + p.r_window // 2)] = False
        second = scores[mask].min() if mask.any() else np.inf
        matches[n] = (k + ds // 2, scores[k] / second if second > 0 else 1.0)
    return matches


def _kmeans(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(points)
    except ImportError:
        from scipy.cluster.vq import kmeans2

        return kmeans2(points.astype(float), k, seed=seed, minit="++")[1]


def _ransac_line(pts: np.ndarray, iters: int, rng: np.random.Generator) -> np.ndarray:
    if len(pts) < 4:
        return np.arange(len(pts))
    best = np.arange(0)
    tol = max(10.0, 0.1 * len(pts))
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        if abs(pts[j, 0] - pts[i, 0]) < 1e-6:
            continue
        a = (pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0])
        b = pts[i, 1] - a * pts[i, 0]
        inl = np.nonzero(np.abs(pts[:, 1] - (a * pts[:, 0] + b)) < tol)[0]
        if len(inl) > len(best):
            best = inl
    return best if len(best) else np.arange(len(pts))


def automerge_regions(affinity: np.ndarray, p: AutoMergeParams = AutoMergeParams()
                      ) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Regions (a0, a1, b0, b1) with their mean match ratio, best first."""
    DD = enhance_contrast(np.clip(1.0 - affinity, 0, 1), p.contrast_r)
    m = sequence_matches(DD, p)
    good = [(int(m[q, 0]), q, m[q, 1]) for q in range(len(m)) if np.isfinite(m[q, 0]) and m[q, 1] < p.ratio]
    if not good:
        return []
    pts = np.array([(a, b) for a, b, _ in good], dtype=float)
    k = max(2, min(p.n_clusters, len(good) // 10))
    labels = _kmeans(pts, k, p.seed) if len(good) >= k else np.zeros(len(good), int)
    rng = np.random.default_rng(p.seed)
    regions = []
    for c in range(labels.max() + 1):
        idx = np.nonzero(labels == c)[0]
        if len(idx) <= 2:
            continue
        idx = idx[_ransac_line(pts[idx], p.ransac_iters, rng)]
        if len(idx) < 3:
            continue
        a0, a1 = pts[idx, 0].min(), pts[idx, 0].max()
        b0, b1 = pts[idx, 1].min(), pts[idx, 1].max()
        box = (int(max(0, a0 - p.padding)), int(min(affinity.shape[0], a1 + p.padding)),
               int(max(0, b0 - p.padding)), int(min(affinity.shape[1], b1 + p.padding)))
        if (box[1] - box[0]) * (box[3] - box[2]) >= p.min_area:
            regions.append((box, float(np.mean([good[i][2] for i in idx]))))

    def score(r):  # as in our comparison: similar sequences and large overlap first
        (a0, a1, b0, b1), ratio = r
        sim = float(affinity[a0:a1, b0:b1].mean())
        return 0.7 * sim + 0.3 * min((a1 - a0) * (b1 - b0) / 50000.0, 1.0)

    return sorted(regions, key=score, reverse=True)


def automerge_select(affinity: np.ndarray, p: AutoMergeParams = AutoMergeParams(), log=print) -> Selection:
    """The chosen region(s), every keyframe of which is exchanged."""
    t = time.perf_counter()
    regions = automerge_regions(affinity, p)
    if p.best_only:
        regions = regions[:1]
    out = []
    for (a0, a1, b0, b1), _ in regions:
        r = Region((a0, a1, b0, b1))
        r.scans_a, r.scans_b = list(range(a0, a1)), list(range(b0, b1))
        r.candidates = int((affinity[a0:a1, b0:b1] > 0).sum())
        r.seconds["automerge"] = time.perf_counter() - t
        out.append(r)
    log(f"[automerge] {len(out)} region(s), {sum(len(r.keyframes_a) + len(r.keyframes_b) for r in out)} scans "
        f"in {time.perf_counter() - t:.2f} s")
    return Selection(out, "", "-")
