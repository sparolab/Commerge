"""Commerge scan selection: which keyframes two robots should exchange to merge their maps.

1. ``regions`` - sequential matching: connected regions of the affinity matrix above ``threshold``
   whose bounding box covers at least ``min_area`` cells (places the robots visited in sequence).
2. ``cover``   - per region, the minimum vertex cover of the bipartite graph of candidate pairs
   (robot A keyframes x robot B keyframes): the fewest keyframes that still touch every pair.
3. ``clique``  - among the pairs between covered keyframes, the maximum edge-weight clique of the
   pairwise-consistency graph: pairs (i, j), (k, l) agree when the distance between A's
   keyframes i, k matches the distance between B's keyframes j, l.

Each stage can be switched off (ablation): without ``regions`` the whole matrix is one region,
without ``cover`` every candidate pair goes on, without ``clique`` every remaining pair is kept.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import find_objects, label

from . import solvers

STAGES = ("regions", "cover", "clique")


def cosine_affinity(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """(Na, Nb) cosine similarity between two sets of descriptors (0 where a descriptor is empty)."""
    a = np.nan_to_num(np.asarray(desc_a, dtype=float))
    b = np.nan_to_num(np.asarray(desc_b, dtype=float))
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    a = np.divide(a, na, out=np.zeros_like(a), where=na > 0)
    b = np.divide(b, nb, out=np.zeros_like(b), where=nb > 0)
    return a @ b.T


def extract_regions(affinity: np.ndarray, threshold: float, min_area: int) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes (a0, a1, b0, b1), half-open, of the connected regions above ``threshold``
    whose box area is at least ``min_area``; rows index robot A, columns robot B."""
    labeled = label(affinity > threshold)[0]
    boxes = []
    for sl in find_objects(labeled):
        if sl is None:
            continue
        a0, a1, b0, b1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if (a1 - a0) * (b1 - b0) >= min_area:
            boxes.append((a0, a1, b0, b1))
    return boxes


def consistency_scores(pairs: np.ndarray, weights: np.ndarray, pos_a: np.ndarray, pos_b: np.ndarray,
                       max_error: float) -> np.ndarray:
    """Pair-score matrix M (upper triangle): M_pq = w_p * exp(-e_pq) * w_q if e_pq <= max_error,
    with e_pq = | |a_p - a_q| - |b_p - b_q| | the distance mismatch between pairs p and q."""
    pa = pos_a[pairs[:, 0]]
    pb = pos_b[pairs[:, 1]]
    da = np.linalg.norm(pa[:, None, :] - pa[None, :, :], axis=2)
    db = np.linalg.norm(pb[:, None, :] - pb[None, :, :], axis=2)
    err = np.abs(da - db)
    C = np.where(err <= max_error, np.exp(-err), 0.0)
    M = weights[:, None] * C * weights[None, :]
    return np.triu(M, 1)


@dataclass
class Region:
    box: Tuple[int, int, int, int]                 # a0, a1, b0, b1 (half-open)
    candidates: int = 0                           # candidate pairs above the threshold
    cover_a: List[int] = field(default_factory=list)
    cover_b: List[int] = field(default_factory=list)
    pairs: List[Tuple[int, int]] = field(default_factory=list)  # selected (a, b) keyframe pairs
    seconds: dict = field(default_factory=dict)
    scans_a: List[int] | None = None              # explicit scans (baselines that send whole regions)
    scans_b: List[int] | None = None

    @property
    def keyframes_a(self) -> List[int]:
        return sorted(self.scans_a) if self.scans_a is not None else sorted({a for a, _ in self.pairs})

    @property
    def keyframes_b(self) -> List[int]:
        return sorted(self.scans_b) if self.scans_b is not None else sorted({b for _, b in self.pairs})


@dataclass
class Selection:
    regions: List[Region]
    stages: Tuple[str, ...]
    solver: str

    @property
    def keyframes_a(self) -> List[int]:
        return sorted({a for r in self.regions for a in r.keyframes_a})

    @property
    def keyframes_b(self) -> List[int]:
        return sorted({b for r in self.regions for b in r.keyframes_b})

    @property
    def pairs(self) -> List[Tuple[int, int]]:
        return [p for r in self.regions for p in r.pairs]


def select(affinity: np.ndarray, pos_a: np.ndarray, pos_b: np.ndarray, threshold: float,
           min_area: int, consistency: float = 3.0, stages: Sequence[str] = STAGES,
           solver: str = "auto", time_limit: float = 0.0, verbose: bool = False,
           tie_break: bool = True, log=print) -> Selection:
    """Run the selected stages on an (Na, Nb) affinity matrix; ``pos_a``/``pos_b`` are the
    keyframe positions (Na, 3), (Nb, 3) of each robot in its own frame."""
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
    stages = tuple(s for s in STAGES if s in stages)
    if "cover" in stages or "clique" in stages:
        solver = solvers.resolve(solver)
    na, nb = affinity.shape
    if len(pos_a) < na or len(pos_b) < nb:
        raise ValueError(f"affinity {affinity.shape} but {len(pos_a)} / {len(pos_b)} poses")

    t0 = time.perf_counter()
    boxes = extract_regions(affinity, threshold, min_area) if "regions" in stages else [(0, na, 0, nb)]
    t_regions = time.perf_counter() - t0
    log(f"[commerge] {len(boxes)} region(s) (threshold {threshold}, min area {min_area}) "
        f"in {t_regions:.3f} s; stages {'+'.join(stages) or 'none'}, solver {solver}")

    regions = []
    for k, (a0, a1, b0, b1) in enumerate(boxes):
        r = Region((a0, a1, b0, b1))
        sub = affinity[a0:a1, b0:b1]
        ii, jj = np.nonzero(sub >= threshold)
        pairs = np.c_[ii + a0, jj + b0]
        r.candidates = len(pairs)
        if r.candidates == 0:
            regions.append(r)
            continue

        # 2. minimum vertex cover of the bipartite candidate graph
        if "cover" in stages:
            t = time.perf_counter()
            a_nodes = [("A", int(a)) for a in np.unique(pairs[:, 0])]
            b_nodes = [("B", int(b)) for b in np.unique(pairs[:, 1])]
            edges = [(("A", int(a)), ("B", int(b))) for a, b in pairs]
            # Minimum-cardinality cover (unit weights, as in the paper). Many covers share that
            # size; a tie-break far below 1 picks, among them, the keyframes with the strongest
            # candidate pairs, so Gurobi and HiGHS return the same cover.
            score = {}
            for (a, b) in pairs:
                s = float(affinity[a, b])
                score[("A", int(a))] = score.get(("A", int(a)), 0.0) + s
                score[("B", int(b))] = score.get(("B", int(b)), 0.0) + s
            top = max(score.values())
            eps = 0.5 / (len(score) + 1)
            weights = {v: 1.0 - eps * score[v] / top for v in score} if tie_break else None
            cover = solvers.min_vertex_cover(a_nodes + b_nodes, edges, a_nodes, b_nodes, weights,
                                             solver=solver, time_limit=time_limit, verbose=verbose)
            r.cover_a = sorted(i for s, i in cover if s == "A")
            r.cover_b = sorted(i for s, i in cover if s == "B")
            keep = np.isin(pairs[:, 0], r.cover_a) & np.isin(pairs[:, 1], r.cover_b)
            pairs = pairs[keep]
            r.seconds["cover"] = time.perf_counter() - t
        else:
            r.cover_a = sorted(set(pairs[:, 0].tolist()))
            r.cover_b = sorted(set(pairs[:, 1].tolist()))

        # 3. maximum edge-weight clique of the consistency graph
        if "clique" in stages and len(pairs) > 1:
            t = time.perf_counter()
            w = affinity[pairs[:, 0], pairs[:, 1]]
            M = consistency_scores(pairs, w, pos_a, pos_b, consistency)
            chosen = solvers.max_edge_weight_clique(M, solver=solver, time_limit=time_limit, verbose=verbose)
            pairs = pairs[chosen]
            r.seconds["clique"] = time.perf_counter() - t

        r.pairs = [(int(a), int(b)) for a, b in pairs]
        log(f"[commerge] region {k}: A {a0}-{a1 - 1} x B {b0}-{b1 - 1}: {r.candidates} candidate pairs"
            f" -> cover {len(r.cover_a)}+{len(r.cover_b)} keyframes -> {len(r.pairs)} pairs "
            f"({len(r.keyframes_a)} A + {len(r.keyframes_b)} B scans)"
            + "".join(f", {s} {v:.2f} s" for s, v in r.seconds.items()))
        regions.append(r)
    return Selection(regions, stages, solver)
