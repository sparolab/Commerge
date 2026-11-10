"""commerge: select the scans two robots exchange to merge their maps.

    commerge <session A> <session B> --out merged/ [--threshold 0.94 --min-area 10000]

Sessions are SOLiD outputs (LT-mapper format: Scans/, SOLiDs/, *_poses.txt). The affinity is
the cosine similarity of their SOLiD descriptors, or any (Na, Nb) matrix given with --affinity.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .selection import STEPS, cosine_affinity, select
from .session import Session
from .solvers import SOLVERS
from .submap import build_submap, save_submap


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="commerge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_a", type=Path, help="robot A: SOLiD session folder")
    p.add_argument("session_b", type=Path, help="robot B: SOLiD session folder")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    g = p.add_argument_group("affinity")
    g.add_argument("--affinity", type=Path, help="(Na, Nb) .npy instead of the SOLiD descriptors")
    g.add_argument("--descriptor", choices=("rsolid", "asolid", "both"), default="rsolid",
                   help="SOLiD part used for the cosine affinity (default: rsolid)")
    g.add_argument("--poses", default="auto", help="auto | optimized | odom: keyframe poses to use")
    g = p.add_argument_group("selection")
    g.add_argument("--threshold", type=float, required=True, help="affinity threshold of a candidate pair")
    g.add_argument("--min-area", type=int, required=True, help="smallest region (cells of its bounding box)")
    g.add_argument("--consistency", type=float, default=3.0,
                   help="largest distance mismatch (m) between two consistent pairs (default: 3)")
    g.add_argument("--steps", default="ABC",
                   help="steps to run, any subset of ABC (default): (A) " + STEPS["A"] + ", (B) " + STEPS["B"]
                   + ", (C) " + STEPS["C"])
    g.add_argument("--solver", choices=SOLVERS, default="auto",
                   help="gurobi (paper, needs a license) | highs (SciPy, no license) | auto (default)")
    g.add_argument("--time-limit", type=float, default=0.0, help="per-problem solver time limit, s (0: none)")
    g = p.add_argument_group("output")
    g.add_argument("--no-submaps", action="store_true", help="only write the selection")
    g.add_argument("--min-range", type=float, default=3.0, help="drop points closer than this (m)")
    g.add_argument("--voxel", type=float, default=0.0, help="submap voxel size (m, 0: none)")
    g.add_argument("--plot", action="store_true", help="save the affinity with regions and pairs")
    g.add_argument("--verbose", action="store_true", help="solver logs")
    p.add_argument("--version", action="version", version=f"commerge {__version__}")
    return p.parse_args(argv)


def descriptor(s: Session, which: str) -> np.ndarray:
    if s.rsolid is None:
        raise SystemExit(f"{s.name}: no SOLiDs/ folder; give the affinity with --affinity")
    if which == "rsolid":
        return s.rsolid
    if s.asolid is None:
        raise SystemExit(f"{s.name}: SOLiDs/ has no A-SOLiD line")
    return s.asolid if which == "asolid" else np.c_[s.rsolid, s.asolid]


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    A = Session.load(args.session_a, args.poses)
    B = Session.load(args.session_b, args.poses)
    print(f"[commerge] {A.name}: {len(A)} keyframes | {B.name}: {len(B)} keyframes")
    if args.affinity:
        aff = np.load(args.affinity)
    else:
        t = time.perf_counter()
        aff = cosine_affinity(descriptor(A, args.descriptor), descriptor(B, args.descriptor))
        print(f"[commerge] affinity {aff.shape} from {args.descriptor} in {time.perf_counter() - t:.2f} s")
        np.save(out / "affinity.npy", aff.astype(np.float32))

    sel = select(aff, A.positions, B.positions, args.threshold, args.min_area, args.consistency,
                 args.steps, args.solver, args.time_limit, args.verbose)

    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "keyframe_a", "keyframe_b", "affinity"])
        for k, r in enumerate(sel.regions):
            for a, b in r.pairs:
                w.writerow([k, a, b, f"{aff[a, b]:.6f}"])

    summary = {
        "session_a": str(args.session_a), "session_b": str(args.session_b),
        "keyframes": [len(A), len(B)], "threshold": args.threshold, "min_area": args.min_area,
        "consistency": args.consistency, "steps": sel.steps, "solver": sel.solver,
        "selected_scans": [len(sel.keyframes_a), len(sel.keyframes_b)],
        "regions": [{"box_a": [r.box[0], r.box[1]], "box_b": [r.box[2], r.box[3]],
                     "A_candidates": r.candidates, "B_cover": [len(r.cover_a), len(r.cover_b)],
                     "C_pairs": len(r.pairs), "keyframes_a": r.keyframes_a, "keyframes_b": r.keyframes_b,
                     "seconds": r.seconds} for r in sel.regions],
    }
    if not args.no_submaps:
        for k, r in enumerate(sel.regions):
            if not r.pairs:
                continue
            pa = save_submap(out / f"submap_A_{k}.pcd", build_submap(A, r.keyframes_a, args.min_range, args.voxel))
            pb = save_submap(out / f"submap_B_{k}.pcd", build_submap(B, r.keyframes_b, args.min_range, args.voxel))
            summary["regions"][k]["submaps"] = [pa.name, pb.name]
    (out / "selection.json").write_text(json.dumps(summary, indent=1))

    if args.plot:
        from .viz import plot_selection

        plot_selection(aff, sel, out / "selection.png", args.threshold)

    total = len(A) + len(B)
    sent = len(sel.keyframes_a) + len(sel.keyframes_b)
    print(f"[commerge] selected {len(sel.keyframes_a)} A + {len(sel.keyframes_b)} B scans "
          f"({100.0 * sent / max(total, 1):.1f} % of {total}) in {len(sel.regions)} region(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
