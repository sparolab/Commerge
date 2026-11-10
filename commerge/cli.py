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
from .merge import POLICIES, alignment_error, choose_regions, register
from .session import Session, read_kitti_poses
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
    g = p.add_argument_group("exchange and merging")
    g.add_argument("--policy", choices=POLICIES, default="score",
                   help="region(s) whose scans are exchanged: score (largest step-(C) clique weight, default), "
                        "area, candidates or all")
    g.add_argument("--no-register", action="store_true",
                   help="do not register the exchanged submaps (registration needs kiss-matcher, small-gicp)")
    g.add_argument("--reg-voxel", type=float, default=0.5, help="registration voxel size (m)")
    g.add_argument("--gt-a", type=Path, help="ground-truth keyframe poses of A (KITTI, common frame): report the error")
    g.add_argument("--gt-b", type=Path, help="ground-truth keyframe poses of B (same frame as --gt-a)")
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
    # exchange policy: the region(s) whose selected scans the robots actually send
    chosen = choose_regions(sel, aff, A.positions, B.positions, args.policy, args.consistency)
    ka = sorted({a for k in chosen for a in sel.regions[k].keyframes_a})
    kb = sorted({b for k in chosen for b in sel.regions[k].keyframes_b})
    summary["exchange"] = {"policy": args.policy, "regions": chosen, "keyframes_a": ka, "keyframes_b": kb}
    if ka and kb and not (args.no_submaps and args.no_register):
        sub_a = build_submap(A, ka, args.min_range, args.voxel)
        sub_b = build_submap(B, kb, args.min_range, args.voxel)
        if not args.no_submaps:
            save_submap(out / "submap_A.pcd", sub_a)
            save_submap(out / "submap_B.pcd", sub_b)
        if not args.no_register:
            try:
                reg = register(sub_a, sub_b, args.reg_voxel)
            except ImportError as e:
                reg = False
                print(f"[commerge] registration skipped: {e}")
            if reg is False:
                pass
            elif reg is None:
                print("[commerge] registration failed (too few inliers); try another --policy")
            else:
                np.savetxt(out / "T_a_b.txt", reg.T_a_b, fmt="%.9f")
                merged = np.einsum("ij,njk->nik", reg.T_a_b, B.poses)[:, :3, :4].reshape(len(B), 12)
                np.savetxt(out / "poses_b_in_a.txt", merged, fmt="%.9f")
                summary["registration"] = {"inliers": reg.inliers, "gicp_error": reg.fitness, "seconds": reg.seconds}
                msg = f"[commerge] registered B to A: {reg.inliers} KISS-Matcher inliers, GICP in {reg.seconds:.1f} s"
                if args.gt_a and args.gt_b:
                    et, er, em = alignment_error(reg.T_a_b, A.poses, B.poses, read_kitti_poses(args.gt_a),
                                                 read_kitti_poses(args.gt_b))
                    summary["registration"]["error"] = {"trans_m": et, "rot_deg": er, "merged_mean_m": em}
                    msg += f"; vs ground truth: {et:.2f} m, {er:.2f} deg (B keyframes {em:.2f} m off on average)"
                print(msg)
    (out / "selection.json").write_text(json.dumps(summary, indent=1))

    if args.plot:
        from .viz import plot_selection

        plot_selection(aff, sel, out / "selection.png", args.threshold)

    total = len(A) + len(B)
    sent = len(ka) + len(kb)
    mb = sum(A.scans[k].stat().st_size for k in ka) / 1e6 + sum(B.scans[k].stat().st_size for k in kb) / 1e6 \
        if A.scans and B.scans else float("nan")
    print(f"[commerge] exchanged {len(ka)} A + {len(kb)} B scans ({100.0 * sent / max(total, 1):.1f} % of {total}, "
          f"{mb:.1f} MB) from region(s) {chosen} of {len(sel.regions)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
