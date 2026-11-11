"""commerge-compare: Commerge, its ablations (without step (B) or (C), step (A) only) and AutoMerge
on the same two sessions.

    commerge-compare <session A> <session B> --threshold 0.96 --min-area 2000 --out cmp/
                     [--gt-a gt_a.txt --gt-b gt_b.txt]

Reports, per method, the scans each robot sends (count, share of all keyframes, megabytes of the
keyframe clouds) and the run time. With ground-truth keyframe poses in a common frame
(KITTI format, one row per keyframe), it also reports the share of selected pairs that are
really the same place (within --gt-radius meters).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .baselines.automerge import AutoMergeParams, automerge_select
from .cli import descriptor
from .merge import alignment_error, choose_regions, register
from .selection import select
from .submap import build_submap
from .session import Session, read_kitti_poses
from .solvers import SOLVERS

METHODS = {
    "Commerge (A+B+C)": "ABC",
    "w/o (B)": "AC",
    "w/o (C)": "AB",
    "(A) only": "A",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="commerge-compare", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_a", type=Path)
    p.add_argument("session_b", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--affinity", type=Path, help="(Na, Nb) .npy instead of SOLiD descriptors")
    p.add_argument("--descriptor", choices=("rsolid", "asolid", "both"), default="rsolid")
    p.add_argument("--poses", default="auto")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--min-area", type=int, required=True)
    p.add_argument("--consistency", type=float, default=3.0)
    p.add_argument("--solver", choices=SOLVERS, default="auto")
    p.add_argument("--time-limit", type=float, default=60.0,
                   help="per-problem solver limit, s (default 60: the ablations without the cover can be huge)")
    p.add_argument("--automerge-min-area", type=int, help="AutoMerge region size (default: --min-area)")
    p.add_argument("--gt-a", type=Path, help="ground-truth keyframe poses of A (KITTI, common frame)")
    p.add_argument("--gt-b", type=Path, help="ground-truth keyframe poses of B (same frame as --gt-a)")
    p.add_argument("--gt-radius", type=float, default=5.0)
    p.add_argument("--policy", default="score", help="exchange policy of the Commerge variants (see commerge --help)")
    p.add_argument("--no-register", action="store_true", help="skip registering each method's exchanged scans")
    return p.parse_args(argv)


def scan_megabytes(s: Session, keyframes) -> float:
    return sum(s.scans[k].stat().st_size for k in keyframes) / 1e6 if s.scans else float("nan")


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    A, B = Session.load(args.session_a, args.poses), Session.load(args.session_b, args.poses)
    aff = np.load(args.affinity) if args.affinity else \
        __import__("commerge").cosine_affinity(descriptor(A, args.descriptor), descriptor(B, args.descriptor))
    gt = gtT = None
    if args.gt_a and args.gt_b:
        gtT = (read_kitti_poses(args.gt_a), read_kitti_poses(args.gt_b))
        gt = (gtT[0][:, :3, 3], gtT[1][:, :3, 3])

    rows, selections = [], {}
    quiet = lambda *a: None  # noqa: E731
    for name, steps in METHODS.items():
        t = time.perf_counter()
        sel = select(aff, A.positions, B.positions, args.threshold, args.min_area, args.consistency,
                     steps, args.solver, args.time_limit, log=quiet)
        selections[name] = (sel, time.perf_counter() - t)
    t = time.perf_counter()
    am = AutoMergeParams(min_area=args.automerge_min_area or args.min_area)
    selections["AutoMerge"] = (automerge_select(aff, am, log=quiet), time.perf_counter() - t)

    total = len(A) + len(B)
    for name, (sel, sec) in selections.items():
        # what is actually exchanged: the chosen region (AutoMerge keeps its best region already)
        chosen = choose_regions(sel, aff, A.positions, B.positions, args.policy if name != "AutoMerge" else "all")
        ka = sorted({a for k in chosen for a in sel.regions[k].keyframes_a})
        kb = sorted({b for k in chosen for b in sel.regions[k].keyframes_b})
        pairs = [p for k in chosen for p in sel.regions[k].pairs]
        row = {"method": name, "scans_a": len(ka), "scans_b": len(kb),
               "share": 100.0 * (len(ka) + len(kb)) / total,
               "megabytes": scan_megabytes(A, ka) + scan_megabytes(B, kb),
               "pairs": len(pairs), "seconds": sec}
        if gt is not None and pairs:
            d = np.array([np.linalg.norm(gt[0][a] - gt[1][b]) for a, b in pairs])
            row["pair_precision"] = float((d <= args.gt_radius).mean())
        if not args.no_register and ka and kb:
            try:
                reg = register(build_submap(A, ka), build_submap(B, kb))
            except ImportError:
                args.no_register = True
                print("[compare] kiss-matcher not installed: registration skipped")
                reg = None
            row["registered"] = reg is not None
            if reg is not None and gtT is not None:
                et, er, em = alignment_error(reg.T_a_b, A.poses, B.poses, *gtT)
                row.update(trans_err=et, rot_err=er, merged_err=em)
        rows.append(row)

    print("| method | scans A + B | share of keyframes | data (MB) | pairs | pair precision | merge error | time (s) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        prec = f"{100 * r['pair_precision']:.1f} %" if "pair_precision" in r else "-"
        if "trans_err" in r:
            err = f"{r['trans_err']:.2f} m, {r['rot_err']:.2f} deg"
        else:
            err = "failed" if r.get("registered") is False else "-"
        print(f"| {r['method']} | {r['scans_a']} + {r['scans_b']} | {r['share']:.1f} % | {r['megabytes']:.1f} | "
              f"{r['pairs'] or '-'} | {prec} | {err} | {r['seconds']:.2f} |")
    (args.out / "comparison.json").write_text(json.dumps(rows, indent=1))

    from .viz import plot_comparison

    plot_comparison(aff, {k: v[0] for k, v in selections.items()}, args.out / "comparison.png", args.threshold, rows)
    print(f"-> {args.out}/comparison.json, comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
