"""Radar points of each keyframe, for the map drawings (make_gif.py): RadLoc's polar-to-Cartesian detections,
the strongest returns of each Navtech scan (MulRan: 0.0595 m range bins), in the radar frame.

    python points.py --seq <MulRan>/KAIST_03 --sessions runs/radar/robot_a runs/radar/robot_b
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "thirdparty/RadLoc/python"))
import radloc  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--seq", required=True)
p.add_argument("--sessions", nargs="+", required=True)
p.add_argument("--resolution", type=float, default=0.0595)
p.add_argument("--keep", type=float, default=99.7, help="percentile of the returns kept")
p.add_argument("--max-range", type=float, default=80.0)
a = p.parse_args()

for s in a.sessions:
    out = Path(s) / "points"
    out.mkdir(exist_ok=True)
    frames = (Path(s) / "frames.txt").read_text().split()
    for k, f in enumerate(frames):
        scan = radloc.load_polar_image(str(Path(a.seq) / "polar" / f))
        scan = scan[: int(a.max_range / a.resolution)]
        q = radloc.to_cartesian_points(scan, a.resolution, np.percentile(scan, a.keep), range_offset=40)
        np.save(out / f"{k:06d}.npy", q.astype(np.float32))
    print(f"{out}: {len(frames)} keyframes")
