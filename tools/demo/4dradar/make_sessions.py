"""4D radar with SOLiD: two SOLiD radar sessions (accumulated radar submaps per keyframe, as SOLiD's radar SLAM
writes them), their keyframe poses replaced by ground truth and robot B's moved into its own frame.

    # 1. one SOLiD session per robot (thirdparty/SOLiD, ROS 2, GeoRIO odometry), e.g. SNAIL-Radar st route:
    ros2 launch solid_ros run.launch.py odometry:=georio save_dir:=$HOME/st_20240113   # while data1 of 20240113 plays
    ros2 launch solid_ros run.launch.py odometry:=georio save_dir:=$HOME/st_20231213
    # 2. ground-truth poses at the keyframe times (SNAIL ref_tls_T_xt32.csv), B moved by yaw 60 deg, (30, -20, 0) m
    python make_sessions.py --a $HOME/st_20240113 --gt-a <gt>/20240113/data1/ref_tls_T_xt32.csv \
                            --b $HOME/st_20231213 --gt-b <gt>/20231213/data1/ref_tls_T_xt32.csv --out runs/4dradar
    # 3. merge
    commerge runs/4dradar/robot_a runs/4dradar/robot_b --threshold 0.96 --min-area 30 --registration pairs \
        --policy all --out runs/4dradar/merged --gt-a runs/4dradar/robot_a/gt_poses.txt --gt-b runs/4dradar/robot_b/gt_poses.txt

The sessions keep their Scans/ and SOLiDs/ (symbolic links), so --registration kiss also works on them.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import interpolate_poses, rigid, write_session  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--a", required=True, help="robot A's SOLiD session (Scans/, SOLiDs/, times.txt)")
p.add_argument("--b", required=True, help="robot B's SOLiD session")
p.add_argument("--gt-a", required=True, help="ground truth of A: time, x, y, z, qx, qy, qz, qw (csv or text)")
p.add_argument("--gt-b", required=True)
p.add_argument("--out", required=True)
p.add_argument("--yaw", type=float, default=60.0)
p.add_argument("--shift", type=float, nargs=3, default=[30.0, -20.0, 0.0])
a = p.parse_args()


def load_gt(path):
    g = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None, comments="#")[:, :8]
    T = np.tile(np.eye(4), (len(g), 1, 1))
    T[:, :3, 3] = g[:, 1:4]
    T[:, :3, :3] = Rotation.from_quat(g[:, 4:8]).as_matrix()
    return g[:, 0], T


for name, src, gt, move in (("robot_a", a.a, a.gt_a, None), ("robot_b", a.b, a.gt_b, rigid(a.yaw, a.shift))):
    t = np.loadtxt(Path(src) / "times.txt", ndmin=1)
    T = interpolate_poses(t, *load_gt(gt))
    root = write_session(Path(a.out) / name, T, t, move=move)
    for d in ("Scans", "SOLiDs"):
        if (Path(src) / d).is_dir() and not (root / d).exists():
            os.symlink(Path(src).resolve() / d, root / d)
    print(f"{root}: {len(t)} keyframes" + ("" if move is None else f", moved by yaw {a.yaw} deg, {a.shift} m"))
