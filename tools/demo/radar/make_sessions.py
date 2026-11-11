"""Spinning radar (Navtech) with RadLoc: two robots from one MulRan sequence, one of them in its own frame.

    python make_sessions.py --seq /data/MulRan/KAIST_03 --gt /data/MulRan/KAIST_03/global_pose.csv --out runs/radar
    commerge runs/radar/robot_a runs/radar/robot_b --affinity runs/radar/affinity.npy --threshold 0.93 \
        --min-area 50 --registration pairs --policy all --out runs/radar/merged \
        --gt-a runs/radar/robot_a/gt_poses.txt --gt-b runs/radar/robot_b/gt_poses.txt

Robot A drives the first half of the sequence, robot B the second half, which re-drives much of A's route.
Keyframes every 2 m; descriptor: RadLoc (thirdparty/RadLoc, ReFeree++ bands), compared by mean absolute
difference, as RadLoc's fine stage does.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE / "thirdparty/RadLoc/python")]
import radloc  # noqa: E402
from common import (distance_to_affinity, interpolate_poses, keyframes_by_distance, report_affinity, rigid,  # noqa: E402
                    write_session)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--seq", required=True, help="MulRan sequence folder with polar/ (Navtech polar images)")
p.add_argument("--gt", required=True, help="MulRan global_pose.csv (time ns, 3x4 pose)")
p.add_argument("--out", required=True)
p.add_argument("--step", type=float, default=2.0, help="keyframe spacing (m)")
p.add_argument("--split", type=float, default=0.5, help="share of the keyframes driven by robot A")
p.add_argument("--yaw", type=float, default=60.0, help="robot B's frame: yaw (deg) ...")
p.add_argument("--shift", type=float, nargs=3, default=[30.0, -20.0, 0.0], help="... and translation (m)")
a = p.parse_args()

images = sorted((Path(a.seq) / "polar").glob("*.png"))
t = np.array([int(f.stem) for f in images], dtype=np.int64)
g = np.loadtxt(a.gt, delimiter=",")
G = np.tile(np.eye(4), (len(g), 1, 1))
G[:, :3, :4] = g[:, 1:13].reshape(-1, 3, 4)
G[:, :3, 3] -= G[0, :3, 3]                      # UTM -> local
T = interpolate_poses((t - g[0, 0]) / 1e9, (g[:, 0] - g[0, 0]) / 1e9, G)
k = keyframes_by_distance(T, a.step)
print(f"{len(images)} scans, {len(k)} keyframes every {a.step} m")

desc = []
for n, i in enumerate(k):
    desc.append(radloc.compute_descriptor(str(images[i])))
    if n % 200 == 0:
        print(f"  RadLoc {n} / {len(k)}", end="\r", flush=True)
desc = np.array(desc)
s = int(len(k) * a.split)
ka, kb = k[:s], k[s:]
D = np.abs(desc[:s, None, :] - desc[None, s:, :]).mean(2)  # RadLoc fine distance (mean absolute difference)
S = distance_to_affinity(D)

out = Path(a.out)
write_session(out / "robot_a", T[ka], t[ka] / 1e9, [images[i].name for i in ka])
write_session(out / "robot_b", T[kb], t[kb] / 1e9, [images[i].name for i in kb], move=rigid(a.yaw, a.shift))
np.save(out / "affinity.npy", S.astype(np.float32))
report_affinity(S, T[ka], T[kb])
print(f"-> {out}: robot_a {len(ka)}, robot_b {len(kb)} keyframes (B moved by yaw {a.yaw} deg, {a.shift} m)")
