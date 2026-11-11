"""Camera points of each keyframe, for the map drawings (make_gif.py): stereo depth of KITTI's color pair
(image_2 / image_3, OpenCV SGBM), road removed, in the LiDAR frame of the session poses.

    python points.py --seq <KITTI color>/sequences/00 --sessions robot_a robot_b [--calib <KITTI>/sequences/00/calib.txt]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# velodyne -> cam0 of KITTI odometry sequences 00-02 (calib.txt "Tr"), used when --calib has no Tr line
TR_00 = [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
         -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
         9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--seq", required=True, help="KITTI color sequence folder (image_2/, image_3/, calib.txt)")
p.add_argument("--sessions", nargs="+", required=True)
p.add_argument("--calib", help="calib.txt with P2, P3 and Tr (default: <seq>/calib.txt, Tr of sequence 00)")
p.add_argument("--max-depth", type=float, default=35.0)
p.add_argument("--per-frame", type=int, default=3000)
p.add_argument("--t0", type=float, default=1e9)
p.add_argument("--period", type=float, default=0.1)
a = p.parse_args()

cal = {}
for line in open(a.calib or Path(a.seq) / "calib.txt"):
    k, *v = line.split()
    cal[k.rstrip(":")] = np.array(v, dtype=float)
P2, P3 = cal["P2"].reshape(3, 4), cal["P3"].reshape(3, 4)
f, cx, cy = P2[0, 0], P2[0, 2], P2[1, 2]
base = (P2[0, 3] - P3[0, 3]) / f                       # stereo baseline (m)
off2 = P2[0, 3] / f                                    # cam2 is this far along x from cam0
Tr = np.eye(4)
Tr[:3, :4] = np.reshape(cal.get("Tr", TR_00), (3, 4))
cam0_to_velo = np.linalg.inv(Tr)
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5, P1=8 * 3 * 25, P2=32 * 3 * 25,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
rng = np.random.default_rng(0)

for s in a.sessions:
    out = Path(s) / "points"
    out.mkdir(exist_ok=True)
    frames = np.rint((np.loadtxt(Path(s) / "times.txt", ndmin=1) - a.t0) / a.period).astype(int)
    for k, fr in enumerate(frames):
        L = cv2.imread(str(Path(a.seq) / "image_2" / f"{fr:06d}.png"), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(Path(a.seq) / "image_3" / f"{fr:06d}.png"), cv2.IMREAD_GRAYSCALE)
        d = sgbm.compute(L, R).astype(np.float32) / 16.0
        v, u = np.nonzero(d > 1.0)
        z = f * base / d[v, u]
        keep = z < a.max_depth
        u, v, z = u[keep], v[keep], z[keep]
        cam = np.c_[(u - cx) * z / f + off2, (v - cy) * z / f, z, np.ones(len(z))]  # cam0 frame
        q = (cam @ cam0_to_velo.T)[:, :3]
        q = q[q[:, 2] > -1.3]                              # drop the road (LiDAR ~1.73 m above it)
        if len(q) > a.per_frame:
            q = q[rng.choice(len(q), a.per_frame, replace=False)]
        np.save(out / f"{k:06d}.npy", q.astype(np.float32))
    print(f"{out}: {len(frames)} keyframes")
