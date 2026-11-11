"""Camera with MegaLoc: the KITTI 00 two-robot split of the main demo (robot A frames 0-1999, robot B 2000-4540, the
same keyframes), described by the left color camera instead of SOLiD. Keyframe poses: KITTI ground truth (poses.txt,
in the LiDAR frame), robot B's moved into its own frame (yaw 60 deg, (30, -20, 0) m).

    python make_sessions.py --images <KITTI color>/sequences/00/image_2 --poses <KITTI>/poses/00.txt \
        --a <SOLiD session of A> --b <SOLiD session of B> --out runs/camera
    python points.py --seq <KITTI color>/sequences/00 --sessions runs/camera/robot_a runs/camera/robot_b
    commerge runs/camera/robot_a runs/camera/robot_b --affinity runs/camera/affinity.npy --threshold 0.8 \
        --min-area 300 --registration pairs --policy all --out runs/camera/merged \
        --gt-a runs/camera/robot_a/gt_poses.txt --gt-b runs/camera/robot_b/gt_poses.txt

--a / --b: the main demo's SOLiD sessions (AGENTS.md, step 4); their times.txt give each keyframe's KITTI frame.
Descriptor: MegaLoc (torch.hub gmberton/MegaLoc, 8448-D), cosine affinity.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--images", required=True, help="KITTI color sequences/00/image_2")
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--poses", required=True, help="KITTI ground truth poses/00.txt (camera frame)")
p.add_argument("--calib", help="calib.txt with Tr (default: Tr of sequence 00)")
p.add_argument("--yaw", type=float, default=60.0)
p.add_argument("--shift", type=float, nargs=3, default=[30.0, -20.0, 0.0])
p.add_argument("--out", required=True)
p.add_argument("--period", type=float, default=0.1, help="KITTI frame period (s)")
p.add_argument("--t0", type=float, default=1e9, help="time of frame 0 in times.txt (SOLiD's KITTI reader: 1e9)")
p.add_argument("--batch", type=int, default=16)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True).eval().to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def frames_of(session):
    t = np.loadtxt(Path(session) / "times.txt", ndmin=1)
    return t, np.rint((t - a.t0) / a.period).astype(int)


def describe(session):
    t, frames = frames_of(session)
    out = []
    for i in range(0, len(frames), a.batch):
        ims = [np.asarray(Image.open(Path(a.images) / f"{f:06d}.png").convert("RGB")) for f in frames[i:i + a.batch]]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(dev) / 255.0
        x = torch.nn.functional.interpolate(x, size=(322, 1064), mode="bilinear", align_corners=False)  # 14 x 23, 14 x 76
        with torch.no_grad():
            d = model((x - mean) / std)
        out.append(torch.nn.functional.normalize(d, dim=1).cpu().numpy())
    print(f"{session}: {len(frames)} keyframes, frames {frames[0]}..{frames[-1]}")
    return np.concatenate(out)


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import report_affinity, rigid, write_session  # noqa: E402

Tr = np.eye(4)
tr = None
if a.calib:
    for line in open(a.calib):
        if line.startswith("Tr:"):
            tr = np.array(line.split()[1:], dtype=float)
if tr is None:
    src = (HERE / "points.py").read_text()
    ns = {}
    exec(src[src.index("TR_00"):src.index("p = argparse")], ns)
    tr = np.array(ns["TR_00"])
Tr[:3, :4] = tr.reshape(3, 4)
P = np.loadtxt(a.poses).reshape(-1, 3, 4)
G = np.tile(np.eye(4), (len(P), 1, 1))
G[:, :3, :4] = P
G = np.linalg.inv(Tr) @ G @ Tr                       # camera-frame poses -> LiDAR frame
out = Path(a.out)
(ta, fa), (tb, fb) = frames_of(a.a), frames_of(a.b)
write_session(out / "robot_a", G[fa], ta)
write_session(out / "robot_b", G[fb], tb, move=rigid(a.yaw, a.shift))
da, db = describe(a.a), describe(a.b)
S = da @ db.T
np.save(out / "affinity.npy", S.astype(np.float32))
report_affinity(S, G[fa], G[fb])
print(f"-> {out}: robot_a {len(fa)}, robot_b {len(fb)} keyframes (B moved by yaw {a.yaw} deg, {a.shift} m)")
