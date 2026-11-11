"""Thermal camera with AnyThermal: day and night of ViViD++ campus as two robots, the night one in its own frame.

    python extract.py campus_day1.bag runs/thermal/day                       # prints the GPS origin
    python extract.py campus_night.bag runs/thermal/night --origin LAT LON   # same origin
    python make_sessions.py --day runs/thermal/day --night runs/thermal/night --ckpt <dir with pretrained_checkpoints/> \
        --out runs/thermal
    commerge runs/thermal/robot_a runs/thermal/robot_b --affinity runs/thermal/affinity.npy --threshold 0.91 \
        --min-area 300 --registration pairs --policy all --out runs/thermal/merged \
        --gt-a runs/thermal/robot_a/gt_poses.txt --gt-b runs/thermal/robot_b/gt_poses.txt

Descriptor: AnyThermal (thirdparty/AnyThermal, ICRA 2026: DINOv2 distilled to thermal + SALAD head, checkpoints from
huggingface.co/theairlabcmu/AnyThermal), cosine affinity. Note: AnyThermal's backbone was distilled on data that
include ViViD++. Keyframe poses: GPS positions, heading along the track.
"""
import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE / "thirdparty/AnyThermal"
sys.path[:0] = [str(HERE.parent), str(ROOT)]
from common import report_affinity, rigid, write_session  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--day", required=True)
p.add_argument("--night", required=True)
p.add_argument("--ckpt", required=True, help="folder holding pretrained_checkpoints/ (AnyThermal's zip, extracted)")
p.add_argument("--out", required=True)
p.add_argument("--yaw", type=float, default=60.0)
p.add_argument("--shift", type=float, nargs=3, default=[30.0, -20.0, 0.0])
p.add_argument("--batch", type=int, default=16)
a = p.parse_args()
out = Path(a.out).resolve()

os.chdir(a.ckpt)  # AnyThermal loads pretrained_checkpoints/... relative to the working directory
from custom_models.str_to_cls import get_model_from_string  # noqa: E402

args = types.SimpleNamespace(same_backbone=False)
model = get_model_from_string(args, "vpr_mmdistill_salad_anythermal_full_thr", "vpr")
model.eval()


def to_8bit(img16):
    lo, hi = np.percentile(img16, [1, 99])
    return (np.clip((img16.astype(float) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def describe(folder):
    files = sorted((Path(folder) / "images").glob("*.png"))
    feats = []
    for i in range(0, len(files), a.batch):
        ims = [np.repeat(to_8bit(np.asarray(Image.open(f)))[..., None], 3, axis=2) for f in files[i:i + a.batch]]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float() / 255.0
        feats.append(model.extract_feature(x).cpu().numpy())
    print(f"{folder}: {len(files)} keyframes")
    return np.concatenate(feats)


def poses(folder):
    k = np.loadtxt(Path(folder) / "keyframes.txt", ndmin=2)  # time x y z lat lon sigma
    xyz = k[:, 1:4]
    d = np.gradient(xyz[:, :2], axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    T = np.tile(np.eye(4), (len(k), 1, 1))
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(yaw), -np.sin(yaw), np.sin(yaw), np.cos(yaw)
    T[:, :3, 3] = xyz - [0, 0, xyz[0, 2]]
    return k[:, 0], T


fa, fb = describe(a.day), describe(a.night)
S = fa @ fb.T
ta, Ta = poses(a.day)
tb, Tb = poses(a.night)
write_session(out / "robot_a", Ta, ta)
write_session(out / "robot_b", Tb, tb, move=rigid(a.yaw, a.shift))
np.save(out / "affinity.npy", S.astype(np.float32))
report_affinity(S, Ta, Tb)
print(f"-> {out}: robot_a (day) {len(ta)}, robot_b (night) {len(tb)} keyframes")
