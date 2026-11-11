"""Thermal points of each keyframe, for the map drawings (make_gif.py): monocular depth from the thermal image alone
(AnyThermal's MiDaS head, thirdparty/BridgeMultiSpectralDepth + the AnyThermal checkpoints), back-projected with
the FLIR A65 13 mm lens (f = 13 mm / 17 um = 765 px; ViViD++'s camera_info is empty), ground removed, in the
vehicle frame (x forward, y left, z up).

    PYTHONPATH=<deps> python points.py --ckpt <dir with pretrained_checkpoints/> --sessions runs/thermal/day runs/thermal/night \
        --into runs/thermal/robot_a runs/thermal/robot_b
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
DEPTH = HERE / "thirdparty/BridgeMultiSpectralDepth"
sys.modules["pytorch3d"] = types.ModuleType("pytorch3d")   # imported by a training loss only
sys.modules["pytorch3d.loss"] = types.SimpleNamespace(chamfer_distance=None)
sys.path.insert(0, str(DEPTH))

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ckpt", required=True)
p.add_argument("--sessions", nargs="+", required=True, help="extract.py outputs (images/)")
p.add_argument("--into", nargs="+", required=True, help="the Commerge sessions that get points/")
p.add_argument("--f", type=float, default=765.0)
p.add_argument("--max-depth", type=float, default=35.0)
p.add_argument("--height", type=float, default=1.2, help="points lower than this below the camera are ground (m)")
p.add_argument("--per-frame", type=int, default=3000)
a = p.parse_args()
ckpt = Path(a.ckpt).resolve() / "pretrained_checkpoints"
into = [Path(s).resolve() for s in a.into]
sessions = [Path(s).resolve() for s in a.sessions]

os.chdir(DEPTH)
from mmcv import Config  # noqa: E402
from models import MODELS  # noqa: E402
import dataloaders.custom_transforms as cs_tf  # noqa: E402

cfg = Config.fromfile("configs/AnyThermal/Midas_anythermal.yaml")
cfg.model.backbone_path = str(ckpt / "backbone/AnyThermal_full/model20.pth")
model = MODELS.build(name=cfg.model.name, option=cfg)
w = torch.load(ckpt / "depth/Midas_anythermal/ckpt_epoch=28_step=145000.ckpt", map_location="cpu", weights_only=False)
model.load_state_dict(w["state_dict"])
model = model.cuda().eval()
SIZE = [448, 560]  # 640 x 512 at the same aspect, a multiple of the ViT patch (14)
tf = cs_tf.CustomCompose([cs_tf.RescaleTo(SIZE), cs_tf.ArrayToTensor(Itype="thr"),
                          cs_tf.TensorImgEnhance("thr", cfg.dataset["Augmentation"]["thr"]), cs_tf.Normalize()])
rng = np.random.default_rng(0)

for src, dst in zip(sessions, into):
    out = dst / "points"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((src / "images").glob("*.png"))
    for k, fpath in enumerate(files):
        img = np.asarray(Image.open(fpath)).astype(np.float32)
        H, W = img.shape
        x = tf([img[..., None]], None, None)[0]["img_in"][0].cuda()
        with torch.no_grad():
            d = model.inference_depth(x.unsqueeze(0)).squeeze().cpu().numpy() / 1000.0  # mm -> m
        v, u = np.mgrid[0:SIZE[0], 0:SIZE[1]]
        su, sv = W / SIZE[1], H / SIZE[0]
        uu, vv, z = (u.ravel() + 0.5) * su, (v.ravel() + 0.5) * sv, d.ravel()
        X, Y = (uu - W / 2) * z / a.f, (vv - H / 2) * z / a.f      # camera: x right, y down, z forward
        keep = (z > 2.0) & (z < a.max_depth) & (Y < a.height)
        q = np.c_[z[keep], -X[keep], -Y[keep]]                     # vehicle: x forward, y left, z up
        if len(q) > a.per_frame:
            q = q[rng.choice(len(q), a.per_frame, replace=False)]
        np.save(out / f"{k:06d}.npy", q.astype(np.float32))
    print(f"{out}: {len(files)} keyframes")
