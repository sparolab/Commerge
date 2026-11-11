"""Sonar points of each keyframe, for the map drawings (make_gif.py): the strongest returns of the forward-looking
sonar image (BlueView P900-130: 130 deg, 50 m; Aracati polar images: 300 beams x 1500 range bins), in the vehicle
frame (x forward, y left).

    python points.py --images polar_image.zip --sessions runs/sonar/robot_a runs/sonar/robot_b
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--images", required=True, help="polar_image.zip or a folder")
p.add_argument("--sessions", nargs="+", required=True)
p.add_argument("--range", type=float, default=50.0)
p.add_argument("--fov", type=float, default=130.0)
p.add_argument("--keep", type=float, default=99.5, help="percentile of the returns kept")
p.add_argument("--min-range", type=float, default=4.0, help="skip the bright near field (m)")
p.add_argument("--flip", action="store_true", help="beams ordered right to left")
a = p.parse_args()

src = Path(a.images)
z = zipfile.ZipFile(src) if src.suffix == ".zip" else None
index = {Path(n).name: n for n in z.namelist()} if z else {}
for s in a.sessions:
    out = Path(s) / "points"
    out.mkdir(exist_ok=True)
    frames = (Path(s) / "frames.txt").read_text().split()
    for k, f in enumerate(frames):
        img = Image.open(io.BytesIO(z.read(index[f]))) if z else Image.open(src / f)
        im = np.asarray(img.convert("L")).astype(float)          # (beams, range bins)
        nb, nr = im.shape
        rr = (np.arange(nr) + 0.5) / nr * a.range
        im[:, rr < a.min_range] = 0
        b, r = np.nonzero(im > max(np.percentile(im, a.keep), 1))
        th = np.radians((b + 0.5) / nb * a.fov - a.fov / 2) * (-1 if a.flip else 1)
        q = np.c_[rr[r] * np.cos(th), rr[r] * np.sin(th)]
        np.save(out / f"{k:06d}.npy", q.astype(np.float32))
    print(f"{out}: {len(frames)} keyframes")
