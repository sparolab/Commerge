"""Thermal keyframes from a ViViD++ bag: one 16-bit thermal image every ``--step`` m of GPS (/bestpos) motion,
streamed (no image is kept in memory beyond the current one).

    python extract.py campus_day1.bag runs/thermal/day   --origin 37.xxx 127.xxx
    python extract.py campus_night.bag runs/thermal/night --origin 37.xxx 127.xxx

Writes <out>/images/<k>.png (16-bit), <out>/keyframes.txt (time x y z lat lon sigma), positions in metres east /
north of ``--origin`` (the same origin for every session; default: the bag's first fix).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("bag")
p.add_argument("out")
p.add_argument("--origin", type=float, nargs=2, metavar=("LAT", "LON"))
p.add_argument("--step", type=float, default=2.0)
a = p.parse_args()

out = Path(a.out)
(out / "images").mkdir(parents=True, exist_ok=True)
R = 6378137.0
origin = a.origin
fix = None          # latest GPS fix: (time, x, y, z, lat, lon, sigma)
last = None
rows = []
with AnyReader([Path(a.bag)]) as r:
    con = [c for c in r.connections if c.topic in ("/bestpos", "/thermal/image_raw")]
    for c, _, raw in r.messages(connections=con):
        m = r.deserialize(raw, c.msgtype)
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if c.topic == "/bestpos":
            if origin is None:
                origin = (m.lat, m.lon)
                print(f"origin {origin[0]:.8f} {origin[1]:.8f}")
            x = np.radians(m.lon - origin[1]) * R * np.cos(np.radians(origin[0]))
            y = np.radians(m.lat - origin[0]) * R
            fix = (t, x, y, m.height, m.lat, m.lon, max(m.lat_sigma, m.lon_sigma))
            continue
        if fix is None or abs(t - fix[0]) > 0.2:
            continue
        if last is not None and np.hypot(fix[1] - last[0], fix[2] - last[1]) < a.step:
            continue
        dt = np.uint16 if m.encoding in ("mono16", "16UC1") else np.uint8
        img = np.frombuffer(m.data, dtype=dt).reshape(m.height, m.step // np.dtype(dt).itemsize)[:, : m.width]
        Image.fromarray(img).save(out / "images" / f"{len(rows):06d}.png")
        rows.append([t, *fix[1:4], fix[4], fix[5], fix[6]])
        last = fix[1:3]
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} keyframes ({m.encoding} {m.width}x{m.height})", flush=True)
np.savetxt(out / "keyframes.txt", np.array(rows), fmt="%.9f", header="time x y z lat lon sigma")
print(f"-> {out}: {len(rows)} keyframes every {a.step} m")
