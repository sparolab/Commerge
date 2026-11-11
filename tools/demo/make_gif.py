"""The Commerge demo GIF for any two sessions and any sensor: both robots' maps, steps (A), (B), (C) on the
affinity matrix, the keyframes that are exchanged, and the maps merging with the transform registered from them
(KISS-Matcher + GICP on the exchanged scans, or --registration pairs: the selected pairs' positions). Every number
shown comes from the run.

    python tools/demo/make_gif.py <session A> <session B> --threshold 0.96 --min-area 2000 --out demo.gif
        [--affinity A_to_B.npy --affinity-label "affinity (my descriptor)" --label-a "robot A" --label-b "robot B"]
        [--registration pairs --policy all]

The maps are drawn from each keyframe's own sensor points: <session>/points/<k:06d>.npy ((N, 2+) in the keyframe's
sensor frame: radar, sonar, camera depth...) when the folder exists, the LiDAR scans (Scans/) otherwise.

Needs Pillow, matplotlib and ffmpeg (for the GIF).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from commerge import Session, cosine_affinity, select  # noqa: E402
from commerge.merge import choose_regions, register, register_pairs  # noqa: E402
from commerge.selection import extract_regions  # noqa: E402
from commerge.session import read_pcd  # noqa: E402
from commerge.submap import build_submap  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("session_a", type=Path)
ap.add_argument("session_b", type=Path)
ap.add_argument("--out", type=Path, required=True, help="output .gif")
ap.add_argument("--threshold", type=float, required=True)
ap.add_argument("--min-area", type=int, required=True)
ap.add_argument("--affinity", type=Path, help="(Na, Nb) .npy instead of the SOLiD affinity")
ap.add_argument("--affinity-label", default=None)
ap.add_argument("--label-a", default="robot A")
ap.add_argument("--label-b", default="robot B")
ap.add_argument("--intro", default="Two robots, two maps")
ap.add_argument("--poses", default="auto", help="auto | optimized | odom")
ap.add_argument("--registration", choices=("kiss", "pairs"), default="kiss")
ap.add_argument("--policy", default=None, help="exchange policy (default: score for kiss, all for pairs)")
ap.add_argument("--gt-b-moved", action="store_true", help=argparse.SUPPRESS)
ap.add_argument("--point-size", type=int, default=320, help="points drawn per keyframe")
ap.add_argument("--point-alpha", type=float, default=0.9, help="map point opacity (higher = darker)")
ap.add_argument("--point-px", type=int, default=1, help="map point size in pixels (supersampled)")
ap.add_argument("--range", type=float, default=35.0, help="points kept around each keyframe (m)")
ap.add_argument("--frames", type=Path, help="keep the PNG frames here")
ap.add_argument("--only", default="", help="render only these frames, e.g. 40,150 (preview)")
args = ap.parse_args()

TH, AREA = args.threshold, args.min_area
AFF_LABEL = args.affinity_label or ("affinity" if args.affinity else "affinity (SOLiD)")
LAB_A, LAB_B = args.label_a, args.label_b
RMAX = args.range
ONLY = [int(x) for x in args.only.split(",") if x]
OUT = str(args.frames or tempfile.mkdtemp(prefix="commerge_gif_"))
os.makedirs(OUT, exist_ok=True)
W, H, SS = 960, 480, 2
MW = 600                       # map panel width; the affinity panel takes the rest
BG = (255, 255, 255)
INK = (33, 37, 41)
BLUE, TEAL, LIME, CYAN = (63, 81, 181), (0, 137, 123), (46, 204, 64), (0, 172, 193)
MAG = (216, 27, 96)                               # the exchanged scans (logo's "Map" color)


def mix(c0, c1, u):
    return tuple(int(round(a * (1 - u) + b * u)) for a, b in zip(c0, c1))
def _font_file(names):
    for n in names:
        for d in ("/usr/share/fonts", "/Library/Fonts", "/System/Library/Fonts", str(Path.home() / ".fonts")):
            hits = list(Path(d).rglob(n)) if Path(d).is_dir() else []
            if hits:
                return str(hits[0])
    return None


F = _font_file(["Inter-SemiBold.otf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"])
FR = _font_file(["Inter-Regular.otf", "DejaVuSans.ttf", "Arial.ttf"])


def font(s, f=F):
    return ImageFont.truetype(f, s * SS) if f else ImageFont.load_default()

# phases (frames at 20 fps)
P = dict(intro=24, regions=26, cover=26, clique=30, send=40, merge=40, hold=44)
T0 = np.cumsum([0] + list(P.values()))
STARTS = dict(zip(P, T0[:-1]))
TOTAL = int(T0[-1])

A = Session.load(args.session_a, args.poses)
B = Session.load(args.session_b, args.poses)
aff = np.load(args.affinity) if args.affinity else cosine_affinity(A.rsolid, B.rsolid)
sel = select(aff, A.positions, B.positions, TH, AREA, 3.0, steps="ABC", solver="highs", log=lambda *a: None)
boxes = extract_regions(aff, TH, AREA)
cand = np.argwhere(aff >= TH)
in_box = np.zeros(len(cand), bool)
for a0, a1, b0, b1 in boxes:
    in_box |= (cand[:, 0] >= a0) & (cand[:, 0] < a1) & (cand[:, 1] >= b0) & (cand[:, 1] < b1)
cand = cand[in_box]
cover = np.array([(a, b) for r in sel.regions for a, b in cand
                  if a in set(r.cover_a) and b in set(r.cover_b) and r.box[0] <= a < r.box[1] and r.box[2] <= b < r.box[3]])
pairs = np.array(sel.pairs)                       # step (C) pairs of every region
ka_all, kb_all = sel.keyframes_a, sel.keyframes_b
# exchange policy: the region(s) whose keyframes are exchanged and registered
POLICY = args.policy or ("all" if args.registration == "pairs" else "score")
CHS = choose_regions(sel, aff, A.positions, B.positions, POLICY)
CH = CHS[0]
CH_BOXES = {tuple(sel.regions[k].box) for k in CHS}
pairs_sel = np.array([p for k in CHS for p in sel.regions[k].pairs])
ka = sorted({a for k in CHS for a in sel.regions[k].keyframes_a})
kb = sorted({b for k in CHS for b in sel.regions[k].keyframes_b})
PAIRS = args.registration == "pairs"
if PAIRS:
    reg = register_pairs(A.poses, B.poses, [tuple(p) for p in pairs_sel])
else:
    reg = register(build_submap(A, ka, 3.0), build_submap(B, kb, 3.0), voxel=0.5)
if reg is None:
    sys.exit("registration failed: try other --threshold / --min-area")
T_AB = reg.T_a_b
print(f"exchange regions {CHS}: {len(ka)} + {len(kb)} keyframes; T from " + ("the selected pairs" if PAIRS else "KISS-Matcher + GICP"))
n_sent, n_total = len(ka) + len(kb), len(A) + len(B)
if PAIRS:  # only keyframe poses travel
    UNIT, SIZE_U = "keyframes", "KB"
    MB_ALL, MB_SENT = n_total * 96 / 1e3, n_sent * 96 / 1e3
else:
    UNIT, SIZE_U = "scans", "MB"
    mb = lambda s, ks: sum(s.scans[k].stat().st_size for k in ks) / 1e6  # noqa: E731
    MB_ALL = mb(A, range(len(A))) + mb(B, range(len(B)))
    MB_SENT = mb(A, ka) + mb(B, kb)
print(f"data {MB_SENT:.1f} {SIZE_U} of {MB_ALL:.0f} {SIZE_U}")
print(f"regions {len(boxes)}, candidates {len(cand)}, cover pairs {len(cover)}, selected {len(pairs)} pairs, "
      f"{n_sent}/{n_total} {UNIT}")

# maps: a few points per keyframe, in each robot's frame
rng = np.random.default_rng(0)


def local_points(s, root, n, stride):
    out = []
    pdir = Path(root) / "points"
    for k in range(0, len(s), stride):
        f = pdir / f"{k:06d}.npy"
        if f.exists():
            q = np.load(f).astype(float)
            q = np.c_[q[:, :2], q[:, 2] if q.shape[1] > 2 else np.zeros(len(q))]
        else:
            q = read_pcd(s.scans[k])[:, :3]
        q = q[np.linalg.norm(q[:, :2], axis=1) < RMAX]
        if len(q) > n:
            q = q[rng.choice(len(q), n, replace=False)]
        out.append((k, q))
    return out


ptsA = local_points(A, args.session_a, args.point_size, 2)
ptsB = local_points(B, args.session_b, args.point_size, 2)
D = T_AB                                        # B's frame -> A's frame, from the exchanged scans
span = np.ptp(A.positions[:, :2], axis=0)
OFFSET = np.eye(4)
OFFSET[:3, 3] = [span[0] * 1.45, 0, 0]          # B shown next to A before merging
OFFSET[:3, :3] = Rot.from_euler("z", 20, degrees=True).as_matrix() @ D[:3, :3]  # near the final heading
OFFSET[:3, 3] += A.positions[:, :3].mean(0) - OFFSET[:3, :3] @ B.positions.mean(0)


CB = B.positions.mean(0)  # B's map centre (in B's frame): rotate about it, not the origin


def b_transform(u):
    """B's display transform: OFFSET (apart) -> D (merged), u in [0, 1]."""
    R = Slerp([0, 1], Rot.from_matrix([OFFSET[:3, :3], D[:3, :3]]))(u).as_matrix()
    c0 = OFFSET[:3, :3] @ CB + OFFSET[:3, 3]
    c1 = D[:3, :3] @ CB + D[:3, 3]
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = (1 - u) * c0 + u * c1 - R @ CB
    return M


def world(s_poses, pts, M=np.eye(4)):
    out, kf = [], []
    for k, q in pts:
        T = M @ s_poses[k]
        out.append(q @ T[:3, :3].T + T[:3, 3])
        kf.append(np.full(len(q), k))
    return np.concatenate(out), np.concatenate(kf)


# map view fit: everything that is ever shown
allA, kfA = world(A.poses, ptsA)
probe = [allA, world(B.poses, ptsB, b_transform(0))[0], world(B.poses, ptsB, b_transform(1))[0]]
pr = np.concatenate(probe)[:, :2]
lo, hi = np.percentile(pr, 0.5, axis=0), np.percentile(pr, 99.5, axis=0)
CEN0 = (lo + hi) / 2
SCALE0 = min((MW - 40) * SS / (hi[0] - lo[0]), (H - 110) * SS / (hi[1] - lo[1]))
# merged view: the camera moves onto the merged map and zooms in while B slides over
pm = np.concatenate([allA, world(B.poses, ptsB, b_transform(1))[0]])[:, :2]
lo1, hi1 = np.percentile(pm, 0.5, axis=0), np.percentile(pm, 99.5, axis=0)
CEN1 = (lo1 + hi1) / 2
SCALE1 = min((MW - 40) * SS / (hi1[0] - lo1[0]), (H - 110) * SS / (hi1[1] - lo1[1]))
CEN, SCALE = CEN0.copy(), SCALE0


def set_view(u):
    global CEN, SCALE
    CEN = (1 - u) * CEN0 + u * CEN1
    SCALE = SCALE0 * (SCALE1 / SCALE0) ** u          # geometric: a steady zoom


def uv(xy):
    return np.stack([MW * SS / 2 + SCALE * (xy[:, 0] - CEN[0]), (H * SS) / 2 + 22 * SS - SCALE * (xy[:, 1] - CEN[1])], 1)


def splat(img, p, col, a=0.35):
    u, v = p[:, 0].astype(int), p[:, 1].astype(int)
    ok = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
    u, v, col = u[ok], v[ok], col[ok]
    acc = np.zeros(img.shape[:2] + (3,))
    cnt = np.zeros(img.shape[:2])
    np.add.at(cnt, (v, u), 1)
    for c in range(3):
        np.add.at(acc[:, :, c], (v, u), col[:, c])
    m = cnt > 0
    alpha = 1 - np.exp(-a * cnt[m])
    img[m] = img[m] * (1 - alpha[:, None]) + (acc[m] / cnt[m][:, None]) * alpha[:, None]


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def phase(f, name):
    return (f - STARTS[name]) / P[name]


# affinity panel geometry
AX0, AY0 = MW + 30, 92
AW = W - AX0 - 26
AH = min(int(AW * aff.shape[0] / aff.shape[1]), 230)  # room for the counts below
lo_v = min(TH - 0.035, float(np.percentile(aff, 98)))  # show the structure below the threshold too
norm = np.clip((aff - lo_v) / (1 - lo_v), 0, 1) ** 1.5
import matplotlib.cm as cm  # noqa: E402

heat = (cm.magma(norm)[:, :, :3] * 255).astype(np.uint8)
FULL = np.array([0, aff.shape[0], 0, aff.shape[1]], float)
bb = np.array([min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes)], float)
pad_a, pad_b = 0.35 * (bb[1] - bb[0]), 0.35 * (bb[3] - bb[2])
ZOOM = np.array([bb[0] - pad_a, bb[1] + pad_a, bb[2] - pad_b, bb[3] + pad_b])
# keep the panel aspect ratio
want = AH / AW
ha, wb = ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2]
if ha / wb < want:
    c = ZOOM[:2].mean(); ZOOM[:2] = c - want * wb / 2, c + want * wb / 2
else:
    c = ZOOM[2:].mean(); ZOOM[2:] = c - ha / want / 2, c + ha / want / 2
ZOOM = np.clip(ZOOM, [0, 0, 0, 0], [aff.shape[0], aff.shape[0], aff.shape[1], aff.shape[1]])
WIN = FULL.copy()


_lab = Image.new("RGBA", (AH * SS, 14 * SS), (255, 255, 255, 0))
ImageDraw.Draw(_lab).text((0, 0), "robot A keyframes  →", font=font(11, FR), fill=(110, 110, 110))
YLABEL = _lab.rotate(-90, expand=True)                  # reads top to bottom, along the rows


def heat_view():
    a0, a1, b0, b1 = WIN.astype(int)
    return Image.fromarray(heat[a0:max(a1, a0 + 2), b0:max(b1, b0 + 2)]).resize((AW * SS, AH * SS), Image.NEAREST)


def mat_xy(a, b):
    a0, a1, b0, b1 = WIN
    return (AX0 * SS + (b + 0.5 - b0) / (b1 - b0) * AW * SS, AY0 * SS + (a + 0.5 - a0) / (a1 - a0) * AH * SS)


def inside(x, y):
    return AX0 * SS <= x <= (AX0 + AW) * SS and AY0 * SS <= y <= (AY0 + AH) * SS


setA, setB = set(ka), set(kb)                   # exchanged (chosen region)
selA_all, selB_all = set(ka_all), set(kb_all)   # selected in some region
STEP = [("Step (A)  Sequential matching", "regions both robots visited in sequence"),
        ("Step (B)  Balanced min. vertex cover", "fewest keyframes, shared by both robots"),
        ("Step (C)  Max. edge-weighted clique", "only geometrically consistent pairs"),
        (f"Send only these {UNIT}", "one region: the most consistent pairs" if len(CHS) == 1 else
         f"{len(CHS)} regions: consistent pairs, spread out"),
        ("Maps merged", "registered from the exchanged scans only" if not PAIRS else
         f"rigid fit of {reg.inliers} selected pairs' positions")]

for f in (ONLY or range(TOTAL)):
    img = np.full((H * SS, W * SS, 3), 255.0)
    u_merge = ease(phase(f, "merge"))
    lit = ease(phase(f, "clique") * 1.5) if f >= STARTS["clique"] else 0.0
    fade = ease(phase(f, "send") * 2) if f >= STARTS["send"] else 0.0   # other regions fade out
    # --- maps
    Mb = b_transform(u_merge)
    set_view(ease(phase(f, "merge") * 1.15))
    wA, kA = allA, kfA
    wB, kB = world(B.poses, ptsB, Mb)
    for w, k, base, S, SA in ((wA, kA, BLUE, setA, selA_all), (wB, kB, TEAL, setB, selB_all)):
        chosen = np.isin(k, list(S))
        other = np.isin(k, list(SA)) & ~chosen
        col = np.tile(np.array(base, float), (len(w), 1))
        dim = 1 - 0.55 * lit
        col = 255 - (255 - col) * dim
        col[chosen] = (1 - lit) * col[chosen] + lit * np.array(mix(LIME, MAG, fade), float)
        col[other] = (1 - lit * (1 - fade)) * col[other] + lit * (1 - fade) * np.array(LIME, float)
        q = uv(w[:, :2])
        if args.point_px > 1:  # thicker points: copies shifted by up to point_px - 1 pixels
            o = np.array([(i, j) for i in range(args.point_px) for j in range(args.point_px)], float)
            q = (q[None, :, :] + o[:, None, :]).reshape(-1, 2)
            col = np.tile(col, (len(o), 1))
        splat(img, q, col, args.point_alpha)
    pim = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(pim)
    # trajectories
    for poses, M, col in ((A.poses, np.eye(4), BLUE), (B.poses, Mb, TEAL)):
        xy = uv(np.einsum("ij,njk->nik", M, poses)[:, :2, 3])
        d.line([tuple(p) for p in xy], fill=col, width=2 * SS)
    # selected keyframes pulse
    if lit > 0:
        r = (3 + 2 * np.sin(f / 3.0)) * SS
        ring = mix(LIME, MAG, fade)
        for poses, M, S in ((A.poses, np.eye(4), ka if fade > 0.5 else ka_all), (B.poses, Mb, kb if fade > 0.5 else kb_all)):
            xy = uv(np.einsum("ij,njk->nik", M, poses[S])[:, :2, 3])
            for x, y in xy:
                d.ellipse([x - r, y - r, x + r, y + r], outline=ring, width=(2 if fade > 0.5 else 1) * SS)
    # send: packets from B's selected keyframes towards A's matched keyframes
    if STARTS["send"] <= f < STARTS["merge"] + 6:
        u = np.clip(phase(f, "send") * 1.3, 0, 1)
        for i, (a, b) in enumerate(pairs_sel[:: max(1, len(pairs_sel) // 40)]):
            s = np.clip(u * 1.6 - (i % 10) * 0.06, 0, 1)
            if 0 < s < 1:
                pb = uv((Mb @ B.poses[b])[None, :2, 3])[0]
                pa = uv(A.poses[a][None, :2, 3])[0]
                x, y = pb + s * (pa - pb)
                d.ellipse([x - 4 * SS, y - 4 * SS, x + 4 * SS, y + 4 * SS], fill=MAG, outline=(255, 255, 255))
    # --- affinity panel (zooms onto the regions while they are found)
    z = ease(phase(f, "regions") * 1.4)
    WIN[:] = (1 - z) * FULL + z * ZOOM
    pim.paste(heat_view(), (AX0 * SS, AY0 * SS))
    d.rectangle([AX0 * SS, AY0 * SS, (AX0 + AW) * SS, (AY0 + AH) * SS], outline=(200, 200, 200), width=SS)
    d.text((AX0 * SS, (AY0 - 26) * SS), AFF_LABEL, font=font(12, FR), fill=(110, 110, 110))
    d.text((AX0 * SS, (AY0 + AH + 6) * SS), "robot B keyframes  →", font=font(11, FR), fill=(110, 110, 110))
    pim.paste(YLABEL, ((AX0 - 18) * SS, AY0 * SS), YLABEL)
    if f >= STARTS["regions"]:
        u = ease(phase(f, "regions"))
        for bx in boxes:
            a0, a1, b0, b1 = bx
            x0, y0 = mat_xy(a0, b0)
            x1, y1 = mat_xy(a1, b1)
            x0, x1 = np.clip([x0, x1], AX0 * SS, (AX0 + AW) * SS)
            y0, y1 = np.clip([y0, y1], AY0 * SS, (AY0 + AH) * SS)
            is_ch = tuple(bx) in CH_BOXES
            col = mix(CYAN, MAG, fade) if is_ch else tuple(int(c * (1 - 0.6 * fade) + 70 * 0.6 * fade) for c in CYAN)
            d.rectangle([x0, y0, x0 + u * (x1 - x0), y0 + u * (y1 - y0)], outline=col, width=(3 if is_ch and fade > 0 else 2) * SS)
        pts_show = cand if f < STARTS["cover"] + P["cover"] // 2 else cover
        if f >= STARTS["cover"] + P["cover"] // 2 and f < STARTS["clique"]:
            pts_show = cover
        if f < STARTS["clique"]:
            for a, b in pts_show[:: max(1, len(pts_show) // 1500)]:
                x, y = mat_xy(a, b)
                if inside(x, y):
                    d.ellipse([x - SS, y - SS, x + SS, y + SS], fill=(255, 255, 255))
    if f >= STARTS["clique"]:
        sel_set = {tuple(p) for p in pairs_sel.tolist()}
        for a, b in pairs:
            x, y = mat_xy(a, b)
            r = 3 * SS
            if not inside(x, y):
                continue
            if (a, b) in sel_set or fade == 0:
                d.ellipse([x - r, y - r, x + r, y + r], fill=mix(LIME, MAG, fade if (a, b) in sel_set else 0), outline=(255, 255, 255))
            else:  # pairs of the other regions fade to grey
                g = tuple(int(c * (1 - fade) + 110 * fade) for c in LIME)
                d.ellipse([x - r, y - r, x + r, y + r], fill=g)
    # --- funnel: how much each stage keeps
    FY = AY0 + AH + 34
    rows = [("(A) candidate pairs", len(cand), "regions", (190, 190, 200)),
            ("(B) after vertex cover", len(cover), "cover", CYAN),
            ("(C) after clique", len(pairs), "clique", LIME),
            (f"{UNIT} exchanged ({len(CHS)} region{'s' if len(CHS) > 1 else ''})", n_sent, "send", MAG)]
    bw = AW - 10
    for i, (label_, n, ph, col) in enumerate(rows):
        if f < STARTS[ph]:
            continue
        u = ease(phase(f, ph) * 2)
        y = (FY + i * 30) * SS
        frac = np.log1p(n) / np.log1p(max(len(cand), n_total))
        d.rectangle([AX0 * SS, y + 14 * SS, AX0 * SS + u * frac * bw * SS, y + 24 * SS], fill=col)
        tail = f" of {n_total:,}" if ph == "send" else ""
        d.text((AX0 * SS, y), f"{label_}", font=font(11, FR), fill=(90, 90, 90))
        d.text(((AX0 + bw) * SS, y), f"{int(round(u * n)):,}{tail}", font=font(11), fill=INK, anchor="ra")
    # --- titles
    stage = 0 if f < STARTS["cover"] else 1 if f < STARTS["clique"] else 2 if f < STARTS["send"] else 3 if f < STARTS["merge"] + 10 else 4
    if f >= STARTS["regions"]:
        t1, t2 = STEP[stage]
        d.text((AX0 * SS, 16 * SS), t1, font=font(17), fill=INK)
        if t2:
            d.text((AX0 * SS, 40 * SS), t2, font=font(12, FR), fill=(100, 100, 100))
    else:
        d.text((AX0 * SS, 16 * SS), args.intro, font=font(17), fill=INK)
        d.text((AX0 * SS, 40 * SS), f"which {UNIT} should they exchange?", font=font(12, FR), fill=(100, 100, 100))
    # legend + counter on the map
    d.text((18 * SS, 14 * SS), "● " + LAB_A, font=font(13), fill=BLUE)
    xb = 18 * SS + d.textlength("● " + LAB_A, font=font(13)) + 22 * SS
    d.text((xb, 14 * SS), "● " + LAB_B, font=font(13), fill=TEAL)
    if f >= STARTS["send"]:
        a = ease(phase(f, "send") * 2)
        shown = int(round(a * n_sent))
        txt = (f"{shown} of {n_total:,} {UNIT}  ·  {MB_SENT * shown / n_sent:.1f} {SIZE_U} of poses" if PAIRS else
               f"{shown} of {n_total:,} {UNIT}  ·  {MB_SENT * shown / n_sent:.1f} of {MB_ALL:,.0f} {SIZE_U}")
        d.rounded_rectangle([14 * SS, (H - 46) * SS, 360 * SS, (H - 14) * SS], radius=8 * SS, fill=(252, 236, 242))
        d.text((24 * SS, (H - 41) * SS), txt, font=font(14), fill=MAG)
    d.line([MW * SS, 16 * SS, MW * SS, (H - 16) * SS], fill=(228, 228, 228), width=SS)
    pim.resize((W, H), Image.LANCZOS).save(f"{OUT}/{f:04d}.png")
print("frames", TOTAL)
if not ONLY:
    if shutil.which("ffmpeg") is None:
        sys.exit(f"ffmpeg not found; the frames are in {OUT}")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", f"{OUT}/%04d.png", "-vf",
                    "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(args.out)], check=True)
    print(f"-> {args.out}")
    if not args.frames:
        shutil.rmtree(OUT)
