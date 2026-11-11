"""Imaging sonar with SonarContext: two robots (sessions) from Aracati 2017, one of them in its own frame.

    python make_sessions.py --images polar_image.zip --gt pose_gt.txt --out runs/sonar
    commerge runs/sonar/robot_a runs/sonar/robot_b --affinity runs/sonar/affinity.npy --threshold 0.9 \
        --min-area 50 --registration pairs --policy all --out runs/sonar/merged \
        --gt-a runs/sonar/robot_a/gt_poses.txt --gt-b runs/sonar/robot_b/gt_poses.txt

Robot A is the first half of the run, robot B the second half. Keyframes every 1 m. Descriptor: Sonar Context
(thirdparty/sonar_context: max over 4 x 4 patches of the polar image), compared as in its col_distance_sc (mean
column cosine similarity, best column shift), vectorized here and checked against the original on a few pairs.
``--gt``: the /pose_gt topic of the Aracati bag as text (stamp bagtime x y z qx qy qz qw).
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE / "thirdparty/sonar_context/place_recognition/utils")]
from common import interpolate_poses, keyframes_by_distance, report_affinity, rigid, write_session  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def sonar_context(img: np.ndarray, patch=(4, 4)) -> np.ndarray:
    """img2scpk.ptcloud2sc: the max of each patch; ``img`` with rows = range bins, columns = beams, so that the
    column shift of the matching is a rotation (yaw)."""
    r, c = img.shape[0] // patch[0], img.shape[1] // patch[1]
    return img[: r * patch[0], : c * patch[1]].reshape(r, patch[0], c, patch[1]).max(axis=(1, 3)).astype(float)


def column_similarity(A: np.ndarray, B: np.ndarray, bound: float = 0.5, chunk: int = 16) -> np.ndarray:
    """(Na, Nb) best mean column cosine similarity over column shifts up to ``bound`` of the width (both ways)."""
    def unit(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return np.divide(X, n, out=np.zeros_like(X), where=n > 0), (n[:, 0] > 0)
    ua, va = unit(A)  # (Na, rows, cols): columns normalized
    ub, vb = unit(B)
    cols = A.shape[2]
    shifts = range(1, int(cols * bound) + 1)
    S = np.full((len(A), len(B)), -1.0)
    for i0 in range(0, len(A), chunk):
        X = ua[i0:i0 + chunk].transpose(0, 2, 1).reshape(-1, ua.shape[1])         # (a*cols, rows)
        Y = ub.transpose(1, 0, 2).reshape(ub.shape[1], -1)                       # (rows, b*cols)
        C = (X @ Y).reshape(-1, cols, len(B), cols).transpose(0, 2, 1, 3)        # (a, b, cols, cols) cosines
        V = va[i0:i0 + chunk, None, :, None] & vb[None, :, None, :]
        best = np.full(C.shape[:2], -1.0)
        for s in shifts:
            for sign in (1, -1):  # A's column j - s (right shift) or j + s (left shift) against B's column j
                j = np.arange(s, cols) if sign > 0 else np.arange(0, cols - s)
                d = C[:, :, j - sign * s, j]
                v = V[:, :, j - sign * s, j]
                m = (d * v).sum(-1) / np.maximum(v.sum(-1), 1)
                best = np.maximum(best, m)
        S[i0:i0 + chunk] = best
    return S


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--images", required=True, help="Aracati polar_image.zip (or a folder of <stamp>.png)")
p.add_argument("--gt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--step", type=float, default=1.0, help="keyframe spacing (m)")
p.add_argument("--split", type=float, default=0.5, help="share of the keyframes in robot A's session")
p.add_argument("--yaw", type=float, default=60.0)
p.add_argument("--shift", type=float, nargs=3, default=[30.0, -20.0, 0.0])
a = p.parse_args()

src = Path(a.images)
if src.suffix == ".zip":
    z = zipfile.ZipFile(src)
    names = sorted((n for n in z.namelist() if n.endswith(".png")), key=lambda n: float(Path(n).stem))
    load = lambda n: Image.open(io.BytesIO(z.read(n)))
else:
    names = sorted((str(f) for f in src.glob("*.png")), key=lambda n: float(Path(n).stem))
    load = Image.open
t = np.array([float(Path(n).stem) for n in names])
g = np.loadtxt(a.gt)
G = np.tile(np.eye(4), (len(g), 1, 1))
G[:, :3, 3] = g[:, 2:5]
G[:, :3, :3] = Rotation.from_quat(g[:, 5:9]).as_matrix()
T = interpolate_poses(t, g[:, 0], G)
k = keyframes_by_distance(T, a.step)
print(f"{len(names)} sonar images, {len(k)} keyframes every {a.step} m")
cache = Path(a.out) / "sonar_context_range_beam.npy"
if cache.exists():
    SC = np.load(cache)
else:
    # Aracati's polar images are (beams, range bins): transposed to (range, beams)
    SC = np.array([sonar_context(np.asarray(load(names[i]).convert("L")).T) for i in k])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, SC)
s = int(len(k) * a.split)
ka, kb = k[:s], k[s:]
S = column_similarity(SC[:s], SC[s:])

try:  # against the original, on a few pairs
    from SonarContextManager import col_distance_sc
    for i, j in [(0, 0), (5, 17), (s // 2, (len(k) - s) // 3)]:
        print(f"  pair ({i}, {j}): sonar_context {1 - col_distance_sc(SC[i].copy(), SC[s + j].copy()):.4f}, here {S[i, j]:.4f}")
except ImportError:
    pass

out = Path(a.out)
write_session(out / "robot_a", T[ka], t[ka], [Path(names[i]).name for i in ka])
write_session(out / "robot_b", T[kb], t[kb], [Path(names[i]).name for i in kb], move=rigid(a.yaw, a.shift))
np.save(out / "affinity.npy", S.astype(np.float32))
report_affinity(S, T[ka], T[kb], near=3.0)
print(f"-> {out}: robot_a {len(ka)}, robot_b {len(kb)} keyframes (B moved by yaw {a.yaw} deg, {a.shift} m)")
