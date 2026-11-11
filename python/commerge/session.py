"""A robot's run as an LT-mapper session, as written by SOLiD (``save_dir`` / ``--out``):

    <session>/
    ├── Scans/000000.pcd        keyframe cloud in the sensor frame (x y z intensity)
    ├── SOLiDs/000000.txt       SOLiD descriptor (line 1: R-SOLiD, line 2: A-SOLiD)
    ├── optimized_poses.txt     keyframe poses after loop closing (KITTI, 3x4 per row)
    └── odom_poses.txt          keyframe poses from the odometry (KITTI)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


def read_kitti_poses(path: Path) -> np.ndarray:
    """(N, 4, 4) poses from a KITTI file (12 values per row, spaces or commas)."""
    text = Path(path).read_text().replace(",", " ")
    a = np.array(text.split(), dtype=float).reshape(-1, 12)
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :4] = a.reshape(-1, 3, 4)
    return T


def read_pcd(path: Path) -> np.ndarray:
    """(N, 4) x y z intensity from a PCD file (ascii, binary or binary_compressed)."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            key, *vals = line.split()
            header[key.upper()] = vals
            if key.upper() == "DATA":
                break
        data = f.read()
    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    n = int(header["POINTS"][0])
    kind = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
            ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}
    dtype = np.dtype([(f, kind[(t, s)], (c,)) if c > 1 else (f, kind[(t, s)])
                      for f, s, t, c in zip(fields, sizes, types, counts)])
    mode = header["DATA"][0].lower()
    if mode == "ascii":
        cols = np.loadtxt(data.decode().splitlines(), ndmin=2)
        rec = np.zeros(n, dtype=dtype)
        for k, f in enumerate(fields[: cols.shape[1]]):
            rec[f] = cols[:, k]
    elif mode == "binary":
        rec = np.frombuffer(data[: n * dtype.itemsize], dtype=dtype)
    elif mode == "binary_compressed":
        rec = _decompress_pcd(data, dtype, n)
    else:
        raise ValueError(f"{path}: unsupported PCD DATA {mode}")
    out = np.zeros((n, 4), dtype=np.float32)
    names = rec.dtype.names or ()
    for k, f in enumerate(("x", "y", "z", "intensity")):
        if f in names:
            out[:, k] = rec[f]
    return out


def _decompress_pcd(data: bytes, dtype: np.dtype, n: int) -> np.ndarray:
    import struct

    comp, uncomp = struct.unpack("<II", data[:8])
    raw = _lzf_decompress(data[8: 8 + comp], uncomp)
    rec = np.zeros(n, dtype=dtype)
    off = 0
    for name in dtype.names or ():  # binary_compressed stores each field contiguously
        dt = dtype[name]
        size = dt.itemsize * n
        rec[name] = np.frombuffer(raw[off: off + size], dtype=dt).reshape(rec[name].shape)
        off += size
    return rec


def _lzf_decompress(src: bytes, out_len: int) -> bytes:
    out = bytearray()
    i = 0
    while i < len(src):
        ctrl = src[i]
        i += 1
        if ctrl < 32:  # literal run
            out += src[i: i + ctrl + 1]
            i += ctrl + 1
        else:  # back reference
            length = ctrl >> 5
            ref = len(out) - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += src[i]
                i += 1
            ref -= src[i]
            i += 1
            for _ in range(length + 2):
                out.append(out[ref])
                ref += 1
    if len(out) != out_len:
        raise ValueError("corrupt binary_compressed PCD")
    return bytes(out)


def write_pcd(path: Path, xyzi: np.ndarray) -> None:
    """Binary PCD with x y z intensity (float32)."""
    xyzi = np.ascontiguousarray(xyzi[:, :4], dtype=np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n"
        "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {len(xyzi)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyzi)}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(xyzi.tobytes())


@dataclass
class Session:
    """One robot's keyframes: poses (N, 4, 4), scan files and SOLiD descriptors."""

    name: str
    poses: np.ndarray
    scans: List[Path] = field(default_factory=list)
    rsolid: Optional[np.ndarray] = None  # (N, D) R-SOLiD
    asolid: Optional[np.ndarray] = None  # (N, D) A-SOLiD

    @classmethod
    def load(cls, root: Path, poses: str = "auto", name: Optional[str] = None) -> "Session":
        """``poses``: "auto" (optimized if present, else odometry), "optimized", "odom" or a file."""
        root = Path(root)
        if poses == "auto":
            p = root / "optimized_poses.txt"
            p = p if p.exists() else root / "odom_poses.txt"
        elif poses in ("optimized", "odom"):
            p = root / f"{poses}_poses.txt"
        else:
            p = Path(poses)
        T = read_kitti_poses(p)
        scans = sorted((root / "Scans").glob("*.pcd"))
        rs, as_ = None, None
        sdir = root / "SOLiDs"
        if sdir.is_dir():
            files = sorted(sdir.glob("*.txt"))
            lines = [f.read_text().splitlines() for f in files]
            rs = np.array([np.array(l[0].split(), dtype=float) for l in lines])
            as_ = np.array([np.array(l[1].split(), dtype=float) for l in lines if len(l) > 1]) if lines else None
        s = cls(name or root.name, T, scans, rs, as_)
        if s.scans and len(s.scans) != len(s.poses):
            raise ValueError(f"{root}: {len(s.scans)} scans but {len(s.poses)} poses")
        return s

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def positions(self) -> np.ndarray:
        return self.poses[:, :3, 3]

    def cloud_in_world(self, k: int, min_range: float = 0.0) -> np.ndarray:
        """Keyframe ``k`` (N, 4) in the session frame, dropping points closer than ``min_range``."""
        pts = read_pcd(self.scans[k])
        if min_range > 0:
            pts = pts[np.linalg.norm(pts[:, :3], axis=1) > min_range]
        T = self.poses[k]
        out = pts.copy()
        out[:, :3] = pts[:, :3] @ T[:3, :3].T + T[:3, 3]
        return out
