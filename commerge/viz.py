"""Plot of the affinity matrix with the regions and the selected pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_selection(affinity: np.ndarray, selection, path: Path, threshold: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7 * affinity.shape[0] / max(affinity.shape[1], 1) + 0.6))
    vmin = threshold - 0.1 if threshold is not None else None
    im = ax.imshow(affinity, cmap="magma", vmin=vmin, interpolation="nearest")
    for k, r in enumerate(selection.regions):
        a0, a1, b0, b1 = r.box
        ax.add_patch(plt.Rectangle((b0, a0), b1 - b0, a1 - a0, edgecolor="cyan", facecolor="none", lw=1.2))
        ax.text(b0, a0, str(k), color="cyan", fontsize=8, va="bottom")
        if r.pairs:
            p = np.array(r.pairs)
            ax.scatter(p[:, 1], p[:, 0], s=6, c="lime", marker="o", linewidths=0)
    ax.set_xlabel("robot B keyframe")
    ax.set_ylabel("robot A keyframe")
    ax.set_title(f"{len(selection.keyframes_a)} + {len(selection.keyframes_b)} scans selected "
                 f"(steps {'+'.join(f'({s})' for s in selection.steps) or 'none'})")
    fig.colorbar(im, ax=ax, fraction=0.04, label="affinity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(affinity: np.ndarray, selections: dict, path: Path, threshold: float | None = None,
                    rows: list | None = None) -> None:
    """Top: each method's selection on the affinity matrix, zoomed onto the regions.
    Bottom: data each method exchanges (and the share of correct pairs, if known)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(selections)
    fig = plt.figure(figsize=(3.3 * n, 4.9))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.0], hspace=0.3, wspace=0.08)
    th = threshold if threshold is not None else float(np.percentile(affinity, 99.5))
    shown = np.clip((affinity - (th - 0.035)) / 0.035 + 0.0, 0, None)
    shown = np.clip(shown / max(shown.max(), 1e-9), 0, 1) ** 1.5
    boxes = [r.box for sel in selections.values() for r in sel.regions]
    if boxes:
        a0, a1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
        b0, b1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
        pa, pb = 0.15 * (a1 - a0) + 10, 0.15 * (b1 - b0) + 10
        win = (max(0, a0 - pa), min(affinity.shape[0], a1 + pa), max(0, b0 - pb), min(affinity.shape[1], b1 + pb))
    else:
        win = (0, affinity.shape[0], 0, affinity.shape[1])
    total = sum(affinity.shape)
    green, cyan = "#2ecc40", "#00acc1"
    for k, (name, sel) in enumerate(selections.items()):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(shown, cmap="magma", interpolation="nearest")
        for r in sel.regions:
            ra0, ra1, rb0, rb1 = r.box
            ax.add_patch(plt.Rectangle((rb0, ra0), rb1 - rb0, ra1 - ra0, edgecolor=cyan, facecolor="none", lw=1.2))
            if r.pairs:
                p = np.array(r.pairs)
                ax.scatter(p[:, 1], p[:, 0], s=9, c=green, linewidths=0.3, edgecolors="white")
            elif r.scans_a is not None:
                ax.add_patch(plt.Rectangle((rb0, ra0), rb1 - rb0, ra1 - ra0, color=green, alpha=0.35, lw=0))
        ax.set_xlim(win[2], win[3])
        ax.set_ylim(win[1], win[0])
        sent = len(sel.keyframes_a) + len(sel.keyframes_b)
        ax.set_title(f"{name}\n{sent} scans ({100.0 * sent / total:.1f} %)", fontsize=11,
                     fontweight="bold" if k == 0 else "normal")
        ax.set_xticks([])
        ax.set_yticks([])
    if rows:
        ax = fig.add_subplot(gs[1, :])
        names = [r["method"] for r in rows]
        mb = [r["megabytes"] for r in rows]
        cols = ["#3F51B5" if i == 0 else "#b0b7c3" for i in range(len(rows))]
        bars = ax.bar(names, mb, color=cols, width=0.55)
        for b, r in zip(bars, rows):
            lab = f"{r['megabytes']:.1f} MB"
            if "pair_precision" in r:
                lab += f"\n{100 * r['pair_precision']:.0f} % correct pairs"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), lab, ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("data exchanged (MB)")
        ax.set_ylim(0, max(mb) * 1.35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
