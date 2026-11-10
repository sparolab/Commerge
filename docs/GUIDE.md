# Commerge guide

Everything the [README](../README.md) leaves out: outputs, options, the Python API and the comparison.

## Outputs and options of `commerge`

| Output | |
| --- | --- |
| `selection.json` | per region: candidate pairs of step (A), cover of step (B), pairs of step (C); the exchanged region and keyframes; registration (and its error, with ground truth) |
| `submap_A.pcd`, `submap_B.pcd` | the exchanged scans of each robot, in its own frame |
| `T_a_b.txt`, `poses_b_in_a.txt` | robot B's frame in robot A's frame (4x4), and B's keyframes in A's frame (KITTI): the merged map |
| `pairs.csv` | selected keyframe pairs of every region (region, keyframe of A, keyframe of B, affinity) |
| `affinity.npy`, `selection.png` | the SOLiD affinity matrix; with `--plot`, the regions and selected pairs on it |

| Option | Default | |
| --- | --- | --- |
| `--threshold`, `--min-area` | required | step (A): affinity of a candidate pair, smallest region (cells of its bounding box) |
| `--consistency` | `3.0` | step (C): largest distance mismatch (m) between two consistent pairs |
| `--steps` | `ABC` | any subset, for ablations: without (A) the whole matrix is one region, without (B) every candidate goes on, without (C) every covered pair is kept |
| `--solver` | `auto` | `gurobi`, `highs` or `auto` |
| `--policy` | `score` | region(s) whose scans are exchanged: `score` (largest step-(C) clique weight), `area`, `candidates` or `all` |
| `--no-register`, `--reg-voxel` | `0.5` | skip the registration; its voxel size (m) |
| `--gt-a`, `--gt-b` | | ground-truth keyframe poses (KITTI, common frame) to report the merge error |
| `--affinity` | SOLiD | any `(N_A, N_B)` `.npy` instead (another descriptor) |
| `--poses` | `auto` | `optimized` (loop-closed) or `odom` keyframe poses |
| `--min-range`, `--voxel` | `3.0`, `0` | submap: drop points near the sensor, optional voxel size |

## Python

```python
from commerge import Session, choose_regions, cosine_affinity, select
A, B = Session.load("robot_a"), Session.load("robot_b")
aff = cosine_affinity(A.rsolid, B.rsolid)
sel = select(aff, A.positions, B.positions, threshold=0.96, min_area=2000, steps="ABC")
region = choose_regions(sel, aff, A.positions, B.positions)[0]       # the region exchanged
print(sel.regions[region].keyframes_a, sel.regions[region].keyframes_b)
```

## Compare: `commerge-compare`

`commerge-compare` runs Commerge, its ablations and AutoMerge (our re-implementation of its SeqSLAM-based region selection, see [`commerge/baselines`](../commerge/baselines)) on the same pair of sessions:

```bash
commerge-compare $HOME/robot_a $HOME/robot_b --threshold 0.96 --min-area 2000 --out compare/ [--gt-a gt_a.txt --gt-b gt_b.txt]
```

On KITTI 00 split into two robots (frames 0-1999 and 2000-4540, 2,712 keyframes, SOLiD affinity, threshold 0.96, min. area 2000). Each method sends the scans of one region, which are registered with KISS-Matcher + GICP; a pair is correct within 5 m in ground truth:

| Method | Scans sent (A + B) | Share of keyframes | Data | Correct pairs | Merge error | Time |
| --- | --- | --- | --- | --- | --- | --- |
| **Commerge, steps (A)+(B)+(C)** | **14 + 17** | **1.1 %** | **4.6 MB** | **100 %** | **0.35 m, 0.06°** | 1.0 s |
| w/o step (B) | 83 + 110 | 7.1 % | 30.7 MB | 100 % | 0.26 m, 0.02° | 94 s (solver time limit) |
| w/o step (C) | 23 + 41 | 2.4 % | 9.4 MB | 47 % | 0.32 m, 0.06° | 0.6 s |
| step (A) only | 132 + 129 | 9.6 % | 41.1 MB | 58 % | 0.37 m, 0.07° | 0.02 s |
| AutoMerge (region selection) | 79 + 77 | 5.8 % | 22.3 MB | – | 1.05 m, 0.06° | 17 s |
