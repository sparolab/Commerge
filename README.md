<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg" />
    <img src="assets/logo_light.svg" alt="Commerge: Communication-Efficient Map Merging" width="560" />
  </picture>
  <br />
  <a href="https://github.com/sparolab/Commerge/tree/main/commerge"><img src="https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54" /></a>
  <a href="#prerequisites"><img src="https://img.shields.io/badge/Gurobi-EE3524" /></a>
  <a href="#prerequisites"><img src="https://img.shields.io/badge/HiGHS-license--free-2E7D32" /></a>
  <a href="https://github.com/sparolab/SOLiD"><img src="https://img.shields.io/badge/SOLiD-descriptor-3F51B5" /></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu&logoColor=white" /></a>
  <a href="#installation"><img src="https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-green" /></a>
  <a href="https://github.com/sparolab/Commerge/actions/workflows/ci.yml"><img src="https://github.com/sparolab/Commerge/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI" /></a>
  <br />
  <a href="https://sparolab.github.io/research/commerge/"><img src="https://img.shields.io/badge/Project-Page-6f42c1" /></a>
  <a href="https://arxiv.org/abs/2606.25386"><img src="https://img.shields.io/badge/arXiv-b33737?logo=arXiv" /></a>
  <a href="https://sparolab.github.io/research/commerge/"><img src="https://img.shields.io/badge/IJRR-2026-1f6feb" /></a>
  <a href="https://youtu.be/LV_8_jZbHV8"><img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white" /></a>
  <br />
  <img src="https://readme-typing-svg.demolab.com?background=0D1117&color=22C55E&font=Fira+Code&size=14&duration=4500&pause=1000&center=true&vCenter=true&width=680&height=24&lines=Only+a+small+subset+of+carefully+selected+scans+is+sufficient+for+map+merging%21" alt="Only a small subset of carefully selected scans is sufficient for map merging!" />
  <br />
  <img src="assets/commerge_demo.gif" alt="Commerge on KITTI 00 split into two robots: the affinity of their SOLiD descriptors goes through step (A) sequential matching, step (B) balanced minimum vertex cover and step (C) maximum edge-weighted clique; the most consistent region's scans (magenta) are the only ones exchanged, and the transform registered from them merges the two maps." width="95%" />
  <p>
    <strong>[IJRR 2026] Commerge: Communication-Efficient, Robust, and Fast LiDAR Map Merging Framework<br />
    for Multi-Robot Coordination in Resource-Constrained Scenarios</strong>
  </p>
</div>

______________________________________________________________________

Robots that map on their own must exchange scans to merge their maps, and over a weak link that exchange is the bottleneck. Commerge picks the few scans that carry the overlap and sends only those, in three steps:

- :robot: **Before the steps, each robot runs its own SLAM with [SOLiD](https://github.com/sparolab/SOLiD)**: keyframe scans, poses and SOLiD descriptors, saved as a session.
- :link: **Step (A) Sequential matching**: regions of the inter-robot affinity matrix where both robots saw the same places in sequence.
- :balance_scale: **Step (B) Balanced minimum vertex cover**: the fewest keyframes that still touch every candidate match, shared fairly between the robots.
- :gem: **Step (C) Maximum edge-weighted clique**: among those, only the pairs that agree with each other geometrically.
- :handshake: **Exchange and merge**: the robots exchange the scans of the most consistent region only; their submaps are registered ([KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher) + GICP) and the transform merges the whole maps.
- :package: **MB, not GB**: up to 99.98 % less data exchanged (7,000 MB → 1.3 MB on HeLiPR) with comparable alignment, also on a Jetson Nano.

<p align="center"><img src="assets/summary.png" alt="Data exchanged after each step on HeLiPR: 7,000 MB for the whole map, 60 MB after step (A), 26 MB after step (B) and 1.3 MB after step (C)" width="95%" /></p>

______________________________________________________________________

## :package: Prerequisites

| | |
| --- | --- |
| Python | 3.9 or newer, with NumPy, SciPy and Matplotlib (installed with the package) |
| Optimization solver | **[Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/)**, as in the paper: **free for academic use** (named academic license, `pip install gurobipy`, then `grbgetkey <key>`). <br/> **No Gurobi license?** Use **HiGHS**: it ships with SciPy, so there is nothing to install. It returns the same selection, only slower on the largest regions. |
| Keyframe sessions | one per robot, from [SOLiD](https://github.com/sparolab/SOLiD) (included as `thirdparty/SOLiD`): keyframe scans, poses and SOLiD descriptors |
| Registration | [KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher) (included as `thirdparty/KISS-Matcher`) and [small_gicp](https://github.com/koide3/small_gicp), installed with the package (`kiss-matcher`, `small-gicp` wheels) |

The solver is picked with `--solver`: `auto` (default: Gurobi if it is installed and licensed, HiGHS otherwise), `gurobi` or `highs`.

## :hammer_and_wrench: Installation

```bash
git clone --recursive https://github.com/sparolab/Commerge.git
cd Commerge
pip install .                 # + ".[gurobi]" for gurobipy, ".[compare]" for the AutoMerge comparison
# KISS-Matcher from the submodule instead of the wheel: pip install thirdparty/KISS-Matcher/python
commerge --version
```

Ubuntu and macOS (Python only, no ROS needed for Commerge itself).

## :rocket: Run

**1. One session per robot** with SOLiD, with any odometry it supports ([FAST-LIO, KISS-ICP, GeoRIO, …](https://github.com/sparolab/SOLiD/tree/main/slam)) or without ROS:

```bash
cd thirdparty/SOLiD
roslaunch solid_ros run.launch odometry:=fast_lio save_dir:=$HOME/robot_a     # ROS 1, while the bag of robot A plays
ros2 launch solid_ros run.launch.py odometry:=kiss_icp save_dir:=$HOME/robot_b  # ROS 2
python slam/common/mac/solid_loops.py --kitti <sequence> --config slam/config/kitti.yaml --out $HOME/robot_a  # no ROS
```

**2. Select the scans to exchange, register them and merge:**

```bash
commerge $HOME/robot_a $HOME/robot_b --threshold 0.96 --min-area 2000 --out merged/ --plot [--gt-a gt_a.txt --gt-b gt_b.txt]
```

```
[commerge] kitti00_a: 1011 keyframes | kitti00_b: 1701 keyframes
[commerge] steps (A)+(B)+(C), solver gurobi: (A) 4 region(s) (threshold 0.96, min area 2000) in 0.021 s
[commerge] region 2: A 250-381 x B 899-1027: (A) 1077 candidate pairs -> (B) 66+66 keyframes -> (C) 24 pairs = 14 A + 17 B scans, (B) 0.34 s, (C) 0.13 s
...
[commerge] registered B to A: 540 KISS-Matcher inliers, GICP in 1.3 s; vs ground truth: 0.35 m, 0.06 deg (B keyframes 0.15 m off on average)
[commerge] exchanged 14 A + 17 B scans (1.1 % of 2712, 4.6 MB) from region(s) [2] of 4 -> merged/
```

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

From Python:

```python
from commerge import Session, choose_regions, cosine_affinity, select
A, B = Session.load("robot_a"), Session.load("robot_b")
aff = cosine_affinity(A.rsolid, B.rsolid)
sel = select(aff, A.positions, B.positions, threshold=0.96, min_area=2000, steps="ABC")
region = choose_regions(sel, aff, A.positions, B.positions)[0]       # the region exchanged
print(sel.regions[region].keyframes_a, sel.regions[region].keyframes_b)
```

## :bar_chart: Compare

`commerge-compare` runs Commerge, its ablations and AutoMerge (our re-implementation of its SeqSLAM-based region selection, see [`commerge/baselines`](commerge/baselines)) on the same pair of sessions:

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

______________________________________________________________________

## Citation

If you use Commerge in your research, please cite our [paper](https://arxiv.org/abs/2606.25386).

<details>
  <summary><strong>See bibtex</strong></summary>

```bibtex
@article{kim2026commerge,
  author  = {Kim, Hogyun and Choi, Jiwon and Kim, Juwon and Yang, Geonmo and Jeong, Seokhwan and Lim, Hyungtae and Cho, Younggun},
  title   = {Commerge: Communication-Efficient, Robust, and Fast LiDAR Map Merging Framework for Multi-Robot Coordination in Resource-Constrained Scenarios},
  journal = {International Journal of Robotics Research (IJRR)},
  year    = {2026}
}
```

</details>

## Contributing

Issues and pull requests are welcome.

<a href="https://github.com/sparolab/Commerge/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=sparolab/Commerge" />
</a>

## Acknowledgements

Commerge uses [SOLiD](https://github.com/sparolab/SOLiD) for inter-robot place recognition. The comparison re-implements the region selection of [AutoMerge](https://github.com/MetaSLAM/AutoMerge_Server), which builds on [SeqSLAM](https://doi.org/10.1109/ICRA.2012.6224623). Related research: [KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher), [SKiD-SLAM](https://github.com/sparolab/SKiD-SLAM).

## Special Thanks

We thank [Minwoo Jung](https://github.com/minwoo0611).
