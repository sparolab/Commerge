<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sparolab/Commerge/main/assets/logo_dark.svg" />
    <img src="https://raw.githubusercontent.com/sparolab/Commerge/main/assets/logo_light.svg" alt="Commerge: Communication-Efficient Map Merging" width="560" />
  </picture>
  <br />
  <a href="https://github.com/sparolab/Commerge/tree/main/commerge"><img src="https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54" /></a>
  <a href="https://pypi.org/project/commerge/"><img src="https://img.shields.io/pypi/v/commerge?logo=pypi&logoColor=white&label=PyPI" /></a>
  <a href="#prerequisites"><img src="https://img.shields.io/badge/Gurobi-EE3524" /></a>
  <a href="#prerequisites"><img src="https://img.shields.io/badge/HiGHS-license--free-2E7D32" /></a>
  <a href="https://github.com/sparolab/SOLiD"><img src="https://img.shields.io/badge/SOLiD-descriptor-3F51B5" /></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu&logoColor=white" /></a>
  <a href="#installation"><img src="https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white" /></a>
  <a href="#docker"><img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" /></a>
  <a href="https://github.com/sparolab/Commerge/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-green" /></a>
  <a href="https://github.com/sparolab/Commerge/actions/workflows/ci.yml"><img src="https://github.com/sparolab/Commerge/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI" /></a>
  <br />
  <a href="https://sparolab.github.io/research/commerge/"><img src="https://img.shields.io/badge/Project-Page-6f42c1" /></a>
  <a href="https://arxiv.org/abs/2606.25386"><img src="https://img.shields.io/badge/arXiv-b33737?logo=arXiv" /></a>
  <a href="https://sparolab.github.io/research/commerge/"><img src="https://img.shields.io/badge/IJRR-2026-1f6feb" /></a>
  <a href="https://youtu.be/LV_8_jZbHV8"><img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white" /></a>
  <a href="#robot-run-with-your-agent"><img src="https://img.shields.io/badge/Claude%20Code-ready-D97757?logo=claude&logoColor=white" alt="Claude Code ready" /></a>
  <a href="#robot-run-with-your-agent"><img src="https://img.shields.io/badge/Codex-ready-000000?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHJlY3QgeD0iMS41IiB5PSIzIiB3aWR0aD0iMjEiIGhlaWdodD0iMTgiIHJ4PSIzIiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiLz48cGF0aCBkPSJNNiA5bDMuNSAzTDYgMTUiIGZpbGw9Im5vbmUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PHBhdGggZD0iTTEyIDE1LjVoNiIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48L3N2Zz4=" alt="Codex ready" /></a>
  <br />
  <a href="https://pypi.org/project/commerge/"><img src="https://readme-typing-svg.demolab.com?background=0D1117&color=22C55E&font=Fira+Code&size=14&duration=2500&pause=800&center=true&vCenter=true&width=230&height=24&lines=%24+pip+install+commerge" alt="pip install commerge" /></a>
  <br />
  <img src="https://raw.githubusercontent.com/sparolab/Commerge/main/assets/commerge_demo.gif" alt="Commerge on KITTI 00 split into two robots: the affinity of their SOLiD descriptors goes through step (A) sequential matching, step (B) balanced minimum vertex cover and step (C) maximum edge-weighted clique; the most consistent region's scans (magenta) are the only ones exchanged, and the transform registered from them merges the two maps." width="95%" />
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

<p align="center"><img src="https://raw.githubusercontent.com/sparolab/Commerge/main/assets/summary.png" alt="Data exchanged after each step on HeLiPR: 7,000 MB for the whole map, 60 MB after step (A), 26 MB after step (B) and 1.3 MB after step (C)" width="95%" /></p>

______________________________________________________________________

## :package: Prerequisites

- **Python 3.9+**. `pip install commerge` brings NumPy, SciPy, Matplotlib and the registration ([KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher), wheels for Python 3.9-3.12, and [small_gicp](https://github.com/koide3/small_gicp)).
- **Solver: [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/), free for academic use**, as in the paper (`pip install gurobipy`, then `grbgetkey <key>`). **No license? HiGHS** ships with SciPy and returns the same selection, only slower on the largest regions (`--solver highs`; `auto`, the default, picks Gurobi when it is licensed).
- **One session per robot** from [SOLiD](https://github.com/sparolab/SOLiD) (`thirdparty/SOLiD`): keyframe scans, poses and SOLiD descriptors.

## :hammer_and_wrench: Installation

```bash
pip install commerge          # + "commerge[gurobi]" for gurobipy, "commerge[compare]" for the AutoMerge comparison
# or from a clone, with SOLiD and KISS-Matcher as submodules:
git clone --recursive https://github.com/sparolab/Commerge.git && cd Commerge && pip install .
# KISS-Matcher from the submodule instead of the wheel: pip install thirdparty/KISS-Matcher/python
commerge --version
```

Ubuntu and macOS (Python only, no ROS needed for Commerge itself).

### Docker

```bash
docker pull ghcr.io/sparolab/commerge     # or: docker build -f docker/Dockerfile -t ghcr.io/sparolab/commerge .
docker run --rm -v $HOME/robot_a:/a -v $HOME/robot_b:/b -v $PWD/merged:/out ghcr.io/sparolab/commerge \
    commerge /a /b --threshold 0.96 --min-area 2000 --out /out --plot
# Gurobi: add -v $HOME/gurobi.lic:/opt/gurobi/gurobi.lic (otherwise HiGHS)
```

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

All outputs, options and the Python API: [docs/GUIDE.md](https://github.com/sparolab/Commerge/blob/main/docs/GUIDE.md).

## :bar_chart: Compare

`commerge-compare` runs Commerge, its ablations and AutoMerge on the same sessions. On KITTI 00 split into two robots, Commerge sends **31 of 2,712 scans (4.6 MB)** and merges the maps within **0.35 m**; AutoMerge's region sends 156 scans and merges within 1.05 m. Details and the ablations: [docs/GUIDE.md](https://github.com/sparolab/Commerge/blob/main/docs/GUIDE.md#compare-commerge-compare).

______________________________________________________________________

## :robot: Run with your agent

Open [Claude Code](https://claude.com/claude-code) or [Codex](https://openai.com/codex/) in an empty folder and paste:

```text
Clone https://github.com/sparolab/Commerge, read its AGENTS.md, and follow it to install Commerge and merge KITTI 00 split into two robots on this machine. Ask me what you need along the way.
```

The agent follows [`AGENTS.md`](https://github.com/sparolab/Commerge/blob/main/AGENTS.md): it installs Commerge with Gurobi or HiGHS, makes one SOLiD session per robot, selects and exchanges the scans, merges the maps, and checks the result.

______________________________________________________________________

## :warning: Limitation: heterogeneous LiDARs

Commerge selects scans from the affinity of the two robots' descriptors. With the same LiDAR on both robots, SOLiD gives an affinity whose overlapping regions stand out. With different LiDARs (a spinning Ouster and a Livox Avia with a 70° field of view, for instance), SOLiD describes the same place differently and step (A) finds hundreds of regions with no structure. A descriptor made for heterogeneous LiDARs, such as [HeLiOS](https://github.com/minwoo0611/HeLiOS) or [InLiER](https://github.com/LTU-RAI/InLiER), passed with `--affinity` (and `--enhance`), brings the right region back; registering the few scans exchanged across such different fields of view is still unreliable (`--registration grid` is experimental). Contributions are welcome.

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

We thank [Minwoo Jung](https://github.com/minwoo0611) for the [HeLiPR](https://sites.google.com/view/heliprdataset) dataset, and [Nikolaos Stathoulopoulos](https://github.com/nstathou) for [InLiER](https://github.com/LTU-RAI/InLiER), learning-free heterogeneous LiDAR place recognition.
