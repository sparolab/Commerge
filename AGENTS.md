# Commerge — setup guide for coding agents

You are setting up and running Commerge (communication-efficient multi-robot LiDAR map merging: it selects the few scans two robots must exchange, registers them and merges the maps) for a user. Take them from a fresh machine to KITTI 00 split into two robots, merged, with the amount of data exchanged and the merge error. Follow the steps in order. Run every command yourself, check its output before moving on, and stop to ask the user only where a step says **Ask**.

## 0. Ask the user first

**Ask**, in a single message:

1. Where their KITTI odometry sequence 00 is (the `sequences/00` folder with `velodyne/`, `poses.txt`, `calib.txt`). If they do not have it, tell them to download it from the KITTI website (registration required): the velodyne scans, poses and calibration.
2. Whether they have a **Gurobi** license (free for academic use: https://www.gurobi.com/academia/academic-program-and-licenses/). Without one, Commerge uses **HiGHS** (shipped with SciPy), which gives the same selection, only slower on the largest regions.
3. **pip** (a Python 3.9–3.12 virtual environment, recommended) or **Docker** (`ghcr.io/sparolab/commerge`).
4. Where to save results (default `~/commerge_results`).

## 1. Check the machine

- `python3 --version` must be 3.9–3.12 (KISS-Matcher has no wheel for 3.13 yet; Commerge still selects scans there but skips the registration).
- Docker path: `docker --version`.
- Free disk space with `df -h`: about 10 GB for the two sessions (keyframe scans) and the outputs.

## 2. Get the code

```bash
git clone --recursive https://github.com/sparolab/Commerge.git    # thirdparty/: SOLiD, KISS-Matcher
cd Commerge
```

Layout: `python/commerge/` is the Python package (`selection.py`: step (A) sequential matching, step (B) balanced minimum vertex cover, step (C) maximum edge-weighted clique; `solvers.py`: Gurobi / HiGHS; `merge.py`: exchange policy, KISS-Matcher + GICP registration, error; `cli.py`: the `commerge` command; `compare.py`: `commerge-compare`; `baselines/automerge.py`). `cpp/` is the same pipeline in C++ (library, `commerge_cpp` command, `commerge_cpp` Python bindings). `docs/GUIDE.md` lists every output and option. `thirdparty/SOLiD` makes the per-robot sessions.

## 3. Install

### pip

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install ".[solid,compare]" rosbags pyyaml     # commerge + solid-descriptor (SOLiD in Python) + AutoMerge baseline
commerge --version
python -c "import kiss_matcher, small_gicp, solid; print('ok')"    # must print: ok
```

With a Gurobi license: `pip install gurobipy`, then `grbgetkey <key>` (or place `gurobi.lic` in the home folder). Check with `python -c "import gurobipy; gurobipy.Model()"`: no license error means Gurobi is used.

### Docker

```bash
docker pull ghcr.io/sparolab/commerge     # or: docker build -f docker/Dockerfile -t ghcr.io/sparolab/commerge .
docker run --rm ghcr.io/sparolab/commerge commerge --version
```

Run the session step (4) with the pip install or in the same container (`-v` the KITTI folder and the results folder, `-w /work`); mount a Gurobi license with `-v $HOME/gurobi.lic:/opt/gurobi/gurobi.lic`.

## 4. One session per robot

Split KITTI 00 into robot A (frames 0–1999) and robot B (2000–4540). Each robot starts its own frame at its first keyframe (`--rebase`), as two real robots would:

```bash
R=~/commerge_results; K=/path/to/sequences/00
cd thirdparty/SOLiD/slam/common/mac
python solid_loops.py --kitti $K --config ../../config/kitti.yaml --out $R/robot_a --range 0:2000 --rebase
python solid_loops.py --kitti $K --config ../../config/kitti.yaml --out $R/robot_b --range 2000:4541 --rebase
# ground truth in one common frame, for the check only (same keyframes, not rebased)
python solid_loops.py --kitti $K --config ../../config/kitti.yaml --out $R/gt_a --range 0:2000
python solid_loops.py --kitti $K --config ../../config/kitti.yaml --out $R/gt_b --range 2000:4541
cd -
```

Each session holds `Scans/`, `SOLiDs/`, `odom_poses.txt` and `times.txt`. Expected: about **1011** keyframes for A and **1701** for B. With a ROS robot, the session comes from SOLiD's `run.launch` with `save_dir:=` instead (see the SOLiD README).

## 5. Run

```bash
commerge $R/robot_a $R/robot_b --threshold 0.96 --min-area 2000 --out $R/merged --plot \
    --gt-a $R/gt_a/odom_poses.txt --gt-b $R/gt_b/odom_poses.txt
```

Add `--solver highs` to force HiGHS. Optional comparison with the ablations and AutoMerge:

```bash
commerge-compare $R/robot_a $R/robot_b --threshold 0.96 --min-area 2000 --out $R/compare \
    --gt-a $R/gt_a/odom_poses.txt --gt-b $R/gt_b/odom_poses.txt
```

## 6. Check the result

- `$R/merged/` must hold `selection.json`, `pairs.csv`, `submap_A.pcd`, `submap_B.pcd`, `T_a_b.txt`, `poses_b_in_a.txt` and the plot.
- Expected on KITTI 00: about **4 regions** after step (A); the exchanged region has a few dozen scans (the paper's run: **14 A + 17 B scans, about 1 % of 2,712 keyframes, 4.6 MB**), and the merge error against ground truth is **below 1 m and 1°** (0.35 m, 0.06° in our run). Small differences come from the KITTI folder versus a bag and from the solver.
- Report to the user: the scans and MB exchanged, the share of keyframes, and the merge error. Show the plot.

## Troubleshooting

- **`registration skipped (ImportError)`**: KISS-Matcher or small_gicp is missing, usually Python 3.13. Use Python 3.12, or `pip install thirdparty/KISS-Matcher/python`.
- **Gurobi license error**: the license is node-locked or expired; run with `--solver highs`.
- **No region after step (A)**: the threshold is too high for these descriptors (e.g. another sensor); lower `--threshold` or `--min-area`. Options in [docs/GUIDE.md](docs/GUIDE.md).
- **Large merge error**: the chosen region is wrong; look at the plot (the affinity has a robot A keyframe per row and a robot B keyframe per column) and at `pairs.csv`.
