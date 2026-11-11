# Demos: any sensor, any descriptor

Commerge only needs one place descriptor per keyframe and the keyframe poses. Each demo here splits real data into two robots, moves robot B into its own frame (yaw 60°, (30, −20, 0) m), computes the affinity with that sensor's own place-recognition method, and lets Commerge select the keyframes to exchange and merge the maps (`--registration pairs`: from the selected pairs' positions, no point clouds needed).

| Folder | Sensor | Descriptor (submodule) | Data |
| --- | --- | --- | --- |
| [4dradar](4dradar) | 4D radar | SOLiD (`thirdparty/SOLiD` of the repository) | SNAIL-Radar st route, two days |
| [radar](radar) | spinning radar (Navtech) | RadLoc (`radar/thirdparty/RadLoc`) | MulRan KAIST_03 |
| [sonar](sonar) | imaging sonar | Sonar Context (`sonar/thirdparty/sonar_context`) | Aracati 2017 |
| [thermal](thermal) | thermal camera | AnyThermal (`thermal/thirdparty/AnyThermal`) | ViViD++ campus, day vs night |
| [camera](camera) | camera | MegaLoc (`torch.hub`) | KITTI 00 |

## Guide

**Your own sensor.** Compute one descriptor per keyframe with any place-recognition method, save the `(N_A, N_B)` affinity (higher = more similar: cosine similarity, or `1 - distance / max`), write each robot's keyframe poses as `odom_poses.txt` (KITTI format, one row per keyframe), and run:

```bash
commerge robot_a robot_b --affinity affinity.npy --threshold T --min-area N --registration pairs --policy all --out merged/
```

- `--threshold`: around the top 0.1 % of the affinity (`np.percentile(S, 99.9)`); `common.report_affinity` prints, per percentile, the share of pairs that are true revisits when ground truth is at hand.
- `--min-area`: from the keyframe spacing, a few tens of keyframes in sequence (e.g. 50 at 2 m, 300 at KITTI's ~1.5 m with a 1,000 x 1,700 affinity).
- `--registration pairs` fits B's keyframe positions of the selected pairs onto A's (yaw and translation, outlier pairs dropped): no point clouds needed. `--policy all` hands it the pairs of every region, which pins down the rotation; with one short region it is poorly constrained.
- With point clouds on both robots, `--registration kiss` (KISS-Matcher + GICP on the exchanged scans) is more precise; the descriptor can still come from another sensor (e.g. the camera selects, the LiDAR registers).
- The maps of the GIFs (`make_gif.py`) are drawn from `<session>/points/<k>.npy`, each keyframe's points in its sensor frame. Check the axes: the map is sharpest when they are right (`points.py` of each demo documents its frame; a radar spinning clockwise needs y flipped, a sensor whose ground truth is another sensor's pose needs the extrinsic).

## Results

Each demo splits real data into two robots, moves robot B into its own frame (**yaw 60°, (30, −20, 0) m**) and lets Commerge find it back; every map is drawn from that sensor's own points (radar detections, sonar returns, stereo or thermal depth), no LiDAR:

| Sensor | Descriptor | Data | Keyframes exchanged | Rotation error | B's keyframes off | Demo |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 4D radar | [SOLiD](https://github.com/sparolab/SOLiD) | SNAIL-Radar, two days | 1.7 % | 0.31° | 0.57 m | [4dradar](4dradar) |
| Radar (Navtech) | [RadLoc](https://github.com/sparolab/RadLoc) | MulRan KAIST_03 | 1.2 % | 0.05° | 0.45 m | [radar](radar) |
| Imaging sonar | [Sonar Context](https://github.com/sparolab/sonar_context) | Aracati 2017 | 4.5 % | 3.0° | 1.33 m | [sonar](sonar) |
| Thermal camera | [AnyThermal](https://github.com/castacks/AnyThermal) | ViViD++ campus, day vs night | 2.1 % | 0.16° | 1.09 m | [thermal](thermal) |
| Camera | [MegaLoc](https://github.com/gmberton/MegaLoc) | KITTI 00 | 2.1 % | 0.22° | 1.90 m | [camera](camera) |

Keyframes exchanged: the share of both robots' keyframes; with `--registration pairs` only their poses travel (a few KB). Rotation and position errors are against each dataset's ground truth.

## Shared pieces

- `common.py`: ground-truth keyframe poses, the rigid move of robot B, sessions in Commerge's format, a quick check of how well an affinity separates revisits.
- `make_gif.py`: the demo GIF for any two sessions (`--registration pairs`, `--affinity`); the maps are drawn from each keyframe's own sensor points (`<session>/points/<k>.npy`, written by each demo's `points.py`), or from LiDAR scans (`Scans/`).

```bash
git submodule update --init tools/demo/radar/thirdparty/RadLoc tools/demo/sonar/thirdparty/sonar_context tools/demo/thermal/thirdparty/AnyThermal tools/demo/thermal/thirdparty/BridgeMultiSpectralDepth
python tools/demo/make_gif.py runs/radar/robot_a runs/radar/robot_b --affinity runs/radar/affinity.npy \
    --threshold 0.93 --min-area 50 --registration pairs --range 80 --out radar.gif
```
