# Demos: any sensor, any descriptor

Commerge only needs one place descriptor per keyframe and the keyframe poses. Each demo here splits real data into two robots, moves robot B into its own frame (yaw 60°, (30, −20, 0) m), computes the affinity with that sensor's own place-recognition method, and lets Commerge select the keyframes to exchange and merge the maps (`--registration pairs`: from the selected pairs' positions, no point clouds needed).

| Folder | Sensor | Descriptor (submodule) | Data |
| --- | --- | --- | --- |
| [4dradar](4dradar) | 4D radar | SOLiD (`thirdparty/SOLiD` of the repository) | SNAIL-Radar st route, two days |
| [radar](radar) | spinning radar (Navtech) | RadLoc (`radar/thirdparty/RadLoc`) | MulRan KAIST_03 |
| [sonar](sonar) | imaging sonar | Sonar Context (`sonar/thirdparty/sonar_context`) | Aracati 2017 |
| [thermal](thermal) | thermal camera | AnyThermal (`thermal/thirdparty/AnyThermal`) | ViViD++ campus, day vs night |
| [camera](camera) | camera | MegaLoc (`torch.hub`) | KITTI 00 |

Shared pieces:

- `common.py`: ground-truth keyframe poses, the rigid move of robot B, sessions in Commerge's format, a quick check of how well an affinity separates revisits.
- `make_gif.py`: the demo GIF for any two sessions (`--registration pairs`, `--affinity`); the maps are drawn from each keyframe's own sensor points (`<session>/points/<k>.npy`, written by each demo's `points.py`), or from LiDAR scans (`Scans/`).

```bash
git submodule update --init --recursive tools/demo     # the descriptors
python tools/demo/make_gif.py runs/radar/robot_a runs/radar/robot_b --affinity runs/radar/affinity.npy \
    --threshold 0.93 --min-area 50 --registration pairs --range 80 --out radar.gif
```
