# Commerge C++

The C++ core of [Commerge](https://github.com/sparolab/Commerge): the same scan selection (steps (A), (B), (C)),
solvers (HiGHS, optional Gurobi), registration (KISS-Matcher + GICP) and outputs as the Python package,
as a library (`commerge::commerge`), a command (`commerge_cpp`) and Python bindings (`commerge_cpp`).
See the main README for the method and the options.

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build -j    # from the repository root
cpp/build/commerge_cpp robot_a robot_b --threshold 0.96 --min-area 2000 --out merged
pip install commerge-cpp                                    # Python bindings (HiGHS; selection)
```
