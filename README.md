# Lidar Point Cloud Registration - User Manual

This repository benchmarks point-cloud registration pipelines across four datasets:

- **Task 1**: Stanford object scans (Bunny/Dragon/Buddha)
- **Task 2**: ScanNet indoor scenes
- **Task 3**: PAVIN outdoor LiDAR scans
- **Task 4**: 3DMatch indoor scans

The workflow is config-driven: edit a task's `config.py`, run `main.py`, then inspect outputs with `viewer.py`.

## 1. Quick start

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run one pipeline:

```bash
python task1/main.py
```

View results:

```bash
python task1/viewer.py
```

## 2. Project layout

- `task1/`, `task2/`, `task3/`, `task4/`: task-specific pipelines
- `source_codes/sift3d/`: local C++ SIFT3D keypoint executable
- `source_codes/Go-ICP-master/Go-ICP-master/`: local C++ Go-ICP executable
- `Scannet/`, `pavin/`, `3dmatch/`, and task1 data folders: datasets

## 3. Running each task

For each task, configure pair + method in `taskX/config.py`, then run:

```bash
python taskX/main.py
```

### Config fields you will use most

- `SOURCE_NAME`, `TARGET_NAME`: scan pair
- `FEATURE_METHOD`: `NONE`, `ISS`, `3DSIFT`, `HARRIS3D`, `FPFH`, `SHOT` (Task 4 also supports `MULTI_SCALE`)
- `REGISTRATION_METHOD`: `ICP`, `ROBUST_ICP`, `FGR`, `GO_ICP`
- `VOXEL_SIZE` (Tasks 2/3/4), thresholds, and Go-ICP parameters
- `VISUALIZE`, `SAVE_RESULTS`

### Output format

Each run writes a folder like:

`<REGISTRATION_METHOD>_<FEATURE_METHOD>_<SOURCE_NAME>_<TARGET_NAME>`

Inside each run folder:

- `metrics.txt` (fitness, rmse, rotation error, translation error)
- registered `.ply` output cloud
- Task 4 also writes a full-resolution `*_full.ply`

## 4. Batch and combination runs

Run predefined scan-pair batches:

```bash
python task1/batch.py
python task2/batch.py
python task3/batch.py
python task4/batch.py
```

Run all feature/registration combinations:

```bash
python task3/run_combinations.py
python task4/run_combinations.py
```

These scripts edit `taskX/config.py` while running.

## 5. Viewer commands

Interactive viewers:

```bash
python task1/viewer.py
python task2/viewer.py
python task3/viewer.py
python task4/viewer.py
```

Direct run selection examples:

```bash
python task2/viewer.py ICP_FPFH_scene0027_02_vh_clean_scene0027_00_vh_clean
python task4/viewer.py --list
```

There is also a root-level PAVIN scan browser:

```bash
python view.py
python view.py scan300 --color distance
```

## 6. Native dependencies for optional methods

Some methods require local C++ binaries:

- `FEATURE_METHOD="3DSIFT"` needs `source_codes/sift3d/build/sift3d`
- `REGISTRATION_METHOD="GO_ICP"` needs `source_codes/Go-ICP-master/Go-ICP-master/GoICP`

Build SIFT3D:

```bash
cd source_codes/sift3d
cmake -S . -B build
cmake --build build -j
```

Build Go-ICP:

```bash
cd source_codes/Go-ICP-master/Go-ICP-master
cmake -S . -B build
cmake --build build -j
cp build/GoICP .
```

## 7. Troubleshooting

- **`FGR` warning about missing FPFH**: FGR needs feature data (`FPFH`, or another mode that produces descriptors).
- **Empty cloud / failed feature extraction**: verify `SOURCE_NAME`, `TARGET_NAME`, file names, and dataset paths in `config.py`.
- **GUI issues on Linux/Wayland**: viewers force X11 environment variables; run locally with desktop/X11 support.
- **Go-ICP/SIFT3D not found**: build binaries in `source_codes` and keep paths unchanged or update `GO_ICP_BIN_PATH`.
