# LiDAR Point Cloud Registration Pipeline — Detailed Implementation Context

This document is written as **handoff context for another AI agent** that needs to modify or extend this repository safely.

The repo is a registration benchmarking framework across 4 datasets/tasks:

- `task1`: Stanford-style object scans (Bunny/Dragon/Buddha)
- `task2`: ScanNet indoor scenes
- `task3`: PAVIN outdoor LiDAR CSV scans
- `task4`: 3DMatch indoor scans

The same conceptual pipeline is reused everywhere, but each task overrides I/O format, scaling, thresholds, feature heuristics, and output behavior.

---

## 0) New-user setup

Run the setup commands from the repository root. Using a virtual environment
keeps this project's pinned Python packages separate from the system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the model required by the DGR registration methods:

```bash
python scripts/download_models.py
```

The script is safe to run more than once. It skips an existing checkpoint and
downloads the DGR 3DMatch model to:

```text
source_codes/DeepGlobalRegistration/ResUNetBN2C-feat32-3dmatch-v0.05.pth
```

This is the same path used by `task4/config.py` and the DGR loader. The
GeoTransformer checkpoint is already kept in
`source_codes/GeoTransformer/weights/geotransformer-3dmatch.pth.tar`, and the
R-PointHop checkpoint is already kept in
`source_codes/R-PointHop/model/R-PointHop.pkl`; neither is downloaded by the
setup script.

Before running an experiment, select one dataset block in the task's
`config.py`, set `SOURCE_NAME` and `TARGET_NAME` (or `PAIRS` for a batch), and
choose `FEATURE_METHOD` and `REGISTRATION_METHOD`. For example:

```bash
python task4/main.py
```

Use the task-specific `main.py`, `batch.py`, or `run_combinations.py` for the
workflow you want. Dataset files are resolved from the paths configured in
`config.py`; update those paths when using data stored outside this checkout.

---

## 0.1) Repository layout

```text
task1/                  Shared/canonical algorithms and Stanford scan pipeline
task2/                  ScanNet pipeline and task-specific configuration
task3/                  PAVIN LiDAR pipeline and task-specific configuration
task4/                  3DMatch pipeline, learned registration dispatch, outputs
source_codes/            Vendored third-party code, native tools, and checkpoints
scripts/                Setup and utility scripts, including model downloads
requirements.txt        Pinned Python dependencies
IMPLEMENTATION.md       Detailed architecture and extension notes
README.md               Short user-facing command reference
```

The task directories are executable entry points. `task1` contains much of the
shared registration and preprocessing code; tasks 2–4 reuse it where
appropriate while overriding dataset loading, configuration, and output
handling. `source_codes` contains external implementations such as
DeepGlobalRegistration, GeoTransformer, MinkowskiEngine, and R-PointHop.

---

## 1) Pipeline mental model (high-level)

Every run does:

1. Load constants from `taskX/config.py`.
2. Load source + target clouds.
3. Optional voxel downsample (task-dependent).
4. Build feature/initialization package using `prepare_registration_inputs(...)`.
5. Execute chosen registration backend (`ICP`, `ROBUST_ICP`, `FGR`, `GO_ICP`).
6. Compute metrics.
7. Save outputs to deterministic folder name.
8. Optionally visualize.

Core concept: **feature stage and registration stage are decoupled by a dictionary contract**.

---

## 2) Code structure and ownership

## 2.1 Shared logic mostly lives in `task1`

`task1` is the canonical implementation source for:

- registration algorithms (`task1/registration/*.py`)
- core feature extractors/wrappers (`task1/preprocessing/*.py`)
- utility functions (`task1/utils/*.py`)

`task2`, `task3`, `task4` reuse most of this but provide their own:

- `main.py`
- `config.py`
- `preprocessing/feature_pipeline.py`
- viewer and batch scripts

## 2.2 Import path behavior in task2/3/4

`task2/main.py`, `task3/main.py`, `task4/main.py` prepend their own task directory to `sys.path` before imports, so their local `config.py` and local `preprocessing` package take precedence.

Practical implication for edits:

- Changing `task1/preprocessing/fpfh.py` affects all tasks.
- Changing `task3/preprocessing/feature_pipeline.py` only affects task3.

---

## 3) Main execution flow by task

## 3.1 Task 1 (`task1/main.py`)

### Inputs

- PLY scans loaded by `task1/utils/io.py::load_scan()` from `DATA_DIR`.
- Ground truth loaded from task `.conf` file via `task1/utils/groundtruth.py`.

### Ground-truth math

`groundtruth.py` parses lines starting with `bmesh`:

- reads translation + quaternion
- converts to 4x4 transform
- computes relative transform as:

`T_gt = T_target @ inv(T_source)`

Then computes:

- `rotation_error` in degrees from trace-based SO(3) angle
- `translation_error` as L2 norm of translation difference

### Registration flow

- Calls `prepare_registration_inputs(feature_method, registration_method, source, target)`
- Initializes a **default failed `RegistrationResult`**:
  - transform = identity
  - fitness = 0
  - rmse = inf
- If feature stage yields empty registration clouds → skip registration and keep failed result.
- Else dispatch by registration method.

### Output

Folder name:

`<REGISTRATION_METHOD>_<FEATURE_METHOD>_<SOURCE_NAME>_<TARGET_NAME>`

Saves:

- `metrics.txt`
- aligned full-resolution source as `<run_name>.ply`

Prints one machine-parseable metric line:

`fitness\trmse\trot_deg\ttrans`

## 3.2 Task 2 (`task2/main.py`)

Differences from task1:

- Scan source: ScanNet PLY dataset (`Scannet/`).
- Downsamples source and target first (`voxel_down_sample(VOXEL_SIZE)`).
- Uses ScanNet GT conf (`scannet.conf`) for rot/trans error.
- Uses task2 feature pipeline tuned for room-scale data.
- Saves aligned **downsampled** source cloud to reduce disk footprint.

Everything else (dispatch semantics, failed default, metric format, output naming) stays structurally identical.

## 3.3 Task 3 (`task3/main.py`)

Important differences:

- Loader reads CSV-like files:
  - accepts with/without `.csv` in name
  - scans first ~20 lines for a `DATA` marker and skips header before numeric parse
  - keeps first 3 columns as XYZ
  - removes `[0,0,0]` dummy rows
- No GT file integrated here: rotation/translation errors forced to `0.0`.
- Downsamples source+target by `VOXEL_SIZE`.
- Evaluates final transformation via `o3d.pipelines.registration.evaluate_registration` on downsampled full clouds; this is used for reported `fitness/rmse`.
- For FGR call, threshold is fed as `voxel_size=RANSAC_THRESHOLD * 2.0`.

## 3.4 Task 4 (`task4/main.py`)

Differences:

- PLY input from `3dmatch/`.
- Downsamples for registration path.
- No GT in current workspace; rot/trans error fields set to `0.0`.
- Uses `task4/go_icp_helper.py::run_go_icp_task4` instead of task1 Go-ICP wrapper.
- Saves both:
  - compact aligned cloud `<run_name>.ply` (downsampled)
  - full-res aligned cloud `<run_name>_full.ply` (for high-quality viewing)

---

## 4) Feature pipeline contract (critical interface)

All task-specific `prepare_registration_inputs(...)` return:

```python
{
  "registration_source": o3d.geometry.PointCloud,
  "registration_target": o3d.geometry.PointCloud,
  "initial_transform": np.ndarray shape (4,4),
  "source_fpfh": o3d.pipelines.registration.Feature | None,
  "target_fpfh": o3d.pipelines.registration.Feature | None,
}
```

Main scripts rely on this exact contract.  
If you add a new feature mode, preserve these keys.

`registration_source/target` can be:

- full clouds
- keypoint clouds
- scaled/unscaled derived keypoint clouds

depending on method and whether registration is FGR.

---

## 5) Feature methods — detailed behavior

## 5.1 `NONE`

- No preprocessing.
- Uses full clouds.
- `initial_transform = I`.
- Works with ICP / ROBUST_ICP / GO_ICP.
- For FGR this is usually invalid unless descriptors are explicitly provided by pipeline branch.

## 5.2 `WHOLE_SET`

- Keeps full cloud as registration input.
- If registration is FGR, computes descriptors on full cloud and stores in `source_fpfh/target_fpfh`.

## 5.3 `FPFH`

- Compute normals and FPFH on source/target (`compute_fpfh`).
- Run Open3D RANSAC feature matching (`get_initial_transform`) to estimate coarse alignment.
- Save descriptors for possible FGR use.

### Parameter scaling

`compute_fpfh(point_cloud, voxel_size)`:

- normal radius = `2 * voxel_size`
- feature radius = `5 * voxel_size`

RANSAC threshold in `get_initial_transform`:

- `threshold = 1.5 * voxel_size`

## 5.4 `ISS`

- Detect ISS keypoints.
- Compute descriptors either:
  - directly on keypoint clouds (task1/task2/task3)
  - by dense-cloud descriptor indexing at nearest keypoint locations (`compute_indexed_fpfh`, task4).
- Estimate initial transform via feature RANSAC on keypoint clouds.
- For FGR mode: registration clouds are replaced with keypoint clouds.

## 5.5 `HARRIS3D`

- Uses external binary in `pcl_tools/harris_3d/build/harris3d`.
- Pipeline writes temporary PCD, calls binary, reads output keypoint PCD.
- Errors if binary missing or zero keypoints produced.

Initial transform:

- Estimated via ICP point-to-point on keypoints (`get_harris3d_initial_transform`).

Room-scale tasks (`task2/3/4`) scale trick:

- scale cloud down by factor 10 before extraction
- extract keypoints
- scale keypoints back up

Reason: adapt fixed-radius assumptions of Harris code to meter-scale scenes.

## 5.6 `3DSIFT` / aliases (`SIFT3D`, `SIFT`, etc.)

- Uses external binary: `source_codes/sift3d/build/sift3d`.
- Runs with params from active `config.py`:
  - normal radius
  - min scale
  - octaves
  - scales per octave
  - min contrast
- Returns keypoints as PCD.
- Initial transform estimated by ICP over keypoints.

## 5.7 `SHOT`

- Keypoints produced by voxel downsampling (`extract_shot_keypoints`).
- External binary (`pcl_tools/shot/build/shot`) computes 352D descriptors.
- `prepare_shot_registration_data` removes non-finite descriptor rows and matching keypoints.
- Returns Open3D Feature with `feature.data` shape `(352, N)`.

Task differences:

- Task1:
  - For non-FGR branch, SHOT initial-transform block is wrapped in a `try/except` and silently falls back if SHOT prep fails.
- Task2/3/4:
  - Use room-scale scaling heuristic (`/10`) and explicit keypoint voxel tuning (`kp_voxel_size=0.02` in scaled space).

## 5.8 `MULTI_SCALE` / `COARSE_TO_FINE` / `C2F`

Implemented in task4 pipeline.

Flow:

1. coarse downsample
2. coarse FPFH + RANSAC
3. medium-scale ICP refinement
4. return refined transform as `initial_transform`

Task4 values:

- coarse voxel ~0.20
- medium voxel ~0.10
- coarse threshold ~1.00
- medium ICP threshold ~0.30

FGR compatibility in this mode: full-cloud descriptors computed if needed.

---

## 6) Registration backends — exact behavior

## 6.1 ICP (`task1/registration/icp.py`)

- Open3D `registration_icp`
- estimation: `TransformationEstimationPointToPoint`
- uses provided `initial_transform`

## 6.2 ROBUST_ICP (`task1/registration/robust_icp.py`)

- Ensures normals exist on both clouds.
- If no normals, estimates with radius roughly tied to threshold.
- Uses Tukey robust loss (`loss_k` from config).
- Estimation model: point-to-plane with robust kernel.

## 6.3 FGR (`task1/registration/fgr.py`)

- Uses Open3D `registration_fgr_based_on_feature_matching`.
- `maximum_correspondence_distance = voxel_size * 0.5` inside wrapper.
- Requires feature objects for both clouds.

## 6.4 GO_ICP (`task1/registration/go_icp.py`)

Wrapper around compiled C++ executable:

1. Deep-copy source and apply initial transform.
2. Normalize source+target to centered `[-1,1]^3`.
3. Optional deterministic source subsample if `num_points > 0`.
4. Write temporary text files in Go-ICP format (`N` then XYZ lines).
5. Build Go-ICP config text (MSE threshold, rotation/translation bounds, trim fraction, distance transform size).
6. Call binary via `subprocess.run(..., check=True)`.
7. Parse output rotation matrix and translation.
8. Unnormalize translation back to original coordinate system.
9. Compose with initial transform:
   - `final_transform = T_goicp @ initial_transform`
10. Return Open3D evaluation result using `evaluate_registration(...)`.

Search-space behavior:

- If initial transform is identity: global bounds
- Otherwise: local narrow bounds (faster convergence)

## 6.5 Task4 Go-ICP helper (`task4/go_icp_helper.py`)

Adds identity-init heuristic:

- if no init transform, align centroids first for better initial translation.
- then keep global rotation search but tighter translation box around centroid-aligned state.

This usually stabilizes and accelerates 3DMatch runs.

---

## 7) Failure handling and fallback semantics

Mains (`task1..4`) intentionally start with a default failed `RegistrationResult`:

- identity transform
- fitness = 0
- rmse = inf

This failed result survives when:

- feature preprocessing yields empty registration clouds
- FGR selected but descriptors are missing
- method combinations are semantically invalid

Meaning: batch scripts still get parseable output without crashing entire sweep.

Exceptions still propagate for hard failures in many external-binary paths unless locally caught.

---

## 8) Metrics, evaluation, and run artifacts

## 8.1 Console line (batch-parsed)

All mains print final line:

`fitness<TAB>rmse<TAB>rotation_error_deg<TAB>translation_error`

Batch scripts parse from stdout by scanning lines in reverse and splitting on tab/comma.

## 8.2 Saved metrics file

`metrics.txt` stores key-value lines:

- `fitness`
- `rmse`
- `rotation_error_deg`
- `translation_error`

## 8.3 Output folder naming convention

`<REGISTRATION_METHOD>_<FEATURE_METHOD>_<SOURCE_NAME>_<TARGET_NAME>`

Examples:

- `FGR_SHOT_scene0027_02_vh_clean_scene0027_00_vh_clean`
- `GO_ICP_ISS_scan300_scan400`

This naming is used by viewers to recover pair metadata by parsing folder names.

---

## 9) Batch and combination orchestration

## 9.1 Batch scripts (`taskX/batch.py`)

Pattern:

1. Rewrite `SOURCE_NAME` and `TARGET_NAME` in `taskX/config.py` using regex.
2. Run `python taskX/main.py` via subprocess.
3. Parse metrics.
4. Accumulate averages and report elapsed time.

Side effect:

- Pair batch scripts **do not restore original source/target settings** at the end.
- Last pair remains written to config after script exits.

## 9.2 Combination sweeps (`task3/run_combinations.py`, `task4/run_combinations.py`)

Pattern:

1. Backup full original config text.
2. Loop over registration × feature list.
3. Rewrite feature + registration in config.
4. Execute `taskX/batch.py`.
5. Parse `AVERAGE` output from batch stdout.
6. Append result object and checkpoint to JSON after each combo.
7. Restore original config in `finally`.

Saved checkpoints:

- `task3_combos_checkpoint.json`
- `task4_combos_checkpoint.json`

Each entry stores:

- selected methods
- success flag
- averaged metrics
- elapsed seconds
- raw stdout/stderr

---

## 10) Viewer subsystem details

All viewers force Linux env vars:

- `GDK_BACKEND=x11`
- `XDG_SESSION_TYPE=x11`

to avoid Wayland/Open3D GLFW issues.

## 10.1 `task1/viewer.py`

- Multi-dataset viewer with dataset switching (bunny/dragon/buddha).
- Parses run directories, reads metrics, extracts source/target names.
- Interactive key controls:
  - show both/source/target
  - color scheme cycling
  - stats print
- Fallback to basic `draw_geometries` if callback visualizer fails.

## 10.2 `task2/viewer.py`

- Reads nested outputs: `task2/outputs_scannet/<scene>/<run>/`.
- Uses original target scan downsampled to `VOXEL_SIZE` for comparison.
- Supports direct run arg or interactive filtered list.

## 10.3 `task3/viewer.py`

- Handles PAVIN CSV target loading with header parsing.
- Displays aligned source + downsampled target.
- Multiple color schemes including normals and height gradients.

## 10.4 `task4/viewer.py`

- Prefers full-resolution aligned file `<run_name>_full.ply` if present.
- Falls back to standard run `.ply` otherwise.
- Loads full-resolution target for visual quality.
- Includes `--list` mode.

## 10.5 Root `view.py`

- Not a registration-result viewer.
- It is a raw PAVIN scan explorer with optional color modes (`height`, `distance`, `uniform`).

---

## 11) Native binary dependencies and expected locations

Required when using certain methods:

- SIFT3D: `source_codes/sift3d/build/sift3d`
- Harris3D: `pcl_tools/harris_3d/build/harris3d`
- SHOT: `pcl_tools/shot/build/shot`
- Go-ICP: `source_codes/Go-ICP-master/Go-ICP-master/GoICP`

If binaries are missing:

- wrappers raise explicit `FileNotFoundError` with build hint path.

---

## 12) Config variables that actually drive behavior

Common important fields:

- `DATA_DIR`, `OUTPUT_DIR`
- `SOURCE_NAME`, `TARGET_NAME`
- `FEATURE_METHOD`
- `REGISTRATION_METHOD`
- `VOXEL_SIZE` (task2/3/4, and feature behavior tuning)
- `ICP_THRESHOLD`, `ROBUST_ICP_THRESHOLD`, `ROBUST_ICP_LOSS_K`
- `RANSAC_THRESHOLD` (task3/task4 tuning)
- `GO_ICP_*` family
- `VISUALIZE`, `SAVE_RESULTS`

SIFT tuning fields (used by SIFT external wrapper):

- `SIFT3D_NORMAL_RADIUS`
- `SIFT3D_MIN_SCALE`
- `SIFT3D_OCTAVES`
- `SIFT3D_SCALES_PER_OCTAVE`
- `SIFT3D_MIN_CONTRAST`
- `SIFT3D_INITIAL_THRESHOLD`

---

## 13) Subtle implementation notes / gotchas

1. `FEATURE_METHOD` comments mention `"NONE" (Whole Set)`, but code also has explicit `"WHOLE_SET"` branch; they are not exactly the same branch in implementation.
2. FGR requires feature objects. If absent, code does not crash; it returns failed-result metrics.
3. Task3 config comments mention `MULTI_SCALE`, but task3 feature pipeline currently does not implement that branch; task4 does.
4. Batch scripts mutate config files on disk. Don’t assume config remains untouched after batch unless using combination runner that restores.
5. Task1 SHOT non-FGR path swallows exceptions in initial-transform prep and silently falls back to identity-ish flow.
6. Go-ICP wrappers return Open3D **evaluation result**, not an ICP optimization object.
7. Viewer run-name parsers are custom per task; if naming format changes, viewers can break run metadata extraction.

---

## 14) Extension playbook (for future codegen AI)

## 14.1 Add a new feature method

1. Implement branch in each intended task’s `preprocessing/feature_pipeline.py`.
2. Always produce valid dict keys with defaults.
3. If method supports FGR, provide `source_fpfh/target_fpfh` as Open3D Feature objects with aligned columns.
4. Add method string to config comments and sweep scripts if benchmarking is needed.
5. Ensure empty-feature cases degrade to failed result, not crash all batch jobs.

## 14.2 Add a new registration method

1. Implement backend under `task1/registration/` if shared.
2. Add dispatch branch in each `taskX/main.py`.
3. Define required config params and safe defaults.
4. Preserve standard output conventions:
   - one metric line in stdout
   - deterministic run folder naming
   - `metrics.txt` with same keys

## 14.3 Add a new dataset task

Minimum files to mirror:

- `taskN/config.py`
- `taskN/main.py`
- `taskN/preprocessing/feature_pipeline.py`
- `taskN/batch.py`
- `taskN/viewer.py`

Recommended strategy:

- reuse task1 registration and low-level feature modules first
- only fork when data format or scale demands it

---

## 15) Practical command map

- Run one configuration:
  - `python task1/main.py`
  - `python task2/main.py`
  - `python task3/main.py`
  - `python task4/main.py`
- Run pair batches:
  - `python task1/batch.py`
  - `python task2/batch.py`
  - `python task3/batch.py`
  - `python task4/batch.py`
- Run full method sweeps:
  - `python task3/run_combinations.py`
  - `python task4/run_combinations.py`
- Inspect outputs:
  - `python task1/viewer.py`
  - `python task2/viewer.py`
  - `python task3/viewer.py`
  - `python task4/viewer.py --list`

---

## 16) One-line summary for another AI

This repo is a **modular registration benchmark harness** where task-specific loaders and feature heuristics feed a shared registration backend set, with stable output/metric conventions and script-driven config mutation for benchmarking sweeps.