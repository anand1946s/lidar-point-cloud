#!/usr/bin/env python3
import sys
import os
import copy
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

# Force X11/XWayland backend on Linux to avoid Wayland window placement & GLEW errors in Open3D/GLFW
if sys.platform.startswith("linux"):
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["XDG_SESSION_TYPE"] = "x11"

# Find paths relative to task2
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT.parent))

from config import DATA_DIR, VOXEL_SIZE

OUTPUTS_ROOT = PROJECT_ROOT / "outputs_scannet"


def parse_run_name(run_name):
    """
    Parses a run directory name into registration method, feature method,
    source scan name, and target scan name.
    """
    parts = run_name.split('_')
    if len(parts) >= 10:
        if parts[-8].startswith("scene") and parts[-4].startswith("scene"):
            source_name = "_".join(parts[-8:-4])
            target_name = "_".join(parts[-4:])
            rem = parts[:-8]

            if len(rem) >= 2 and rem[0] == "GO" and rem[1] == "ICP":
                registration_method = "GO_ICP"
                feature_method = "_".join(rem[2:])
            elif len(rem) >= 2 and rem[0] == "ROBUST" and rem[1] == "ICP":
                registration_method = "ROBUST_ICP"
                feature_method = "_".join(rem[2:])
            else:
                registration_method = rem[0]
                feature_method = "_".join(rem[1:])

            return registration_method, feature_method, source_name, target_name

    return None, None, None, None


def load_scan(scan_name):
    """Loads raw target PLY point cloud from the Scannet folder."""
    path = DATA_DIR / f"{scan_name}.ply"
    if not path.exists():
        print(f"\033[91mWarning: scan file not found at {path}\033[0m")
        return None
    try:
        pcd = o3d.io.read_point_cloud(str(path))
        return pcd
    except Exception as e:
        print(f"\033[91mError loading target scan {scan_name}: {e}\033[0m")
        return None


class ScanNetViewerState:
    """Manages color modes and view selection for ScanNet registration outputs."""

    def __init__(self, source_pcd, target_pcd, run_name):
        self.run_name = run_name

        # Deep copies to preserve original geometries and colors
        self.orig_source = copy.deepcopy(source_pcd) if source_pcd else None
        self.orig_target = copy.deepcopy(target_pcd) if target_pcd else None

        # Display mode: 1 = Both, 2 = Source Only, 3 = Target Only
        self.display_mode = 1

        # Color schemes: "original", "red_green", "pink_cyan", "normals", "height"
        self.color_scheme = "original"
        self.color_schemes = ["original", "red_green", "pink_cyan", "normals", "height"]

        self.geometry_container = o3d.geometry.PointCloud()
        self.update_geometries()

    def _get_colored_pcd(self, pcd, is_source=True):
        if pcd is None or len(pcd.points) == 0:
            return None

        p_copy = copy.deepcopy(pcd)
        pts = np.asarray(p_copy.points)

        if self.color_scheme == "original":
            if not p_copy.has_colors():
                # Fallback to height gradient if original RGB is missing
                p_copy.colors = o3d.utility.Vector3dVector(self._get_height_colors(pts))
            return p_copy

        elif self.color_scheme == "red_green":
            color = [0.9, 0.15, 0.15] if is_source else [0.1, 0.65, 0.25]
            p_copy.paint_uniform_color(color)
            return p_copy

        elif self.color_scheme == "pink_cyan":
            color = [1.0, 0.15, 0.58] if is_source else [0.0, 0.85, 0.85]
            p_copy.paint_uniform_color(color)
            return p_copy

        elif self.color_scheme == "normals":
            if not p_copy.has_normals():
                p_copy.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
            normals = np.asarray(p_copy.normals)
            if len(normals) == len(pts):
                p_copy.colors = o3d.utility.Vector3dVector(np.clip((normals + 1.0) / 2.0, 0.0, 1.0))
            return p_copy

        elif self.color_scheme == "height":
            p_copy.colors = o3d.utility.Vector3dVector(self._get_height_colors(pts))
            return p_copy

        return p_copy

    def _get_height_colors(self, points):
        if len(points) == 0:
            return np.zeros((0, 3))
        vals = points[:, 2]
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            return np.tile([0.0, 0.7, 0.7], (len(points), 1))
        norm_vals = (vals - vmin) / (vmax - vmin)
        colors = np.zeros_like(points)
        colors[:, 0] = norm_vals  # Red channel
        colors[:, 1] = 0.5 * (1 - norm_vals)  # Green channel
        colors[:, 2] = 1.0 - norm_vals  # Blue channel
        return colors

    def update_geometries(self):
        combined_pts = []
        combined_cols = []

        if self.orig_source and self.display_mode in [1, 2]:
            s_colored = self._get_colored_pcd(self.orig_source, is_source=True)
            if s_colored and len(s_colored.points) > 0:
                combined_pts.append(np.asarray(s_colored.points))
                combined_cols.append(np.asarray(s_colored.colors))

        if self.orig_target and self.display_mode in [1, 3]:
            t_colored = self._get_colored_pcd(self.orig_target, is_source=False)
            if t_colored and len(t_colored.points) > 0:
                combined_pts.append(np.asarray(t_colored.points))
                combined_cols.append(np.asarray(t_colored.colors))

        if combined_pts:
            pts_all = np.vstack(combined_pts)
            cols_all = np.vstack(combined_cols)
            self.geometry_container.points = o3d.utility.Vector3dVector(pts_all)
            self.geometry_container.colors = o3d.utility.Vector3dVector(cols_all)
        else:
            self.geometry_container.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.geometry_container.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

    def set_display_mode(self, mode):
        self.display_mode = mode
        mode_names = {1: "BOTH (Source + Target)", 2: "ALIGNED SOURCE ONLY", 3: "TARGET ONLY"}
        print(f"\n\033[92m[Viewer State] View Mode: {mode_names[mode]}\033[0m")
        self.update_geometries()

    def cycle_color_scheme(self):
        idx = self.color_schemes.index(self.color_scheme)
        self.color_scheme = self.color_schemes[(idx + 1) % len(self.color_schemes)]
        labels = {
            "original": "ORIGINAL NATURAL RGB COLORS",
            "red_green": "RED (Source) / GREEN (Target) OVERLAY",
            "pink_cyan": "PINK (Source) / CYAN (Target) OVERLAY",
            "normals": "SURFACE NORMALS SHADING",
            "height": "Z-HEIGHT GRADIENT MAP",
        }
        print(f"\n\033[92m[Viewer State] Color Scheme: {labels[self.color_scheme]}\033[0m")
        self.update_geometries()


def print_help_menu():
    print("\n\033[1;95m=== Interactive ScanNet Visualizer Controls ===\033[0m")
    print("  \033[93m[1]\033[0m : Show BOTH Aligned Source & Target Point Clouds")
    print("  \033[93m[2]\033[0m : Show ALIGNED SOURCE Point Cloud Only")
    print("  \033[93m[3]\033[0m : Show TARGET Point Cloud Only")
    print("  \033[93m[C]\033[0m : Cycle Color Mode (ORIGINAL RGB -> RED/GREEN -> PINK/CYAN -> NORMALS -> HEIGHT)")
    print("  \033[93m[P]\033[0m : Print Point Cloud Statistics & Evaluation Metrics")
    print("  \033[93m[H]\033[0m : Show Help Menu")
    print("  \033[93m[Q] / [Esc]\033[0m : Close Visualizer Window")


def visualize_registration(aligned_source, target, window_name, run_dir=None):
    """Visualizes aligned source and target point clouds with full interactive controls."""
    state = ScanNetViewerState(aligned_source, target, window_name)

    src_count = len(aligned_source.points) if aligned_source else 0
    tgt_count = len(target.points) if target else 0

    print(f"\nShowing registration result: \033[92m{window_name}\033[0m")
    print(f"  - Aligned Source points: \033[94m{src_count:,}\033[0m")
    print(f"  - Target points:         \033[94m{tgt_count:,}\033[0m")
    print(f"  - Default Color Mode:    \033[92mORIGINAL NATURAL RGB COLORS\033[0m")

    print_help_menu()

    try:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"ScanNet Registration Viewer - {window_name}", width=1200, height=800)
        vis.add_geometry(state.geometry_container)

        opt = vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])  # Pure White background
        opt.point_size = 2.0

        def cb_mode_1(v):
            state.set_display_mode(1)
            v.update_geometry(state.geometry_container)
            v.update_renderer()
            return True

        def cb_mode_2(v):
            state.set_display_mode(2)
            v.update_geometry(state.geometry_container)
            v.update_renderer()
            return True

        def cb_mode_3(v):
            state.set_display_mode(3)
            v.update_geometry(state.geometry_container)
            v.update_renderer()
            return True

        def cb_color(v):
            state.cycle_color_scheme()
            v.update_geometry(state.geometry_container)
            v.update_renderer()
            return True

        def cb_stats(v):
            print(f"\n\033[1;94m=== Registration Run Info ===\033[0m")
            print(f"  Run Name:          {window_name}")
            print(f"  Source Points:     {src_count:,}")
            print(f"  Target Points:     {tgt_count:,}")
            print(f"  Active Color Mode: {state.color_scheme.upper()}")
            if run_dir:
                metrics_file = run_dir / "metrics.txt"
                if metrics_file.exists():
                    print("\n\033[1;92m=== Metrics for this run ===\033[0m")
                    print(metrics_file.read_text().strip())
            return True

        def cb_help(v):
            print_help_menu()
            return True

        vis.register_key_callback(ord("1"), cb_mode_1)
        vis.register_key_callback(ord("2"), cb_mode_2)
        vis.register_key_callback(ord("3"), cb_mode_3)

        for key in [ord("C"), ord("c")]:
            vis.register_key_callback(key, cb_color)
        for key in [ord("P"), ord("p")]:
            vis.register_key_callback(key, cb_stats)
        for key in [ord("H"), ord("h")]:
            vis.register_key_callback(key, cb_help)

        vis.run()
        vis.destroy_window()

    except Exception as e:
        print(f"\n\033[91mFailed to launch interactive Visualizer GUI: {e}\033[0m")
        print("Attempting fallback visualization (draw_geometries)...")
        try:
            geoms = []
            if aligned_source:
                geoms.append(aligned_source)
            if target:
                geoms.append(target)
            o3d.visualization.draw_geometries(geoms, window_name=window_name, width=1200, height=800)
        except Exception as e2:
            print(f"\033[91mFallback visualization also failed: {e2}\033[0m")


def view_run(run_dir):
    """Loads files and launches visualization for a selected output run directory."""
    print(f"\nLoading run: \033[95m{run_dir.name}\033[0m")

    reg, feat, source_name, target_name = parse_run_name(run_dir.name)

    ply_files = list(run_dir.glob("*.ply"))
    if not ply_files:
        print(f"\033[91mError: No .ply file found in {run_dir.name}\033[0m")
        return

    aligned_source_file = ply_files[0]
    print(f"Reading aligned source: {aligned_source_file.name} ...")
    aligned_source = o3d.io.read_point_cloud(str(aligned_source_file))
    if len(aligned_source.points) == 0:
        print("\033[91mError: Aligned source cloud is empty.\033[0m")
        return

    target = None
    if target_name:
        print(f"Loading target scan: {target_name} ...")
        target_raw = load_scan(target_name)
        if target_raw:
            print(f"Downsampling target scan with voxel size {VOXEL_SIZE}...")
            target = target_raw.voxel_down_sample(VOXEL_SIZE)

    visualize_registration(aligned_source, target, run_dir.name, run_dir=run_dir)

    metrics_file = run_dir / "metrics.txt"
    if metrics_file.exists():
        print("\n\033[1;92m=== Metrics for this run ===\033[0m")
        print(metrics_file.read_text().strip())
        print("============================\n")


def get_filtered_runs(runs, query):
    """Filters run list based on query string."""
    query = query.lower().strip()
    if not query:
        return runs
    return [r for r in runs if query in r.name.lower() or query in r.parent.name.lower()]


def resolve_exact_run(choice, all_runs):
    """Checks if input matches any run name exactly (case-insensitive)."""
    for r in all_runs:
        if r.name.lower() == choice.lower():
            return r
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize any point cloud registration output from task2 outputs.")
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help="Optional directory name of the run output to view directly (e.g. 'ICP_NONE_scene0006_01_vh_clean_scene0006_00_vh_clean')."
    )
    args = parser.parse_args()

    if not OUTPUTS_ROOT.exists():
        print(f"\033[91mError: Outputs directory not found at {OUTPUTS_ROOT}\033[0m")
        sys.exit(1)

    # Get all runs recursively that contain a .ply file
    all_runs = []
    for scene_dir in OUTPUTS_ROOT.iterdir():
        if scene_dir.is_dir():
            for run_dir in scene_dir.iterdir():
                if run_dir.is_dir() and any(run_dir.glob("*.ply")):
                    all_runs.append(run_dir)
    all_runs.sort(key=lambda x: (x.parent.name, x.name))

    if not all_runs:
        print(f"\033[91mNo valid output runs found in {OUTPUTS_ROOT}\033[0m")
        sys.exit(1)

    # Direct CLI loading
    if args.run is not None:
        selected_run = resolve_exact_run(args.run, all_runs)
        if selected_run is None:
            matches = [r for r in all_runs if args.run.lower() in r.name.lower()]
            if len(matches) == 1:
                selected_run = matches[0]
            elif len(matches) > 1:
                print(f"\033[93mMultiple matches found for '{args.run}':\033[0m")
                for idx, m in enumerate(matches):
                    print(f"  [{idx}] {m.name}")
                sys.exit(1)
            else:
                print(f"\033[91mCould not find any output run matching '{args.run}'\033[0m")
                sys.exit(1)

        view_run(selected_run)
        sys.exit(0)

    # Interactive Menu
    filter_query = ""
    while True:
        all_runs = []
        for scene_dir in OUTPUTS_ROOT.iterdir():
            if scene_dir.is_dir():
                for run_dir in scene_dir.iterdir():
                    if run_dir.is_dir() and any(run_dir.glob("*.ply")):
                        all_runs.append(run_dir)
        all_runs.sort(key=lambda x: (x.parent.name, x.name))

        filtered_runs = get_filtered_runs(all_runs, filter_query)

        print("\n\033[1;95m=== ScanNet Point Cloud Registration Viewer ===\033[0m")
        if filter_query:
            print(f"Filtered by: '\033[93m{filter_query}\033[0m' ({len(filtered_runs)} of {len(all_runs)} runs)")
        else:
            print(f"Total available runs: \033[94m{len(all_runs)}\033[0m")

        print("-" * 110)
        for idx, run_dir in enumerate(filtered_runs):
            scene_name = run_dir.parent.name
            reg, feat, src, tgt = parse_run_name(run_dir.name)
            if reg:
                info_str = f"Scene: {scene_name:<10} | Reg: {reg:<12} | Feat: {feat:<10} | {src} -> {tgt}"
            else:
                info_str = f"Scene: {scene_name:<10} | {run_dir.name}"
            print(f"  \033[94m[{idx:3d}]\033[0m {info_str}")

        if not filtered_runs:
            print("  \033[91mNo runs match the current query.\033[0m")

        print("-" * 110)
        print("Commands:")
        print("  - \033[96m<Index>\033[0m (e.g. \033[96m0\033[0m) to view a specific run")
        print("  - \033[96m<Keyword>\033[0m (e.g. \033[96mscene0006\033[0m, \033[96mFGR\033[0m) to filter the list")
        print("  - \033[96mc\033[0m to clear filters")
        print("  - \033[93mq\033[0m to quit")

        try:
            choice = input("\033[96mSelect > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if choice.lower() == 'q':
            print("Goodbye!")
            break
        elif choice.lower() == 'c':
            filter_query = ""
            continue

        exact = resolve_exact_run(choice, all_runs)
        if exact:
            view_run(exact)
        elif choice.isdigit() and 0 <= int(choice) < len(filtered_runs):
            view_run(filtered_runs[int(choice)])
        else:
            filter_query = choice


if __name__ == "__main__":
    main()
