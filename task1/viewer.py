#!/usr/bin/env python3
import argparse
import copy
import os
import re
import sys
from pathlib import Path

import numpy as np

# Force X11/XWayland backend on Linux to avoid Wayland window placement & GLEW errors in Open3D/GLFW
if sys.platform.startswith("linux"):
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d

# Find script directory and project root
SCRIPT_DIR = Path(__file__).resolve().parent

# Default dataset option: "bunny", "dragon", or "buddha"
DEFAULT_DATASET = "dragon"

# Dataset configurations mapping
DATASETS = {
    "bunny": {
        "output_dir": SCRIPT_DIR / "outputs_bunny",
        "data_dir": SCRIPT_DIR / "data",
        "display_name": "Stanford Bunny",
    },
    "dragon": {
        "output_dir": SCRIPT_DIR / "outputs_dragon_stand",
        "data_dir": SCRIPT_DIR / "dragon_stand",
        "display_name": "Stanford Dragon",
    },
    "buddha": {
        "output_dir": SCRIPT_DIR / "outputs_buddha",
        "data_dir": SCRIPT_DIR / "buddha" / "happy_side",
        "display_name": "Happy Buddha",
    },
}


def parse_run_metrics(metrics_file):
    """Parses key-value metrics from metrics.txt if present."""
    metrics = {}
    if not metrics_file.exists():
        return metrics
    try:
        with open(metrics_file, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, val = line.split(":", 1)
                    try:
                        metrics[key.strip()] = float(val.strip())
                    except ValueError:
                        metrics[key.strip()] = val.strip()
    except Exception:
        pass
    return metrics


def parse_pair_names(run_name):
    """Extracts source and target scan names from run folder name."""
    # Match pattern: ..._SOURCE_TARGET
    # e.g., FGR_SHOT_bun000_bun045 -> bun000, bun045
    # e.g., GO_ICP_ISS_dragonSideRight_0_dragonSideRight_24 -> dragonSideRight_0, dragonSideRight_24
    m = re.search(r"([A-Za-z0-9_]+)_([A-Za-z0-9]+_\d+)_([A-Za-z0-9]+_\d+)$", run_name)
    if not m:
        m = re.search(r"([A-Za-z0-9_]+)_(bun\d+)_(bun\d+)$", run_name)

    if m:
        groups = m.groups()
        return groups[-2], groups[-1]
    return None, None


def get_runs_for_dataset(dataset_key):
    """Retrieves all output run directories for a dataset."""
    cfg = DATASETS.get(dataset_key)
    if not cfg:
        return []

    output_dir = cfg["output_dir"]
    if not output_dir.exists():
        return []

    runs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    run_info_list = []

    for r in runs:
        metrics_file = r / "metrics.txt"
        ply_files = list(r.glob("*.ply"))
        ply_path = ply_files[0] if ply_files else None
        metrics = parse_run_metrics(metrics_file)
        src_name, tgt_name = parse_pair_names(r.name)

        run_info_list.append(
            {
                "run_dir": r,
                "run_name": r.name,
                "dataset": dataset_key,
                "ply_path": ply_path,
                "metrics_path": metrics_file,
                "metrics": metrics,
                "source_name": src_name,
                "target_name": tgt_name,
            }
        )

    return run_info_list


def resolve_run_or_dataset(query, current_dataset="bunny"):
    """
    Resolves user query string into a dataset key or specific run dictionary.
    """
    query_str = str(query).strip()
    if not query_str:
        return None, None, []

    # Check if query directly specifies a dataset
    query_lower = query_str.lower()
    if query_lower in DATASETS:
        return query_lower, None, []

    # Direct directory / path check
    path_check = Path(query_str)
    if path_check.exists():
        # Find which dataset it belongs to
        for dkey, dcfg in DATASETS.items():
            if (
                path_check.is_relative_to(dcfg["output_dir"])
                or path_check.parent == dcfg["output_dir"]
            ):
                runs = get_runs_for_dataset(dkey)
                for r in runs:
                    if r["run_dir"] == path_check or r["run_name"] == path_check.name:
                        return dkey, r, []

    # Resolve within current_dataset
    runs = get_runs_for_dataset(current_dataset)

    # By index
    if query_str.isdigit():
        idx = int(query_str)
        if 0 <= idx < len(runs):
            return current_dataset, runs[idx], []

    # Search by exact name or substring
    exact_matches = [r for r in runs if r["run_name"].lower() == query_lower]
    if len(exact_matches) == 1:
        return current_dataset, exact_matches[0], []

    substr_matches = [r for r in runs if query_lower in r["run_name"].lower()]
    if len(substr_matches) == 1:
        return current_dataset, substr_matches[0], []
    elif len(substr_matches) > 1:
        return current_dataset, None, substr_matches

    # Try searching across ALL datasets if nothing found in current
    for dkey in DATASETS:
        if dkey == current_dataset:
            continue
        all_runs = get_runs_for_dataset(dkey)
        sub_all = [r for r in all_runs if query_lower in r["run_name"].lower()]
        if len(sub_all) == 1:
            return dkey, sub_all[0], []
        elif len(sub_all) > 1:
            return dkey, None, sub_all

    return current_dataset, None, []


class RegistrationViewerState:
    """Manages view mode (Both, Source Only, Target Only) and color palettes."""

    def __init__(self, source_pcd, target_pcd, run_info):
        self.source_pcd = source_pcd
        self.target_pcd = target_pcd
        self.run_info = run_info

        # Display mode: 1 = Both, 2 = Source Only, 3 = Target Only
        self.mode = 1
        self.color_scheme = "red_green"  # "red_green", "cyan_magenta", "height"

        self.geometry_container = o3d.geometry.PointCloud()
        self.update_geometries()

    def update_geometries(self):
        combined_pts = []
        combined_colors = []

        # Color definitions optimized for crisp contrast on white background
        if self.color_scheme == "red_green":
            color_src = np.array([0.9, 0.15, 0.15])  # Deep Red
            color_tgt = np.array([0.1, 0.65, 0.25])  # Deep Green
        elif self.color_scheme == "cyan_magenta":
            color_src = np.array([0.85, 0.1, 0.65])  # Vibrant Magenta
            color_tgt = np.array([0.0, 0.55, 0.85])  # Deep Cyan/Blue
        else:  # height / amber_blue
            color_src = np.array([0.9, 0.45, 0.0])  # Amber / Orange
            color_tgt = np.array([0.15, 0.35, 0.9])  # Royal Blue

        if self.source_pcd is not None and self.mode in [1, 2]:
            pts_src = np.asarray(self.source_pcd.points)
            if len(pts_src) > 0:
                combined_pts.append(pts_src)
                combined_colors.append(np.tile(color_src, (len(pts_src), 1)))

        if self.target_pcd is not None and self.mode in [1, 3]:
            pts_tgt = np.asarray(self.target_pcd.points)
            if len(pts_tgt) > 0:
                combined_pts.append(pts_tgt)
                combined_colors.append(np.tile(color_tgt, (len(pts_tgt), 1)))

        if combined_pts:
            pts_all = np.vstack(combined_pts)
            cols_all = np.vstack(combined_colors)
            self.geometry_container.points = o3d.utility.Vector3dVector(pts_all)
            self.geometry_container.colors = o3d.utility.Vector3dVector(cols_all)
        else:
            self.geometry_container.points = o3d.utility.Vector3dVector(
                np.zeros((0, 3))
            )
            self.geometry_container.colors = o3d.utility.Vector3dVector(
                np.zeros((0, 3))
            )

    def set_mode(self, mode):
        self.mode = mode
        mode_str = {1: "BOTH (Source + Target)", 2: "SOURCE ONLY", 3: "TARGET ONLY"}[
            mode
        ]
        print(f"\n\033[92m[Viewer State] Display Mode: {mode_str}\033[0m")
        self.update_geometries()

    def toggle_color_scheme(self):
        schemes = ["red_green", "cyan_magenta", "height"]
        idx = schemes.index(self.color_scheme)
        self.color_scheme = schemes[(idx + 1) % len(schemes)]
        print(
            f"\n\033[92m[Viewer State] Color Palette: {self.color_scheme.upper()}\033[0m"
        )
        self.update_geometries()


def print_help_menu():
    print("\n\033[1;95m=== Interactive Visualizer Controls ===\033[0m")
    print("  \033[93m[1]\033[0m : Display BOTH Registered Source & Target Point Clouds")
    print("  \033[93m[2]\033[0m : Display SOURCE Point Cloud Only")
    print("  \033[93m[3]\033[0m : Display TARGET Point Cloud Only")
    print(
        "  \033[93m[C]\033[0m : Cycle Color Palette (Red/Green -> Cyan/Magenta -> Amber/Blue)"
    )
    print("  \033[93m[P]\033[0m : Print Point Cloud Statistics & Evaluation Metrics")
    print("  \033[93m[H]\033[0m : Show Help Menu")
    print("  \033[93m[Q] / [Esc]\033[0m : Close Visualizer Window")


def print_run_details(run_info, src_pts_count, tgt_pts_count):
    """Prints clean formatted metrics and info table to terminal."""
    print("\n" + "\033[1;36m=" * 60 + "\033[0m")
    print(f"\033[1;95m  Task 1 Registration Output Viewer\033[0m")
    print("\033[1;36m=" * 60 + "\033[0m")
    print(
        f"  \033[94mDataset:\033[0m           {DATASETS[run_info['dataset']]['display_name']} ({run_info['dataset']})"
    )
    print(f"  \033[94mRun Folder:\033[0m        {run_info['run_name']}")
    print(
        f"  \033[94mSource Scan:\033[0m       {run_info['source_name'] or 'Unknown'} ({src_pts_count:,} pts)"
    )
    print(
        f"  \033[94mTarget Scan:\033[0m       {run_info['target_name'] or 'Unknown'} ({tgt_pts_count:,} pts)"
    )

    metrics = run_info["metrics"]
    if metrics:
        print("\033[1;36m" + "-" * 60 + "\033[0m")
        print("  \033[1;93mEvaluation Metrics:\033[0m")
        if "fitness" in metrics:
            print(f"    - Fitness:            \033[92m{metrics['fitness']:.6f}\033[0m")
        if "rmse" in metrics:
            print(f"    - Inlier RMSE:        \033[92m{metrics['rmse']:.6f}\033[0m")
        if "rotation_error_deg" in metrics:
            print(
                f"    - Rotation Error:     \033[92m{metrics['rotation_error_deg']:.4f}°\033[0m"
            )
        if "translation_error" in metrics:
            print(
                f"    - Translation Error:  \033[92m{metrics['translation_error']:.6f}\033[0m"
            )
    print("\033[1;36m=" * 60 + "\033[0m\n")


def visualize_run(run_info):
    """Loads source and target point clouds and launches the visualizer."""
    dataset_key = run_info["dataset"]
    data_dir = DATASETS[dataset_key]["data_dir"]

    # 1. Load Aligned Source PLY
    source_pcd = None
    if run_info["ply_path"] and run_info["ply_path"].exists():
        try:
            source_pcd = o3d.io.read_point_cloud(str(run_info["ply_path"]))
        except Exception as e:
            print(
                f"\033[91mFailed to load aligned PLY ({run_info['ply_path'].name}): {e}\033[0m"
            )

    # If aligned PLY is missing or empty, fall back to raw source scan
    if source_pcd is None or len(source_pcd.points) == 0:
        if run_info["source_name"]:
            raw_src_file = data_dir / f"{run_info['source_name']}.ply"
            if raw_src_file.exists():
                print(
                    f"\033[93mNote: Using raw unaligned source point cloud from {raw_src_file.name}\033[0m"
                )
                source_pcd = o3d.io.read_point_cloud(str(raw_src_file))

    # 2. Load Target PLY
    target_pcd = None
    if run_info["target_name"]:
        target_file = data_dir / f"{run_info['target_name']}.ply"
        if target_file.exists():
            try:
                target_pcd = o3d.io.read_point_cloud(str(target_file))
            except Exception as e:
                print(
                    f"\033[91mFailed to load target PLY ({target_file.name}): {e}\033[0m"
                )

    src_count = len(source_pcd.points) if source_pcd else 0
    tgt_count = len(target_pcd.points) if target_pcd else 0

    if src_count == 0 and tgt_count == 0:
        print(
            f"\033[91mError: No valid point cloud data found for run '{run_info['run_name']}'.\033[0m"
        )
        return

    # Print summary & metrics
    print_run_details(run_info, src_count, tgt_count)
    print_help_menu()

    state = RegistrationViewerState(source_pcd, target_pcd, run_info)
    window_title = f"Task 1 Viewer [{dataset_key.upper()}] - {run_info['run_name']}"

    try:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_title, width=1200, height=800)
        vis.add_geometry(state.geometry_container)

        opt = vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])  # Pure white background
        opt.point_size = 2.0

        # Callbacks
        def cb_mode_1(v):
            state.set_mode(1)
            v.update_geometry(state.geometry_container)
            v.update_renderer()
            return True

        def cb_mode_2(v):
            state.set_mode(2)
            v.update_geometry(state.geometry_container)
            v.update_renderer()
            return True

        def cb_mode_3(v):
            state.set_mode(3)
            v.update_geometry(state.geometry_container)
            v.update_renderer()
            return True

        def cb_color(v):
            state.toggle_color_scheme()
            v.update_geometry(state.geometry_container)
            v.update_renderer()
            return True

        def cb_stats(v):
            print_run_details(run_info, src_count, tgt_count)
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
        print(f"\033[91mGUI Window initialization error: {e}\033[0m")
        print("Attempting fallback non-interactive visualization...")
        try:
            geoms = []
            if source_pcd:
                s_temp = copy.deepcopy(source_pcd)
                s_temp.paint_uniform_color([0.95, 0.25, 0.25])
                geoms.append(s_temp)
            if target_pcd:
                t_temp = copy.deepcopy(target_pcd)
                t_temp.paint_uniform_color([0.25, 0.85, 0.35])
                geoms.append(t_temp)
            o3d.visualization.draw_geometries(
                geoms, window_name=window_title, width=1200, height=800
            )
        except Exception as e2:
            print(f"\033[91mFallback visualization failed: {e2}\033[0m")


def display_interactive_menu(current_dataset="bunny"):
    """Interactive CLI menu to switch datasets, filter runs, and view pairs."""
    dataset = current_dataset.lower()

    while True:
        runs = get_runs_for_dataset(dataset)
        cfg = DATASETS[dataset]

        print("\n" + "\033[1;95m" + "=" * 70 + "\033[0m")
        print(
            f"\033[1;95m=== Task 1 Registration Output Viewer [{cfg['display_name']}] ===\033[0m"
        )
        print("\033[1;95m" + "=" * 70 + "\033[0m")
        print("Options:")
        print("  \033[93m[B]\033[0m Switch to Bunny dataset")
        print("  \033[93m[D]\033[0m Switch to Dragon dataset")
        print("  \033[93m[H]\033[0m Switch to Buddha dataset")
        print("  \033[93m[Q]\033[0m Quit")
        print("-" * 70)
        print(
            f"Total available runs in \033[94m{cfg['output_dir'].name}\033[0m: \033[92m{len(runs)}\033[0m\n"
        )

        print("  Index  | Fitness   | RMSE      | Run Name")
        print("  " + "-" * 65)
        for idx, r in enumerate(runs):
            fitness_str = (
                f"{r['metrics']['fitness']:.4f}"
                if "fitness" in r["metrics"]
                else " N/A  "
            )
            rmse_str = (
                f"{r['metrics']['rmse']:.4f}" if "rmse" in r["metrics"] else " N/A  "
            )
            status_flag = "" if r["ply_path"] else " \033[91m[NO PLY]\033[0m"
            print(
                f"  \033[94m[{idx:3d}]\033[0m  | {fitness_str}   | {rmse_str}   | {r['run_name']}{status_flag}"
            )

        print(
            "\nEnter run index (e.g. 0), search query (e.g. 'ROBUST', 'FGR', '0_24'), dataset switch ('b'/'d'/'h'), or 'q' to quit:"
        )
        try:
            choice = input("\033[96mSelect Run > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not choice:
            continue

        c_lower = choice.lower()
        if c_lower in ["q", "quit", "exit"]:
            print("Exiting viewer. Goodbye!")
            break

        if c_lower == "b" or c_lower == "bunny":
            dataset = "bunny"
            continue
        elif c_lower == "d" or c_lower == "dragon":
            dataset = "dragon"
            continue
        elif c_lower == "h" or c_lower == "buddha":
            dataset = "buddha"
            continue

        dkey, run_info, matches = resolve_run_or_dataset(choice, dataset)
        if run_info:
            visualize_run(run_info)
        elif matches:
            print(f"\n\033[93mMultiple matches found ({len(matches)}):\033[0m")
            for idx, m in enumerate(matches):
                f_str = (
                    f"{m['metrics'].get('fitness', 0.0):.4f}"
                    if "fitness" in m["metrics"]
                    else "N/A"
                )
                print(
                    f"  \033[94m[{idx:3d}]\033[0m ({m['dataset'].upper()}) Fitness: {f_str} | {m['run_name']}"
                )
            print("\nPlease select a run by exact index or full name.")
        else:
            print(
                f"\033[91mNo run matching '{choice}' was found in dataset '{dataset}'.\033[0m"
            )


def main():
    parser = argparse.ArgumentParser(
        description="View registration outputs and pairs for Task 1 (Bunny, Dragon, Buddha)."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Dataset name ('bunny', 'dragon', 'buddha'), run index, search query, or run folder path.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["bunny", "dragon", "buddha"],
        default=DEFAULT_DATASET,
        help=f"Dataset to view (default: '{DEFAULT_DATASET}').",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all output runs for the selected dataset and exit.",
    )
    args = parser.parse_args()

    active_dataset = args.dataset.lower()

    # CLI query provided
    if args.query is not None:
        q_lower = args.query.lower()

        # If user passed a dataset name directly as positional arg (e.g. 'python viewer.py dragon')
        if q_lower in DATASETS:
            active_dataset = q_lower
            if args.list:
                runs = get_runs_for_dataset(active_dataset)
                print(
                    f"\n\033[1;95m=== {DATASETS[active_dataset]['display_name']} Runs ({len(runs)}) ===\033[0m"
                )
                for idx, r in enumerate(runs):
                    print(f"  [{idx:3d}] {r['run_name']}")
                sys.exit(0)
            display_interactive_menu(active_dataset)
            sys.exit(0)

        # Resolve direct query / run name / index / path
        dkey, run_info, matches = resolve_run_or_dataset(args.query, active_dataset)
        if run_info:
            visualize_run(run_info)
            sys.exit(0)

        if matches:
            print(f"\033[93mMultiple matches found for query '{args.query}':\033[0m")
            for idx, m in enumerate(matches):
                print(f"  [{idx:2d}] ({m['dataset']}) {m['run_name']}")
            sys.exit(1)
        else:
            print(
                f"\033[91mCould not find any output run matching query '{args.query}'.\033[0m"
            )
            sys.exit(1)

    if args.list:
        runs = get_runs_for_dataset(active_dataset)
        print(
            f"\n\033[1;95m=== {DATASETS[active_dataset]['display_name']} Runs ({len(runs)}) ===\033[0m"
        )
        for idx, r in enumerate(runs):
            print(f"  [{idx:3d}] {r['run_name']}")
        sys.exit(0)

    # Default interactive menu
    display_interactive_menu(active_dataset)


if __name__ == "__main__":
    main()
