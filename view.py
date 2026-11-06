#!/usr/bin/env python3
import argparse
import copy
import os
import re
import sys
from pathlib import Path

# Force X11/XWayland backend on Linux to avoid Wayland window placement & GLEW errors in Open3D/GLFW
if sys.platform.startswith("linux"):
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ["GLFW_PLATFORM"] = "x11"

import numpy as np
import open3d as o3d

BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "bunny": {
        "title": "Stanford Bunny",
        "path": BASE_DIR / "bunny",
        "extensions": [".ply"],
        "aliases": ["bunny", "bun", "0"],
    },
    "pavin": {
        "title": "PAVIN Dataset",
        "path": BASE_DIR / "pavin",
        "extensions": [".csv", ".txt"],
        "aliases": ["pavin", "1"],
    },
    "task1/buddha": {
        "title": "Happy Buddha (Task 1)",
        "path": BASE_DIR / "task1" / "buddha",
        "extensions": [".ply"],
        "aliases": ["buddha", "task1/buddha", "2"],
    },
    "task1/dragon_stand": {
        "title": "Stanford Dragon (Task 1)",
        "path": BASE_DIR / "task1" / "dragon_stand",
        "extensions": [".ply"],
        "aliases": ["dragon", "dragon_stand", "task1/dragon_stand", "3"],
    },
    "3dmatch": {
        "title": "3DMatch (Scene 1)",
        "path": BASE_DIR / "3dmatch",
        "extensions": [".ply"],
        "aliases": ["3dmatch", "4"],
    },
    "3dmatch2": {
        "title": "3DMatch (Scene 2)",
        "path": BASE_DIR / "3dmatch2",
        "extensions": [".ply"],
        "aliases": ["3dmatch2", "5"],
    },
    "3dmatch3": {
        "title": "3DMatch (Scene 3)",
        "path": BASE_DIR / "3dmatch3",
        "extensions": [".ply"],
        "aliases": ["3dmatch3", "6"],
    },
    "scannet": {
        "title": "ScanNet Dataset",
        "path": BASE_DIR / "Scannet",
        "extensions": [".ply"],
        "aliases": ["scannet", "scan-net", "scan_net", "7"],
    },
}

COLOR_SCHEMES = [
    ("neon", "Neon Cyberpunk (Height)"),
    ("rainbow", "Rainbow / Turbo (Height)"),
    ("turbo", "Turbo Rainbow (Height)"),
    ("viridis", "Viridis (Height)"),
    ("plasma", "Plasma / Sunset (Height)"),
    ("distance", "Radial Distance"),
    ("normals", "Surface Normal Shading"),
    ("teal", "Uniform Teal"),
    ("gold", "Uniform Gold"),
]
COLOR_MODES = [s[0] for s in COLOR_SCHEMES] + ["original", "height", "uniform", "normal"]


def natural_sort_key(path):
    """Sorts file paths in human/natural numerical order (e.g. bin_2 before bin_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def get_dataset_files(dataset_key):
    """Recursively finds all valid files for a given dataset."""
    info = DATASETS.get(dataset_key)
    if not info:
        return []
    folder = info["path"]
    if not folder.exists():
        return []
    exts = tuple(info.get("extensions", [".ply", ".csv"]))
    files = [
        f for f in folder.rglob("*")
        if f.is_file()
        and f.suffix.lower() in exts
        and not f.name.startswith(".")
        and not f.name.endswith(".Identifier")
    ]
    return sorted(files, key=natural_sort_key)


def get_all_workspace_files():
    """Gathers all point cloud/scan files across all datasets."""
    all_files = []
    for d_key in DATASETS:
        all_files.extend(get_dataset_files(d_key))
    return sorted(all_files, key=natural_sort_key)


def interpolate_colormap(t, stops):
    """Piecewise linear RGB colormap interpolation for normalized values [0, 1]."""
    colors = np.zeros((len(t), 3))
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        mask = (t >= p0) & (t <= p1)
        if not np.any(mask):
            continue
        factor = ((t[mask] - p0) / (p1 - p0))[:, None]
        colors[mask] = (1 - factor) * np.array(c0) + factor * np.array(c1)
    return colors


def compute_point_colors(points, mode="neon", original_colors=None, normals=None):
    """
    Computes RGB point colors for various color schemes:
      - 'neon': Cyberpunk violet -> magenta -> bright cyan.
      - 'rainbow': Turbo/rainbow blue -> cyan -> green -> yellow -> red.
      - 'viridis': Perceptually uniform purple -> blue -> teal -> yellow.
      - 'plasma': Sunset navy -> purple -> orange -> bright yellow.
      - 'distance': Radial distance from origin.
      - 'normals': Surface normal shading (normal XYZ mapped to RGB).
      - 'teal': Uniform clean light cyan/teal.
      - 'gold': Uniform warm amber/gold.
      - 'original': Keep original RGB vertex colors if present.
    """
    if len(points) == 0:
        return np.empty((0, 3))

    if mode == "original" and original_colors is not None and len(original_colors) == len(points):
        return original_colors

    if mode in ["normal", "normals"]:
        if normals is not None and len(normals) == len(points):
            n = np.asarray(normals)
            # Normalize to unit vectors and map from [-1, 1] to [0, 1].
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            norm[norm == 0.0] = 1.0
            n = n / norm
            return np.clip(0.5 * (n + 1.0), 0.0, 1.0)
        return np.tile([0.0, 0.85, 0.75], (len(points), 1))

    if mode in ["teal", "uniform"]:
        return np.tile([0.0, 0.85, 0.75], (len(points), 1))

    if mode == "gold":
        return np.tile([0.96, 0.75, 0.15], (len(points), 1))

    if mode == "distance":
        vals = np.linalg.norm(points, axis=1)
    else:  # Height along Z axis
        vals = points[:, 2]

    val_min, val_max = vals.min(), vals.max()
    if val_max == val_min:
        return np.tile([0.0, 0.85, 0.75], (len(points), 1))

    t = np.clip((vals - val_min) / (val_max - val_min), 0.0, 1.0)

    if mode in ["neon", "height"]:
        stops = [
            (0.00, [0.15, 0.05, 0.40]),
            (0.50, [0.90, 0.15, 0.60]),
            (1.00, [0.00, 0.95, 0.95]),
        ]
    elif mode in ["rainbow", "turbo"]:
        stops = [
            (0.00, [0.18, 0.25, 0.85]),
            (0.35, [0.10, 0.75, 0.75]),
            (0.65, [0.30, 0.85, 0.20]),
            (0.85, [0.95, 0.85, 0.10]),
            (1.00, [0.90, 0.20, 0.10]),
        ]
    elif mode == "viridis":
        stops = [
            (0.00, [0.27, 0.00, 0.33]),
            (0.33, [0.19, 0.41, 0.56]),
            (0.67, [0.12, 0.63, 0.53]),
            (1.00, [0.99, 0.90, 0.14]),
        ]
    elif mode == "plasma":
        stops = [
            (0.00, [0.05, 0.03, 0.53]),
            (0.35, [0.55, 0.05, 0.65]),
            (0.70, [0.95, 0.40, 0.25]),
            (1.00, [0.95, 0.95, 0.15]),
        ]
    elif mode == "distance":
        stops = [
            (0.00, [0.00, 0.95, 0.85]),
            (0.50, [0.90, 0.20, 0.75]),
            (1.00, [1.00, 0.80, 0.10]),
        ]
    else:  # fallback neon
        stops = [
            (0.00, [0.15, 0.05, 0.40]),
            (0.50, [0.90, 0.15, 0.60]),
            (1.00, [0.00, 0.95, 0.95]),
        ]

    return interpolate_colormap(t, stops)


def load_csv_scan(file_path):
    """Loads point coordinates from a space-separated or comma-separated CSV/TXT file."""
    try:
        skip_rows = 0
        with open(file_path, "r", errors="ignore") as f:
            for idx, line in enumerate(f):
                if idx > 25:
                    break
                if "DATA" in line:
                    skip_rows = idx + 1
                    break

        points = np.loadtxt(file_path, skiprows=skip_rows)
        if len(points.shape) == 2 and points.shape[1] >= 3:
            points = points[:, :3]
        elif len(points.shape) == 1 and len(points) >= 3:
            points = points[:3].reshape(1, 3)
        else:
            raise ValueError(f"Unexpected array shape: {points.shape}")

        # Filter out origin points [0, 0, 0]
        points = points[~np.all(points == 0, axis=1)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    except Exception as e:
        print(f"\n\033[91mError loading CSV/TXT {file_path.name}: {e}\033[0m")
        return None


def load_point_cloud(file_path):
    """Loads point cloud from PLY or CSV/TXT file."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".ply":
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            if len(pcd.points) > 0:
                return pcd
            # Fallback in case PLY is saved as a triangle mesh
            mesh = o3d.io.read_triangle_mesh(str(path))
            if len(mesh.vertices) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = mesh.vertices
                if mesh.has_vertex_colors():
                    pcd.colors = mesh.vertex_colors
                if mesh.has_vertex_normals():
                    pcd.normals = mesh.vertex_normals
                return pcd
            print(f"\033[91mNotice: {path.name} contains 0 points or vertices.\033[0m")
            return None
        except Exception as e:
            print(f"\033[91mError reading PLY {path.name}: {e}\033[0m")
            return None
    elif ext in [".csv", ".txt", ".xyz", ".pts"]:
        return load_csv_scan(path)
    else:
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            if len(pcd.points) > 0:
                return pcd
        except Exception:
            pass
        return load_csv_scan(path)


def visualize_pcd(pcd, file_name, color_mode="neon"):
    """Displays the point cloud in an interactive Open3D window with keypress callbacks."""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        print("\033[91mCannot visualize: point cloud has 0 points.\033[0m")
        return color_mode

    has_orig = pcd.has_colors()
    orig_colors = np.asarray(pcd.colors).copy() if has_orig else None
    if not pcd.has_normals():
        try:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
        except Exception:
            pass
    normals = np.asarray(pcd.normals).copy() if pcd.has_normals() else None

    # Available color schemes
    active_schemes = list(COLOR_SCHEMES)
    if has_orig:
        active_schemes.insert(0, ("original", "Original RGB Colors"))

    scheme_keys = [s[0] for s in active_schemes]
    norm_mode = "neon" if color_mode in ["height", "neon"] else ("teal" if color_mode == "uniform" else color_mode)
    if norm_mode == "normal":
        norm_mode = "normals"
    if norm_mode not in scheme_keys:
        norm_mode = "original" if has_orig else "neon"
    current_idx = scheme_keys.index(norm_mode)

    state = {
        "idx": current_idx,
    }

    # Apply initial colors
    init_key, init_title = active_schemes[state["idx"]]
    initial_cols = compute_point_colors(points, init_key, orig_colors, normals=normals)
    pcd.colors = o3d.utility.Vector3dVector(initial_cols)

    print(f"\nShowing: \033[92m{file_name}\033[0m")
    print(f"Points:  \033[94m{len(points):,}\033[0m | Color Scheme: \033[95m{init_title}\033[0m")
    print("\033[1;93m=== 3D Window Controls ===\033[0m")
    print("  \033[1;92m[C]\033[0m : Switch Color Scheme (Neon -> Rainbow -> Turbo -> Viridis -> Plasma -> Distance -> Normals -> Teal -> Gold)")
    print("  \033[1;92m[B]\033[0m : Toggle Background (Dark / Light)")
    print("  \033[1;92m[Q] / [Esc]\033[0m : Close Window & Return to Menu")
    print("  \033[90mMouse: Left Click+Drag to Rotate | Scroll to Zoom | Right Click or Shift+Left to Pan\033[0m\n")

    bg_state = [1]  # 1 = White, 0 = Dark
    dark_bg = np.array([0.06, 0.06, 0.08])
    light_bg = np.array([1.0, 1.0, 1.0])  # Pure white background

    try:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        created = vis.create_window(window_name=f"Point Cloud Viewer - {file_name}", width=1200, height=800)
        if not created:
            raise RuntimeError("Failed to create Open3D visualizer window")
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        if opt is not None:
            opt.background_color = light_bg
            opt.point_size = 2.0

        def cb_change_color(v):
            state["idx"] = (state["idx"] + 1) % len(active_schemes)
            new_key, new_title = active_schemes[state["idx"]]
            new_cols = compute_point_colors(points, new_key, orig_colors, normals=normals)
            pcd.colors = o3d.utility.Vector3dVector(new_cols)
            v.update_geometry(pcd)
            v.update_renderer()
            print(f"\033[92m>> Color Scheme:\033[0m \033[1;95m{new_title}\033[0m")
            return True

        def cb_toggle_bg(v):
            bg_state[0] = 1 - bg_state[0]
            new_bg = light_bg if bg_state[0] == 1 else dark_bg
            bg_name = "White" if bg_state[0] == 1 else "Dark"
            r_opt = v.get_render_option()
            if r_opt is not None:
                r_opt.background_color = new_bg
            v.update_renderer()
            print(f"\033[92m>> Background:\033[0m \033[1;94m{bg_name}\033[0m")
            return True

        for key in [ord("C"), ord("c")]:
            vis.register_key_callback(key, cb_change_color)
        for key in [ord("B"), ord("b")]:
            vis.register_key_callback(key, cb_toggle_bg)

        vis.run()
        vis.destroy_window()
        return active_schemes[state["idx"]][0]
    except Exception as e:
        print(f"\033[91mVisualizer GUI error: {e}\033[0m")
        print("Attempting fallback visualization (draw_geometries)...")
        try:
            o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud Viewer - {file_name}", width=1200, height=800)
        except Exception as e2:
            print(f"\033[91mFallback visualization also failed: {e2}\033[0m")
        return color_mode


def resolve_file(query, file_list):
    """Resolves a query string against a list of file paths."""
    query_str = str(query).strip()
    if not query_str:
        return None, []

    # 1. Direct path check
    direct_path = Path(query_str)
    if direct_path.exists() and direct_path.is_file():
        return direct_path, []

    # 2. Direct index check
    if query_str.isdigit():
        idx = int(query_str)
        if 0 <= idx < len(file_list):
            return file_list[idx], []

    query_clean = query_str.lower()
    # Strip common extensions for stem matching
    stem_query = query_clean
    for ext in [".ply", ".csv", ".txt"]:
        if stem_query.endswith(ext):
            stem_query = stem_query[:-len(ext)]
            break

    # 3. Exact stem or filename match
    exact_matches = []
    for f in file_list:
        if f.name.lower() == query_clean or f.stem.lower() == stem_query:
            exact_matches.append(f)
    if len(exact_matches) == 1:
        return exact_matches[0], []
    elif len(exact_matches) > 1:
        return None, exact_matches

    # 4. Number match within filename (e.g., query '15' -> matches 'cloud_bin_15.ply' or 'scan15.csv')
    if query_clean.isdigit():
        num_matches = []
        for f in file_list:
            nums = re.findall(r"\d+", f.name)
            if query_clean in [str(int(n)) for n in nums]:
                num_matches.append(f)
        if len(num_matches) == 1:
            return num_matches[0], []
        elif len(num_matches) > 1:
            return None, num_matches

    # 5. Substring match
    sub_matches = [f for f in file_list if stem_query in f.name.lower() or stem_query in str(f).lower()]
    if len(sub_matches) == 1:
        return sub_matches[0], []

    return None, sub_matches


def resolve_dataset_key(query):
    """Resolves a query string to a dataset key."""
    q = query.strip().lower()
    for d_key, info in DATASETS.items():
        if q == d_key.lower() or q in [a.lower() for a in info.get("aliases", [])]:
            return d_key
    for d_key, info in DATASETS.items():
        if q in d_key.lower() or q in info["title"].lower():
            return d_key
    return None


def interactive_file_browser(files, title="Dataset Files", default_color="height"):
    """Interactive navigation inside a list of point cloud files."""
    color_mode = default_color
    page = 0
    page_size = 25

    while True:
        total_files = len(files)
        total_pages = (total_files + page_size - 1) // page_size if total_files > page_size else 1

        print(f"\n\033[1;95m=== {title} ({total_files} files) ===\033[0m")
        if total_pages > 1:
            print(f"\033[90mPage {page + 1} of {total_pages} (showing files {page * page_size} - {min((page + 1) * page_size - 1, total_files - 1)})\033[0m")

        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_files)

        for idx in range(start_idx, end_idx):
            f = files[idx]
            size_mb = f.stat().st_size / (1024 * 1024)
            disp_name = f.name
            print(f"  \033[94m[{idx:3d}]\033[0m {disp_name:<28s} \033[90m({size_mb:5.2f} MB)\033[0m")

        prompt_hints = []
        if total_pages > 1:
            prompt_hints.append("\033[93m'n'\033[0m next | \033[93m'p'\033[0m prev | \033[93m'all'\033[0m list all")
        prompt_hints.append("\033[93m'c'\033[0m color [" + color_mode + "]")
        prompt_hints.append("\033[93m'b'\033[0m back")
        prompt_hints.append("\033[93m'q'\033[0m quit")

        print(f"\nOptions: { ' | '.join(prompt_hints) }")
        try:
            choice = input("\033[96mSelect File Index or Name > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return "quit", color_mode

        if not choice:
            continue

        c_lower = choice.lower()
        if c_lower in ["q", "quit", "exit"]:
            return "quit", color_mode
        if c_lower in ["b", "back"]:
            return "back", color_mode
        if c_lower in ["n", "next"] and total_pages > 1:
            page = (page + 1) % total_pages
            continue
        if c_lower in ["p", "prev", "previous"] and total_pages > 1:
            page = (page - 1 + total_pages) % total_pages
            continue
        if c_lower == "all" and total_pages > 1:
            page_size = total_files
            page = 0
            continue
        if c_lower == "c":
            scheme_keys = [s[0] for s in COLOR_SCHEMES]
            curr_idx = scheme_keys.index(color_mode) if color_mode in scheme_keys else 0
            color_mode = scheme_keys[(curr_idx + 1) % len(scheme_keys)]
            print(f"\033[92mColor mode changed to: {color_mode}\033[0m")
            continue

        # Try to resolve selected file
        target_file, matches = resolve_file(choice, files)
        if target_file is None:
            if matches:
                print(f"\033[93mMultiple matches found for '{choice}':\033[0m")
                for m in matches[:10]:
                    idx_in_list = files.index(m)
                    print(f"  [{idx_in_list}] {m.name}")
                if len(matches) > 10:
                    print(f"  ... and {len(matches) - 10} more.")
            else:
                print(f"\033[91mNo file matched '{choice}'. Enter an index, filename, or 'b' for back.\033[0m")
            continue

        # Load and visualize the point cloud
        pcd = load_point_cloud(target_file)
        if pcd:
            new_color = visualize_pcd(pcd, target_file.name, color_mode=color_mode)
            if new_color:
                color_mode = new_color


def interactive_dataset_menu(default_color="neon"):
    """Top-level dataset selection CLI."""
    color_mode = default_color

    while True:
        print("\n\033[1;95m====================================================\033[0m")
        print("\033[1;95m       3D Point Cloud Interactive Dataset Viewer    \033[0m")
        print("\033[1;95m====================================================\033[0m")
        print("Available datasets:")

        keys = list(DATASETS.keys())
        for idx, k in enumerate(keys):
            info = DATASETS[k]
            file_count = len(get_dataset_files(k))
            status = f"({file_count} files)" if file_count > 0 else "\033[91m(not found)\033[0m"
            print(f"  \033[94m[{idx}]\033[0m \033[1m{k:<20s}\033[0m {info['title']:<24s} \033[90m{status}\033[0m")

        print("  \033[90m----------------------------------------------------\033[0m")
        print(f"  \033[94m[a]\033[0m \033[1mBrowse All\033[0m           All datasets combined")
        print(f"  \033[94m[c]\033[0m \033[1mColor Mode\033[0m           Current: [\033[95m{color_mode}\033[0m]")
        print(f"  \033[94m[q]\033[0m \033[1mQuit\033[0m")

        try:
            choice = input("\n\033[96mSelect Dataset [0-6, name, a, c, q] > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not choice:
            continue

        c_lower = choice.lower()
        if c_lower in ["q", "quit", "exit"]:
            print("Exiting viewer. Goodbye!")
            break

        if c_lower == "c":
            scheme_keys = [s[0] for s in COLOR_SCHEMES]
            curr_idx = scheme_keys.index(color_mode) if color_mode in scheme_keys else 0
            color_mode = scheme_keys[(curr_idx + 1) % len(scheme_keys)]
            print(f"\033[92mColor mode changed to: {color_mode}\033[0m")
            continue

        if c_lower in ["a", "all"]:
            all_files = get_all_workspace_files()
            if not all_files:
                print("\033[91mNo point cloud files found across datasets.\033[0m")
                continue
            ret, color_mode = interactive_file_browser(all_files, title="All Datasets Combined", default_color=color_mode)
            if ret == "quit":
                break
            continue

        # Check dataset match
        matched_key = resolve_dataset_key(choice)
        if matched_key:
            files = get_dataset_files(matched_key)
            if not files:
                print(f"\033[91mNo valid files found for {matched_key}.\033[0m")
                continue
            ret, color_mode = interactive_file_browser(files, title=DATASETS[matched_key]["title"], default_color=color_mode)
            if ret == "quit":
                break
            continue

        # Check if user typed a direct file path
        candidate_path = Path(choice)
        if candidate_path.exists() and candidate_path.is_file():
            pcd = load_point_cloud(candidate_path)
            if pcd:
                new_color = visualize_pcd(pcd, candidate_path.name, color_mode=color_mode)
                if new_color:
                    color_mode = new_color
            continue

        print(f"\033[91mInvalid choice: '{choice}'. Enter 0-{len(keys)-1}, dataset name, or 'q' to quit.\033[0m")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI Point Cloud Viewer for Stanford Bunny, PAVIN, Buddha, Dragon, 3DMatch, and ScanNet datasets."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Dataset name (e.g. 'bunny', 'buddha', '3dmatch', 'scannet'), direct file path, or scan name."
    )
    parser.add_argument(
        "file_query",
        nargs="?",
        default=None,
        help="Optional file index or filename query when target is a dataset (e.g. '0', 'bun045')."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Dataset identifier (e.g. bunny, pavin, buddha, dragon_stand, 3dmatch, 3dmatch2, 3dmatch3, scannet)."
    )
    parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="Specific file path or filename to open."
    )
    parser.add_argument(
        "--color",
        "-c",
        choices=COLOR_MODES,
        default="neon",
        help="Point cloud coloring mode: 'neon', 'rainbow', 'turbo', 'viridis', 'plasma', 'distance', 'normals', 'teal', 'gold', or 'original' (default: 'neon')."
    )
    args = parser.parse_args()

    # 1. Direct file passed via --file or positional arg
    direct_file = args.file or (args.target if args.target and Path(args.target).exists() and Path(args.target).is_file() else None)
    if direct_file:
        p = Path(direct_file)
        pcd = load_point_cloud(p)
        if pcd:
            visualize_pcd(pcd, p.name, color_mode=args.color)
        sys.exit(0)

    # 2. Dataset passed via --dataset or positional target
    dataset_req = args.dataset or args.target
    if dataset_req:
        matched_key = resolve_dataset_key(dataset_req)
        if matched_key:
            files = get_dataset_files(matched_key)
            if not files:
                print(f"\033[91mNo files found for dataset '{matched_key}'.\033[0m")
                sys.exit(1)

            # If a specific file within the dataset was requested
            file_req = args.file_query or args.file
            if file_req:
                selected, matches = resolve_file(file_req, files)
                if selected:
                    pcd = load_point_cloud(selected)
                    if pcd:
                        visualize_pcd(pcd, selected.name, color_mode=args.color)
                    sys.exit(0)
                else:
                    if matches:
                        print(f"\033[93mMultiple matches found for '{file_req}':\033[0m")
                        for m in matches:
                            print(f"  [{files.index(m)}] {m.name}")
                    else:
                        print(f"\033[91mNo file matched '{file_req}' in {matched_key}.\033[0m")
                    sys.exit(1)

            # Interactive browser for that specific dataset
            interactive_file_browser(files, title=DATASETS[matched_key]["title"], default_color=args.color)
            sys.exit(0)

        # Check if the positional query matches any file across all datasets
        all_files = get_all_workspace_files()
        selected, matches = resolve_file(dataset_req, all_files)
        if selected:
            pcd = load_point_cloud(selected)
            if pcd:
                visualize_pcd(pcd, selected.name, color_mode=args.color)
            sys.exit(0)
        elif matches:
            print(f"\033[93mMultiple matches across datasets for '{dataset_req}':\033[0m")
            for m in matches[:10]:
                print(f"  {m.relative_to(BASE_DIR)}")
            sys.exit(1)
        else:
            print(f"\033[91mUnknown dataset or file query: '{dataset_req}'.\033[0m")
            print("Available datasets: " + ", ".join(DATASETS.keys()))
            sys.exit(1)

    # 3. Interactive Main Menu
    interactive_dataset_menu(default_color=args.color)


if __name__ == "__main__":
    main()
