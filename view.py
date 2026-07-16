#!/usr/bin/env python3
import sys
import os

# Force X11/XWayland backend on Linux to avoid Wayland window placement & GLEW errors in Open3D/GLFW
if sys.platform.startswith("linux"):
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["XDG_SESSION_TYPE"] = "x11"

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

# Find the workspace root
BASE_DIR = Path(__file__).resolve().parent
PAVIN_DIR = BASE_DIR / "pavin"

def find_csv_scans():
    """Finds all scan*.csv files recursively under pavin_dir."""
    if not PAVIN_DIR.exists():
        print(f"Error: PAVIN directory not found at {PAVIN_DIR}")
        return []
    # Find all CSV files matching scan*.csv recursively
    csv_files = sorted(list(PAVIN_DIR.glob("**/scan*.csv")))
    # If nothing found, try finding any .csv file in pavin
    if not csv_files:
        csv_files = sorted(list(PAVIN_DIR.glob("**/*.csv")))
    return csv_files

def apply_aesthetic_coloring(points, color_mode="height"):
    """
    Applies a premium gradient color map to the point cloud.
    Modes:
      - 'height': Gradient based on Z coordinate (default, beautiful neon height map).
      - 'distance': Gradient based on distance from the sensor origin.
      - 'uniform': Uniform clean light teal/green.
    """
    if color_mode == "uniform":
        return np.tile([0.0, 0.8, 0.7], (len(points), 1))
        
    if color_mode == "distance":
        vals = np.linalg.norm(points, axis=1)
    else: # height (Z)
        vals = points[:, 2]
        
    val_min, val_max = vals.min(), vals.max()
    if val_max == val_min:
        return np.tile([0.0, 0.8, 0.7], (len(points), 1))
        
    # Normalize values to [0, 1] for colormap interpolation
    norm_vals = (vals - val_min) / (val_max - val_min)
    
    # Premium Neon Cyberpunk gradient: Deep violet -> Vibrant Magenta -> Cyan
    color_start = np.array([0.15, 0.05, 0.35]) # Deep violet (low/close)
    color_mid = np.array([0.85, 0.15, 0.55])   # Neon pink/magenta (mid)
    color_end = np.array([0.0, 0.9, 0.9])      # Bright cyan (high/far)
    
    colors = np.zeros_like(points)
    mask = norm_vals < 0.5
    # Interpolate [0, 0.5] -> [color_start, color_mid]
    colors[mask] = (1 - norm_vals[mask][:, None]/0.5) * color_start + (norm_vals[mask][:, None]/0.5) * color_mid
    # Interpolate [0.5, 1.0] -> [color_mid, color_end]
    colors[~mask] = (1 - (norm_vals[~mask][:, None]-0.5)/0.5) * color_mid + ((norm_vals[~mask][:, None]-0.5)/0.5) * color_end
    
    return colors

def load_scan(file_path):
    """Loads point cloud coordinates from a space-separated CSV/TXT file."""
    try:
        # Check if there is a header in the first few lines of the file
        skip_rows = 0
        with open(file_path, "r") as f:
            for idx, line in enumerate(f):
                if idx > 20: # Only scan first 20 lines
                    break
                if "DATA" in line:
                    skip_rows = idx + 1
                    break

        # PAVIN scan CSV files are space-separated XYZ files
        points = np.loadtxt(file_path, skiprows=skip_rows)
        if len(points.shape) == 2 and points.shape[1] >= 3:
            points = points[:, :3]
        elif len(points.shape) == 1 and len(points) >= 3:
            points = points[:3].reshape(1, 3)
        else:
            raise ValueError(f"Expected file to have shape (N, >=3), got {points.shape}")

        # Filter out dummy viewpoint/origin point [0, 0, 0]
        points = points[~np.all(points == 0, axis=1)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    except Exception as e:
        print(f"\n\033[91mError loading {file_path.name}: {e}\033[0m")
        return None

def visualize_pcd(pcd, window_name, color_mode="height"):
    """Visualizes the point cloud with a premium dark theme and custom colors."""
    points = np.asarray(pcd.points)
    colors = apply_aesthetic_coloring(points, color_mode)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\nShowing: \033[92m{window_name}\033[0m")
    print(f"Number of points: \033[94m{len(points):,}\033[0m")
    print("Press \033[93m'q'\033[0m or close the window to exit the visualizer.")
    
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"PAVIN Point Cloud Viewer - {window_name}", width=1200, height=800)
        vis.add_geometry(pcd)
        
        opt = vis.get_render_option()
        opt.background_color = np.array([0.06, 0.06, 0.08]) # sleek dark background
        opt.point_size = 1.5                                # customized point size for clarity
        
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"\n\033[91mFailed to launch visualizer GUI: {e}\033[0m")
        print("This might happen if you are running in a headless environment or SSH without X-forwarding.")
        print("Attempting fallback visualization (standard draw_geometries)...")
        try:
            o3d.visualization.draw_geometries([pcd], window_name=window_name, width=1200, height=800)
        except Exception as e2:
            print(f"\033[91mFallback visualization also failed: {e2}\033[0m")

def resolve_scan(query, scans):
    """
    Resolves a scan query string to a file from the list of scan files.
    Tries:
    1. Direct file path check.
    2. Exact filename match (case-insensitive).
    3. Exact name match without extension (case-insensitive).
    4. Exact match with 'scan' prefix (e.g., '300' -> 'scan300.csv').
    5. Substring match.
    """
    query_str = str(query).strip()
    if not query_str:
        return None, []

    # 1. Direct path check
    path_check = Path(query_str)
    if path_check.exists() and path_check.is_file():
        return path_check, []

    query_clean = query_str.lower()
    if query_clean.endswith(".csv"):
        query_clean = query_clean[:-4]

    # 2, 3, & 4. Exact matches (e.g., '300' matches 'scan300.csv' or 'scan300')
    exact_matches = []
    for s in scans:
        s_name_lower = s.name.lower()
        s_stem_lower = s.stem.lower()
        
        if (s_name_lower == query_clean or 
            s_stem_lower == query_clean or 
            s_stem_lower == f"scan{query_clean}"):
            exact_matches.append(s)
            
    if len(exact_matches) == 1:
        return exact_matches[0], []
    elif len(exact_matches) > 1:
        return None, exact_matches

    # 5. Substring matches as a fallback
    substring_matches = [s for s in scans if query_clean in s.name.lower()]
    if len(substring_matches) == 1:
        return substring_matches[0], []
    
    return None, substring_matches

def main():
    parser = argparse.ArgumentParser(description="Interactive and command-line point cloud viewer for the PAVIN dataset.")
    parser.add_argument(
        "scan",
        nargs="?",
        default=None,
        help="Scan identifier (e.g., '300', 'scan300', 'scan300.csv') or direct file path."
    )
    parser.add_argument(
        "--color",
        choices=["height", "distance", "uniform"],
        default="height",
        help="Color mapping mode: height-based, distance-based, or uniform teal (default: 'height')."
    )
    args = parser.parse_args()

    scans = find_csv_scans()
    if not scans:
        print("\033[91mNo point cloud scans found under the 'pavin' directory.\033[0m")
        sys.exit(1)

    # 1. If a specific scan has been requested via command line:
    if args.scan is not None:
        selected_file, matches = resolve_scan(args.scan, scans)
        if selected_file is None:
            if matches:
                print(f"\033[93mMultiple matches found for '{args.scan}':\033[0m")
                for i, m in enumerate(matches):
                    print(f"  [{i}] {m.name}")
            else:
                print(f"\033[91mCould not find any scan matching '{args.scan}' in the PAVIN dataset.\033[0m")
                print("Available scans:")
                for s in scans:
                    print(f"  - {s.name}")
            sys.exit(1)

        pcd = load_scan(selected_file)
        if pcd:
            visualize_pcd(pcd, selected_file.name, args.color)
        sys.exit(0)

    # 2. Interactive Selection Menu (if no command line scan was provided):
    while True:
        print("\n\033[1;95m=== PAVIN Dataset Point Cloud Viewer ===\033[0m")
        print("Available point cloud scans:")
        
        for idx, scan_path in enumerate(scans):
            size_mb = scan_path.stat().st_size / (1024 * 1024)
            print(f"  \033[94m[{idx:2d}]\033[0m {scan_path.name:<15s} ({size_mb:.2f} MB)")
            
        print("\nEnter a scan index (e.g., 15), name (e.g., 300), or '\033[93mq\033[0m' to quit:")
        try:
            choice = input("\033[96mSelect Scan > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
            
        if choice.lower() in ['q', 'quit', 'exit']:
            print("Exiting viewer. Goodbye!")
            break
            
        selected_file = None
        if choice.isdigit() and 0 <= int(choice) < len(scans):
            selected_file = scans[int(choice)]
        else:
            selected_file, matches = resolve_scan(choice, scans)
            if selected_file is None:
                if matches:
                    print(f"\033[93mMultiple matches found for '{choice}':\033[0m")
                    for m in matches:
                        print(f"  - {m.name}")
                else:
                    print(f"\033[91mNo scan matches the name or query '{choice}'.\033[0m")
                continue

        if selected_file:
            pcd = load_scan(selected_file)
            if pcd:
                visualize_pcd(pcd, selected_file.name, args.color)

if __name__ == "__main__":
    main()
