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

# Find paths relative to task3
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT.parent))

from config import DATA_DIR, OUTPUT_DIR, VOXEL_SIZE

def parse_run_name(run_name):
    """
    Parses a run directory name into registration method, feature method,
    source scan name, and target scan name.
    E.g., "FGR_FPFH_scan000_scan100" -> ("FGR", "FPFH", "scan000", "scan100")
    """
    parts = run_name.split('_')
    if len(parts) >= 4:
        # Check if the last two parts correspond to source and target scan names
        if parts[-2].startswith("scan") and parts[-1].startswith("scan"):
            source_name = parts[-2]
            target_name = parts[-1]
            rem = parts[:-2]
            
            # Resolve registration and feature method
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
    """Loads raw target point cloud from the dataset folder."""
    if not str(scan_name).endswith(".csv"):
        scan_file = DATA_DIR / f"{scan_name}.csv"
    else:
        scan_file = DATA_DIR / scan_name

    if not scan_file.exists():
        print(f"\033[91mWarning: scan file not found at {scan_file}\033[0m")
        return None

    try:
        skip_rows = 0
        with open(scan_file, "r") as f:
            for idx, line in enumerate(f):
                if idx > 20:
                    break
                if "DATA" in line:
                    skip_rows = idx + 1
                    break
        
        points = np.loadtxt(scan_file, skiprows=skip_rows)
        if len(points.shape) == 2 and points.shape[1] >= 3:
            points = points[:, :3]
        elif len(points.shape) == 1 and len(points) >= 3:
            points = points[:3].reshape(1, 3)
        else:
            raise ValueError(f"Expected file to have shape (N, >=3), got {points.shape}")

        points = points[~np.all(points == 0, axis=1)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    except Exception as e:
        print(f"\033[91mError loading target scan {scan_name}: {e}\033[0m")
        return None

def apply_aesthetic_coloring(points):
    """Generates neon cyberpunk gradient colors for single cloud view."""
    vals = points[:, 2] # height (Z)
    val_min, val_max = vals.min(), vals.max()
    if val_max == val_min:
        return np.tile([0.0, 0.8, 0.7], (len(points), 1))
        
    norm_vals = (vals - val_min) / (val_max - val_min)
    
    color_start = np.array([0.15, 0.05, 0.35]) # Deep violet
    color_mid = np.array([0.85, 0.15, 0.55])   # Neon pink/magenta
    color_end = np.array([0.0, 0.9, 0.9])      # Bright cyan
    
    colors = np.zeros_like(points)
    mask = norm_vals < 0.5
    colors[mask] = (1 - norm_vals[mask][:, None]/0.5) * color_start + (norm_vals[mask][:, None]/0.5) * color_mid
    colors[~mask] = (1 - (norm_vals[~mask][:, None]-0.5)/0.5) * color_mid + ((norm_vals[~mask][:, None]-0.5)/0.5) * color_end
    
    return colors

def visualize_pcd_single(pcd, window_name):
    """Fallback visualizer for showing only the aligned cloud."""
    points = np.asarray(pcd.points)
    pcd.colors = o3d.utility.Vector3dVector(apply_aesthetic_coloring(points))
    
    print(f"\nShowing aligned cloud only: \033[92m{window_name}\033[0m")
    print(f"Number of points: \033[94m{len(points):,}\033[0m")
    print("Press \033[93m'q'\033[0m or close the window to exit the visualizer.")
    
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"PAVIN Point Cloud - {window_name}", width=1200, height=800)
        vis.add_geometry(pcd)
        
        opt = vis.get_render_option()
        opt.background_color = np.array([0.06, 0.06, 0.08])
        opt.point_size = 1.5
        
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"\033[91mGUI visualization failed: {e}\033[0m")

def visualize_registration(aligned_source, target, window_name):
    """Aesthetic visualizer showing aligned source (pink) and target (teal)."""
    # Aligned Source: Neon Pink/Magenta
    aligned_source.paint_uniform_color([1.0, 0.15, 0.58])
    # Target: Neon Cyan/Teal
    target.paint_uniform_color([0.0, 0.85, 0.85])

    print(f"\nShowing registration result: \033[92m{window_name}\033[0m")
    print(f"Aligned Source points: \033[94m{len(aligned_source.points):,}\033[0m")
    print(f"Target points:         \033[94m{len(target.points):,}\033[0m")
    print("Color legend:")
    print("  - \033[95mNeon Pink/Magenta\033[0m: Aligned Source Cloud")
    print("  - \033[96mNeon Cyan/Teal\033[0m: Target Cloud")
    print("Press \033[93m'q'\033[0m or close the window to exit the visualizer.")

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"PAVIN Registration Viewer - {window_name}", width=1200, height=800)
        vis.add_geometry(aligned_source)
        vis.add_geometry(target)
        
        opt = vis.get_render_option()
        opt.background_color = np.array([0.06, 0.06, 0.08]) # sleek dark background
        opt.point_size = 2.0                                # point size for contrast
        
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"\n\033[91mFailed to launch visualizer GUI: {e}\033[0m")
        print("Attempting fallback visualization (draw_geometries)...")
        try:
            o3d.visualization.draw_geometries([aligned_source, target], window_name=window_name, width=1200, height=800)
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
            
    if target is None:
        print("\033[93mWarning: Target scan could not be loaded. Visualizing aligned source only.\033[0m")
        visualize_pcd_single(aligned_source, run_dir.name)
    else:
        visualize_registration(aligned_source, target, run_dir.name)
        
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
    return [r for r in runs if query in r.name.lower()]

def resolve_exact_run(choice, all_runs):
    """Checks if input matches any run name exactly (case-insensitive)."""
    for r in all_runs:
        if r.name.lower() == choice.lower():
            return r
    return None

def main():
    parser = argparse.ArgumentParser(description="Visualize any point cloud registration output from task3 outputs.")
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help="Optional directory name of the run output to view directly (e.g. 'ICP_NONE_scan000_scan100')."
    )
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        print(f"\033[91mError: Outputs directory not found at {OUTPUT_DIR}\033[0m")
        sys.exit(1)

    # Get all runs that contain a .ply file
    all_runs = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir() and any(d.glob("*.ply"))])

    if not all_runs:
        print(f"\033[91mNo valid output runs found in {OUTPUT_DIR}\033[0m")
        sys.exit(1)

    # Direct CLI loading
    if args.run is not None:
        selected_run = resolve_exact_run(args.run, all_runs)
        if selected_run is None:
            # Try partial match
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
        # Re-fetch in case runs changed
        all_runs = sorted([d for d in OUTPUT_DIR.iterdir() if d.is_dir() and any(d.glob("*.ply"))])
        filtered_runs = get_filtered_runs(all_runs, filter_query)
        
        print("\n\033[1;95m=== PAVIN Point Cloud Registration Viewer ===\033[0m")
        if filter_query:
            print(f"Filtered by: '\033[93m{filter_query}\033[0m' ({len(filtered_runs)} of {len(all_runs)} runs)")
        else:
            print(f"Total available runs: \033[94m{len(all_runs)}\033[0m")
            
        print("-" * 85)
        for idx, run_dir in enumerate(filtered_runs):
            reg, feat, src, tgt = parse_run_name(run_dir.name)
            if reg:
                info_str = f"Reg: {reg:<12} | Feat: {feat:<10} | {src} -> {tgt}"
            else:
                info_str = run_dir.name
            print(f"  \033[94m[{idx:2d}]\033[0m {info_str}")
            
        if not filtered_runs:
            print("  \033[91mNo runs match the current query.\033[0m")

        print("-" * 85)
        print("Commands:")
        print("  - \033[96m<Index>\033[0m (e.g. \033[96m0\033[0m) to view a specific run")
        print("  - \033[96m<Keyword>\033[0m (e.g. \033[96mFGR\033[0m, \033[96mSHOT\033[0m) to filter the list")
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
