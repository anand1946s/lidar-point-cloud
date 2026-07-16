#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Force X11/XWayland backend on Linux to avoid Wayland window placement & GLEW errors in Open3D/GLFW
if sys.platform.startswith("linux"):
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d

# Find the directory of this script
BASE_DIR = Path(__file__).resolve().parent

def sort_key(p):
    """Sort key helper to ensure cloud_bin_2.ply comes before cloud_bin_10.ply."""
    stem = p.stem
    if stem.startswith("cloud_bin_"):
        try:
            return (0, int(stem.replace("cloud_bin_", "")))
        except ValueError:
            pass
    return (1, stem)

def find_3dmatch_files():
    """Finds all PLY files within the 3dmatch dataset directory, looking in multiple logical places."""
    search_dirs = [
        BASE_DIR / "3dmatch",
        Path.cwd() / "3dmatch",
    ]
    # Also search recursively under BASE_DIR as a fallback
    for d in BASE_DIR.glob("**/3dmatch"):
        if d.is_dir() and d not in search_dirs:
            search_dirs.append(d)

    for d in search_dirs:
        if d.exists() and d.is_dir():
            ply_files = list(d.glob("*.ply"))
            if ply_files:
                return sorted(ply_files, key=sort_key)
    return []

def apply_aesthetic_coloring(pcd, points, color_mode="cyberpunk"):
    """
    Applies custom colormaps to the point cloud.
    Modes:
      - 'cyberpunk': Z-based Neon Cyberpunk gradient (Deep violet -> Vibrant Magenta -> Cyan).
      - 'normals': Color map based on normal vector orientations.
      - 'rainbow': Multi-segment spectral height gradient.
      - 'distance': Heatmap showing distance from centroid.
      - 'uniform': Clean modern uniform neon cyan/teal.
    """
    if len(points) == 0:
        return np.zeros((0, 3))

    if color_mode == "uniform":
        return np.tile([0.0, 0.9, 0.7], (len(points), 1))

    if color_mode == "normals":
        try:
            if not pcd.has_normals():
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            normals = np.asarray(pcd.normals)
            if len(normals) == len(points):
                # Map [-1, 1] to [0, 1] for RGB visualization
                colors = (normals + 1.0) / 2.0
                return np.clip(colors, 0.0, 1.0)
        except Exception as e:
            print(f"\033[93m[Warning] Could not estimate normals for coloring: {e}\033[0m")
        return np.tile([0.8, 0.8, 0.8], (len(points), 1))

    if color_mode == "distance":
        centroid = np.mean(points, axis=0)
        vals = np.linalg.norm(points - centroid, axis=1)
    else:  # Z-height
        vals = points[:, 2]

    val_min, val_max = vals.min(), vals.max()
    if val_max == val_min:
        return np.tile([0.0, 0.9, 0.7], (len(points), 1))

    norm_vals = (vals - val_min) / (val_max - val_min)

    if color_mode == "rainbow":
        # Blue -> Cyan -> Green -> Yellow -> Red
        colors = np.zeros_like(points)
        c0 = np.array([0.0, 0.0, 1.0])
        c1 = np.array([0.0, 1.0, 1.0])
        c2 = np.array([0.0, 1.0, 0.0])
        c3 = np.array([1.0, 1.0, 0.0])
        c4 = np.array([1.0, 0.0, 0.0])

        # Segment 1 [0.0, 0.25]
        m1 = (norm_vals >= 0.0) & (norm_vals < 0.25)
        t = norm_vals[m1] / 0.25
        colors[m1] = (1 - t)[:, None] * c0 + t[:, None] * c1

        # Segment 2 [0.25, 0.50]
        m2 = (norm_vals >= 0.25) & (norm_vals < 0.5)
        t = (norm_vals[m2] - 0.25) / 0.25
        colors[m2] = (1 - t)[:, None] * c1 + t[:, None] * c2

        # Segment 3 [0.50, 0.75]
        m3 = (norm_vals >= 0.5) & (norm_vals < 0.75)
        t = (norm_vals[m3] - 0.5) / 0.25
        colors[m3] = (1 - t)[:, None] * c2 + t[:, None] * c3

        # Segment 4 [0.75, 1.00]
        m4 = (norm_vals >= 0.75)
        t = np.clip((norm_vals[m4] - 0.75) / 0.25, 0.0, 1.0)
        colors[m4] = (1 - t)[:, None] * c3 + t[:, None] * c4
        return colors

    elif color_mode == "distance":
        # Fire & Ice gradient: Gold/Yellow -> Orange -> Dark Purple
        c_start = np.array([1.0, 0.8, 0.2])
        c_mid = np.array([1.0, 0.3, 0.0])
        c_end = np.array([0.1, 0.0, 0.25])

        colors = np.zeros_like(points)
        mask = norm_vals < 0.5
        colors[mask] = (1 - norm_vals[mask][:, None]/0.5) * c_start + (norm_vals[mask][:, None]/0.5) * c_mid
        colors[~mask] = (1 - (norm_vals[~mask][:, None]-0.5)/0.5) * c_mid + ((norm_vals[~mask][:, None]-0.5)/0.5) * c_end
        return colors

    else:  # cyberpunk
        # Deep violet -> Vibrant Magenta -> Cyan
        color_start = np.array([0.15, 0.05, 0.35])
        color_mid = np.array([0.85, 0.15, 0.55])
        color_end = np.array([0.0, 0.9, 0.9])

        colors = np.zeros_like(points)
        mask = norm_vals < 0.5
        colors[mask] = (1 - norm_vals[mask][:, None]/0.5) * color_start + (norm_vals[mask][:, None]/0.5) * color_mid
        colors[~mask] = (1 - (norm_vals[~mask][:, None]-0.5)/0.5) * color_mid + ((norm_vals[~mask][:, None]-0.5)/0.5) * color_end
        return colors

class VisualizerState:
    """Manages the visualizer state and coordinates key callbacks."""
    def __init__(self, pcd, filepath, color_mode="cyberpunk"):
        self.filepath = filepath
        self.original_pcd = pcd
        self.display_pcd = o3d.geometry.PointCloud()
        
        # Populate initial point coordinates and geometry attributes
        self.display_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            self.display_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        if pcd.has_normals():
            self.display_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))

        self.color_modes = ["cyberpunk", "normals", "rainbow", "distance", "uniform"]
        self.color_mode = color_mode
        self.is_downsampled = False

        # Paint the initial view
        self.update_colors()

    def update_colors(self):
        pts = np.asarray(self.display_pcd.points)
        colors = apply_aesthetic_coloring(self.display_pcd, pts, self.color_mode)
        self.display_pcd.colors = o3d.utility.Vector3dVector(colors)

    def cycle_colors(self):
        current_idx = self.color_modes.index(self.color_mode)
        next_idx = (current_idx + 1) % len(self.color_modes)
        self.color_mode = self.color_modes[next_idx]
        print(f"\n\033[92m[State] Switched colormap to: {self.color_mode.upper()}\033[0m")
        self.update_colors()

    def toggle_downsample(self):
        self.is_downsampled = not self.is_downsampled
        if self.is_downsampled:
            bbox = self.original_pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_max_bound() - bbox.get_min_bound()
            avg_extent = np.mean(extent)
            voxel_size = avg_extent / 100.0  # Voxel size is ~1% of bounding box dimensions
            pcd_down = self.original_pcd.voxel_down_sample(voxel_size)
            print(f"\n\033[93m[State] Downsampled to voxel size {voxel_size:.4f} ({len(pcd_down.points):,} points)\033[0m")
        else:
            pcd_down = self.original_pcd
            print(f"\n\033[92m[State] Restored to full resolution ({len(pcd_down.points):,} points)\033[0m")

        self.display_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_down.points))
        if pcd_down.has_normals():
            self.display_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd_down.normals))
        else:
            self.display_pcd.normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        
        self.update_colors()

def print_help_menu():
    print("\n\033[1;95m=== Interactive Visualizer Controls ===\033[0m")
    print("  \033[93m[C]\033[0m : Cycle Colormap (CYBERPUNK -> NORMALS -> RAINBOW -> DISTANCE -> UNIFORM)")
    print("  \033[93m[D]\033[0m : Toggle Voxel Downsampling (approx 1% of extent)")
    print("  \033[93m[P]\033[0m : Print Point Cloud Statistics (console output)")
    print("  \033[93m[H]\033[0m : Print this Help Menu")
    print("  \033[93m[+/-]\033[0m : Increase / Decrease Point Size (standard Open3D)")
    print("  \033[93m[[/]]\033[0m : Increase / Decrease Field of View (standard Open3D)")
    print("  \033[93m[R]\033[0m : Reset Viewpoint (standard Open3D)")
    print("  \033[93m[Q] / [Esc]\033[0m : Close Visualizer Window")

def visualize_pcd(pcd, filepath, initial_color_mode="cyberpunk", point_size=1.5):
    """Launches the GLFW Visualizer with dynamic state controls and callbacks."""
    state = VisualizerState(pcd, filepath, initial_color_mode)
    window_name = f"3DMatch Point Cloud Viewer - {filepath.name}"
    
    print(f"\nShowing: \033[92m{filepath.name}\033[0m")
    print(f"Total Points: \033[94m{len(pcd.points):,}\033[0m")
    
    print_help_menu()

    try:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_name, width=1200, height=800)
        vis.add_geometry(state.display_pcd)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.06, 0.06, 0.08])
        opt.point_size = point_size

        # Define callbacks
        def key_c_callback(vis):
            state.cycle_colors()
            vis.update_geometry(state.display_pcd)
            vis.update_renderer()
            return True

        def key_d_callback(vis):
            state.toggle_downsample()
            vis.update_geometry(state.display_pcd)
            vis.update_renderer()
            return True

        def key_p_callback(vis):
            pts = np.asarray(state.display_pcd.points)
            bbox = state.display_pcd.get_axis_aligned_bounding_box()
            min_b = bbox.get_min_bound()
            max_b = bbox.get_max_bound()
            extent = max_b - min_b
            centroid = np.mean(pts, axis=0) if len(pts) > 0 else np.array([0.0, 0.0, 0.0])
            
            print("\n\033[1;94m=== Point Cloud Stats ===\033[0m")
            print(f"  File:            {state.filepath.name}")
            print(f"  Points:          {len(pts):,}")
            print(f"  Color Mode:      {state.color_mode.upper()}")
            print(f"  Downsampled:     {state.is_downsampled}")
            print(f"  Min Bound:       [{min_b[0]:.3f}, {min_b[1]:.3f}, {min_b[2]:.3f}]")
            print(f"  Max Bound:       [{max_b[0]:.3f}, {max_b[1]:.3f}, {max_b[2]:.3f}]")
            print(f"  Extent (X,Y,Z):  [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
            print(f"  Centroid:        [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
            return True

        def key_h_callback(vis):
            print_help_menu()
            return True

        # Register keys for both lowercase and uppercase GLFW bindings
        for key in [ord('C'), ord('c')]:
            vis.register_key_callback(key, key_c_callback)
        for key in [ord('D'), ord('d')]:
            vis.register_key_callback(key, key_d_callback)
        for key in [ord('P'), ord('p')]:
            vis.register_key_callback(key, key_p_callback)
        for key in [ord('H'), ord('h')]:
            vis.register_key_callback(key, key_h_callback)

        vis.run()
        vis.destroy_window()

    except Exception as e:
        print(f"\n\033[91mFailed to launch interactive Visualizer GUI: {e}\033[0m")
        print("Falling back to standard non-interactive draw_geometries...")
        try:
            o3d.visualization.draw_geometries([state.display_pcd], window_name=window_name, width=1200, height=800)
        except Exception as e2:
            print(f"\033[91mFallback visualization also failed: {e2}\033[0m")

def resolve_file(query, files):
    """Resolves user search query or index to a specific path from files."""
    query_str = str(query).strip()
    if not query_str:
        return None, []

    path_check = Path(query_str)
    if path_check.exists() and path_check.is_file():
        return path_check, []

    if query_str.isdigit():
        idx = int(query_str)
        if 0 <= idx < len(files):
            return files[idx], []

    query_clean = query_str.lower()
    
    exact_matches = []
    for f in files:
        name_lower = f.name.lower()
        stem_lower = f.stem.lower()
        if (name_lower == query_clean or 
            stem_lower == query_clean or 
            stem_lower == f"cloud_bin_{query_clean}"):
            exact_matches.append(f)

    if len(exact_matches) == 1:
        return exact_matches[0], []
    elif len(exact_matches) > 1:
        return None, exact_matches

    substring_matches = [f for f in files if query_clean in f.name.lower()]
    if len(substring_matches) == 1:
        return substring_matches[0], []

    return None, substring_matches

def load_ply_file(filepath, downsample_size=None):
    """Loads point cloud geometry from PLY format."""
    try:
        pcd = o3d.io.read_point_cloud(str(filepath))
        if len(pcd.points) == 0:
            raise ValueError("Point cloud contains 0 points.")
        if downsample_size:
            pcd = pcd.voxel_down_sample(downsample_size)
        return pcd
    except Exception as e:
        print(f"\n\033[91mError loading file {filepath.name}: {e}\033[0m")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="A premium interactive 3D point cloud visualizer for the 3DMatch dataset."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Index (e.g. '0'), file name (e.g. 'cloud_bin_0.ply'), query string ('bin_10'), or path."
    )
    parser.add_argument(
        "--color",
        choices=["cyberpunk", "normals", "rainbow", "distance", "uniform"],
        default="cyberpunk",
        help="Initial colormap mode (default: 'cyberpunk')."
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Initial point size for visualization (default: 1.5)."
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=None,
        help="Voxel size for initial downsampling (optional)."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print list of all available files in the 3DMatch dataset and exit."
    )
    args = parser.parse_args()

    files = find_3dmatch_files()
    if not files:
        print("\033[91mNo point clouds found in the '3dmatch' dataset directory.\033[0m")
        sys.exit(1)

    if args.list:
        print("\n\033[1;95m=== Available 3DMatch Dataset Files ===\033[0m")
        for idx, f in enumerate(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  \033[94m[{idx:2d}]\033[0m {f.name:<18s} ({size_mb:.2f} MB)")
        sys.exit(0)

    # 1. Direct CLI Query Mode
    if args.query is not None:
        selected_file, matches = resolve_file(args.query, files)
        if selected_file is None:
            if matches:
                print(f"\033[93mMultiple matches found for query '{args.query}':\033[0m")
                for i, m in enumerate(matches):
                    print(f"  [{i}] {m.name}")
            else:
                print(f"\033[91mCould not find any 3DMatch file matching query '{args.query}'.\033[0m")
                print("Use --list to view all files.")
            sys.exit(1)

        pcd = load_ply_file(selected_file, args.downsample)
        if pcd:
            visualize_pcd(pcd, selected_file, args.color, args.point_size)
        sys.exit(0)

    # 2. Interactive Shell Mode
    while True:
        print("\n\033[1;95m=== 3DMatch Point Cloud Dataset Viewer ===\033[0m")
        print("Available point cloud files:")
        for idx, f in enumerate(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  \033[94m[{idx:2d}]\033[0m {f.name:<18s} ({size_mb:.2f} MB)")

        print("\nEnter a file index (e.g. 5), search term (e.g. bin_12), or '\033[93mq\033[0m' to quit:")
        try:
            choice = input("\033[96mSelect File > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break

        selected_file = None
        if choice.isdigit() and 0 <= int(choice) < len(files):
            selected_file = files[int(choice)]
        else:
            selected_file, matches = resolve_file(choice, files)
            if selected_file is None:
                if matches:
                    print(f"\033[93mMultiple matches found for '{choice}':\033[0m")
                    for m in matches:
                        print(f"  - {m.name}")
                else:
                    print(f"\033[91mNo files matching query '{choice}' were found.\033[0m")
                continue

        if selected_file:
            pcd = load_ply_file(selected_file, args.downsample)
            if pcd:
                visualize_pcd(pcd, selected_file, args.color, args.point_size)

if __name__ == "__main__":
    main()
