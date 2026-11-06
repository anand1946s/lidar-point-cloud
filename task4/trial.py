#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import os
import copy

# Match task4/viewer.py runtime behavior on Linux to avoid GLFW/GLEW issues.
if sys.platform.startswith("linux"):
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_DIR, SCENE_NAME


def get_dataset_files(dataset_dir: Path):
    return sorted(dataset_dir.glob("*.ply"))


def print_dataset(files):
    print(f"\n{SCENE_NAME} dataset: {DATA_DIR}")
    for idx, file_path in enumerate(files):
        print(f"[{idx:3d}] {file_path.name}")


def load_cloud(file_path: Path):
    pcd = o3d.io.read_point_cloud(str(file_path))
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty: {file_path}")
    if not pcd.has_colors():
        print(
            f"Note: {file_path.name} has no RGB colors in the file. "
            "Showing raw geometry only.",
            file=sys.stderr,
        )
    return pcd


def print_help_menu():
    print("\n=== Interactive Dataset Viewer Controls ===")
    print("  [N] : Next file")
    print("  [B] : Previous file")
    print("  [I] : Show current cloud as a 2D color image")
    print("  [P] : Print current file stats")
    print("  [H] : Show this help")
    print("  [Q] / [Esc] : Close window")


def choose_index_interactively(files):
    print_dataset(files)
    while True:
        choice = input("\nEnter file index to open (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(files):
                return idx
        print(f"Invalid input. Enter an index from 0 to {len(files)-1}.")


class DatasetViewerState:
    def __init__(self, files, index):
        self.files = files
        self.index = index
        self.geometry = o3d.geometry.PointCloud()
        self._load_current()

    @property
    def current_file(self):
        return self.files[self.index]

    def _load_current(self):
        pcd = load_cloud(self.current_file)
        self.geometry.points = copy.deepcopy(pcd.points)
        if pcd.has_colors():
            self.geometry.colors = copy.deepcopy(pcd.colors)
        else:
            self.geometry.colors = o3d.utility.Vector3dVector()
        if pcd.has_normals():
            self.geometry.normals = copy.deepcopy(pcd.normals)
        else:
            self.geometry.normals = o3d.utility.Vector3dVector()

    def move(self, step):
        self.index = (self.index + step) % len(self.files)
        self._load_current()
        print(f"\nNow viewing: [{self.index}] {self.current_file.name}")

    def print_stats(self):
        points = np.asarray(self.geometry.points)
        print(f"\nFile: [{self.index}] {self.current_file.name}")
        print(f"Points: {len(points):,}")
        if len(points) > 0:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            print(f"Bounds min: ({mins[0]:.4f}, {mins[1]:.4f}, {mins[2]:.4f})")
            print(f"Bounds max: ({maxs[0]:.4f}, {maxs[1]:.4f}, {maxs[2]:.4f})")
        print("Colors: original" if self.geometry.has_colors() else "Colors: none in file")


def render_cloud_as_color_image(pcd, width=1280, height=800):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Cannot render image from an empty point cloud.")

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.tile(np.array([[0.2, 0.6, 0.9]]), (len(points), 1))

    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    # Project to the two dominant principal directions for a stable 2D view.
    proj_2d = centered @ vt[:2].T
    depth = centered @ vt[2]

    min_xy = proj_2d.min(axis=0)
    max_xy = proj_2d.max(axis=0)
    span_xy = np.maximum(max_xy - min_xy, 1e-8)
    norm_xy = (proj_2d - min_xy) / span_xy

    px = np.clip((norm_xy[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
    py = np.clip(((1.0 - norm_xy[:, 1]) * (height - 1)).astype(np.int32), 0, height - 1)

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_order = np.argsort(depth)  # Draw far points first, near points last.

    for idx in draw_order:
        x = px[idx]
        y = py[idx]
        c = (colors[idx] * 255.0).astype(np.uint8)
        x0 = max(0, x - 1)
        x1 = min(width, x + 2)
        y0 = max(0, y - 1)
        y1 = min(height, y + 2)
        image[y0:y1, x0:x1] = c

    return image


def show_cloud_image(pcd, title):
    image = render_cloud_as_color_image(pcd)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_interactive_viewer(files, start_index):
    state = DatasetViewerState(files, start_index)
    print(f"\nOpening: [{start_index}] {state.current_file.name}")
    print_help_menu()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{SCENE_NAME} Dataset Viewer",
        width=1280,
        height=800
    )
    vis.add_geometry(state.geometry)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = 2.0

    def refresh():
        vis.clear_geometries()
        vis.add_geometry(state.geometry)
        vis.update_geometry(state.geometry)
        vis.update_renderer()

    def next_file(_):
        state.move(1)
        refresh()
        return True

    def prev_file(_):
        state.move(-1)
        refresh()
        return True

    def print_stats(_):
        state.print_stats()
        return True

    def show_image(_):
        show_cloud_image(state.geometry, f"{SCENE_NAME}: {state.current_file.name}")
        return True

    def show_help(_):
        print_help_menu()
        return True

    for key in [ord("N"), ord("n")]:
        vis.register_key_callback(key, next_file)
    for key in [ord("B"), ord("b")]:
        vis.register_key_callback(key, prev_file)
    for key in [ord("P"), ord("p")]:
        vis.register_key_callback(key, print_stats)
    for key in [ord("I"), ord("i")]:
        vis.register_key_callback(key, show_image)
    for key in [ord("H"), ord("h")]:
        vis.register_key_callback(key, show_help)

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Open and view point clouds from the 3dmatch3 dataset."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scans and exit.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index of the scan to open.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Exact file name to open (e.g., cloud_bin_5.ply).",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Open the selected scan as a 2D color image instead of 3D interactive viewer.",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Dataset folder not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    files = get_dataset_files(DATA_DIR)
    if not files:
        print(f"No .ply files found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print_dataset(files)
        return

    selected_index = None

    if args.name is not None:
        for idx, file_path in enumerate(files):
            if file_path.name == args.name:
                selected_index = idx
                break
        if selected_index is None:
            print(f"File not found in dataset: {args.name}", file=sys.stderr)
            sys.exit(1)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(files):
            print(
                f"Invalid index {args.index}. Use --list to see valid indices 0..{len(files)-1}.",
                file=sys.stderr,
            )
            sys.exit(1)
        selected_index = args.index
    else:
        selected_index = choose_index_interactively(files)
        if selected_index is None:
            return

    try:
        selected_pcd = load_cloud(files[selected_index])
        if args.image:
            show_cloud_image(selected_pcd, f"{SCENE_NAME}: {files[selected_index].name}")
        else:
            run_interactive_viewer(files, selected_index)
    except Exception as exc:
        print(f"Interactive visualizer failed: {exc}", file=sys.stderr)
        print("Trying Open3D draw_geometries fallback...", file=sys.stderr)
        pcd = load_cloud(files[selected_index])
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=files[selected_index].name,
            width=1280,
            height=800,
        )


if __name__ == "__main__":
    main()
