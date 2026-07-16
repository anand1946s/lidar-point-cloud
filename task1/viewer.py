import open3d as o3d
from pathlib import Path


PLY_PATH = Path(
    "task2/outputs_scannet/scene0027/ROBUST_ICP_SHOT_scene0027_01_vh_clean_scene0027_00_vh_clean/ROBUST_ICP_SHOT_scene0027_01_vh_clean_scene0027_00_vh_clean.ply"
)
#"task2/outputs_scannet/scene0024/ROBUST_ICP_SHOT/ROBUST_ICP_SHOT_scene0024_01_vh_clean_scene0024_00_vh_clean"

def main():

    pcd = o3d.io.read_point_cloud(
        str(PLY_PATH)
    )

    print(
        f"Loaded {len(pcd.points)} points"
    )

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=PLY_PATH.name,
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()