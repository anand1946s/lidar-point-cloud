import open3d as o3d
from pathlib import Path


PLY_PATH = Path(
    "task1/outputs/FGR_FPFH_bun000_bun045/FGR_FPFH_bun000_bun045.ply"
)


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