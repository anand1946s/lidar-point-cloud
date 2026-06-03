import open3d as o3d

from config import OUTPUT_DIR

pcd = o3d.io.read_point_cloud(
    str(OUTPUT_DIR / "final.ply")
)

print(pcd)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="final.ply"
)