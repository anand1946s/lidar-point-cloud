import open3d as o3d

#pcd = o3d.io.read_point_cloud("keypoint_outputs/ear_back_keypoints.ply")
pcd = o3d.io.read_point_cloud("bunny/bunny/data/ear_back.ply")

print(pcd)

o3d.visualization.draw_geometries([pcd])