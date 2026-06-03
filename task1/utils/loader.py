import open3d as o3d

def load_scan(path):
    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud: {path}")

    return pcd