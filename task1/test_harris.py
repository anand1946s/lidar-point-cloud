import open3d as o3d

from config import DATA_DIR
from preprocessing.harris3d import (
    extract_harris3d
)

cloud = o3d.io.read_point_cloud(
    str(DATA_DIR / "dragonSideRight_72.ply")
)

print(cloud)
keypoints = extract_harris3d(cloud)

print(keypoints)