import copy
import open3d as o3d

from config import DATA_DIR
from utils.loader import load_scan
from preprocessing.preprocess import preprocess_point_cloud
from registration.fgr import run_fgr
from registration.icp import run_icp

# =====================================================
# CONFIG
# =====================================================

VOXEL_SIZE = 0.003

# =====================================================
# LOAD SCANS
# =====================================================

source = load_scan(DATA_DIR / "top2.ply")
target = load_scan(DATA_DIR / "bun090.ply")

print(f"Source points : {len(source.points)}")
print(f"Target points : {len(target.points)}")

# =====================================================
# PREPROCESS
# =====================================================

source_down, source_fpfh = preprocess_point_cloud(
    source,
    VOXEL_SIZE
)

target_down, target_fpfh = preprocess_point_cloud(
    target,
    VOXEL_SIZE
)

print("\nAfter Downsampling")
print(f"Source points : {len(source_down.points)}")
print(f"Target points : {len(target_down.points)}")

print("\nFPFH Shapes")
print(f"Source FPFH : {source_fpfh.data.shape}")
print(f"Target FPFH : {target_fpfh.data.shape}")

# =====================================================
# FGR
# =====================================================

result = run_fgr(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    VOXEL_SIZE
)

print("\n=== FGR RESULT ===")
print("Fitness:", result.fitness)
print("RMSE:", result.inlier_rmse)
print("Transformation:")
print(result.transformation)

# =====================================================
# VISUALIZE FGR
# =====================================================

source_fgr = copy.deepcopy(source_down)

source_fgr.transform(
    result.transformation
)

source_fgr.paint_uniform_color([1, 0, 0])  # red
target_down.paint_uniform_color([0, 1, 0])  # green

# o3d.visualization.draw_geometries(
#     [source_fgr, target_down],
#     window_name="After FGR"
# )

# =====================================================
# ICP
# =====================================================

icp_result = run_icp(
    source_down,
    target_down,
    result.transformation,
    VOXEL_SIZE
)

print("\n=== ICP RESULT ===")
print("Fitness:", icp_result.fitness)
print("RMSE:", icp_result.inlier_rmse)
print("Transformation:")
print(icp_result.transformation)

print("\n=== IMPROVEMENT ===")
print(
    "RMSE Reduction:",
    result.inlier_rmse - icp_result.inlier_rmse
)

# =====================================================
# VISUALIZE ICP
# =====================================================

source_icp = copy.deepcopy(source_down)

source_icp.transform(
    icp_result.transformation
)

source_icp.paint_uniform_color([1, 0, 0])  # red
target_down.paint_uniform_color([0, 1, 0])  # green

o3d.visualization.draw_geometries(
    [source_icp, target_down],
    window_name="After ICP"
)