import sys
from pathlib import Path

# Find the workspace root
PROJECT_ROOT = Path(__file__).resolve().parent

SCENE_NAME = "3DMatch"
DATA_DIR = PROJECT_ROOT.parent / "3dmatch"
OUTPUT_DIR = PROJECT_ROOT / "output1_new"

# To run on the first 3DMatch dataset instead, use:
#   SCENE_NAME = "3DMatch"
#   DATA_DIR = PROJECT_ROOT.parent / "3dmatch"
#   OUTPUT_DIR = PROJECT_ROOT / "output1_new"

SOURCE_NAME = "cloud_bin_4"
TARGET_NAME = "cloud_bin_5"

# The voxel size used for initial downsampling of the point clouds (in meters)
# Optimized for 3DMatch room-scale scans (5cm provides a great balance of speed and detail)
VOXEL_SIZE = 0.05

# Options: "NONE" (Whole Set), "ISS", "3DSIFT", "HARRIS3D", "FPFH", "SHOT", "MULTI_SCALE" (or "C2F")
FEATURE_METHOD = "FPFH"

# Options: "ICP", "ROBUST_ICP", "FGR", "GO_ICP"
REGISTRATION_METHOD = "ICP"

VISUALIZE = False
SAVE_RESULTS = True

# PCL curvature-based SIFT3D keypoint detector parameters
SIFT3D_NORMAL_RADIUS = 0.1
SIFT3D_MIN_SCALE = 0.05
SIFT3D_OCTAVES = 3
SIFT3D_SCALES_PER_OCTAVE = 4
SIFT3D_MIN_CONTRAST = 0.0001
SIFT3D_INITIAL_THRESHOLD = 0.1

# Registration thresholds (optimized for 3DMatch dataset scale)
ICP_THRESHOLD = 0.05          # 5cm max correspondence distance for standard alignment
ROBUST_ICP_THRESHOLD = 0.05   # 5cm max correspondence distance
ROBUST_ICP_LOSS_K = 0.02      # Tukey loss parameter
RANSAC_THRESHOLD = 0.075      # 7.5cm distance threshold for RANSAC matching

# Go-ICP specific parameters
_go_icp_exe_name = "GoICP"
if sys.platform == "win32":
    _go_icp_exe_name += ".exe"

GO_ICP_BIN_PATH = (
    PROJECT_ROOT.parent
    / "source_codes"
    / "Go-ICP-master"
    / "Go-ICP-master"
    / _go_icp_exe_name
)

GO_ICP_MSE_THRESH = 0.005
GO_ICP_TRIM_FRACTION = 0.1
GO_ICP_DT_SIZE = 150
GO_ICP_NUM_POINTS = 300
