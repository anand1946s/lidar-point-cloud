import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SCENE_NAME = "PAVIN"
DATA_DIR = PROJECT_ROOT.parent / "pavin" / "PAVIN-20260709T123119Z-2-001" / "PAVIN"
OUTPUT_DIR = PROJECT_ROOT / "outputs_pavin"

SOURCE_NAME = "scan300"
TARGET_NAME = "scan400"

# The voxel size used for initial downsampling of the point clouds (in meters)
VOXEL_SIZE = 0.05

# Options: "NONE" (Whole Set), "ISS", "3DSIFT", "HARRIS3D", "FPFH", "SHOT", "MULTI_SCALE"
FEATURE_METHOD = "SHOT"

# Options: "ICP", "ROBUST_ICP", "FGR", "GO_ICP"
REGISTRATION_METHOD = "FGR"
VISUALIZE = False
SAVE_RESULTS = True

# PCL curvature-based SIFT3D keypoint detector (scaled up for room-scale in meters)
SIFT3D_NORMAL_RADIUS = 0.1
SIFT3D_MIN_SCALE = 0.05
SIFT3D_OCTAVES = 3
SIFT3D_SCALES_PER_OCTAVE = 4
SIFT3D_MIN_CONTRAST = 0.0001
SIFT3D_INITIAL_THRESHOLD = 0.1

# Registration parameters (scaled up for meters)
ICP_THRESHOLD = 0.15
ROBUST_ICP_THRESHOLD = 0.15
ROBUST_ICP_LOSS_K = 0.05
RANSAC_THRESHOLD = 0.5

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
