import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "buddha" / "happy_side"
OUTPUT_DIR = PROJECT_ROOT / "outputs_buddha"

SOURCE_NAME = "happySideRight_72"
TARGET_NAME = "happySideRight_96"
CONF_NAME = "happySideRight.conf"



# Options: "NONE" (Whole Set), "ISS", "3DSIFT", "HARRIS3D", "FPFH", "SHOT"
FEATURE_METHOD = "NONE"
 
# Options: "ICP", "ROBUST_ICP", "FGR", "GO_ICP"
REGISTRATION_METHOD = "GO_ICP"

VISUALIZE = False
SAVE_RESULTS = True

# PCL curvature-based SIFT3D keypoint detector
SIFT3D_NORMAL_RADIUS = 0.01
SIFT3D_MIN_SCALE = 0.001
SIFT3D_OCTAVES = 3
SIFT3D_SCALES_PER_OCTAVE = 4
SIFT3D_MIN_CONTRAST = 0.00000001
SIFT3D_INITIAL_THRESHOLD = 0.02

# Registration parameters
ICP_THRESHOLD = 0.02
ROBUST_ICP_THRESHOLD = 0.02
ROBUST_ICP_LOSS_K = 0.01

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
# Increase threshold to stop the Branch-and-Bound search earlier
GO_ICP_MSE_THRESH = 0.005
# Set to 0.0 to disable trimming. Trimming forces a costly sorting operation at EVERY iteration.
# If your point clouds have decent overlap, 0.0 will be exponentially faster.
GO_ICP_TRIM_FRACTION = 0.1
GO_ICP_DT_SIZE = 150  # Default was 300. Decrease to speed up initialization, increase for better distance accuracy.
GO_ICP_NUM_POINTS = 300  # Number of points to downsample for Go-ICP global search. 0 means all points.
