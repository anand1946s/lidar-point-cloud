import sys
from pathlib import Path

# Find the workspace root
PROJECT_ROOT = Path(__file__).resolve().parent

# Canonical roots used by the batch matrix.  The active commented block below
# remains the simplest way to select a single experiment interactively.
DATASET_ROOTS = {
    "BUNNY": PROJECT_ROOT.parent / "task1" / "data",
    "DRAGON": PROJECT_ROOT.parent / "task1" / "dragon_stand",
    "BUDDHA": PROJECT_ROOT.parent / "task1" / "buddha" / "happy_side",
    "3DMATCH": PROJECT_ROOT.parent / "3dmatch",
    "3DMATCH2": PROJECT_ROOT.parent / "3dmatch2",
    "SCANNET": PROJECT_ROOT.parent / "Scannet",
    "SCANNET2": PROJECT_ROOT.parent / "Scannet",
    "PAVIN": PROJECT_ROOT.parent / "pavin" / "PAVIN-20260709T123119Z-2-001" / "PAVIN",
}

# Dataset scale profiles are configuration, not branches in registration
# algorithms. PAVIN retains the thresholds already used by its working Task 3
# pipeline; the generic profile remains the 3DMatch/ScanNet default.
DATASET_PROFILES = {
    "BUNNY": {
        "voxel_size": 0.003,
        "icp_threshold": 0.02,
        "robust_icp_threshold": 0.02,
        "robust_icp_loss_k": 0.01,
        "ransac_threshold": 0.03,
    },
    "DRAGON": {
        "voxel_size": 0.005,
        "icp_threshold": 0.02,
        "robust_icp_threshold": 0.02,
        "robust_icp_loss_k": 0.01,
        "ransac_threshold": 0.03,
    },
    "BUDDHA": {
        "voxel_size": 0.005,
        "icp_threshold": 0.02,
        "robust_icp_threshold": 0.02,
        "robust_icp_loss_k": 0.01,
        "ransac_threshold": 0.03,
    },
    "PAVIN": {
        "voxel_size": 0.05,
        "icp_threshold": 0.15,
        "robust_icp_threshold": 0.15,
        "robust_icp_loss_k": 0.05,
        "ransac_threshold": 0.5,
        "dgr_threshold": 0.15,
        "dgr_icp_threshold": 0.15,
    },
}

# ============================================================
# Dataset selection (activate exactly one block)
# ============================================================

# --- 1) Stanford Bunny ---
# SCENE_NAME = "Bunny"
# DATASET = "Bunny"
# DATA_DIR = PROJECT_ROOT.parent / "task1" / "data"
# OUTPUT_DIR = PROJECT_ROOT / "output_bunny"
# SOURCE_NAME = "bun000"
# TARGET_NAME = "bun045"
# PAIRS = [("bun000", "bun045"), ("bun045", "bun090"), ("bun000", "bun315")]

# --- 2) Stanford Dragon ---
# SCENE_NAME = "Dragon"
# DATASET = "Dragon"
# DATA_DIR = PROJECT_ROOT.parent / "task1" / "dragon_stand"
# OUTPUT_DIR = PROJECT_ROOT / "output_dgr_dragon"
# SOURCE_NAME = "dragonStandRight_72"
# TARGET_NAME = "dragonStandRight_96"
# PAIRS = [("dragonStandRight_0", "dragonStandRight_24"), ("dragonStandRight_24", "dragonStandRight_48"), ("dragonStandRight_48", "dragonStandRight_72"), ("dragonStandRight_72", "dragonStandRight_96")]

# --- 3) Stanford Buddha ---
# SCENE_NAME = "Buddha"
# DATASET = "Buddha"
# DATA_DIR = PROJECT_ROOT.parent / "task1" / "buddha" / "happy_side"
# OUTPUT_DIR = PROJECT_ROOT / "output_buddha"
# SOURCE_NAME = "happySideRight_0"
# TARGET_NAME = "happySideRight_24"
# PAIRS = [("happySideRight_0", "happySideRight_24"), ("happySideRight_24", "happySideRight_48"), ("happySideRight_48", "happySideRight_72"), ("happySideRight_72", "happySideRight_96")]

# --- 4) 3DMatch ---
# SCENE_NAME = "3DMatch"
# DATASET = "3DMatch"
# DATA_DIR = PROJECT_ROOT.parent / "3dmatch"
# OUTPUT_DIR = PROJECT_ROOT / "outputs_3dmatch2"
# SOURCE_NAME = "cloud_bin_0"
# TARGET_NAME = "cloud_bin_1"
# PAIRS = [("cloud_bin_0", "cloud_bin_1"), ("cloud_bin_1", "cloud_bin_2"), ("cloud_bin_2", "cloud_bin_3"), ("cloud_bin_3", "cloud_bin_4")]

# --- 4b) 3DMatch2 ---
SCENE_NAME = "3DMatch2"
DATASET = "3DMatch"  # or "3DMatch2"
DATA_DIR = PROJECT_ROOT.parent / "3dmatch2"
OUTPUT_DIR = PROJECT_ROOT / "outputs_3dmatch2"
SOURCE_NAME = "cloud_bin_0"
TARGET_NAME = "cloud_bin_1"
PAIRS = [("cloud_bin_0", "cloud_bin_1"), ("cloud_bin_1", "cloud_bin_2"), ("cloud_bin_2", "cloud_bin_3"), ("cloud_bin_3", "cloud_bin_4")]

# --- 5) ScanNet ---
# SCENE_NAME = "scene0027"
# DATASET = "ScanNet"
# DATA_DIR = PROJECT_ROOT.parent / "Scannet"
# OUTPUT_DIR = PROJECT_ROOT / "output_scannet" / SCENE_NAME
# # To save into output_scannet2 instead:
# # OUTPUT_DIR = PROJECT_ROOT / "output_scannet2" / SCENE_NAME
# # OUTPUT_DIR = PROJECT_ROOT / "output_scannet2"
# SOURCE_NAME = "scene0027_02_vh_clean"
# TARGET_NAME = "scene0027_00_vh_clean"
# PAIRS = [("scene0027_02_vh_clean", "scene0027_00_vh_clean"), ("scene0027_01_vh_clean", "scene0027_00_vh_clean"), ("scene0024_02_vh_clean", "scene0024_00_vh_clean"), ("scene0046_02_vh_clean", "scene0046_00_vh_clean")]

# --- 5b) ScanNet2 / DGR_ROBUSTICP (output_scannet2) ---
# SCENE_NAME = "scene0027"
# DATASET = "ScanNet"  # or "ScanNet2"
# DATA_DIR = PROJECT_ROOT.parent / "Scannet"
# OUTPUT_DIR = PROJECT_ROOT / "output_scannet2" / SCENE_NAME
# # Or to save directly in output_scannet2 without scene subfolder:
# # OUTPUT_DIR = PROJECT_ROOT / "output_scannet2"
# SOURCE_NAME = "scene0027_02_vh_clean"
# TARGET_NAME = "scene0027_00_vh_clean"
# PAIRS = [
#     ("scene0027_02_vh_clean", "scene0027_00_vh_clean"),
#     ("scene0027_01_vh_clean", "scene0027_00_vh_clean"),
#     ("scene0024_02_vh_clean", "scene0024_00_vh_clean"),
#     ("scene0046_02_vh_clean", "scene0046_00_vh_clean"),
# ]

# To run DGR_ROBUSTICP with this block, configure:
#   FEATURE_METHOD = "NONE"                     # Whole-set input for DGR
#   REGISTRATION_METHOD = "DGR_ICP_HYBRID"       # DGR_ROBUSTICP (or "DGR_ROBUSTICP")

# --- 6) PAVIN ---
# SCENE_NAME = "PAVIN"
# DATASET = "PAVIN"
# DATA_DIR = PROJECT_ROOT.parent / "pavin" / "PAVIN-20260709T123119Z-2-001" / "PAVIN"
# OUTPUT_DIR = PROJECT_ROOT / "output_pavin"
# SOURCE_NAME = "scan300"
# TARGET_NAME = "scan400"
# PAIRS = [("scan300", "scan400"), ("scan400", "scan500"), ("scan500", "scan600"), ("scan600", "scan700")]

# The voxel size used for initial downsampling of the point clouds (in meters)
# Optimized for 3DMatch room-scale scans (5cm provides a great balance of speed and detail)
VOXEL_SIZE = 0.05

# Options: "NONE" (Whole Set), "ISS", "3DSIFT", "HARRIS3D", "FPFH", "SHOT", "MULTI_SCALE" (or "C2F"), "RPOINTHOP"
# For DGR_ROBUSTICP: use "NONE" for standard end-to-end alignment (or feature method for seeded initial pose)
# FEATURE_METHOD = "NONE"
FEATURE_METHOD = "NONE"

# Options: "ICP", "ROBUST_ICP", "FGR", "GO_ICP", "DEEP_GLOBAL", "DGR_ICP_HYBRID", "DGR_ROBUSTICP", "RPOINTHOP", "RPOINTHOP_ICP_HYBRID", "GEOTRANSFORMER", "GEOTRANSFORMER_ICP_HYBRID"
# To run DGR_ROBUSTICP (Deep Global Registration + Robust ICP hybrid):
# REGISTRATION_METHOD = "DGR_ICP_HYBRID"  # or "DGR_ROBUSTICP"
REGISTRATION_METHOD = "DGR_ROBUSTICP"

# GeoTransformer parameters
GEOTRANSFORMER_MODEL_PATH = (
    PROJECT_ROOT.parent
    / "source_codes"
    / "GeoTransformer"
    / "weights"
    / "geotransformer-3dmatch.pth.tar"
)
GEOTRANSFORMER_VOXEL_SIZE = 0.04
GEOTRANSFORMER_MAX_POINTS = 25000
GEOTRANSFORMER_USE_GPU = False
GEOTRANSFORMER_AUTO_SCALE = True
GEOTRANSFORMER_THRESHOLD = 0.05

# R-PointHop parameters
RPOINTHOP_MODEL_PATH = (
    PROJECT_ROOT.parent
    / "source_codes"
    / "R-PointHop"
    / "model"
    / "R-PointHop.pkl"
)
RPOINTHOP_THRESHOLD = 0.05
RPOINTHOP_RANSAC_THRESHOLD = 0.075
RPOINTHOP_NUM_POINTS = (1024, 896, 768, 640)
RPOINTHOP_NUM_SAMPLES = (64, 32, 32, 32)

# Deep Global Registration parameters
DGR_INLIER_THRESHOLD = 0.05
DGR_MODEL_PATH = (
    PROJECT_ROOT.parent
    / "source_codes"
    / "DeepGlobalRegistration"
    / "ResUNetBN2C-feat32-3dmatch-v0.05.pth"
)
DGR_USE_GPU = False

# DGR+ICP Hybrid Registration parameters
DGR_ICP_HYBRID_DGR_THRESHOLD = 0.05
DGR_ICP_HYBRID_DGR_VOXEL_SIZE = 0.25
# Uniform cap for all datasets; this bounds CPU DGR memory/runtime without
# introducing hidden dataset-specific tuning.
DGR_ICP_HYBRID_DGR_MAX_POINTS = 1000
# DGR's reference fallback otherwise launches an 80,000-iteration RANSAC.
# This applies only when learned DGR cannot produce a weighted solution.
DGR_SAFEGUARD_MAX_ITERATIONS = 500
DGR_ICP_HYBRID_ICP_THRESHOLD = 0.05
DGR_ICP_HYBRID_ICP_MAX_ITERATIONS = 8
DGR_ICP_HYBRID_LOSS_K = 0.02
DGR_ICP_HYBRID_ENABLE_AGGRESSIVE_REFINEMENT = False

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
# Bound feature-derived RANSAC used before registration. The old 100,000
# iteration setting made large ScanNet/3DMatch/PAVIN batches appear hung.
FEATURE_RANSAC_MAX_ITERATIONS = 1000
FEATURE_RANSAC_MAX_VALIDATIONS = 1000
# Each batch pair runs in a fresh worker to isolate native DGR/Open3D state.
# This is a hard per-pair ceiling, not a dataset-specific parameter.
BATCH_PAIR_TIMEOUT_SECONDS = 90

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
