import open3d as o3d
import numpy as np
from pathlib import Path
from utils.io import load_scan
from preprocessing.feature_pipeline import prepare_registration_inputs
from utils.groundtruth import load_groundtruth, relative_transform, rotation_error, translation_error

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dragon_side"

source = load_scan("dragonSideRight_48")
target = load_scan("dragonSideRight_72")

pipeline_data = prepare_registration_inputs(
    "FPFH",
    "GO_ICP",
    source,
    target
)

initial_transform = pipeline_data.get("initial_transform", np.identity(4))

poses = load_groundtruth(str(DATA_DIR / "dragonSideRight.conf"))
T_gt = relative_transform(poses, "dragonSideRight_48", "dragonSideRight_72")

print("RANSAC Initial Transform:")
print(initial_transform)
print("Ground Truth Transform:")
print(T_gt)

rot_err = rotation_error(T_gt, initial_transform)
trans_err = translation_error(T_gt, initial_transform)
print(f"RANSAC Rotation Error: {rot_err:.6f} deg, Translation Error: {trans_err:.6f}")
