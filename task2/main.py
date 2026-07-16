import sys
from pathlib import Path
import copy
import numpy as np

# Ensure task2 config overrides task1 config
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT.parent))

from config import *
import open3d as o3d

from task1.utils.io import load_scan, save_metrics
from task1.utils.metrics import extract_metrics
from task1.utils.visualization import visualize_registration

from task1.utils.groundtruth import (
    load_groundtruth,
    relative_transform,
    rotation_error,
    translation_error,
)

from preprocessing.feature_pipeline import (
    prepare_registration_inputs
)

from task1.registration.icp import run_icp
from task1.registration.fgr import run_fgr

from task1.registration.robust_icp import (
    run_robust_icp
)

from task1.registration.go_icp import run_go_icp

def main():
    feature_method = str(FEATURE_METHOD).strip().upper()
    registration_method = str(REGISTRATION_METHOD).strip().upper()

    # -----------------------------
    # Load Point Clouds
    # -----------------------------
    source_raw = load_scan(SOURCE_NAME)
    target_raw = load_scan(TARGET_NAME)

    # -----------------------------
    # Downsample point clouds for speed and memory efficiency
    # -----------------------------
    source = source_raw.voxel_down_sample(VOXEL_SIZE)
    target = target_raw.voxel_down_sample(VOXEL_SIZE)

    # -----------------------------
    # Ground Truth
    # -----------------------------
    poses = load_groundtruth(
        str(DATA_DIR / CONF_NAME)
    )

    T_gt = relative_transform(
        poses,
        SOURCE_NAME,
        TARGET_NAME
    )

    # -----------------------------
    # Feature Pipeline
    # -----------------------------
    pipeline_data = (
        prepare_registration_inputs(
            feature_method,
            registration_method,
            source,
            target
        )
    )

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = np.identity(4)
    result.fitness = 0.0
    result.inlier_rmse = float('inf')

    can_register = (
        pipeline_data and
        pipeline_data["registration_source"].has_points() and
        pipeline_data["registration_target"].has_points()
    )

    if not can_register:
        print(
            f"ERROR: Feature method '{feature_method}' resulted in empty point clouds. Skipping registration.",
            file=sys.stderr,
        )
    else:
        registration_source = pipeline_data["registration_source"]
        registration_target = pipeline_data["registration_target"]
        initial_transform = pipeline_data.get("initial_transform", np.identity(4))

        # -----------------------------
        # Registration Selection
        # -----------------------------
        if registration_method == "ICP":
            result = run_icp(
                registration_source,
                registration_target,
                initial_transform,
                threshold=ICP_THRESHOLD,
            )

        elif registration_method == "ROBUST_ICP":
            result = run_robust_icp(
                registration_source,
                registration_target,
                initial_transform,
                threshold=ROBUST_ICP_THRESHOLD,
                loss_k=ROBUST_ICP_LOSS_K,
            )
            
        elif registration_method == "GO_ICP":
            result = run_go_icp(
                registration_source,
                registration_target,
                initial_transform,
                threshold=ICP_THRESHOLD,
                bin_path=GO_ICP_BIN_PATH,
                mse_thresh=GO_ICP_MSE_THRESH,
                trim_fraction=GO_ICP_TRIM_FRACTION,
                dt_size=GO_ICP_DT_SIZE,
                num_points=GO_ICP_NUM_POINTS
            )

        elif registration_method == "FGR":
            if pipeline_data.get("source_fpfh") is None or pipeline_data.get("target_fpfh") is None:
                print(
                    f"WARNING: REGISTRATION_METHOD='FGR' requires FPFH features, but they were not generated for FEATURE_METHOD='{feature_method}'. Returning a failed result.",
                    file=sys.stderr,
                )
            else:
                source_fpfh = pipeline_data["source_fpfh"]
                target_fpfh = pipeline_data["target_fpfh"]

                result = run_fgr(
                    registration_source,
                    registration_target,
                    source_fpfh,
                    target_fpfh,
                    voxel_size=VOXEL_SIZE
                )

        else:
            raise ValueError(
                f"Unsupported registration method: "
                f"{REGISTRATION_METHOD!r}"
            )

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = extract_metrics(result)

    rot_err = rotation_error(
        T_gt,
        result.transformation
    )

    trans_err = translation_error(
        T_gt,
        result.transformation
    )

    print(
        f"{metrics['fitness']:.6f}\t"
        f"{metrics['rmse']:.6f}\t"
        f"{rot_err:.6f}\t"
        f"{trans_err:.6f}"
    )

    # -----------------------------
    # Save Results
    # -----------------------------
    run_name = (
        f"{registration_method}_"
        f"{feature_method}_"
        f"{SOURCE_NAME}_"
        f"{TARGET_NAME}"
    )

    output_dir = OUTPUT_DIR / run_name

    metrics_to_save = {
        "fitness": metrics["fitness"],
        "rmse": metrics["rmse"],
        "rotation_error_deg": rot_err,
        "translation_error": trans_err,
    }

    if SAVE_RESULTS:
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_metrics(
            output_dir,
            metrics_to_save
        )

        # Save downsampled aligned source cloud to save space
        aligned_source = copy.deepcopy(source)
        aligned_source.transform(
            result.transformation
        )

        o3d.io.write_point_cloud(
            str(
                output_dir /
                f"{run_name}.ply"
            ),
            aligned_source
        )

    # -----------------------------
    # Visualization
    # -----------------------------
    if VISUALIZE:
        visualize_registration(
            source,
            target,
            result.transformation
        )


if __name__ == "__main__":
    main()
