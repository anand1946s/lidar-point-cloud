from config import *
import open3d as o3d
import copy
import numpy as np
import sys

from utils.io import load_scan, save_metrics
from utils.metrics import extract_metrics
from utils.visualization import visualize_registration

from utils.groundtruth import (
    load_groundtruth,
    relative_transform,
    rotation_error,
    translation_error,
)

from preprocessing.feature_pipeline import (
    prepare_registration_inputs
)

from registration.icp import run_icp
from registration.fgr import run_fgr

from registration.robust_icp import (
    run_robust_icp
)

from registration.go_icp import run_go_icp

def main():

    feature_method = str(FEATURE_METHOD).strip().upper()
    registration_method = str(REGISTRATION_METHOD).strip().upper()

    # -----------------------------
    # Load Point Clouds
    # -----------------------------

    source = load_scan(SOURCE_NAME)
    target = load_scan(TARGET_NAME)

    # -----------------------------
    # Ground Truth
    # -----------------------------

    poses = load_groundtruth(
        str(DATA_DIR / "bun.conf")
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

    # Default to a failed result. This will be used if a combination is invalid
    # or if the feature extraction fails.
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = np.identity(4)
    result.fitness = 0.0
    result.inlier_rmse = float('inf')

    # Check if the feature pipeline produced valid (non-empty) point clouds
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

        # If no initial transform is provided (e.g., for FEATURE_METHOD="NONE", or if
        # feature matching fails), default to an identity matrix.
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
                dt_size=GO_ICP_DT_SIZE
            )

        elif registration_method == "FGR":
            # FGR requires FPFH features. If they are not present, we use the default failed result.
            if pipeline_data.get("source_fpfh") is None or pipeline_data.get("target_fpfh") is None:
                print(
                    f"WARNING: REGISTRATION_METHOD='FGR' requires FPFH features, but they were not generated for FEATURE_METHOD='{feature_method}'. Returning a failed result.",
                    file=sys.stderr,
                )
                # The 'result' is already set to a failed state, so we just let it pass.
            else:
                source_fpfh = pipeline_data["source_fpfh"]
                target_fpfh = pipeline_data["target_fpfh"]

                result = run_fgr(
                    registration_source,
                    registration_target,
                    source_fpfh,
                    target_fpfh
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
        f"{metrics['fitness']:.6f},"
        f"{metrics['rmse']:.6f},"
        f"{rot_err:.6f},"
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

        # -------------------------
        # Save Registered Cloud
        # -------------------------

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
