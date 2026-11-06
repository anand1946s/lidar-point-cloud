"""Dataset-independent registration experiment entry point."""
import copy
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from config import *
from datasets import create_dataset
from evaluation import evaluate, save_metrics
from preprocessing.feature_pipeline import prepare_registration_inputs
from task1.registration.icp import run_icp
from task1.registration.fgr import run_fgr
from task1.registration.robust_icp import run_robust_icp
from go_icp_helper import run_go_icp_task4
from dgr_helper import run_dgr_task4
from dgr_icp_hybrid import run_dgr_icp_hybrid
from rpointhop_helper import run_rpointhop_task4
from rpointhop_icp_hybrid import run_rpointhop_icp_hybrid
from geotransformer_helper import run_geotransformer_task4
from geotransformer_icp_hybrid import run_geotransformer_icp_hybrid
from task1.utils.visualization import visualize_registration


def _failed_result():
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = np.identity(4)
    result.fitness, result.inlier_rmse = 0.0, float("inf")
    return result


def run_registration(source_name=SOURCE_NAME, target_name=TARGET_NAME, *, dataset_name=DATASET, output_dir=OUTPUT_DIR,
                     feature_method=FEATURE_METHOD, registration_method=REGISTRATION_METHOD):
    """Run one pair without relying on dataset-specific branches.

    Dataset adapters own file resolution. Feature preparation owns selected
    registration inputs. Registration receives only clouds and transforms.
    """
    dataset_root = DATA_DIR if str(dataset_name).upper() == str(DATASET).upper() else DATASET_ROOTS[str(dataset_name).upper()]
    dataset = create_dataset(dataset_name, dataset_root)
    profile = DATASET_PROFILES.get(str(dataset_name).upper(), {})
    voxel_size = profile.get("voxel_size", VOXEL_SIZE)
    icp_threshold = profile.get("icp_threshold", ICP_THRESHOLD)
    robust_icp_threshold = profile.get("robust_icp_threshold", ROBUST_ICP_THRESHOLD)
    robust_icp_loss_k = profile.get("robust_icp_loss_k", ROBUST_ICP_LOSS_K)
    ransac_threshold = profile.get("ransac_threshold", RANSAC_THRESHOLD)
    dgr_threshold = profile.get("dgr_threshold", DGR_ICP_HYBRID_DGR_THRESHOLD)
    dgr_icp_threshold = profile.get("dgr_icp_threshold", DGR_ICP_HYBRID_ICP_THRESHOLD)
    pair = dataset.load_pair(source_name, target_name)
    source_raw, target_raw = pair.source, pair.target
    source = source_raw.voxel_down_sample(voxel_size)
    target = target_raw.voxel_down_sample(voxel_size)
    if not source.has_points() or not target.has_points():
        raise ValueError("Voxel downsampling produced an empty point cloud; reduce VOXEL_SIZE.")

    feature_method = str(feature_method).upper()
    registration_method = str(registration_method).upper()
    inputs = prepare_registration_inputs(feature_method, registration_method, source, target, voxel_size=voxel_size,
                                         ransac_threshold=ransac_threshold)
    coarse_source, coarse_target = inputs["registration_source"], inputs["registration_target"]
    initial_transform = inputs["initial_transform"]
    result = _failed_result()

    if not coarse_source.has_points() or not coarse_target.has_points():
        raise ValueError(f"{feature_method} returned an empty registration input.")
    if registration_method == "ICP":
        result = run_icp(coarse_source, coarse_target, initial_transform, threshold=icp_threshold)
    elif registration_method == "ROBUST_ICP":
        result = run_robust_icp(coarse_source, coarse_target, initial_transform, threshold=robust_icp_threshold, loss_k=robust_icp_loss_k)
    elif registration_method == "FGR":
        if inputs["source_fpfh"] is None:
            raise ValueError(f"FGR requires descriptors; {feature_method} did not provide them.")
        result = run_fgr(coarse_source, coarse_target, inputs["source_fpfh"], inputs["target_fpfh"], voxel_size=ransac_threshold * 2.0)
    elif registration_method == "GO_ICP":
        result = run_go_icp_task4(coarse_source, coarse_target, initial_transform, threshold=icp_threshold, bin_path=GO_ICP_BIN_PATH,
                                  mse_thresh=GO_ICP_MSE_THRESH, trim_fraction=GO_ICP_TRIM_FRACTION, dt_size=GO_ICP_DT_SIZE, num_points=GO_ICP_NUM_POINTS)
    elif registration_method in {"DEEP_GLOBAL", "DGR"}:
        # The feature-derived seed is used by transforming DGR's actual input;
        # dgr_helper remains a reusable geometry-only implementation.
        seeded = o3d.geometry.PointCloud(coarse_source)
        seeded.transform(initial_transform)
        residual = run_dgr_task4(seeded, coarse_target, threshold=dgr_threshold, weights_path=str(DGR_MODEL_PATH), use_gpu=DGR_USE_GPU)
        result = residual
        result.transformation = residual.transformation @ initial_transform
    elif registration_method in {"DGR_ICP_HYBRID", "DGR_ROBUSTICP", "DGR_ROBUST_ICP"}:
        result = run_dgr_icp_hybrid(coarse_source, coarse_target, feature_initial_transform=initial_transform,
                                    refinement_source=source, refinement_target=target,
                                    dgr_threshold=dgr_threshold, dgr_weights_path=str(DGR_MODEL_PATH), dgr_use_gpu=DGR_USE_GPU,
                                    dgr_voxel_size=DGR_ICP_HYBRID_DGR_VOXEL_SIZE, dgr_max_points=DGR_ICP_HYBRID_DGR_MAX_POINTS,
                                    dgr_safeguard_max_iterations=DGR_SAFEGUARD_MAX_ITERATIONS,
                                    icp_threshold=dgr_icp_threshold, icp_max_iterations=DGR_ICP_HYBRID_ICP_MAX_ITERATIONS,
                                    robust_icp_loss_k=robust_icp_loss_k, enable_aggressive_refinement=DGR_ICP_HYBRID_ENABLE_AGGRESSIVE_REFINEMENT)
    elif registration_method in {"RPOINTHOP", "R_POINT_HOP", "R_POINTHOP"}:
        result = run_rpointhop_task4(
            coarse_source,
            coarse_target,
            initial_transform=initial_transform,
            threshold=icp_threshold,
            model_path=str(RPOINTHOP_MODEL_PATH) if "RPOINTHOP_MODEL_PATH" in globals() else None,
            ransac_threshold=RPOINTHOP_RANSAC_THRESHOLD if "RPOINTHOP_RANSAC_THRESHOLD" in globals() else None,
        )
    elif registration_method in {"RPOINTHOP_ICP_HYBRID", "R_POINT_HOP_ICP_HYBRID", "RPOINTHOP_HYBRID"}:
        result = run_rpointhop_icp_hybrid(
            coarse_source,
            coarse_target,
            feature_initial_transform=initial_transform,
            refinement_source=source,
            refinement_target=target,
            rpointhop_threshold=profile.get("rpointhop_threshold", RPOINTHOP_THRESHOLD if "RPOINTHOP_THRESHOLD" in globals() else icp_threshold),
            rpointhop_ransac_threshold=profile.get("rpointhop_ransac_threshold", RPOINTHOP_RANSAC_THRESHOLD if "RPOINTHOP_RANSAC_THRESHOLD" in globals() else 0.15),
            rpointhop_model_path=str(RPOINTHOP_MODEL_PATH) if "RPOINTHOP_MODEL_PATH" in globals() else None,
            icp_threshold=icp_threshold,
            robust_icp_loss_k=robust_icp_loss_k,
        )
    elif registration_method in {"GEOTRANSFORMER", "GEO_TRANSFORMER"}:
        result = run_geotransformer_task4(
            coarse_source,
            coarse_target,
            threshold=profile.get("geotransformer_threshold", GEOTRANSFORMER_THRESHOLD if "GEOTRANSFORMER_THRESHOLD" in globals() else icp_threshold),
            weights_path=str(GEOTRANSFORMER_MODEL_PATH) if "GEOTRANSFORMER_MODEL_PATH" in globals() else None,
            voxel_size=GEOTRANSFORMER_VOXEL_SIZE if "GEOTRANSFORMER_VOXEL_SIZE" in globals() else None,
            max_points=GEOTRANSFORMER_MAX_POINTS if "GEOTRANSFORMER_MAX_POINTS" in globals() else None,
            use_gpu=GEOTRANSFORMER_USE_GPU if "GEOTRANSFORMER_USE_GPU" in globals() else False,
            auto_scale=GEOTRANSFORMER_AUTO_SCALE if "GEOTRANSFORMER_AUTO_SCALE" in globals() else True,
        )
    elif registration_method in {"GEOTRANSFORMER_ICP_HYBRID", "GEO_TRANSFORMER_ICP_HYBRID", "GEOTRANSFORMER_HYBRID"}:
        result = run_geotransformer_icp_hybrid(
            coarse_source,
            coarse_target,
            feature_initial_transform=initial_transform,
            refinement_source=source,
            refinement_target=target,
            geotransformer_threshold=profile.get("geotransformer_threshold", GEOTRANSFORMER_THRESHOLD if "GEOTRANSFORMER_THRESHOLD" in globals() else icp_threshold),
            weights_path=str(GEOTRANSFORMER_MODEL_PATH) if "GEOTRANSFORMER_MODEL_PATH" in globals() else None,
            voxel_size=GEOTRANSFORMER_VOXEL_SIZE if "GEOTRANSFORMER_VOXEL_SIZE" in globals() else None,
            use_gpu=GEOTRANSFORMER_USE_GPU if "GEOTRANSFORMER_USE_GPU" in globals() else False,
            auto_scale=GEOTRANSFORMER_AUTO_SCALE if "GEOTRANSFORMER_AUTO_SCALE" in globals() else True,
            icp_threshold=icp_threshold,
            robust_icp_loss_k=robust_icp_loss_k,
        )
    else:
        raise ValueError(f"Unsupported registration method: {registration_method!r}")

    before = evaluate(source, target, initial_transform, icp_threshold, pair.ground_truth)
    metrics = evaluate(source, target, result.transformation, icp_threshold, pair.ground_truth)
    metrics.update({"dataset": dataset.name, "source_id": source_name, "target_id": target_name,
                    "feature_method": feature_method, "registration_method": registration_method,
                    "before_registration_fitness": before["fitness"]})
    run_name = f"{dataset.name}_{registration_method}_{feature_method}_{source_name}_{target_name}"
    run_dir = Path(output_dir) / run_name
    save_metrics(run_dir, metrics)
    if SAVE_RESULTS:
        aligned = copy.deepcopy(source_raw)
        aligned.transform(result.transformation)
        o3d.io.write_point_cloud(str(run_dir / f"{run_name}_full.ply"), aligned)
    if VISUALIZE:
        visualize_registration(source, target, result.transformation)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run one point-cloud registration pair.")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--source", default=SOURCE_NAME)
    parser.add_argument("--target", default=TARGET_NAME)
    parser.add_argument("--feature", default=FEATURE_METHOD)
    parser.add_argument("--registration", default=REGISTRATION_METHOD)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()
    metrics = run_registration(
        args.source, args.target, dataset_name=args.dataset, output_dir=args.output,
        feature_method=args.feature, registration_method=args.registration,
    )
    metrics["execution_time_seconds"] = time.perf_counter() - started
    print_metric_summary(metrics)


def _format_metric(value):
    return "N/A" if value is None else f"{value:.6f}"


def print_metric_summary(metrics, label="RESULT"):
    """Print the common experiment metric contract for one pair or a batch."""
    print(label)
    print(f"  fitness: {_format_metric(metrics.get('fitness'))}")
    print(f"  rmse: {_format_metric(metrics.get('rmse'))}")
    print(f"  rotation_error_deg: {_format_metric(metrics.get('rotation_error_deg'))}")
    print(f"  translation_error: {_format_metric(metrics.get('translation_error'))}")
    print(f"  before_registration_fitness: {_format_metric(metrics.get('before_registration_fitness'))}")
    print(f"  execution_time_seconds: {_format_metric(metrics.get('execution_time_seconds'))}")


if __name__ == "__main__":
    main()
