import sys
from pathlib import Path
import numpy as np
import open3d as o3d

from task1.registration.robust_icp import run_robust_icp
from rpointhop_helper import (
    RPOINTHOP_REPO_DIR,
    extract_rpointhop_features,
    run_rpointhop_task4,
)


def run_rpointhop_icp_hybrid(
    source,
    target,
    feature_initial_transform=None,
    refinement_source=None,
    refinement_target=None,
    rpointhop_threshold=0.05,
    rpointhop_ransac_threshold=0.15,
    rpointhop_model_path=None,
    icp_threshold=0.05,
    icp_max_iterations=50,
    robust_icp_loss_k=0.02,
    enable_aggressive_refinement=False,
    verbose=False,
):
    """
    Hybrid registration combining R-PointHop (coarse) and Robust ICP (fine).

    Step 1: Run R-PointHop feature matching + RANSAC for coarse alignment.
    Step 2: Refine using Robust ICP for high-precision local alignment.

    Args:
        source: Open3D PointCloud to align onto `target`.
        target: Open3D PointCloud reference scan.
        feature_initial_transform: Optional initial transform seed.
        refinement_source: Source cloud used for ICP refinement (defaults to source).
        refinement_target: Target cloud used for ICP refinement (defaults to target).
        rpointhop_threshold: Metric evaluation distance for R-PointHop.
        rpointhop_ransac_threshold: Distance threshold for feature RANSAC.
        rpointhop_model_path: Path to R-PointHop.pkl checkpoint.
        icp_threshold: Distance threshold for ICP refinement.
        icp_max_iterations: Maximum iterations for Robust ICP.
        robust_icp_loss_k: Tukey loss parameter for Robust ICP.
        enable_aggressive_refinement: Whether to perform secondary stricter refinement.
        verbose: Verbosity flag.

    Returns:
        o3d.pipelines.registration.RegistrationResult
    """
    if refinement_source is None:
        refinement_source = source
    if refinement_target is None:
        refinement_target = target

    # Step 1: Coarse global registration via R-PointHop
    rpointhop_result = run_rpointhop_task4(
        source,
        target,
        initial_transform=feature_initial_transform,
        threshold=rpointhop_threshold,
        model_path=rpointhop_model_path,
        ransac_threshold=rpointhop_ransac_threshold,
    )
    T_coarse = rpointhop_result.transformation

    rpointhop_eval_at_icp = o3d.pipelines.registration.evaluate_registration(
        refinement_source, refinement_target, icp_threshold, T_coarse
    )

    # Step 2: Fine registration via Robust ICP
    robust_icp_result = run_robust_icp(
        refinement_source,
        refinement_target,
        T_coarse,
        threshold=icp_threshold,
        loss_k=robust_icp_loss_k,
        max_iterations=icp_max_iterations,
    )

    if enable_aggressive_refinement and abs(robust_icp_result.fitness - rpointhop_result.fitness) < 0.01:
        aggressive_result = run_robust_icp(
            refinement_source,
            refinement_target,
            robust_icp_result.transformation,
            threshold=icp_threshold * 0.8,
            loss_k=robust_icp_loss_k,
            max_iterations=icp_max_iterations * 2,
        )
        if aggressive_result.fitness > robust_icp_result.fitness:
            robust_icp_result = aggressive_result

    robust_eval = o3d.pipelines.registration.evaluate_registration(
        refinement_source, refinement_target, icp_threshold, robust_icp_result.transformation
    )

    robust_corr = len(robust_eval.correspondence_set)
    is_near_zero = robust_corr <= 1 or robust_eval.fitness <= 1e-6
    is_clearly_worse = robust_eval.fitness + 1e-6 < rpointhop_eval_at_icp.fitness

    if is_near_zero or is_clearly_worse:
        accepted_transform = T_coarse
        accepted_eval = rpointhop_eval_at_icp
    else:
        accepted_transform = robust_icp_result.transformation
        accepted_eval = robust_eval

    final_result = o3d.pipelines.registration.RegistrationResult()
    final_result.transformation = accepted_transform
    final_result.fitness = accepted_eval.fitness
    final_result.inlier_rmse = accepted_eval.inlier_rmse

    return final_result
