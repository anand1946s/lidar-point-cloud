"""
Hybrid registration method combining GeoTransformer (global alignment)
and Robust ICP (fine local refinement).
"""

import copy
from pathlib import Path
import numpy as np
import open3d as o3d

from task1.registration.robust_icp import run_robust_icp
from geotransformer_helper import run_geotransformer_task4


def run_geotransformer_icp_hybrid(
    coarse_source,
    coarse_target,
    feature_initial_transform=None,
    refinement_source=None,
    refinement_target=None,
    geotransformer_threshold=0.05,
    weights_path=None,
    use_gpu=False,
    voxel_size=None,
    auto_scale=True,
    icp_threshold=0.05,
    icp_max_iterations=30,
    robust_icp_loss_k=0.02,
):
    """
    Two-stage hybrid registration:
      Stage 1: Official GeoTransformer predicts global SE(3) transformation.
      Stage 2: Robust ICP refines transformation on higher-resolution point clouds.

    Args:
        coarse_source: Open3D PointCloud source downsampled for global matching.
        coarse_target: Open3D PointCloud target downsampled for global matching.
        feature_initial_transform: Optional prior transformation.
        refinement_source: Fine-resolution source PointCloud (or coarse if None).
        refinement_target: Fine-resolution target PointCloud (or coarse if None).
        geotransformer_threshold: Distance threshold for GeoTransformer metric evaluation.
        weights_path: Path to geotransformer checkpoint.
        use_gpu: Whether to use CUDA.
        voxel_size: Pre-downsampling voxel size.
        auto_scale: Whether to scale non-room scale point clouds.
        icp_threshold: Distance threshold for Robust ICP refinement.
        icp_max_iterations: Maximum ICP iterations.
        robust_icp_loss_k: Huber/Tukey loss parameter for robust ICP.

    Returns:
        o3d.pipelines.registration.RegistrationResult
    """
    ref_src = coarse_source if refinement_source is None else refinement_source
    ref_tgt = coarse_target if refinement_target is None else refinement_target

    # Stage 1: GeoTransformer global registration
    try:
        geo_result = run_geotransformer_task4(
            coarse_source,
            coarse_target,
            threshold=geotransformer_threshold,
            weights_path=weights_path,
            voxel_size=voxel_size,
            use_gpu=use_gpu,
            auto_scale=auto_scale,
        )
        t_global = geo_result.transformation
    except Exception as e:
        # Fallback to feature_initial_transform if GeoTransformer raises an error
        t_global = feature_initial_transform if feature_initial_transform is not None else np.eye(4)

    # Stage 2: Robust ICP refinement
    refined_result = run_robust_icp(
        ref_src,
        ref_tgt,
        t_global,
        threshold=icp_threshold,
        max_iterations=icp_max_iterations,
        loss_k=robust_icp_loss_k,
    )

    return refined_result
