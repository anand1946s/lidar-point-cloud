"""
Hybrid registration method combining Deep Global Registration (coarse)
and Robust ICP (fine registration).

This approach leverages DGR's ability to find accurate global alignments
and then refines them using robust ICP for maximum accuracy.
"""

import sys
import contextlib
import io
import types
from pathlib import Path
import numpy as np
import open3d as o3d
from easydict import EasyDict

from task1.registration.robust_icp import run_robust_icp
import warnings

warnings.filterwarnings(
    "ignore",
    message="The MinkowskiEngine was compiled with CPU_ONLY flag.*"
)

DGR_REPO_DIR = (
    Path(__file__).resolve().parent.parent
    / "source_codes"
    / "DeepGlobalRegistration"
)
if str(DGR_REPO_DIR) not in sys.path:
    sys.path.append(str(DGR_REPO_DIR))

_DGR_MODEL_CACHE = {}


def _bound_dgr_safeguard(model, max_iterations):
    """Replace DGR's unbounded fallback RANSAC with a bounded equivalent.

    The official implementation hard-codes 80,000 iterations (and a four
    million iteration ceiling) when its learned weighted fit fails. On CPU,
    this is the source of multi-hour batch runs. The normal learned DGR path
    is unchanged; only its fallback is bounded by explicit experiment config.
    """
    def safeguard(_self, pcd0, pcd1, idx0, idx1, _feats0, _feats1, distance, num_iterations):
        correspondences = np.column_stack((idx0.cpu().numpy(), idx1.cpu().numpy()))
        return o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd0, pcd1, o3d.utility.Vector2iVector(correspondences), distance,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, 1000),
        ).transformation
    model.safeguard_registration = types.MethodType(safeguard, model)


def _load_dgr_model(weights_path, use_gpu):
    """Load the official DeepGlobalRegistration model."""
    orig_path = list(sys.path)
    if str(DGR_REPO_DIR) in sys.path:
        sys.path.remove(str(DGR_REPO_DIR))
    sys.path.insert(0, str(DGR_REPO_DIR))
    if "model" in sys.modules:
        mod_file = getattr(sys.modules["model"], "__file__", "") or ""
        if str(DGR_REPO_DIR) not in mod_file:
            del sys.modules["model"]
    try:
        from core.deep_global_registration import (
            DeepGlobalRegistration as OfficialDeepGlobalRegistration,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Deep Global Registration requires MinkowskiEngine, which is not "
            "installed. Build it from source_codes/MinkowskiEngine first:\n"
            "  pip install source_codes/MinkowskiEngine"
        ) from e
    finally:
        sys.path[:] = orig_path

    device = "cuda" if use_gpu else "cpu"

    config = EasyDict({
        "weights": str(weights_path),
        "clip_weight_thresh": 0.05,
        "use_gpu": use_gpu,
        "inlier_feature_type": "coords",
        "inlier_direct_loss_weight": 1.0,
        "procrustes_loss_weight": 1.0,
        "trans_weight": 1.0,
        "hit_ratio_thresh": 0.1,
    })

    cache_key = (str(weights_path), bool(use_gpu))
    if cache_key not in _DGR_MODEL_CACHE:
        _DGR_MODEL_CACHE[cache_key] = OfficialDeepGlobalRegistration(
            config,
            device=device,
        )
    return _DGR_MODEL_CACHE[cache_key]


def run_dgr_icp_hybrid(
    source,
    target,
    feature_initial_transform=None,
    refinement_source=None,
    refinement_target=None,
    dgr_threshold=0.05,
    dgr_weights_path=None,
    dgr_use_gpu=False,
    icp_threshold=0.05,
    icp_max_iterations=50,
    robust_icp_loss_k=0.02,
    dgr_voxel_size=None,
    dgr_max_points=12000,
    dgr_safeguard_max_iterations=5000,
    enable_aggressive_refinement=False,
    verbose=False,
):
    """
    Hybrid registration combining DGR (coarse) and Robust ICP (fine).

    Step 1: Run Deep Global Registration to find a coarse global alignment
    Step 2: Refine using Robust ICP for high-precision local alignment

    Args:
        source: Open3D PointCloud to align onto `target`.
        target: Open3D PointCloud reference scan.
        dgr_threshold: Correspondence distance (m) used for DGR metrics.
        dgr_weights_path: Path to the official DGR checkpoint (.pth).
        dgr_use_gpu: Whether to run DGR inference on CUDA (False = CPU).
        icp_threshold: Correspondence distance (m) for ICP refinement.
        icp_max_iterations: Maximum iterations for Robust ICP.
        robust_icp_loss_k: Tukey loss parameter for Robust ICP.
        dgr_voxel_size: Optional voxel size for DGR-only downsampling.
        dgr_max_points: Max point count used in DGR stage.
        enable_aggressive_refinement: Whether to run a second expensive ICP pass.

    Returns:
        o3d.pipelines.registration.RegistrationResult with combined transformation
    """
    if feature_initial_transform is None:
        feature_initial_transform = np.identity(4)
    if refinement_source is None:
        refinement_source = source
    if refinement_target is None:
        refinement_target = target

    if dgr_weights_path is None:
        dgr_weights_path = DGR_REPO_DIR / "ResUNetBN2C-feat32-3dmatch-v0.05.pth"

    dgr_weights_path = Path(dgr_weights_path)
    if not dgr_weights_path.exists():
        raise FileNotFoundError(
            f"Official DGR checkpoint not found at {dgr_weights_path}. "
            f"Ensure ResUNetBN2C-feat32-3dmatch-v0.05.pth exists in {DGR_REPO_DIR}."
        )

    if dgr_voxel_size is None:
        dgr_voxel_size = max(dgr_threshold * 3.0, 0.1)

    # DGR has no descriptor/initial-pose argument.  Applying a selected
    # feature method's seed here makes that feature part of DGR's actual input,
    # rather than computing it and discarding it.  DGR's residual is composed
    # back into the original source frame below.
    dgr_source = o3d.geometry.PointCloud(source)
    dgr_source.transform(feature_initial_transform)
    dgr_target = target
    if dgr_voxel_size > 0:
        ds_source = dgr_source.voxel_down_sample(dgr_voxel_size)
        ds_target = dgr_target.voxel_down_sample(dgr_voxel_size)
        if ds_source.has_points() and ds_target.has_points():
            dgr_source = ds_source
            dgr_target = ds_target

    if dgr_max_points and dgr_max_points > 0:
        if len(dgr_source.points) > dgr_max_points:
            step = int(np.ceil(len(dgr_source.points) / dgr_max_points))
            dgr_source = dgr_source.uniform_down_sample(step)
        if len(dgr_target.points) > dgr_max_points:
            step = int(np.ceil(len(dgr_target.points) / dgr_max_points))
            dgr_target = dgr_target.uniform_down_sample(step)

    # Step 1: DGR for coarse registration
    # The official reference implementation prints model/safeguard diagnostics
    # for every pair. Keep batch output quiet unless explicitly requested.
    output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with output:
        dgr_model = _load_dgr_model(dgr_weights_path, dgr_use_gpu)
        _bound_dgr_safeguard(dgr_model, dgr_safeguard_max_iterations)
    xyz_source = np.asarray(dgr_source.points)
    xyz_target = np.asarray(dgr_target.points)

    with output:
        T_dgr_residual = dgr_model.register(xyz_source, xyz_target)
    T_dgr = T_dgr_residual @ feature_initial_transform
    dgr_result = o3d.pipelines.registration.evaluate_registration(
        refinement_source, refinement_target, dgr_threshold, T_dgr
    )
    dgr_eval_at_icp = o3d.pipelines.registration.evaluate_registration(
        refinement_source, refinement_target, icp_threshold, T_dgr
    )

    # Step 2: Robust ICP for fine registration using DGR's transformation as
    # the initial guess. This deliberately reuses the canonical Robust ICP
    # implementation from task1, including its Tukey robust loss.
    robust_icp_result = run_robust_icp(
        refinement_source,
        refinement_target,
        T_dgr,
        threshold=icp_threshold,
        loss_k=robust_icp_loss_k,
        max_iterations=icp_max_iterations,
    )

    # Additional refinement pass with stricter threshold if improvement was minimal
    if enable_aggressive_refinement and abs(robust_icp_result.fitness - dgr_result.fitness) < 0.01:
        # Try with smaller threshold
        aggressive_result = run_robust_icp(
            refinement_source,
            refinement_target,
            robust_icp_result.transformation,
            threshold=icp_threshold * 0.8,
            loss_k=robust_icp_loss_k,
            max_iterations=icp_max_iterations * 2,
        )

        # Use better result
        if aggressive_result.fitness > robust_icp_result.fitness:
            robust_icp_result = aggressive_result

    robust_eval = o3d.pipelines.registration.evaluate_registration(
        refinement_source, refinement_target, icp_threshold, robust_icp_result.transformation
    )

    dgr_corr = len(dgr_eval_at_icp.correspondence_set)
    robust_corr = len(robust_eval.correspondence_set)
    is_near_zero = robust_corr <= 1 or robust_eval.fitness <= 1e-6
    is_clearly_worse = robust_eval.fitness + 1e-6 < dgr_eval_at_icp.fitness

    if is_near_zero or is_clearly_worse:
        accepted_transform = T_dgr
        accepted_eval = dgr_eval_at_icp
    else:
        accepted_transform = robust_icp_result.transformation
        accepted_eval = robust_eval

    final_result = o3d.pipelines.registration.RegistrationResult()
    final_result.transformation = accepted_transform
    final_result.fitness = accepted_eval.fitness
    final_result.inlier_rmse = accepted_eval.inlier_rmse

    return final_result
