import sys
import os
from pathlib import Path
import numpy as np
import open3d as o3d
from easydict import EasyDict

# Add official DGR repository root to sys.path
DGR_REPO_DIR = Path(__file__).resolve().parent.parent.parent / "source_codes" / "DeepGlobalRegistration"
if str(DGR_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(DGR_REPO_DIR))

# Official DGR modules from original repo
from core.deep_global_registration import DeepGlobalRegistration as OfficialDeepGlobalRegistration


def dgr_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    weights_path: str = None,
    voxel_size: float = 0.05,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Official Deep Global Registration (DGR) inference function (Choy et al., CVPR 2020).
    Wraps the official DeepGlobalRegistration pipeline from source_codes/DeepGlobalRegistration.
    
    Args:
        source: Open3D PointCloud (source scan)
        target: Open3D PointCloud (target scan)
        weights_path: Path to official pretrained DGR checkpoint (.pth)
        voxel_size: Preprocessing voxel size for DGR
        use_gpu: Whether to run inference on CUDA GPU (False for CPU)
        
    Returns:
        T: 4x4 numpy array rigid transformation matrix
    """
    if weights_path is None:
        weights_path = str(DGR_REPO_DIR / "ResUNetBN2C-feat32-3dmatch-v0.05.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Official DGR checkpoint not found at {weights_path}. "
            f"Please ensure ResUNetBN2C-feat32-3dmatch-v0.05.pth exists in {DGR_REPO_DIR}."
        )

    # Configuration dictionary for official DeepGlobalRegistration
    config = EasyDict({
        "weights": weights_path,
        "clip_weight_thresh": 0.05,
        "use_gpu": use_gpu,
        "voxel_size": voxel_size,
        "inlier_feature_type": "coords",
        "inlier_direct_loss_weight": 1.0,
        "procrustes_loss_weight": 1.0,
        "trans_weight": 1.0,
        "hit_ratio_thresh": 0.1,
    })

    # Instantiate official DGR model from official repository
    device = "cuda" if use_gpu else "cpu"
    dgr_model = OfficialDeepGlobalRegistration(config, device=device)

    # Call official registration pipeline from original implementation
    T_matrix = dgr_model.register(source, target)

    return T_matrix


def run_deep_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray = None,
    source_fpfh=None,
    target_fpfh=None,
    threshold: float = 0.05,
    weights_path: str = None,
    use_gpu: bool = False
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Wrapper adapting official DGR inference to the benchmarking framework's
    RegistrationResult interface (compatible with ICP, FGR, and Go-ICP).
    """
    T = dgr_registration(
        source,
        target,
        weights_path=weights_path,
        voxel_size=threshold,
        use_gpu=use_gpu
    )

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T

    # Compute evaluation metrics using Open3D evaluate_registration
    eval_res = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, T
    )
    result.fitness = eval_res.fitness
    result.inlier_rmse = eval_res.inlier_rmse

    return result
