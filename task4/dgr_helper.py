import sys
import os
from pathlib import Path
import numpy as np
import open3d as o3d
from easydict import EasyDict
import warnings

# Official Deep Global Registration repository root

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


def _load_dgr_model(weights_path, use_gpu):
    """
    Lazily loads the official DeepGlobalRegistration model (Choy et al., CVPR 2020).

    Imported inside the function so MinkowskiEngine is only required at runtime.
    """
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

    return OfficialDeepGlobalRegistration(config, device=device)


def run_dgr_task4(
    source,
    target,
    threshold=0.05,
    weights_path=None,
    use_gpu=False,
):
    """
    Runs the official Deep Global Registration (DGR) pipeline end-to-end.

    DGR is a global registration method, so no initial transform is required
    (any FPFH/RANSAC seed computed upstream is ignored). The returned Open3D
    RegistrationResult behaves identically to run_icp/run_go_icp_task4 so the
    rest of the Task 4 pipeline can consume it unchanged.

    Args:
        source: Open3D PointCloud to align onto `target`.
        target: Open3D PointCloud reference scan.
        threshold: Correspondence distance (m) used for Open3D metrics.
        weights_path: Path to the official DGR checkpoint (.pth).
        use_gpu: Whether to run inference on CUDA (False = CPU).

    Returns:
        o3d.pipelines.registration.RegistrationResult
    """
    if weights_path is None:
        weights_path = DGR_REPO_DIR / "ResUNetBN2C-feat32-3dmatch-v0.05.pth"

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Official DGR checkpoint not found at {weights_path}. "
            f"Ensure ResUNetBN2C-feat32-3dmatch-v0.05.pth exists in {DGR_REPO_DIR}."
        )

    dgr_model = _load_dgr_model(weights_path, use_gpu)

    T_matrix = dgr_model.register(source, target)

    result = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, T_matrix
    )
    return result
