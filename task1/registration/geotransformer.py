import sys
import os
from pathlib import Path
import numpy as np
import open3d as o3d

# Add task4 to sys.path to reuse geotransformer_helper
TASK4_DIR = Path(__file__).resolve().parent.parent.parent / "task4"
if str(TASK4_DIR) not in sys.path:
    sys.path.insert(0, str(TASK4_DIR))

from geotransformer_helper import run_geotransformer_task4, _load_geotransformer_model


def geotransformer_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    weights_path: str = None,
    voxel_size: float = None,
    use_gpu: bool = False,
    auto_scale: bool = True,
) -> np.ndarray:
    """
    Official GeoTransformer (Qin et al., CVPR 2022) rigid transformation estimation.

    Args:
        source: Open3D PointCloud (source scan)
        target: Open3D PointCloud (target scan)
        weights_path: Path to checkpoint (.pth.tar)
        voxel_size: Downsampling voxel size
        use_gpu: Whether to run inference on CUDA GPU
        auto_scale: Whether to scale non-meter-scale objects

    Returns:
        T: 4x4 numpy array rigid transformation matrix
    """
    result = run_geotransformer_task4(
        source,
        target,
        weights_path=weights_path,
        voxel_size=voxel_size,
        use_gpu=use_gpu,
        auto_scale=auto_scale,
    )
    return result.transformation


def run_geotransformer(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transform: np.ndarray = None,
    threshold: float = 0.05,
    weights_path: str = None,
    voxel_size: float = None,
    use_gpu: bool = False,
    auto_scale: bool = True,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Wrapper adapting official GeoTransformer inference to the benchmarking framework's
    RegistrationResult interface (compatible with ICP, FGR, Go-ICP, DGR).
    """
    return run_geotransformer_task4(
        source,
        target,
        threshold=threshold,
        weights_path=weights_path,
        voxel_size=voxel_size,
        use_gpu=use_gpu,
        auto_scale=auto_scale,
    )
