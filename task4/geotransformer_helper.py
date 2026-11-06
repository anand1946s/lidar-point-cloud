import sys
import os
import copy
from pathlib import Path
import numpy as np
import open3d as o3d
import torch

# Official GeoTransformer paths
GEOTRANSFORMER_ROOT = (
    Path(__file__).resolve().parent.parent
    / "source_codes"
    / "GeoTransformer"
)
EXP_DIR = (
    GEOTRANSFORMER_ROOT
    / "experiments"
    / "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
)

if str(GEOTRANSFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(GEOTRANSFORMER_ROOT))

_CACHED_MODEL = None
_CACHED_DEVICE = None
_CACHED_WEIGHTS = None


import importlib.util

_CACHED_MAKE_CFG = None
_CACHED_CREATE_MODEL = None


def _get_geotransformer_modules():
    global _CACHED_MAKE_CFG, _CACHED_CREATE_MODEL
    if _CACHED_MAKE_CFG is not None and _CACHED_CREATE_MODEL is not None:
        return _CACHED_MAKE_CFG, _CACHED_CREATE_MODEL

    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))

    spec_cfg = importlib.util.spec_from_file_location("gt_exp_cfg", EXP_DIR / "config.py")
    cfg_mod = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(cfg_mod)

    spec_model = importlib.util.spec_from_file_location("gt_exp_model", EXP_DIR / "model.py")
    model_mod = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(model_mod)

    _CACHED_MAKE_CFG = cfg_mod.make_cfg
    _CACHED_CREATE_MODEL = model_mod.create_model
    return _CACHED_MAKE_CFG, _CACHED_CREATE_MODEL


def _load_geotransformer_model(weights_path=None, use_gpu=False):
    """
    Lazily loads and caches the official GeoTransformer model and checkpoint.
    """
    global _CACHED_MODEL, _CACHED_DEVICE, _CACHED_WEIGHTS

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    if weights_path is None:
        weights_path = GEOTRANSFORMER_ROOT / "weights" / "geotransformer-3dmatch.pth.tar"
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"GeoTransformer checkpoint not found at {weights_path}. "
            f"Ensure geotransformer-3dmatch.pth.tar exists in {GEOTRANSFORMER_ROOT / 'weights'}."
        )

    weights_str = str(weights_path)
    device_str = str(device)

    if _CACHED_MODEL is not None and _CACHED_WEIGHTS == weights_str and _CACHED_DEVICE == device_str:
        return _CACHED_MODEL, device

    make_cfg, create_model = _get_geotransformer_modules()

    cfg = make_cfg()
    model = create_model(cfg).to(device)
    model.eval()

    checkpoint = torch.load(weights_str, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)

    _CACHED_MODEL = model
    _CACHED_DEVICE = device_str
    _CACHED_WEIGHTS = weights_str

    return model, device


def run_geotransformer_task4(
    source,
    target,
    threshold=0.05,
    weights_path=None,
    voxel_size=None,
    max_points=25000,
    use_gpu=False,
    auto_scale=True,
):
    """
    Runs official GeoTransformer (Qin et al., CVPR 2022) end-to-end to align source onto target.

    Args:
        source: Open3D PointCloud to align onto `target`.
        target: Open3D PointCloud reference scan.
        threshold: Inlier distance threshold (m) for registration evaluation.
        weights_path: Path to geotransformer checkpoint (.pth.tar).
        voxel_size: Downsampling voxel size for inference (default: 0.04m, or auto-derived).
        max_points: Point limit cap for efficient CPU/GPU inference.
        use_gpu: Whether to use CUDA device if available.
        auto_scale: Whether to scale non-room-scale objects (e.g. Stanford Bunny) to ~4m extent.

    Returns:
        o3d.pipelines.registration.RegistrationResult
    """
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    make_cfg, _ = _get_geotransformer_modules()
    cfg = make_cfg()

    model, device = _load_geotransformer_model(weights_path=weights_path, use_gpu=use_gpu)

    src_pcd = copy.deepcopy(source)
    tgt_pcd = copy.deepcopy(target)

    # Scale adaptation: 3DMatch model expects meter/room-scale objects (~2-8m).
    # If object extent is tiny (< 1.5m, e.g. Stanford models) or huge (> 50m), rescale for inference.
    scale = 1.0
    if auto_scale:
        tgt_extent = np.max(tgt_pcd.get_max_bound() - tgt_pcd.get_min_bound())
        if tgt_extent > 1e-4 and (tgt_extent < 1.5 or tgt_extent > 40.0):
            scale = 4.0 / float(tgt_extent)
            src_pcd.scale(scale, center=(0.0, 0.0, 0.0))
            tgt_pcd.scale(scale, center=(0.0, 0.0, 0.0))

    # Downsampling for inference
    eff_voxel_size = voxel_size if voxel_size is not None else 0.04
    if auto_scale and scale != 1.0 and voxel_size is None:
        eff_voxel_size = 0.04
    elif auto_scale and scale != 1.0 and voxel_size is not None:
        eff_voxel_size = voxel_size * scale

    if eff_voxel_size > 0:
        src_down = src_pcd.voxel_down_sample(eff_voxel_size)
        tgt_down = tgt_pcd.voxel_down_sample(eff_voxel_size)
        if len(src_down.points) > 100 and len(tgt_down.points) > 100:
            src_pcd, tgt_pcd = src_down, tgt_down

    src_pts = np.asarray(src_pcd.points, dtype=np.float32)
    tgt_pts = np.asarray(tgt_pcd.points, dtype=np.float32)

    # Cap point limit if exceeded
    if max_points is not None:
        if len(src_pts) > max_points:
            idx = np.random.choice(len(src_pts), max_points, replace=False)
            src_pts = src_pts[idx]
        if len(tgt_pts) > max_points:
            idx = np.random.choice(len(tgt_pts), max_points, replace=False)
            tgt_pts = tgt_pts[idx]

    data_dict = {
        "ref_points": tgt_pts,
        "src_points": src_pts,
        "ref_feats": np.ones_like(tgt_pts[:, :1], dtype=np.float32),
        "src_feats": np.ones_like(src_pts[:, :1], dtype=np.float32),
        "transform": np.eye(4, dtype=np.float32),
    }

    neighbor_limits = [38, 36, 36, 38]
    collated = registration_collate_fn_stack_mode(
        [data_dict],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
    )

    if device.type == "cuda":
        from geotransformer.utils.torch import to_cuda
        collated = to_cuda(collated)

    with torch.no_grad():
        output_dict = model(collated)

    T_est = output_dict["estimated_transform"].detach().cpu().numpy()
    if T_est.ndim == 3 and T_est.shape[0] == 1:
        T_est = T_est[0]

    # Invert scaling on translation component
    if scale != 1.0:
        T_est[:3, 3] /= scale

    result = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, T_est
    )
    return result
