import sys
import os
from pathlib import Path
import pickle
import numpy as np
import open3d as o3d
import sklearn.metrics

# Official R-PointHop repository root
RPOINTHOP_REPO_DIR = (
    Path(__file__).resolve().parent.parent
    / "source_codes"
    / "R-PointHop"
)
if str(RPOINTHOP_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(RPOINTHOP_REPO_DIR))

_CACHED_MODEL_PARAMS = None
_CACHED_MODEL_PATH = None


def _load_rpointhop_model(model_path=None):
    """
    Lazily loads and caches the official R-PointHop model parameters.
    """
    global _CACHED_MODEL_PARAMS, _CACHED_MODEL_PATH
    if model_path is None:
        model_path = RPOINTHOP_REPO_DIR / "model" / "R-PointHop.pkl"
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Official R-PointHop model checkpoint not found at {model_path}. "
            f"Ensure R-PointHop.pkl exists in {RPOINTHOP_REPO_DIR / 'model'}."
        )

    if _CACHED_MODEL_PARAMS is None or _CACHED_MODEL_PATH != str(model_path):
        with open(model_path, "rb") as f:
            _CACHED_MODEL_PARAMS = pickle.load(f)
        _CACHED_MODEL_PATH = str(model_path)

    return _CACHED_MODEL_PARAMS


def extract_rpointhop_features(
    source,
    target,
    model_path=None,
    num_points=(1024, 896, 768, 640),
    num_samples=(64, 32, 32, 32),
):
    """
    Extracts R-PointHop multi-hop features and keypoints for source and target point clouds.

    Args:
        source: Open3D PointCloud source scan.
        target: Open3D PointCloud target scan.
        model_path: Path to R-PointHop.pkl model checkpoint.
        num_points: Number of points at each hop.
        num_samples: Number of KNN neighbors at each hop.

    Returns:
        tuple: (source_keypoints_pcd, target_keypoints_pcd, source_features_o3d, target_features_o3d)
    """
    try:
        import rpointhop
    except ImportError as e:
        raise ImportError(
            "R-PointHop module could not be imported. Ensure source_codes/R-PointHop is present."
        ) from e

    params = _load_rpointhop_model(model_path)
    initial_n = num_points[0]

    # Sample initial points
    if len(source.points) > initial_n:
        s_ds = source.farthest_point_down_sample(initial_n)
    else:
        s_ds = source
    if len(target.points) > initial_n:
        t_ds = target.farthest_point_down_sample(initial_n)
    else:
        t_ds = target

    s_pts = np.asarray(s_ds.points)
    t_pts = np.asarray(t_ds.points)

    # Pad if fewer than initial_n points
    if len(s_pts) < initial_n:
        pad_idx = np.random.choice(len(s_pts), initial_n - len(s_pts), replace=True)
        s_pts = np.concatenate([s_pts, s_pts[pad_idx]], axis=0)
    if len(t_pts) < initial_n:
        pad_idx = np.random.choice(len(t_pts), initial_n - len(t_pts), replace=True)
        t_pts = np.concatenate([t_pts, t_pts[pad_idx]], axis=0)

    # Normalize to unit sphere for model invariance
    s_center = np.mean(s_pts, axis=0)
    t_center = np.mean(t_pts, axis=0)
    s_scale = max(float(np.max(np.linalg.norm(s_pts - s_center, axis=1))), 1e-6)
    t_scale = max(float(np.max(np.linalg.norm(t_pts - t_center, axis=1))), 1e-6)

    s_norm = (s_pts - s_center) / s_scale
    t_norm = (t_pts - t_center) / t_scale

    data_c = np.concatenate([np.expand_dims(t_norm, 0), np.expand_dims(s_norm, 0)], axis=0)

    leaf_nodes, sampled_pts = rpointhop.pointhop_pred(
        False, data_c, pca_params=params, n_newpoint=list(num_points), n_sample=list(num_samples)
    )

    leaf_arr = np.array(leaf_nodes)
    features = np.reshape(leaf_arr, (leaf_arr.shape[0], leaf_arr.shape[1], leaf_arr.shape[2]))
    features = np.moveaxis(features, 0, 2)

    target_fea, source_fea = features[0], features[1]

    # Map sampled keypoints back to original coordinate system
    target_pts_unnorm = sampled_pts[0] * t_scale + t_center
    source_pts_unnorm = sampled_pts[1] * s_scale + s_center

    src_kps = o3d.geometry.PointCloud()
    src_kps.points = o3d.utility.Vector3dVector(source_pts_unnorm)
    tgt_kps = o3d.geometry.PointCloud()
    tgt_kps.points = o3d.utility.Vector3dVector(target_pts_unnorm)

    src_feat = o3d.pipelines.registration.Feature()
    src_feat.data = source_fea.T.astype(np.float64)
    tgt_feat = o3d.pipelines.registration.Feature()
    tgt_feat.data = target_fea.T.astype(np.float64)

    return src_kps, tgt_kps, src_feat, tgt_feat


def run_rpointhop_task4(
    source,
    target,
    initial_transform=None,
    threshold=0.05,
    model_path=None,
    ransac_threshold=None,
    num_points=(1024, 896, 768, 640),
    num_samples=(64, 32, 32, 32),
):
    """
    Runs global registration using R-PointHop feature descriptors and RANSAC matching.

    Args:
        source: Open3D PointCloud to align onto `target`.
        target: Open3D PointCloud reference scan.
        initial_transform: Optional 4x4 initial transform seed.
        threshold: Correspondence distance threshold (m) for evaluation.
        model_path: Path to R-PointHop.pkl checkpoint.
        ransac_threshold: Distance threshold for feature-based RANSAC (defaults to max(threshold * 3, 0.15)).
        num_points: Point count schedule across hops.
        num_samples: KNN neighborhood schedule across hops.

    Returns:
        o3d.pipelines.registration.RegistrationResult
    """
    if ransac_threshold is None:
        ransac_threshold = max(threshold * 1.5, 0.075)

    src_kps, tgt_kps, src_feat, tgt_feat = extract_rpointhop_features(
        source, target, model_path=model_path, num_points=num_points, num_samples=num_samples
    )

    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_kps,
        tgt_kps,
        src_feat,
        tgt_feat,
        mutual_filter=False,
        max_correspondence_distance=ransac_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 5000),
    )

    T_matrix = ransac_result.transformation
    result = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, T_matrix
    )
    return result
