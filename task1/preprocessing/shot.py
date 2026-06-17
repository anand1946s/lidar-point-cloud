import subprocess
import tempfile
import numpy as np
import open3d as o3d
import sys

from config import PROJECT_ROOT

_shot_executable_name = "shot"
if sys.platform == "win32":
    _shot_executable_name += ".exe"
SHOT_EXECUTABLE = (
    PROJECT_ROOT.parent / "pcl_tools" / "shot" / "build" / _shot_executable_name
)

SHOT_KEYPOINT_VOXEL_SIZE = 0.005


def extract_shot_keypoints(cloud, voxel_size=SHOT_KEYPOINT_VOXEL_SIZE):
    return cloud.voxel_down_sample(voxel_size)


def _run_shot(cloud, keypoints):
    if keypoints is None:
        keypoints = extract_shot_keypoints(cloud)

    if len(keypoints.points) == 0:
        raise ValueError("Cannot compute SHOT descriptors for zero keypoints.")

    if not SHOT_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"SHOT executable not found at {SHOT_EXECUTABLE}. "
            "Build pcl_tools/shot first."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Open3D writes PLY coordinates as doubles, which PCL PointXYZ may not
        # map correctly. PCD uses float coordinates and preserves the geometry.
        cloud_path = f"{temp_dir}/cloud.pcd"
        keypoints_path = f"{temp_dir}/keypoints.pcd"
        output_path = f"{temp_dir}/shot.csv"

        if not o3d.io.write_point_cloud(cloud_path, cloud):
            raise RuntimeError("Failed to write the temporary SHOT input cloud.")
        if not o3d.io.write_point_cloud(keypoints_path, keypoints):
            raise RuntimeError("Failed to write the temporary SHOT keypoints.")

        try:
            completed = subprocess.run(
                [
                    str(SHOT_EXECUTABLE),
                    cloud_path,
                    keypoints_path,
                    output_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip()
            raise RuntimeError(f"PCL SHOT computation failed: {detail}") from exc

        try:
            descriptors = np.loadtxt(output_path, delimiter=",")
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Could not read SHOT descriptors. "
                f"SHOT output: {completed.stdout.strip()}"
            ) from exc

        descriptors = np.atleast_2d(descriptors)
        if descriptors.shape != (len(keypoints.points), 352):
            raise RuntimeError(
                "SHOT descriptor count does not match the keypoint count: "
                f"{descriptors.shape[0]} descriptors for "
                f"{len(keypoints.points)} keypoints."
            )

        return descriptors


def _to_open3d_feature(descriptors):
    feature = o3d.pipelines.registration.Feature()
    feature.data = np.asarray(descriptors, dtype=np.float64).T
    return feature


def compute_shot(cloud, keypoints=None):
    """
    Compute one SHOT descriptor for each keypoint.

    For registration, use prepare_shot_registration_data() so invalid SHOT
    rows and their corresponding keypoints are removed together.
    """
    if keypoints is None:
        keypoints = extract_shot_keypoints(cloud)

    return _to_open3d_feature(_run_shot(cloud, keypoints))


def prepare_shot_registration_data(cloud, keypoints=None):
    """Return keypoints and finite SHOT descriptors with matching columns."""
    if keypoints is None:
        keypoints = extract_shot_keypoints(cloud)

    descriptors = _run_shot(cloud, keypoints)
    valid_mask = np.isfinite(descriptors).all(axis=1)
    valid_indices = np.flatnonzero(valid_mask).tolist()

    if not valid_indices:
        raise RuntimeError(
            "PCL produced no finite SHOT descriptors. Increase the SHOT "
            "support radius or use a denser point cloud."
        )

    valid_keypoints = keypoints.select_by_index(valid_indices)
    valid_features = _to_open3d_feature(descriptors[valid_mask])
    return valid_keypoints, valid_features
