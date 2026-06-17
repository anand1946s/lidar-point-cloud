import subprocess
import tempfile

import numpy as np
import open3d as o3d

from config import PROJECT_ROOT


HARRIS_EXECUTABLE = (
    PROJECT_ROOT.parent / "pcl_tools" / "harris_3d" / "build" / "harris3d"
)


def extract_harris3d(cloud):
    if not HARRIS_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Harris3D executable not found at {HARRIS_EXECUTABLE}. "
            "Build pcl_tools/harris_3d first."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = f"{temp_dir}/input.pcd"
        output_path = f"{temp_dir}/keypoints.pcd"

        if not o3d.io.write_point_cloud(input_path, cloud):
            raise RuntimeError(
                "Failed to write the temporary Harris3D input cloud."
            )

        try:
            subprocess.run(
                [
                    str(HARRIS_EXECUTABLE),
                    input_path,
                    output_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip()
            raise RuntimeError(
                f"PCL Harris3D extraction failed: {detail}"
            ) from exc

        keypoints = o3d.io.read_point_cloud(output_path)
        if len(keypoints.points) == 0:
            raise RuntimeError("PCL Harris3D returned zero keypoints.")

        return keypoints


def get_harris3d_initial_transform(
    source_keypoints,
    target_keypoints,
    threshold=0.02,
):
    result = o3d.pipelines.registration.registration_icp(
        source_keypoints,
        target_keypoints,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return result.transformation
