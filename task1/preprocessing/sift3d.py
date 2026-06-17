import subprocess
import tempfile

import numpy as np
import open3d as o3d

from config import (
    PROJECT_ROOT,
    SIFT3D_INITIAL_THRESHOLD,
    SIFT3D_MIN_CONTRAST,
    SIFT3D_MIN_SCALE,
    SIFT3D_NORMAL_RADIUS,
    SIFT3D_OCTAVES,
    SIFT3D_SCALES_PER_OCTAVE,
)


SIFT3D_EXECUTABLE = (
    PROJECT_ROOT.parent / "source_codes" / "sift3d" / "build" / "sift3d"
)


def extract_sift3d(cloud):
    if not SIFT3D_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"SIFT3D executable not found at {SIFT3D_EXECUTABLE}. "
            "Build source_codes/sift3d first."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = f"{temp_dir}/input.pcd"
        output_path = f"{temp_dir}/keypoints.pcd"

        if not o3d.io.write_point_cloud(input_path, cloud):
            raise RuntimeError("Failed to write the temporary SIFT3D input cloud.")

        try:
            subprocess.run(
                [
                    str(SIFT3D_EXECUTABLE),
                    input_path,
                    output_path,
                    str(SIFT3D_NORMAL_RADIUS),
                    str(SIFT3D_MIN_SCALE),
                    str(SIFT3D_OCTAVES),
                    str(SIFT3D_SCALES_PER_OCTAVE),
                    str(SIFT3D_MIN_CONTRAST),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip()
            raise RuntimeError(f"PCL SIFT3D extraction failed: {detail}") from exc

        keypoints = o3d.io.read_point_cloud(output_path)
        if len(keypoints.points) == 0:
            raise RuntimeError("PCL SIFT3D returned zero keypoints.")

        return keypoints


def get_sift3d_initial_transform(
    source_keypoints,
    target_keypoints,
    threshold=SIFT3D_INITIAL_THRESHOLD,
):
    result = o3d.pipelines.registration.registration_icp(
        source_keypoints,
        target_keypoints,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return result.transformation
