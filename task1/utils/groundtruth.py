import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    """
    Convert translation + quaternion to 4x4 transform.
    """

    T = np.eye(4)

    R = Rotation.from_quat(
        [qx, qy, qz, qw]
    ).as_matrix()

    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    return T

def load_groundtruth(conf_path):
    """
    Returns:
        {
            "bun000": T0,
            "bun045": T1,
            ...
        }
    """

    poses = {}

    with open(conf_path, "r") as f:

        for line in f:

            if not line.startswith("bmesh"):
                continue

            parts = line.split()

            filename = parts[1]

            scan_name = filename.replace(".ply", "")

            tx, ty, tz = map(float, parts[2:5])

            qx, qy, qz, qw = map(float, parts[5:9])

            poses[scan_name] = pose_to_matrix(
                tx, ty, tz,
                qx, qy, qz, qw
            )

    return poses


def relative_transform(
    poses,
    source_name,
    target_name
):

    T_source = poses[source_name]

    T_target = poses[target_name]

    return (
        T_target
        @
        np.linalg.inv(T_source)
    )



def translation_error(
    T_gt,
    T_est
):

    return np.linalg.norm(
        T_gt[:3, 3]
        -
        T_est[:3, 3]
    )


def rotation_error(
    T_gt,
    T_est
):

    R_gt = T_gt[:3, :3]

    R_est = T_est[:3, :3]

    R_diff = R_gt.T @ R_est

    trace = np.trace(R_diff)

    value = (trace - 1) / 2

    value = np.clip(
        value,
        -1.0,
        1.0
    )

    theta = np.arccos(value)

    return np.degrees(theta)