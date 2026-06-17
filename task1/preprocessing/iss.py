import open3d as o3d


def extract_iss_keypoints(pcd):

    keypoints = (
        o3d.geometry.keypoint
        .compute_iss_keypoints(
            pcd
        )
    )

    return keypoints