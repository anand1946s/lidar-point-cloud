import open3d as o3d

#point to plance icp


def run_icp(
    source,
    target,
    initial_transform,
    voxel_size
):

    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return result