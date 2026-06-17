import open3d as o3d


def run_icp(
    source,
    target,
    initial_transform,
    threshold=0.02
):

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return result