import open3d as o3d



def _ensure_normals(point_cloud, radius=None, max_nn=30):

    if point_cloud.has_normals():
        return

    if radius is None:
        extent = point_cloud.get_max_bound() - point_cloud.get_min_bound()
        radius = max(float(extent.max()) * 0.1, 0.01)

    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )


def run_robust_icp(
    source,
    target,
    initial_transform,
    threshold=0.02,
    loss_k=0.01,
):

    _ensure_normals(source)
    _ensure_normals(target)

    loss = (
        o3d.pipelines.registration
        .TukeyLoss(k=loss_k)
    )

    estimation = (
        o3d.pipelines.registration
        .TransformationEstimationPointToPlane(
            loss
        )
    )

    result = (
        o3d.pipelines.registration
        .registration_icp(
            source,
            target,
            threshold,
            initial_transform,
            estimation
        )
    )

    return result