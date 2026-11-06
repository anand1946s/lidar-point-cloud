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
    max_iterations=50,
):
    """Run point-to-plane ICP with Tukey robust loss.

    The initial_transform is used as the starting pose. Normals are
    estimated automatically when they are missing. max_iterations is
    optional so existing standalone Robust ICP callers remain compatible.
    """
    _ensure_normals(source, radius=threshold * 2.0)
    _ensure_normals(target, radius=threshold * 2.0)

    loss = o3d.pipelines.registration.TukeyLoss(k=loss_k)

    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            loss
        )
    )

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations
    )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        initial_transform,
        estimation,
        criteria,
    )

    return result
