import open3d as o3d


def compute_fpfh(point_cloud, voxel_size=0.005):

    radius_normal = voxel_size * 2

    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30
        )
    )

    radius_feature = voxel_size * 5

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100
        )
    )

    return fpfh




def get_initial_transform(
    source,
    target,
    source_fpfh,
    target_fpfh,
    voxel_size
):

    threshold = voxel_size * 1.5

    result = (
        o3d.pipelines.registration
        .registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            True,
            threshold,
            o3d.pipelines.registration.
            TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),

                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnDistance(
                    threshold
                )
            ],
            o3d.pipelines.registration.
            RANSACConvergenceCriteria(
                100000,
                0.999
            )
        )
    )

    return result