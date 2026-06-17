import open3d as o3d


def run_fgr(
    source,
    target,
    source_fpfh,
    target_fpfh,
    voxel_size=0.005
):

    distance_threshold = voxel_size * 0.5

    result = (
        o3d.pipelines.registration
        .registration_fgr_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration
            .FastGlobalRegistrationOption(
                maximum_correspondence_distance=
                distance_threshold
            )
        )
    )

    return result