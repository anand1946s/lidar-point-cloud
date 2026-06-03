import open3d as o3d


def run_fgr(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size
):

    distance_threshold = voxel_size * 1.5

    result = (
        o3d.pipelines.registration
        .registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold
            )
        )
    )

    return result