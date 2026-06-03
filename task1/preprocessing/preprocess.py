import open3d as o3d


def preprocess_point_cloud(pcd, voxel_size):

    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Normals
    radius_normal = voxel_size * 2

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30
        )
    )

    # FPFH
    radius_feature = voxel_size * 5

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100
        )
    )

    return pcd_down, fpfh