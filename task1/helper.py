import copy

from preprocessing.preprocess import preprocess_point_cloud
from registration.fgr import run_fgr
from registration.icp import run_icp


def register_and_merge(
    merged_cloud,
    new_scan,
    voxel_size
):
    """
    Register new_scan to merged_cloud
    and return updated merged cloud.
    """

    # ---------------------------------
    # Preprocess
    # ---------------------------------

    merged_down, merged_fpfh = preprocess_point_cloud(
        merged_cloud,
        voxel_size
    )

    scan_down, scan_fpfh = preprocess_point_cloud(
        new_scan,
        voxel_size
    )

    # ---------------------------------
    # FGR
    # ---------------------------------

    fgr_result = run_fgr(
        scan_down,
        merged_down,
        scan_fpfh,
        merged_fpfh,
        voxel_size
    )

    print("\n=== FGR ===")
    print("Fitness:", fgr_result.fitness)

    # ---------------------------------
    # ICP
    # ---------------------------------

    icp_result = run_icp(
        scan_down,
        merged_down,
        fgr_result.transformation,
        voxel_size
    )

    print("\n=== ICP ===")
    print("Fitness:", icp_result.fitness)

    # ---------------------------------
    # Apply final transform
    # ---------------------------------

    aligned_scan = copy.deepcopy(
        new_scan
    )

    aligned_scan.transform(
        icp_result.transformation
    )

    # ---------------------------------
    # Merge
    # ---------------------------------

    merged_cloud += aligned_scan

    return merged_cloud


def clean_cloud(
    cloud,
    voxel_size=0.001
):
    """
    Final downsampling after all merges.
    """

    return cloud.voxel_down_sample(
        voxel_size
    )