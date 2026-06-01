import open3d as o3d
import numpy as np
import os

# =====================================================
# CONFIG
# =====================================================

KEYPOINT_DIR = "keypoints/keypoint_outputs"
OUTPUT_DIR = "descriptor"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS = [
    ("000", "045"),
    ("045", "090"),
    ("090", "180"),
    ("180", "270"),
    ("270", "315")
]

# =====================================================
# FPFH
# =====================================================

def compute_fpfh(pcd):

    # Estimate normals
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01,
            max_nn=30
        )
    )

    # Compute FPFH descriptors
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.025,
            max_nn=100
        )
    )

    return fpfh


# =====================================================
# CORRESPONDENCE MATCHING
# =====================================================

def build_correspondence_matrix(fpfh_source, fpfh_target):

    source_desc = np.asarray(fpfh_source.data).T
    target_desc = np.asarray(fpfh_target.data).T

    correspondences = []

    for i in range(len(source_desc)):

        descriptor = source_desc[i]

        distances = np.linalg.norm(
            target_desc - descriptor,
            axis=1
        )

        best_match = np.argmin(distances)

        correspondences.append(
            [i, best_match]
        )

    return np.array(
        correspondences,
        dtype=np.int32
    )


# =====================================================
# PROCESS ONE PAIR
# =====================================================

def process_pair(source_id, target_id):

    source_path = os.path.join(
        KEYPOINT_DIR,
        f"bun{source_id}_keypoints.ply"
    )

    target_path = os.path.join(
        KEYPOINT_DIR,
        f"bun{target_id}_keypoints.ply"
    )

    print("\n--------------------------------")
    print(f"Processing {source_id} -> {target_id}")
    print("--------------------------------")

    source_cloud = o3d.io.read_point_cloud(
        source_path
    )

    target_cloud = o3d.io.read_point_cloud(
        target_path
    )

    print(
        f"Source keypoints: {len(source_cloud.points)}"
    )

    print(
        f"Target keypoints: {len(target_cloud.points)}"
    )

    print("Computing FPFH...")

    source_fpfh = compute_fpfh(
        source_cloud
    )

    target_fpfh = compute_fpfh(
        target_cloud
    )

    print(
        f"Source FPFH shape: {source_fpfh.data.shape}"
    )

    print(
        f"Target FPFH shape: {target_fpfh.data.shape}"
    )

    print("Matching descriptors...")

    correspondences = build_correspondence_matrix(
        source_fpfh,
        target_fpfh
    )

    print(
        f"Correspondence matrix shape: "
        f"{correspondences.shape}"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"correspondence_{source_id}_{target_id}.npy"
    )

    np.save(
        output_path,
        correspondences
    )

    print(f"Saved: {output_path}")


# =====================================================
# MAIN
# =====================================================

def main():

    print("Generating correspondence matrices...")

    for source_id, target_id in PAIRS:

        process_pair(
            source_id,
            target_id
        )

    print("\nDone.")


if __name__ == "__main__":
    main()