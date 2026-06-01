import open3d as o3d
import numpy as np
import os
import json

# =====================================================
# CONFIG
# =====================================================

KEYPOINT_DIR = "keypoints/keypoint_outputs"
CORR_DIR = "descriptor"
OUTPUT_DIR = "transforms"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS = [
    ("000", "045"),
    ("045", "090"),
    ("090", "180"),
    ("180", "270"),
    ("270", "315")
]

# =====================================================
# PROCESS ONE PAIR
# =====================================================

def process_pair(source_id, target_id):

    print("\n================================")
    print(f"RANSAC {source_id} -> {target_id}")
    print("================================")

    source_path = os.path.join(
        KEYPOINT_DIR,
        f"bun{source_id}_keypoints.ply"
    )

    target_path = os.path.join(
        KEYPOINT_DIR,
        f"bun{target_id}_keypoints.ply"
    )

    corr_path = os.path.join(
        CORR_DIR,
        f"correspondence_{source_id}_{target_id}.npy"
    )

    source_cloud = o3d.io.read_point_cloud(
        source_path
    )

    target_cloud = o3d.io.read_point_cloud(
        target_path
    )

    corr = np.load(corr_path)

    print(
        f"Source keypoints: {len(source_cloud.points)}"
    )

    print(
        f"Target keypoints: {len(target_cloud.points)}"
    )

    print(
        f"Correspondences: {corr.shape}"
    )

    corr_o3d = o3d.utility.Vector2iVector(
        corr.astype(np.int32)
    )

    result = (
        o3d.pipelines.registration
        .registration_ransac_based_on_correspondence(
            source_cloud,
            target_cloud,
            corr_o3d,
            max_correspondence_distance=0.01
        )
    )

    transform = result.transformation

    print("\nTransformation Matrix:")
    print(transform)

    print("\nFitness:", result.fitness)
    print("RMSE:", result.inlier_rmse)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"transform_{source_id}_{target_id}.json"
    )

    data = {
        "pair": f"{source_id}_{target_id}",
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "matrix": transform.tolist()
    }

    with open(output_file, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )

    print(f"\nSaved: {output_file}")

# =====================================================
# MAIN
# =====================================================

def main():

    for source_id, target_id in PAIRS:

        process_pair(
            source_id,
            target_id
        )

    print("\nAll transforms generated.")

if __name__ == "__main__":
    main()