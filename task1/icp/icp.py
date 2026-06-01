import open3d as o3d
import numpy as np
import json
import os

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "../bunny/bunny/data"
RANSAC_DIR = "transforms"
OUTPUT_DIR = "icp/icp_transforms"

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
    print(f"ICP {source_id} -> {target_id}")
    print("================================")

    source_path = os.path.join(
        DATA_DIR,
        f"bun{source_id}.ply"
    )

    target_path = os.path.join(
        DATA_DIR,
        f"bun{target_id}.ply"
    )

    transform_path = os.path.join(
        RANSAC_DIR,
        f"transform_{source_id}_{target_id}.json"
    )

    source = o3d.io.read_point_cloud(
        source_path
    )

    target = o3d.io.read_point_cloud(
        target_path
    )

    with open(transform_path, "r") as f:
        ransac_data = json.load(f)

    initial_transform = np.array(
        ransac_data["matrix"]
    )

    print(
        f"Source points: {len(source.points)}"
    )

    print(
        f"Target points: {len(target.points)}"
    )

    print("Running ICP...")

    result = (
        o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=0.01,
            init=initial_transform,
            estimation_method=
            o3d.pipelines.registration.
            TransformationEstimationPointToPoint()
        )
    )

    print("\nFitness:", result.fitness)
    print("RMSE:", result.inlier_rmse)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"icp_{source_id}_{target_id}.json"
    )

    output_data = {
        "pair": f"{source_id}_{target_id}",
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "matrix": result.transformation.tolist()
    }

    with open(output_path, "w") as f:
        json.dump(
            output_data,
            f,
            indent=4
        )

    print(f"Saved: {output_path}")


# =====================================================
# MAIN
# =====================================================

def main():

    for source_id, target_id in PAIRS:

        process_pair(
            source_id,
            target_id
        )

    print("\nAll ICP transforms generated.")


if __name__ == "__main__":
    main()