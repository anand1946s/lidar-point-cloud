import open3d as o3d
import numpy as np
import json
import os

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "../bunny/bunny/data"

ICP_DIR = "icp/icp_transforms"
OUTPUT_DIR = "icp/final_transforms"

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
    print(f"ROBUST ICP {source_id} -> {target_id}")
    print("================================")

    source = o3d.io.read_point_cloud(
        os.path.join(
            DATA_DIR,
            f"bun{source_id}.ply"
        )
    )

    target = o3d.io.read_point_cloud(
        os.path.join(
            DATA_DIR,
            f"bun{target_id}.ply"
        )
    )

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01,
            max_nn=30
        )
    )

    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01,
            max_nn=30
        )
    )

    transform_path = os.path.join(
        ICP_DIR,
        f"icp_{source_id}_{target_id}.json"
    )

    with open(transform_path, "r") as f:
        icp_data = json.load(f)

    init_transform = np.array(
        icp_data["matrix"]
    )

    loss = (
        o3d.pipelines.registration
        .TukeyLoss(k=0.01)
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
            max_correspondence_distance=0.01,
            init=init_transform,
            estimation_method=estimation
        )
    )

    print("Fitness:", result.fitness)
    print("RMSE:", result.inlier_rmse)

    output_data = {
        "pair": f"{source_id}_{target_id}",
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "matrix": result.transformation.tolist()
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        f"final_{source_id}_{target_id}.json"
    )

    with open(output_path, "w") as f:
        json.dump(
            output_data,
            f,
            indent=4
        )

    print("Saved:", output_path)

# =====================================================
# MAIN
# =====================================================

def main():

    for source_id, target_id in PAIRS:

        process_pair(
            source_id,
            target_id
        )

    print("\nAll robust ICP transforms generated.")

if __name__ == "__main__":
    main()