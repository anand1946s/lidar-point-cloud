import open3d as o3d
import numpy as np
import json
import os

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "../bunny/bunny/data"

TRANSFORM_DIR = "icp/final_transforms"

OUTPUT_FILE = "reconstruction/reconstructed_bunny.ply"

# =====================================================
# LOAD TRANSFORM
# =====================================================

def load_transform(filename):

    path = os.path.join(
        TRANSFORM_DIR,
        filename
    )

    with open(path, "r") as f:

        data = json.load(f)

    return np.array(
        data["matrix"]
    )

# =====================================================
# MAIN
# =====================================================

def main():

    print("Loading transforms...")

    T_000_045 = load_transform(
        "final_000_045.json"
    )

    T_045_090 = load_transform(
        "final_045_090.json"
    )

    T_090_180 = load_transform(
        "final_090_180.json"
    )

    T_180_270 = load_transform(
        "final_180_270.json"
    )

    T_270_315 = load_transform(
        "final_270_315.json"
    )

    # -------------------------------------------------
    # CUMULATIVE TRANSFORMS
    # -------------------------------------------------

    T000 = np.eye(4)

    T045 = T_000_045

    T090 = T045 @ T_045_090

    T180 = T090 @ T_090_180

    T270 = T180 @ T_180_270

    T315 = T270 @ T_270_315

    print("Loading scans...")

    bun000 = o3d.io.read_point_cloud(
        os.path.join(DATA_DIR, "bun000.ply")
    )

    bun045 = o3d.io.read_point_cloud(
        os.path.join(DATA_DIR, "bun045.ply")
    )

    bun090 = o3d.io.read_point_cloud(
        os.path.join(DATA_DIR, "bun090.ply")
    )

    bun180 = o3d.io.read_point_cloud(
        os.path.join(DATA_DIR, "bun180.ply")
    )

    bun270 = o3d.io.read_point_cloud(
        os.path.join(DATA_DIR, "bun270.ply")
    )

    bun315 = o3d.io.read_point_cloud(
        os.path.join(DATA_DIR, "bun315.ply")
    )

    print("Applying transforms...")

    bun045.transform(T045)

    bun090.transform(T090)

    bun180.transform(T180)

    bun270.transform(T270)

    bun315.transform(T315)

    print("Merging clouds...")

    merged = (
        bun000 +
        bun045 +
        bun090 +
        bun180 +
        bun270 +
        bun315
    )

    print(
        f"Merged points before cleanup: "
        f"{len(merged.points)}"
    )

    merged = merged.voxel_down_sample(
        voxel_size=0.001
    )

    print(
        f"Merged points after cleanup: "
        f"{len(merged.points)}"
    )

    o3d.io.write_point_cloud(
        OUTPUT_FILE,
        merged
    )

    print(
        f"Saved reconstruction: "
        f"{OUTPUT_FILE}"
    )

    o3d.visualization.draw_geometries(
        [merged],
        window_name="Reconstructed Bunny"
    )

if __name__ == "__main__":
    main()