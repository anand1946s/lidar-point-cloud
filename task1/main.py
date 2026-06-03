import open3d as o3d

from config import DATA_DIR,OUTPUT_DIR
from utils.loader import load_scan

from helper import (
    register_and_merge,
    clean_cloud
)

# =====================================================
# CONFIG
# =====================================================

VOXEL_SIZE = 0.003

SCAN_FILES = [
    "bun000.ply",
    "bun045.ply",
    "bun090.ply",
    # "bun180.ply",
    # "bun270.ply",
    "bun315.ply",
    "top2.ply",
    "top3.ply",
    "chin.ply",
    "ear_back.ply"    
]

# =====================================================
# INITIAL CLOUD
# =====================================================

merged = load_scan(
    DATA_DIR / SCAN_FILES[0]
)

print(f"Starting with {SCAN_FILES[0]}")

# =====================================================
# INCREMENTAL REGISTRATION
# =====================================================

for filename in SCAN_FILES[1:]:

    print(f"\nAdding {filename}")

    scan = load_scan(
        DATA_DIR / filename
    )

    merged = register_and_merge(
        merged,
        scan,
        VOXEL_SIZE
    )

    print(
        f"Current merged points: {len(merged.points)}"
    )

# =====================================================
# CLEANUP
# =====================================================

merged = clean_cloud(
    merged,
    voxel_size=0.001
)

print(
    f"\nFinal points: {len(merged.points)}"
)

# =====================================================
# VISUALIZE
# =====================================================

o3d.visualization.draw_geometries(
    [merged],
    window_name="Reconstructed Bunny"
)

output_file = OUTPUT_DIR / "final.ply"

o3d.io.write_point_cloud(
    str(output_file),
    merged
)

print(f"\nSaved: {output_file}")