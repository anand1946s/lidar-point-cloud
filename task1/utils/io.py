import open3d as o3d
from pathlib import Path

from config import DATA_DIR


def load_scan(scan_name):
    path = DATA_DIR / f"{scan_name}.ply"

    return o3d.io.read_point_cloud(str(path))

def save_metrics(output_dir, metrics):

    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / "metrics.txt"

    with open(file_path, "w") as f:

        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")