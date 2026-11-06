"""Dataset-neutral experiment metrics and result persistence."""
from pathlib import Path
import json
import numpy as np
import open3d as o3d


def evaluate(source, target, transformation, threshold, ground_truth=None):
    result = o3d.pipelines.registration.evaluate_registration(source, target, threshold, transformation)
    metrics = {"fitness": float(result.fitness), "rmse": float(result.inlier_rmse)}
    if ground_truth is None:
        metrics.update({"rotation_error_deg": None, "translation_error": None, "ground_truth_available": False})
    else:
        delta = np.linalg.inv(ground_truth) @ transformation
        cosine = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        metrics.update({"rotation_error_deg": float(np.degrees(np.arccos(cosine))), "translation_error": float(np.linalg.norm(delta[:3, 3])), "ground_truth_available": True})
    return metrics


def save_metrics(output_dir, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output_dir / "metrics.txt").open("w") as stream:
        for key, value in metrics.items():
            stream.write(f"{key}: {value}\n")
