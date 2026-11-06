"""One batch runner for every dataset adapter."""
import sys
from pathlib import Path
import subprocess
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import (
    DATASET,
    DATA_DIR,
    DATASET_ROOTS,
    PAIRS,
    OUTPUT_DIR,
    BATCH_PAIR_TIMEOUT_SECONDS,
    FEATURE_METHOD,
    REGISTRATION_METHOD,
)
from datasets import create_dataset
from main import print_metric_summary


def _run_pair_in_worker(dataset, source_id, target_id, output_dir, experiment):
    """Use process isolation for native DGR/Minkowski/Open3D resources."""
    feature_method = str(experiment.get("feature_method", FEATURE_METHOD)).upper()
    registration_method = str(experiment.get("registration_method", REGISTRATION_METHOD)).upper()
    command = [
        sys.executable, str(PROJECT_ROOT / "main.py"),
        "--dataset", dataset.name,
        "--source", source_id,
        "--target", target_id,
        "--output", str(output_dir),
        "--feature", feature_method,
        "--registration", registration_method,
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT.parent, text=True, capture_output=True,
        timeout=BATCH_PAIR_TIMEOUT_SECONDS,
    )
    run_name = f"{dataset.name}_{registration_method}_{feature_method}_{source_id}_{target_id}"
    metrics_file = Path(output_dir) / run_name / "metrics.json"
    if completed.returncode != 0 or not metrics_file.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip() or f"worker exited with {completed.returncode}"
        raise RuntimeError(detail)
    metrics = json.loads(metrics_file.read_text())
    metrics["execution_time_seconds"] = time.perf_counter() - started
    return metrics


def run_batch(pairs=None, **experiment):
    dataset_name = experiment.get("dataset_name", DATASET)
    dataset_root = DATA_DIR if str(dataset_name).upper() == str(DATASET).upper() else DATASET_ROOTS[str(dataset_name).upper()]
    dataset = create_dataset(dataset_name, dataset_root)
    if pairs is None:
        # A config block's PAIRS apply only to its active dataset. Product
        # sweeps must use the selected adapter's own defaults instead.
        pairs = PAIRS if str(dataset_name).upper() == str(DATASET).upper() else tuple(dataset.default_pairs())
    
    output_dir = experiment.get("output_dir", OUTPUT_DIR)
    results = []
    for source_id, target_id in pairs:
        try:
            metrics = _run_pair_in_worker(dataset, source_id, target_id, output_dir, experiment)
            results.append(metrics)
            print_metric_summary(metrics, label=f"PAIR: {source_id} -> {target_id}")
        except Exception as error:
            print(f"FAILED {source_id} -> {target_id}: {error}", file=sys.stderr)
    if results:
        numeric = (
            "fitness", "rmse", "rotation_error_deg", "translation_error",
            "before_registration_fitness", "execution_time_seconds",
        )
        averages = {
            key: (sum(values) / len(values) if (values := [item[key] for item in results if item.get(key) is not None]) else None)
            for key in numeric
        }
        print("\n" + "=" * 40)
        print_metric_summary(averages, label=f"AVERAGE (across {len(results)} pairs)")
        print("=" * 40)
        return results, averages
    return [], {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run batch pairwise registration.")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--feature", default=FEATURE_METHOD)
    parser.add_argument("--registration", default=REGISTRATION_METHOD)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_batch(
        dataset_name=args.dataset,
        feature_method=args.feature,
        registration_method=args.registration,
        output_dir=args.output,
    )

