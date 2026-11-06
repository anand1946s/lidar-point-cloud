"""Run Dataset × Feature × Registration without editing config.py."""
from itertools import product
from pathlib import Path
import json
import time

from config import DATASET, OUTPUT_DIR
from batch import run_batch

FEATURE_METHODS = ["NONE", "ISS", "3DSIFT", "HARRIS3D", "FPFH", "SHOT", "MULTI_SCALE"]
REGISTRATION_METHODS = ["DGR_ICP_HYBRID", "GEOTRANSFORMER", "GEOTRANSFORMER_ICP_HYBRID"]
DATASETS = [DATASET]  # Change this list or import a sweep definition.


def run_combinations(datasets=DATASETS, features=FEATURE_METHODS, registrations=REGISTRATION_METHODS):
    results = []
    for dataset, feature, registration in product(datasets, features, registrations):
        started = time.time()
        print(f"\n=== Dataset={dataset}, Feature={feature}, Registration={registration} ===")
        try:
            runs, averages = run_batch(dataset_name=dataset, feature_method=feature, registration_method=registration,
                                       output_dir=Path(OUTPUT_DIR) / "batch")
            results.append({"dataset": dataset, "feature_method": feature, "registration_method": registration,
                            "success": bool(runs), "averages": averages, "time_seconds": time.time() - started})
        except Exception as error:
            results.append({"dataset": dataset, "feature_method": feature, "registration_method": registration,
                            "success": False, "error": str(error), "time_seconds": time.time() - started})
    (Path(OUTPUT_DIR) / "batch_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    run_combinations()
