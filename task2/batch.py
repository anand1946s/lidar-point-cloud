from pathlib import Path
import subprocess
import re
import sys
import time

import importlib.util

CONFIG_PATH = Path("task2/config.py")

# Dynamically load the current config to get the configured SCENE_NAME
spec = importlib.util.spec_from_file_location("config", str(CONFIG_PATH))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
scene_name = config.SCENE_NAME

# The three scans in ScanNet are _00, _01, and _02.
# We run 3 pairwise registrations to align all three.
PAIRS = [
    (f"{scene_name}_01_vh_clean", f"{scene_name}_00_vh_clean"),
    (f"{scene_name}_02_vh_clean", f"{scene_name}_01_vh_clean"),
    (f"{scene_name}_02_vh_clean", f"{scene_name}_00_vh_clean"),
]

fitness_vals = []
rmse_vals = []
rot_vals = []
trans_vals = []

start_time = time.time()

for source_name, target_name in PAIRS:
    text = CONFIG_PATH.read_text()

    text = re.sub(
        r'SOURCE_NAME\s*=\s*".*"',
        f'SOURCE_NAME = "{source_name}"',
        text
    )

    text = re.sub(
        r'TARGET_NAME\s*=\s*".*"',
        f'TARGET_NAME = "{target_name}"',
        text
    )

    CONFIG_PATH.write_text(text)

    print(f"\nRunning: {source_name} -> {target_name}")

    result = subprocess.run(
        [sys.executable, "task2/main.py"],
        capture_output=True,
        text=True
    )

    output = result.stdout.strip()
    print(output)

    try:
        # Parse lines in reverse to find the metrics output, ignoring trailing print statements
        for line in reversed(output.strip().split("\n")):
            try:
                delimiter = "\t" if "\t" in line else ","
                fitness, rmse, rot, trans = map(float, line.split(delimiter))
                fitness_vals.append(fitness)
                rmse_vals.append(rmse)
                rot_vals.append(rot)
                trans_vals.append(trans)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Metrics line not found in output.")

    except Exception:
        print("Could not parse output:")
        print("--- STDOUT ---")
        print(output)
        print("--- STDERR ---")
        print(result.stderr.strip())
        print("--------------")

# -------------------------
# Average
# -------------------------
if fitness_vals:
    avg_fitness = sum(fitness_vals) / len(fitness_vals)
    avg_rmse = sum(rmse_vals) / len(rmse_vals)
    avg_rot = sum(rot_vals) / len(rot_vals)
    avg_trans = sum(trans_vals) / len(trans_vals)

    print("\nAVERAGE")
    print(
        f"{avg_fitness:.6f}\t"
        f"{avg_rmse:.6f}\t"
        f"{avg_rot:.6f}\t"
        f"{avg_trans:.6f}"
    )
else:
    print("\nNo successful registrations to compute average.")

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
