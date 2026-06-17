from pathlib import Path
import subprocess
import re
import open3d as o3d
import sys
import time


PAIRS = [
    ("bun000", "bun045"),
    ("bun045", "bun090"),
    ("bun270", "bun315"),
    ("bun315", "bun000"),
]

CONFIG_PATH = Path("task1/config.py")

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

    print(
        f"\nRunning: "
        f"{source_name} -> {target_name}"
    )

    result = subprocess.run(
        [sys.executable, "task1/main.py"],
        capture_output=True,
        text=True
    )
   

    output = result.stdout.strip()

    print(output)

    try:
        # Parse lines in reverse to find the metrics output, ignoring trailing print statements
        for line in reversed(output.strip().split("\n")):
            try:
                fitness, rmse, rot, trans = map(float, line.split(","))
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

avg_fitness = (
    sum(fitness_vals)
    / len(fitness_vals)
)

avg_rmse = (
    sum(rmse_vals)
    / len(rmse_vals)
)

avg_rot = (
    sum(rot_vals)
    / len(rot_vals)
)

avg_trans = (
    sum(trans_vals)
    / len(trans_vals)
)

print("\nAVERAGE")

print(
    f"{avg_fitness:.6f},"
    f"{avg_rmse:.6f},"
    f"{avg_rot:.6f},"
    f"{avg_trans:.6f}"
)

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")