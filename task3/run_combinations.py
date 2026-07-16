import subprocess
import sys
import re
from pathlib import Path
import json
import time

CONFIG_PATH = Path("task3/config.py")

FEATURE_METHODS = ["NONE", "ISS", "3DSIFT", "HARRIS3D", "FPFH", "SHOT"]
REGISTRATION_METHODS = ["ICP", "ROBUST_ICP", "FGR", "GO_ICP"]

# Backup original config.py
original_config = CONFIG_PATH.read_text()

results = []

try:
    for reg in REGISTRATION_METHODS:
        for feat in FEATURE_METHODS:
            print(f"\n==========================================")
            print(f"Running Combo: Feature={feat}, Registration={reg}")
            print(f"==========================================")
            
            # Read config and update FEATURE_METHOD and REGISTRATION_METHOD
            text = CONFIG_PATH.read_text()
            
            text = re.sub(
                r'FEATURE_METHOD\s*=\s*".*"',
                f'FEATURE_METHOD = "{feat}"',
                text
            )
            text = re.sub(
                r'REGISTRATION_METHOD\s*=\s*".*"',
                f'REGISTRATION_METHOD = "{reg}"',
                text
            )
            
            CONFIG_PATH.write_text(text)
            
            start_time = time.time()
            
            # Ensure environment variables are set correctly for subprocess
            import os
            env = os.environ.copy()
            if sys.platform.startswith("linux"):
                env["GDK_BACKEND"] = "x11"
                env["XDG_SESSION_TYPE"] = "x11"
                
            res = subprocess.run(
                [sys.executable, "task3/batch.py"],
                capture_output=True,
                text=True,
                env=env
            )
            elapsed = time.time() - start_time
            
            stdout = res.stdout
            stderr = res.stderr
            
            print(f"Done in {elapsed:.2f} seconds.")
            
            # Parse AVERAGE block from stdout
            # Output of batch.py looks like:
            # AVERAGE
            # <fitness>\t<rmse>\t<rot>\t<trans>
            
            fitness, rmse, rot, trans = None, None, None, None
            success = False
            
            try:
                lines = stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if "AVERAGE" in line and i + 1 < len(lines):
                        metrics_line = lines[i+1].strip()
                        delimiter = "\t" if "\t" in metrics_line else ","
                        fitness, rmse, rot, trans = map(float, metrics_line.split(delimiter))
                        success = True
                        break
            except Exception as e:
                print(f"Failed to parse output: {e}")
            
            combo_result = {
                "feature_method": feat,
                "registration_method": reg,
                "success": success,
                "fitness": fitness,
                "rmse": rmse,
                "rot_error": rot,
                "trans_error": trans,
                "time_seconds": elapsed,
                "stdout": stdout,
                "stderr": stderr
            }
            print(f"Result: success={success}, fitness={fitness}, rmse={rmse}")
            results.append(combo_result)
            
            # Write checkpoint
            with open("task3_combos_checkpoint.json", "w") as f:
                json.dump(results, f, indent=4)

finally:
    # Restore original config.py
    CONFIG_PATH.write_text(original_config)
    print("Original configuration restored.")
