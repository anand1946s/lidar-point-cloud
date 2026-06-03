from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "bunny" / "bunny" / "data"

OUTPUT_DIR = PROJECT_ROOT / "task1" / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)