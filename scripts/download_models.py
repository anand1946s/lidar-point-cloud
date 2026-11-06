import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS = {
    "dgr_3dmatch": {
        "url": (
            "http://node2.chrischoy.org/data/projects/DGR/"
            "ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        ),
        "path": (
            BASE_DIR
            / "source_codes/DeepGlobalRegistration/"
            "ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        ),
    }
}


def download_model(name, model):
    destination = model["path"]

    if destination.exists():
        print(f"[OK] {name} already exists")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".part")

    print(f"[DOWNLOAD] {name}")

    try:
        urllib.request.urlretrieve(
            model["url"],
            temporary
        )

        temporary.rename(destination)
        print(f"[OK] Saved to {destination}")

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    for name, model in MODELS.items():
        download_model(name, model)
