import urllib.request
from pathlib import Path

MODELS = {
    "haarcascade_frontalface_default.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    ),
    "haarcascade_eye_tree_eyeglasses.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    ),
}


def download_models() -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in MODELS.items():
        target_path = models_dir / filename
        if target_path.exists():
            print(f"[EXISTS] {filename} is already downloaded.")
            continue

        print(f"[DOWNLOADING] {filename}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"[SUCCESS] Saved to {target_path}")
        except Exception as error:
            print(f"[FAILED] Could not download {filename}: {error}")


if __name__ == "__main__":
    download_models()