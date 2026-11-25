"""Fetch the pinned public checkpoint into this project for offline inference."""
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODEL_ID = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
MODEL_DIR = ROOT / ".models" / REVISION
WEIGHTS_SHA256 = "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "config.json", "model.safetensors"):
        path = MODEL_DIR / name
        if name == "model.safetensors" and path.is_file() and digest(path) == WEIGHTS_SHA256:
            continue
        print(f"Downloading {MODEL_ID} / {name}", flush=True)
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        with tempfile.NamedTemporaryFile(dir=MODEL_DIR, delete=False) as dest:
            partial = Path(dest.name)
        try:
            with urlopen(url, timeout=120) as source, partial.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
            if name == "model.safetensors" and digest(partial) != WEIGHTS_SHA256:
                raise ValueError("Checkpoint SHA-256 mismatch")
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
    print(f"Ready: {MODEL_DIR}")


if __name__ == "__main__":
    main()
