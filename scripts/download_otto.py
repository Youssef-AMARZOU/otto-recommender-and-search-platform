"""Download the OTTO dataset (Kaggle dataset: otto/recsys-dataset).

The competition API rejects downloads until the competition rules are
accepted; the dataset endpoint does not, so this script fetches the train
JSONL through the dataset API and normalises it to data/raw/otto/train.jsonl
(the ETL's default input path).

Credentials are read from KAGGLE_USERNAME / KAGGLE_KEY environment variables
(see .env.example) or from the standard ~/.kaggle/kaggle.json token file.
Nothing secret is stored in this repository.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

DATASET = "otto/recsys-dataset"
FILE = "otto-recsys-train.jsonl"
RAW_DIR = Path("data/raw/otto")
TARGET = RAW_DIR / "train.jsonl"

SETUP_INSTRUCTIONS = """\
Kaggle CLI not found or not configured. To fetch OTTO:
  1. pip install kaggle
  2. Create an API token on kaggle.com (Account -> Create API Token)
  3. Export KAGGLE_USERNAME and KAGGLE_KEY (or place kaggle.json in ~/.kaggle/)
  4. Re-run: python scripts/download_otto.py
Expected result: data/raw/otto/train.jsonl (11+ GB)
"""


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"{TARGET} already present; nothing to do.")
        return 0
    if shutil.which("kaggle") is None:
        print(SETUP_INSTRUCTIONS)
        return 1

    command = ["kaggle", "datasets", "download", "-d", DATASET, "-f", FILE, "-p", str(RAW_DIR)]
    result = subprocess.run(command, check=False)  # noqa: S603 — fixed literal command, no user input
    if result.returncode != 0:
        return result.returncode

    archive = RAW_DIR / f"{FILE}.zip"
    if not archive.exists():
        print(f"Expected archive {archive} after download; not found.")
        return 1
    with zipfile.ZipFile(archive) as zf:
        member = next((n for n in zf.namelist() if n.endswith(".jsonl")), None)
        if member is None:
            print(f"No .jsonl member inside {archive}.")
            return 1
        with zf.open(member) as src, TARGET.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    archive.unlink()
    print(f"Extracted {member} to {TARGET}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
