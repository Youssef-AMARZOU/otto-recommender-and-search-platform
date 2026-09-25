"""Download the OTTO dataset (Kaggle: otto-recommender-system).

Credentials are read from KAGGLE_USERNAME / KAGGLE_KEY environment variables
(see .env.example) or from the standard ~/.kaggle/kaggle.json token file.
Nothing secret is stored in this repository.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

COMPETITION = "otto-recommender-system"
RAW_DIR = Path("data/raw/otto")

SETUP_INSTRUCTIONS = """\
Kaggle CLI not found or not configured. To fetch OTTO:
  1. pip install kaggle
  2. Create an API token on kaggle.com (Account -> Create API Token)
  3. Export KAGGLE_USERNAME and KAGGLE_KEY (or place kaggle.json in ~/.kaggle/)
  4. Re-run: python scripts/download_otto.py
Expected files after download/unzip: train.parquet, test.parquet, sample_submission.parquet
"""


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        print(SETUP_INSTRUCTIONS)
        return 1

    command = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)]
    result = subprocess.run(command, check=False)  # noqa: S603 — fixed literal command, no user input
    if result.returncode == 0:
        print(f"OTTO archive downloaded to {RAW_DIR}. Unzip before running the ETL.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
