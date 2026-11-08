"""Download the public datasets used by MedBox and print what they contain.

Usage (from the repo root):
    python -m tools.fetch_data

What it does:
    1. Downloads the Kaggle dataset "les-medicaments-en-tunisie" into data/raw/ (ignored by Git).
    2. Prints the SHA-256 of every downloaded file (to copy into data/SOURCES.md).
    3. Prints the shape, the columns and the first rows of every CSV file,
       so we can decide how to build data/drugs_tn.csv from REAL columns, not guessed ones.

Credentials:
    The Kaggle key must be in  C:\\Users\\<you>\\.kaggle\\kaggle.json  (never in this repo, never in the chat).
"""

import hashlib                      # SHA-256 of each file, for reproducibility
from pathlib import Path            # OS-independent paths (Windows and Linux)

import pandas as pd                 # read the CSV to inspect columns (installed with gradio)

# --- Settings ---------------------------------------------------------------------------
KAGGLE_SLUG = "skanderkammoun/les-medicaments-en-tunisie"   # owner/dataset on kaggle.com
RAW_DIR = Path("data/raw/medicaments_tn")                     # raw files stay local, never committed


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in binary mode (line endings do not change it)."""
    h = hashlib.sha256()
    with path.open("rb") as f:                                # "rb" = read raw bytes
        for chunk in iter(lambda: f.read(1 << 20), b""):      # 1 MB at a time: works for big files too
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    """Download and unzip the Kaggle dataset into RAW_DIR."""
    # Import here: "import kaggle" tries to read kaggle.json immediately and fails without it.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()                                        # reads ~/.kaggle/kaggle.json
    RAW_DIR.mkdir(parents=True, exist_ok=True)                # create data/raw/... if missing
    api.dataset_download_files(KAGGLE_SLUG, path=str(RAW_DIR), unzip=True)
    print(f"Downloaded {KAGGLE_SLUG} -> {RAW_DIR}")


def inspect() -> None:
    """Print hash, size and content summary of every downloaded file."""
    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No file found in {RAW_DIR}. Did the download fail?")

    for p in files:
        print("=" * 80)
        print(f"FILE   : {p}")
        print(f"SIZE   : {p.stat().st_size} bytes")
        print(f"SHA256 : {sha256_of(p)}")

        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)                               # if this fails, we will look at the encoding
            print(f"SHAPE  : {df.shape[0]} rows x {df.shape[1]} columns")
            print("COLUMNS:", list(df.columns))
            print("FIRST 5 ROWS:")
            print(df.head(5).to_string())
            # Count unique values per column: helps to find the trade-name column
            print("UNIQUE VALUES PER COLUMN:")
            print(df.nunique().to_string())


if __name__ == "__main__":
    download()
    inspect()
