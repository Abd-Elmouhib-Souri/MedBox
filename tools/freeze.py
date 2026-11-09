"""Write the ground-truth CSVs and freeze the test set with SHA-256.

Usage (from the repo root), after tools.strip_exif:
    python -m tools.freeze

Writes:
    data/dev_truth.csv, data/test_truth.csv   columns: file,true_name   (true_name = NONE for negatives)
    data/test_manifest.sha256                 one line per frozen file: "<sha256>  <path>"
The final evaluation (medbox.evaluate --final) recomputes these hashes and refuses to run if any file changed.
"""

import hashlib
from pathlib import Path

import pandas as pd

SPLIT_CSV = Path("data/split.csv")
MANIFEST = Path("data/test_manifest.sha256")


def sha256_of(path: Path) -> str:
    """SHA-256 of the raw bytes (binary mode: line endings cannot change it)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    split = pd.read_csv(SPLIT_CSV)
    split["file"] = split["photo"].str.split(" ").str[0]           # 'dn1.jpg (cereal box)' -> 'dn1.jpg'

    for name in ["dev", "test"]:
        part = split[split["split"] == name][["file", "true_name"]]
        for f in part["file"]:                                      # every photo must exist, without EXIF
            if not (Path("data") / name / f).exists():
                raise SystemExit(f"Missing data/{name}/{f}: run tools.strip_exif first")
        out = Path("data") / f"{name}_truth.csv"
        part.to_csv(out, index=False, encoding="utf-8", lineterminator="\n")
        print(f"written: {out} ({len(part)} rows)")

    # Manifest: every test photo + the test truth file
    frozen = sorted(Path("data/test").glob("*.jpg")) + [Path("data/test_truth.csv")]
    lines = [f"{sha256_of(p)}  {p.as_posix()}" for p in frozen]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {MANIFEST} ({len(lines)} files)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
