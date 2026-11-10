"""Prepare the field test set (own phone photos) from data/field_map.csv, then freeze it.

Usage (from the repo root):
    python -m tools.field_set "C:\\path\\to\\folder\\with\\the\\photos"

For each row of data/field_map.csv (photo, true_name, source_file, group):
    1. finds <folder>/<source_file>;
    2. applies EXIF orientation, longest side <= 2048 px, saves a JPEG WITHOUT metadata to data/field/<photo>.jpg
       (exactly the same rules as tools/strip_exif.py);
    3. writes data/field_truth.csv (file,true_name,group) and data/field_manifest.sha256 (all images + truth file).
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

MAP = Path("data/field_map.csv")
OUT_DIR = Path("data/field")
TRUTH = Path("data/field_truth.csv")
MANIFEST = Path("data/field_manifest.sha256")
MAX_SIDE = 2048                                                  # same as tools/strip_exif.py


def main(src: Path) -> None:
    m = pd.read_csv(MAP)
    missing = [f for f in m["source_file"] if not (src / f).exists()]
    if missing:
        raise SystemExit(f"Missing in {src}: {missing}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for photo, source in zip(m["photo"], m["source_file"]):
        out = OUT_DIR / f"{photo}.jpg"
        with Image.open(src / source) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")      # rotate pixels, drop alpha
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            clean = Image.new("RGB", im.size)
            clean.paste(im)                                      # pixels only, no metadata
            clean.save(out, "JPEG", quality=95)
        with Image.open(out) as chk:
            if len(chk.getexif()) > 0:
                raise SystemExit(f"EXIF still present in {out}")
            print(f"OK  {out}  {chk.size[0]}x{chk.size[1]}  <- {source}")

    truth = pd.DataFrame({"file": m["photo"] + ".jpg", "true_name": m["true_name"], "group": m["group"]})
    truth.to_csv(TRUTH, index=False, encoding="utf-8", lineterminator="\n")
    files = sorted(OUT_DIR.glob("*.jpg")) + [TRUTH]
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.as_posix()}" for p in files]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {TRUTH} ({len(truth)} rows), {MANIFEST} ({len(lines)} files)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tools.field_set <folder with the photos>")
    main(Path(sys.argv[1]))
