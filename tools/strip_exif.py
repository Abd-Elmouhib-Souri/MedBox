"""Copy the original phone photos to data/dev and data/test WITHOUT any metadata (EXIF: GPS position, phone model, date).

Usage (from the repo root):
    python -m tools.strip_exif

Input : data/raw/photos/<name>.jpg   (originals, ignored by Git, never committed)
        names must be the ones given by data/split.csv: d01.jpg, dn1.jpg, t01.jpg, tn1.jpg ...
Output: data/dev/<name>.jpg and data/test/<name>.jpg   (clean copies, these are committed)

What it does for each photo:
    1. applies the EXIF orientation to the pixels (otherwise a portrait photo would appear rotated once EXIF is gone);
    2. resizes so the longest side is at most 2048 px (keeps the repo small; YOLO works at 640 px anyway);
    3. saves a new JPEG built from the pixels only (no EXIF, no GPS, no thumbnail);
    4. re-opens the result and checks that no EXIF is left.
"""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

RAW_DIR = Path("data/raw/photos")
SPLIT_CSV = Path("data/split.csv")
MAX_SIDE = 2048          # fixed before any result (same for every photo, dev and test)


def file_name(photo_field: str) -> str:
    """'dn1.jpg (cereal box)' -> 'dn1.jpg'"""
    return photo_field.split(" ")[0]


def main() -> None:
    split = pd.read_csv(SPLIT_CSV)
    expected = {file_name(p): s for p, s in zip(split["photo"], split["split"])}   # name -> dev/test

    missing = [n for n in expected if not (RAW_DIR / n).exists()]
    extra = sorted(p.name for p in RAW_DIR.glob("*") if p.is_file() and p.name not in expected)
    if missing:
        raise SystemExit(f"Missing photos in {RAW_DIR}: {missing}")
    if extra:
        print(f"WARNING: ignored files in {RAW_DIR} (not in split.csv): {extra}")

    for name, split_name in sorted(expected.items()):
        out_dir = Path("data") / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(RAW_DIR / name) as im:
            im = ImageOps.exif_transpose(im)               # 1. rotate pixels as the phone intended
            im = im.convert("RGB")                         # drop alpha / palette modes
            im.thumbnail((MAX_SIDE, MAX_SIDE))             # 2. longest side <= 2048, keeps the ratio
            clean = Image.new("RGB", im.size)
            clean.paste(im)                                # 3. new image = pixels only, no metadata
            clean.save(out_dir / name, "JPEG", quality=95)

        with Image.open(out_dir / name) as check:          # 4. proof: no EXIF left
            exif = check.getexif()
            if len(exif) > 0 or "exif" in check.info:
                raise SystemExit(f"EXIF still present in {out_dir / name}")
            print(f"OK  {out_dir / name}  {check.size[0]}x{check.size[1]}  EXIF tags: 0")

    print(f"\n{len(expected)} photos written without metadata.")


if __name__ == "__main__":
    main()
