"""Prepare and freeze the v2 independent test set (PROTOCOL.md section 12) from data/test_v2_map.csv.

Usage (from the repo root):
    python -m tools.test_v2_set "C:\\path\\to\\the\\folder\\with\\the\\photos"

For each row of data/test_v2_map.csv (photo, true_name, source_file, group, note):
    - group "excluded" rows are skipped (they stay in the map, with their reason, for the README);
    - other rows: finds <folder>/<source_file>, applies EXIF orientation, longest side <= 2048 px, saves a JPEG
      WITHOUT metadata to data/test_v2/<photo>.jpg (exactly the rules of tools/field_set.py);
Then writes data/test_v2_truth.csv (file,true_name,group) and data/test_v2_manifest.sha256 (all images + truth file).
It refuses to run if the manifest already exists: the test set is frozen once.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

MAP = Path("data/test_v2_map.csv")
OUT_DIR = Path("data/test_v2")
TRUTH = Path("data/test_v2_truth.csv")
MANIFEST = Path("data/test_v2_manifest.sha256")
MAX_SIDE = 2048                                                  # same as tools/field_set.py


def main(src: Path) -> None:
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} already exists: the v2 test set is frozen (PROTOCOL.md section 12).")
    m = pd.read_csv(MAP, keep_default_na=False)                  # keep "NONE" as text, empty notes as ""
    keep = m[m["group"] != "excluded"]
    missing = [f for f in keep["source_file"] if not (src / f).exists()]
    if missing:
        raise SystemExit(f"Missing in {src}: {missing}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for photo, source in zip(keep["photo"], keep["source_file"]):
        out = OUT_DIR / f"{photo}.jpg"
        with Image.open(src / source) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")      # rotate pixels, drop alpha
            im.thumbnail((MAX_SIDE, MAX_SIDE))                   # longest side <= 2048, ratio kept
            clean = Image.new("RGB", im.size)
            clean.paste(im)                                      # pixels only, no metadata
            clean.save(out, "JPEG", quality=95)
        with Image.open(out) as chk:                             # proof: no EXIF left
            if len(chk.getexif()) > 0:
                raise SystemExit(f"EXIF still present in {out}")
            print(f"OK  {out}  {chk.size[0]}x{chk.size[1]}  <- {source}")

    truth = pd.DataFrame({"file": keep["photo"] + ".jpg", "true_name": keep["true_name"], "group": keep["group"]})
    truth.to_csv(TRUTH, index=False, encoding="utf-8", lineterminator="\n")
    files = sorted(OUT_DIR.glob("*.jpg")) + [TRUTH]
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.as_posix()}" for p in files]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    counts = truth["group"].value_counts().to_dict()
    print(f"written: {TRUTH} ({len(truth)} rows: {counts}), {MANIFEST} ({len(lines)} files)")
    print(f"excluded and reported: {int((m['group'] == 'excluded').sum())} photos (see {MAP})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tools.test_v2_set <folder with the photos>")
    main(Path(sys.argv[1]))
