"""Copy 3 random images of the Kaggle test split into examples/ (resized, no metadata).

Source: Drug Name Detection dataset (CC BY 4.0). Redistribution is allowed with
attribution, which this script writes to examples/ATTRIBUTION.md.
Run once: python -m tools.make_examples
"""
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

SRC = Path("data/raw/kabul/test/images")  # local Kaggle copy (ignored by Git)
DST = Path("examples")                    # shipped with the repository
N, SEED = 3, 42                           # 3 images, drawn at random (not hand-picked)
MAX_SIDE, QUALITY = 1024, 90              # keep the repository light


def dataset_url() -> str:
    # Reuse the dataset link already written in the README (no URL typed by hand)
    readme = Path("README.md").read_text(encoding="utf-8")
    m = re.search(r"\[Drug Name Detection\]\((https?://[^)\s]+)\)", readme)
    if not m:
        sys.exit("ERREUR: lien 'Drug Name Detection' introuvable dans README.md")
    return m.group(1)


def main() -> None:
    files = sorted(SRC.glob("*.jpg"))  # sorted: same order on every machine
    if len(files) < N:
        sys.exit(f"ERREUR: moins de {N} images dans {SRC}")
    if DST.exists() and any(DST.glob("*.jpg")):
        sys.exit("ERREUR: examples/ contient deja des images - rien n'a ete modifie")
    url = dataset_url()

    picks = random.Random(SEED).sample(files, N)  # reproducible random draw
    DST.mkdir(exist_ok=True)
    rows = []
    for i, src in enumerate(picks, start=1):
        out = DST / f"kaggle_test_{i}.jpg"
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")  # apply orientation, drop alpha
            im.thumbnail((MAX_SIDE, MAX_SIDE))               # longest side <= 1024, ratio kept
            im.save(out, "JPEG", quality=QUALITY)            # no exif passed: no metadata
            print(f"{out}  <-  {src.name}  {im.size}")
        rows.append(f"| `{out.name}` | `{src.name}` |")

    attribution = [
        "# Example images",
        "",
        f"These {N} images come from the test split of the **Drug Name Detection** dataset",
        "by P. Karimi Darabi (origin: Roboflow, Kabul University), licensed under",
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).",
        "",
        f"Dataset: {url}",
        "",
        f"Changes: drawn at random (seed {SEED}, `tools/make_examples.py`), resized",
        f"(longest side <= {MAX_SIDE} px), re-encoded as JPEG (quality {QUALITY}), metadata removed.",
        "",
        "| File | Original file in the dataset |",
        "|---|---|",
        *rows,
    ]
    (DST / "ATTRIBUTION.md").write_text("\n".join(attribution) + "\n", encoding="utf-8")
    print(f"OK - {N} images + examples/ATTRIBUTION.md")


if __name__ == "__main__":
    main()