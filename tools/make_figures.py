"""Figure for the README: how detection works, step by step, on 4 images of the Kaggle TEST split.

Usage (from the repo root):
    python -m tools.make_figures

Images: Kaggle "Drug Name Detection" test split (CC BY 4.0), downloaded into data/raw/kabul/ (ignored by Git).
They are NOT our frozen test set. The 4 images are drawn at random with seed 42, not chosen by hand,
so the figure shows typical cases (successes and failures), not the best ones.

Each row: (1) photo + YOLO box and confidence  (2) crop given to TrOCR  (3) text read, closest name, decision.
Output: results/figures/detection_examples.png
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                  # no window: just write the PNG
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from medbox.reader import Reader, load_config

KABUL = Path("data/raw/kabul")
OUT = Path("results/figures/detection_examples.png")
N, SEED = 4, 42


def find_test_images() -> list[Path]:
    dirs = [d for d in KABUL.rglob("images") if d.parent.name == "test"]
    if not dirs:
        raise SystemExit(f"No test/images folder under {KABUL}. Download the Kaggle dataset first.")
    return sorted(dirs[0].glob("*.jpg"))


def main() -> None:
    reader = Reader(load_config())
    images = random.Random(SEED).sample(find_test_images(), N)       # random draw, not hand-picked

    fig, axes = plt.subplots(N, 3, figsize=(15, 4.2 * N), gridspec_kw={"width_ratios": [1.2, 1, 1.3]})
    for row, path in zip(axes, images):
        im = Image.open(path).convert("RGB")
        box = reader.detect(im)                                       # (x1, y1, x2, y2, conf) or None
        result = reader.identify(im)                                  # full pipeline, threshold from config

        # (1) photo + box
        row[0].imshow(im)
        row[0].set_title(f"1. YOLO: {'no box' if box is None else f'box, confidence {box[4]:.2f}'}", fontsize=11)
        if box is not None:
            x1, y1, x2, y2 = box[:4]
            row[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2.5))

        # (2) crop (+5 %), rotated as kept by the best-scoring reading
        if box is not None:
            crop = reader.crop(im, box)
            rot = result["rotation"] or 0
            row[1].imshow(crop if rot == 0 else crop.rotate(rot, expand=True))
            row[1].set_title(f"2. Crop given to TrOCR (rotation {rot}°)", fontsize=11)
        else:
            row[1].set_title("2. No crop (nothing detected)", fontsize=11)

        # (3) text read and decision
        decision = f"ACCEPTED: {result['name']}" if result["status"] == "ok" else "REFUSED"
        text = (f"TrOCR read:\n  {result['ocr_text']!r}\n\n"
                f"Closest Tunisian name:\n  {result['candidate']}  (score {result['score']})\n\n"
                f"Threshold {reader.config['threshold']}  ->  {decision}")
        row[2].text(0.02, 0.5, text, fontsize=12, va="center", family="monospace")
        row[2].set_title("3. Reading and decision", fontsize=11)
        for ax in row:
            ax.axis("off")
        print(f"{path.name}: box={'none' if box is None else round(box[4], 2)} ocr={result['ocr_text']!r} -> {decision}")

    fig.suptitle("How MedBox reads a box — 4 random images of the Kaggle test split (CC BY 4.0), seed 42", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])                          # leave room for the title
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=90)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
