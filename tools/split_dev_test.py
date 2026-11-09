"""Random dev/test assignment of the boxes and negatives (rule fixed in PROTOCOL.md, section 2).

Usage (from the repo root):
    python -m tools.split_dev_test

Inputs:
    data/inventory_check.csv  (from tools.check_inventory) -> only EXACT rows are used
    data/negatives.txt        one non-medicine object per line, short generic description (e.g. "tea box")
Output:
    data/split.csv            columns: split, kind, true_name, photo
                              photo = the file name to give to the photo (d01.jpg, t01.jpg, dn1.jpg, tn1.jpg...)
"""

import random
from pathlib import Path

import pandas as pd

N_DEV_BOXES, N_DEV_NEG = 3, 2          # fixed in PROTOCOL.md before the draw
SEED = 42

INV = Path("data/inventory_check.csv")
NEG = Path("data/negatives.txt")
OUT = Path("data/split.csv")


def draw(items: list[str], n_dev: int) -> tuple[list[str], list[str]]:
    """Sort (so the input order does not matter), shuffle with a fixed seed, cut."""
    items = sorted(items)
    random.Random(SEED).shuffle(items)
    return items[:n_dev], items[n_dev:]


def main() -> None:
    inv = pd.read_csv(INV)
    boxes = inv.loc[inv["status"] == "EXACT", "list_name"].tolist()
    if len(boxes) != len(set(boxes)):
        raise SystemExit("Two boxes have the same trade name: keep only one per name, then run again.")
    negs = [l.strip() for l in NEG.read_text(encoding="utf-8").splitlines() if l.strip()]

    dev_b, test_b = draw(boxes, N_DEV_BOXES)
    dev_n, test_n = draw(negs, N_DEV_NEG)

    rows = []
    rows += [("dev", "box", n, f"d{i:02d}.jpg") for i, n in enumerate(dev_b, 1)]
    rows += [("dev", "negative", "NONE", f"dn{i}.jpg ({d})") for i, d in enumerate(dev_n, 1)]
    rows += [("test", "box", n, f"t{i:02d}.jpg") for i, n in enumerate(test_b, 1)]
    rows += [("test", "negative", "NONE", f"tn{i}.jpg ({d})") for i, d in enumerate(test_n, 1)]
    df = pd.DataFrame(rows, columns=["split", "kind", "true_name", "photo"])
    df.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")

    print(df.to_string(index=False))
    print(f"\ndev: {len(dev_b)} boxes + {len(dev_n)} negatives | test: {len(test_b)} boxes + {len(test_n)} negatives")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
