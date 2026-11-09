"""Check which medicine boxes can be used as positives (their trade name must be in data/drugs_tn.csv).

Usage (from the repo root):
    1. Write data/box_names.txt : one box per line, the big name printed on the box (dosage not needed).
    2. python -m tools.check_inventory

For each line it prints:
    EXACT     -> the normalized name is in the list: the box is eligible, use the name shown.
    CHECK     -> no exact match; the 3 closest list names are shown. If one is clearly the same
                 product (spelling difference), write that list name in box_names.txt and run again.
    The tool never decides for you: a box is eligible only with an EXACT line.
Result: data/inventory_check.csv (also used later by the dev/test random split).
"""

from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from medbox.text_norm import base_name   # same normalization as the future matcher

NAMES_TXT = Path("data/box_names.txt")
LIST_CSV = Path("data/drugs_tn.csv")
OUT_CSV = Path("data/inventory_check.csv")


def main() -> None:
    names = pd.read_csv(LIST_CSV)["name"].tolist()                 # 1,639 trade names
    known = set(names)
    lines = [l.strip() for l in NAMES_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    for raw in lines:
        q = base_name(raw)                                           # "Doliprane 500mg" -> "DOLIPRANE"
        if q in known:
            rows.append({"box_text": raw, "status": "EXACT", "list_name": q, "closest_3": ""})
        else:
            top = process.extract(q, names, scorer=fuzz.WRatio, limit=3)   # (name, score, index)
            closest = " | ".join(f"{n} ({s:.0f})" for n, s, _ in top)
            rows.append({"box_text": raw, "status": "CHECK", "list_name": "", "closest_3": closest})

    df = pd.DataFrame(rows)
    dup = df[(df.status == "EXACT") & df.duplicated("list_name", keep=False)]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\n")

    print(df.to_string(index=False))
    print(f"\nEXACT (eligible): {(df.status == 'EXACT').sum()}   CHECK: {(df.status == 'CHECK').sum()}")
    if len(dup):
        print("WARNING: the same trade name appears twice (two boxes of the same product):")
        print(dup[["box_text", "list_name"]].to_string(index=False))
    print(f"written: {OUT_CSV}")


if __name__ == "__main__":
    main()
