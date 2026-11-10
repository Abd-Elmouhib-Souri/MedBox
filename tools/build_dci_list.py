"""Build the generic-name (DCI) files used by MedBox v2 (PROTOCOL.md section 11).

Usage (from the repo root), after tools.fetch_data:
    python -m tools.build_dci_list

Reads data/raw/medoc/latest.json (same pinned file as the trade-name list) and writes:
    data/dci_tn.csv      one row per normalized DCI (column `dci`)          -> list for molecule matching
    data/brand_dci.csv   trade name -> DCI (columns `name`, `dci`)          -> ground truth for molecule answers
"""

import json
from pathlib import Path

import pandas as pd

from medbox.text_norm import base_name, normalize   # same normalization as the matcher

RAW_JSON = Path("data/raw/medoc/latest.json")


def main() -> None:
    rows = json.loads(RAW_JSON.read_text(encoding="utf-8"))["data"]   # [name, form, dosage, pres, price, reimb, DCI]
    pairs = pd.DataFrame(
        [(base_name(r[0]), normalize(r[6])) for r in rows if r and r[0] and r[6] and str(r[6]).strip()],
        columns=["name", "dci"],
    ).drop_duplicates().sort_values(["name", "dci"])
    dci = pd.DataFrame({"dci": sorted(pairs["dci"].unique())})

    dci.to_csv("data/dci_tn.csv", index=False, encoding="utf-8", lineterminator="\n")
    pairs.to_csv("data/brand_dci.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(f"unique DCI          : {len(dci)}")
    print(f"trade name -> DCI   : {len(pairs)} pairs, {pairs['name'].nunique()} trade names")
    print("written: data/dci_tn.csv, data/brand_dci.csv")


if __name__ == "__main__":
    main()
