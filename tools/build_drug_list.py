"""Build data/drugs_tn.csv (closed list for fuzzy matching) from the Pharma Tunisie list (medoc latest.json).

Source decided in PROTOCOL.md: github.com/ballouchi/medoc, js/latest.json, pinned commit (see tools/fetch_data.py).
Each row of latest.json: [name, form, dosage, presentation, price, reimbursed, DCI]. Only the name is used.

Usage (from the repo root):
    python -m tools.build_drug_list

Why a "base name":
    The raw list has one row per product variant ("CARBOLESS 50", "CARBOLESS 100").
    MedBox does not read the dosage (declared limit). If both variants stay in the list,
    an OCR text "CARBOLESS" matches both with the SAME score, and the choice is arbitrary.
    So we match at trade-name level: the dosage at the end of the name is removed.
"""

import json
from pathlib import Path

import pandas as pd

from medbox.text_norm import base_name, normalize   # same functions as the matcher

RAW_JSON = Path("data/raw/medoc/latest.json")
OUT_CSV = Path("data/drugs_tn.csv")


def main() -> None:
    rows = json.loads(RAW_JSON.read_text(encoding="utf-8"))["data"]   # list of lists (read as data only)
    df = pd.DataFrame(rows)
    names = df[0].dropna().astype(str).str.strip()               # column 0 = official name, one per row

    table = pd.DataFrame({"official": names})
    table["normalized"] = table["official"].map(normalize)       # case/accents/spaces removed
    table["name"] = table["official"].map(base_name)             # + trailing dosage removed

    # One row per base name, with how many official variants it groups and 3 examples
    grouped = (
        table.drop_duplicates("official")
        .groupby("name")["official"]
        .agg(n_variants="count", examples=lambda s: " | ".join(sorted(s)[:3]))
        .reset_index()
        .sort_values("name")
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\n")  # LF: stable hash

    # ---- Report: every number below comes from the data, not from a guess ----
    print(f"raw rows                 : {len(df)}")
    print(f"unique official names    : {table['official'].nunique()}")
    print(f"unique after normalize   : {table['normalized'].nunique()}")
    print(f"unique base names (list) : {grouped.shape[0]}")
    changed = table.drop_duplicates("official")
    changed = changed[changed["normalized"] != changed["name"]]
    print(f"official names that lost a dosage suffix: {len(changed)}")
    print("\n15 examples of grouped names (n_variants > 1):")
    print(grouped[grouped["n_variants"] > 1].head(15).to_string(index=False))
    print(f"\nwritten: {OUT_CSV}")


if __name__ == "__main__":
    main()
