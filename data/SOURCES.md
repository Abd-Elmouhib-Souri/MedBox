# Data sources

Raw files are downloaded by scripts into `data/raw/` (ignored by Git). Only small derived files are committed.

## 1. Tunisian drug list — `data/drugs_tn.csv`

| Field | Value |
|---|---|
| Source | Kaggle, "les medicaments en tunisie" by Skander Kammoun |
| URL | https://www.kaggle.com/datasets/skanderkammoun/les-medicaments-en-tunisie |
| Version | Version 1 |
| Licence | Apache 2.0 (as shown on the dataset page) |
| Raw file | `medicine_data_tn.csv` — 254,012 bytes — 3,245 rows × 7 columns |
| Raw SHA-256 | `95feb4c2cc0607f69844ee9a4317ed18c8a99a8820547323eb439ba9c4166d13` |
| Downloaded | 2026-11-08, with `python -m tools.fetch_data` |

**Modification (Apache 2.0 requires stating it):** `python -m tools.build_drug_list` keeps only the `Name` column,
normalizes it (uppercase, no accents, single spaces) and removes a dosage written at the end of the name
(`CARBOLESS 50` and `CARBOLESS 100` → `CARBOLESS`). Reason: MedBox does not read the dosage, so two variants of the
same trade name would get the same matching score.

Measured result: 1,756 unique official names → **1,639 trade names** in `data/drugs_tn.csv`
(225 official names lost a dosage suffix).

Known side effect: a trade name that really ends with a number (for example a name like "X 3") also loses that number.

## 2. Drug name detection images (YOLO training) — not committed, used on Kaggle only

| Field | Value |
|---|---|
| Source used | Kaggle, "The Drug Name Detection Dataset" by Parisa Karimi Darabi |
| URL | https://www.kaggle.com/datasets/pkdarabi/the-drug-name-detection-dataset |
| Original source (from its `data.yaml`) | Roboflow, workspace `kabul-university-evptq`, project `drug-name-detection`, version 2: https://universe.roboflow.com/kabul-university-evptq/drug-name-detection/dataset/2 |
| Licence | CC BY 4.0 (Kaggle page and `data.yaml`) |
| Content | 1,823 images, all 640×640 — train 1,276 · valid 365 · test 182 — one class: `drug-name` |
| Label format | YOLO text files; 14 % of the label rows are polygons, not boxes (train 182/1,277). Ultralytics converts them to boxes. |
| Leakage check | 0 images from the same original photo in two splits; 0 identical files across splits (`results/yolo/audit.json`) |

Not modified. The trained weights (`weights/best.pt`) are a derivative work of this dataset; attribution above (CC BY 4.0).

## 3. Considered and not used

- **PharmAI** (Mendeley Data, DOI 10.17632/sdfcbmjfnd.2): no bounding boxes (images are sorted by first letter only),
  and its "augmented" folder is derived from the originals, which would leak between splits. The protocol does not fine-tune OCR.
