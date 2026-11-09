# Data sources

Raw files are downloaded by scripts into `data/raw/` (ignored by Git). Only small derived files are committed.

## 1. Tunisian drug list — `data/drugs_tn.csv`

| Field | Value |
|---|---|
| Source | "Pharma Tunisie — Prix des médicaments en Tunisie", by ballouchi (sources cited on the site: CNAM lists, PCT circulars) |
| Site / repository | https://ballouchi.github.io/medoc/ — https://github.com/ballouchi/medoc (branch `gh-pages`) |
| File used | `js/latest.json` at commit `b03f9d97856b405c136fee5a5e01944b98d46c14` (prices published 2026-10-27) |
| Licence | Apache 2.0 (`LICENSE` file of the repository) |
| Raw file | 4,127 rows: [name, form, dosage, presentation, price, reimbursed, DCI] |
| Raw SHA-256 | `c4f9c895ca032d23aa598198f5a39ad8a9018a9739a0b4fed0af38a54f972964` (checked by `tools/fetch_data.py`) |
| Downloaded | 2026-11-09, with `python -m tools.fetch_data` |

**Modification (Apache 2.0 requires stating it):** `python -m tools.build_drug_list` keeps only the name column,
normalizes it (uppercase, no accents, single spaces) and removes a dosage written at the end of the name
(`CARBOLESS 50` and `CARBOLESS 100` → `CARBOLESS`, including `10 mg/ml`-style suffixes). Reason: MedBox does not
read the dosage, so two variants of the same trade name would get the same matching score.

Measured result: 2,296 unique official names → **2,160 trade names** in `data/drugs_tn.csv`
(265 official names lost a dosage suffix).

Known side effect: a trade name that really ends with a number (for example a name like "X 3") also loses that number.

**Replaced list:** Kaggle "les medicaments en tunisie" by Skander Kammoun (Apache 2.0, `medicine_data_tn.csv`,
3,245 rows, 1,639 trade names after the same cleaning, SHA-256 `95feb4c2…66d13`, page "updated 2 years ago").
Same 7 columns, probably an older copy of the list above. Replaced on 2026-11-09 for recency and completeness,
before checking which owned boxes either list contains (see `PROTOCOL.md`).

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

- **PCT price list on idaraty.tn** (July 2026, circular 2026/30): only the 306 drugs whose price was revised, not a full registry.
- **Roboflow "Med-Name" (DetectTest)**: 50 images, stock photos with a Shutterstock watermark (licence label not trustworthy).
- **Roboflow "Drug names 2"**: handwritten drug names and scanned prescriptions, not medicine boxes.
- These candidates came from an LLM-assisted search; each page was opened and checked by hand.

- **PharmAI** (Mendeley Data, DOI 10.17632/sdfcbmjfnd.2): no bounding boxes (images are sorted by first letter only),
  and its "augmented" folder is derived from the originals, which would leak between splits. The protocol does not fine-tune OCR.
- **med.tn** (Tunisian drug directory): commercial site, no open licence, no download; used only for manual checks, never redistributed.
- **Roboflow "pevvo-drug"**: real phone photos, but polygons around the whole box (not the name), few products, Gulf packaging. Possible future work (box detection first, then name).
- **LaboData**: paid French product database with white-background studio photos; licence does not allow redistribution.
