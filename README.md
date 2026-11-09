# MedBox Reader

**Prototype: read the trade name of a Tunisian medicine from a photo of its box, or refuse.**
YOLOv8 (PyTorch) finds the drug-name region, TrOCR (Hugging Face) reads it, fuzzy matching compares it with a closed
list of 2,160 Tunisian trade names, and the system **refuses** when the match score is below a threshold.

> ⚠️ Prototype for research and learning. **Not a medical tool.** It does not replace a pharmacist.
> It does not read Arabic, dosage, or prescriptions.

---

## Results in one table

| What was measured | Result | Source |
|---|---|---|
| Detector (YOLOv8n), Kaggle **val** split | mAP@0.5 = **0.690** (used to choose between 2 runs, so slightly optimistic) | `results/yolo/yolo_results.json` |
| Detector, Kaggle **test** split, evaluated **once** | mAP@0.5 = **0.667**, mAP@0.5:0.95 = 0.378 (182 images) | `results/yolo/yolo_results.json` |
| End-to-end, frozen pilot test (14 web-collected images: 9 boxes + 5 non-medicine objects) | **0/9 boxes recognized** (95% CI 0.00–0.30) · **0 wrong accepts** (0/14, 95% CI 0.00–0.22) · **5/5 negatives refused** (95% CI 0.57–1.00) | `results/final_test.json` |
| Baseline without detector (TrOCR on the whole image), same test | 0/9 recognized · 0 wrong accepts · 5/5 negatives refused | `results/final_test.json` |
| Decision rule "YOLO is useful" | **undefined**: neither system accepted any image, so top-1 precision has n = 0 | `PROTOCOL.md` §8 |
| Decision rule "demo can be published" (≤ 1 wrong accept) | yes (0 wrong accepts) | `PROTOCOL.md` §8 |

**In plain words:** the system never gave a wrong drug name, but it also never recognized a drug on the test images.
It refused everything. Below is why, and what this teaches.

---

## Pipeline

```
photo ──► YOLOv8n (drug-name region) ──► crop (+5 %) ──► TrOCR at 0°, 90°, 270° ──► rapidfuzz WRatio vs 2,160 names
                                                                                     │
                                       score ≥ threshold (91) ──► trade name   ◄─────┤
                                       score < threshold or no box ──► REFUSED ◄─────┘
Baseline: the same, without YOLO (TrOCR reads the whole image).
```

| Component | Why |
|---|---|
| **YOLOv8n** (Ultralytics, PyTorch), fine-tuned on Kaggle | A box has a lot of text; the detector should isolate the name line. |
| **TrOCR** `microsoft/trocr-base-printed` (Hugging Face), used as is | Encoder–decoder (ViT + Transformer) trained on printed text lines. |
| **rapidfuzz** WRatio + closed list | Corrects OCR errors against known names only. |
| **Refusal threshold** chosen on dev images only | A confident wrong answer is worse than a refusal. |

---

## Error analysis (test, pipeline, 9 boxes)

| Error type | Count | Images | What happened |
|---|---|---|---|
| Detector found no name region | 3 | t03, t04, t06 | Images are tiny (see below); after upscaling to 640 px the text is blurry. |
| Detector picked the **generic name** (DCI), not the brand | 2 | t01, t09 | t01: TrOCR read `DOMPERIDONE` perfectly (Motilium's molecule) → closest brand `DOMPER` (score 90, refused). t09: read `DISMETITE` (diosmectite) → `SMECTITE` (82, refused). |
| OCR produced unrelated text | 4 | t02, t05, t07, t08 | Readings like `1` or `AMOUNT` on crops of a few dozen pixels. |

Non-medicine objects: 5/5 refused (one was read as `-JABON NATURAL-` → `XATRAL`, score 75, refused).

No threshold would have helped: on the test set, **no candidate name was correct** for any box, in the pipeline or in
the baseline (`candidate` field in `results/final_test.json`). A lower threshold would only have added wrong answers.

### Why it failed (findings)

1. **Tiny test images.** The test images are web images and screenshots, not phone photos. Median size of the 9 box
   images: **245 × 182 px** (a phone photo is about 4000 × 3000 px). Only t01 is large (2000 × 2000).
2. **The training labels mix brand and generic names.** In the Kaggle/Roboflow dataset, some boxes are labelled on the
   generic name line (e.g. "Mycophenolate" instead of the brand "Mycept"). The detector learned this, and on t01 and t09
   it cropped the generic name. Matching a generic name against a list of brands produces a *different* brand —
   exactly the dangerous case. The threshold refused it.
3. **WRatio gives 90 to any fragment.** If the OCR text is a short piece of a longer name (`TAX` in `KETAXOL`),
   WRatio's partial match scores 100 × 0.9 = **90**. The dev rule (smallest threshold with 0 wrong accepts) gave
   **91**, just above this artifact. This is why the baseline, which mostly reads fragments, never got accepted.
4. **TrOCR reads one line.** On a whole box (baseline), it outputs a short fragment of one line.

### What I would change next (decided before any new test)

- Take **real phone photos** of real boxes (the planned test, not done; see deviations).
- Refuse OCR readings shorter than 4 characters, before matching.
- Match against **brand and generic names** (the list has a DCI column), and answer "generic name X" instead of guessing a brand.
- Re-label the detector data to mark the brand line only; or detect the whole box first, then the name.

---

## Demo example (not part of the test)

On a larger, sharper web image of a box labelled "Metformin" (not in the test set), the full pipeline worked
mechanically: YOLO found the name line, TrOCR read `METFORMIN` exactly, and the closest list entry was
`METFORMINE SANOFI` (score 90). The system **refused**: "Metformin" is the molecule, and naming a specific
product from another laboratory would be a confident wrong answer. Screenshot not included (image not owned).

---

## Evaluation protocol (pre-registered)

The rules were written in [`PROTOCOL.md`](PROTOCOL.md) and committed **before** each result they govern.
Key commits (`git log --oneline` shows the full order):

| Commit | What it fixed in advance |
|---|---|
| `9f5662b` | Initial protocol: metrics, refusal rule, single final run, decision rules |
| `702180f` | Kaggle leakage rule and single evaluation on the Kaggle test split (before the data audit) |
| `8da3a85` | OCR rotation rule (from the Kaggle audit, before any OCR code) |
| `967ba2a` | Closed list switched to Pharma Tunisie (before checking which boxes it contains) |
| `8e3a28d` | Random dev/test split (seed 42) |
| `b50fa45` | Deviation 2: web-collected images (see below) |
| `4133f68` | Test set frozen with SHA-256 manifest |
| `0d918dc` | Reader and evaluation code |
| `3a35c59` | Threshold 91 chosen on dev |
| `7253798` | Final test run, executed once (`results/final_test.json`) |

The final run (`python -m medbox.evaluate --final`) checks the SHA-256 manifest first and refuses to run twice.

### Deviations (all written before the results they affect)

1. **Fewer boxes than planned:** 12 eligible boxes instead of 30+ → dev 3 boxes + 2 negatives, test 9 boxes + 5 negatives. Pilot scale.
2. **No own photos.** The dev/test images were collected from the web or are screenshots (checked with EXIF: none
   comes from the author's phone). They are cleaner and much smaller than real phone photos, so the end-to-end result
   is **not representative of real use**. They are **not redistributed** (copyright); only their SHA-256 hashes and the
   ground truth are in this repository.
3. **Closed list changed** from the Kaggle list (1,639 names, about two years old) to the more recent and complete
   Pharma Tunisie list (2,160 names), before checking the boxes.
4. **First data audit was wrong** on 14 % of the labels (polygons read as boxes). A corrected audit gives 21 % vertical
   name boxes. Training was not affected (Ultralytics converts polygons to boxes).

---

## Data and licences

| Data | Use | Licence |
|---|---|---|
| [Drug Name Detection](https://www.kaggle.com/datasets/pkdarabi/the-drug-name-detection-dataset) (P. Karimi Darabi; origin Roboflow, Kabul University) — 1,823 images | Train YOLO | CC BY 4.0 |
| [Pharma Tunisie](https://github.com/ballouchi/medoc) `js/latest.json` @ `b03f9d9` — 4,127 rows → 2,160 trade names | Closed list | Apache 2.0 |
| Web-collected test images | Pilot test (local only) | Not redistributed |

Details, hashes and sources considered but not used: [`data/SOURCES.md`](data/SOURCES.md).
The weights `weights/best.pt` are trained on CC BY 4.0 data (attribution above).

---

## Reproduce

```bash
conda create -n medbox python=3.11 -y && conda activate medbox
pip install -r requirements.txt
python -m pytest -q                      # 11 tests, fake models, < 1 s
python -m tools.fetch_data               # drug list (checks SHA-256); needs a Kaggle key in ~/.kaggle
python -m tools.build_drug_list          # -> data/drugs_tn.csv (2,160 names)
python app.py                            # local demo (Gradio)
```

YOLO training: [`notebooks/train_yolo_kaggle.ipynb`](notebooks/train_yolo_kaggle.ipynb) (Kaggle T4, outputs included).
The dev/test evaluation needs the local images, which are not published.

## Limits

Pilot scale (9 test boxes, wide intervals) · web images, not phone photos · no Arabic · no dosage · closed list ·
detector trained on non-Tunisian data · TrOCR not fine-tuned.

## Licence

AGPL-3.0 (compatible with Ultralytics). Author: Abd Elmouhib Souri.
