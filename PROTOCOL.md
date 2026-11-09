# MedBox Reader — Pre-registered evaluation protocol

This file is committed **before** any test photo is taken.
Its commit hash is written in the README. The rules below do not change after that commit.
If a rule must change, the change is a new commit, it is explained in the README, and the old rule stays visible in the history.

## 1. Unit of evaluation
- One photo = one unit.
- Closed list (decided 2026-11-09, **before** checking which owned boxes it contains): the Pharma Tunisie list,
  github.com/ballouchi/medoc, file `js/latest.json` at commit `b03f9d9` (prices published 2026-10-27), Apache 2.0.
  It replaces the Kaggle list "les-medicaments-en-tunisie" (Apache 2.0, last updated about two years earlier),
  which has the same 7 columns and is probably an older copy of it. Criteria: more recent, more complete
  (4,127 rows vs 3,245; 2,160 trade names vs 1,639), original source, open licence.
- Ground truth: `data/test_truth.csv` with columns `file,true_name`.
  - `true_name` is the exact name from `data/drugs_tn.csv` (unique trade name, no dosage).
  - `true_name = NONE` for negative photos (not a medicine).
- Name comparison: both names are normalized the same way as in `medbox.reader.match()`
  (uppercase, accents removed, multiple spaces collapsed). A prediction is correct only if the normalized names are equal.

## 2. Data splits (own photos)
- **Dev:** about 10 medicine boxes + 2 negative photos. Used **only** to choose the refusal threshold.
- **Test:** 20–30 **other** medicine boxes (never a box used in dev) + 3–5 negative photos.
- Test photos and `test_truth.csv` are frozen with SHA-256 (`data/test_manifest.sha256`) and committed **before** any threshold is tuned.
- Privacy: no prescription, no name, no address, no face in the frame. EXIF metadata is removed before commit.
- Photo processing, identical for dev and test (`tools/strip_exif.py`): EXIF orientation applied to the pixels,
  longest side reduced to at most 2048 px, saved as JPEG (quality 95) without any metadata. Originals stay local.
- Photo-taking rule: one photo per object, phone camera, normal home light, the way a user would take it.
  A photo is retaken only if a human cannot read the trade name on it (decided before any system output is seen).

### Deviation (decided 2026-11-09, before any own photo was taken)
- Fewer boxes are available than planned (boxes physically owned, trade name present in `data/drugs_tn.csv`).
  The planned sizes (about 10 dev boxes, 20–30 test boxes) cannot be met.
- New sizes, fixed before knowing the final number N of eligible boxes:
  **dev = 3 boxes + 2 negatives**, **test = all other eligible boxes (N − 3) + the other negatives (5)**.
- Assignment is random, not chosen by hand: eligible names sorted alphabetically, shuffled with `random.Random(42)`,
  first 3 to dev, the rest to test (`python -m tools.split_dev_test`, output `data/split.csv`). Same rule for negatives.
- Consequence, stated in advance: this is a **pilot-scale** evaluation. With about 10 test boxes, the 95 % Wilson
  intervals are very wide, and no general accuracy claim is made. Results are reported as counts (x / n), never as a
  percentage alone.
- One photo per box (photos of the same box are not independent).

### Deviation 2 (decided 2026-11-09, before any system output on these images)
- **No own phone photos are used.** The dev and test images are images **collected from the web and screenshots**
  (cropped from a few larger photos). Checked with EXIF: none comes from the author's phone.
- Consequences, stated in advance: the images are cleaner than real phone photos (often studio-like or already
  cropped around the box), so the results are **not representative of real use**; some images may also exist online,
  so a pretrained model may have seen them. Results are reported as a **pilot on web-collected images**.
- The images are **not redistributed** (copyright). They stay on the author's computer. Only the ground truth and
  the SHA-256 manifest are committed, so the final run still refuses to start if any image changed.
- Selection rule for each name drawn in `data/split.csv`: the **lowest-numbered** file (`medN` / `negN`) that shows
  that product **alone**; written in `data/image_map.csv` and committed before any OCR or matching code is run.
- The tool `tools/strip_exif.py` is still applied (same orientation, size and JPEG rules for every image).

## 3. Metrics (on the test set)
Let P = positive photos, A = accepted positive photos.
- **Top-1 precision:** correct names among accepted positive photos (x / A).
- **Coverage:** accepted positive photos among all positive photos (A / P).
- **Wrong accepts:** accepted photos (positive or negative) with a wrong name. **This is the most important metric.**
- **Negatives refused:** refused negative photos / all negative photos.
- Every proportion is reported as `x / n` with a **95 % Wilson interval**.
- If a denominator is 0 (for example, no photo accepted), the metric is reported as **"undefined"**, never as 0 % or 100 %.

## 4. Systems compared
- **Pipeline:** YOLOv8n detects the drug-name region → best box is cropped → TrOCR reads the crop → rapidfuzz match → threshold.
- **Baseline (no detector):** TrOCR on the whole resized image → same rapidfuzz match → same threshold.
- In both systems, if the detector finds nothing, or the match score is below the threshold, the output is **refused**.
- Rotation (decided from the Kaggle data audit, where about one name box in five is taller than wide: 21.1 % in train after converting polygon labels to boxes):
  TrOCR reads horizontal lines, so OCR is run on the image at 0°, 90° and 270°, and the reading with the best
  matching score is kept. The pipeline does this on the crop, the baseline on the whole image: same rule for both.

## 5. Refusal threshold
- Matching score: `rapidfuzz.fuzz.WRatio`, range 0–100.
- Short names: trade names of **3 characters or less** (5 in the list used: `LRJ`, `MAG`, `MP`, `T`, `VAL`) get high
  partial-match scores against unrelated text. They are accepted **only on an exact match** (normalized OCR text equal to the name).
- Sweep on **dev only**, integer thresholds from 0 to 100 (step 1). A photo is accepted if `score >= threshold`.
- Rule: choose the **smallest threshold that gives 0 wrong accepts on dev** (pipeline system).
- If no threshold gives 0 wrong accepts, the threshold is set to 100 and this is reported.
- The chosen value is written in `config.yaml` and saved in `results/dev_sweep.json`. It does not change after that.

## 6. YOLO training (Kaggle GPU)
- Model: `yolov8n.pt` (Ultralytics), `epochs=50`, `imgsz=640`, `batch=16`, `seed=42`, `deterministic=True`.
  Optimizer and learning rate: Ultralytics defaults (`optimizer="auto"`). No learning-rate search.
- Early stopping: `patience=10` (training stops if val fitness does not improve for 10 epochs). The kept weights are `best.pt`.
- No dropout: in Ultralytics, `dropout` only acts on classification heads, not on the detection head. Regularization comes from
  pretrained weights, default weight decay and training-time augmentation.
- Data: Kaggle "Drug Name Detection" dataset, with its own train/val split. Our own photos are **never** used for training or for choosing YOLO settings.
- **Data audit first** (notebook, Kaggle data only): licence, label format, class names, box sizes, image sizes, corrupted files, empty labels, exact duplicates between train and val.
  If exact duplicates are found between train and val, they are removed from val before training, and their number is reported.
  The dataset comes from Roboflow: file names look like `<original>_jpg.rf.<hash>.jpg`. If images made from the same
  `<original>` appear in two splits, those images are removed from valid and test (kept in train), and their number is reported.
- **Kaggle test split** (182 images): never used for training or for choosing between run A and run B.
  After the choice, the kept run is evaluated **once** on it, and its mAP@0.5 is reported next to the val value.
  If the labels do not mark the drug name, training stops and the plan is reviewed before any change.
- **No offline image filters** (sharpening, edge/contour maps, denoising, retouching) on training images:
  an image must look the same at training time and at inference time. Only random training-time augmentation is used.
- **At most 2 training runs**, decided now:
  - Run A: Ultralytics default augmentation.
  - Run B: text-aware augmentation: `fliplr=0.0` (mirrored text never exists on real boxes), `degrees=5`, `perspective=0.0005`; everything else default.
  - The run with the higher **mAP@0.5 on the Kaggle val split** is kept. Both values are reported.
    Because val is used to choose, the kept val mAP is slightly optimistic; this is stated in the README.
- Reported: **mAP@0.5 on the Kaggle val split**. This is a development metric, not the final result.

## 7. Single final run
- The test is run **once**, with `python -m medbox.evaluate --final`, which writes `results/final_test.json`.
- The script checks `data/test_manifest.sha256` first and refuses to run if any test file has changed.
- If a bug is found after the run, it is fixed and documented. A second score is never presented as the first one.
- Each error is classified as: detection failed, OCR failed, name not in list, confusion between two close names.

## 8. Decision rules (fixed in advance)
- **"YOLO is useful"** only if the pipeline has a higher top-1 precision than the baseline **and** no more wrong accepts.
- **"Demo can be published"** only if wrong accepts on the test set are **≤ 1**.

## 9. Declared limits
- No Arabic text reading. No dosage reading.
- Closed list: a drug that is not in `data/drugs_tn.csv` cannot be recognized.
- Small test set, so confidence intervals are wide.
- YOLO training data is not Tunisian.
- This is a prototype. It is not a medical tool and does not replace a pharmacist.
