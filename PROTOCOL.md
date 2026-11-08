# MedBox Reader — Pre-registered evaluation protocol

This file is committed **before** any test photo is taken.
Its commit hash is written in the README. The rules below do not change after that commit.
If a rule must change, the change is a new commit, it is explained in the README, and the old rule stays visible in the history.

## 1. Unit of evaluation
- One photo = one unit.
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
