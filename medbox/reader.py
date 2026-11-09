"""MedBox Reader: photo of a medicine box -> official trade name, or refusal.

Pipeline:  YOLOv8 (drug-name region) -> crop -> TrOCR (read text, 3 rotations) -> rapidfuzz (closed list) -> threshold
Baseline:  same, but TrOCR reads the whole image (no detector).

The models are loaded once, on first use. Tests inject fake models, so they run without downloading anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from rapidfuzz import fuzz, process

from medbox.text_norm import base_name

CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml (paths, model names, fixed settings, threshold)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------
# Real models (loaded lazily). Each one exposes a tiny interface, so tests can replace it.
# ----------------------------------------------------------------------------------------------
class YoloDetector:
    """Wraps Ultralytics YOLO: image -> best box (x1, y1, x2, y2, confidence) or None."""

    def __init__(self, weights: str, conf: float):
        from ultralytics import YOLO                      # imported here: tests do not need it
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, image: Image.Image):
        result = self.model.predict(image, conf=self.conf, verbose=False)[0]
        if len(result.boxes) == 0:
            return None                                   # nothing detected -> the pipeline will refuse
        best = int(result.boxes.conf.argmax())            # keep the most confident box only
        x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
        return (x1, y1, x2, y2, float(result.boxes.conf[best]))


class TrOcr:
    """Wraps Hugging Face TrOCR: image (one text line) -> string."""

    def __init__(self, model_name: str, max_new_tokens: int):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"   # CPU is enough for a demo
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, image: Image.Image) -> str:
        pixels = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        with self.torch.no_grad():                        # inference only: no gradients, less memory
            ids = self.model.generate(pixels, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


# ----------------------------------------------------------------------------------------------
# The reader
# ----------------------------------------------------------------------------------------------
@dataclass
class Reader:
    config: dict
    names: list[str] | None = None                        # closed list; read from config if None
    detector: object | None = None                        # injected in tests, else YoloDetector
    ocr: object | None = None                             # injected in tests, else TrOcr
    _long: list[str] = field(default_factory=list, init=False)
    _short: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        if self.names is None:
            self.names = pd.read_csv(self.config["drug_list"])["name"].astype(str).tolist()
        limit = self.config["short_name_max_len"]
        self._short = {n for n in self.names if len(n) <= limit}      # exact match only (PROTOCOL.md)
        self._long = [n for n in self.names if len(n) > limit]        # fuzzy match

    # -- lazy model loading -------------------------------------------------------------------
    def _get_detector(self):
        if self.detector is None:
            self.detector = YoloDetector(self.config["weights"], self.config["yolo_conf"])
        return self.detector

    def _get_ocr(self):
        if self.ocr is None:
            self.ocr = TrOcr(self.config["ocr_model"], self.config["max_new_tokens"])
        return self.ocr

    # -- steps ----------------------------------------------------------------------------------
    def detect(self, image: Image.Image):
        """Best YOLO box or None."""
        return self._get_detector()(image)

    def crop(self, image: Image.Image, box) -> Image.Image:
        """Crop the box, enlarged by crop_pad on each side, clipped to the image."""
        x1, y1, x2, y2 = box[:4]
        pad_x, pad_y = (x2 - x1) * self.config["crop_pad"], (y2 - y1) * self.config["crop_pad"]
        w, h = image.size
        return image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)))

    def read_text(self, image: Image.Image) -> str:
        """Raw TrOCR text for one image (no rotation)."""
        return self._get_ocr()(image)

    def match(self, text: str) -> tuple[str | None, float]:
        """Closest trade name and its score (0-100). Same normalization as the list (text_norm.base_name)."""
        query = base_name(text)
        if not query:
            return None, 0.0
        if query in self._short:                          # short names: exact match only
            return query, 100.0
        best = process.extractOne(query, self._long, scorer=fuzz.WRatio)   # (name, score, index) or None
        if best is None:
            return None, 0.0
        return best[0], float(best[1])

    def read_best(self, image: Image.Image) -> dict:
        """OCR at each rotation; keep the reading with the highest matching score."""
        best = None
        for angle in self.config["rotations"]:
            img = image if angle == 0 else image.rotate(angle, expand=True)
            text = self.read_text(img)
            name, score = self.match(text)
            if best is None or score > best["score"]:     # ties: the first rotation (0°) wins
                best = {"ocr_text": text, "candidate": name, "score": score, "rotation": angle}
        return best

    def identify(self, image: Image.Image, use_detector: bool = True, threshold: float | None = None) -> dict:
        """Full decision for one image. use_detector=False is the baseline (whole image)."""
        thr = self.config["threshold"] if threshold is None else threshold
        if thr is None:
            raise RuntimeError("No threshold: run `python -m medbox.evaluate --dev` first")
        image = image.convert("RGB")

        box = None
        if use_detector:
            box = self.detect(image)
            if box is None:                               # no name region found -> refuse
                return {"status": "refused", "reason": "no_detection", "name": None, "candidate": None,
                        "score": 0.0, "ocr_text": "", "box": None, "rotation": None}
            region = self.crop(image, box)
        else:
            region = image

        r = self.read_best(region)
        ok = r["candidate"] is not None and r["score"] >= thr
        return {
            "status": "ok" if ok else "refused",
            "reason": None if ok else "below_threshold",
            "name": r["candidate"] if ok else None,       # the answer shown to the user
            "candidate": r["candidate"],                  # kept for error analysis, even when refused
            "score": round(r["score"], 2),
            "ocr_text": r["ocr_text"],
            "box": None if box is None else [round(v, 1) for v in box[:4]],
            "rotation": r["rotation"],
        }


_READER: Reader | None = None


def identify(image: Image.Image, use_detector: bool = True) -> dict:
    """Module-level helper used by the demo: one shared Reader, models loaded once."""
    global _READER
    if _READER is None:
        _READER = Reader(load_config())
    return _READER.identify(image, use_detector=use_detector)
