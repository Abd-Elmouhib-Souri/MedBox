"""MedBox Reader v2 (PROTOCOL.md section 11): read ALL text lines, keep the tallest ones, match brands AND molecules.

    text detector (CRAFT, EasyOCR) -> 5 tallest text boxes -> crop (+5 %) -> TrOCR at 0/90/180/270
    -> drop readings < 4 letters/digits -> rapidfuzz `ratio` vs 2,160 trade names and 728 DCI
    -> trade name if score >= T_brand, else "molecule X" if score >= T_dci, else refuse.

No YOLO in v2. Models are loaded lazily; tests inject fakes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image
from rapidfuzz import fuzz, process

from medbox.reader import TrOcr, load_config
from medbox.text_norm import base_name, normalize


class CraftDetector:
    """EasyOCR's CRAFT text detector only (no EasyOCR recognizer): image -> list of (x1, y1, x2, y2)."""

    def __init__(self, langs: list[str]):
        import easyocr                                         # imported here: tests do not need it
        self.reader = easyocr.Reader(langs, gpu=False, recognizer=False, verbose=False)

    def __call__(self, image: Image.Image) -> list[tuple[float, float, float, float]]:
        horizontal, free = self.reader.detect(np.array(image.convert("RGB")))
        boxes = [(x1, y1, x2, y2) for x1, x2, y1, y2 in horizontal[0]]          # [x_min, x_max, y_min, y_max]
        for poly in free[0]:                                                   # rotated text: its bounding box
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return [tuple(float(v) for v in b) for b in boxes]


def alnum_count(text: str) -> int:
    """Number of letters and digits after normalization (the < 4 filter)."""
    return len(re.sub(r"[^A-Z0-9]", "", normalize(text)))


@dataclass
class ReaderV2:
    config: dict                                   # the `v2` block of config.yaml
    brands: list[str] | None = None
    dcis: list[str] | None = None
    detector: object | None = None                 # injected in tests, else CraftDetector
    ocr: object | None = None                      # injected in tests, else TrOcr
    _long: list[str] = field(default_factory=list, init=False)
    _short: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        if self.brands is None:
            self.brands = pd.read_csv(self.config["drug_list"])["name"].astype(str).tolist()
        if self.dcis is None:
            self.dcis = pd.read_csv(self.config["dci_list"])["dci"].astype(str).tolist()
        self._short = {n for n in self.brands if len(n) <= self.config["short_name_max_len"]}
        self._long = [n for n in self.brands if len(n) > self.config["short_name_max_len"]]

    def _det(self):
        if self.detector is None:
            self.detector = CraftDetector(self.config["easyocr_langs"])
        return self.detector

    def _ocr(self):
        if self.ocr is None:
            self.ocr = TrOcr(self.config["ocr_model"], self.config["max_new_tokens"])
        return self.ocr

    # -- matching ---------------------------------------------------------------------------------
    def match_brand(self, text: str) -> tuple[str | None, float]:
        q = base_name(text)
        if not q:
            return None, 0.0
        if q in self._short:                                       # short trade names: exact only
            return q, 100.0
        best = process.extractOne(q, self._long, scorer=fuzz.ratio)
        return (best[0], float(best[1])) if best else (None, 0.0)

    def match_dci(self, text: str) -> tuple[str | None, float]:
        q = normalize(text)
        if not q:
            return None, 0.0
        best = process.extractOne(q, self.dcis, scorer=fuzz.ratio)
        return (best[0], float(best[1])) if best else (None, 0.0)

    # -- reading ----------------------------------------------------------------------------------
    def candidates(self, image: Image.Image) -> list[dict]:
        """Every reading of the K tallest text boxes, at every rotation, with its best brand and DCI."""
        image = image.convert("RGB")
        boxes = sorted(self._det()(image), key=lambda b: b[3] - b[1], reverse=True)[: self.config["top_k"]]
        out = []
        w, h = image.size
        for x1, y1, x2, y2 in boxes:
            px, py = (x2 - x1) * self.config["crop_pad"], (y2 - y1) * self.config["crop_pad"]
            crop = image.crop((max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)))
            for angle in self.config["rotations"]:
                text = self._ocr()(crop if angle == 0 else crop.rotate(angle, expand=True))
                if alnum_count(text) < self.config["min_chars"]:   # filter: too short to trust
                    continue
                brand, bs = self.match_brand(text)
                dci, ds = self.match_dci(text)
                out.append({"box": [round(v, 1) for v in (x1, y1, x2, y2)], "height": round(y2 - y1, 1),
                            "rotation": angle, "text": text, "brand": brand, "brand_score": round(bs, 2),
                            "dci": dci, "dci_score": round(ds, 2)})
        return out

    @staticmethod
    def best(cands: list[dict], kind: str) -> dict | None:
        """Highest score for `kind` ('brand' or 'dci'); ties -> taller text."""
        valid = [c for c in cands if c[kind] is not None]
        return max(valid, key=lambda c: (c[f"{kind}_score"], c["height"])) if valid else None

    @staticmethod
    def decide(cands: list[dict], t_brand: float, t_dci: float) -> dict:
        b, d = ReaderV2.best(cands, "brand"), ReaderV2.best(cands, "dci")
        if b and b["brand_score"] >= t_brand:
            return {"status": "brand", "answer": b["brand"], "score": b["brand_score"], "from": b}
        if d and d["dci_score"] >= t_dci:
            return {"status": "molecule", "answer": d["dci"], "score": d["dci_score"], "from": d}
        return {"status": "refused", "answer": None, "score": max([0.0] + [c["brand_score"] for c in cands]),
                "from": b}

    def identify(self, image: Image.Image) -> dict:
        t_b, t_d = self.config["T_brand"], self.config["T_dci"]
        if t_b is None or t_d is None:
            raise RuntimeError("No v2 thresholds: run `python -m medbox.evaluate_v2 --dev` first")
        cands = self.candidates(image)
        return {**self.decide(cands, t_b, t_d), "candidates": cands}


def load_v2_config() -> dict:
    cfg = load_config()
    return {**cfg["v2"], "ocr_model": cfg["ocr_model"], "max_new_tokens": cfg["max_new_tokens"],
            "drug_list": cfg["drug_list"], "short_name_max_len": cfg["short_name_max_len"]}
