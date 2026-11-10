"""v2 tests with fake detector and OCR (no model download)."""

from PIL import Image

from medbox.evaluate_v2 import outcome
from medbox.reader_v2 import ReaderV2, alnum_count

CFG = {"short_name_max_len": 3, "top_k": 2, "crop_pad": 0.05, "rotations": [0], "min_chars": 4,
       "T_brand": 85, "T_dci": 85}
BRANDS = ["URPEX", "CLAMOXYL", "T", "KETAXOL"]
DCIS = ["ACICLOVIR", "AMOXICILLINE"]
IMG = Image.new("RGB", (400, 300), "white")


def reader(boxes, texts):
    """Fake detector returns `boxes`; fake OCR returns texts in order of calls."""
    it = iter(texts)
    return ReaderV2(CFG, brands=BRANDS, dcis=DCIS, detector=lambda im: boxes, ocr=lambda im: next(it, ""))


def test_short_reading_is_dropped():
    assert alnum_count("T") == 1 and alnum_count("TAX:") == 3
    r = reader([(0, 0, 100, 40)], ["T"]).identify(IMG)            # the Allergica case
    assert r["status"] == "refused" and r["candidates"] == []


def test_fragment_no_longer_scores_90():
    name, score = reader([], []).match_brand("TAXO")              # fragment of KETAXOL
    assert score < 85                                              # ratio, not WRatio


def test_tallest_box_brand_wins():
    boxes = [(0, 0, 100, 20), (0, 50, 300, 150)]                  # 2nd box is taller -> read first
    r = reader(boxes, ["CLAMOXYL", "AMOXICILLINE 500"]).identify(IMG)
    assert r["status"] == "brand" and r["answer"] == "CLAMOXYL"


def test_molecule_answer_when_no_brand():
    r = reader([(0, 0, 200, 60)], ["ACICLOVIR 5%"]).identify(IMG)  # the Urpex case
    assert r["status"] == "molecule" and r["answer"] == "ACICLOVIR"


def test_outcome_rules():
    dci_of = {"URPEX": {"ACICLOVIR"}}
    assert outcome({"status": "molecule", "answer": "ACICLOVIR"}, "URPEX", dci_of) == "correct_molecule"
    assert outcome({"status": "brand", "answer": "KETAXOL"}, "URPEX", dci_of) == "wrong"
    assert outcome({"status": "molecule", "answer": "ACICLOVIR"}, "NONE", dci_of) == "wrong"
    assert outcome({"status": "refused", "answer": None}, "NONE", dci_of) == "refused"
