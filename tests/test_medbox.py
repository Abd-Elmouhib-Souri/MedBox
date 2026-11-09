"""Fast tests: no model is downloaded. Fake detector and fake OCR are injected into the Reader."""

from PIL import Image

from medbox.evaluate import accepted, metrics, wrong_accept
from medbox.reader import Reader
from medbox.stats import proportion, wilson

NAMES = ["DOLIPRANE", "AMLOVAR", "ATOR", "AMLOR", "VAL", "T"]   # small closed list for the tests
CFG = {"short_name_max_len": 3, "crop_pad": 0.05, "rotations": [0, 90, 270], "threshold": 80}
IMG = Image.new("RGB", (200, 100), "white")


def make_reader(box=(10, 10, 100, 50, 0.9), texts=None):
    """Reader with fakes: the detector returns `box`; the OCR returns texts[angle] (by image size)."""
    texts = texts or {0: "DOLIPRANE"}

    def fake_ocr(image):
        # 200x100 image: rotated by 90/270 it becomes 100x200 (+ crop). We tell angles apart by orientation.
        landscape = image.size[0] >= image.size[1]
        return texts.get(0 if landscape else 90, "")

    return Reader(CFG, names=NAMES, detector=lambda image: box, ocr=fake_ocr)


# ---- matching ---------------------------------------------------------------------------------
def test_noisy_ocr_finds_the_right_name():
    name, score = make_reader().match("D0LIPRANE 1000mg")         # OCR error 0 -> O, dosage removed
    assert name == "DOLIPRANE" and score >= 80


def test_short_name_needs_exact_match():
    r = make_reader()
    assert r.match("VAL") == ("VAL", 100.0)                       # exact: accepted
    name, _ = r.match("TOTAL VALUE")                              # contains VAL and T, but not exact
    assert name not in {"VAL", "T"}


def test_empty_text_gives_no_candidate():
    assert make_reader().match("   ") == (None, 0.0)


# ---- decisions --------------------------------------------------------------------------------
def test_no_detection_is_refused():
    r = make_reader(box=None).identify(IMG)
    assert r["status"] == "refused" and r["reason"] == "no_detection" and r["name"] is None


def test_score_below_threshold_is_refused():
    r = make_reader(texts={0: "XQZW"}).identify(IMG)              # garbage text -> low score
    assert r["status"] == "refused" and r["reason"] == "below_threshold" and r["name"] is None


def test_good_reading_is_accepted():
    r = make_reader().identify(IMG)
    assert r["status"] == "ok" and r["name"] == "DOLIPRANE"


def test_rotation_with_best_score_wins():
    # crop is landscape at 0°, portrait at 90°/270°: the OCR only reads correctly when rotated
    r = make_reader(texts={0: "XQZW", 90: "AMLOVAR"}).identify(IMG)
    assert r["name"] == "AMLOVAR" and r["rotation"] in (90, 270)


def test_baseline_uses_whole_image_without_detector():
    r = make_reader(box=None).identify(IMG, use_detector=False)   # detector would refuse, baseline does not use it
    assert r["status"] == "ok" and r["box"] is None


# ---- evaluation rules -------------------------------------------------------------------------
def test_wrong_accept_counts_negatives():
    neg = {"true_name": "NONE", "candidate": "ATOR", "score": 90.0, "reason": None}
    pos_ok = {"true_name": "ATOR", "candidate": "ATOR", "score": 90.0, "reason": None}
    assert wrong_accept(neg, 80) and not wrong_accept(pos_ok, 80)
    assert not accepted(neg, 95)                                  # higher threshold -> refused
    m = metrics([neg, pos_ok], 80)
    assert m["wrong_accepts"]["count"] == 1 and m["top1_precision"]["k"] == 1


# ---- statistics -------------------------------------------------------------------------------
def test_wilson_known_values():
    lo, hi = wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3   # reference values for 5/10
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3                  # 0/10 still has a real upper bound


def test_zero_denominator_is_undefined():
    assert wilson(0, 0) is None
    assert proportion(0, 0)["text"].startswith("undefined")
