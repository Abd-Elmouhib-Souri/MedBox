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


# ---- v2 independent test runner (PROTOCOL.md §12) --------------------------------------------------------------
import hashlib

import pytest

from medbox.evaluate_v2 import run_test

CFG_TEST = {**CFG, "T_brand": 88, "T_dci": 72}                      # fixed thresholds, as in config.yaml


class FakeV1:
    """v1 reader stand-in: always refuses (enough to check the runner, no YOLO needed)."""
    def identify(self, image, use_detector=True):
        return {"status": "refused", "name": None, "score": 0.0, "ocr_text": ""}


def test_run_test_refuses_without_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                    # empty folder: no frozen test set
    with pytest.raises(SystemExit):
        run_test(v2_reader=reader([], []), v1_reader=FakeV1(), cfg=CFG_TEST)


def test_run_test_runs_once_and_never_writes_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("T_brand: 88\n", encoding="utf-8")   # must stay byte-identical
    data = tmp_path / "data"
    (data / "test_v2").mkdir(parents=True)
    (data / "brand_dci.csv").write_text("name,dci\nCLAMOXYL,AMOXICILLINE\n", encoding="utf-8")
    IMG.save(data / "test_v2" / "v01.jpg")
    (data / "test_v2_truth.csv").write_text("file,true_name,group\nv01.jpg,CLAMOXYL,in_list\n", encoding="utf-8")
    files = ["data/test_v2/v01.jpg", "data/test_v2_truth.csv"]
    (data / "test_v2_manifest.sha256").write_text(
        "".join(f"{hashlib.sha256((tmp_path / p).read_bytes()).hexdigest()}  {p}\n" for p in files), encoding="utf-8")

    before = (tmp_path / "config.yaml").read_bytes()
    out = run_test(v2_reader=reader([(0, 0, 100, 40)], ["CLAMOXYL"]), v1_reader=FakeV1(), cfg=CFG_TEST)
    assert (tmp_path / "config.yaml").read_bytes() == before        # thresholds read, never written
    assert out["summary_v2"]["brand_correct"]["k"] == 1
    assert out["summary_v1_same_photos"]["brand_correct"]["k"] == 0
    assert (tmp_path / "results" / "test_v2.json").exists()
    with pytest.raises(SystemExit):                                # second run refused
        run_test(v2_reader=reader([(0, 0, 100, 40)], ["CLAMOXYL"]), v1_reader=FakeV1(), cfg=CFG_TEST)
