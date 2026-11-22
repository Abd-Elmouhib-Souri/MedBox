"""Evaluation of MedBox v2 (PROTOCOL.md section 11).

    python -m medbox.evaluate_v2 --dev    # 13 dev photos (data/field): choose T_brand then T_dci (once) -> config.yaml
    python -m medbox.evaluate_v2 --test   # independent test (data/test_v2, PROTOCOL.md §12): fixed thresholds, ONCE

Numbers from --dev are DEVELOPMENT results (thresholds are chosen on these photos), never test results.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from medbox.evaluate import check_manifest, git_commit, versions
from medbox.reader import CONFIG_PATH
from medbox.reader_v2 import ReaderV2, load_v2_config
from medbox.stats import proportion

RESULTS = Path("results")


def truth_dci(path: str = "data/brand_dci.csv") -> dict[str, set[str]]:
    """Trade name -> set of its DCI (a molecule answer is correct if it is in this set)."""
    df = pd.read_csv(path)
    return df.groupby("name")["dci"].apply(set).to_dict()


def outcome(dec: dict, true_name: str, dci_of: dict) -> str:
    """correct_brand | correct_molecule | wrong | refused."""
    if dec["status"] == "refused":
        return "refused"
    if true_name == "NONE":
        return "wrong"                                             # any answer on a product not in the list
    if dec["status"] == "brand":
        return "correct_brand" if dec["answer"] == true_name else "wrong"
    return "correct_molecule" if dec["answer"] in dci_of.get(true_name, set()) else "wrong"


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["true_name"] != "NONE"]
    neg = [r for r in rows if r["true_name"] == "NONE"]
    count = lambda rs, o: sum(r["outcome"] == o for r in rs)
    return {
        "brand_correct": proportion(count(pos, "correct_brand"), len(pos)),
        "brand_or_molecule_correct": proportion(count(pos, "correct_brand") + count(pos, "correct_molecule"), len(pos)),
        "wrong_answers": {"count": count(rows, "wrong"), **proportion(count(rows, "wrong"), len(rows))},
        "not_in_list_refused": proportion(count(neg, "refused"), len(neg)),
    }


def collect(reader: ReaderV2, truth: pd.DataFrame, folder: Path) -> list[dict]:
    rows = []
    for f, t in zip(truth["file"], truth["true_name"]):
        with Image.open(folder / f) as im:
            cands = reader.candidates(im)
        b, d = ReaderV2.best(cands, "brand"), ReaderV2.best(cands, "dci")
        print(f"  {f:8} truth={t:14} brand={str(b and b['brand']):18} {b and b['brand_score']!s:>6} "
              f"dci={str(d and d['dci'])[:24]:24} {d and d['dci_score']!s:>6}  text={b and b['text']!r}")
        rows.append({"file": f, "true_name": t, "candidates": cands})
    return rows


def write_thresholds(t_b: int, t_d: int) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text, n1 = re.subn(r"^(\s+T_brand:)\s*null", rf"\g<1> {t_b}", text, flags=re.M)
    text, n2 = re.subn(r"^(\s+T_dci:)\s*null", rf"\g<1> {t_d}", text, flags=re.M)
    if n1 != 1 or n2 != 1:
        raise SystemExit("config.yaml: 'T_brand: null' / 'T_dci: null' not found (already set?)")
    CONFIG_PATH.write_text(text, encoding="utf-8")


def run_dev() -> None:
    cfg = load_v2_config()
    if cfg["T_brand"] is not None or cfg["T_dci"] is not None:
        raise SystemExit("v2 thresholds already set: the protocol forbids changing them.")
    check_manifest(Path("data/field_manifest.sha256"))
    truth, dci_of = pd.read_csv("data/field_truth.csv"), truth_dci()
    print("DEV v2 (13 field photos, now the development set)")
    rows = collect(ReaderV2(cfg), truth, Path("data/field"))

    def wrong_at(t_b, t_d, kind):
        n = 0
        for r in rows:
            dec = ReaderV2.decide(r["candidates"], t_b, t_d)
            if dec["status"] == kind and outcome(dec, r["true_name"], dci_of) == "wrong":
                n += 1
        return n

    ok_b = [t for t in range(101) if wrong_at(t, 101, "brand") == 0]           # rule: smallest with 0 wrong
    t_b = ok_b[0] if ok_b else 100
    ok_d = [t for t in range(101) if wrong_at(t_b, t, "molecule") == 0]
    t_d = ok_d[0] if ok_d else 100

    for r in rows:
        dec = ReaderV2.decide(r["candidates"], t_b, t_d)
        r.update(decision={k: dec[k] for k in ("status", "answer", "score")}, outcome=outcome(dec, r["true_name"], dci_of))
    out = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
           "versions": versions(), "note": "DEVELOPMENT results: thresholds chosen on these 13 photos (not a test)",
           "T_brand": t_b, "T_dci": t_d, "summary": summarize(rows), "rows": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dev_v2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    write_thresholds(t_b, t_d)
    for r in rows:
        print(f"  {r['file']:8} truth={r['true_name']:14} -> {r['decision']['status']:9} "
              f"{str(r['decision']['answer']):30} [{r['outcome']}]")
    print(f"\nT_brand = {t_b}, T_dci = {t_d}")
    print({k: v["text"] for k, v in out["summary"].items()})
    print("written: results/dev_v2.json and config.yaml  (DEVELOPMENT results, not a test)")


MARGIN_ZONE = (85, 88)          # PROTOCOL.md §12: thin-margin zone seen on dev, reported separately


def v1_decision(v1_reader, image) -> dict:
    """v1 pipeline (YOLO, threshold 91 from config.yaml) mapped to the v2 decision format, so outcome() applies."""
    r = v1_reader.identify(image, use_detector=True)
    return {"status": "brand" if r["status"] == "ok" else "refused", "answer": r["name"],
            "score": r["score"], "ocr_text": r["ocr_text"]}


def run_test(v2_reader: ReaderV2 | None = None, v1_reader=None, cfg: dict | None = None,
             manifest: Path = Path("data/test_v2_manifest.sha256"), truth_csv: str = "data/test_v2_truth.csv",
             folder: Path = Path("data/test_v2")) -> dict:
    """PROTOCOL.md §12: frozen photos, FIXED thresholds (read, never written), v2 and v1 on the same photos, ONCE.

    Readers and cfg can be injected (tests use fakes); by default the real models and config.yaml are used.
    """
    out_path = RESULTS / "test_v2.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists: the v2 test is run ONCE (PROTOCOL.md §12).")
    if not Path(manifest).exists():
        raise SystemExit(f"No frozen v2 test set ({manifest}): run tools.test_v2_set and commit it first.")
    cfg = cfg or load_v2_config()
    t_b, t_d = cfg["T_brand"], cfg["T_dci"]
    if t_b is None or t_d is None:
        raise SystemExit("v2 thresholds missing in config.yaml: they must come from the dev run (§11).")
    check_manifest(Path(manifest))                                  # refuses if any frozen file changed
    truth, dci_of = pd.read_csv(truth_csv, keep_default_na=False), truth_dci()

    if v2_reader is None:
        v2_reader = ReaderV2(cfg)
    if v1_reader is None:
        from medbox.reader import Reader, load_config               # real v1 (YOLO), loaded only here
        v1_reader = Reader(load_config())

    print(f"TEST v2 (once): {len(truth)} photos, T_brand={t_b}, T_dci={t_d} (fixed, not tuned)")
    rows = []
    for f, t in zip(truth["file"], truth["true_name"]):
        with Image.open(Path(folder) / f) as im:
            im = im.convert("RGB")
            cands = v2_reader.candidates(im)
            dec = ReaderV2.decide(cands, t_b, t_d)                  # fixed thresholds
            v1 = v1_decision(v1_reader, im)
        best_b = ReaderV2.best(cands, "brand")
        row = {"file": f, "true_name": t,
               "decision": {k: dec[k] for k in ("status", "answer", "score")},
               "outcome": outcome(dec, t, dci_of),
               "best_brand_score": best_b["brand_score"] if best_b else 0.0,
               "v1": {**v1, "outcome": outcome(v1, t, dci_of)},
               "candidates": cands}
        rows.append(row)
        print(f"  {f:8} truth={t:14} v2 -> {dec['status']:9} {str(dec['answer']):28} [{row['outcome']}]"
              f"   v1 -> {v1['status']:8} [{row['v1']['outcome']}]")

    lo, hi = MARGIN_ZONE
    margin = [{"file": r["file"], "true_name": r["true_name"], "best_brand_score": r["best_brand_score"],
               "outcome": r["outcome"]} for r in rows if lo <= r["best_brand_score"] < hi]
    v1_rows = [{"true_name": r["true_name"], "outcome": r["v1"]["outcome"]} for r in rows]
    out = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
           "versions": versions(), "note": "INDEPENDENT TEST (PROTOCOL.md §12): thresholds fixed, run once",
           "T_brand": t_b, "T_dci": t_d,
           "summary_v2": summarize(rows), "summary_v1_same_photos": summarize(v1_rows),
           "margin_zone_85_88": margin,
           "decision_rule": {"wrong_answers_v2": sum(r["outcome"] == "wrong" for r in rows),
                             "publish_as_v2.0": sum(r["outcome"] == "wrong" for r in rows) <= 1},
           "rows": rows}
    RESULTS.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nv2:", {k: v["text"] for k, v in out["summary_v2"].items()})
    print("v1:", {k: v["text"] for k, v in out["summary_v1_same_photos"].items()})
    print("margin zone 85-88:", margin)
    print("decision rule:", out["decision_rule"])
    print(f"written: {out_path}  (config.yaml not modified)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--dev", action="store_true")
    m.add_argument("--test", action="store_true")
    a = ap.parse_args()
    run_dev() if a.dev else run_test()
