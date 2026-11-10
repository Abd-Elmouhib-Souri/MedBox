"""Evaluation of MedBox v2 (PROTOCOL.md section 11).

    python -m medbox.evaluate_v2 --dev    # 13 dev photos (data/field): choose T_brand then T_dci (once) -> config.yaml
    python -m medbox.evaluate_v2 --test   # future new photos (data/test_v2): run ONCE (not available yet)

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


def run_test() -> None:
    if (RESULTS / "test_v2.json").exists():
        raise SystemExit("results/test_v2.json already exists: the v2 test is run ONCE.")
    if not Path("data/test_v2_manifest.sha256").exists():
        raise SystemExit("No v2 test set yet (PROTOCOL.md §11): photograph >= 20 new boxes first.")
    raise SystemExit("v2 test runner: to be enabled with the new test set (same rules as --dev, thresholds fixed).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--dev", action="store_true")
    m.add_argument("--test", action="store_true")
    a = ap.parse_args()
    run_dev() if a.dev else run_test()
