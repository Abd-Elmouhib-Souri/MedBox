"""Evaluation of MedBox Reader (rules in PROTOCOL.md).

    python -m medbox.evaluate --dev     # dev images only: sweep 0..100, choose the threshold, write it in config.yaml (once)
    python -m medbox.evaluate --final   # test images: checks the SHA-256 manifest, runs pipeline AND baseline ONCE
    python -m medbox.evaluate --field   # own phone photos (v1.1, PROTOCOL.md section 10): same rules, ONCE

Both modes refuse to run twice: --dev refuses if config.yaml already has a threshold,
--final refuses if results/final_test.json already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from medbox.reader import CONFIG_PATH, Reader, load_config
from medbox.stats import proportion

RESULTS = Path("results")


# ---------------------------------------------------------------------------------------------
# Running the systems
# ---------------------------------------------------------------------------------------------
def run_system(reader: Reader, truth: pd.DataFrame, folder: Path, use_detector: bool) -> list[dict]:
    """Raw output per image, computed with threshold 0 (the threshold is applied afterwards)."""
    rows = []
    for f, true_name in zip(truth["file"], truth["true_name"]):
        with Image.open(folder / f) as im:
            r = reader.identify(im, use_detector=use_detector, threshold=0)
        rows.append({"file": f, "true_name": true_name, **r})
        print(f"  {f:10} truth={true_name:22} candidate={str(r['candidate']):22} score={r['score']:6.2f} "
              f"rot={r['rotation']} ocr={r['ocr_text']!r}")
    return rows


def accepted(row: dict, thr: float) -> bool:
    """Accepted at threshold thr: something was read, matched, and scored high enough."""
    return row["reason"] != "no_detection" and row["candidate"] is not None and row["score"] >= thr


def wrong_accept(row: dict, thr: float) -> bool:
    """Accepted with a wrong name (for a negative, any accepted name is wrong)."""
    return accepted(row, thr) and (row["true_name"] == "NONE" or row["candidate"] != row["true_name"])


# ---------------------------------------------------------------------------------------------
# Metrics (PROTOCOL.md, section 3)
# ---------------------------------------------------------------------------------------------
def metrics(rows: list[dict], thr: float) -> dict:
    pos = [r for r in rows if r["true_name"] != "NONE"]
    neg = [r for r in rows if r["true_name"] == "NONE"]
    acc_pos = [r for r in pos if accepted(r, thr)]
    correct = [r for r in acc_pos if r["candidate"] == r["true_name"]]
    wrong = [r for r in rows if wrong_accept(r, thr)]
    return {
        "top1_precision": proportion(len(correct), len(acc_pos)),        # correct / accepted positives
        "coverage": proportion(len(acc_pos), len(pos)),                  # accepted positives / positives
        "wrong_accepts": {"count": len(wrong), **proportion(len(wrong), len(rows))},
        "negatives_refused": proportion(sum(not accepted(r, thr) for r in neg), len(neg)),
    }


def error_category(row: dict, thr: float) -> str | None:
    """First automatic classification; the README adds the manual reading of ocr_text."""
    if row["true_name"] == "NONE":
        return "negative_accepted" if accepted(row, thr) else None
    if row["reason"] == "no_detection":
        return "detection_failed"
    if wrong_accept(row, thr):
        return "wrong_name_accepted"
    if not accepted(row, thr):
        return "correct_name_below_threshold" if row["candidate"] == row["true_name"] else "ocr_or_matching_failed"
    return None                                                          # correct


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def versions() -> dict:
    out = {}
    for mod in ["torch", "ultralytics", "transformers", "rapidfuzz"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "not installed"
    return out


def check_manifest(path: Path) -> None:
    """Refuse to run if any frozen file changed since the freeze commit."""
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, file = line.split("  ", 1)
        got = hashlib.sha256(Path(file).read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f"MANIFEST MISMATCH: {file} changed since the freeze. Final run refused.")
    print(f"manifest OK ({path})")


def write_threshold(thr: float) -> None:
    """Replace `threshold: null` in config.yaml, keeping the comments."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    new, n = re.subn(r"^threshold:\s*null", f"threshold: {thr}", text, flags=re.M)
    if n != 1:
        raise SystemExit("config.yaml: 'threshold: null' not found (threshold already set?)")
    CONFIG_PATH.write_text(new, encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------------
def run_dev() -> None:
    cfg = load_config()
    if cfg["threshold"] is not None:
        raise SystemExit(f"Threshold already set ({cfg['threshold']}). The protocol forbids changing it.")
    truth = pd.read_csv("data/dev_truth.csv")
    reader = Reader(cfg)

    print("DEV — pipeline (YOLO + TrOCR + match)")
    rows = run_system(reader, truth, Path("data/dev"), use_detector=True)
    print("DEV — baseline (TrOCR on the whole image), for information only")
    base = run_system(reader, truth, Path("data/dev"), use_detector=False)

    # Rule: smallest threshold with 0 wrong accepts on dev (pipeline); if none, 100.
    sweep = []
    for t in range(0, 101):
        sweep.append({"threshold": t, "wrong_accepts": sum(wrong_accept(r, t) for r in rows),
                      "accepted": sum(accepted(r, t) for r in rows)})
    ok = [s["threshold"] for s in sweep if s["wrong_accepts"] == 0]
    thr = ok[0] if ok else 100
    rule = "smallest threshold with 0 wrong accepts on dev" if ok else "no threshold gives 0 wrong accepts -> 100"

    RESULTS.mkdir(exist_ok=True)
    out = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
           "versions": versions(), "chosen_threshold": thr, "rule_applied": rule,
           "pipeline_rows": rows, "baseline_rows": base, "sweep": sweep}
    (RESULTS / "dev_sweep.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    write_threshold(thr)
    print(f"\nCHOSEN THRESHOLD = {thr}  ({rule})\nwritten: results/dev_sweep.json and config.yaml")


def run_once(name: str, truth_csv: str, folder: str, manifest: str, description: str) -> None:
    """One frozen test set, run ONCE: manifest check, pipeline + baseline, metrics, decision rules."""
    out_path = RESULTS / f"{name}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists: each test set is run ONCE (PROTOCOL.md).")
    cfg = load_config()
    thr = cfg["threshold"]
    if thr is None:
        raise SystemExit("No threshold in config.yaml: run --dev first.")
    check_manifest(Path(manifest))
    truth = pd.read_csv(truth_csv)
    reader = Reader(cfg)

    systems = {}
    for sys_name, use_det in [("pipeline", True), ("baseline", False)]:
        print(f"{name.upper()} — {sys_name}")
        rows = run_system(reader, truth, Path(folder), use_detector=use_det)
        for r in rows:
            r["accepted"] = accepted(r, thr)
            r["error"] = error_category(r, thr)
        systems[sys_name] = {"metrics": metrics(rows, thr), "rows": rows}

    p, b = systems["pipeline"]["metrics"], systems["baseline"]["metrics"]
    pv, bv = p["top1_precision"]["value"], b["top1_precision"]["value"]
    yolo_useful = ("undefined (a precision has n = 0)" if pv is None or bv is None
                   else bool(pv > bv and p["wrong_accepts"]["count"] <= b["wrong_accepts"]["count"]))
    out = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "versions": versions(), "threshold": thr,
        "test_set": description,
        "decision_rules": {"yolo_useful": yolo_useful,
                           "demo_publishable": p["wrong_accepts"]["count"] <= 1},
        **systems,
    }
    RESULTS.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["threshold", "decision_rules"]}, indent=2))
    for sys_name in systems:
        print(sys_name, {k: v["text"] for k, v in systems[sys_name]["metrics"].items()})
    print(f"written: {out_path}")


def run_final() -> None:
    run_once("final_test", "data/test_truth.csv", "data/test", "data/test_manifest.sha256",
             "14 web-collected images (9 boxes + 5 negatives), not own photos (PROTOCOL.md, deviation 2)")


def run_field() -> None:
    run_once("field_test", "data/field_truth.csv", "data/field", "data/field_manifest.sha256",
             "13 own phone photos (8 boxes in the list + 5 products not in the list), PROTOCOL.md section 10")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true")
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--field", action="store_true")              # v1.1: own phone photos (PROTOCOL.md §10)
    args = ap.parse_args()
    run_dev() if args.dev else run_final() if args.final else run_field()
