"""Unified evaluation runner — compares a prediction JSON with a ground truth
JSON and produces a report covering all available components."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.ball import evaluate_ball
from src.evaluation.events import evaluate_events
from src.evaluation.rally import evaluate_rally
from src.evaluation.score import evaluate_final_score, evaluate_score
from src.evaluation.tracking import evaluate_tracking

logger = logging.getLogger(__name__)

SECTIONS = [
    "ball_tracking",
    "player_tracking",
    "low_level_events",
    "high_level_events",
    "rally_states",
    "ocr_readings",
]


def evaluate_all(
    pred: dict,
    gt: dict,
    *,
    event_tolerance: int = 5,
    ball_distance_threshold: float = 15.0,
    iou_threshold: float = 0.3,
    ocr_tolerance: int = 30,
) -> dict[str, Any]:
    """Run evaluation on every section present in **both** pred and gt.

    Args:
        pred: full pipeline output dictionary.
        gt: ground truth dictionary (same schema — only sections you want
            to evaluate need to be present).
        event_tolerance: temporal tolerance for event matching (frames).
        ball_distance_threshold: max pixel distance for ball TP.
        iou_threshold: min IoU for player tracking match.
        ocr_tolerance: temporal tolerance for OCR reading matching (frames).

    Returns:
        Dictionary with one key per evaluated section, plus ``summary``.
    """
    report: dict[str, Any] = {}

    if "ball_tracking" in gt and "ball_tracking" in pred:
        logger.info("Evaluating ball detection …")
        report["ball_detection"] = evaluate_ball(
            pred["ball_tracking"], gt["ball_tracking"],
            distance_threshold=ball_distance_threshold,
        )

    if "player_tracking" in gt and "player_tracking" in pred:
        logger.info("Evaluating player tracking …")
        report["player_tracking"] = evaluate_tracking(
            pred["player_tracking"], gt["player_tracking"],
            iou_threshold=iou_threshold,
        )

    if "low_level_events" in gt and "low_level_events" in pred:
        logger.info("Evaluating low-level events …")
        report["low_level_events"] = evaluate_events(
            pred["low_level_events"], gt["low_level_events"],
            tolerance=event_tolerance,
            label="low_level_events",
        )

    if "high_level_events" in gt and "high_level_events" in pred:
        logger.info("Evaluating high-level events …")
        report["high_level_events"] = evaluate_events(
            pred["high_level_events"], gt["high_level_events"],
            tolerance=event_tolerance,
            label="high_level_events",
        )

    if "rally_states" in gt and "rally_states" in pred:
        logger.info("Evaluating rally state classification …")
        report["rally_state"] = evaluate_rally(
            pred["rally_states"], gt["rally_states"],
        )

    if "ocr_readings" in gt and "ocr_readings" in pred:
        logger.info("Evaluating OCR score readings …")
        report["ocr_score"] = evaluate_score(
            pred["ocr_readings"], gt["ocr_readings"],
            tolerance=ocr_tolerance,
        )

    if "high_level_events" in gt and "high_level_events" in pred:
        report["final_score"] = evaluate_final_score(
            pred["high_level_events"], gt["high_level_events"],
        )

    report["summary"] = _build_summary(report)
    return report


def _build_summary(report: dict) -> dict:
    """One-line-per-component summary suitable for terminal output."""
    s: dict[str, Any] = {}

    if "ball_detection" in report:
        b = report["ball_detection"]
        s["ball"] = (
            f"det_rate={b['detection_rate']:.1%}  "
            f"P={b['precision']:.1%}  R={b['recall']:.1%}  F1={b['f1']:.1%}  "
            f"mean_err={b['position_error_px']['mean']:.1f}px"
        )

    if "player_tracking" in report:
        t = report["player_tracking"]
        s["tracking"] = (
            f"HOTA={t['hota']:.1%}  "
            f"DetA={t['deta']:.1%}  AssA={t['assa']:.1%}  "
            f"MOTA={t['mota']:.1%}  "
            f"id_sw={t['id_switches']}"
        )

    if "low_level_events" in report:
        a = report["low_level_events"]["aggregate"]
        s["low_level_events"] = (
            f"P={a['precision']:.1%}  R={a['recall']:.1%}  F1={a['f1']:.1%}  "
            f"(tol={report['low_level_events']['tolerance_frames']}f)"
        )

    if "high_level_events" in report:
        a = report["high_level_events"]["aggregate"]
        s["high_level_events"] = (
            f"P={a['precision']:.1%}  R={a['recall']:.1%}  F1={a['f1']:.1%}  "
            f"(tol={report['high_level_events']['tolerance_frames']}f)"
        )

    if "rally_state" in report:
        r = report["rally_state"]
        prf = r["rally_active_prf"]
        s["rally_state"] = (
            f"acc={r['accuracy']:.1%}  "
            f"P={prf['precision']:.1%}  R={prf['recall']:.1%}  F1={prf['f1']:.1%}"
        )

    if "ocr_score" in report:
        o = report["ocr_score"]
        s["ocr_score"] = (
            f"score_acc={o['score_accuracy']:.1%}  "
            f"matched={o['matched_readings']}/{o['total_gt_readings']}"
        )

    if "final_score" in report:
        fs = report["final_score"]
        s["final_score"] = "MATCH" if fs["points_match"] else "MISMATCH"

    return s


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def print_report(report: dict) -> None:
    """Pretty-print the evaluation report to stdout."""
    summary = report.get("summary", {})
    print("\n" + "=" * 64)
    print("EVALUATION REPORT")
    print("=" * 64)
    for section, line in summary.items():
        print(f"  {section:22s} {line}")
    print("=" * 64)

    for section in [
        "ball_detection", "player_tracking",
        "low_level_events", "high_level_events",
        "rally_state", "ocr_score", "final_score",
    ]:
        if section not in report:
            continue
        print(f"\n── {section} {'─' * (58 - len(section))}")
        _print_dict(report[section], indent=2)


def _print_dict(d: dict, indent: int = 0) -> None:
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            _print_dict(v, indent + 2)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            print(f"{prefix}{k}: [{len(v)} items]")
        else:
            print(f"{prefix}{k}: {v}")
