"""Event detection evaluation (low-level and high-level).

Ground truth format — same as pipeline output:
    "low_level_events":  [{"type": "floor_bounce", "frame_idx": 150, ...}, ...]
    "high_level_events": [{"type": "serve", "frame_idx": 100, ...}, ...]

Events are matched by *type* and *temporal proximity* (frame tolerance).
"""
from __future__ import annotations

from collections import defaultdict

from src.evaluation.metrics import PRF, match_events_temporal


def _group_by_type(events: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for e in events:
        groups[e["type"]].append(e["frame_idx"])
    return groups


def evaluate_events(
    pred: list[dict],
    gt: list[dict],
    tolerance: int = 5,
    label: str = "events",
) -> dict:
    """Evaluate event detection (works for both low-level and high-level).

    Each event type is evaluated independently.  A predicted event is a
    true positive if there is a ground truth event of the **same type**
    within *tolerance* frames.

    Args:
        pred: predicted events list.
        gt: ground truth events list.
        tolerance: temporal tolerance in frames.
        label: human-readable label for the result section.

    Returns:
        Dictionary with per-type and aggregate metrics.
    """
    pred_groups = _group_by_type(pred)
    gt_groups = _group_by_type(gt)

    all_types = sorted(set(pred_groups.keys()) | set(gt_groups.keys()))
    per_type: dict[str, dict] = {}
    agg = PRF()

    for t in all_types:
        pf = pred_groups.get(t, [])
        gf = gt_groups.get(t, [])
        matched, unmatched_p, unmatched_g = match_events_temporal(pf, gf, tolerance)

        prf = PRF(tp=len(matched), fp=len(unmatched_p), fn=len(unmatched_g))
        per_type[t] = {
            "pred_count": len(pf),
            "gt_count": len(gf),
            **prf.to_dict(),
        }
        agg.tp += prf.tp
        agg.fp += prf.fp
        agg.fn += prf.fn

    frame_errors: list[int] = []
    for t in all_types:
        pf = pred_groups.get(t, [])
        gf = gt_groups.get(t, [])
        matched, _, _ = match_events_temporal(pf, gf, tolerance)
        for pframe, gframe in matched:
            frame_errors.append(abs(pframe - gframe))

    result: dict = {
        "tolerance_frames": tolerance,
        "aggregate": agg.to_dict(),
        "per_type": per_type,
    }
    if frame_errors:
        import numpy as np
        result["temporal_error_frames"] = {
            "mean": round(float(np.mean(frame_errors)), 2),
            "median": round(float(np.median(frame_errors)), 2),
            "max": int(np.max(frame_errors)),
        }
    return result
