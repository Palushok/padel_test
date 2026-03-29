"""Ball detection evaluation.

Ground truth format — same as pipeline output ``ball_tracking``:
    [{"frame_idx": 0, "position": [x, y] | null, "visible": true|false}, ...]

When ``visible`` is absent, visibility is inferred from ``position != null``.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import PRF, DistanceStats


def evaluate_ball(
    pred: list[dict],
    gt: list[dict],
    distance_threshold: float = 15.0,
) -> dict:
    """Evaluate ball detection quality.

    Args:
        pred: predicted ``ball_tracking`` list.
        gt: ground truth ``ball_tracking`` list.
        distance_threshold: maximum pixel distance for a detection to count
            as a true positive.

    Returns:
        Dictionary with detection_rate, precision/recall/f1, and
        position_error statistics.
    """
    gt_map = {e["frame_idx"]: e for e in gt}
    pred_map = {e["frame_idx"]: e for e in pred}

    prf = PRF()
    dist_stats = DistanceStats()
    gt_visible_count = 0
    gt_total = len(gt_map)

    for fidx, g in gt_map.items():
        gt_pos = g.get("position")
        gt_vis = g.get("visible", gt_pos is not None)
        if gt_vis and gt_pos is not None:
            gt_visible_count += 1

        p = pred_map.get(fidx)
        pred_pos = p.get("position") if p else None
        pred_has = pred_pos is not None

        if gt_vis and gt_pos is not None:
            if pred_has:
                d = np.linalg.norm(
                    np.array(pred_pos, dtype=float) - np.array(gt_pos, dtype=float)
                )
                dist_stats.errors.append(float(d))
                if d <= distance_threshold:
                    prf.tp += 1
                else:
                    prf.fp += 1
                    prf.fn += 1
            else:
                prf.fn += 1
        else:
            if pred_has:
                prf.fp += 1

    detection_rate = (
        (prf.tp / gt_visible_count) if gt_visible_count else 0.0
    )

    return {
        "detection_rate": round(detection_rate, 4),
        "gt_visible_frames": gt_visible_count,
        "gt_total_frames": gt_total,
        "distance_threshold_px": distance_threshold,
        **prf.to_dict(),
        "position_error_px": dist_stats.to_dict(),
    }
