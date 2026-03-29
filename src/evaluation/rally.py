"""Rally state classification evaluation.

Ground truth format — same as pipeline output ``rally_states``:
    [{"frame_idx": 0, "is_rally_active": false}, ...]

Positive class = ``rally_active`` (is_rally_active == true).
"""
from __future__ import annotations

from src.evaluation.metrics import PRF


def evaluate_rally(
    pred: list[dict],
    gt: list[dict],
) -> dict:
    """Evaluate per-frame rally state classification.

    Args:
        pred: predicted ``rally_states`` list.
        gt: ground truth ``rally_states`` list.

    Returns:
        Dictionary with accuracy, precision/recall/f1 (positive = active),
        and per-class counts.
    """
    gt_map = {e["frame_idx"]: e["is_rally_active"] for e in gt}
    pred_map = {e["frame_idx"]: e["is_rally_active"] for e in pred}

    common_frames = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common_frames:
        return {"error": "no overlapping frames between pred and gt"}

    correct = 0
    prf_active = PRF()
    prf_inactive = PRF()
    total = len(common_frames)

    for fidx in common_frames:
        g = gt_map[fidx]
        p = pred_map[fidx]

        if p == g:
            correct += 1

        if g and p:
            prf_active.tp += 1
        elif g and not p:
            prf_active.fn += 1
            prf_inactive.fp += 1
        elif not g and p:
            prf_active.fp += 1
            prf_inactive.fn += 1
        else:
            prf_inactive.tp += 1

    gt_active = sum(1 for f in common_frames if gt_map[f])
    gt_inactive = total - gt_active

    return {
        "total_frames_compared": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "gt_active_frames": gt_active,
        "gt_inactive_frames": gt_inactive,
        "rally_active_prf": prf_active.to_dict(),
        "rally_inactive_prf": prf_inactive.to_dict(),
    }
