"""Player tracking evaluation.

Ground truth format — same as pipeline output ``player_tracking``:
    [{
        "frame_idx": 0,
        "players": [
            {"track_id": 1, "bbox": [x1, y1, x2, y2], ...},
            ...
        ]
    }, ...]

Metrics:
    - HOTA (Higher Order Tracking Accuracy) with DetA, AssA, LocA sub-scores
    - MOTA (Multi-Object Tracking Accuracy)
    - MOTP (Multi-Object Tracking Precision)
    - Detection rate, ID switches, position error
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.evaluation.metrics import DistanceStats


def _bbox_center(bbox: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def _iou(a: list[float], b: list[float]) -> float:
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_frame(
    gt_players: list[dict],
    pred_players: list[dict],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], int, int]:
    """Match GT and predicted players on a single frame.

    Returns:
        matches: list of (gt_track_id, pred_track_id, iou)
        fn: unmatched GT count
        fp: unmatched pred count
    """
    if not gt_players or not pred_players:
        return [], len(gt_players), len(pred_players)

    candidates: list[tuple[float, int, int, float]] = []
    for gi, gp in enumerate(gt_players):
        for pi, pp in enumerate(pred_players):
            iou_val = _iou(gp["bbox"], pp["bbox"])
            if iou_val >= iou_threshold:
                candidates.append((1 - iou_val, gi, pi, iou_val))

    candidates.sort()
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for _, gi, pi, iou_val in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((
            gt_players[gi]["track_id"],
            pred_players[pi]["track_id"],
            iou_val,
        ))

    fn = len(gt_players) - len(used_gt)
    fp = len(pred_players) - len(used_pred)
    return matches, fn, fp


# ── HOTA ──

def _compute_hota(
    gt_map: dict[int, dict],
    pred_map: dict[int, dict],
    alpha_thresholds: np.ndarray | None = None,
) -> dict:
    """Compute HOTA and its sub-metrics (DetA, AssA, LocA).

    HOTA = sqrt(DetA * AssA), averaged over a range of IoU thresholds.

    Reference: Luiten et al., "HOTA: A Higher Order Metric for Evaluating
    Multi-Object Tracking", IJCV 2020.
    """
    if alpha_thresholds is None:
        alpha_thresholds = np.arange(0.05, 1.0, 0.05)

    all_frames = sorted(set(gt_map.keys()) | set(pred_map.keys()))

    # Pre-compute: how many frames each GT/pred ID is present in
    gt_id_count: dict[int, int] = defaultdict(int)
    pred_id_count: dict[int, int] = defaultdict(int)
    for fidx in all_frames:
        g = gt_map.get(fidx)
        p = pred_map.get(fidx)
        if g:
            for pl in g["players"]:
                gt_id_count[pl["track_id"]] += 1
        if p:
            for pl in p["players"]:
                pred_id_count[pl["track_id"]] += 1

    hota_vals = []
    deta_vals = []
    assa_vals = []
    loca_vals = []

    for alpha in alpha_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_iou = 0.0

        # (gt_id, pred_id) → count of frames matched together (TPA)
        pair_tp_count: dict[tuple[int, int], int] = defaultdict(int)

        for fidx in all_frames:
            g = gt_map.get(fidx)
            p = pred_map.get(fidx)
            gt_players = g["players"] if g else []
            pred_players = p["players"] if p else []

            matches, fn, fp = _match_frame(gt_players, pred_players, alpha)
            total_tp += len(matches)
            total_fn += fn
            total_fp += fp

            for gt_id, pred_id, iou_val in matches:
                total_iou += iou_val
                pair_tp_count[(gt_id, pred_id)] += 1

        denom = total_tp + total_fp + total_fn
        deta = total_tp / denom if denom else 0.0
        loca = total_iou / total_tp if total_tp else 0.0

        # AssA: for each TP, look up its (gt_id, pred_id) pair and compute
        # the track-level Ass-IoU.  All TPs sharing the same pair get the
        # same Ass-IoU, but each contributes individually to the average.
        ass_iou_sum = 0.0
        for (gt_id, pred_id), tpa in pair_tp_count.items():
            fpa = pred_id_count[pred_id] - tpa
            fna = gt_id_count[gt_id] - tpa
            ass_iou = tpa / (tpa + fpa + fna)
            # This pair contributes `tpa` TPs, each with this Ass-IoU
            ass_iou_sum += ass_iou * tpa

        assa = ass_iou_sum / total_tp if total_tp else 0.0
        hota = float(np.sqrt(deta * assa))

        hota_vals.append(hota)
        deta_vals.append(deta)
        assa_vals.append(assa)
        loca_vals.append(loca)

    return {
        "hota": round(float(np.mean(hota_vals)), 4),
        "deta": round(float(np.mean(deta_vals)), 4),
        "assa": round(float(np.mean(assa_vals)), 4),
        "loca": round(float(np.mean(loca_vals)), 4),
    }


# ── Main entry point ──

def evaluate_tracking(
    pred: list[dict],
    gt: list[dict],
    iou_threshold: float = 0.3,
) -> dict:
    """Evaluate player tracking quality.

    Args:
        pred: predicted ``player_tracking`` list.
        gt: ground truth ``player_tracking`` list.
        iou_threshold: minimum IoU for MOTA/MOTP matching.  HOTA uses its
            own range of thresholds (0.05 – 0.95).

    Returns:
        Dictionary with HOTA, MOTA, MOTP, detection rate, ID switches,
        and position error.
    """
    gt_map = {e["frame_idx"]: e for e in gt}
    pred_map = {e["frame_idx"]: e for e in pred}

    total_gt_boxes = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    dist_stats = DistanceStats()

    prev_gt_to_pred: dict[int, int] = {}

    all_frames = sorted(set(gt_map.keys()) | set(pred_map.keys()))

    for fidx in all_frames:
        g = gt_map.get(fidx)
        p = pred_map.get(fidx)
        gt_players = g["players"] if g else []
        pred_players = p["players"] if p else []
        total_gt_boxes += len(gt_players)

        matches, fn, fp = _match_frame(gt_players, pred_players, iou_threshold)
        total_tp += len(matches)
        total_fn += fn
        total_fp += fp

        gt_to_pred: dict[int, int] = {}
        for gt_id, pred_id, _ in matches:
            gt_to_pred[gt_id] = pred_id

        for m in matches:
            gt_p = next(p for p in gt_players if p["track_id"] == m[0])
            pr_p = next(p for p in pred_players if p["track_id"] == m[1])
            d = float(np.linalg.norm(
                _bbox_center(gt_p["bbox"]) - _bbox_center(pr_p["bbox"])
            ))
            dist_stats.errors.append(d)

        for gt_id, pred_id in gt_to_pred.items():
            if gt_id in prev_gt_to_pred and prev_gt_to_pred[gt_id] != pred_id:
                total_id_switches += 1

        prev_gt_to_pred.update(gt_to_pred)

    mota = (
        1 - (total_fn + total_fp + total_id_switches) / total_gt_boxes
        if total_gt_boxes
        else 0.0
    )
    motp = dist_stats.mean

    hota_metrics = _compute_hota(gt_map, pred_map)

    return {
        "total_gt_boxes": total_gt_boxes,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "id_switches": total_id_switches,
        "detection_rate": round(total_tp / total_gt_boxes, 4) if total_gt_boxes else 0.0,
        "hota": hota_metrics["hota"],
        "deta": hota_metrics["deta"],
        "assa": hota_metrics["assa"],
        "loca": hota_metrics["loca"],
        "mota": round(mota, 4),
        "motp_px": round(motp, 2),
        "iou_threshold": iou_threshold,
        "position_error_px": dist_stats.to_dict(),
    }
