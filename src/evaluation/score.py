"""OCR / score evaluation.

Ground truth format — same as pipeline output ``ocr_readings``:
    [{
        "frame_idx": 0,
        "team_a_scores": [0],
        "team_b_scores": [0],
        "team_a_name": "...",
        "team_b_name": "...",
    }, ...]

Also evaluates final score correctness using ``high_level_events``
(looking at the last ``point_won`` and accumulated score).
"""
from __future__ import annotations


def evaluate_score(
    pred_readings: list[dict],
    gt_readings: list[dict],
    tolerance: int = 5,
) -> dict:
    """Evaluate OCR score reading accuracy.

    For each GT reading, finds the closest predicted reading within
    *tolerance* frames and compares the score values.

    Args:
        pred_readings: predicted ``ocr_readings`` list.
        gt_readings: ground truth ``ocr_readings`` list.
        tolerance: temporal tolerance in frames for matching readings.

    Returns:
        Dictionary with score accuracy, team name accuracy, and details.
    """
    if not gt_readings:
        return {"error": "no ground truth OCR readings"}

    pred_by_frame = {r["frame_idx"]: r for r in pred_readings}

    matched = 0
    score_correct = 0
    name_correct = 0
    total = len(gt_readings)

    per_reading: list[dict] = []

    for g in gt_readings:
        gf = g["frame_idx"]
        best: dict | None = None
        best_d = tolerance + 1
        for pf, pr in pred_by_frame.items():
            d = abs(pf - gf)
            if d < best_d:
                best_d = d
                best = pr
        if best is None or best_d > tolerance:
            per_reading.append({
                "gt_frame": gf, "matched": False,
            })
            continue

        matched += 1
        gt_a = g.get("team_a_scores", [])
        gt_b = g.get("team_b_scores", [])
        pr_a = best.get("team_a_scores", [])
        pr_b = best.get("team_b_scores", [])

        scores_ok = (gt_a == pr_a and gt_b == pr_b)
        if scores_ok:
            score_correct += 1

        names_ok = (
            g.get("team_a_name", "") == best.get("team_a_name", "")
            and g.get("team_b_name", "") == best.get("team_b_name", "")
        )
        if names_ok:
            name_correct += 1

        per_reading.append({
            "gt_frame": gf,
            "pred_frame": best["frame_idx"],
            "matched": True,
            "score_correct": scores_ok,
            "name_correct": names_ok,
        })

    return {
        "total_gt_readings": total,
        "matched_readings": matched,
        "score_accuracy": round(score_correct / matched, 4) if matched else 0.0,
        "name_accuracy": round(name_correct / matched, 4) if matched else 0.0,
        "details": per_reading,
    }


def evaluate_final_score(
    pred_events: list[dict],
    gt_events: list[dict],
) -> dict:
    """Compare final accumulated score from high-level events.

    Counts ``point_won`` events per team in both prediction and GT,
    then checks if totals match.
    """
    def _count_points(events: list[dict]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for e in events:
            if e.get("type") == "point_won":
                team = e.get("details", {}).get("winning_team")
                if team is not None:
                    counts[team] = counts.get(team, 0) + 1
        return counts

    pred_pts = _count_points(pred_events)
    gt_pts = _count_points(gt_events)

    all_teams = sorted(set(pred_pts.keys()) | set(gt_pts.keys()))
    per_team: dict[str, dict] = {}
    total_correct = True

    for t in all_teams:
        p = pred_pts.get(t, 0)
        g = gt_pts.get(t, 0)
        ok = p == g
        per_team[f"team_{t}"] = {"pred_points": p, "gt_points": g, "match": ok}
        if not ok:
            total_correct = False

    return {
        "points_match": total_correct,
        "per_team": per_team,
    }
